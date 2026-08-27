"""PotentialBuilder: builds scattering potential volumes from atomic coordinates."""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Callable
from importlib import resources
from typing import Sequence

import gemmi
import lightning as L
import torch

from ..arrays import (
    radial_grid_2d,
    radial_grid_3d,
    soft_voxelize_coordinates,
    soft_voxelize_xy_coordinates,
)
from ..atom import (
    atom_symbol,
    kirkland_atomic_potential_2d,
    load_shtyrov_species_parameters,
    lobato_atomic_potential_2d,
)
from ..fft import spatial_convolve2d_same
from ..progress import TqdmProgress
from ._builders import (
    build_atomic_potential_kernel,
    build_potential_volume_analytic_scatter,
    build_potential_volume_analytic_scatter_kirkland,
    build_potential_volume_analytic_scatter_lobato,
    compute_supersampling_parameters,
    potential_from_deltas,
    recommended_rcut,
)

logger = logging.getLogger(__name__)

# Whether this process has already warned that some atoms fall back to
# per-element Peng factors. Every structure rendered under
# `parameterization="shtyrov"` without complete typing hits this, and a run
# renders one builder per species, so a 27-species tomogram reported the same
# fact 54 times: once per species from each of the two paths below, which
# derive it independently for the same builder.
#
# The counts differ per structure but the finding and its remedy do not, so
# the warning is a process-level one and the per-structure counts are DEBUG
# log records instead. DEBUG rather than INFO deliberately: a caller that has
# set INFO (e.g. to follow a tomogram's section timings) is asking for
# progress, not for a line per rendered species.
_peng_fallback_warned = False

# Cache of built kernels, keyed by everything one depends on.
#
# `build_atomic_potential_kernel`'s result is a pure function of the voxel size,
# the parameterization, and the element or bonded species: `compute_
# supersampling_parameters(dx)` fixes `ssn`/`ssdx`/`ssf`, and `sR_3d` and
# `avgpool3d` follow from those, so `dx` determines every grid argument passed
# alongside the species.
#
# Building them is not cheap -- a CPU `torch.exp` over the supersampled radial
# grid, per species -- and specimen assembly builds a separate `PotentialBuilder`
# per species, so the same handful of kernels is rebuilt over and over. Measured
# on the default `configs/tomogram.toml`: 321 kernel builds, 29 of them distinct,
# ~11 s of the 14 s cytosol render stage (the actual GPU `forward()` is 0.4 s).
#
# Kernels are small (the supersampled kernel binned down to the target grid, a
# few hundred KB) and the key space is bounded by the species tables, so this is
# capped rather than evicted; the cap only ever binds if a caller sweeps `dx`.
_KERNEL_CACHE: dict[tuple, torch.Tensor] = {}
_KERNEL_CACHE_MAX = 512


def _cached_atomic_potential_kernel(
    cache_key: tuple, build: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """
    Return a cached potential kernel, building it on first request.

    Parameters
    ----------
    cache_key : tuple
        Everything the kernel depends on -- see `_KERNEL_CACHE`.
    build : callable
        Zero-argument builder, called only on a miss.

    Returns
    -------
    torch.Tensor
        A fresh copy each call, so a caller mutating the result (or moving it to
        a device) cannot corrupt the entry for everyone else.
    """
    hit = _KERNEL_CACHE.get(cache_key)
    if hit is None:
        hit = build()
        if len(_KERNEL_CACHE) < _KERNEL_CACHE_MAX:
            _KERNEL_CACHE[cache_key] = hit
    return hit.clone()


class PotentialBuilder(L.LightningModule):
    """
    Module for building 3D electrostatic potential volumes from atomic coordinates.

    Computes potentials using supersampled atomic potential kernels and
    convolution, supporting multiple parameterizations (Kirkland, Lobato, Shtyrov).

    Parameters
    ----------
    n_xyz : int or tuple of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Pixel/voxel size in Å.
    atomic_numbers : torch.Tensor
        Atomic numbers of all atoms in structure.
    progressbars : bool, optional
        Enable progress bars during computation. Default is True.
    parameterization : str, optional
        Atomic potential parameterization: 'kirkland', 'lobato', or 'shtyrov'.
        Default is 'shtyrov' (bonded-species-aware if `atom_species` is
        given, otherwise per-element Peng `c4322` factors — see
        `atom_species` below).
    conv_backend : str, optional
        Convolution backend: 'fftconvolve' or 'conv3d'. Default is 'fftconvolve'.
    atom_species : sequence of str or None, optional
        Per-atom bonded-species descriptors (e.g. `"O(HH)"`, `"C(HHHC)"`,
        as produced by `specter.pdb.PDB.get_atom_species`), same length
        and atom order as `atomic_numbers`. Only used when
        `parameterization='shtyrov'`; ignored by Kirkland/Lobato. Atoms
        with no matching species (`None`, or not covered by
        `shtyrov_params_path`) fall back to gemmi's built-in per-element
        `c4322` (Peng et al.) scattering factors for that atom — the same
        fallback sffit itself uses. Default is None.
    shtyrov_params_path : str, optional
        Path to an sffit-format JSON species parameter file (see
        `specter.atom.load_shtyrov_species_parameters`). Only used when
        `atom_species` is given. Defaults to the bundled
        `specter.atom_data/params_cat.json`.
    rcut : float, optional
        Radius (Å) of the local evaluation window used by
        `forward(method="analytic")`, for every `parameterization`. Only
        used by `method="analytic"`. Default is None, which auto-selects
        the smallest radius that still captures >=99.5% of the total
        integrated potential of every element/species actually present
        (see `recommended_rcut`) — e.g. ~2-2.5 Å for a light-element-only
        (H/C/N/O) structure, up to ~5-6 Å if heavier or more diffuse
        elements (e.g. K, Na, or heavier alkali metals) are present. Pass
        an explicit value to override the auto-selection.
    periodic : bool, optional
        If True, wrap out-of-bounds voxel indices with periodic boundary
        conditions during soft voxelization. Use when coordinates were
        generated with periodic BCs (e.g. from GradientSKIcemaker). Only
        applies to the '3d' method; raises ValueError if used with '2d'.
        Default is False.

    Attributes
    ----------
    atomic_potentials_2d : torch.Tensor
        Precomputed 2D atomic potentials for each unique element.
    atomic_potentials_3d : torch.Tensor
        Precomputed 3D atomic potentials for each unique element.
    """

    def __init__(
        self,
        n_xyz: int | Sequence[int],
        dx: float,
        atomic_numbers: torch.Tensor,
        progressbars: bool = True,
        parameterization: str = "shtyrov",
        conv_backend: str = "fftconvolve",
        atom_species: Sequence[str | None] | None = None,
        shtyrov_params_path: str | None = None,
        rcut: float | None = None,
        periodic: bool = False,
    ):
        super().__init__()

        if isinstance(n_xyz, int):
            self.nx = self.ny = self.nz = n_xyz
        else:
            self.nx, self.ny, self.nz = n_xyz
        self.dx = dx
        self.progressbars = progressbars
        self.conv_backend = conv_backend
        # Captured before atom_species may get normalized from None to
        # [None]*N below (see get_3d_atomic_potentials) -- used to silence
        # the "fell back to Peng" warning when the user never provided
        # species typing at all (the common default case), while still
        # warning when they explicitly gave species and some didn't match.
        self._atom_species_explicitly_given = atom_species is not None
        # Both the voxelized-kernel and the analytic coefficient path derive
        # the Peng fallback from this builder's own atom set, so without this
        # each builder reported the same fact twice (see
        # _report_peng_fallback).
        self._peng_fallback_reported = False
        self.atom_species = atom_species
        self.shtyrov_params_path = shtyrov_params_path
        self.rcut = (
            rcut
            if rcut is not None
            else recommended_rcut(atomic_numbers, parameterization, atom_species)
        )
        self.periodic = periodic
        # Cache for method="analytic"'s per-atom (a, b) coefficients — built
        # lazily on first use since it needs a full atom_species lookup pass.
        self._analytic_coefs: tuple[torch.Tensor, torch.Tensor] | None = None

        self.ssn, self.ssdx, self.ssf = compute_supersampling_parameters(dx)
        sR_2d = radial_grid_2d(self.ssn, self.ssdx, convention="torch")
        sR_3d = radial_grid_3d(self.ssn, self.ssdx, convention="torch")
        self.register_buffer("sR_2d", sR_2d, persistent=False)
        self.register_buffer("sR_3d", sR_3d, persistent=False)

        self.atomic_numbers = atomic_numbers
        self.unique_elements = torch.unique(atomic_numbers)
        # Populated instead of unique_elements when parameterization is
        # 'shtyrov' and atom_species is given (see get_3d_atomic_potentials).
        self.shtyrov_groups: list[tuple[str, int | str]] | None = None
        atomic_potentials_2d = torch.empty(
            len(self.unique_elements), self.ssn // self.ssf, self.ssn // self.ssf
        )
        atomic_potentials_3d = torch.empty(
            len(self.unique_elements),
            self.ssn // self.ssf,
            self.ssn // self.ssf,
            self.ssn // self.ssf,
        )
        self.register_buffer("atomic_potentials_2d", atomic_potentials_2d)
        self.register_buffer("atomic_potentials_3d", atomic_potentials_3d)

        self.parameterization = parameterization
        self.avgpool2d = torch.nn.AvgPool2d(self.ssf, stride=self.ssf)
        self.avgpool3d = torch.nn.AvgPool3d(self.ssf, stride=self.ssf)
        if parameterization in ("kirkland", "lobato"):
            self.get_2d_atomic_potentials()
        self.get_3d_atomic_potentials()

    def get_2d_atomic_potentials(
        self, unique_elements: torch.Tensor | None = None
    ) -> None:
        """
        Compute and cache 2D atomic potential kernels for unique elements.

        Parameters
        ----------
        unique_elements : torch.Tensor, optional
            Elements to compute potentials for. If None, uses all unique
            elements from initialization. Default is None.

        Notes
        -----
        Potentials are supersampled and downsampled to main grid resolution.
        Results are stored in `self.atomic_potentials_2d`.
        """
        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            self.unique_elements = unique_elements
            self.atomic_potentials_2d = torch.empty(
                len(unique_elements), self.ssn // self.ssf, self.ssn // self.ssf
            )

        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == "kirkland":
                pot = kirkland_atomic_potential_2d(int(elem), self.sR_2d)
            elif self.parameterization == "lobato":
                pot = lobato_atomic_potential_2d(int(elem), self.sR_2d)
            else:
                raise ValueError(
                    f"Unknown parameterization '{self.parameterization}'. "
                    "Choose 'kirkland' or 'lobato'."
                )

            if self.ssf != 1:
                pot = self.avgpool2d(pot[None, None]).squeeze(0).squeeze(0)

            self.atomic_potentials_2d[i] = pot

    def get_3d_atomic_potentials(
        self, unique_elements: torch.Tensor | None = None
    ) -> None:
        """
        Compute and cache 3D atomic potential kernels for unique elements.

        Parameters
        ----------
        unique_elements : torch.Tensor, optional
            Elements to compute potentials for. If None, uses all unique
            elements from initialization. Default is None.

        Notes
        -----
        Potentials are supersampled and downsampled to main grid resolution.
        Results are stored in `self.atomic_potentials_3d`.
        Supports Kirkland, Lobato, and Shtyrov parameterizations. When
        `parameterization='shtyrov'`, kernels are always grouped by
        bonded-species descriptor instead of by element (see
        `_get_3d_shtyrov_species_potentials`) -- if `self.atom_species` is
        None, it is treated as `[None] * len(atomic_numbers)`, i.e. every
        atom falls back to plain per-element Peng `c4322` factors.
        """
        if self.parameterization == "shtyrov" and self.atom_species is None:
            self.atom_species = [None] * len(self.atomic_numbers)

        if self.parameterization == "shtyrov":
            self._get_3d_shtyrov_species_potentials()
            return

        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            # update unique elements
            self.unique_elements = unique_elements
            self.atomic_potentials_3d = torch.empty(
                len(unique_elements),
                self.ssn // self.ssf,
                self.ssn // self.ssf,
                self.ssn // self.ssf,
            )

        for i, elem in enumerate(self.unique_elements):
            pot = _cached_atomic_potential_kernel(
                (self.dx, self.parameterization, "element", int(elem)),
                functools.partial(
                    build_atomic_potential_kernel,
                    self.dx,
                    parameterization=self.parameterization,
                    atomic_number=int(elem),
                    sR=self.sR_3d,
                    avgpool3d=self.avgpool3d,
                    ssf=self.ssf,
                ),
            )

            self.atomic_potentials_3d[i] = pot

    def _report_peng_fallback(self, n_atoms: int, elements: Sequence[str]) -> None:
        """
        Report atoms that fell back to per-element Peng scattering factors.

        Warns at most once per process and logs the per-structure counts at
        DEBUG; see `_peng_fallback_warned` for why. Also collapses this
        builder's two independent discoveries of the same fact -- the
        voxelized-kernel and analytic coefficient paths each derive it from
        the same atom set -- into one record.

        Parameters
        ----------
        n_atoms : int
            How many of this structure's atoms fell back.
        elements : sequence of str
            Element symbols those atoms belong to.
        """
        if self._peng_fallback_reported:
            return
        self._peng_fallback_reported = True

        logger.debug(
            "%d atom(s) had no matching Shtyrov species (elements: %s); "
            "falling back to gemmi's built-in Peng et al. scattering factors "
            "for those atoms.",
            n_atoms,
            list(elements),
        )

        global _peng_fallback_warned
        if _peng_fallback_warned:
            return
        _peng_fallback_warned = True
        warnings.warn(
            "Some atoms had no matching Shtyrov species and fall back to "
            "gemmi's built-in Peng et al. scattering factors. The usual cause "
            "is incomplete bond typing, most often a missing Monomer Library "
            "(see PDB's monomer_library_path). Reported once per process; for "
            "per-structure atom counts and elements, call "
            "specter.set_verbosity(logging.DEBUG).",
            stacklevel=3,
        )

    def _get_3d_shtyrov_species_potentials(self) -> None:
        """
        Build 3D Shtyrov kernels grouped by bonded-species descriptor.

        Atoms whose `atom_species` entry is `None` or not present in the
        species parameter table fall back to a per-element, unbonded
        scattering factor instead (gemmi's built-in `c4322` Peng et al.
        table — the same fallback sffit itself uses for atom types missing
        from a fitted table, see `sffit/fit.py::do_mmcif`), with a single
        aggregated warning (no atom is silently dropped). Populates
        `self.shtyrov_groups` (a list of `("species", name)` /
        `("element", z)` tags, one per row of `self.atomic_potentials_3d`)
        for `forward` to consume.
        """
        params_path = self.shtyrov_params_path
        if params_path is None:
            params_path = str(
                resources.files("specter.atom_data").joinpath("params_cat.json")
            )
        species_table = load_shtyrov_species_parameters(params_path)

        assert self.atom_species is not None
        matched = sorted(
            {s for s in self.atom_species if s is not None and s in species_table}
        )
        unmatched_mask = torch.tensor(
            [s is None or s not in species_table for s in self.atom_species]
        )
        fallback_elements = sorted(
            {int(z) for z in self.atomic_numbers[unmatched_mask].tolist()}
        )

        if fallback_elements and self._atom_species_explicitly_given:
            self._report_peng_fallback(
                int(unmatched_mask.sum()),
                [str(atom_symbol(z)) for z in fallback_elements],
            )

        self.shtyrov_groups = [("species", s) for s in matched] + [
            ("element", z) for z in fallback_elements
        ]

        self.atomic_potentials_3d = torch.empty(
            len(self.shtyrov_groups),
            self.ssn // self.ssf,
            self.ssn // self.ssf,
            self.ssn // self.ssf,
        )
        for i, (kind, key) in enumerate(self.shtyrov_groups):
            if kind == "species":
                assert isinstance(key, str)
                pot = _cached_atomic_potential_kernel(
                    (self.dx, "shtyrov", "species", key, params_path),
                    functools.partial(
                        build_atomic_potential_kernel,
                        self.dx,
                        parameterization="shtyrov",
                        shtyrov_species=key,
                        species_table=species_table,
                        sR=self.sR_3d,
                        avgpool3d=self.avgpool3d,
                        ssf=self.ssf,
                    ),
                )
            else:
                assert isinstance(key, int)
                pot = _cached_atomic_potential_kernel(
                    (self.dx, "peng", "element", key),
                    functools.partial(
                        build_atomic_potential_kernel,
                        self.dx,
                        parameterization="peng",
                        atomic_number=key,
                        sR=self.sR_3d,
                        avgpool3d=self.avgpool3d,
                        ssf=self.ssf,
                    ),
                )

            self.atomic_potentials_3d[i] = pot

    def _cached_analytic_coefs(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Return the cached analytic coefficients, migrated if the module has moved.

        The cache is a plain attribute rather than a registered buffer, since it
        is built lazily on first use and its size depends on the structure. That
        keeps it invisible to ``nn.Module.to()``, so a builder first used on CPU
        and then moved to GPU would otherwise return CPU coefficients and fail
        mid-``forward`` with a device mismatch. Callers that rebuild the
        potential every forward pass (``ImageGeneratorFromCoordinates``) hit this
        on every GPU run.

        Returns
        -------
        tuple of torch.Tensor, or None
            The cached ``(a_coefs, b_coefs)`` on the module's current device, or
            None if nothing has been cached yet.
        """
        if self._analytic_coefs is None:
            return None
        a_coefs, b_coefs = self._analytic_coefs
        if a_coefs.device != self.device:
            self._analytic_coefs = (a_coefs.to(self.device), b_coefs.to(self.device))
        return self._analytic_coefs

    def _get_analytic_atom_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (and cache) per-atom Gaussian coefficients for `forward(method="analytic")`.

        Unlike the grouped-kernel path (`_get_3d_shtyrov_species_potentials`),
        analytic evaluation needs no shared kernel per group — each atom's
        own `(a, b)` coefficients (matched species, or the Peng `c4322`
        fallback) are looked up directly.

        Vectorized as: one O(N) pass over `self.atom_species` assigning each
        atom an integer group id (matched species string, or `("elem", z)`
        for the fallback case) via plain dict/set lookups (cheap, no tensor
        allocation per atom -- unlike an earlier version of this method,
        which called `gemmi.Element(...)`/`torch.tensor(...)` inside a
        Python loop over every individual atom; for a ~224k-atom structure
        that took ~8s, ~12x slower than the equivalent Kirkland analytic
        path on the same structure, entirely from this one loop), then a
        single vectorized gather from a small (n_groups, 5) coefficient
        table -- n_groups is bounded by the number of distinct species/
        elements actually present (rarely more than a few dozen), not by
        atom count.

        Returns
        -------
        a_coefs, b_coefs : torch.Tensor
            Per-atom Gaussian coefficients, shape (N, 5) each.
        """
        cached = self._cached_analytic_coefs()
        if cached is not None:
            return cached

        assert self.atom_species is not None
        params_path = self.shtyrov_params_path
        if params_path is None:
            params_path = str(
                resources.files("specter.atom_data").joinpath("params_cat.json")
            )
        species_table = load_shtyrov_species_parameters(params_path)

        group_of_atom: list[str | tuple[str, int]] = [
            s if (s is not None and s in species_table) else ("elem", int(z))
            for s, z in zip(self.atom_species, self.atomic_numbers.tolist())
        ]

        group_index: dict[str | tuple[str, int], int] = {}
        group_ids: list[int] = []
        for g in group_of_atom:
            idx = group_index.get(g)
            if idx is None:
                idx = len(group_index)
                group_index[g] = idx
            group_ids.append(idx)

        n_groups = len(group_index)
        a_table = torch.empty(n_groups, 5)
        b_table = torch.empty(n_groups, 5)
        fallback_elements: set[str] = set()
        fallback_group_id_list: list[int] = []
        for g, idx in group_index.items():
            if isinstance(g, tuple):
                z = g[1]
                coef = gemmi.Element(z).c4322
                a_table[idx] = torch.tensor(coef.a)
                b_table[idx] = torch.tensor(coef.b)
                fallback_elements.add(str(atom_symbol(z)))
                fallback_group_id_list.append(idx)
            else:
                P = species_table[g]
                a_table[idx] = P[:, 0]
                b_table[idx] = P[:, 1]

        group_ids_t = torch.tensor(group_ids, dtype=torch.long)
        n_fallback = 0
        if fallback_elements:
            fallback_group_ids = torch.tensor(fallback_group_id_list)
            n_fallback = int(torch.isin(group_ids_t, fallback_group_ids).sum().item())
        a_coefs = a_table[group_ids_t].to(self.device)
        b_coefs = b_table[group_ids_t].to(self.device)

        if fallback_elements and self._atom_species_explicitly_given:
            self._report_peng_fallback(n_fallback, sorted(fallback_elements))

        self._analytic_coefs = (a_coefs, b_coefs)
        return self._analytic_coefs

    def _get_analytic_peng_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (and cache) per-atom Peng `c4322` coefficients for
        `forward(method="analytic")` when `parameterization='shtyrov'` but no
        `atom_species` bonded typing was given — every atom uses its plain,
        unbonded per-element scattering factor (no species table lookup).

        Vectorized per unique element (see `_get_analytic_atom_coefficients`
        for why: same per-atom-Python-loop cost this method used to pay,
        fixed the same way) rather than per atom.

        Returns
        -------
        a_coefs, b_coefs : torch.Tensor
            Per-atom Gaussian coefficients, shape (N, 5) each.
        """
        cached = self._cached_analytic_coefs()
        if cached is not None:
            return cached

        atomic_numbers = self.atomic_numbers.to(self.device)
        unique_z, group_ids_t = torch.unique(atomic_numbers, return_inverse=True)

        a_table = torch.empty(len(unique_z), 5, device=self.device)
        b_table = torch.empty(len(unique_z), 5, device=self.device)
        for i, z in enumerate(unique_z.tolist()):
            coef = gemmi.Element(int(z)).c4322
            a_table[i] = torch.tensor(coef.a)
            b_table[i] = torch.tensor(coef.b)

        a_coefs = a_table[group_ids_t]
        b_coefs = b_table[group_ids_t]

        self._analytic_coefs = (a_coefs, b_coefs)
        return self._analytic_coefs

    def forward(
        self,
        coordinates: torch.Tensor,
        method: str = "analytic",
        conv_backend: str | None = None,
    ) -> torch.Tensor:
        """
        Build potential volume(s) from atomic coordinates.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates. Shape (N, 3) for single volume or (B, N, 3)
            for batch of volumes.
        method : str, optional
            Voxelization method: '2d' (soft XY, hard Z), '3d' (trilinear), or
            'analytic' (per-atom closed-form evaluation in a local window,
            no splat/FFT/kernel precomputation — supported for every
            `parameterization`: Kirkland/Lobato dispatch on `atomic_numbers`
            directly via `build_potential_volume_analytic_scatter_kirkland`/
            `_lobato`; Shtyrov dispatches on per-atom bonded species via
            `build_potential_volume_analytic_scatter` when `atom_species` is
            given, falling back to plain per-element Peng `c4322` factors
            for atoms without a species, or for every atom when
            `atom_species` is None entirely). Default is 'analytic'. Not
            supported with `periodic=True` (raises `ValueError`) — use
            `method='3d'` for periodic boundary conditions.
        conv_backend : str, optional
            Convolution backend override. Default is None (uses self.conv_backend).
            Unused for `method='analytic'`.

        Returns
        -------
        potential_volume : torch.Tensor
            Electrostatic potential volume(s). Shape (nz, ny, nx) for single
            input or (B, nz, ny, nx) for batched input.

        Notes
        -----
        '2d'/'3d' use soft voxelization followed by convolution with
        precomputed atomic potential kernels. The 2d method is faster but
        less accurate. 'analytic' skips voxelization/convolution/kernel
        precomputation entirely — dramatically faster than '3d' for typical
        atom counts (benchmarked ~26-90x on CPU, ~18-28x on GPU for a
        ~1600-atom structure) and fully differentiable w.r.t. `coordinates`.
        """
        if method == "analytic":
            if self.periodic:
                raise ValueError(
                    "method='analytic' does not implement periodic boundary "
                    "wrapping (unlike '2d'/'3d', which wrap out-of-bounds "
                    "voxel indices when periodic=True). Use method='3d' for "
                    "periodic boundary conditions."
                )
            coordinates = coordinates.to(self.device)
            if coordinates.ndim == 2:
                coordinates = coordinates.unsqueeze(0)
            B = coordinates.shape[0]

            if self.parameterization == "kirkland":
                atomic_numbers = self.atomic_numbers.to(self.device)
                analytic_fn = functools.partial(
                    build_potential_volume_analytic_scatter_kirkland, atomic_numbers
                )
            elif self.parameterization == "lobato":
                atomic_numbers = self.atomic_numbers.to(self.device)
                analytic_fn = functools.partial(
                    build_potential_volume_analytic_scatter_lobato, atomic_numbers
                )
            elif self.parameterization == "shtyrov":
                if self.atom_species is not None:
                    a_coefs, b_coefs = self._get_analytic_atom_coefficients()
                else:
                    a_coefs, b_coefs = self._get_analytic_peng_coefficients()
                analytic_fn = functools.partial(
                    build_potential_volume_analytic_scatter,
                    a_coefs=a_coefs,
                    b_coefs=b_coefs,
                )
            else:
                raise ValueError(
                    f"Unknown parameterization '{self.parameterization}'. "
                    "Choose 'kirkland', 'lobato', or 'shtyrov'."
                )

            potential_volume = torch.stack(
                [
                    analytic_fn(
                        coordinates[b],
                        grid_shape=(self.nz, self.ny, self.nx),
                        dx=self.dx,
                        rcut=self.rcut,
                    )
                    for b in range(B)
                ]
            )
            if B == 1:
                potential_volume = potential_volume.squeeze(0)
            return potential_volume

        if conv_backend is None:
            conv_backend = self.conv_backend
        coordinates = coordinates.to(self.device)

        if coordinates.ndim == 2:  # (N,3) -> add batch dimension
            coordinates = coordinates.unsqueeze(0)

        B, N, _ = coordinates.shape

        potential_volume = torch.zeros(
            (B, self.nz, self.ny, self.nx), device=self.device
        )

        if method == "2d" and self.shtyrov_groups is not None:
            raise ValueError(
                "method='2d' is not supported with species-grouped Shtyrov "
                "potentials (atom_species given). Use method='3d'."
            )

        groups: list[tuple[str, int | str]] = self.shtyrov_groups or [
            ("element", int(z)) for z in self.unique_elements
        ]

        with TqdmProgress(transient=True, disable=not self.progressbars) as progress:
            task = progress.add_task("Building element ...", total=len(groups))
            for i, (kind, key) in enumerate(groups):
                label: str
                if kind == "species":
                    assert isinstance(key, str)
                    assert self.atom_species is not None
                    label = key
                    mask = torch.tensor(
                        [s == key for s in self.atom_species], device=self.device
                    )
                else:
                    assert isinstance(key, int)
                    label = str(atom_symbol(key))
                    mask = self.atomic_numbers == key
                progress.update(
                    task,
                    description=f"Building element {label}",
                    advance=1,
                )
                atomic_indices = torch.argwhere(mask).squeeze(-1)
                coords_elem = coordinates[:, atomic_indices, :]  # (B, Nelem, 3)

                if method == "2d":
                    if self.periodic:
                        raise ValueError(
                            "periodic=True is not supported with method='2d'. "
                            "Use method='3d'."
                        )
                    temp_volume = soft_voxelize_xy_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx,
                    )
                    convolved_flat = spatial_convolve2d_same(
                        temp_volume.reshape(-1, self.ny, self.nx),
                        self.atomic_potentials_2d[i],
                    )
                    potential_volume += convolved_flat.reshape(
                        B, self.nz, self.ny, self.nx
                    )

                elif method == "3d":
                    temp_volume = soft_voxelize_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx,
                        periodic=self.periodic,
                    )
                    potential_volume += potential_from_deltas(
                        temp_volume, self.atomic_potentials_3d[i], backend=conv_backend
                    )
                else:
                    raise ValueError(f"Unknown method '{method}'. Choose '2d' or '3d'.")

        if B == 1:
            potential_volume = potential_volume.squeeze(0)
        return potential_volume
