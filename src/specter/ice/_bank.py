"""
Extracts ice sub-volumes from a cache of pre-generated, pre-converged ice
coordinate configurations, instead of running :class:`GradientSKIcemaker`'s
expensive L-BFGS optimisation per request.

See :func:`build_ice_cache` for how the cache itself is produced.
"""

from __future__ import annotations

import glob
import math
import os
from typing import TYPE_CHECKING

import lightning as L
import matplotlib.figure
import torch


from ..arrays import (
    soft_voxelize_coordinates,
)
from ..progress import track
from ..rotations import random_rotation_matrix_from_generator
from ._energy import MLBOP
from ..potential import (
    potential_from_deltas,
)
from ._kernels import build_water_kernel
from specter.options import IceModel, ScatteringFactors

if TYPE_CHECKING:
    pass

from ._library import FIXED_POINT_ENCODING, decode_positions, default_ice_cache_dir
from ._tiling import _IceTiling


class IceBank(_IceTiling, L.LightningModule):
    """
    Draws randomly rotated, randomly translated ice sub-volumes from a
    library of pre-generated ice coordinate configurations.

    Each cached config is a single :class:`GradientSKIcemaker` run's
    positions, converged under the standard MLBOP recipe (see
    :func:`build_ice_cache`) and stored as a periodic cubic cell. Since
    generation is a one-time cost (~tens of minutes for a production-scale
    config), this class turns "get a new, statistically independent ice
    volume" into a request costing single-digit milliseconds on GPU: pick a
    cached config, pick a random rotation and a random center, gather
    candidate atoms via periodic wraparound of the *source*, rotate them
    into the requested volume's own local frame, and soft-voxelize.

    Unlike :class:`GradientSKIcemaker`/:class:`RandomIcemaker`, whose
    ``(n, dx, nz)`` are fixed at construction, this class takes them
    per-call — the whole point is serving arbitrarily-sized requests from a
    fixed-size cache.

    Rotation is exact (no interpolation) since it's applied to continuous
    coordinates before voxelization, unlike rotating an already-voxelized
    grid.

    Two request sizes are handled differently:

    - **Fits within a single cached config** (``n*dx``, ``n*dx``, ``nz*dx``
      each <= some cached config's own box size): use
      :meth:`generate_ice`/:meth:`generate_ice_deltas` -- a single crop,
      no relaxation needed, cheapest path.
    - **Larger than any single cached config**: use
      :meth:`generate_big_ice`/:meth:`generate_big_ice_deltas` -- tiles
      multiple independently rotated crops together and heals the tile
      boundaries with a short local MLBOP relaxation (naive concatenation
      alone leaves severe damage at the seams).

    Parameters
    ----------
    cache_dir : str, optional
        Directory containing cached config ``.pt`` files (see
        :func:`build_ice_cache`). Every ``*.pt`` file in this directory is
        treated as a cache entry. Defaults to the bundled
        ``ice_data/ice_cache`` shipped with the repository.
    device : str or torch.device, optional
        Computation device. Default is ``"cpu"``.
    parameterization : str, optional
        Atomic scattering-factor parameterization for the ice kernel:
        ``'kirkland'``, ``'lobato'``, or ``'shtyrov'``. Default ``'kirkland'``
        -- ice is a bulk material, outside the biomolecular domain Shtyrov is
        fitted for (see `build_water_kernel`). Note this only
        affects the kernel used to voxelize a crop's coordinates -- it does
        not change which cached configs (already-optimized coordinate sets)
        are drawn from.
    progressbars : bool, optional
        Whether to show progress bars over batched generation. Default is
        True.
    """

    method: IceModel = "gd"

    """str: Always ``"gd"`` -- every cached config was generated via
    :class:`GradientSKIcemaker`. Kept for compatibility with code that reads
    ``icemaker.method`` (e.g. to log/report which ice model an
    externally-constructed icemaker uses)."""

    def __init__(
        self,
        cache_dir: str | None = None,
        device: str | torch.device = "cpu",
        parameterization: ScatteringFactors = "kirkland",
        progressbars: bool = True,
    ) -> None:
        super().__init__()
        if cache_dir is None:
            cache_dir = default_ice_cache_dir()
        self.cache_dir = cache_dir
        self.parameterization = parameterization
        self._config_paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self._config_paths:
            raise FileNotFoundError(
                f"No cached ice configs (*.pt) found in {cache_dir!r}"
            )
        # Loaded eagerly (not lazily): cache configs are small (a few MB each,
        # even smaller on disk at float16) and every config's box_L is needed
        # up front to pick a config that actually fits a given request --
        # checking box size *after* randomly picking a config would make
        # in-range requests non-deterministically fail depending on which
        # config got drawn, for any cache mixing config sizes.
        self._configs: list[dict] = [self._load_config(p) for p in self._config_paths]
        self._kernel_cache: dict[float, torch.Tensor] = {}
        self._source_pos_cache: dict[int, tuple[torch.device, torch.Tensor]] = {}
        self.progressbars = progressbars

        self.positions: torch.Tensor | None = None
        """torch.Tensor or None: The most recently extracted crop's
        coordinates (N, 3), in the crop's own local frame, centered at the
        origin. Non-periodic -- unlike the other icemaker classes' outputs, a
        crop is a finite chunk of a larger periodic source, not periodic on
        its own."""
        self.current_icedeltas: torch.Tensor | None = None
        """torch.Tensor or None: The most recently generated batch's
        soft-voxelized deltas."""
        self.n: int | None = None
        self.dx: float | None = None
        self.nz: int | None = None
        self.box_x: float | None = None
        self.box_y: float | None = None
        self.box_z: float | None = None

        if torch.device(device).type != "cpu":
            self.to(device)

    def __len__(self) -> int:
        return len(self._config_paths)

    def _load_config(self, path: str) -> dict:
        data = torch.load(path, weights_only=False)
        # Two on-disk coordinate formats, distinguished by an explicit key
        # rather than by dtype: fixed-point indices (see `encode_positions`)
        # for anything generated since that encoding landed, raw floats for
        # everything before it -- including the bundled `ice_data/ice_cache`,
        # which must keep loading unchanged. Sniffing `int16` instead of
        # reading the key would misread any future raw-integer format.
        if data.get("coord_encoding") == FIXED_POINT_ENCODING:
            data["positions"] = decode_positions(data["positions"], data["box_L"])
        else:
            data["positions"] = data["positions"].float()  # upcast from float16
        return data

    def _get_kernel(self, dx: float) -> torch.Tensor:
        if dx not in self._kernel_cache:
            self._kernel_cache[dx] = build_water_kernel(dx, self.parameterization)
        return self._kernel_cache[dx]

    def _get_source_pos(self, config: dict) -> torch.Tensor:
        """
        Device-resident copy of ``config["positions"]``, cached per config.

        ``config["positions"]`` itself always stays on CPU (that's the
        eagerly-loaded copy from disk, shared regardless of which device
        this instance is on). The crop-extraction math in
        :meth:`_extract_crop` is otherwise CPU-only even when
        ``self.device`` is a GPU -- moving just the candidate-gathering
        computation there is a real win when many small crops are drawn in
        a loop (as :meth:`_place_tiles` does): ~4x faster per call in
        practice, and cheap in memory (a single ~527k-atom config is a few
        MB, and the transient candidate tensor per call is ~150-350 MB,
        trivial next to a modern GPU's total memory). Invalidated and
        rebuilt if ``self.device`` changes between calls.
        """
        key = id(config)
        cached = self._source_pos_cache.get(key)
        if cached is not None and cached[0] == self.device:
            return cached[1]
        pos = config["positions"].to(self.device)
        self._source_pos_cache[key] = (self.device, pos)
        return pos

    def _extract_crop(
        self,
        crop_extent: tuple[float, float, float],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Extract one randomly rotated, randomly translated crop.

        Parameters
        ----------
        crop_extent : tuple of float
            Requested ``(box_x, box_y, box_z)`` in Å.

        Returns
        -------
        torch.Tensor
            Shape (N, 3), local frame, centered at the origin.
        """
        max_extent = max(crop_extent)
        candidates = [c for c in self._configs if c["box_L"] >= max_extent]
        if not candidates:
            largest = max(c["box_L"] for c in self._configs)
            raise ValueError(
                f"Requested crop extent {crop_extent} Å exceeds every cached "
                f"config's own box size (largest available: {largest} Å) along "
                "at least one axis. This method only crops from a single "
                "config; use generate_big_ice()/generate_big_ice_deltas() "
                "instead for volumes larger than the cache."
            )
        idx = int(torch.randint(len(candidates), (1,), generator=generator).item())
        config = candidates[idx]
        box_L = config["box_L"]

        # Random draws stay on CPU regardless of self.device -- generator
        # (if given) is a CPU torch.Generator by convention throughout this
        # module/its tests, and these are tiny tensors anyway. Only the
        # actual candidate-gathering math below (O(n_source_atoms x
        # n_periodic_images), the real cost for small crops out of a large
        # source -- see _get_source_pos) runs on self.device.
        center_cpu = (torch.rand(3, generator=generator) - 0.5) * box_L
        R_cpu = random_rotation_matrix_from_generator(generator)

        source_pos = self._get_source_pos(config)
        half_extent = torch.tensor(
            [e / 2 for e in crop_extent], device=source_pos.device
        )
        reach = float(
            half_extent.norm()
        )  # corner-to-center distance of the (possibly non-cubic) crop box
        center = center_cpu.to(source_pos.device)
        R = R_cpu.to(source_pos.device)

        # Gather candidates via periodic wraparound of the source, using
        # enough periodic images to safely cover `reach` regardless of how
        # it compares to box_L. For center c and atom a each confined to
        # [-box_L/2, box_L/2), the offset c - a spans (-box_L, box_L), so
        # the largest image index that can ever matter is the largest
        # integer strictly below (box_L + reach) / box_L = 1 + reach/box_L,
        # i.e. m = floor(1 + reach/box_L) -- equivalently
        # 1 + floor(reach/box_L), which is always >= 1 (never just the
        # unwrapped primary cell, even for small reach: a center and atom
        # near opposite box edges still need the neighboring image to see
        # each other). The tempting m = ceil(reach/box_L - 0.5) silently
        # gives m=0 (no wraparound at all) whenever reach <= box_L/2, which
        # is wrong for any center near a box edge and shows up as a sharp,
        # reach/box_L-dependent gap in atom count.
        m = 1 + math.floor(reach / box_L)
        shifts = (
            torch.arange(-m, m + 1, dtype=source_pos.dtype, device=source_pos.device)
            * box_L
        )
        sx, sy, sz = torch.meshgrid(shifts, shifts, shifts, indexing="ij")
        image_offsets = torch.stack([sx.flatten(), sy.flatten(), sz.flatten()], dim=1)

        candidates = (
            source_pos.unsqueeze(0) + image_offsets.unsqueeze(1) - center
        ).reshape(-1, 3)
        candidates = candidates[candidates.norm(dim=1) <= reach]

        local = candidates @ R
        in_crop = (local.abs() <= half_extent).all(dim=1)
        # Left on the device the gather ran on. Every consumer either splats
        # it into a device-resident grid, filters it, or feeds it to MLBOP on
        # `positions.device`; a `.cpu()` here cost a synchronising 6 MB
        # pageable copy per tile and then ran the tile's mask/offset/cat
        # arithmetic on the host, for nothing.
        return local[in_crop]

    def generate_ice_deltas(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        batchsize: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Extract ``batchsize`` independent crops and soft-voxelize them.

        Each crop draws a fresh random config choice, rotation, and
        translation. The last crop's coordinates are left in
        ``self.positions`` (for :meth:`mlbop_energy`); the full batch is
        stored in ``self.current_icedeltas``.

        Parameters
        ----------
        n : int
            Number of voxels along x and y.
        dx : float
            Voxel size in Å.
        nz : int, optional
            Number of voxels along z. Defaults to ``n``.
        batchsize : int, optional
            Number of independent crops. Default is 1.
        generator : torch.Generator, optional
            RNG for reproducibility. Default is the global RNG.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape (batchsize, nz, n, n).
        """
        self.n, self.dx, self.nz = n, dx, nz if nz is not None else n
        self.box_x = self.n * self.dx
        self.box_y = self.n * self.dx
        self.box_z = self.nz * self.dx
        crop_extent = (self.box_x, self.box_y, self.box_z)

        results = []
        for _ in track(
            range(batchsize),
            description="Extracting ice crops",
            disable=not self.progressbars or batchsize == 1,
            transient=True,
        ):
            self.positions = self._extract_crop(crop_extent, generator=generator)
            # periodic=True here isn't claiming the crop is self-periodic -- it's a
            # fix for a splat-grid fencepost bug: origin=n//2 centers voxel 0 exactly
            # at -half_extent (no atoms exist below it) while voxel n-1 sits a full
            # voxel inside +half_extent (atoms on both sides), so voxel 0 gets only
            # ~half the local density of an interior voxel and voxel n-1 gets nearly
            # all of it. Wrapping the overflow that would otherwise drop past index n
            # back onto index 0 restores voxel 0's missing "neighbor below"
            # contribution -- measured to bring low-face density from ~0.49-0.50x
            # interior up to ~0.92-0.94x, matching the high-face level. Cheap:
            # the same index-tensor op as a mask, no extra cost.
            vox = soft_voxelize_coordinates(
                self.positions,
                grid_shape=(self.nz, self.n, self.n),
                voxel_size=self.dx,
                periodic=True,
            )
            results.append(vox)
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_ice(
        self,
        n: int,
        dx: float,
        nz: int | None = None,
        batchsize: int = 1,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Extract ``batchsize`` ice sub-volumes and convolve with the atomic
        potential kernel.

        Parameters
        ----------
        n : int
            Number of voxels along x and y.
        dx : float
            Voxel size in Å.
        nz : int, optional
            Number of voxels along z. Defaults to ``n``.
        batchsize : int, optional
            Number of independent ice volumes. Default is 1.
        device : torch.device or str, optional
            Target device for the output. Default is ``self.device``.
        generator : torch.Generator, optional
            RNG for reproducibility. Default is the global RNG.

        Returns
        -------
        icecubes : torch.Tensor
            Convolved ice potential volumes, shape (batchsize, nz, n, n).
        """
        if device is None:
            device = self.device
        self.generate_ice_deltas(
            n=n, dx=dx, nz=nz, batchsize=batchsize, generator=generator
        )
        assert self.current_icedeltas is not None
        kernel = self._get_kernel(dx)
        return potential_from_deltas(
            self.current_icedeltas.to(device),
            kernel.to(device),
            backend="auto",
            boundary="periodic",
        )

    def mlbop_energy(self, pbc: bool = False) -> dict[str, float]:
        """
        Score the last-extracted crop against the ML-BOP potential.

        See :meth:`specter.ice._energy.MLBOP.compute_energy`. Defaults to
        non-periodic (``pbc=False``), unlike the other icemaker classes --
        an extracted crop is a finite chunk of a larger periodic source, not
        periodic on its own (see :meth:`generate_ice_deltas`). Pass
        ``pbc=True`` only if you know the requested extent exactly matches
        the source config's own box size.

        Parameters
        ----------
        pbc : bool, optional
            Whether to treat the crop as periodic. Default is False.

        Returns
        -------
        dict[str, float]
            See :meth:`specter.ice._energy.MLBOP.compute_energy` for the
            fields returned.
        """
        assert self.positions is not None, (
            "No positions -- call generate_ice_deltas() first"
        )
        assert (
            self.box_x is not None and self.box_y is not None and self.box_z is not None
        )
        box = (self.box_x, self.box_y, self.box_z)
        model = MLBOP(device=self.positions.device)
        with torch.no_grad():
            result = model.compute_energy(self.positions, box_size=box, pbc=pbc)
        return {k: v.item() for k, v in result.items()}

    def plot_diagnostics(
        self, save_path: str | None = None, show: bool = True
    ) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
        """
        Compute ML-BOP energy/structure stats and S(k) radial profiles for
        every config in the cache, and plot them for a quick visual
        quality check of the library (e.g. after (re)running
        :func:`build_ice_cache`).

        Each config is scored with ``pbc=True`` (unlike :meth:`mlbop_energy`'s
        own ``pbc=False`` default) since a full cached config -- unlike an
        extracted crop -- genuinely is the periodic cell it was optimised
        as, and on ``self.device`` (under ``torch.no_grad()`` -- no
        gradients are needed here): scoring a full ~527k-atom config takes
        ~0.2s on GPU, so across 20 configs this stays a quick quality check
        rather than several minutes of CPU time.

        Parameters
        ----------
        save_path : str, optional
            If given, saves the energy-summary plot to
            ``f"{save_path}_energy.png"`` and the S(k) overlay to
            ``f"{save_path}_sk.png"``.
        show : bool, optional
            Whether to display the plots. Default True.

        Returns
        -------
        energy_fig, sk_fig : matplotlib.figure.Figure
            The two diagnostic figures.
        """
        from ._diagnostics import plot_ice_bank_diagnostics

        return plot_ice_bank_diagnostics(self, save_path, show)
