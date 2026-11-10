"""
Public membrane specimen generator.

Ties the membrane submodules together into one class: the organelle recipe
(``_organelles``) composed from skeleton primitives (``_primitives``) and
regularised on a coarse grid (``_geometry``), the calibrated bilayer
potential (``_profile``) rendered in a narrow band about each mid-surface
(``_render``), and normal-aligned transmembrane protein insertion
(``_placement``).

Everything shares one centred physical frame: physical ``(0, 0, 0)`` is the
organelle's centre and lies on voxel ``n // 2`` of each output axis. This
generator does not place a membrane off-centre within a larger tomogram;
that compositing step belongs to
``specter.specimen.tomogram.TomogramSpecimenGenerator``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from specter.options import MembraneKind, ScatteringFactors

from ...arrays import clip_insert_bounds
from ...pdb import DEFAULT_PDB_CACHE_DIR, PDB
from ...random_seed import derive_seed
from ...rotations import build_affine_matrix, fold_subvoxel_offsets, rotate_volume
from .._parallel_render import (
    build_templates_concurrently,
    resolve_render_devices,
    resolve_render_workers,
)
from ..packing import estimate_protein_box_size
from ._field import MembraneField, _centred_origin_xyz
from ._geometry import Grid, ReachError
from ._organelles import OrganellePlan, plan_organelle
from ._placement import (
    align_principal_axis_to_z,
    align_transmembrane_depth,
    orientation_for_normal,
    sample_surface_sites,
)
from ._profile import (
    BilayerProfile,
    build_measured_bilayer_profile,
)
from ._render import render_bilayer

# Voxels of rotated transmembrane templates held at once while placing them
# (512 MB of float32): `place_transmembrane` rotates a chunk of sites per
# batched `rotate_volume` call rather than one site per call.
_TRANSMEMBRANE_ROTATION_BATCH_VOXELS = 2**27

#: Default fraction of the bilayer's own peak density above which a
#: transmembrane protein is treated as having displaced the lipid it sits
#: in -- see `MembraneGenerator.transmembrane_occupancy_fraction`. Named
#: because `_insert_blend` also falls back to it for LABELLING when
#: displacement is switched off entirely.
_DEFAULT_TRANSMEMBRANE_OCCUPANCY_FRACTION = 0.05

# Spacing of the coarse grid the geometry is built on, A. A third of the
# smallest radius of curvature the regularisation leaves (30 A).
_COARSE_SPACING = 10.0
# Most coarse-grid nodes one membrane may take. The geometry holds about
# 0.4 GiB per million nodes while it is regularised; past this the grid is
# coarsened, which rounds the tightest features more.
_MAX_COARSE_NODES = 24_000_000
# How far the bilayer's potential reaches from the mid-surface, A; the
# margin an auto-sized output box leaves around the organelle.
_BILAYER_MARGIN = 45.0
# Times the path-dependent randomness of a kind (undulation, a tubule's
# walk) is redrawn when a draw cannot be regularised to a kink-free surface.
_BUILD_ATTEMPTS = 4


@dataclass
class TransmembraneSpec:
    """
    One transmembrane protein species to attempt placing in the membrane.

    Parameters
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF path. Used as the species id and, when
        ``template`` is not supplied, as the input to ``PotentialBuilder``.
    frequency : int, optional
        Relative placement weight among species. Default 1.
    tm_span_mask : torch.Tensor, optional
        Boolean mask, same atom order as the loaded structure's
        coordinates, selecting the transmembrane span. Default None, which
        centers the full structure's z-extent instead (see
        :func:`~specter.specimen.membrane._placement.align_transmembrane_depth`).
        Ignored when ``template`` is supplied -- depth alignment must
        already be baked into a prebuilt template.
    parameterization : str, optional
        PotentialBuilder parameterization for PDB-backed specs. Default
        "shtyrov".
    template : torch.Tensor, optional
        Prebuilt density template with shape ``(Z, Y, X)``, already
        depth-aligned (transmembrane span centered on the template's own
        z-center). If supplied, no PDB is loaded for this species.
    """

    pdb_source: str
    frequency: int = 1
    tm_span_mask: torch.Tensor | None = None
    parameterization: ScatteringFactors = "shtyrov"
    template: torch.Tensor | None = None


def render_transmembrane_template(
    spec: TransmembraneSpec,
    voxel_size: float,
    pdb_cache_dir: str,
    device: str | torch.device,
    readd_hydrogens: bool | str = "auto",
    monomer_library_path: str | None = None,
    use_deposited_bfactors: bool = False,
) -> torch.Tensor:
    """
    Build one transmembrane species' potential template from its PDB source.

    Pure/stateless (no `MembraneGenerator` instance involved) so a caller
    driving several `MembraneGenerator`s that all share the same
    `transmembrane_specs` -- e.g. `n_copies` copies of one `[[membrane]]`
    entry in `specter build tomogram` -- can call this ONCE per species and
    pass the result via `TransmembraneSpec.template`, instead of letting
    each instance's own `MembraneGenerator._build_template` rebuild the same
    species redundantly. `MembraneGenerator._build_template` itself is a
    thin device-routing wrapper around this function.

    Parameters
    ----------
    spec : TransmembraneSpec
        Species to render (``spec.template``, if set, is ignored here --
        callers that already have a template have no reason to call this).
    voxel_size : float
        Output voxel size, Å.
    pdb_cache_dir : str
        Passed to `PDB` for fetching/caching `spec.pdb_source`.
    device : str or torch.device
        Device to build the `PotentialBuilder` (and run its `forward`) on.
    readd_hydrogens : {"auto", True, False}, optional
        Forwarded to `PDB`; see `specter.pdb.PDB`. Only meaningful for
        `spec.parameterization == "shtyrov"` with a Monomer Library available.
    monomer_library_path : str, optional
        Forwarded to `PDB`; see `specter.pdb.PDB`. Unset falls back to
        `$CLIBD_MON`.
    use_deposited_bfactors : bool, optional
        Damp each atom by the B-factor its structure deposits, rather than
        rendering the model statically. Requires
        `spec.parameterization == "shtyrov"`; see `PotentialBuilder`'s own
        `b_factors` for why the other backends refuse it, and why a
        deposited column is not a measured displacement. Default False.

    Returns
    -------
    torch.Tensor
        Density template, shape ``(Z, Y, X)``, on `device`.
    """
    pdb = PDB(
        spec.pdb_source,
        pdb_cache_dir=pdb_cache_dir,
        verbose=False,
        # Shtyrov fits scattering factors per bonded species, so it needs the
        # bond topology to beat plain per-element factors; the other
        # parameterizations are per-element by construction.
        compute_atom_species=spec.parameterization == "shtyrov",
        readd_hydrogens=readd_hydrogens,
        monomer_library_path=monomer_library_path,
    )
    coordinates = align_principal_axis_to_z(pdb.coordinates)
    coordinates = align_transmembrane_depth(coordinates, spec.tm_span_mask)
    n = estimate_protein_box_size(pdb.max_diameter, voxel_size)
    # Imported here: specter.imaging imports this package.
    from ...imaging import Structure

    # "analytic" (PotentialBuilder's own documented default), not "3d" --
    # confirmed empirically the two methods integrate to the same total
    # potential (ratio ~1.001 across a 10x span of voxel sizes, on a real
    # structure) but "3d"'s convolution-based approach spreads that same
    # total over a visibly lower, smoother peak (25-45% lower max value
    # than "analytic" for identical atoms). The bilayer profile is
    # rendered with "analytic"; using "3d" here compared two differently-
    # sharpened renderings of the same underlying physics, not a
    # deliberate contrast difference between membrane and protein.
    # align_principal_axis_to_z/align_transmembrane_depth only rotate and
    # translate, so species stay in step with coordinates.
    return (
        Structure(pdb, spec.parameterization)
        .render(
            voxel_size,
            n,
            coordinates=coordinates,
            use_deposited_bfactors=use_deposited_bfactors,
            device=device,
        )
        .potential
    )


@dataclass
class TransmembranePlacement:
    """One accepted transmembrane placement."""

    species_id: str
    center_xyz: torch.Tensor
    rotation_matrix: torch.Tensor


class MembraneGenerator:
    """
    Generate one organelle's membranes, with transmembrane inserts.

    An organelle is drawn as a `kind`, a recipe over thickened-skeleton
    primitives whose settings are measurable sizes. The solid it composes
    is regularised on a coarse grid, so that no membrane curves tighter
    than 30 Å or passes within 50 Å of another, and the calibrated bilayer
    potential is rendered about each mid-surface at the output voxel size.

    Parameters
    ----------
    target_shape : tuple of int, optional
        Output volume shape, ``(Z, Y, X)``. Default None: sized to hold the
        organelle that was drawn. A given shape is used as it is; a
        membrane that reaches past it is cut at the faces and
        `clipped_at_boundary` is set.
    voxel_size : float, optional
        Output voxel size, Å. Also used to render transmembrane protein
        templates. Default 5.0.
    kind : {"vesicle", "tubules", "cisterna", "mitochondrion"}, optional
        The organelle recipe. ``"vesicle"`` is a closed, gently undulating
        spheroid, optionally with nested layers; ``"tubules"`` a branching
        network of tubes; ``"cisterna"`` a flattened, curved sac, optionally
        fenestrated; ``"mitochondrion"`` a capsule with an outer membrane
        and an inner boundary membrane folded into lamellar cristae.
        Default ``"vesicle"``.
    n_lipids_per_leaflet : int, optional
        Reference lipid patch size, for callers building a patch of their
        own with
        :func:`~specter.specimen.membrane._profile.build_reference_lipid_patch`.
        It has no effect on `generate`. Default 200.
    parameterization : str, optional
        Scattering factors of the lipid reference patch and the
        transmembrane templates. Default "shtyrov".
    bilayer_thickness : float, optional
        Phosphate-to-phosphate spacing, Å. The measured profile
        (:func:`~specter.specimen.membrane._profile.build_measured_bilayer_profile`)
        is rescaled along its normal to this value at fixed amplitude, so
        the integrated potential scales with it. Default 37.0, the spacing
        measured for fully hydrated fluid POPC at 30 °C [2]_; fluid
        phosphatidylcholines span 35-38 Å [2]_ [3]_.
    bilayer_layer_sigma_angstrom : float, optional
        Additional Gaussian broadening of the profile along the normal, Å.
        Default 0.0.
    transmembrane_specs : list of TransmembraneSpec, optional
        Transmembrane protein species to place. Default None.
    transmembrane_occupancy_fraction : float, optional
        Fraction of the bilayer's peak potential above which a protein
        template displaces the lipid it sits in instead of adding to it;
        see `_insert_blend` for the blend between the two. Default 0.05.
    pdb_cache_dir : str, optional
        Passed to `PDB` for PDB-backed transmembrane specs.
    readd_hydrogens : bool or "auto", optional
        Passed to `PDB`. Default "auto".
    monomer_library_path : str, optional
        Passed to `PDB`. Default None.
    use_deposited_bfactors : bool, optional
        Whether transmembrane templates are blurred by their deposited
        B-factors. Default False.
    device : str or torch.device, optional
        Device for generation. Default "cpu".
    seed : int, optional
        Random seed. Default None: drawn from torch's global generator.
    render_workers : int or "auto", optional
        Number of transmembrane species rendered concurrently. Default 1.
    render_devices : list of str or torch.device, optional
        Device pool those species are spread across. Default None:
        `device`.
    max_radius : float, optional
        Largest `bounding_radius` the organelle may have, Å. A draw that
        exceeds it has the size settings left at their defaults scaled
        down to fit; a size given in `kind_settings` is never changed.
        Default None: no limit.
    **kind_settings
        The settings of `kind`, each a scalar or a ``[low, high]`` range
        from which this membrane's value is drawn, in Å or as a count:

        - ``"vesicle"``: ``radius``, ``elongation`` (long axis over short,
          at least 1), ``undulation`` (amplitude), ``layers``.
        - ``"tubules"``: ``radius``, ``length`` (of each branch),
          ``branches``.
        - ``"cisterna"``: ``diameter``, ``lumen_width``, ``fenestrations``.
        - ``"mitochondrion"``: ``radius``, ``length`` (of the axis between
          the end caps), ``crista_spacing``, ``membrane_gap`` (between the
          outer and inner boundary membranes).

        An omitted setting takes the default in
        `~specter.specimen.membrane.ORGANELLE_DEFAULTS`. A setting of
        another kind is refused.

    Attributes
    ----------
    params : dict
        The value drawn for each setting of `kind`.
    bounding_radius : float
        Radius of a sphere about the organelle's centre that holds its
        membranes, Å. Known before `generate`.
    field : MembraneField or None
        Signed distance to the outermost membrane's mid-surface after
        `generate`, on the coarse grid.
    clipped_at_boundary : bool or None
        Whether the outermost membrane reaches past the output box.

    Notes
    -----
    Transmembrane proteins are placed on the outermost membrane only.

    References
    ----------
    .. [1] Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024).
       Simulating the cellular context in synthetic datasets for cryo-electron
       tomography. *IEEE Transactions on Medical Imaging*, 43(11), 3742–3754.
       https://doi.org/10.1109/TMI.2024.3398401
    .. [2] Kučerka, N., Tristram-Nagle, S., & Nagle, J. F. (2005). Structure of
       fully hydrated fluid phase lipid bilayers with monounsaturated chains.
       *Journal of Membrane Biology*, 208(3), 193–202.
       https://doi.org/10.1007/s00232-005-7006-8
    .. [3] Nagle, J. F., & Tristram-Nagle, S. (2000). Structure of lipid
       bilayers. *Biochimica et Biophysica Acta (BBA) – Reviews on
       Biomembranes*, 1469(3), 159–195.
       https://doi.org/10.1016/S0304-4157(00)00016-2
    """

    def __init__(
        self,
        target_shape: tuple[int, int, int] | None = None,
        voxel_size: float = 5.0,
        kind: MembraneKind = "vesicle",
        *,
        n_lipids_per_leaflet: int = 200,
        parameterization: ScatteringFactors = "shtyrov",
        bilayer_thickness: float = 37.0,
        bilayer_layer_sigma_angstrom: float = 0.0,
        transmembrane_specs: list[TransmembraneSpec] | None = None,
        transmembrane_occupancy_fraction: float = _DEFAULT_TRANSMEMBRANE_OCCUPANCY_FRACTION,
        pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
        readd_hydrogens: bool | str = "auto",
        monomer_library_path: str | None = None,
        use_deposited_bfactors: bool = False,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        render_workers: int | Literal["auto"] = 1,
        render_devices: list[str | torch.device] | None = None,
        max_radius: float | None = None,
        **kind_settings: Any,
    ):
        self.voxel_size = float(voxel_size)
        self.kind: MembraneKind = kind
        self.seed = seed
        # One base seed for the sizes and for every redraw of the geometry.
        self._base_seed = (
            int(seed)
            if seed is not None
            else int(torch.randint(0, 2**31 - 1, (1,)).item())
        )
        self._plan: OrganellePlan = plan_organelle(
            kind,
            np.random.default_rng(derive_seed(self._base_seed, 0)),
            max_radius=None if max_radius is None else max_radius - _BILAYER_MARGIN,
            **kind_settings,
        )
        if target_shape is None:
            reach = self._plan.half_extent + _BILAYER_MARGIN
            nx, ny, nz = (2 * int(math.ceil(r / self.voxel_size)) + 2 for r in reach)
            target_shape = (nz, ny, nx)
        tz, ty, tx = target_shape
        self.target_shape: tuple[int, int, int] = (int(tz), int(ty), int(tx))
        self.n_lipids_per_leaflet = n_lipids_per_leaflet
        self.parameterization = parameterization
        self.bilayer_thickness = bilayer_thickness
        self.bilayer_layer_sigma_angstrom = bilayer_layer_sigma_angstrom
        self.transmembrane_specs = transmembrane_specs or []
        self.transmembrane_occupancy_fraction = transmembrane_occupancy_fraction
        self.pdb_cache_dir = pdb_cache_dir
        self.readd_hydrogens = readd_hydrogens
        self.use_deposited_bfactors = use_deposited_bfactors
        self.monomer_library_path = monomer_library_path
        self.device = torch.device(device)
        self.render_workers = resolve_render_workers(
            render_workers, len(self.transmembrane_specs)
        )
        self.render_devices = resolve_render_devices(self.device, render_devices)

        self.field: MembraneField | None = None
        self.profile: BilayerProfile | None = None
        self.volume: torch.Tensor | None = None
        # Which voxels each transmembrane protein displaced lipid from,
        # 1-based and local to this membrane; 0 is lipid or empty. Set by
        # `place_transmembrane`, and the basis for BOTH ground-truth
        # labels: the proteins are these voxels, the bilayer is the rest of
        # the rendered shell. None until `place_transmembrane` runs (and
        # when there are no transmembrane specs at all).
        self.transmembrane_labels: torch.Tensor | None = None
        self.placements: list[TransmembranePlacement] = []
        self.clipped_at_boundary: bool | None = None
        self._origin_xyz: torch.Tensor | None = None

    @property
    def params(self) -> dict[str, float]:
        """The value drawn for each setting of `kind`."""
        return dict(self._plan.params)

    @property
    def bounding_radius(self) -> float:
        """Radius of a sphere about the centre holding the membranes, Å."""
        return self._plan.bounding_radius + _BILAYER_MARGIN

    def _coarse_grid(self) -> Grid:
        """The grid the geometry is built on, covering the output box."""
        nz, ny, nx = self.target_shape
        box = (nx * self.voxel_size, ny * self.voxel_size, nz * self.voxel_size)
        assert self._origin_xyz is not None
        lo = tuple(float(v) for v in self._origin_xyz)
        grid = Grid(box, h=_COARSE_SPACING, lo=lo, device=self.device)
        if grid.n_nodes > _MAX_COARSE_NODES:
            h = _COARSE_SPACING * (grid.n_nodes / _MAX_COARSE_NODES) ** (1 / 3)
            warnings.warn(
                f"MembraneGenerator: a {box[0]:.0f} x {box[1]:.0f} x "
                f"{box[2]:.0f} A membrane box needs {grid.n_nodes:,} geometry "
                f"nodes at {_COARSE_SPACING:.0f} A; building it at {h:.1f} A "
                "instead, which rounds the tightest membrane features more.",
                stacklevel=3,
            )
            grid = Grid(box, h=h, lo=lo, device=self.device)
        return grid

    def generate(self) -> torch.Tensor:
        """
        Build the organelle's membranes and render them into ``self.volume``.

        Sets ``self.profile``, ``self.field`` (the outermost membrane) and
        ``self.clipped_at_boundary``.

        Returns
        -------
        torch.Tensor
            Potential volume, shape ``target_shape``.

        Raises
        ------
        ReachError
            If no redraw of the kind's path-dependent randomness could be
            regularised to a kink-free surface.
        """
        # The MEASURED psi(z), not a two-Gaussian form: two Gaussians on
        # vacuum discard the acyl core, which carries most of the bilayer's
        # integrated potential. See build_measured_bilayer_profile.
        self.profile = build_measured_bilayer_profile(
            thickness_angstrom=self.bilayer_thickness,
            extra_sigma_angstrom=self.bilayer_layer_sigma_angstrom,
            parameterization=self.parameterization,
            device=self.device,
        )
        # Physical (0, 0, 0) -- the organelle's own centre, and the point a
        # caller's `position_xyz` places -- on voxel n // 2 of each axis,
        # which is where `clip_insert_bounds` puts a local grid's insertion
        # point.
        self._origin_xyz = _centred_origin_xyz(self.target_shape, self.voxel_size)
        grid = self._coarse_grid()

        surfaces: list[torch.Tensor] | None = None
        for attempt in range(_BUILD_ATTEMPTS):
            rng = np.random.default_rng(derive_seed(self._base_seed, 1 + attempt))
            try:
                surfaces = self._plan.build(grid, rng)
                break
            except ReachError:
                if attempt == _BUILD_ATTEMPTS - 1:
                    raise
        assert surfaces is not None

        outer = surfaces[0]
        self.field = MembraneField(
            phi=outer.permute(2, 1, 0).contiguous(),
            spacing_angstrom=grid.h,
            origin_xyz=grid.origin.detach().cpu().to(torch.float32),
        )
        solid = (outer < 0).nonzero().to(torch.float32) * grid.h + grid.origin
        lo = self._origin_xyz.to(solid)
        nz, ny, nx = self.target_shape
        hi = lo + (torch.tensor([nx, ny, nz]).to(solid) - 1) * self.voxel_size
        self.clipped_at_boundary = bool(
            solid.numel() > 0 and ((solid < lo) | (solid > hi)).any()
        )
        self.field.clipped_at_boundary = self.clipped_at_boundary

        self.volume = render_bilayer(
            surfaces,
            grid,
            self.profile,
            self.target_shape,
            self.voxel_size,
            origin_xyz=self._origin_xyz,
        )
        return self.volume

    def place_transmembrane(
        self, min_spacing_angstrom: float = 40.0, max_attempts: int | None = None
    ) -> list[TransmembranePlacement]:
        """
        Sample surface sites and insert transmembrane protein templates.

        Must be called after :meth:`generate`. Species are chosen per site
        by weighted random draw (weights = ``frequency``), the same
        weighted-selection idea `specimen.tomogram.
        TomogramSpecimenGenerator`'s own ratio-mode filler species use
        (there via `draw_species_pool`).

        Parameters
        ----------
        min_spacing_angstrom : float, optional
            Minimum center-to-center spacing between placed sites,
            Å. Default 40.0.
        max_attempts : int, optional
            Passed to
            :func:`~specter.specimen.membrane._placement.sample_surface_sites`.
            Default is that function's own default.

        Returns
        -------
        list of TransmembranePlacement
        """
        if self.field is None or self.volume is None or self._origin_xyz is None:
            raise RuntimeError("call generate() before place_transmembrane()")
        if not self.transmembrane_specs:
            return []

        n_sites = sum(spec.frequency for spec in self.transmembrane_specs)
        sites_xyz, normals_xyz = sample_surface_sites(
            self.field,
            n_sites=n_sites,
            min_spacing_angstrom=min_spacing_angstrom,
            max_attempts=max_attempts,
            seed=self.seed,
        )
        n_found = sites_xyz.shape[0]
        if n_found < n_sites:
            severity = "zero" if n_found == 0 else "only"
            warnings.warn(
                f"MembraneGenerator.place_transmembrane: {severity} {n_found}/{n_sites} "
                "requested transmembrane sites were found -- sample_surface_sites' "
                "Newton-projection surface search exhausted max_attempts before "
                "reaching the target count. Common causes: the membrane's surface "
                "area is too small for this many well-spaced sites at this "
                "min_spacing_angstrom (try reducing min_spacing_angstrom or n_sites/frequency), "
                "or the membrane is mostly outside its box. This is not "
                "raised as an error since a partial/empty result is sometimes "
                "intended (e.g. deliberately testing a too-small membrane), but "
                f"placements will be missing/absent if not: {n_found} of "
                f"{n_sites} transmembrane instances will be placed.",
                stacklevel=2,
            )

        # Keyed by index, not pdb_source, while building (a duplicate
        # pdb_source across specs is legal -- the original serial dict
        # comprehension just let the later one silently win); remapped to
        # pdb_source -> template below to preserve that exact "last spec
        # wins" behaviour for lookups in the placement loop further down.
        index_templates = build_templates_concurrently(
            keys=list(range(len(self.transmembrane_specs))),
            build_one=lambda i, device: self._build_template(
                self.transmembrane_specs[i], device
            ),
            devices=self.render_devices,
            max_workers=self.render_workers,
        )
        templates = {
            spec.pdb_source: index_templates[i]
            for i, spec in enumerate(self.transmembrane_specs)
        }
        weights = torch.tensor(
            [spec.frequency for spec in self.transmembrane_specs], dtype=torch.float32
        )
        chooser = torch.Generator(device="cpu")
        if self.seed is not None:
            chooser.manual_seed(self.seed)

        # int16, not int32: these ids are LOCAL to this membrane (1..n_sites,
        # a few hundred at most), remapped onto the shared counter by the
        # caller. Halves the footprint of an array that is the same shape as
        # `self.volume`, which for a large organelle is hundreds of MB.
        self.transmembrane_labels = torch.zeros(
            self.volume.shape, dtype=torch.int16, device=self.volume.device
        )

        # Every site's species in one draw, with replacement: the same
        # distribution as one draw per site, without a host round trip each.
        choices = (
            torch.multinomial(
                weights, n_found, replacement=True, generator=chooser
            ).tolist()
            if n_found
            else []
        )
        rotations = [
            orientation_for_normal(
                normals_xyz[i], seed=None if self.seed is None else self.seed + i
            )
            for i in range(n_found)
        ]
        sites_cpu = sites_xyz.detach().cpu()

        # Templates are rotated a chunk of consecutive sites at a time, one
        # batched `rotate_volume` per species in the chunk, and inserted in
        # site order: `_insert_blend` replaces density, so where two
        # templates overlap the order decides which one survives. The chunk
        # bounds the rotated copies held at once to ~2^27 voxels.
        largest = max(
            (templates[spec.pdb_source].numel() for spec in self.transmembrane_specs),
            default=1,
        )
        chunk = max(1, _TRANSMEMBRANE_ROTATION_BATCH_VOXELS // largest)

        # A site is a continuous point on the surface. Each template is
        # inserted at the voxel nearest its site and the sub-voxel remainder
        # is folded into the rotation that resamples it anyway
        # (`fold_subvoxel_offsets`), so the protein's centre lands on the
        # site its placement records rather than up to half a voxel from it.
        # `self._origin_xyz` puts physical (0, 0, 0) on voxel n // 2, so a
        # site's coordinates are already its offset from that voxel.
        centre_zyx = torch.tensor([s // 2 for s in self.volume.shape], dtype=torch.long)
        placements: list[TransmembranePlacement] = []
        for c0 in range(0, n_found, chunk):
            site_ids = range(c0, min(c0 + chunk, n_found))
            rotated: dict[int, torch.Tensor] = {}
            insert_at: dict[int, torch.Tensor] = {}
            for species in sorted({choices[i] for i in site_ids}):
                group = [i for i in site_ids if choices[i] == species]
                template = templates[self.transmembrane_specs[species].pdb_source]
                # `rotate_volume` turns density by R^T for
                # `build_affine_matrix(R)`; each R here maps the template's
                # +z onto the membrane normal, so it is passed transposed to
                # turn the density by R itself.
                theta = build_affine_matrix(
                    torch.stack([rotations[i] for i in group])
                    .transpose(-1, -2)
                    .to(self.device)
                )
                theta, whole = fold_subvoxel_offsets(
                    theta, sites_cpu[group], self.voxel_size, template.shape
                )
                volumes = rotate_volume(template, theta, padding_mode="zeros")
                rotated.update(zip(group, volumes))
                whole_zyx = (whole / self.voxel_size).round().long().flip(-1)
                insert_at.update(zip(group, centre_zyx + whole_zyx))

            for i in site_ids:
                spec = self.transmembrane_specs[choices[i]]
                self._insert_blend(rotated.pop(i), insert_at.pop(i), instance_id=i + 1)
                placements.append(
                    TransmembranePlacement(
                        species_id=spec.pdb_source,
                        center_xyz=sites_cpu[i],
                        rotation_matrix=rotations[i].detach().cpu(),
                    )
                )

        self.placements = placements
        return placements

    def _build_template(
        self, spec: TransmembraneSpec, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Build (or reuse) one transmembrane species' potential template.

        Parameters
        ----------
        spec : TransmembraneSpec
            Species to render, or reuse via `spec.template` if supplied.
        device : torch.device, optional
            Device to build the `PotentialBuilder` (and run its `forward`)
            on -- lets concurrent callers (`build_templates_concurrently`)
            spread species across `self.render_devices` instead of
            serializing everything onto `self.device`. Default None: use
            `self.device`, matching pre-parallel behaviour exactly. Either
            way, the returned template always ends up on `self.device`
            (this method's only externally-visible contract), since that's
            what every downstream consumer -- `_insert_blend` chief among
            them -- assumes.
        """
        if spec.template is not None:
            return spec.template.detach().to(self.device, dtype=torch.float32)

        build_device = self.device if device is None else device
        return render_transmembrane_template(
            spec,
            self.voxel_size,
            self.pdb_cache_dir,
            build_device,
            readd_hydrogens=self.readd_hydrogens,
            monomer_library_path=self.monomer_library_path,
            use_deposited_bfactors=self.use_deposited_bfactors,
        ).to(self.device)

    def _physical_to_voxel_index(self, site_xyz: torch.Tensor) -> torch.Tensor:
        assert self._origin_xyz is not None
        origin = self._origin_xyz.cpu()
        idx_xyz = (site_xyz.detach().cpu() - origin) / self.voxel_size
        idx_zyx = torch.stack([idx_xyz[2], idx_xyz[1], idx_xyz[0]])
        return idx_zyx.round().long()

    def _insert_blend(
        self,
        local_density: torch.Tensor,
        center_zyx: torch.Tensor,
        instance_id: int | None = None,
    ) -> None:
        """
        Insert a transmembrane protein template, replacing (not adding to)
        the membrane's own density wherever the template is occupied -- see
        `transmembrane_occupancy_fraction`'s own docstring for why plain
        addition double-counts density and how this threshold is chosen.

        Parameters
        ----------
        local_density : torch.Tensor
            The rotated protein template.
        center_zyx : torch.Tensor
            Where to insert it, in voxel indices.
        instance_id : int, optional
            When given (and `transmembrane_labels` has been allocated),
            stamp this id into `transmembrane_labels` wherever the protein
            displaced lipid. Default None: density only, no labelling.
        """
        assert self.volume is not None
        assert self.profile is not None
        bounds = clip_insert_bounds(
            center_zyx.tolist(), local_density.shape, self.volume.shape
        )
        if bounds is None:
            return
        dst, src = bounds
        protein = local_density[src].to(self.volume.device)

        peak = float(self.profile.psi.max())
        threshold = self.transmembrane_occupancy_fraction * peak

        # Ground truth for "this voxel is protein, not lipid" is the same
        # quantity the density blend below already uses: the smoothstep
        # weight reaches 0.5 exactly where `protein == threshold`, so that
        # is the displacement boundary, not a second independently-tuned
        # cutoff. `transmembrane_occupancy_fraction = 0` means the protein
        # coexists with lipid rather than displacing it -- there is then no
        # displacement region, but the protein still OCCUPIES voxels and
        # still needs a label, so fall back to the default fraction for
        # labelling alone rather than silently labelling nothing.
        if instance_id is not None and self.transmembrane_labels is not None:
            label_threshold = (
                threshold
                if threshold > 0
                else _DEFAULT_TRANSMEMBRANE_OCCUPANCY_FRACTION * peak
            )
            claimed = self.transmembrane_labels[dst]
            self.transmembrane_labels[dst] = torch.where(
                (protein > label_threshold) & (claimed == 0),
                torch.full_like(claimed, instance_id),
                claimed,
            )

        if threshold <= 0:
            self.volume[dst] += protein
            return

        # Smoothstep over a band centered on `threshold`, not an
        # unbounded-tail sigmoid: a sigmoid never truly reaches 0, so a
        # template's near-zero background (e.g. rotate_volume's own
        # zero-padding, or a real potential's slowly-decaying tail) still
        # picks up a small but nonzero weight EVERYWHERE the template was
        # inserted, incorrectly attenuating membrane density across the
        # whole (padded) bounding box rather than just near the protein's
        # actual mass (caught directly: a synthetic single-hot-voxel
        # template made this visible as a net *decrease* in total density
        # after insertion). Smoothstep instead saturates to EXACTLY 0/1
        # outside its band, so density far from the protein is left
        # untouched.
        low, high = 0.5 * threshold, 1.5 * threshold
        t = torch.clamp((protein - low) / (high - low), 0.0, 1.0)
        weight = t * t * (3.0 - 2.0 * t)
        self.volume[dst] = (1.0 - weight) * self.volume[dst] + weight * protein


__all__ = [
    "MembraneGenerator",
    "TransmembraneSpec",
    "TransmembranePlacement",
]
