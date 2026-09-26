"""
TomogramSpecimenGenerator: the specimen generator behind `specter build
tomogram`. Assembles a tomogram from any combination of organic membranes
(with transmembrane proteins on the bilayer), scattered filaments (F-actin,
microtubules), a carbon support film, gold fiducial beads, and densely
packed cytosolic and vesicle-lumen protein populations. Any combination is
valid as long as at least one is non-empty; with no membranes the whole
volume is one cytosol region (see `._regions`).

Generation order is carbon film, membranes, filaments, beads, then protein
fill; each stage avoids what the earlier ones placed. Protein species are
region-gated against the composited membranes, so a "lumen" species can
only land inside an enclosed compartment and a "cytosol" species only
outside one, and are placed either at an exact count
(`TomogramProteinSpec.n_copies`, ground-truth "target" semantics, placed
first within each region) or ratio-weighted up to `occupancy_fraction` of
the region (`TomogramProteinSpec.ratio`, "filler" semantics).

Protein placement is `specimen.packing.pack_shapes_3d`: each instance is
tested at its rotated voxel footprint (van der Waals shell plus `gap`)
against a running occupancy grid that already holds the region's
complement and everything placed before it, so proteins collide with each
other, filaments, beads, the membrane shell and the carbon film at voxel
resolution. Membrane instances and gold beads are placed by Random
Sequential Addition of bounding spheres against an exclusion distance
field (`pack_hard_spheres_3d`), which is an approximation: an instance's
true rendered shape can still graze what it avoided close to the boundary.
Anything that does not fit is dropped rather than retried ("reject and
move on"). Two consequences of the approximate stages:

- The carbon film is painted into the canvas first and everything after
  it avoids it, but membrane placement only avoids it as a bounding
  sphere, so whatever part of an instance's rendered density would land on
  carbon is zeroed as it is merged into the canvas (the `to_composite`
  loop), keeping the volume and the instance's shell label consistent.
  Filament placement has no obstacle-avoiding random walk, so a monomer
  landing inside the film is dropped after the fact (`_stamp_filaments`).
- Filaments are not region-gated (they have no fixed geometry to gate
  against) and do not avoid the membrane shell or each other; they are
  rendered before protein packing purely so the packer can avoid them.

Membranes: each :class:`MembraneInstance` has its own `MembraneGenerator`
(potentially a different `shape_backend`), renders in its own centered
local frame on a working grid auto-sized to the organelle, and is
max-merged into the shared canvas at a `position_xyz` resolved by
`pack_hard_spheres_3d` (treating the instance as a bounding sphere against
other instances, the box walls and the carbon film). Shell/lumen/cytosol
regions are classified once on the composite; `classify_membrane_regions`'s
connected-components approach handles several disjoint compartments
without special-casing. An instance whose `clipped_at_boundary` is set
after generation (its local grid was too small for what was drawn) is
dropped with a warning; the bounding-sphere check is necessarily
approximate for `swept_spline`'s wandering shape.

Labels: `membrane_labels` records which membrane instance a shell voxel
belongs to; `instance_labels` records which protein object a voxel belongs
to, with one id counter shared in generation order: transmembrane
proteins, filaments (one id per filament) and microtubules (one id per
tube), gold beads, then cytosol/lumen proteins.

Code layout: `TomogramSpecimenGenerator` keeps the orchestration, carbon
film, beads, and protein packing/rendering; the membrane stage, the
filament/microtubule stage and pick export are mixins it inherits from
`._membranes`, `._filaments` and `._picks`.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

import numpy as np
import torch

from ...config import ScalarOrRange
from ...arrays import count_nonzero_chunked
from ...crowding import insert_particles_into_micrograph
from ...pdb import DEFAULT_PDB_CACHE_DIR, PDB, canonical_pdb_source
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, random_rotation_matrix
from .._carbon import CarbonFilmGenerator, CarbonFilmSpec, edge_hole_center
from .._grid import BeadGenerator
from .._parallel_render import (
    build_pdb_cache_concurrently,
    build_templates_concurrently,
    resolve_render_devices,
    resolve_render_workers,
)
from ..filament import (
    FilamentInstance,
    FilamentSpec,
    MicrotubuleInstance,
    MicrotubuleSpec,
)
from ..membrane import TransmembranePlacement
from ..packing import (
    build_species_mask,
    draw_species_pool,
    estimate_protein_box_size,
    pack_hard_spheres_3d,
    pack_shapes_3d,
)
from ._helpers import (
    _MAX_PACKING_GRID_VOXELS,
    _allowed_region_exclusion_field,
    _coarsen_grid_by,
    _coarsen_grid_to_budget,
    _downsample_mask_maxpool,
    _insert_instance_labels,
    _insert_rotated_copies,
    _wants_atom_species,
    resolve_accumulator_device,
)
from ._specs import (
    BeadPlacement,
    MembraneInstance,
    TomogramBeadSpec,
    TomogramPlacement,
    TomogramProteinSpec,
)
from ._filaments import _FilamentStageMixin
from ._membranes import _MembraneStageMixin
from ._picks import _PickExportMixin
from ...progress import TqdmProgress, phase, phase_done, phase_start, status
from specter.options import ScatteringFactors


class TomogramSpecimenGenerator(
    _MembraneStageMixin, _FilamentStageMixin, _PickExportMixin
):
    """
    Assemble a specimen tomogram from any combination of pre-configured
    membranes, filaments, carbon film, gold fiducials, and densely packed
    cytosolic/vesicle-lumen protein populations -- every one of those is
    optional (see the module docstring), so this is the single generator
    behind `specter build tomogram` for membrane and membrane-free
    specimens alike.

    Places any number of distinct species (see `protein_specs` below),
    purely for DENSITY -- each region (cytosol/lumen) is packed as densely
    as `occupancy_fraction` allows by exact-footprint shape packing
    (`..packing.pack_shapes_3d`), uniformly throughout it, with no distributional shaping of any one species' own
    spatial statistics. Contrast `specter.specimen.
    MicrographSpecimenGenerator` (the single-particle-micrograph backend):
    single-species only (many duplicate copies of ONE template), but its
    placement (`~specter.crowding.CrowdWithDuplicates`) supports an
    optional water-air-interface adsorption bias that this generator has
    no equivalent of -- crowding realism here instead comes from
    region-gating against real membrane geometry, not from shaping any
    species' own Z-distribution.

    Parameters
    ----------
    membrane_instances : list of MembraneInstance
        Membranes to composite into the shared tomogram (see
        `MembraneInstance`) -- each instance's own shape/transmembrane_specs
        are used as-is and not duplicated here. May be EMPTY (no membranes
        at all -- `generate()` then treats the whole tomogram as one
        cytosol region, no lumen; see module docstring). Every instance's
        own `generator.voxel_size` must match `voxel_size` below (raises
        `ValueError` naming the offending index otherwise).
    target_shape : tuple of int
        Shared tomogram canvas shape, `(Z, Y, X)` voxels -- every instance
        composites into this same grid.
    voxel_size : float
        Shared voxel size, Å -- must match every instance's own
        `generator.voxel_size`.
    protein_specs : list of TomogramProteinSpec
        Cytosolic/lumen species to pack, exact-count (`n_copies`) and/or
        ratio-weighted (`ratio`). May be EMPTY (membranes/filaments with no
        packed protein population).
    microtubule_specs : list of MicrotubuleSpec, optional
        Microtubule species to scatter through the tomogram via
        `specimen.filament.place_microtubules` -- whole 13-protofilament
        tubes (lumen, A-lattice seam and all), each rendered as many rigid
        copies of one alpha-beta tubulin dimer. Placement shares filaments'
        limitations below (no region-gating, no collision avoidance), and
        the tubes are likewise avoided by targets/filler afterwards. Unlike
        filaments, every dimer of one tube shares a single instance-label
        id, so a microtubule is one object in the segmentation.
    filament_specs : list of FilamentSpec, optional
        Filament species (e.g. `specimen.filament.ACTIN_SPEC`/
        `PROTOFILAMENT_SPEC`) to scatter through the tomogram via
        `specimen.filament.place_filaments` -- specter-native random-walk
        placement, with no region-gating and no collision avoidance
        against the membrane shell or each other, but DOES get avoided by
        the `protein_specs` packing stage that follows it (see module
        docstring). Default None (no filaments).
    carbon_film_spec : CarbonFilmSpec, optional
        Carbon support film to paint into the shared canvas before
        anything else is placed (see module docstring). Default None (no
        film -- pure ice, the original behaviour).
    bead_specs : list of TomogramBeadSpec, optional
        Gold fiducial bead populations to scatter (see module docstring
        and `TomogramBeadSpec`). Default None (no beads).
    bead_roughness : float or [low, high], optional
        How irregular each fiducial's boundary is, as an RMS fraction of
        its radius -- see `.._grid.BeadGenerator`. A ``[low, high]`` pair
        draws per bead, mixing near-round and misshapen particles. Default
        0.12.
    occupancy_fraction : float, optional
        Target packing density, as a fraction of real footprint volume (see
        `draw_species_pool`), applied independently per region -- e.g. 0.2
        for `"lumen"` species targets 20% of the LUMEN's own volume, not
        20% of the whole tomogram. Default 0.2.
    n_orientations : int, optional
        Size of the per-species rotation cache used by protein packing.
        Rotating per attempt instead of indexing a cache is what dominates
        a naive implementation's cost (profiled at 6.8 A: 3.27 ms to rotate
        vs 0.08 ms to collision-test). Default 256.
    packing_max_retries : int, optional
        Trial positions per instance. The dominant density/speed knob,
        paired with that function's `stall_patience` -- see
        `..packing.pack_shapes_3d`'s own table. Default 1500.
    packing_voxel_size : float, optional
        Run protein collision on a COARSER grid than
        the render, an integer multiple of `voxel_size`.

        Default None = automatic: pack at `voxel_size` until the grid would
        exceed `_MAX_PACKING_GRID_VOXELS`, then coarsen by the smallest
        integer factor that fits, mirroring what
        `_resolve_exclusion_field_grid` already does for the sphere
        backend. The budget sits high enough that ordinary boxes (a
        200x1200x1200 grid at 5 A) never trigger it; it exists so a fine
        `voxel_size` degrades to coarser collision instead of failing
        outright. Pass a value to control it explicitly.

        The collision grid, not the render, is what makes a fine
        `voxel_size` expensive: a 1 A occupancy grid for a
        1000x6000x6000 A box is 36 GB and the rotation cache is 20 GB for a
        243 A species, against 0.29 GB and 0.18 GB at 5 A. Rendering stays
        at `voxel_size` either way, since `..packing.pack_shapes_3d` returns
        positions in Angstrom rather than grid indices.

        Footprints are built at `voxel_size`, and each ROTATED orientation
        is max-pooled onto the coarse grid inside the packer, never
        rasterized at the coarse size directly. Both halves of that matter.
        A directly rasterized coarse mask omits the van der Waals shell
        entirely (a 1.9 A pad rounds to zero dilation at 2 A), so instances
        pack ~2 A closer than van der Waals contact allows and the extra
        density is an artifact. Pooling before rotating rather than after
        loses containment, since rotation interpolates, leaving the
        collision guarantee approximate at render resolution -- see
        `..packing._shape._rotation_cache`. Measured at 2 A against a
        native 1 A pack: volume fraction 0.264 vs 0.269, zero overlapping
        voxels, 8.8x faster; the same comparison with direct coarse masks
        reads a denser 0.348, with instances interpenetrating.
    clip_axes : tuple of bool, optional
        (z, y, x), matching `target_shape`'s axis order -- passed
        straight through to every packer call here (`pack_hard_spheres_3d`
        for auto-placed membrane instances and gold beads,
        `pack_shapes_3d` for cytosol/lumen protein packing). True
        on an axis lets a placed instance's center stay in-bounds while its
        body pokes past that wall (truncated at render time) instead of
        being rejected outright -- e.g. for a tomogram whose xy field of
        view is a crop of a larger cellular region. Default all False.
    region_density_threshold : float, optional
        Passed to `classify_membrane_regions`. Default None (that
        function's own default).
    region_max_passes : int, optional
        `max_passes` for the gold-bead `pack_hard_spheres_3d` call, whose
        placement is restricted to outside the membrane shell and the
        already-placed filaments by a `sampling_mask`. Default 300, higher
        than `pack_hard_spheres_3d`'s own default 200, since a crowded
        volume can need more attempts before a valid spot turns up.
        Cytosol/lumen protein packing does not read it; its attempt budget
        is `packing_max_retries`.
    min_transmembrane_spacing : float, optional
        Passed to `MembraneGenerator.place_transmembrane`. Default 40.0.
    pdb_cache_dir : str, optional
        Directory for downloaded PDB/mmCIF files. Default is
        `$SPECTER_PDB_CACHE`, else `$XDG_CACHE_HOME/specter/pdb`, else
        `~/.cache/specter/pdb` (see `config.default_pdb_cache_dir`).
    parameterization : str, optional
        Atomic scattering-factor parameterization for `PotentialBuilder`.
        Default "shtyrov", matching `PotentialBuilder`'s own default.
    monomer_library_path : str, optional
        Forwarded to every `PDB` built here; see `specter.pdb.PDB`. Unset
        falls back to `$CLIBD_MON`.
    readd_hydrogens : {"auto", True, False}, optional
        Forwarded to every `PDB` built here; see `specter.pdb.PDB`. Only
        takes effect for `parameterization="shtyrov"`, which is the only one
        that types atoms, and only when a Monomer Library is available via
        `$CLIBD_MON`. Default "auto": keep the hydrogens a structure carries,
        add them only when it has none.
    use_deposited_bfactors : bool, optional
        Damp each atom by the B-factor its structure deposits, rather than
        rendering the model statically. Requires
        `parameterization="shtyrov"`; see `PotentialBuilder`'s own
        `b_factors` for why the other backends refuse it, and why a
        deposited column is not a measured displacement. Default False.
    seed : int, optional
        Random seed.
    device : str or torch.device, optional
        Device for cytosol/lumen protein packing (`pack_shapes_3d`) AND,
        by default, for `_render_species_pool`'s own `PotentialBuilder`
        step -- override the latter alone via `render_devices` below. The
        bounding-sphere placement of membrane instances and beads always
        runs on the CPU (see `pack_hard_spheres_3d`'s own docstring).
        Default "cpu".
    chunk_size : int, optional
        Instances rotated per batch, per species. Default None (all of a
        species' instances at once).
    render_workers : int or "auto", optional
        Number of cytosol/lumen `protein_specs` species rendered/fetched
        concurrently (`_render_species_pool`'s own `PotentialBuilder`
        templates on threads, and `generate()`'s PDB preload on processes
        above `_MIN_SOURCES_FOR_PROCESS_POOL`). Default 1: fully serial,
        identical to the pre-parallel behaviour. `"auto"` resolves via
        `recommend_render_workers(len(protein_specs))` -- min(n_species, 8),
        the measured sweet spot from a full production-scale sweep (see
        that function's own docstring).
    render_devices : list of str or torch.device, optional
        Device pool to round-robin those concurrent species across (e.g.
        multiple GPUs). Default None: every species renders on `device`
        above, still concurrently across `render_workers` threads, just not
        spread across multiple physical devices.
    progressbars : bool, optional
        Show progress bars/status spinners during `generate()` (membrane
        instance generation, filament placement, PDB fetch, packing, and
        per-species rendering) via `specter.progress`'s `TqdmProgress`/
        `status`. Default True. Set False for quiet/scripted runs.
    accumulator_device : str or torch.device or "auto", optional
        Device for the shared canvas tensors (`volume`/`instance_labels`/
        `membrane_labels`), decoupled from `device` (which stays the
        compute device for rendering/rotation regardless). Default None:
        same as `device`, identical to the original one-device behaviour.
        "auto" resolves via `recommend_accumulator_device` -- estimates
        the canvas' own memory footprint from `target_shape` and
        falls back to "cpu" if it would exceed half of `device`'s
        CURRENTLY FREE memory (conservative on purpose: rendering/
        rotation on `device` need real memory too, at the same time).
        Explicit "cpu" always works regardless of that estimate. Set this
        for a large field of view whose canvas exceeds GPU VRAM but fits
        in system RAM -- see this generator's own module-level discussion
        for the numbers this matters at.

    Attributes
    ----------
    regions : dict of str to torch.Tensor
        ``{"shell", "lumen", "cytosol"}`` boolean masks, set after
        `generate()` runs (see `classify_membrane_regions`) -- computed on
        the COMPOSITED (all instances merged) volume.
    membrane_labels : torch.Tensor
        Per-instance integer label volume for the membrane SHELL itself --
        `membrane_labels == i+1` is instance `i`'s own shell (first-write-
        wins where instances overlap, see `_insert_shell_label`), shape
        `target_shape`, dtype int32. Set after `generate()` runs.

        This is the BILAYER only. Voxels where an embedded transmembrane
        protein displaced the lipid are excluded here and carry that
        protein's own id in `instance_labels` instead, so the two volumes
        partition the membrane rather than double-claiming it. The
        displacement boundary is not a second cutoff invented for
        labelling: it is the same one
        `MembraneGenerator.transmembrane_occupancy_fraction` already uses
        to decide where protein density REPLACES lipid density.
    placed_membrane_instances : list of MembraneInstance
        The subset of `membrane_instances` actually composited into the
        volume, set after `generate()` runs. `membrane_instances` is the
        request and is never pruned; an instance is absent here if
        auto-placement could not fit it without colliding, or if its own
        working grid clipped the shape it drew. Report placed-of-requested
        from the two lists together.
    transmembrane_placements : list of TransmembranePlacement
        From every instance's own `MembraneGenerator.place_transmembrane`,
        with `center_xyz` offset into shared-tomogram coordinates by that
        instance's `position_xyz`. Set after `generate()` runs.
    placements : list of TomogramPlacement
        Every placed cytosolic/lumen instance, set after `generate()` runs.
    instance_labels : torch.Tensor
        Per-instance integer label volume for every PROTEIN instance:
        transmembrane proteins first, then filament monomers (when
        `filament_specs` is non-empty), gold beads, and the cytosol/lumen
        fill -- all on one continuous instance-id counter, not separate
        label spaces (see module docstring). Shape `target_shape`, dtype
        int32. Set after `generate()` runs.

        Transmembrane proteins are ordinary instances here despite being
        embedded in a membrane: a consumer training per-instance protein
        segmentation would otherwise find them plainly visible in the
        density and absent from its target.
    filament_instances : list of FilamentInstance
        Every placed filament monomer, from `specimen.filament.
        place_filaments` (`position_xyz` in the same corner-relative,
        `[0, extent)` convention `export_picks` writes directly -- NOT the
        center-relative convention `instance_labels`/`volume` use
        internally, see `_stamp_filaments`). Set after `generate()` runs
        (empty if `filament_specs` was empty/None).
    microtubule_instances : list of MicrotubuleInstance
        Every placed microtubule (axis polyline + lattice), set after
        `generate()` runs (empty if `microtubule_specs` was empty/None).
    microtubule_dimer_instances : list of FilamentInstance
        The individual tubulin dimer copies those microtubules were
        rendered from -- kept separate from `filament_instances` so the two
        species types stay distinguishable in ground truth.
    bead_instances : list of BeadPlacement
        Every placed gold fiducial bead, set after `generate()` runs
        (empty if `bead_specs` was empty/None).
    """

    def __init__(
        self,
        membrane_instances: list[MembraneInstance],
        target_shape: tuple[int, int, int],
        voxel_size: float,
        protein_specs: list[TomogramProteinSpec],
        filament_specs: list[FilamentSpec] | None = None,
        microtubule_specs: list[MicrotubuleSpec] | None = None,
        carbon_film_spec: CarbonFilmSpec | None = None,
        bead_specs: list[TomogramBeadSpec] | None = None,
        bead_roughness: ScalarOrRange = 0.12,
        occupancy_fraction: float = 0.2,
        n_orientations: int = 256,
        packing_max_retries: int = 1500,
        packing_voxel_size: float | None = None,
        clip_axes: tuple[bool, bool, bool] = (False, False, False),
        region_density_threshold: float | None = None,
        region_max_passes: int = 300,
        min_transmembrane_spacing: float = 40.0,
        pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
        parameterization: ScatteringFactors = "shtyrov",
        bulk_parameterization: ScatteringFactors = "kirkland",
        readd_hydrogens: bool | str = "auto",
        monomer_library_path: str | None = None,
        use_deposited_bfactors: bool = False,
        seed: int | None = None,
        device: str | torch.device = "cpu",
        chunk_size: int | None = None,
        render_workers: int | Literal["auto"] = 1,
        render_devices: list[str | torch.device] | None = None,
        progressbars: bool = True,
        accumulator_device: str | torch.device | Literal["auto"] | None = None,
    ):
        if (
            not protein_specs
            and not membrane_instances
            and not filament_specs
            and not microtubule_specs
            and not bead_specs
            and carbon_film_spec is None
        ):
            raise ValueError(
                "TomogramSpecimenGenerator: at least one of "
                "membrane_instances, protein_specs, filament_specs, "
                "microtubule_specs, bead_specs, or carbon_film_spec must be "
                "non-empty/set -- an empty "
                "tomogram has nothing to generate."
            )
        for i, mi in enumerate(membrane_instances):
            if mi.generator.voxel_size != voxel_size:
                raise ValueError(
                    f"TomogramSpecimenGenerator: membrane_instances[{i}]'s own "
                    f"voxel_size ({mi.generator.voxel_size}) does not match the shared "
                    f"voxel_size ({voxel_size}) -- every instance must render on the "
                    "same voxel grid to be compositable."
                )
        self.membrane_instances = membrane_instances
        self.target_shape = target_shape
        self.voxel_size = voxel_size
        self.protein_specs = protein_specs
        self.filament_specs = filament_specs or []
        self.microtubule_specs = microtubule_specs or []
        self.carbon_film_spec = carbon_film_spec
        self.bead_specs = bead_specs or []
        self.bead_roughness = bead_roughness
        self.occupancy_fraction = occupancy_fraction
        # Not a constructor argument: there is no physical basis for a
        # minimum clearance between macromolecules in crowded cytoplasm --
        # they contact each other, and CryoTomoSim uses none either. Under
        # protein packing a nonzero value is also quantized to whole
        # voxels, so a nominal 5 A became a full voxel shell in every
        # direction and cost ~30% of the achievable density (volume fraction
        # 0.197 -> 0.138 on a 121-species filler set at 6.8 A). Gold beads
        # and membrane instances share this value and are content with 0:
        # their bounding spheres merely become tangent.
        self.gap = 0.0
        self.n_orientations = n_orientations
        self.packing_max_retries = packing_max_retries
        self.packing_voxel_size = packing_voxel_size
        self.clip_axes = clip_axes
        self.region_density_threshold = region_density_threshold
        self.region_max_passes = region_max_passes
        self.min_transmembrane_spacing = min_transmembrane_spacing
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
        self.bulk_parameterization = bulk_parameterization
        self.readd_hydrogens = readd_hydrogens
        self.monomer_library_path = monomer_library_path
        self.use_deposited_bfactors = use_deposited_bfactors
        self.seed = seed
        self.device = device
        self.chunk_size = chunk_size
        self.render_workers = resolve_render_workers(render_workers, len(protein_specs))
        self.render_devices = resolve_render_devices(device, render_devices)
        self.progressbars = progressbars
        # Where the shared, potentially very large canvas tensors (volume/
        # instance_labels/membrane_labels) live -- default None resolves
        # to `device` (identical to the pre-existing behaviour: everything
        # on one device). Set to "cpu" to decouple them from `device`: all
        # per-particle/per-instance COMPUTE (PotentialBuilder rendering,
        # rotate_volume, MembraneGenerator field generation) still runs on
        # `device` (GPU, for speed), but each small
        # rotated/rasterized result is moved to `accumulator_device` right
        # before being stamped into the big canvas -- letting the canvas
        # itself be sized by system RAM instead of GPU VRAM (e.g. a
        # (1333, 4000, 4000)-voxel volume at 1.5 A/voxel is ~85 GB, past
        # any single GPU's VRAM but plausible in system RAM on a
        # workstation/cluster node). The per-instance insertion helpers
        # (`_insert_volume_max`/`_insert_shell_label`/
        # `_insert_instance_labels`) already move the SMALL side to the
        # accumulator's device internally, so this is safe regardless of
        # where `device` itself points.
        self.accumulator_device = resolve_accumulator_device(
            device, accumulator_device, target_shape
        )

        self.regions: dict[str, torch.Tensor] | None = None
        self.membrane_labels: torch.Tensor | None = None
        # The subset of `membrane_instances` that actually made it into the
        # volume. `membrane_instances` is the REQUEST and is never pruned;
        # an instance is dropped from this list if auto-placement couldn't
        # fit it without colliding, or if its own working grid clipped the
        # shape it drew. Report placed-of-requested from the two together
        # rather than reading `membrane_instances` alone, which would
        # claim every requested instance is present.
        self.placed_membrane_instances: list[MembraneInstance] = []
        self.transmembrane_placements: list[TransmembranePlacement] = []
        self.placements: list[TomogramPlacement] = []
        self.instance_labels: torch.Tensor | None = None
        # Rasterized footprints for protein packing, keyed by the structure's
        # file and the flags that shape its parse, plus (voxel_size, gap) -- a
        # species reappearing across regions, or across generate() calls,
        # rasterizes once. Not keyed on id(pdb): CPython reuses an id once
        # its object is freed, so a later PDB could hit a stale mask.
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self.filament_instances: list[FilamentInstance] = []
        self.microtubule_instances: list[MicrotubuleInstance] = []
        self.microtubule_dimer_instances: list[FilamentInstance] = []
        self.bead_instances: list[BeadPlacement] = []

    def generate(self) -> torch.Tensor:
        """
        Run the full pipeline and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Shape `target_shape`, dtype float32.
        """
        if self.seed is not None:
            torch.manual_seed(
                self.seed
            )  # random_rotation_matrix has no generator= param

        # Per-run outputs: a second generate() must not append to the first's.
        self.bead_instances = []

        voxel_size = self.voxel_size
        target_shape = self.target_shape
        box = (
            target_shape[0] * voxel_size,
            target_shape[1] * voxel_size,
            target_shape[2] * voxel_size,
        )

        volume, carbon_mask = self._stage_carbon(target_shape, voxel_size)

        # Transmembrane proteins are stamped into this during the membrane
        # loop below, so it is allocated before that loop rather than after
        # it: they are protein instances like any other, and belong in the
        # same id space as filaments, beads and the cytosol/lumen fill (see
        # module docstring). They therefore take the FIRST ids, and
        # everything placed later avoids them through `obstacle_mask`.
        instance_labels = torch.zeros(
            target_shape, dtype=torch.int32, device=self.accumulator_device
        )
        next_instance_id = 1

        volume, instance_labels, next_instance_id = self._stage_membranes(
            volume, instance_labels, next_instance_id, carbon_mask, box, voxel_size
        )
        volume, instance_labels, next_instance_id, obstacle_mask = (
            self._stage_filaments(
                volume, instance_labels, next_instance_id, voxel_size, carbon_mask
            )
        )
        volume, instance_labels, next_instance_id, obstacle_mask = self._stage_beads(
            volume, instance_labels, next_instance_id, voxel_size, obstacle_mask
        )

        self.placements = []
        pdb_cache = self._load_structures()
        for location in ("cytosol", "lumen"):
            specs_here = [s for s in self.protein_specs if s.location == location]
            if not specs_here:
                continue
            volume, instance_labels, next_instance_id = self._stage_species(
                location,
                specs_here,
                volume,
                instance_labels,
                next_instance_id,
                obstacle_mask,
                pdb_cache,
                voxel_size,
            )

        self.instance_labels = instance_labels
        return volume

    def _stage_carbon(
        self, target_shape: tuple[int, int, int], voxel_size: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The empty canvas, with the carbon film painted in if there is one,
        and the film's footprint mask (None without a film)."""
        # Carbon film (if any) is generated first, even before membrane
        # instance positions are solved -- so membrane auto-placement here,
        # and filament placement further below (`_stamp_filaments`), can
        # both be made carbon-aware: nothing should end up placed inside
        # the carbon film itself. `carbon_mask` captures ONLY the carbon
        # footprint (nothing else has been painted into `volume` yet at
        # this point); membrane instances are then composited into the
        # same `volume` via max-merge below, and `classify_membrane_regions`
        # runs once against the full composite afterward -- carbon stays
        # part of it there too (it reads as "shell", same as membrane,
        # which is what already keeps beads/cytosol/lumen protein fill off
        # of it, see `_stamp_beads`/the cytosol/lumen loop below).
        volume = torch.zeros(
            target_shape, dtype=torch.float32, device=self.accumulator_device
        )
        carbon_mask: torch.Tensor | None = None
        if self.carbon_film_spec is not None:
            with (
                phase("Carbon film", disable=not self.progressbars),
                status("Generating carbon support film", disable=not self.progressbars),
            ):
                volume = self._stamp_carbon_film(volume, target_shape, voxel_size)
            carbon_mask = volume > 0
        return volume, carbon_mask

    def _stage_beads(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        obstacle_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
        """Stamp the gold fiducial beads and fold them into the obstacle mask."""
        # Gold fiducial beads render right after filaments, still BEFORE
        # cytosol/lumen protein packing -- avoids the membrane shell and
        # any already-placed filaments (obstacle_mask), and is itself then
        # folded into obstacle_mask so the protein-fill stage below avoids
        # already-placed beads too (see module docstring/_stamp_beads).
        if self.bead_specs:
            _bead_phase_start = phase_start(
                "Gold fiducial beads", disable=not self.progressbars
            )
            volume, instance_labels, next_instance_id = self._stamp_beads(
                volume, instance_labels, next_instance_id, voxel_size, obstacle_mask
            )
            phase_done(
                f"Gold fiducial beads ({len(self.bead_instances)} placed)",
                _bead_phase_start,
                disable=not self.progressbars,
            )
            obstacle_mask = instance_labels > 0
        else:
            self.bead_instances = []
        return volume, instance_labels, next_instance_id, obstacle_mask

    def _load_structures(self) -> dict[str, PDB]:
        """Load every cytosol/lumen structure once, concurrently."""
        pdb_cache: dict[str, PDB] = {}

        # Pre-load every unique cytosol/lumen pdb_source ONCE, up front,
        # concurrently across self.render_workers: done serially, PDB
        # fetch+parse (not rendering) is the single largest cost of a
        # 161-species production-scale run, ~45% of total wall time. Uses
        # PROCESSES, not threads (build_pdb_cache_concurrently, not
        # build_templates_concurrently): thread-pooling this step measured
        # ZERO wall-clock benefit, because Biopython's structure parser
        # doesn't release the GIL for most of its work. See
        # build_pdb_cache_concurrently's own docstring for the spawn/
        # __main__-guard caveat that comes with using processes here.
        unique_sources = sorted({s.pdb_source for s in self.protein_specs})
        if unique_sources:
            # "Loading", not "Fetching": on the common path nothing is
            # downloaded at all, and the time goes on parsing -- 9.7 s of
            # Biopython plus 7.0 s of gemmi typing for a 220k-atom assembly,
            # against ~0 s for a cache hit on the .cif. Calling it a fetch
            # sent readers looking for a network problem that wasn't there.
            _fetch_phase_start = phase_start(
                "Loading PDB structures", disable=not self.progressbars
            )
            with TqdmProgress(
                transient=True, disable=not self.progressbars
            ) as progress:
                fetch_task = progress.add_task(
                    "Loading PDB structures", total=len(unique_sources)
                )
                pdb_cache = build_pdb_cache_concurrently(
                    pdb_sources=unique_sources,
                    pdb_cache_dir=self.pdb_cache_dir,
                    max_workers=self.render_workers,
                    compute_atom_species=_wants_atom_species(self.parameterization),
                    readd_hydrogens=self.readd_hydrogens,
                    monomer_library_path=self.monomer_library_path,
                    on_result=lambda source: progress.update(
                        fetch_task, advance=1, description=f"Loaded {source}"
                    ),
                )
            phase_done(
                # Structures, not spellings: `1fa2` and `1FA2` are one
                # entry and are fetched once (see canonical_pdb_source).
                f"Loaded {len({canonical_pdb_source(s) for s in unique_sources})} "
                "PDB structure(s)",
                _fetch_phase_start,
                disable=not self.progressbars,
            )

        return pdb_cache

    def _stage_species(
        self,
        location: str,
        specs_here: list[TomogramProteinSpec],
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        obstacle_mask: torch.Tensor | None,
        pdb_cache: dict[str, PDB],
        voxel_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pack and render one region's protein species: the exact-count targets
        first, then the ratio-weighted filler."""
        _location_phase_start = phase_start(
            f"{location.capitalize()} species", disable=not self.progressbars
        )

        assert self.regions is not None  # set by _stage_membranes
        region_mask = self.regions[location]
        if obstacle_mask is not None:
            # Negate first, then AND in place: `region_mask & ~obstacle`
            # held the negation AND the result at once, and at a
            # 300x1200x1200 tomogram each bool is 0.40 GiB. Profiled as
            # the single largest allocation site in a default run, 3.22
            # GiB across four blocks, since both regions are built while
            # the originals are still referenced.
            #
            # Writing into the negation, never into self.regions -- that
            # is kept for later stages and must not be mutated here.
            region_mask = ~obstacle_mask
            region_mask &= self.regions[location]
        # Chunked: a plain .sum() on a volume-sized bool promotes every
        # element to int64 first, 3.22 GiB for a 300x1200x1200 mask, and
        # profiled as the largest single allocation in a default run.
        region_voxels = count_nonzero_chunked(region_mask)
        if region_voxels == 0:
            warnings.warn(
                f"TomogramSpecimenGenerator: no '{location}' region found "
                f"(0 voxels, after excluding already-placed filaments) -- "
                f"{len(specs_here)} species declared for it will not be "
                "placed. For 'lumen', this means the membrane has no "
                "enclosed compartment.",
                stacklevel=2,
            )
            return volume, instance_labels, next_instance_id
        region_volume_a3 = region_voxels * voxel_size**3

        pdbs_by_source: dict[str, PDB] = {}
        for spec in specs_here:
            if spec.pdb_source not in pdb_cache:
                pdb_cache[spec.pdb_source] = self._load_pdb(spec.pdb_source)
            pdbs_by_source[spec.pdb_source] = pdb_cache[spec.pdb_source]

        # Protein packing works from one running occupancy grid: True
        # means "an instance may not go here", so it starts as the
        # complement of the region (which already has obstacles removed
        # above) and accumulates every instance placed below. It needs no
        # distance field -- gold beads and membrane instances, which are
        # placed by bounding sphere, are the only things that still build
        # one, and they build their own.
        pack_voxel, pack_shape, pack_factor = self._packing_grid(
            self.target_shape, voxel_size
        )
        occupancy = ~region_mask
        if pack_factor > 1:
            occupancy = _downsample_mask_maxpool(occupancy, pack_factor, pack_shape)

        exact_specs = [s for s in specs_here if s.n_copies is not None]
        ratio_specs = [s for s in specs_here if s.n_copies is None]

        # Exact-count ("target") species are placed FIRST within this
        # region; the ratio-weighted ones then fill what is left of the
        # occupancy grid, which carries the targets forward.
        if exact_specs:
            exact_pdbs = [pdbs_by_source[s.pdb_source] for s in exact_specs]
            exact_radii = torch.cat(
                [
                    torch.full((s.n_copies,), float(pdb.max_diameter) / 2.0)  # type: ignore[arg-type]
                    for s, pdb in zip(exact_specs, exact_pdbs)
                ]
            )
            exact_species_map = torch.cat(
                [
                    torch.full((s.n_copies,), i, dtype=torch.long)  # type: ignore[arg-type]
                    for i, s in enumerate(exact_specs)
                ]
            )
            with (
                phase(
                    f"  Target packing ({location})",
                    disable=not self.progressbars,
                    header=False,
                ),
                status(
                    f"Packing {int(exact_radii.numel())} target instance(s) "
                    f"({location})",
                    disable=not self.progressbars,
                ),
            ):
                coords, exact_rotations, accepted_idx, occupancy = self._pack_shapes(
                    exact_pdbs,
                    exact_species_map,
                    pack_shape,
                    pack_voxel,
                    pack_factor,
                    occupancy,
                )
            n_requested = int(exact_radii.numel())
            n_placed = int(accepted_idx.numel())
            if n_placed < n_requested:
                warnings.warn(
                    f"TomogramSpecimenGenerator: only {n_placed}/"
                    f"{n_requested} exact-count instances fit in the "
                    f"'{location}' region without colliding -- it may be "
                    "too small or too crowded for the requested "
                    "n_copies.",
                    stacklevel=2,
                )
            accepted_species_idx = exact_species_map[accepted_idx]

            with phase(
                f"  Target rendering ({location})",
                disable=not self.progressbars,
                header=False,
            ):
                volume, instance_labels, next_instance_id = self._render_species_pool(
                    exact_specs,
                    exact_pdbs,
                    coords,
                    accepted_species_idx,
                    volume,
                    instance_labels,
                    next_instance_id,
                    location,
                    voxel_size,
                    role="target",
                    rotations=exact_rotations,
                )

        # Ratio-weighted ("filler") species, drawn to fill
        # occupancy_fraction of this region -- avoiding the exact-count
        # placements above (if any), the obstacles (filaments, beads) and
        # the membrane shell, all folded into `occupancy` by this point.
        if ratio_specs:
            ratio_pdbs = [pdbs_by_source[s.pdb_source] for s in ratio_specs]
            species_radii = torch.tensor(
                [float(pdb.max_diameter) / 2.0 for pdb in ratio_pdbs]
            )
            species_ratios = torch.tensor([s.ratio for s in ratio_specs])

            # `occupancy_fraction` is a budget in real footprint volume,
            # which is what the packer collides. `draw_species_pool`'s own
            # default is bounding-sphere volume instead, ~5.6x larger per
            # species; passing the measured masks here is what keeps the
            # setting meaning the same thing as the geometry.
            pool_volumes = torch.tensor(
                [
                    float(self._species_mask(pdb, voxel_size).sum()) * voxel_size**3
                    for pdb in ratio_pdbs
                ]
            )

            pool_radii, pool_species_idx = draw_species_pool(
                species_radii,
                species_ratios,
                self.occupancy_fraction,
                region_volume_a3,
                seed=self.seed,
                species_volumes=pool_volumes,
            )

            with (
                phase(
                    f"  Filler packing ({location})",
                    disable=not self.progressbars,
                    header=False,
                ),
                status(
                    f"Packing filler instances ({location})",
                    disable=not self.progressbars,
                ),
            ):
                coords, filler_rotations, accepted_idx, occupancy = self._pack_shapes(
                    ratio_pdbs,
                    pool_species_idx,
                    pack_shape,
                    pack_voxel,
                    pack_factor,
                    occupancy,
                )
            if accepted_idx.numel() == 0:
                warnings.warn(
                    f"TomogramSpecimenGenerator: placed 0 filler "
                    f"instances in '{location}' -- no rotated footprint "
                    f"fit anywhere in the region's "
                    f"{region_voxels:,} free voxels. Enlarge the "
                    "compartment, or declare a smaller species for "
                    "this region.",
                    stacklevel=2,
                )
            accepted_species_idx = pool_species_idx[accepted_idx]

            with phase(
                f"  Filler rendering ({location})",
                disable=not self.progressbars,
                header=False,
            ):
                volume, instance_labels, next_instance_id = self._render_species_pool(
                    ratio_specs,
                    ratio_pdbs,
                    coords,
                    accepted_species_idx,
                    volume,
                    instance_labels,
                    next_instance_id,
                    location,
                    voxel_size,
                    role="filler",
                    rotations=filler_rotations,
                )

        phase_done(
            f"{location.capitalize()} species",
            _location_phase_start,
            disable=not self.progressbars,
        )

        return volume, instance_labels, next_instance_id

    def _stamp_carbon_film(
        self,
        volume: torch.Tensor,
        target_shape: tuple[int, int, int],
        voxel_size: float,
    ) -> torch.Tensor:
        """Paint `self.carbon_film_spec`'s carbon support film directly into
        `volume` (a plain add -- there's nothing else occupying `volume`
        yet at this point in `generate()`, so max-merge vs. add makes no
        difference here). See module docstring for why placement isn't
        made carbon-aware (a documented, CTS-parity limitation, not new
        here)."""
        carbon_film_spec = self.carbon_film_spec
        assert carbon_film_spec is not None
        # Bulk carbon takes `bulk_parameterization`, NOT this specimen's
        # `parameterization`: Shtyrov is fitted for biomolecules, and its
        # "C(CCC)" proxy puts amorphous carbon 43% above the holography value
        # per unit density, where Kirkland, Lobato and Peng agree to 0.5%.
        carbon_gen = CarbonFilmGenerator(
            voxel_size=voxel_size,
            parameterization=self.bulk_parameterization,
            seed=self.seed,
            device=volume.device,
        )
        grid_rng = np.random.default_rng(self.seed)
        edge_fraction = carbon_film_spec.edge_fraction
        if isinstance(edge_fraction, tuple):
            edge_fraction = grid_rng.uniform(*edge_fraction)
        hole_center = edge_hole_center(
            target_shape=target_shape,
            voxel_size=voxel_size,
            hole_radius=carbon_film_spec.hole_radius,
            edge_fraction=edge_fraction,
            side=carbon_film_spec.edge_side,
            rng=grid_rng,
        )
        film = carbon_gen.generate(
            target_shape=target_shape,
            thickness=carbon_film_spec.thickness,
            hole_radius=carbon_film_spec.hole_radius,
            hole_center=hole_center,
            edge_roughness=carbon_film_spec.edge_roughness,
        )
        # Added in place: the caller reassigns `volume` from this return, so
        # there is no second reference to preserve, and `volume + ...` held
        # two full-size volumes at once -- 1.61 GiB each at the canonical
        # 300x1200x1200.
        volume += film.density.to(volume.device)
        return volume

    def _stamp_beads(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        obstacle_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Place and render every `bead_specs` population -- gold spheres,
        placed by bounding sphere with the RSA backend
        (`pack_hard_spheres_3d`) that also places membrane instances. A
        sphere is its own bounding sphere, so for beads that approximation
        is exact up to the rendered surface roughness.

        Sampling is restricted to outside the membrane shell (beads
        embedded in the bilayer would be a glaring, physically wrong
        artifact -- gold's mean inner potential dwarfs everything else in
        the volume) and outside `obstacle_mask` (already-placed filaments,
        if any). NOT region-gated to cytosol/lumen -- fiducials sit in the
        ice itself, see `TomogramBeadSpec`'s own docstring.
        """
        target_shape = self.target_shape
        box = (
            target_shape[0] * voxel_size,
            target_shape[1] * voxel_size,
            target_shape[2] * voxel_size,
        )

        # self.regions is always set by this point in generate() (region
        # classification runs unconditionally, right after membrane
        # compositing -- see that method's own body).
        assert self.regions is not None
        forbidden = self.regions["shell"].cpu()
        if obstacle_mask is not None:
            forbidden = forbidden | obstacle_mask.cpu()
        allowed = ~forbidden

        allowed_field, exclusion_field, field_voxel_size = (
            _allowed_region_exclusion_field(allowed, target_shape, voxel_size)
        )

        # Radii are drawn here, before packing, so each bead's own size is
        # what the collision test reserves room for.
        rng = torch.Generator().manual_seed(
            0 if self.seed is None else int(self.seed) + 991
        )
        radii = torch.cat(
            [
                torch.full((spec.count,), spec.radius_range[0])
                if spec.radius_range[1] <= spec.radius_range[0]
                else spec.radius_range[0]
                + (spec.radius_range[1] - spec.radius_range[0])
                * torch.rand(spec.count, generator=rng)
                for spec in self.bead_specs
            ]
        )
        with status(
            f"Packing {int(radii.numel())} gold fiducial bead(s)",
            disable=not self.progressbars,
        ):
            coords, accepted_idx = pack_hard_spheres_3d(
                radii,
                box,
                gap=self.gap,
                seed=self.seed,
                device="cpu",  # see self.device's own docstring
                exclusion_distance_field=exclusion_field,
                field_voxel_size=field_voxel_size,
                sampling_mask=allowed_field,
                max_passes=self.region_max_passes,
                clip_axes=self.clip_axes,
            )
        n_requested = int(radii.numel())
        n_placed = int(accepted_idx.numel())
        if n_placed < n_requested:
            warnings.warn(
                f"TomogramSpecimenGenerator: only {n_placed}/{n_requested} "
                "gold fiducial beads fit without colliding with the "
                "membrane shell/already-placed filaments.",
                stacklevel=2,
            )
        if n_placed == 0:
            return volume, instance_labels, next_instance_id

        accepted_radii = radii[accepted_idx]

        # Gold likewise takes `bulk_parameterization` -- it is a bulk metal,
        # and the Shtyrov tables have no elemental gold at all.
        bead_gen = BeadGenerator(
            voxel_size=voxel_size,
            parameterization=self.bulk_parameterization,
            roughness=self.bead_roughness,
        )
        instance_ids = torch.arange(
            next_instance_id, next_instance_id + n_placed, dtype=torch.int32
        )
        next_instance_id += n_placed

        # One bead at a time: each is an independent realisation (its own
        # grain, orientation and, for a [low, high] radius, its own size),
        # so there is no shared template to batch over.
        for i in range(n_placed):
            bead = bead_gen.generate(radius=float(accepted_radii[i]))
            volume = insert_particles_into_micrograph(
                bead.density.to(volume.device).unsqueeze(0),
                coords[i : i + 1],
                pixel_size=voxel_size,
                micrograph=volume,
            )
            # Label from the bead's geometry, not from `density > 0`: the
            # stochastic fill leaves empty voxels inside the boundary,
            # which a density threshold would carve out of the
            # segmentation.
            binarized = bead.mask.to(volume.device).unsqueeze(0).to(torch.int32) * int(
                instance_ids[i]
            )
            instance_labels = _insert_instance_labels(
                binarized,
                coords[i : i + 1],
                voxel_size=voxel_size,
                labels=instance_labels,
            )

        for i in range(n_placed):
            self.bead_instances.append(
                BeadPlacement(
                    radius=float(accepted_radii[i]),
                    position_xyz=coords[i].detach().cpu(),
                    instance_id=int(instance_ids[i]),
                )
            )

        return volume, instance_labels, next_instance_id

    def _packing_grid(
        self, target_shape: tuple[int, int, int], voxel_size: float
    ) -> tuple[float, tuple[int, int, int], int]:
        """Coarse collision grid for `packing_voxel_size`: (voxel, shape, factor)."""
        if self.packing_voxel_size is None:
            return _coarsen_grid_to_budget(
                target_shape, voxel_size, _MAX_PACKING_GRID_VOXELS
            )
        if self.packing_voxel_size <= voxel_size:
            return voxel_size, target_shape, 1
        factor = max(1, int(round(self.packing_voxel_size / voxel_size)))
        return _coarsen_grid_by(target_shape, voxel_size, factor)

    def _species_mask(self, pdb: PDB, voxel_size: float) -> torch.Tensor:
        """
        This species' footprint mask for protein packing, built
        once per (structure, voxel size, gap) and reused across regions,
        across the pool-sizing/packing steps and across `generate` calls.
        The structure is identified by its file and the parse flags that
        change its atoms, never by object identity.

        Always at the RENDER voxel size. Coarsening for a coarser packing
        grid happens per rotated orientation inside the packer, not here --
        see `packing_voxel_size`.
        """
        key = (
            os.path.realpath(pdb.filepath),
            int(pdb.coordinates.shape[0]),
            _wants_atom_species(self.parameterization),
            self.readd_hydrogens,
            self.monomer_library_path,
            voxel_size,
            self.gap,
        )
        if key not in self._mask_cache:
            self._mask_cache[key] = build_species_mask(
                pdb.coordinates, voxel_size, gap=self.gap
            )
        return self._mask_cache[key]

    def _pack_shapes(
        self,
        pdbs: list[PDB],
        species_idx: torch.Tensor,
        pack_shape: tuple[int, int, int],
        pack_voxel: float,
        factor: int,
        occupancy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pack via `..packing.pack_shapes_3d` (exact rotated footprints against
        a running occupancy grid).

        Region restriction and obstacle avoidance both arrive already folded
        into `occupancy` (True = forbidden), so no exclusion distance field
        or sampling mask is needed here -- see `..packing._shape`'s module
        docstring.

        Parameters
        ----------
        pdbs : list of PDB
            One per species, indexed by `species_idx`.
        species_idx : torch.Tensor
            Species index per candidate instance, shape (N,).
        pack_shape : tuple of int
            Collision grid shape, from `_packing_grid`.
        pack_voxel : float
            Collision grid voxel size, A.
        factor : int
            ``pack_voxel / voxel_size``; 1 when packing at render resolution.
        occupancy : torch.Tensor
            Boolean, shape `pack_shape`, True where an instance may not go.
            Already at packing resolution.

        Returns
        -------
        coords : torch.Tensor
            Accepted centers (x, y, z), A, box-centered, shape (M, 3).
        rotations : torch.Tensor
            Orientation each instance was accepted at, shape (M, 3, 3). Must
            be reused at render time.
        accepted_idx : torch.Tensor
            Indices into `species_idx`, shape (M,).
        occupancy : torch.Tensor
            Updated occupancy grid, to carry into the next packing stage.
        """
        # `occupancy` arrives ALREADY at packing resolution -- it is resolved
        # once per region and threaded through both the target and filler
        # stages, so coarsening it here would coarsen the second stage's
        # input a second time.
        # Fine masks, with the factor handed to the packer: it rotates each
        # at this resolution and pools the result onto the packing grid.
        # Pooling here instead loses containment under rotation -- see
        # `..packing._shape._rotation_cache`.
        masks = [self._species_mask(pdb, pack_voxel / factor) for pdb in pdbs]
        region_mask = ~occupancy
        if not bool(region_mask.any()):
            # The region passed generate()'s own `region_voxels == 0` check at
            # render resolution and still has no free voxel HERE: coarsening
            # marks a packing voxel occupied if any fine voxel in it is, so a
            # compartment only a few fine voxels wide anywhere -- a small
            # vesicle lumen at 2 A, on the auto 4 A packing grid -- can lose
            # every voxel it had. Same outcome as the fine-grid case, and for
            # the same reason: nothing fits, so nothing is placed.
            warnings.warn(
                "TomogramSpecimenGenerator: the region has no free voxel on "
                f"the {pack_voxel:.1f} A packing grid (it has some at the "
                f"{pack_voxel / factor:.1f} A render grid, but a compartment "
                "thinner than a packing voxel is swallowed by coarsening) -- "
                "placing nothing in it. A smaller packing_voxel_size keeps "
                "such compartments, at a higher packing cost.",
                stacklevel=2,
            )
            return (
                torch.empty((0, 3)),
                torch.empty((0, 3, 3)),
                torch.empty((0,), dtype=torch.long),
                occupancy,
            )
        return pack_shapes_3d(
            masks,
            species_idx,
            pack_shape,
            pack_voxel,
            occupancy=occupancy,
            region_mask=region_mask,
            n_orientations=self.n_orientations,
            pool_factor=factor,
            max_retries=self.packing_max_retries,
            seed=self.seed,
            device=self.device,
            clip_axes=self.clip_axes,
        )

    def _load_pdb(self, source: str) -> PDB:
        """Parse one structure with this generator's PDB settings.

        Shared by the protein stage's fallback load and filament/microtubule
        rendering, so every structure is parsed with the same flags.
        """
        return PDB(
            source,
            pdb_cache_dir=self.pdb_cache_dir,
            verbose=False,
            compute_atom_species=_wants_atom_species(self.parameterization),
            readd_hydrogens=self.readd_hydrogens,
            monomer_library_path=self.monomer_library_path,
        )

    def _build_species_template(
        self,
        pdb: PDB,
        voxel_size: float,
        device: str | torch.device,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render one structure's potential template, returned on `self.device`.

        `coordinates` replaces ``pdb.coordinates`` (default None: use them
        as-is); filament rendering passes principal-axis-aligned ones.
        """
        n = estimate_protein_box_size(pdb.max_diameter, voxel_size)
        builder = PotentialBuilder(
            n_xyz=n,
            dx=voxel_size,
            atomic_numbers=pdb.atomic_numbers,
            progressbars=False,
            parameterization=self.parameterization,
            # A replacement `coordinates` is rotation-only, so species stay
            # aligned with coordinates.
            atom_species=pdb.atom_species,
            b_factors=pdb.b_factors if self.use_deposited_bfactors else None,
        ).to(device)
        if coordinates is None:
            coordinates = pdb.coordinates
        return builder.forward(coordinates, method="analytic").to(self.device)

    def _render_species_pool(
        self,
        specs: list[TomogramProteinSpec],
        pdbs: list[PDB],
        coords: torch.Tensor,
        accepted_species_idx: torch.Tensor,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        location: str,
        voxel_size: float,
        role: Literal["target", "filler"] = "target",
        rotations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        active_species_i = [
            species_i
            for species_i in range(len(specs))
            if bool((accepted_species_idx == species_i).any())
        ]
        # Build every active species' potential template up front (optionally
        # concurrently across self.render_workers threads/self.render_devices
        # -- see TomogramSpecimenGenerator's own render_workers docstring),
        # then run the rotate/insert loop below. That loop
        # mutates volume/instance_labels/next_instance_id in place across
        # iterations and is comparatively cheap (batched GPU tensor ops), so
        # it stays sequential -- only the per-species PDB fetch/parse +
        # PotentialBuilder.forward is parallelized.
        with TqdmProgress(
            transient=True, disable=not self.progressbars or not active_species_i
        ) as progress:
            template_task = progress.add_task(
                f"Rendering species templates ({location}, {role})",
                total=len(active_species_i),
            )
            templates = build_templates_concurrently(
                keys=active_species_i,
                build_one=lambda species_i, device: self._build_species_template(
                    pdbs[species_i], voxel_size, device
                ),
                devices=self.render_devices,
                max_workers=self.render_workers,
                on_result=lambda species_i: progress.update(
                    template_task,
                    advance=1,
                    description=(
                        f"Rendered {specs[species_i].pdb_source} ({location}, {role})"
                    ),
                ),
            )

        with TqdmProgress(
            transient=True, disable=not self.progressbars or not active_species_i
        ) as progress:
            render_task = progress.add_task(
                f"Placing species ({location}, {role})", total=len(active_species_i)
            )
            for species_i, spec in enumerate(specs):
                mask = accepted_species_idx == species_i
                if not bool(mask.any()):
                    continue
                template = templates[species_i]

                species_coords = coords[mask]
                n_instances = species_coords.shape[0]
                progress.update(
                    render_task,
                    description=(
                        f"Placing {spec.pdb_source} ({n_instances} "
                        f"instance{'' if n_instances == 1 else 's'}, "
                        f"{location}, {role})"
                    ),
                )
                if rotations is not None:
                    # Shape-based packing already committed to an
                    # orientation per instance -- that rotation IS part of
                    # the collision result, so re-drawing here would render
                    # a volume that does not match the geometry the packer
                    # actually tested (overlaps, and picks that miss).
                    R = rotations[mask].to(self.device)
                else:
                    R = random_rotation_matrix(n_instances, device=self.device)
                if R.dim() == 2:
                    R = R.unsqueeze(0)
                theta = build_affine_matrix(R)

                instance_ids = torch.arange(
                    next_instance_id,
                    next_instance_id + n_instances,
                    dtype=torch.int32,
                    device=self.device,
                )
                next_instance_id += n_instances

                volume, instance_labels = _insert_rotated_copies(
                    template,
                    theta,
                    species_coords,
                    instance_ids,
                    volume,
                    instance_labels,
                    voxel_size,
                    self.chunk_size,
                )

                # One device->host transfer per array, not one per
                # instance: a per-instance `.cpu()` is a separate copy AND
                # an implicit sync each, ~31 us apiece, so a 21k-instance
                # filler pool spent ~0.7 s here doing 43k of them.
                coords_host = species_coords.detach().cpu()
                R_host = R.detach().cpu()
                ids_host = instance_ids.cpu().tolist()
                for i in range(n_instances):
                    self.placements.append(
                        TomogramPlacement(
                            species_id=spec.pdb_source,
                            location=location,
                            position_xyz=coords_host[i],
                            rotation_matrix=R_host[i],
                            instance_id=ids_host[i],
                            role=role,
                        )
                    )
                progress.update(render_task, advance=1)

        return volume, instance_labels, next_instance_id
