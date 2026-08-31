"""
TomogramSpecimenGenerator: assembles a full specimen tomogram from an
optional organic membrane -- transmembrane proteins on the bilayer, plus
densely packed cytosolic and vesicle-lumen protein populations, region-
gated against the membrane's own geometry so a "lumen" species can only
land inside an enclosed compartment and a "cytosol" species can only land
outside one -- plus optional scattered filament species (e.g. F-actin),
an optional carbon support film, and optional gold fiducial beads.

This is the ONE specimen generator behind `specter build tomogram`:
`membrane_instances` may be empty (no membranes at all -- the whole volume
is then one cytosol region, since `classify_membrane_regions` already
treats "no membrane" that way by design, see `._regions`'s own docstring),
`protein_specs` may be empty (membranes/filaments with no packed protein
population), and `filament_specs` may be empty -- any combination is
valid as long as at least one of the three is non-empty. There is no
longer a separate non-membrane sphere-packing generator/mode: a species in
`protein_specs` can be placed either at an exact count
(`TomogramProteinSpec.n_copies`, ground-truth "target" semantics) or
ratio-weighted up to `occupancy_fraction` of its region
(`TomogramProteinSpec.ratio`, "filler"/crowding semantics), matching what
the now-deleted `SpherePackingSpecimenGenerator`'s two-stage target/filler
split used to provide, now region-gated too.

Generation order is membranes, then filaments, then protein fill (exact-
count species per region before ratio-weighted ones) -- see `generate()`'s
own body. Placed filament voxels (`instance_labels > 0` right after
`_stamp_filaments`) are folded into the exclusion field/sampling mask used
by the protein-fill stage, so packed spheres are kept clear of already-
placed filaments the same way they're already kept clear of the membrane
shell -- both are bounding-sphere-vs-distance-field approximations, not an
exact voxel-overlap guarantee (a placed protein's true, non-spherical
rendered shape can still graze a filament or the shell very close to the
boundary; consistent with the approximate, "reject and move on" philosophy
already used everywhere else in this generator).

Composes three independently-developed pieces rather than reimplementing
any of them: ``specimen.membrane.MembraneGenerator`` (organic shape +
transmembrane placement, unmodified), :func:`.classify_membrane_regions`
(shell/lumen/cytosol masks via connected-components flood-fill, this
subpackage), and ``specimen.packing.pack_hard_spheres_3d``'s
`exclusion_distance_field` (obstacle- and region-aware RSA, this session).
Deliberately built on the RSA backend, not the periodic force-biased
relaxation that used to live alongside it as `pack_hard_spheres_3d_dense`
(since deleted -- see git history if it's ever needed as a reference
again): a production-scale tomogram (hundreds of voxels per axis) draws
candidate pools far too large for that backend's per-iteration Python-loop
cost to stay practical (verified to run into the hours at that scale) --
RSA's own ~28-41% ceiling, reached in seconds, was the actual target here,
not the force-biased backend's higher-but-impractical one.

A clean-room second approach relative to CTS (CryoTomoSim), not a port of
its placement/membrane algorithms -- those used to live in a separate
generator (``cryotomosim.py``, ``_cts_membrane.py``, ``_cts_placement.py``),
deleted once this generator reached feature parity with it (carbon
film/gold beads were the last two gaps -- see git history for that
generator if the CTS algorithm itself is ever needed as a reference
again). This generator DOES still reuse ``.._carbon``
(``CarbonFilmGenerator``) and ``.._grid`` (``BeadGenerator``) for the
carbon film/gold bead physics below -- those modules are generic
bulk-material potential code with no CTS-specific placement logic of their
own (see each module's own docstring), so they outlived the deleted
generator rather than going with it.
``_cts_membrane.py`` did NOT similarly outlive it: its only other consumer
was ``specimen.membrane``'s own deprecated ``shape_backend="alpha_shape"``,
removed in the same cleanup -- see git history if either is ever needed as
a reference again.

Carbon film (``carbon_film_spec``) is painted directly into the shared canvas
before anything else is placed, then everything placed afterward avoids
it -- membrane placement and filaments here, and (via
``classify_membrane_regions`` reading carbon's own high density as
"shell", the same bucket a real membrane bilayer occupies) beads and
cytosol/lumen protein fill below. Membrane placement itself only avoids
carbon via a bounding-sphere-vs-distance-field approximation (the same
kind used everywhere else in this module), so an irregular organelle's
true rendered shape can still graze it; whatever part of an instance's
own rendered density would land on carbon regardless gets clipped
(zeroed) right before it's merged into the shared canvas, as a safety
net, so both the composited volume and that instance's own ground-truth
shell label consistently exclude it (see the `to_composite` loop below).
Filament placement itself has no obstacle-avoiding random
walk (a bigger algorithmic change than clipping density post-hoc), so a
monomer instance landing inside the film is simply dropped after the
fact (see ``_stamp_filaments``) rather than steered around it -- the one
remaining case unhandled here.

Gold fiducial beads (``bead_specs``) are scattered via the same RSA
backend (``pack_hard_spheres_3d``) used for membrane instances and protein
packing -- unlike the deleted CTS-derived generator's own bead placement,
which stayed on a slower sequential particle placer with a standing TODO
noting beads are "literally already spheres" and a natural RSA candidate.
Beads are placed right after filaments (see
``_stamp_beads``), avoiding the membrane shell and any already-placed
filaments, and are themselves then avoided by the cytosol/lumen protein-
fill stage that follows (folded into the same ``instance_labels``-derived
obstacle mask filaments already use) -- a real accuracy improvement over
the CTS port's fully independent, unaware-of-anything-placed-after-it bead
placement.

Supports MULTIPLE independently-configured membrane instances
(:class:`MembraneInstance`, each with its own `MembraneGenerator` --
potentially a different `shape_backend` per instance) composited into
one shared tomogram volume:
generate each instance in its own centered local frame (unmodified
`MembraneGenerator`, no changes needed there), max-merge the resulting
density volumes into the shared canvas, then classify shell/lumen/cytosol
regions ONCE on the composite -- `classify_membrane_regions`'s connected-
components approach already handles multiple disjoint compartments (several
separate vesicles, whether from one instance or several) without any
special-casing.

Each instance renders on its OWN local working grid -- typically much
smaller than the shared canvas (`MembraneGenerator`'s own `target_shape`
auto-sizes from the organelle's size when omitted; see that class's own
docstring) -- then `clip_insert_bounds`-based compositing crops/places it
into the shared canvas at `position_xyz`, the same mechanism already used
for transmembrane protein templates. This was always how the compositing
math worked (`_insert_volume_max`/`_insert_shell_label` never required a
same-shape `local`); it just wasn't exercised until `MembraneGenerator`
gained auto-sizing, since giving every instance a hand-picked box the size
of the whole tomogram was the only practical option before that.

Every instance's `position_xyz` is resolved by `generate()` via
`pack_hard_spheres_3d` (the same RSA backend used for cytosol/lumen
protein packing below), treating each instance as a bounding sphere -- an
instance that doesn't fit without colliding (with another membrane
instance, the box walls, or the carbon film) is dropped rather than
retried at a new position, matching this module's "reject and move on"
philosophy elsewhere. There is no manual-placement override; every
instance goes through the same collision check. An instance whose
own `clipped_at_boundary` ends up `True` after `generate()` (its local grid
was too small for what actually got drawn) is also dropped, with its own
warning -- caught even though the bounding-sphere check above already tries
to avoid this, since that check is necessarily approximate for
`swept_spline`'s wandering-path shape.

Per-instance voxel labels exist for TWO separate categories:
`membrane_labels` (which membrane instance a shell voxel belongs to, new)
and `instance_labels` (which cytosol/lumen PROTEIN instance a voxel
belongs to, as before). Transmembrane placements still get no per-instance
voxel labels (their density is correctly present in the volume via
`MembraneGenerator.place_transmembrane` itself, unmodified here) -- a
documented gap, not an oversight.

Optionally also scatters filament species (e.g. F-actin, microtubules)
through the tomogram via ``specimen.filament.place_filaments`` --
specter-native random-walk placement, with no region-gating (filaments are
dropped anywhere in the volume regardless of cytosol/lumen/membrane-shell
classification) and no collision avoidance against the membrane shell or
against each other.
Rendered right after membranes, BEFORE cytosol/lumen protein packing (see
module docstring) -- unlike the membrane shell, filaments themselves have
no fixed geometry to region-gate against, so this is purely an ordering
choice, not a region restriction -- and stamped as the first entries in
the shared `instance_labels` volume (protein instances continue the same
instance-id counter afterward) so filament monomers are visible in the
segmentation ground truth alongside cytosol/lumen protein instances.
"""

from __future__ import annotations

import gc
import json
import math
import time
import warnings
from pathlib import Path
from collections.abc import Iterator
from typing import Literal

import numpy as np
import torch
from scipy import ndimage

from ...config import ScalarOrRange
from ...arrays import clip_insert_bounds, count_nonzero_chunked
from ...crowding import insert_particles_into_micrograph
from ...pdb import DEFAULT_PDB_CACHE_DIR, PDB, canonical_pdb_source
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, random_rotation_matrix, rotate_volume
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
    place_filaments,
    place_microtubules,
)
from ..membrane import TransmembranePlacement
from ..membrane._placement import align_principal_axis_to_z
from ..packing import (
    build_species_mask,
    draw_species_pool,
    estimate_protein_box_size,
    pack_hard_spheres_3d,
    pack_shapes_3d,
)
from ._helpers import (
    _MAX_PACKING_GRID_VOXELS,
    _OPEN_REGION_STALL_PATIENCE,
    _TIGHT_REGION_FRACTION_THRESHOLD,
    _build_sphere_exclusion_field,
    _diagnose_zero_placements,
    _downsample_mask_maxpool,
    _insert_instance_labels,
    _insert_local_labels,
    _insert_shell_label,
    _insert_volume_max,
    _instance_bounding_radius,
    _position_to_center_index,
    _resolve_exclusion_field_grid,
    _wants_atom_species,
    resolve_accumulator_device,
)
from ._regions import classify_membrane_regions
from ._specs import (
    BeadPlacement,
    MembraneInstance,
    TomogramBeadSpec,
    TomogramPlacement,
    TomogramProteinSpec,
)
from ...progress import TqdmProgress, phase_done, phase_start, status

_INSTANCE_LABEL_REL_THRESHOLD = 0.01


def _filament_runs(
    instances: list[FilamentInstance],
) -> Iterator[list[FilamentInstance]]:
    """Split placed monomers into one list per filament, in order.

    The single definition of where one filament ends and the next begins,
    shared by instance labelling and pick export so the two ground truths
    cannot disagree about what an object is.

    A boundary is a change of ``(code, filament_id)`` between CONSECUTIVE
    instances, not a change of the key alone. Both placers number
    filaments with ``range(spec.n_copies)``, restarting per spec, so every
    spec contributes a filament 0; each emits one filament's monomers
    consecutively, which is what makes runs the right unit.

    One case this cannot separate: a spec contributing exactly one
    filament, followed immediately by a same-`code` filament 0 from the
    next spec. Separating those needs the placers to number filaments
    globally, which is the real fix if it ever matters -- `filament_id` is
    internal and never written to picks.
    """
    run: list[FilamentInstance] = []
    previous: tuple[str, int] | None = None
    for inst in instances:
        key = (inst.code, inst.filament_id)
        if previous is not None and key != previous:
            yield run
            run = []
        previous = key
        run.append(inst)
    if run:
        yield run


class TomogramSpecimenGenerator:
    """
    Assemble a specimen tomogram from any combination of pre-configured
    membranes, filaments, carbon film, gold fiducials, and densely packed
    cytosolic/vesicle-lumen protein populations -- every one of those is
    optional (see the module docstring), so this is the single generator
    behind `specter build tomogram` for membrane and membrane-free
    specimens alike.

    Places any number of distinct species (see `protein_specs` below),
    purely for DENSITY -- each region (cytosol/lumen) is packed as densely
    as `occupancy_fraction` allows via RSA hard-sphere placement, uniformly
    throughout it, with no distributional shaping of any one species' own
    spatial statistics. Contrast `specter.specimen.single_particle.
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
        Target packing density (see `pack_hard_spheres_3d`/
        `draw_species_pool`), applied independently per region -- e.g. 0.2
        for `"lumen"` species targets 20% of the LUMEN's own volume, not
        20% of the whole tomogram. Default 0.2.
    packing_backend : {"shape", "sphere"}, optional
        Collision geometry for protein packing. Default ``"shape"``.

        ``"sphere"`` collides one circumscribing sphere per
        instance via `..packing.pack_hard_spheres_3d`. Cheap, but a
        molecule's envelope is only ~0.178 of its own bounding sphere, so
        achieved macromolecule volume fraction saturates near 0.09 --
        roughly a third of crowded cytoplasm, and `occupancy_fraction` is
        a *sphere-volume* target rather than a real one.

        ``"shape"`` (default) collides the real rotated footprint against a
        running occupancy grid via `..packing.pack_shapes_3d`, which is what
        CryoTomoSim does. Note it interacts with `gap`: see there. Reaches 0.241 on a matched benchmark against
        CryoTomoSim's own 0.240, i.e. physiological crowding. Costs more
        wall time (see `packing_max_retries`) and ignores
        `exclusion_distance_field`-based region handling in favour of the
        occupancy grid, which subsumes it.
    n_orientations : int, optional
        Size of the per-species rotation cache used by
        ``packing_backend="shape"``. Rotating per attempt instead of
        indexing a cache is what dominates a naive implementation's cost
        (profiled at 6.8 A: 3.27 ms to rotate vs 0.08 ms to collision-test).
        Default 256.
    packing_max_retries : int, optional
        Trial positions per instance for ``packing_backend="shape"``. The
        dominant density/speed knob, paired with that function's
        `stall_patience` -- see `..packing.pack_shapes_3d`'s own table.
        Default 1500.
    packing_voxel_size : float, optional
        Run ``packing_backend="shape"``'s collision on a COARSER grid than
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
        straight through to every `pack_hard_spheres_3d` call here (auto-
        placed membrane instances AND cytosol/lumen protein packing). True
        on an axis lets a placed instance's center stay in-bounds while its
        body pokes past that wall (truncated at render time) instead of
        being rejected outright -- e.g. for a tomogram whose xy field of
        view is a crop of a larger cellular region. Default all False.
    region_density_threshold : float, optional
        Passed to `classify_membrane_regions`. Default None (that
        function's own default).
    region_max_passes : int, optional
        `max_passes` (and `stall_patience`, set equal to it -- see
        `pack_hard_spheres_3d`'s `sampling_mask` docstring for why a small
        region needs the early-exit heuristic disabled) for cytosol/lumen
        packing. Default 300, higher than `pack_hard_spheres_3d`'s own
        default 200 since a tight region (e.g. a small vesicle lumen) can
        need more attempts before a geometrically valid spot turns up.
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
        Device for the packing step (see `pack_hard_spheres_3d`'s own
        docstring on why that specifically stays CPU-bound regardless) AND,
        by default, for `_render_species_pool`'s own `PotentialBuilder`
        step -- override the latter alone via `render_devices` below.
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
        packing_backend: Literal["shape", "sphere"] = "shape",
        n_orientations: int = 256,
        packing_max_retries: int = 1500,
        packing_voxel_size: float | None = None,
        clip_axes: tuple[bool, bool, bool] = (False, False, False),
        region_density_threshold: float | None = None,
        region_max_passes: int = 300,
        min_transmembrane_spacing: float = 40.0,
        pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
        parameterization: str = "shtyrov",
        bulk_parameterization: str = "kirkland",
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
        # packing_backend="shape" a nonzero value is also quantized to whole
        # voxels, so a nominal 5 A became a full voxel shell in every
        # direction and cost ~30% of the achievable density (volume fraction
        # 0.197 -> 0.138 on a 121-species filler set at 6.8 A). Gold beads
        # and membrane instances share this value and are content with 0:
        # their bounding spheres merely become tangent.
        self.gap = 0.0
        self.packing_backend = packing_backend
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
        # `device` (GPU, for speed) exactly as before, but each small
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
        # Rasterized footprints for packing_backend="shape", keyed by
        # (PDB identity, voxel_size, gap) -- a species reappearing across
        # regions rasterizes once.
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

        voxel_size = self.voxel_size
        target_shape = self.target_shape
        box = (
            target_shape[0] * voxel_size,
            target_shape[1] * voxel_size,
            target_shape[2] * voxel_size,
        )

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
            _grid_phase_start = phase_start(
                "Carbon film", disable=not self.progressbars
            )
            with status(
                "Generating carbon support film", disable=not self.progressbars
            ):
                volume = self._stamp_carbon_film(volume, target_shape, voxel_size)
            carbon_mask = volume > 0
            phase_done("Carbon film", _grid_phase_start, disable=not self.progressbars)

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

        # Resolve any omitted position_xyz via collision-rejecting random
        # placement, treating each instance as a bounding sphere (see
        # _instance_bounding_radius) -- an instance that doesn't fit
        # without colliding is dropped (never .generate()-called at all,
        # cheaper than generating first and rejecting after), matching
        # this module's own "reject and move on" packing philosophy rather
        # than retrying at new positions. Instances with an explicit
        # position_xyz are placed as given and NOT included in this
        # collision check (a known v1 gap -- see module docstring).
        _membrane_phase_start = phase_start(
            "Membranes", disable=not self.progressbars or not self.membrane_instances
        )

        to_composite: list[MembraneInstance] = []
        if self.membrane_instances:
            radii = torch.tensor(
                [
                    _instance_bounding_radius(mi.generator)
                    for mi in self.membrane_instances
                ]
            )
            if carbon_mask is not None:
                field_voxel_size, field_shape, field_factor = (
                    _resolve_exclusion_field_grid(target_shape, voxel_size)
                )
                allowed = (~carbon_mask).cpu()
                allowed_field = (
                    _downsample_mask_maxpool(allowed, field_factor, field_shape)
                    if field_factor > 1
                    else allowed
                )
                exclusion_field = (
                    torch.from_numpy(
                        ndimage.distance_transform_edt(allowed_field.numpy())
                    ).float()
                    * field_voxel_size
                )
            with status(
                f"Placing {len(self.membrane_instances)} membrane instance(s)",
                disable=not self.progressbars,
            ):
                coords, accepted_idx = pack_hard_spheres_3d(
                    radii,
                    box,
                    gap=self.gap,
                    seed=self.seed,
                    device="cpu",  # see self.device's own docstring
                    clip_axes=self.clip_axes,
                    exclusion_distance_field=(
                        exclusion_field if carbon_mask is not None else None
                    ),
                    field_voxel_size=field_voxel_size
                    if carbon_mask is not None
                    else None,
                    sampling_mask=(allowed_field if carbon_mask is not None else None),
                )
            n_dropped = len(self.membrane_instances) - accepted_idx.numel()
            if n_dropped:
                warnings.warn(
                    f"TomogramSpecimenGenerator: {n_dropped}/"
                    f"{len(self.membrane_instances)} membrane instances did not "
                    "fit without colliding (with each other, the box walls, or "
                    "the carbon film, if any) and were dropped (never "
                    "generated).",
                    stacklevel=2,
                )
            for k, orig_idx in enumerate(accepted_idx.tolist()):
                mi = self.membrane_instances[orig_idx]
                mi.position_xyz = tuple(coords[k].tolist())
                to_composite.append(mi)

        # Generate + place transmembrane proteins per instance, each in its
        # own centered local frame, then composite densities into the
        # shared canvas (max-merge) before any region classification --
        # classify_membrane_regions needs the full composite, not
        # per-instance pieces.
        self.transmembrane_placements = []
        self.placed_membrane_instances = []
        instance_shell_masks: list[tuple[MembraneInstance, torch.Tensor]] = []
        membrane_progress = TqdmProgress(
            transient=True, disable=not self.progressbars or not to_composite
        )
        with membrane_progress as progress:
            membrane_task = progress.add_task(
                "Generating membrane instances", total=len(to_composite)
            )
            for mi in to_composite:
                # Every mi here has a concrete position_xyz by construction:
                # to_composite only ever collects accepted instances, whose
                # position_xyz was just set above.
                assert mi.position_xyz is not None
                mi.generator.generate()
                progress.update(membrane_task, advance=1)
                if mi.generator.clipped_at_boundary:
                    warnings.warn(
                        "TomogramSpecimenGenerator: a membrane instance's own "
                        "working grid was too small for the organelle size it "
                        "actually drew (clipped_at_boundary=True on its "
                        "MembraneGenerator) -- skipped rather than compositing a "
                        "visibly truncated shape. Increase that instance's own "
                        "target_shape/voxel_size, or omit target_shape "
                        "entirely for auto-sizing.",
                        stacklevel=2,
                    )
                    continue
                # The bilayer's own peak, read BEFORE place_transmembrane
                # inserts protein density. A protein template's peak is
                # typically several times a smoothed bilayer's, so taking
                # this afterwards would inflate the shell threshold below
                # and erode the very thing it is meant to outline.
                bare_bilayer = mi.generator.volume
                assert bare_bilayer is not None  # generate() just ran
                bare_peak = float(bare_bilayer.max())
                tm_placements = mi.generator.place_transmembrane(
                    min_spacing_a=self.min_transmembrane_spacing
                )
                offset = torch.tensor(mi.position_xyz, dtype=torch.float32)
                for tp in tm_placements:
                    tp.center_xyz = tp.center_xyz + offset

                # A membrane instance may extend past the tomogram walls
                # (clip_axes), and `_insert_volume_max` clips the part that
                # does -- so a protein embedded out there contributes no
                # density and gets no instance label. Drop its ground-truth
                # entry too, rather than shipping a pick at coordinates the
                # volume does not cover. Same reasoning, and the same
                # center-point granularity, as the carbon-film drop below.
                if tm_placements:
                    half = torch.tensor(
                        [box[2] / 2, box[1] / 2, box[0] / 2], dtype=torch.float32
                    )
                    centers = torch.stack([tp.center_xyz for tp in tm_placements])
                    outside = (centers.abs() >= half).any(dim=1)
                    n_outside = int(outside.sum())
                    if n_outside:
                        warnings.warn(
                            f"TomogramSpecimenGenerator: dropped {n_outside} "
                            "transmembrane protein placement(s) that fell "
                            "outside the tomogram box, on the part of a "
                            "membrane instance clipped by the volume walls "
                            "(no density was rendered for them, so a pick "
                            "there would claim a particle the volume does "
                            "not contain).",
                            stacklevel=2,
                        )
                        tm_placements = [
                            tp
                            for tp, drop in zip(tm_placements, outside.tolist())
                            if not drop
                        ]

                local_volume = mi.generator.volume
                assert local_volume is not None
                if carbon_mask is not None:
                    # Placed instances shouldn't reach here at all in the
                    # common case (their bounding sphere was already kept
                    # clear of carbon by the RSA exclusion field above), so
                    # this is normally a no-op -- it's a safety net for an
                    # irregular organelle whose true rendered shape extends
                    # past its own bounding-sphere approximation. Zeroes
                    # local_volume wherever it would
                    # land on carbon, BEFORE both compositing into `volume`
                    # and the shell_mask/ground-truth labeling below, so
                    # both consistently reflect the clip -- reuses the same
                    # index math `_insert_volume_max` itself uses, rather
                    # than duplicating it.
                    center_zyx = _position_to_center_index(
                        mi.position_xyz, tuple(volume.shape), voxel_size
                    )
                    bounds = clip_insert_bounds(
                        center_zyx, local_volume.shape, volume.shape
                    )
                    if bounds is not None:
                        dst, src = bounds
                        forbidden = carbon_mask[dst].to(local_volume.device)
                        if forbidden.any():
                            warnings.warn(
                                "TomogramSpecimenGenerator: clipped part of a "
                                "membrane instance (an irregular shape "
                                "exceeding its own bounding-sphere estimate) "
                                "that overlapped the carbon film.",
                                stacklevel=2,
                            )
                            local_volume[src] = local_volume[src] * (~forbidden).to(
                                local_volume.dtype
                            )
                            # Same clip on the protein labels, so a
                            # transmembrane instance whose density was just
                            # removed doesn't keep a ground-truth label
                            # sitting on carbon.
                            tm_labels_clip = mi.generator.transmembrane_labels
                            if tm_labels_clip is not None:
                                tm_labels_clip[src] = tm_labels_clip[src] * (
                                    ~forbidden
                                ).to(tm_labels_clip.dtype)

                    # `place_transmembrane` (above) already baked these
                    # placements' own density into local_volume before this
                    # point, so a placement whose center lands on carbon
                    # just had its density zeroed by the clip above too --
                    # this only fixes the separate ground-truth bookkeeping
                    # list (self.transmembrane_placements, what export_picks
                    # writes out), which would otherwise still claim a
                    # particle sits somewhere with no actual density left.
                    # Checked by center point, not full rendered footprint
                    # (same granularity already used for filament monomers
                    # in _stamp_filaments) -- a placement whose center is
                    # just outside carbon but whose template partially
                    # overlapped it keeps its (partially clipped) entry,
                    # matching how e.g. bead/protein exclusion is also
                    # voxel-level, not footprint-exact.
                    if tm_placements:
                        shape_zyx = tuple(volume.shape)
                        z_c, y_c, x_c = (s // 2 for s in shape_zyx)
                        centers = torch.stack([tp.center_xyz for tp in tm_placements])
                        iz = (
                            (z_c + torch.round(centers[:, 2] / voxel_size))
                            .long()
                            .clamp(0, shape_zyx[0] - 1)
                        )
                        iy = (
                            (y_c + torch.round(centers[:, 1] / voxel_size))
                            .long()
                            .clamp(0, shape_zyx[1] - 1)
                        )
                        ix = (
                            (x_c + torch.round(centers[:, 0] / voxel_size))
                            .long()
                            .clamp(0, shape_zyx[2] - 1)
                        )
                        in_carbon = carbon_mask.cpu()[iz, iy, ix]
                        n_dropped_tm = int(in_carbon.sum())
                        if n_dropped_tm:
                            warnings.warn(
                                f"TomogramSpecimenGenerator: dropped "
                                f"{n_dropped_tm} transmembrane protein "
                                "placement(s) clipped by the carbon film "
                                "(density already removed above; this drops "
                                "their now-stale ground-truth pick entries "
                                "too).",
                                stacklevel=2,
                            )
                            tm_placements = [
                                tp
                                for tp, drop in zip(tm_placements, in_carbon.tolist())
                                if not drop
                            ]
                self.transmembrane_placements.extend(tm_placements)
                volume = _insert_volume_max(
                    volume, local_volume, mi.position_xyz, voxel_size
                )
                # Per-instance shell mask, computed and stashed as a bool
                # (~4x smaller than float32, and ~4-8x smaller again than
                # keeping the full density array around) NOW, while
                # local_volume is still cheaply available, rather than in a
                # second pass after every instance has run. A GLOBAL
                # threshold (shared across every instance) would need the
                # full composite's peak, which isn't known until the loop
                # finishes -- forcing every instance's full-resolution
                # array to stay resident simultaneously until then.
                # Confirmed directly: that OOMs well before this loop even
                # finishes, now that generation-resolution decoupling
                # (MembraneGenerator's max_field_voxels) lets a single
                # instance's own volume reach tens of GB. Using THIS
                # instance's own peak instead when region_density_threshold
                # is auto (None) -- a per-instance peak is also the more
                # correct reference for per-instance shell LABELING, since
                # an instance whose own peak is lower (a smaller organelle
                # resolved on a coarser working grid, say) should not have
                # its true shell mislabeled as background just
                # because a brighter sibling set a higher global bar).
                # When region_density_threshold is explicitly set, it's
                # already an absolute density value (not a fraction, see
                # this same fallback below for self.regions), so using it
                # directly here is identical to a shared global threshold
                # -- no behaviour change in that case.
                if self.region_density_threshold is not None:
                    instance_threshold = self.region_density_threshold
                else:
                    instance_threshold = 0.05 * bare_peak if bare_peak > 0 else 0.0
                shell_mask = local_volume > instance_threshold
                # The membrane label is the BILAYER, not the bilayer plus
                # whatever is embedded in it. `_insert_blend` has already
                # decided which voxels the protein displaced lipid from --
                # reuse that decision rather than making a second, looser
                # one here. The proteins themselves become ordinary protein
                # instances just below, so nothing goes unlabelled: the two
                # volumes partition the membrane between them.
                tm_labels = mi.generator.transmembrane_labels
                if tm_labels is not None:
                    shell_mask = shell_mask & (tm_labels == 0)
                    instance_labels = _insert_local_labels(
                        instance_labels,
                        tm_labels,
                        id_offset=next_instance_id - 1,
                        position_xyz=mi.position_xyz,
                        voxel_size=voxel_size,
                    )
                    # Reserve from the count `place_transmembrane` actually
                    # LABELLED (its own `placements`, ids 1..n), not from
                    # `tm_placements` -- the carbon block above may have
                    # pruned that list, and reserving the short count would
                    # let the next membrane's offset collide with this
                    # one's higher ids.
                    next_instance_id += len(mi.generator.placements)
                    mi.generator.transmembrane_labels = None
                shell_mask = shell_mask.cpu()
                instance_shell_masks.append((mi, shell_mask))
                self.placed_membrane_instances.append(mi)
                mi.generator.volume = None
                # Dropping the last reference above is not enough by
                # itself: PyTorch's CUDA caching allocator keeps freed
                # blocks in its own pool rather than returning them to the
                # driver, and each instance's own working/output grid can
                # be a DIFFERENT size (random per-instance organelle size),
                # so the next instance's allocation can fail on
                # fragmentation even though the previous instance's memory
                # was already dereferenced -- confirmed directly: a second
                # instance's OOM here, with the traceback showing several
                # GiB "reserved but unallocated" at the same time as the
                # failing allocation. gc.collect() first in case any
                # tensor is only reachable via a reference cycle (autograd
                # graphs can create these) that plain refcounting wouldn't
                # free promptly.
                del local_volume
                if torch.device(self.device).type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

        # classify_membrane_regions' own threshold: needs the FULL
        # composite's peak (unlike instance_shell_masks' per-instance
        # thresholds above), so can only be resolved after every instance
        # is merged into volume.
        threshold = self.region_density_threshold
        if threshold is None:
            peak = float(volume.max())
            threshold = 0.05 * peak if peak > 0 else 0.0
        self.regions = classify_membrane_regions(volume, threshold)

        membrane_labels = torch.zeros(
            target_shape, dtype=torch.int32, device=self.accumulator_device
        )
        for instance_id, (mi, shell_mask) in enumerate(instance_shell_masks, start=1):
            assert mi.position_xyz is not None  # see identical assert above
            membrane_labels, overlap = _insert_shell_label(
                membrane_labels, shell_mask, instance_id, mi.position_xyz, voxel_size
            )
            if overlap:
                warnings.warn(
                    f"TomogramSpecimenGenerator: membrane instance {instance_id} "
                    "(1-indexed, in membrane_instances order) overlaps a voxel "
                    "already claimed by an earlier instance in membrane_labels "
                    "-- the earlier instance's label wins there (first-write-"
                    "wins). This can happen even with collision-checked "
                    "placement, since the RSA solve treats each instance as a "
                    "bounding sphere while an irregular organelle's true "
                    "rendered shape can extend past that estimate.",
                    stacklevel=2,
                )
        self.membrane_labels = membrane_labels
        phase_done(
            f"Membranes ({len(instance_shell_masks)}/"
            f"{len(self.membrane_instances)} instance(s) placed)",
            _membrane_phase_start,
            disable=not self.progressbars or not self.membrane_instances,
        )
        # instance_shell_masks (up to several GB/instance -- see where it's
        # built above) is never read again after the membrane_labels loop
        # just above, but stays in scope for the rest of this (long)
        # method otherwise -- filaments/packing below are memory-hungry
        # enough at production scale that leaving it needlessly resident
        # is worth avoiding explicitly rather than waiting for generate()
        # to return.
        del instance_shell_masks

        # Filaments (then gold fiducial beads, see below) render right
        # after membranes, BEFORE cytosol/lumen protein packing (see module
        # docstring) -- obstacle_mask (voxels they actually occupy, from
        # instance_labels before any protein instance touches it) is then
        # folded into the per-region exclusion field/sampling_mask below,
        # the same mechanism already used to keep packed spheres clear of
        # the membrane shell.
        if self.filament_specs or self.microtubule_specs:
            _filament_phase_start = phase_start(
                "Filaments", disable=not self.progressbars
            )
            if self.filament_specs:
                with status(
                    f"Placing {len(self.filament_specs)} filament species",
                    disable=not self.progressbars,
                ):
                    volume, instance_labels, next_instance_id = self._stamp_filaments(
                        volume,
                        instance_labels,
                        next_instance_id,
                        voxel_size,
                        carbon_mask,
                    )
            else:
                self.filament_instances = []
            if self.microtubule_specs:
                with status(
                    f"Placing {len(self.microtubule_specs)} microtubule species",
                    disable=not self.progressbars,
                ):
                    volume, instance_labels, next_instance_id = (
                        self._stamp_microtubules(
                            volume,
                            instance_labels,
                            next_instance_id,
                            voxel_size,
                            carbon_mask,
                        )
                    )
            phase_done(
                f"Filaments ({len(self.filament_instances)} monomer instance(s), "
                f"{len(self.microtubule_instances)} microtubule(s))",
                _filament_phase_start,
                disable=not self.progressbars,
            )
            obstacle_mask = instance_labels > 0
            if self.microtubule_instances:
                # A microtubule's lumen is EMPTY but not accessible: it is
                # sealed by the tube wall. Occupied-voxel exclusion alone
                # would happily pack cytosolic protein inside it, which is
                # exactly what lumenal particles are not (microtubule inner
                # proteins are explicitly out of scope -- see `_lattice`).
                obstacle_mask = obstacle_mask | self._microtubule_lumen_mask(
                    voxel_size, obstacle_mask.device
                )
        else:
            self.filament_instances = []
            obstacle_mask = None

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

        self.placements = []
        pdb_cache: dict[str, PDB] = {}

        # Pre-load every unique cytosol/lumen pdb_source ONCE, up front,
        # concurrently across self.render_workers -- measured directly on a
        # 161-species production-scale run that PDB fetch+parse (not
        # rendering) was the single largest bottleneck (~45% of total wall
        # time) precisely because this loop used to run fully serially, one
        # species at a time, entirely outside render_workers' reach. Uses
        # PROCESSES, not threads (build_pdb_cache_concurrently, not
        # build_templates_concurrently) -- also measured directly:
        # thread-pooling this specific step gave ZERO wall-clock benefit
        # despite dispatching correctly, because Biopython's structure
        # parser doesn't release the GIL for most of its work. See
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

        for location in ("cytosol", "lumen"):
            specs_here = [s for s in self.protein_specs if s.location == location]
            if not specs_here:
                continue
            _location_phase_start = phase_start(
                f"{location.capitalize()} species", disable=not self.progressbars
            )

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
                continue
            region_volume_a3 = region_voxels * voxel_size**3
            region_fraction = region_voxels / (
                target_shape[0] * target_shape[1] * target_shape[2]
            )
            region_stall_patience = (
                self.region_max_passes
                if region_fraction < _TIGHT_REGION_FRACTION_THRESHOLD
                else min(_OPEN_REGION_STALL_PATIENCE, self.region_max_passes)
            )

            pdbs_by_source: dict[str, PDB] = {}
            for spec in specs_here:
                if spec.pdb_source not in pdb_cache:
                    pdb_cache[spec.pdb_source] = PDB(
                        spec.pdb_source,
                        pdb_cache_dir=self.pdb_cache_dir,
                        verbose=False,
                        compute_atom_species=_wants_atom_species(self.parameterization),
                        readd_hydrogens=self.readd_hydrogens,
                        monomer_library_path=self.monomer_library_path,
                    )
                pdbs_by_source[spec.pdb_source] = pdb_cache[spec.pdb_source]

            # field_voxel_size/field_shape: the grid exclusion_field/
            # region_mask_field/sampling_mask below are built and sampled
            # at -- coarser than voxel_size at production scale, see
            # _resolve_exclusion_field_grid's own docstring for why this
            # is safe. exact_specs/ratio_specs' pack_hard_spheres_3d calls
            # pass field_voxel_size (not voxel_size) accordingly.
            field_voxel_size, field_shape, field_factor = _resolve_exclusion_field_grid(
                target_shape, voxel_size
            )
            region_mask_field = (
                _downsample_mask_maxpool(region_mask, field_factor, field_shape)
                if field_factor > 1
                else region_mask
            )

            # Deliberately kept on CPU (not self.device) -- pack_hard_spheres_3d
            # is always called with device="cpu" below (see its own docstring),
            # and its internal _sample_exclusion_distance re-touches this
            # field on EVERY pass, not once -- if it started on a different
            # device (e.g. self.device="cuda"), that per-pass .to() would
            # silently re-copy the whole field GPU<->CPU every single pass.
            # Confirmed directly: this was responsible for 414 of 480s (86%)
            # in a real profiled run, not the RSA algorithm itself.
            exclusion_field = (
                torch.from_numpy(
                    ndimage.distance_transform_edt(region_mask_field.cpu().numpy())
                ).float()
                * field_voxel_size
            )

            # packing_backend="shape" works from one running occupancy grid
            # instead of exclusion_field/region_mask_field: True means "an
            # instance may not go here", so it starts as the complement of
            # the region (which already has obstacles removed above) and
            # accumulates every instance placed below.
            pack_voxel, pack_shape, pack_factor = self._packing_grid(
                target_shape, voxel_size
            )
            occupancy = ~region_mask
            if pack_factor > 1:
                occupancy = _downsample_mask_maxpool(occupancy, pack_factor, pack_shape)

            exact_specs = [s for s in specs_here if s.n_copies is not None]
            ratio_specs = [s for s in specs_here if s.n_copies is None]

            # Exact-count ("target") species placed FIRST within this
            # region -- same two-stage exact-then-exclusion-field pattern
            # the now-deleted SpherePackingSpecimenGenerator used for its
            # own target/filler split (see module docstring), now
            # region-gated instead of whole-box.
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
                _exact_pack_start = time.perf_counter()
                exact_rotations: torch.Tensor | None = None
                with status(
                    f"Packing {int(exact_radii.numel())} target instance(s) "
                    f"({location})",
                    disable=not self.progressbars,
                ):
                    if self.packing_backend == "shape":
                        (
                            coords,
                            exact_rotations,
                            accepted_idx,
                            occupancy,
                        ) = self._pack_shapes(
                            exact_pdbs,
                            exact_species_map,
                            pack_shape,
                            pack_voxel,
                            pack_factor,
                            occupancy,
                        )
                    else:
                        coords, accepted_idx = pack_hard_spheres_3d(
                            exact_radii,
                            box,
                            gap=self.gap,
                            seed=self.seed,
                            device="cpu",  # see self.device's own docstring
                            exclusion_distance_field=exclusion_field,
                            field_voxel_size=field_voxel_size,
                            sampling_mask=region_mask_field,
                            max_passes=self.region_max_passes,
                            stall_patience=region_stall_patience,
                            clip_axes=self.clip_axes,
                        )
                phase_done(
                    f"  Target packing ({location})",
                    _exact_pack_start,
                    disable=not self.progressbars,
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

                _exact_render_start = time.perf_counter()
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
                phase_done(
                    f"  Target rendering ({location})",
                    _exact_render_start,
                    disable=not self.progressbars,
                )

                # The shape backend already recorded these instances in
                # `occupancy`; the sphere backend needs the equivalent baked
                # into its distance field instead.
                if coords.numel() and self.packing_backend != "shape":
                    _rebuild_start = time.perf_counter()
                    placed_radii = exact_radii[accepted_idx]
                    # Also kept on CPU -- see exclusion_field's own comment
                    # above for why moving this to self.device would be a
                    # silent, severe per-pass performance regression, not
                    # just a style choice.
                    exact_exclusion_field = _build_sphere_exclusion_field(
                        coords, placed_radii, field_shape, field_voxel_size
                    )
                    exclusion_field = torch.minimum(
                        exclusion_field, exact_exclusion_field
                    )
                    phase_done(
                        f"  Exclusion field rebuild ({location})",
                        _rebuild_start,
                        disable=not self.progressbars,
                    )

            # Ratio-weighted ("filler") species, drawn to fill
            # occupancy_fraction of this region -- avoiding the exact-count
            # placements above (if any), the filament mask, and the
            # membrane shell, all folded into exclusion_field/region_mask
            # by this point.
            if ratio_specs:
                ratio_pdbs = [pdbs_by_source[s.pdb_source] for s in ratio_specs]
                species_radii = torch.tensor(
                    [float(pdb.max_diameter) / 2.0 for pdb in ratio_pdbs]
                )
                species_ratios = torch.tensor([s.ratio for s in ratio_specs])

                # Measured in whatever the backend collides: real footprint
                # volume for "shape", bounding-sphere volume (draw_species_
                # pool's own default) for "sphere". So `occupancy_fraction`
                # does NOT mean the same thing on both, which is a wart --
                # comparing the backends at one value compares pool sizes
                # rather than geometry, and the packing docs say so.
                #
                # Unifying it on footprint volume was tried and reverted. It
                # hands the sphere backend a ~5.6x larger pool, and that
                # backend resolves a whole pass at once through an
                # independent-set step: more candidates per pass mostly
                # conflict with each other, so a fixed max_passes budget
                # turns the larger pool into FEWER placements. Measured at
                # 10 A on the benchmark specimen, occupancy fell 0.202 ->
                # 0.183 and wall time rose 224 s -> 258 s. The shape backend,
                # which tries candidates serially, has no such behaviour.
                pool_volumes: torch.Tensor | None = None
                if self.packing_backend == "shape":
                    pool_volumes = torch.tensor(
                        [
                            float(self._species_mask(pdb, voxel_size).sum())
                            * voxel_size**3
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

                _filler_pack_start = time.perf_counter()
                filler_rotations: torch.Tensor | None = None
                with status(
                    f"Packing filler instances ({location})",
                    disable=not self.progressbars,
                ):
                    if self.packing_backend == "shape":
                        (
                            coords,
                            filler_rotations,
                            accepted_idx,
                            occupancy,
                        ) = self._pack_shapes(
                            ratio_pdbs,
                            pool_species_idx,
                            pack_shape,
                            pack_voxel,
                            pack_factor,
                            occupancy,
                        )
                    else:
                        coords, accepted_idx = pack_hard_spheres_3d(
                            pool_radii,
                            box,
                            gap=self.gap,
                            seed=self.seed,
                            device="cpu",  # see self.device's own docstring
                            exclusion_distance_field=exclusion_field,
                            field_voxel_size=field_voxel_size,
                            sampling_mask=region_mask_field,
                            max_passes=self.region_max_passes,
                            # A TIGHT region (small fraction of the box, e.g.
                            # a vesicle lumen) can need many more consecutive
                            # misses than a "box is saturated" heuristic
                            # expects before finding a geometrically valid
                            # spot (see pack_hard_spheres_3d's own
                            # sampling_mask docstring) --
                            # region_stall_patience exhausts
                            # region_max_passes there instead of bailing out
                            # early. An OPEN region (e.g. cytosol with no
                            # membrane, or with only a small organelle)
                            # instead gets pack_hard_spheres_3d's own fast
                            # default -- it saturates quickly, so patiently
                            # retrying after that just burns time for
                            # near-zero extra density (see
                            # _TIGHT_REGION_FRACTION_THRESHOLD's own comment
                            # for a benchmark).
                            stall_patience=region_stall_patience,
                            clip_axes=self.clip_axes,
                        )
                phase_done(
                    f"  Filler packing ({location})",
                    _filler_pack_start,
                    disable=not self.progressbars,
                )
                if accepted_idx.numel() == 0 and self.packing_backend == "shape":
                    warnings.warn(
                        f"TomogramSpecimenGenerator: placed 0 filler "
                        f"instances in '{location}' -- no rotated footprint "
                        f"fit anywhere in the region's "
                        f"{region_voxels:,} free voxels. Enlarge the "
                        "compartment, or declare a smaller species for "
                        "this region.",
                        stacklevel=2,
                    )
                elif accepted_idx.numel() == 0:
                    # A non-empty region that still fits nothing is silent
                    # otherwise: no picks file appears for the species and
                    # nothing says why. The naive diagnostic here used to be
                    # exclusion_field[region_mask_field].max() -- clearance
                    # from the shell alone, ignoring the box wall -- which
                    # can dramatically overstate the room available (found
                    # directly: reported ~166 A "available" for a case with
                    # ZERO truly viable positions, because everywhere far
                    # enough from the shell was also too close to the box
                    # wall for this radius). _diagnose_zero_placements
                    # applies the SAME box-containment check
                    # pack_hard_spheres_3d itself uses, so the number
                    # reported is one a caller can actually act on.
                    largest = float(species_radii.max())
                    needed = largest + self.gap
                    viable_voxels, best_clearance = _diagnose_zero_placements(
                        region_mask_field,
                        exclusion_field,
                        field_voxel_size,
                        box,
                        largest,
                        self.gap,
                        self.clip_axes,
                    )
                    if viable_voxels > 0:
                        # Geometrically possible, just unlucky within
                        # max_passes/stall_patience -- a real placement
                        # exists (rare with the "open region" stall_patience
                        # default; see _TIGHT_REGION_FRACTION_THRESHOLD).
                        warnings.warn(
                            f"TomogramSpecimenGenerator: placed 0 filler "
                            f"instances in '{location}' despite "
                            f"{viable_voxels:,} geometrically viable "
                            f"position(s) existing -- an unlucky draw, not "
                            "an impossible one. Increase region_max_passes, "
                            "or accept the occasional miss for a region "
                            "this tight.",
                            stacklevel=2,
                        )
                    else:
                        # No position exists AT ALL that both clears
                        # gap from the boundary and keeps the full
                        # sphere inside the box -- best_clearance is the
                        # most room a box-valid position ever gets, so it's
                        # always < needed here (0.0 if the region has no
                        # box-valid position regardless of clearance).
                        warnings.warn(
                            f"TomogramSpecimenGenerator: placed 0 filler "
                            f"instances in '{location}' -- no position "
                            "exists that is both far enough from the "
                            "boundary/shell AND keeps the whole sphere "
                            f"inside the box. Best available clearance at a "
                            f"box-valid position is {best_clearance:.1f} A, "
                            f"against the {needed:.1f} A this species needs "
                            f"(radius {largest:.1f} A + gap "
                            f"{self.gap:.1f} A), out of "
                            f"{region_voxels:,} region voxels total. "
                            "Enlarge the compartment, reduce gap, "
                            "or declare a smaller species for this region.",
                            stacklevel=2,
                        )
                accepted_species_idx = pool_species_idx[accepted_idx]

                _filler_render_start = time.perf_counter()
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
                    f"  Filler rendering ({location})",
                    _filler_render_start,
                    disable=not self.progressbars,
                )

            phase_done(
                f"{location.capitalize()} species",
                _location_phase_start,
                disable=not self.progressbars,
            )

        self.instance_labels = instance_labels
        return volume

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
        Place and render every `bead_specs` population -- solid gold
        spheres, no rotation needed (spherically symmetric) -- via the same
        RSA backend (`pack_hard_spheres_3d`) used for membrane instances and
        protein packing (see module docstring for why this differs from
        the deleted CTS-derived generator's own sequential bead placement).

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

        field_voxel_size, field_shape, field_factor = _resolve_exclusion_field_grid(
            target_shape, voxel_size
        )
        allowed_field = (
            _downsample_mask_maxpool(allowed, field_factor, field_shape)
            if field_factor > 1
            else allowed
        )
        exclusion_field = (
            torch.from_numpy(
                ndimage.distance_transform_edt(allowed_field.numpy())
            ).float()
            * field_voxel_size
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
        # grain, orientation and -- under radius_cv -- its own size), so
        # there is no shared template to batch over.
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
                pixel_size=voxel_size,
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

    def _one_id_per_filament(
        self, instances: list[FilamentInstance], next_instance_id: int
    ) -> tuple[torch.Tensor, int]:
        """One instance id per filament, rather than one per monomer.

        Segmentation ground truth should mark a filament as an object, the
        way it already marked a microtubule as one rather than as ~950
        loose dimers. Actin was labelled per monomer until 2026-09-01, so
        20 filaments appeared as 765 separate objects and a picker
        evaluated against them was being asked to find monomers.

        Grouped on runs of equal ``(code, filament_id)`` rather than on
        the key alone. Both placers number filaments with
        ``range(spec.n_copies)``, restarting per spec, so two specs each
        contribute a filament 0; keying on the pair alone would merge
        them. Every placer emits one filament's monomers consecutively,
        so a change of key is a filament boundary.

        That leaves one case this cannot separate: a spec contributing
        exactly one filament, immediately followed by a filament of the
        same `code` and id from the next spec. Distinguishing those needs
        the placers to number filaments globally, which is the real fix if
        it ever matters -- `filament_id` is internal, used only here and
        never written to picks.
        """
        ids: list[int] = []
        current = next_instance_id
        for run in _filament_runs(instances):
            ids.extend([current] * len(run))
            current += 1
        return torch.tensor(ids, dtype=torch.int32), current

    def _stamp_filaments(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        carbon_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Place and render every `filament_specs` species, continuing
        `instance_labels`'s own instance-id counter from wherever the
        cytosol/lumen protein loop left it.

        `place_filaments` draws positions in `[0, extent)` -- a corner-
        relative box -- while `volume`/`instance_labels` (and
        `insert_particles_into_micrograph`/`_insert_instance_labels`, both
        already used for cytosol/lumen proteins above) are centered at
        physical (0,0,0). Only the LOCAL `positions_centered` used for
        rendering is shifted by `-extent/2` to bridge that; `self.
        filament_instances` keeps each `FilamentInstance`'s original
        corner-relative `position_xyz` untouched, since that's the
        convention `export_picks` itself writes out directly.

        `place_filaments` itself has no obstacle awareness (a genuine
        collision-avoiding random walk is a bigger algorithmic change than
        this needs -- see module docstring): individual monomer instances
        that land inside `carbon_mask`, if given, are dropped here after
        the fact instead, the same "truncated at render/insert time"
        treatment already applied to monomers that wander outside the
        volume entirely (see `place_filaments`'s own docstring). A dropped
        monomer mid-path just leaves a gap in that filament, not a
        redirected walk around the film.
        """
        target_shape = self.target_shape

        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)
        instances = place_filaments(self.filament_specs, target_shape, voxel_size, rng)
        instances = self._drop_instances_in_carbon(
            instances, carbon_mask, voxel_size, "filament monomer"
        )

        self.filament_instances = instances
        if not instances:
            return volume, instance_labels, next_instance_id
        instance_ids, after = self._one_id_per_filament(instances, next_instance_id)
        return self._render_filament_instances(
            volume,
            instance_labels,
            after,
            voxel_size,
            instances,
            instance_ids=instance_ids,
        )

    def _drop_instances_in_carbon(
        self,
        instances: list[FilamentInstance],
        carbon_mask: torch.Tensor | None,
        voxel_size: float,
        what: str,
    ) -> list[FilamentInstance]:
        """Drop copies whose centre lands inside the carbon film.

        Shared by filament and microtubule stamping -- neither placer is
        obstacle-aware, so this is the same "reject after the fact" pass
        described in `_stamp_filaments`.
        """
        if carbon_mask is None or not instances:
            return instances

        nz, ny, nx = self.target_shape
        pos = torch.stack([inst.position_xyz for inst in instances])  # (N,3) x,y,z
        ix = (pos[:, 0] / voxel_size).long().clamp(0, nx - 1)
        iy = (pos[:, 1] / voxel_size).long().clamp(0, ny - 1)
        iz = (pos[:, 2] / voxel_size).long().clamp(0, nz - 1)
        in_carbon = carbon_mask.cpu()[iz, iy, ix]
        n_dropped = int(in_carbon.sum())
        if n_dropped:
            warnings.warn(
                f"TomogramSpecimenGenerator: dropped {n_dropped} {what} "
                "instance(s) that landed inside the carbon film.",
                stacklevel=2,
            )
            instances = [
                inst for inst, drop in zip(instances, in_carbon.tolist()) if not drop
            ]
        return instances

    def _stamp_microtubules(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        carbon_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Place and render every `microtubule_specs` species.

        A microtubule reaches the renderer as many rigid copies of one
        alpha-beta tubulin dimer -- the same `FilamentInstance` form
        filaments use -- so this shares `_render_filament_instances`
        wholesale. The one difference is instance labelling: every dimer of
        one tube gets the SAME instance id, so segmentation ground truth
        marks microtubules as objects rather than as ~950 loose dimers.
        """
        rng = torch.Generator()
        if self.seed is not None:
            # Offset from the filament seed so two species with the same
            # spec don't land on identical paths.
            rng.manual_seed(self.seed + 1)
        instances, tubes = place_microtubules(
            self.microtubule_specs,
            self.target_shape,
            voxel_size,
            generator=rng,
            pdb_cache_dir=self.pdb_cache_dir,
        )
        instances = self._drop_instances_in_carbon(
            instances, carbon_mask, voxel_size, "microtubule dimer"
        )

        self.microtubule_instances = tubes
        self.microtubule_dimer_instances = instances
        if not instances:
            return volume, instance_labels, next_instance_id

        # One instance id per tube, via the same helper actin uses. The
        # previous spelling grouped on `filament_id` alone, which merged
        # tube 0 of one [[microtubules]] spec with tube 0 of the next --
        # every spec resolves to the same cached dimer `code`, so nothing
        # else separated them.
        instance_ids, after = self._one_id_per_filament(instances, next_instance_id)
        return self._render_filament_instances(
            volume,
            instance_labels,
            after,
            voxel_size,
            instances,
            instance_ids=instance_ids,
            align_to_z=False,
        )

    def _microtubule_lumen_mask(
        self, voxel_size: float, device: torch.device
    ) -> torch.Tensor:
        """Voxels enclosed by a placed microtubule's wall, lumen included.

        Built by stamping a disc of the tube's own radius at every ring of
        every axis polyline. Consecutive rings are one dimer repeat apart
        (82 A) while the radius is ~111 A, so the stamped spheres overlap
        and seal the tube along its whole length without needing a real
        distance transform over the full canvas.
        """
        nz, ny, nx = self.target_shape
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

        for tube in self.microtubule_instances:
            radius_vox = tube.lattice.radius / voxel_size
            reach = int(math.ceil(radius_vox))
            for point in tube.axis_xyz:
                cx, cy, cz = (float(v) / voxel_size for v in point)
                ix0, ix1 = max(0, int(cx) - reach), min(nx, int(cx) + reach + 1)
                iy0, iy1 = max(0, int(cy) - reach), min(ny, int(cy) + reach + 1)
                iz0, iz1 = max(0, int(cz) - reach), min(nz, int(cz) + reach + 1)
                if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
                    continue
                zz, yy, xx = torch.meshgrid(
                    torch.arange(iz0, iz1, device=device, dtype=torch.float32),
                    torch.arange(iy0, iy1, device=device, dtype=torch.float32),
                    torch.arange(ix0, ix1, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                inside = (
                    (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
                ) <= radius_vox**2
                mask[iz0:iz1, iy0:iy1, ix0:ix1] |= inside
        return mask

    def _render_filament_instances(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        voxel_size: float,
        instances: list[FilamentInstance],
        instance_ids: torch.Tensor | None = None,
        align_to_z: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Render placed monomer/dimer copies: one template per species,
        rotated and inserted once per instance.

        Parameters
        ----------
        instance_ids : torch.Tensor, optional
            Per-instance segmentation ids, shape ``(len(instances),)``.
            Default None: number them sequentially from
            ``next_instance_id``, which is then advanced. Microtubule
            stamping passes explicit ids so a whole tube shares one.
        align_to_z : bool, optional
            Pre-rotate the template's longest principal axis onto ``+Z``.
            Default True, as filament monomers need. Microtubules pass
            False: their dimer template is already in the microtubule frame
            (`_tubulin.extract_mt_dimer`), where the roll about ``+Z``
            carries the radial orientation that principal-axis alignment
            has no way to know about and would be free to destroy.
        """
        extent_xyz = (
            torch.tensor(self.target_shape[::-1], dtype=torch.float32) * voxel_size
        )

        if instance_ids is None:
            ids = torch.arange(
                next_instance_id,
                next_instance_id + len(instances),
                dtype=torch.int32,
            )
            next_instance_id += len(instances)
        else:
            ids = instance_ids.to(torch.int32)

        by_code: dict[str, list[tuple[FilamentInstance, int]]] = {}
        for inst, inst_id in zip(instances, ids.tolist()):
            by_code.setdefault(inst.code, []).append((inst, inst_id))

        pdb_cache: dict[str, PDB] = {}
        templates: dict[str, torch.Tensor] = {}
        for code in by_code:
            if code not in pdb_cache:
                pdb_cache[code] = PDB(
                    code,
                    pdb_cache_dir=self.pdb_cache_dir,
                    verbose=False,
                    compute_atom_species=_wants_atom_species(self.parameterization),
                    readd_hydrogens=self.readd_hydrogens,
                    monomer_library_path=self.monomer_library_path,
                )
            pdb = pdb_cache[code]
            n = estimate_protein_box_size(pdb.max_diameter, voxel_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=voxel_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
                # Rotation-only, so species stay aligned with coordinates.
                atom_species=pdb.atom_species,
                b_factors=(pdb.b_factors if self.use_deposited_bfactors else None),
            ).to(self.device)
            coordinates = (
                align_principal_axis_to_z(pdb.coordinates)
                if align_to_z
                else pdb.coordinates
            )
            templates[code] = builder.forward(coordinates, method="analytic").to(
                self.device
            )

        offset = (extent_xyz / 2).to(self.device)
        for code, entries in by_code.items():
            template = templates[code]
            label_threshold = _INSTANCE_LABEL_REL_THRESHOLD * float(template.max())

            insts = [inst for inst, _ in entries]
            n_instances = len(insts)
            positions_centered = (
                torch.stack([inst.position_xyz for inst in insts]).to(self.device)
                - offset
            )
            R = torch.stack([inst.rotation_matrix for inst in insts]).to(self.device)
            theta = build_affine_matrix(R)

            instance_ids = torch.tensor(
                [inst_id for _, inst_id in entries],
                dtype=torch.int32,
                device=self.device,
            )

            step = self.chunk_size or n_instances
            for start in range(0, n_instances, step):
                end = min(start + step, n_instances)
                rotated = rotate_volume(
                    template, theta[start:end], padding_mode="zeros"
                )
                # Moved to the ACCUMULATOR's device (not necessarily
                # self.device) right after the compute-heavy rotation --
                # only this small per-chunk result crosses devices, never
                # the shared canvas itself (see accumulator_device's own
                # docstring).
                rotated = rotated.to(volume.device)
                volume = insert_particles_into_micrograph(
                    rotated,
                    positions_centered[start:end],
                    pixel_size=voxel_size,
                    micrograph=volume,
                )
                binarized = (rotated > label_threshold).to(torch.int32) * instance_ids[
                    start:end
                ].to(volume.device).view(-1, 1, 1, 1)
                instance_labels = _insert_instance_labels(
                    binarized,
                    positions_centered[start:end],
                    pixel_size=voxel_size,
                    labels=instance_labels,
                )

        return volume, instance_labels, next_instance_id

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
        include_transmembrane: bool = True,
        include_filaments: bool = True,
        include_microtubules: bool = True,
        include_filler: bool = True,
        include_beads: bool = True,
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style .ndjson pick file per
        placed cytosol/lumen species (grouped by `(location, species_id)`
        so the same `pdb_source` declared at both locations never collides
        in one file) plus, by default, one per transmembrane species --
        one JSON object per line: ``{"type": "point"|"orientedPoint",
        "location": {"x", "y", "z"}[, "xyz_rotation_matrix"]}``.

        `TomogramPlacement.role == "filler"` placements (species declared
        via `ratio`, not `n_copies`) are INCLUDED by default here --
        `protein_specs` predates the exact-count/ratio split (every
        declared cytosol/lumen species used to be exported
        unconditionally), so defaulting to True preserves that behavior
        for existing `ratio`-only configs. Pass
        `include_filler=False` to export only `n_copies`-declared species
        once you're using the new distinction deliberately. A
        `(species_id, location)` pair placed as BOTH a target and filler
        (declared twice, once with `n_copies` and once with just `ratio`)
        keeps its filler instances in a separate ``-filler``-suffixed file,
        never merged with the target file.

        Transmembrane picks are oriented (a real `rotation_matrix`, unlike
        other membrane picks here, which are plain points) since
        `TransmembranePlacement` actually carries one.

        Coordinates are converted from this generator's box-centered
        convention (`position_xyz`/`center_xyz`, origin at the volume's
        center, matching `MembraneGenerator`'s own convention) to the
        corner-relative (``0..extent``) convention copick/the portal
        actually use -- the same conversion the other two generators'
        `export_picks` perform.

        Must be called after `generate()`.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory to write the .ndjson files into.
        annotation_version : str, optional
            Used only in the output filename
            (``"{name}-{version}_{type}.ndjson"``). Default "1.0".
        oriented : bool, optional
            If True (default), picks are written as ``"orientedPoint"``
            with each instance's rotation matrix included; if False, as
            plain ``"point"`` (location only).
        include_transmembrane : bool, optional
            If True (default), also write pick file(s) for transmembrane
            species, suffixed ``-transmembrane``.
        include_filaments : bool, optional
            If True (default), also write one pick file per filament
            species, suffixed ``-filament``. Each `FilamentInstance`'s own
            `position_xyz` is already in the corner-relative convention
            used here (see `_stamp_filaments`), so -- unlike
            placements/transmembrane above -- it's written directly, with
            no `+ extent_xyz / 2` conversion.
        include_microtubules : bool, optional
            If True (default), also write one pick file per microtubule
            species, suffixed ``-microtubule``: one entry per TUBE, whose
            ``path`` is the axis polyline, not one entry per dimer. A tube
            is a ~950-dimer object, and a pick file listing every dimer is
            rarely what a consumer wants; the per-dimer copies remain in
            `microtubule_dimer_instances` for anyone who does.
        include_filler : bool, optional
            If True, also write pick files for `role == "filler"`
            cytosol/lumen placements (suffixed ``-filler`` on a
            target/filler `(species_id, location)` collision, to avoid
            overwriting the target's own file). Default False.
        include_beads : bool, optional
            If True (default), also write every gold fiducial to a single
            ``gold-bead`` pick file, regardless of radius or which
            `bead_specs` population it came from -- nothing downstream
            distinguishes bead sizes. Always written as plain ``"point"``
            regardless of `oriented`: a bead has no meaningful
            per-instance orientation for picking purposes.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of a grouping key (``"{species}-{location}"`` for
            cytosol/lumen instances, ``"{species}-transmembrane"`` for
            transmembrane instances, ``"{species}-filament"`` for filament
            instances) to written file path.
        """
        if self.instance_labels is None:
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        target_shape = self.target_shape
        voxel_size = self.voxel_size
        extent_xyz = (
            torch.tensor(
                [target_shape[2], target_shape[1], target_shape[0]],
                dtype=torch.float32,
            )
            * voxel_size
        )
        point_type = "orientedPoint" if oriented else "point"

        target_keys = {
            (placed.species_id, placed.location)
            for placed in self.placements
            if placed.role == "target"
        }
        by_key: dict[str, list[TomogramPlacement]] = {}
        for placed in self.placements:
            if placed.role == "filler" and not include_filler:
                continue
            name = Path(placed.species_id).stem
            key = f"{name}-{placed.location}"
            if (
                placed.role == "filler"
                and (placed.species_id, placed.location) in target_keys
            ):
                key = f"{key}-filler"
            by_key.setdefault(key, []).append(placed)
        for key, placed_list in by_key.items():
            path = (
                output_dir / f"{key}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for placed in placed_list:
                    corner_xyz = placed.position_xyz + extent_xyz / 2
                    x, y, z = (float(v) for v in corner_xyz)
                    row: dict = {
                        "type": point_type,
                        "location": {"x": x, "y": y, "z": z},
                    }
                    if oriented:
                        row["xyz_rotation_matrix"] = (
                            placed.rotation_matrix.numpy().tolist()
                        )
                    f.write(json.dumps(row) + "\n")
            written[key] = path

        if include_transmembrane and self.transmembrane_placements:
            by_species: dict[str, list[TransmembranePlacement]] = {}
            for tp in self.transmembrane_placements:
                by_species.setdefault(Path(tp.species_id).stem, []).append(tp)
            for species, tps in by_species.items():
                key = f"{species}-transmembrane"
                path = (
                    output_dir
                    / f"{key}-{annotation_version}_{point_type.lower()}.ndjson"
                )
                with open(path, "w") as f:
                    for tp in tps:
                        corner_xyz = tp.center_xyz + extent_xyz / 2
                        x, y, z = (float(v) for v in corner_xyz)
                        row = {"type": point_type, "location": {"x": x, "y": y, "z": z}}
                        if oriented:
                            row["xyz_rotation_matrix"] = (
                                tp.rotation_matrix.numpy().tolist()
                            )
                        f.write(json.dumps(row) + "\n")
                written[key] = path

        if include_filaments and self.filament_instances:
            by_filament_code: dict[str, list[FilamentInstance]] = {}
            for inst in self.filament_instances:
                by_filament_code.setdefault(inst.code, []).append(inst)
            for code, insts in by_filament_code.items():
                key = f"{Path(code).stem}-filament"
                path = (
                    output_dir
                    / f"{key}-{annotation_version}_{point_type.lower()}.ndjson"
                )
                with open(path, "w") as f:
                    for inst in insts:
                        x, y, z = (float(v) for v in inst.position_xyz)
                        row = {"type": point_type, "location": {"x": x, "y": y, "z": z}}
                        if oriented:
                            row["xyz_rotation_matrix"] = (
                                inst.rotation_matrix.numpy().tolist()
                            )
                        f.write(json.dumps(row) + "\n")
                written[key] = path

                # One `path` per filament as well, so the picks agree with
                # the label volume about what an object is: both now say a
                # filament, where the labels said one object and these
                # points said several dozen.
                #
                # Written in ADDITION to the oriented points rather than
                # instead of them. A path carries no orientations, and the
                # per-monomer rotation matrices above are what subtomogram
                # averaging of F-actin needs; nothing in the volume can
                # recover them. Microtubules ship only a path and so have
                # no per-dimer poses at all.
                path_file = output_dir / f"{key}-{annotation_version}_path.ndjson"
                with open(path_file, "w") as f:
                    for run in _filament_runs(insts):
                        points = torch.stack([i.position_xyz for i in run])
                        centre = points.mean(dim=0)
                        f.write(
                            json.dumps(
                                {
                                    "type": "path",
                                    "location": {
                                        "x": float(centre[0]),
                                        "y": float(centre[1]),
                                        "z": float(centre[2]),
                                    },
                                    "path": [
                                        {
                                            "x": float(p[0]),
                                            "y": float(p[1]),
                                            "z": float(p[2]),
                                        }
                                        for p in points
                                    ],
                                    "n_monomers": len(run),
                                }
                            )
                            + "\n"
                        )
                written[f"{key}-path"] = path_file

        if include_microtubules and self.microtubule_instances:
            by_tube_code: dict[str, list[MicrotubuleInstance]] = {}
            for tube in self.microtubule_instances:
                by_tube_code.setdefault(tube.code, []).append(tube)
            for code, tubes in by_tube_code.items():
                key = f"{Path(code).stem}-microtubule"
                path = output_dir / f"{key}-{annotation_version}_path.ndjson"
                with open(path, "w") as f:
                    for tube in tubes:
                        axis = tube.axis_xyz
                        centre = axis.mean(dim=0)
                        f.write(
                            json.dumps(
                                {
                                    "type": "path",
                                    "location": {
                                        "x": float(centre[0]),
                                        "y": float(centre[1]),
                                        "z": float(centre[2]),
                                    },
                                    "path": [
                                        {
                                            "x": float(p[0]),
                                            "y": float(p[1]),
                                            "z": float(p[2]),
                                        }
                                        for p in axis
                                    ],
                                    "radius": tube.lattice.radius,
                                    "n_protofilaments": (tube.lattice.n_protofilaments),
                                }
                            )
                            + "\n"
                        )
                written[key] = path

        if include_beads and self.bead_instances:
            # Every fiducial goes in one file regardless of radius or
            # population: nothing downstream distinguishes bead sizes, and
            # under a [low, high] radius each instance has a unique size,
            # so grouping by radius would write one file per bead.
            key = "gold-bead"
            path = output_dir / f"{key}-{annotation_version}_point.ndjson"
            with open(path, "w") as f:
                for bead in self.bead_instances:
                    corner_xyz = bead.position_xyz + extent_xyz / 2
                    x, y, z = (float(v) for v in corner_xyz)
                    f.write(
                        json.dumps(
                            {"type": "point", "location": {"x": x, "y": y, "z": z}}
                        )
                        + "\n"
                    )
            written[key] = path

        return written

    def _packing_grid(
        self, target_shape: tuple[int, int, int], voxel_size: float
    ) -> tuple[float, tuple[int, int, int], int]:
        """Coarse collision grid for `packing_voxel_size`: (voxel, shape, factor)."""
        if self.packing_voxel_size is None:
            n = target_shape[0] * target_shape[1] * target_shape[2]
            if n <= _MAX_PACKING_GRID_VOXELS:
                return voxel_size, target_shape, 1
            factor = max(1, math.ceil((n / _MAX_PACKING_GRID_VOXELS) ** (1.0 / 3.0)))
        elif self.packing_voxel_size <= voxel_size:
            return voxel_size, target_shape, 1
        else:
            factor = max(1, int(round(self.packing_voxel_size / voxel_size)))
        shape = tuple(-(-n // factor) for n in target_shape)
        return voxel_size * factor, shape, factor  # type: ignore[return-value]

    def _species_mask(self, pdb: PDB, voxel_size: float) -> torch.Tensor:
        """
        This species' footprint mask for ``packing_backend="shape"``, built
        once per (structure, voxel size, gap) and reused across regions and
        across the pool-sizing/packing steps.

        Always at the RENDER voxel size. Coarsening for a coarser packing
        grid happens per rotated orientation inside the packer, not here --
        see `packing_voxel_size`.
        """
        key = (id(pdb), voxel_size, self.gap)
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
        a running occupancy grid), the `packing_backend="shape"` path.

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

    def _build_species_template(
        self, pdb: PDB, voxel_size: float, device: torch.device
    ) -> torch.Tensor:
        n = estimate_protein_box_size(pdb.max_diameter, voxel_size)
        builder = PotentialBuilder(
            n_xyz=n,
            dx=voxel_size,
            atomic_numbers=pdb.atomic_numbers,
            progressbars=False,
            parameterization=self.parameterization,
            atom_species=pdb.atom_species,
            b_factors=pdb.b_factors if self.use_deposited_bfactors else None,
        ).to(device)
        return builder.forward(pdb.coordinates, method="analytic").to(self.device)

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
        # then run the rotate/insert loop below exactly as before. That loop
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
                label_threshold = _INSTANCE_LABEL_REL_THRESHOLD * float(template.max())

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

                step = self.chunk_size or n_instances
                for start in range(0, n_instances, step):
                    end = min(start + step, n_instances)
                    rotated = rotate_volume(
                        template, theta[start:end], padding_mode="zeros"
                    )
                    # See _stamp_filaments' own identical comment: only
                    # this small per-chunk result crosses devices, never
                    # the shared canvas.
                    rotated = rotated.to(volume.device)
                    volume = insert_particles_into_micrograph(
                        rotated,
                        species_coords[start:end],
                        pixel_size=voxel_size,
                        micrograph=volume,
                    )
                    binarized = (rotated > label_threshold).to(
                        torch.int32
                    ) * instance_ids[start:end].to(volume.device).view(-1, 1, 1, 1)
                    instance_labels = _insert_instance_labels(
                        binarized,
                        species_coords[start:end],
                        pixel_size=voxel_size,
                        labels=instance_labels,
                    )

                for i in range(n_instances):
                    self.placements.append(
                        TomogramPlacement(
                            species_id=spec.pdb_source,
                            location=location,
                            position_xyz=species_coords[i].detach().cpu(),
                            rotation_matrix=R[i].detach().cpu(),
                            instance_id=int(instance_ids[i]),
                            role=role,
                        )
                    )
                progress.update(render_task, advance=1)

        return volume, instance_labels, next_instance_id
