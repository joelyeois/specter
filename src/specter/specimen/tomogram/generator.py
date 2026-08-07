"""
MembraneTomogramGenerator: assembles a full specimen tomogram around an
organic membrane -- transmembrane proteins on the bilayer, plus densely
packed cytosolic and vesicle-lumen protein populations, region-gated
against the membrane's own geometry so a "lumen" species can only land
inside an enclosed compartment and a "cytosol" species can only land
outside one.

Composes three independently-developed pieces rather than reimplementing
any of them: ``specimen.membrane.MembraneGenerator`` (organic shape +
transmembrane placement, unmodified), :func:`.classify_membrane_regions`
(shell/lumen/cytosol masks via connected-components flood-fill, this
subpackage), and ``specimen.packing.pack_hard_spheres_3d``'s
`exclusion_distance_field` (obstacle- and region-aware RSA, this session).
Deliberately uses the RSA backend, not `pack_hard_spheres_3d_dense`: a
production-scale tomogram (hundreds of voxels per axis) draws candidate
pools far too large for the force-biased backend's per-iteration Python-
loop cost to stay practical (verified to run into the hours at that scale)
-- RSA's own ~28-41% ceiling, reached in seconds, is the actual target
here, not `pack_hard_spheres_3d_dense`'s higher-but-impractical one.

Independent of, and not integrated with, the older CTS-derived generators
(``cryotomosim.py``, ``_cts_membrane.py``, ``_cts_placement.py``) -- this is
a clean-room second approach, meant to be benchmarked against CTS later,
not merged with it now.

Supports MULTIPLE independently-configured membrane instances
(:class:`MembraneInstance`, each with its own `MembraneGenerator` --
potentially a different `shape_backend` per instance -- and physical
`position_xyz` offset) composited into one shared tomogram volume:
generate each instance in its own centered local frame (unmodified
`MembraneGenerator`, no changes needed there), max-merge the resulting
density volumes into the shared canvas, then classify shell/lumen/cytosol
regions ONCE on the composite -- `classify_membrane_regions`'s connected-
components approach already handles multiple disjoint compartments (several
separate vesicles, whether from one instance or several) without any
special-casing.

Each instance renders on its OWN local working grid -- typically much
smaller than the shared canvas (`MembraneGenerator`'s own `target_shape_zyx`
auto-sizes from the organelle's size when omitted; see that class's own
docstring) -- then `clip_insert_bounds`-based compositing crops/places it
into the shared canvas at `position_xyz`, the same mechanism already used
for transmembrane protein templates. This was always how the compositing
math worked (`_insert_volume_max`/`_insert_shell_label` never required a
same-shape `local`); it just wasn't exercised until `MembraneGenerator`
gained auto-sizing, since giving every instance a hand-picked box the size
of the whole tomogram was the only practical option before that.

`position_xyz` is optional per instance: when omitted, `generate()` resolves
it via `pack_hard_spheres_3d` (the same RSA backend used for cytosol/lumen
protein packing below), treating each instance as a bounding sphere -- an
instance that doesn't fit without colliding (with another auto-placed
instance; NOT currently checked against explicitly-`position_xyz`'d ones,
a known v1 gap) is dropped rather than retried at a new position, matching
this module's "reject and move on" philosophy elsewhere. An instance whose
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
through the tomogram via ``specimen.filament.place_filaments`` -- the same
specter-native random-walk placement `CryoETSpecimenGenerator` uses, with
no region-gating (filaments are dropped anywhere in the volume regardless
of cytosol/lumen/membrane-shell classification, matching that generator's
own "no occupancy-aware packing against other species" scope) and no
collision avoidance against the membrane or the cytosol/lumen protein
pools. Rendered last, after cytosol/lumen packing, and stamped as
additional entries in the shared `instance_labels` volume (continuing the
same instance-id counter) so filament monomers are visible in the
segmentation ground truth alongside cytosol/lumen protein instances.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from scipy import ndimage

from ...arrays import clip_insert_bounds
from ...crowding import insert_particles_into_micrograph
from ...pdb import DEFAULT_PDB_SAVEFOLDER, PDB
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, random_rotation_matrix, rotate_volume
from .._parallel_render import (
    build_pdb_cache_concurrently,
    build_templates_concurrently,
    resolve_render_devices,
    resolve_render_workers,
)
from ..filament import FilamentInstance, FilamentSpec, place_filaments
from ..membrane import MembraneGenerator, TransmembranePlacement
from ..membrane._placement import align_principal_axis_to_z
from ..packing import draw_species_pool, estimate_protein_box_size, pack_hard_spheres_3d
from ._regions import classify_membrane_regions

# Same convention as specimen/packing/pdb_packing.py's own
# _INSTANCE_LABEL_REL_THRESHOLD -- kept as an independent copy rather than
# imported, matching this codebase's established per-generator
# zero-cross-coupling convention (see e.g. cryotomosim.py's own docstring).
_INSTANCE_LABEL_REL_THRESHOLD = 0.01


def _insert_instance_labels(
    binarized: torch.Tensor,
    positions: torch.Tensor,
    pixel_size: float,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Stamp per-instance integer labels into a shared label volume --
    identical mechanism to specimen/packing/pdb_packing.py's own helper of
    the same name, duplicated rather than imported (see module docstring)."""
    N, Zp, Yp, Xp = binarized.shape
    device = binarized.device
    Z, Y, X = labels.shape
    positions = positions.to(device)
    positions_int = (positions / pixel_size).round().long()
    cz_center, cy_center, cx_center = Z // 2, Y // 2, X // 2

    for i in range(N):
        cx_index = cx_center + int(positions_int[i, 0].item())
        cy_index = cy_center + int(positions_int[i, 1].item())
        cz_index = cz_center + int(positions_int[i, 2].item())
        bounds = clip_insert_bounds(
            (cz_index, cy_index, cx_index), (Zp, Yp, Xp), (Z, Y, X)
        )
        if bounds is None:
            continue
        dst, src = bounds
        chunk = binarized[i][src]
        labels[dst] = torch.where(chunk > 0, chunk, labels[dst])

    return labels


def _position_to_center_index(
    position_xyz: tuple[float, float, float],
    shape_zyx: tuple[int, ...],
    v_size: float,
) -> tuple[int, int, int]:
    """Physical (x, y, z) offset from a volume's own center -> absolute
    (z, y, x) voxel index of that offset -- the center-relative convention
    `MembraneGenerator` itself uses (physical (0,0,0) = volume center),
    matching `_insert_instance_labels`'s own indexing math above. NOT
    `cryoet.py`'s `_insert_clipped`, which uses a corner-relative (0..extent)
    convention for genuinely small local arrays -- the wrong frame for a
    MembraneGenerator instance, which always renders on its own full target
    grid centered at (0,0,0) (see that class's own module docstring)."""
    z_center, y_center, x_center = (
        shape_zyx[0] // 2,
        shape_zyx[1] // 2,
        shape_zyx[2] // 2,
    )
    px, py, pz = position_xyz
    return (
        z_center + int(round(pz / v_size)),
        y_center + int(round(py / v_size)),
        x_center + int(round(px / v_size)),
    )


def _insert_volume_max(
    volume: torch.Tensor,
    local: torch.Tensor,
    position_xyz: tuple[float, float, float],
    v_size: float,
) -> torch.Tensor:
    """Max-merge `local` (same center-relative convention as `volume`,
    i.e. physical (0,0,0) at its own center) into `volume`, shifted by
    `position_xyz`. See `_position_to_center_index` for why this uses a
    different convention than `cryoet.py`'s `_insert_clipped`."""
    center_zyx = _position_to_center_index(position_xyz, tuple(volume.shape), v_size)
    bounds = clip_insert_bounds(center_zyx, local.shape, volume.shape)
    if bounds is None:
        return volume
    dst, src = bounds
    volume[dst] = torch.maximum(volume[dst], local[src])
    return volume


def _instance_bounding_radius(generator: MembraneGenerator) -> float:
    """Conservative bounding-sphere radius for a membrane instance's own
    (already-resolved, see MembraneGenerator.__init__) size, used only for
    collision-rejecting random placement -- deliberately generous rather
    than exact (this module's own "not caring about packing tightly"
    philosophy, see its docstring): for `swept_spline`, a wandering path's
    true bounding box is smaller than its contour length (same reasoning
    MembraneGenerator's own auto-sizing uses), so treating the FULL
    contour length as if straight overestimates, not underestimates."""
    if generator.shape_backend == "spherical_harmonics":
        return max(generator.sh_axes_a)
    if generator.shape_backend == "swept_spline":
        return 0.5 * generator.swept_total_length_a + generator.swept_tube_radius_a
    # metaball/alpha_shape (deprecated): neither has one clean "size"
    # parameter the way the two supported backends do, and target_shape_zyx
    # is required (not auto-sized) for them -- fall back to half that box's
    # own smallest extent as a conservative stand-in.
    tz, ty, tx = generator.target_shape_zyx
    return 0.5 * min(tz, ty, tx) * generator.v_size


def _insert_shell_label(
    labels: torch.Tensor,
    shell_mask: torch.Tensor,
    instance_id: int,
    position_xyz: tuple[float, float, float],
    v_size: float,
) -> tuple[torch.Tensor, bool]:
    """Stamp `instance_id` into `labels` wherever `shell_mask` (same
    center-relative convention, see `_insert_volume_max`) is True, shifted
    by `position_xyz` -- FIRST-write-wins: a voxel already claimed by an
    earlier instance is never overwritten (unlike `_insert_instance_labels`
    above, which is last-write-wins -- harmless there since placed protein
    instances never spatially overlap by construction, but membrane
    instances in v1 have no automatic overlap avoidance, so which instance
    "wins" a genuine overlap must be a deliberate, deterministic choice).

    Returns
    -------
    (torch.Tensor, bool)
        The updated `labels`, and whether this instance's shell overlapped
        any voxel an earlier instance had already claimed.
    """
    center_zyx = _position_to_center_index(position_xyz, tuple(labels.shape), v_size)
    bounds = clip_insert_bounds(center_zyx, shell_mask.shape, labels.shape)
    if bounds is None:
        return labels, False
    dst, src = bounds
    chunk = shell_mask[src].to(labels.dtype) * instance_id
    overlap = bool(((chunk > 0) & (labels[dst] > 0)).any())
    labels[dst] = torch.where(labels[dst] > 0, labels[dst], chunk)
    return labels, overlap


@dataclass
class TomogramProteinSpec:
    """
    One cytosolic- or lumen-dwelling protein species to densely pack.

    Attributes
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF file path.
    location : {"cytosol", "lumen"}, optional
        Which region this species is restricted to -- "cytosol" (outside
        every membrane compartment) or "lumen" (inside an enclosed
        compartment, e.g. a vesicle's interior). Default "cytosol".
    ratio : float, optional
        Relative abundance weight among OTHER specs sharing the same
        `location` (species at different locations are packed
        independently, so ratios don't compare across locations). Default
        1.0.
    """

    pdb_source: str
    location: Literal["cytosol", "lumen"] = "cytosol"
    ratio: float = 1.0


@dataclass
class TomogramPlacement:
    """One placed cytosolic/lumen instance, for ground-truth bookkeeping."""

    species_id: str
    location: str
    position_xyz: torch.Tensor
    rotation_matrix: torch.Tensor
    instance_id: int


@dataclass
class MembraneInstance:
    """
    One membrane, already configured, to composite into a shared tomogram
    alongside others.

    Attributes
    ----------
    generator : MembraneGenerator
        Already-configured (not yet `.generate()`-called) membrane
        generator -- any `shape_backend`, independent per instance. Always
        centered at physical (0,0,0) (`MembraneGenerator`'s own
        convention), then shifted into place via `position_xyz` at
        composite time -- its own `target_shape_zyx` need NOT match the
        owning `MembraneTomogramGenerator`'s (typically shouldn't: leave it
        `None` for a small, auto-sized local grid, see `MembraneGenerator`'s
        own docstring).
    position_xyz : tuple of float, optional
        Physical (x, y, z) offset from the shared tomogram's own center,
        Angstrom. Default `None`: resolved automatically by `generate()`
        via collision-rejecting random placement (see this module's own
        docstring) -- an instance that doesn't fit gets dropped, and
        `position_xyz` is set here (mutated in place) for whichever
        instances are accepted, so it's inspectable afterward. Pass an
        explicit value to place (and exclude from collision-avoidance
        against other auto-placed instances -- a known v1 gap, see module
        docstring) a specific instance manually instead.
    """

    generator: MembraneGenerator
    position_xyz: tuple[float, float, float] | None = None


class MembraneTomogramGenerator:
    """
    Assemble a tomogram from a pre-configured membrane plus densely packed
    cytosolic and vesicle-lumen protein populations.

    Parameters
    ----------
    membrane_instances : list of MembraneInstance
        One or more membranes to composite into the shared tomogram (see
        `MembraneInstance`) -- each instance's own shape/transmembrane_specs
        are used as-is and not duplicated here. Non-empty; every instance's
        own `generator.v_size` must match `v_size` below (raises
        `ValueError` naming the offending index otherwise).
    target_shape_zyx : tuple of int
        Shared tomogram canvas shape, `(Z, Y, X)` voxels -- every instance
        composites into this same grid.
    v_size : float
        Shared voxel size, Angstrom -- must match every instance's own
        `generator.v_size`.
    protein_specs : list of TomogramProteinSpec
        Cytosolic/lumen species to pack.
    filament_specs : list of FilamentSpec, optional
        Filament species (e.g. `specimen.filament.ACTIN_SPEC`/
        `MICROTUBULE_SPEC`) to scatter through the tomogram via
        `specimen.filament.place_filaments` -- specter-native random-walk
        placement, independent of membrane/protein packing (no region-
        gating, no collision avoidance against anything else in the
        tomogram; see module docstring). Default None (no filaments).
    occupancy_fraction : float, optional
        Target packing density (see `pack_hard_spheres_3d`/
        `draw_species_pool`), applied independently per region -- e.g. 0.2
        for `"lumen"` species targets 20% of the LUMEN's own volume, not
        20% of the whole tomogram. Default 0.2.
    gap_angstrom : float, optional
        Minimum clearance between placed spheres' surfaces, between a
        placed sphere and the membrane shell, AND between auto-placed
        membrane instances' own bounding spheres (all three reuse this one
        value). Default 5.0.
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
    min_transmembrane_spacing_a : float, optional
        Passed to `MembraneGenerator.place_transmembrane`. Default 40.0.
    pdb_cache_dir : str, optional
        Directory for downloaded PDB/mmCIF files. Default is
        `specter.pdb.DEFAULT_PDB_SAVEFOLDER` (the repo's own `pdb-data/`,
        anchored to the package location, not the caller's cwd).
    parameterization : str, optional
        Atomic scattering-factor parameterization for `PotentialBuilder`.
        Default "shtyrov", matching `PotentialBuilder`'s own default.
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
        `target_shape_zyx`, dtype int32. Set after `generate()` runs.
    transmembrane_placements : list of TransmembranePlacement
        From every instance's own `MembraneGenerator.place_transmembrane`,
        with `center_xyz` offset into shared-tomogram coordinates by that
        instance's `position_xyz`. Set after `generate()` runs.
    placements : list of TomogramPlacement
        Every placed cytosolic/lumen instance, set after `generate()` runs.
    instance_labels : torch.Tensor
        Per-instance integer label volume for cytosol/lumen PROTEIN
        instances AND, when `filament_specs` is non-empty, filament monomer
        instances too (a continuation of the same instance-id counter, not
        a separate label space -- see module docstring), shape
        `target_shape_zyx`, dtype int32. Set after `generate()` runs.
    filament_instances : list of FilamentInstance
        Every placed filament monomer, from `specimen.filament.
        place_filaments` (`position_xyz` in the same corner-relative,
        `[0, extent)` convention `export_picks` writes directly -- NOT the
        center-relative convention `instance_labels`/`volume` use
        internally, see `_stamp_filaments`). Set after `generate()` runs
        (empty if `filament_specs` was empty/None).
    """

    def __init__(
        self,
        membrane_instances: list[MembraneInstance],
        target_shape_zyx: tuple[int, int, int],
        v_size: float,
        protein_specs: list[TomogramProteinSpec],
        filament_specs: list[FilamentSpec] | None = None,
        occupancy_fraction: float = 0.2,
        gap_angstrom: float = 5.0,
        region_density_threshold: float | None = None,
        region_max_passes: int = 300,
        min_transmembrane_spacing_a: float = 40.0,
        pdb_cache_dir: str = DEFAULT_PDB_SAVEFOLDER,
        parameterization: str = "shtyrov",
        seed: int | None = None,
        device: str | torch.device = "cpu",
        chunk_size: int | None = None,
        render_workers: int | Literal["auto"] = 1,
        render_devices: list[str | torch.device] | None = None,
    ):
        if not protein_specs:
            raise ValueError("protein_specs must be non-empty")
        if not membrane_instances:
            raise ValueError("membrane_instances must be non-empty")
        for i, mi in enumerate(membrane_instances):
            if mi.generator.v_size != v_size:
                raise ValueError(
                    f"MembraneTomogramGenerator: membrane_instances[{i}]'s own "
                    f"v_size ({mi.generator.v_size}) does not match the shared "
                    f"v_size ({v_size}) -- every instance must render on the "
                    "same voxel grid to be compositable."
                )
        self.membrane_instances = membrane_instances
        self.target_shape_zyx = target_shape_zyx
        self.v_size = v_size
        self.protein_specs = protein_specs
        self.filament_specs = filament_specs or []
        self.occupancy_fraction = occupancy_fraction
        self.gap_angstrom = gap_angstrom
        self.region_density_threshold = region_density_threshold
        self.region_max_passes = region_max_passes
        self.min_transmembrane_spacing_a = min_transmembrane_spacing_a
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
        self.seed = seed
        self.device = device
        self.chunk_size = chunk_size
        self.render_workers = resolve_render_workers(render_workers, len(protein_specs))
        self.render_devices = resolve_render_devices(device, render_devices)

        self.regions: dict[str, torch.Tensor] | None = None
        self.membrane_labels: torch.Tensor | None = None
        self.transmembrane_placements: list[TransmembranePlacement] = []
        self.placements: list[TomogramPlacement] = []
        self.instance_labels: torch.Tensor | None = None
        self.filament_instances: list[FilamentInstance] = []

    def generate(self) -> torch.Tensor:
        """
        Run the full pipeline and return the assembled specimen volume.

        Returns
        -------
        torch.Tensor
            Shape `target_shape_zyx`, dtype float32.
        """
        if self.seed is not None:
            torch.manual_seed(
                self.seed
            )  # random_rotation_matrix has no generator= param

        v_size = self.v_size
        target_shape = self.target_shape_zyx
        box = (
            target_shape[0] * v_size,
            target_shape[1] * v_size,
            target_shape[2] * v_size,
        )

        # Resolve any omitted position_xyz via collision-rejecting random
        # placement, treating each instance as a bounding sphere (see
        # _instance_bounding_radius) -- an instance that doesn't fit
        # without colliding is dropped (never .generate()-called at all,
        # cheaper than generating first and rejecting after), matching
        # this module's own "reject and move on" packing philosophy rather
        # than retrying at new positions. Instances with an explicit
        # position_xyz are placed as given and NOT included in this
        # collision check (a known v1 gap -- see module docstring).
        auto_instances = [
            mi for mi in self.membrane_instances if mi.position_xyz is None
        ]
        to_composite = [
            mi for mi in self.membrane_instances if mi.position_xyz is not None
        ]
        if auto_instances:
            radii = torch.tensor(
                [_instance_bounding_radius(mi.generator) for mi in auto_instances]
            )
            coords, accepted_idx = pack_hard_spheres_3d(
                radii,
                box,
                gap=self.gap_angstrom,
                seed=self.seed,
                device=self.device,
            )
            n_dropped = len(auto_instances) - accepted_idx.numel()
            if n_dropped:
                warnings.warn(
                    f"MembraneTomogramGenerator: {n_dropped}/{len(auto_instances)} "
                    "membrane instances with automatic (position_xyz=None) "
                    "placement did not fit without colliding and were dropped "
                    "(never generated).",
                    stacklevel=2,
                )
            for k, orig_idx in enumerate(accepted_idx.tolist()):
                mi = auto_instances[orig_idx]
                mi.position_xyz = tuple(coords[k].tolist())
                to_composite.append(mi)

        # Generate + place transmembrane proteins per instance, each in its
        # own centered local frame, then composite densities into the
        # shared canvas (max-merge) before any region classification --
        # classify_membrane_regions needs the full composite, not
        # per-instance pieces.
        volume = torch.zeros(target_shape, dtype=torch.float32, device=self.device)
        self.transmembrane_placements = []
        instance_volumes: list[tuple[MembraneInstance, torch.Tensor]] = []
        for mi in to_composite:
            # Every mi here has a concrete position_xyz by construction:
            # to_composite starts as the explicitly-positioned instances,
            # then only accepted (position_xyz just set above) auto_instances
            # are appended to it.
            assert mi.position_xyz is not None
            mi.generator.generate()
            if mi.generator.clipped_at_boundary:
                warnings.warn(
                    "MembraneTomogramGenerator: a membrane instance's own "
                    "working grid was too small for the organelle size it "
                    "actually drew (clipped_at_boundary=True on its "
                    "MembraneGenerator) -- skipped rather than compositing a "
                    "visibly truncated shape. Increase that instance's own "
                    "target_shape_zyx/v_size, or omit target_shape_zyx "
                    "entirely for auto-sizing.",
                    stacklevel=2,
                )
                continue
            tm_placements = mi.generator.place_transmembrane(
                min_spacing_a=self.min_transmembrane_spacing_a
            )
            offset = torch.tensor(mi.position_xyz, dtype=torch.float32)
            for tp in tm_placements:
                tp.center_xyz = tp.center_xyz + offset
            self.transmembrane_placements.extend(tm_placements)

            local_volume = mi.generator.volume
            assert local_volume is not None
            instance_volumes.append((mi, local_volume))
            volume = _insert_volume_max(volume, local_volume, mi.position_xyz, v_size)

        # One scalar threshold, shared by the global region classification
        # AND every instance's own shell mask below -- computed from the
        # COMPOSITE (classify_membrane_regions' own default depends on the
        # volume it's given, so it can only be resolved after compositing),
        # then reused verbatim so per-instance shell IDs and the global
        # shell/lumen/cytosol masks agree on every voxel by construction.
        threshold = self.region_density_threshold
        if threshold is None:
            peak = float(volume.max())
            threshold = 0.05 * peak if peak > 0 else 0.0
        self.regions = classify_membrane_regions(volume, threshold)

        membrane_labels = torch.zeros(
            target_shape, dtype=torch.int32, device=self.device
        )
        for instance_id, (mi, local_volume) in enumerate(instance_volumes, start=1):
            assert mi.position_xyz is not None  # see identical assert above
            shell_mask = local_volume > threshold
            membrane_labels, overlap = _insert_shell_label(
                membrane_labels, shell_mask, instance_id, mi.position_xyz, v_size
            )
            if overlap:
                warnings.warn(
                    f"MembraneTomogramGenerator: membrane instance {instance_id} "
                    "(1-indexed, in membrane_instances order) overlaps a voxel "
                    "already claimed by an earlier instance in membrane_labels "
                    "-- the earlier instance's label wins there (first-write-"
                    "wins). Adjust position_xyz if this overlap wasn't intended.",
                    stacklevel=2,
                )
        self.membrane_labels = membrane_labels

        instance_labels = torch.zeros(
            target_shape, dtype=torch.int32, device=self.device
        )
        self.placements = []
        next_instance_id = 1
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
            pdb_cache = build_pdb_cache_concurrently(
                pdb_sources=unique_sources,
                pdb_cache_dir=self.pdb_cache_dir,
                max_workers=self.render_workers,
            )

        for location in ("cytosol", "lumen"):
            specs_here = [s for s in self.protein_specs if s.location == location]
            if not specs_here:
                continue

            region_mask = self.regions[location]
            region_voxels = int(region_mask.sum())
            if region_voxels == 0:
                warnings.warn(
                    f"MembraneTomogramGenerator: no '{location}' region found "
                    f"(0 voxels) -- {len(specs_here)} species declared for it "
                    "will not be placed. For 'lumen', this means the membrane "
                    "has no enclosed compartment.",
                    stacklevel=2,
                )
                continue
            region_volume_a3 = region_voxels * v_size**3

            pdbs = []
            for spec in specs_here:
                if spec.pdb_source not in pdb_cache:
                    pdb_cache[spec.pdb_source] = PDB(
                        spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False
                    )
                pdbs.append(pdb_cache[spec.pdb_source])
            species_radii = torch.tensor(
                [float(pdb.max_diameter) / 2.0 for pdb in pdbs]
            )
            species_ratios = torch.tensor([s.ratio for s in specs_here])

            pool_radii, pool_species_idx = draw_species_pool(
                species_radii,
                species_ratios,
                self.occupancy_fraction,
                region_volume_a3,
                seed=self.seed,
            )

            exclusion_field = (
                torch.from_numpy(
                    ndimage.distance_transform_edt(region_mask.cpu().numpy())
                ).to(device=self.device, dtype=torch.float32)
                * v_size
            )

            coords, accepted_idx = pack_hard_spheres_3d(
                pool_radii,
                box,
                gap=self.gap_angstrom,
                seed=self.seed,
                device=self.device,
                exclusion_distance_field=exclusion_field,
                field_v_size=v_size,
                sampling_mask=region_mask,
                max_passes=self.region_max_passes,
                # region-restricted sampling can need many more consecutive
                # misses than a "box is saturated" heuristic expects before
                # finding a geometrically valid spot (see pack_hard_spheres_3d's
                # own sampling_mask docstring) -- exhaust max_passes instead
                # of bailing out early on a run of misses.
                stall_patience=self.region_max_passes,
            )
            accepted_species_idx = pool_species_idx[accepted_idx]

            volume, instance_labels, next_instance_id = self._render_species_pool(
                specs_here,
                pdbs,
                coords,
                accepted_species_idx,
                volume,
                instance_labels,
                next_instance_id,
                location,
                v_size,
            )

        if self.filament_specs:
            volume, instance_labels, next_instance_id = self._stamp_filaments(
                volume, instance_labels, next_instance_id, v_size
            )
        else:
            self.filament_instances = []

        self.instance_labels = instance_labels
        return volume

    def _stamp_filaments(
        self,
        volume: torch.Tensor,
        instance_labels: torch.Tensor,
        next_instance_id: int,
        v_size: float,
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
        """
        target_shape = self.target_shape_zyx
        extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * v_size

        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)
        instances = place_filaments(self.filament_specs, target_shape, v_size, rng)
        self.filament_instances = instances
        if not instances:
            return volume, instance_labels, next_instance_id

        by_code: dict[str, list[FilamentInstance]] = {}
        for inst in instances:
            by_code.setdefault(inst.code, []).append(inst)

        pdb_cache: dict[str, PDB] = {}
        templates: dict[str, torch.Tensor] = {}
        for code in by_code:
            if code not in pdb_cache:
                pdb_cache[code] = PDB(
                    code, savefolder=self.pdb_cache_dir, verbose=False
                )
            pdb = pdb_cache[code]
            n = estimate_protein_box_size(pdb.max_diameter, v_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=v_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
            ).to(self.device)
            aligned_coordinates = align_principal_axis_to_z(pdb.coordinates)
            templates[code] = builder.forward(
                aligned_coordinates, method="analytic"
            ).to(self.device)

        offset = (extent_xyz / 2).to(self.device)
        for code, insts in by_code.items():
            template = templates[code]
            label_threshold = _INSTANCE_LABEL_REL_THRESHOLD * float(template.max())

            n_instances = len(insts)
            positions_centered = (
                torch.stack([inst.position_xyz for inst in insts]).to(self.device)
                - offset
            )
            R = torch.stack([inst.rotation_matrix for inst in insts]).to(self.device)
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
                volume = insert_particles_into_micrograph(
                    rotated,
                    positions_centered[start:end],
                    pixel_size=v_size,
                    micrograph=volume,
                )
                binarized = (rotated > label_threshold).to(torch.int32) * instance_ids[
                    start:end
                ].view(-1, 1, 1, 1)
                instance_labels = _insert_instance_labels(
                    binarized,
                    positions_centered[start:end],
                    pixel_size=v_size,
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
    ) -> dict[str, Path]:
        """
        Write one copick/CryoET-Data-Portal-style .ndjson pick file per
        placed cytosol/lumen species (grouped by `(location, species_id)`
        so the same `pdb_source` declared at both locations never collides
        in one file) plus, by default, one per transmembrane species --
        same schema as `specimen.packing.pdb_packing.
        SpherePackingSpecimenGenerator.export_picks`/`specimen.cryoet.
        CryoETSpecimenGenerator.export_picks` (one JSON object per line:
        ``{"type": "point"|"orientedPoint", "location": {"x", "y", "z"}[,
        "xyz_rotation_matrix"]}``), so picks from any of these generators
        are interchangeable downstream.

        Transmembrane picks are oriented (a real `rotation_matrix`, unlike
        `CryoETSpecimenGenerator`'s own plain-point membrane picks) since
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

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of a grouping key (``"{species}-{location}"`` for
            cytosol/lumen instances, ``"{species}-transmembrane"`` for
            transmembrane instances, ``"{species}-filament"`` for filament
            instances) to written file path.
        """
        if (
            not self.placements
            and not self.transmembrane_placements
            and not self.filament_instances
        ):
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        target_shape = self.target_shape_zyx
        v_size = self.v_size
        extent_xyz = (
            torch.tensor(
                [target_shape[2], target_shape[1], target_shape[0]],
                dtype=torch.float32,
            )
            * v_size
        )
        point_type = "orientedPoint" if oriented else "point"

        by_key: dict[str, list[TomogramPlacement]] = {}
        for placed in self.placements:
            name = Path(placed.species_id).stem
            by_key.setdefault(f"{name}-{placed.location}", []).append(placed)
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

        return written

    def _build_species_template(
        self, pdb: PDB, v_size: float, device: torch.device
    ) -> torch.Tensor:
        n = estimate_protein_box_size(pdb.max_diameter, v_size)
        builder = PotentialBuilder(
            n_xyz=n,
            dx=v_size,
            atomic_numbers=pdb.atomic_numbers,
            progressbars=False,
            parameterization=self.parameterization,
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
        v_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        active_species_i = [
            species_i
            for species_i in range(len(specs))
            if bool((accepted_species_idx == species_i).any())
        ]
        # Build every active species' potential template up front (optionally
        # concurrently across self.render_workers threads/self.render_devices
        # -- see MembraneTomogramGenerator's own render_workers docstring),
        # then run the rotate/insert loop below exactly as before. That loop
        # mutates volume/instance_labels/next_instance_id in place across
        # iterations and is comparatively cheap (batched GPU tensor ops), so
        # it stays sequential -- only the per-species PDB fetch/parse +
        # PotentialBuilder.forward is parallelized.
        templates = build_templates_concurrently(
            keys=active_species_i,
            build_one=lambda species_i, device: self._build_species_template(
                pdbs[species_i], v_size, device
            ),
            devices=self.render_devices,
            max_workers=self.render_workers,
        )

        for species_i, spec in enumerate(specs):
            mask = accepted_species_idx == species_i
            if not bool(mask.any()):
                continue
            template = templates[species_i]
            label_threshold = _INSTANCE_LABEL_REL_THRESHOLD * float(template.max())

            species_coords = coords[mask]
            n_instances = species_coords.shape[0]
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
                volume = insert_particles_into_micrograph(
                    rotated,
                    species_coords[start:end],
                    pixel_size=v_size,
                    micrograph=volume,
                )
                binarized = (rotated > label_threshold).to(torch.int32) * instance_ids[
                    start:end
                ].view(-1, 1, 1, 1)
                instance_labels = _insert_instance_labels(
                    binarized,
                    species_coords[start:end],
                    pixel_size=v_size,
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
                    )
                )

        return volume, instance_labels, next_instance_id
