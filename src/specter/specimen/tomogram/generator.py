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
it -- membrane auto-placement and filaments here, and (via
``classify_membrane_regions`` reading carbon's own high density as
"shell", the same bucket a real membrane bilayer occupies) beads and
cytosol/lumen protein fill below. Membrane instances given an explicit
``position_xyz`` (never collision-checked against anything else either,
e.g. other membrane instances) are the one case with no upfront placement
check against carbon -- but rather than compositing straight through it
regardless, whatever part of that instance's own rendered density would
land on carbon gets clipped (zeroed) right before it's merged into the
shared canvas, so both the composited volume and that instance's own
ground-truth shell label consistently exclude it (see the `to_composite`
loop below). Filament placement itself has no obstacle-avoiding random
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
smaller than the shared canvas (`MembraneGenerator`'s own `target_shape`
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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from ...config import ScalarOrRange, parse_scalar_or_range
from ...arrays import clip_insert_bounds
from ...crowding import insert_particles_into_micrograph
from ...pdb import DEFAULT_PDB_SAVEFOLDER, PDB
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
from ..membrane import MembraneGenerator, TransmembranePlacement
from ..membrane._placement import align_principal_axis_to_z
from ..packing import draw_species_pool, estimate_protein_box_size, pack_hard_spheres_3d
from ._regions import classify_membrane_regions
from ...progress import TqdmProgress, phase_done, phase_start, status

_INSTANCE_LABEL_REL_THRESHOLD = 0.01


def _wants_atom_species(parameterization: str) -> bool:
    """Whether to spend a gemmi bond-topology pass on a structure.

    The Shtyrov parameterization fits scattering factors per BONDED SPECIES
    (e.g. ``"C(HHHC)"``), so it can only do better than plain per-element
    factors when the bond topology is actually available. Computing it
    roughly doubles per-structure parse time and is wasted for Kirkland/
    Lobato, which are per-element by construction.

    Structures with no resolvable topology (legacy PDB format, isolated
    ions) degrade on their own: `PDB.get_atom_species` returns None for
    those atoms and `PotentialBuilder` falls back to Peng per-element
    factors for exactly them.
    """
    return parameterization == "shtyrov"


# A region covering at least this fraction of the whole tomogram box (e.g.
# a large cytosol with no membrane, or with just one small organelle in a
# big box) behaves like an open box for RSA packing purposes -- it
# saturates fast, so pack_hard_spheres_3d's own default stall_patience
# (15) is enough. Below this threshold (e.g. a small vesicle lumen), the
# region-restricted sampling_mask docstring's own reasoning applies: many
# more consecutive misses can be needed before a geometrically valid spot
# turns up, so region_max_passes' own (much larger) value is used instead.
# Verified directly on a 200x600x600 box (see PR discussion): stall_patience
# =15 packed the same PEI2016 candidate pool to within ~3% of stall_patience
# =300's own density in roughly half the wall time.
_TIGHT_REGION_FRACTION_THRESHOLD = 0.25
_OPEN_REGION_STALL_PATIENCE = 15  # matches pack_hard_spheres_3d's own default


def _insert_instance_labels(
    binarized: torch.Tensor,
    positions: torch.Tensor,
    pixel_size: float,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Stamp per-instance integer labels into a shared label volume.
    `binarized`/`positions` are moved to `labels`' own device (not the
    other way around) -- `labels` is the shared, potentially large
    accumulator (see TomogramSpecimenGenerator's own `accumulator_device`
    docstring), `binarized` is one small per-chunk rotated result."""
    device = labels.device
    binarized = binarized.to(device)
    N, Zp, Yp, Xp = binarized.shape
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
    voxel_size: float,
) -> tuple[int, int, int]:
    """Physical (x, y, z) offset from a volume's own center -> absolute
    (z, y, x) voxel index of that offset -- the center-relative convention
    `MembraneGenerator` itself uses (physical (0,0,0) = volume center),
    matching `_insert_instance_labels`'s own indexing math above. NOT
    `clip_insert_bounds`'s own corner-relative (0..extent) convention,
    which is for genuinely small local arrays -- the wrong frame for a
    MembraneGenerator instance, which always renders on its own full target
    grid centered at (0,0,0) (see that class's own module docstring)."""
    z_center, y_center, x_center = (
        shape_zyx[0] // 2,
        shape_zyx[1] // 2,
        shape_zyx[2] // 2,
    )
    px, py, pz = position_xyz
    return (
        z_center + int(round(pz / voxel_size)),
        y_center + int(round(py / voxel_size)),
        x_center + int(round(px / voxel_size)),
    )


def _insert_volume_max(
    volume: torch.Tensor,
    local: torch.Tensor,
    position_xyz: tuple[float, float, float],
    voxel_size: float,
) -> torch.Tensor:
    """Max-merge `local` (same center-relative convention as `volume`,
    i.e. physical (0,0,0) at its own center) into `volume`, shifted by
    `position_xyz`. See `_position_to_center_index` for why this uses a
    different convention than `clip_insert_bounds`. `local` is
    moved to `volume`'s own device (not the other way around) -- `volume`
    is the shared, potentially large accumulator (see
    TomogramSpecimenGenerator's own `accumulator_device` docstring),
    `local` is one membrane instance's own (much smaller) working grid."""
    center_zyx = _position_to_center_index(
        position_xyz, tuple(volume.shape), voxel_size
    )
    bounds = clip_insert_bounds(center_zyx, local.shape, volume.shape)
    if bounds is None:
        return volume
    dst, src = bounds
    volume[dst] = torch.maximum(volume[dst], local[src].to(volume.device))
    return volume


def _build_sphere_exclusion_field(
    coords_xyz: torch.Tensor,
    radii: torch.Tensor,
    target_shape: tuple[int, int, int],
    voxel_size: float,
) -> torch.Tensor:
    """Rasterize already-placed spheres into a boolean occupied grid and
    return its Euclidean distance transform, ready to combine (elementwise
    minimum, matching `pack_hard_spheres_3d`'s own "union of forbidden
    regions" guidance) with another `exclusion_distance_field` for a
    following packing stage."""
    Z, Y, X = target_shape
    occupied = np.zeros((Z, Y, X), dtype=bool)
    cz, cy, cx = Z / 2.0, Y / 2.0, X / 2.0

    coords_np = coords_xyz.cpu().numpy()
    radii_np = radii.cpu().numpy()
    for (x, y, z), r in zip(coords_np, radii_np):
        vz, vy, vx = cz + z / voxel_size, cy + y / voxel_size, cx + x / voxel_size
        r_vox = r / voxel_size
        z0, z1 = max(0, int(np.floor(vz - r_vox))), min(Z, int(np.ceil(vz + r_vox)) + 1)
        y0, y1 = max(0, int(np.floor(vy - r_vox))), min(Y, int(np.ceil(vy + r_vox)) + 1)
        x0, x1 = max(0, int(np.floor(vx - r_vox))), min(X, int(np.ceil(vx + r_vox)) + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        dist2 = (zz - vz) ** 2 + (yy - vy) ** 2 + (xx - vx) ** 2
        occupied[z0:z1, y0:y1, x0:x1] |= dist2 <= r_vox**2

    field = ndimage.distance_transform_edt(~occupied, sampling=(voxel_size,) * 3)
    return torch.from_numpy(field).float()


# Voxel budget for building an exclusion_distance_field/sampling_mask (see
# _resolve_exclusion_field_grid) -- scipy's distance_transform_edt over the
# WHOLE box is the dominant cost of packing at production scale regardless
# of how sparse the actual "forbidden" voxels are (confirmed directly:
# ~12 minutes and tens of GB of resident memory for one rebuild on a
# 4.5-billion-voxel box, with only ~90 small spheres actually occupied).
#
# Sized against the same 12 GiB host-RAM budget as MembraneGenerator's own
# _MAX_FIELD_VOXELS, but converted through this path's OWN measured cost
# rather than inheriting that one's voxel count: a single scipy EDT here
# (float64 out -> float32 tensor, plus the bool mask) instead of the field
# generator's two-plus-internals, and no cupy path. Measured at a flat 49
# bytes/voxel of peak RSS above baseline (16M/54M/128M/202M-voxel boxes with
# ~90 sparse obstacles: 49.0 at every size, 22 s at 202M).
#
# 12 GiB / 49 B = 263M, so the pre-existing 200M is already inside budget
# (~9.8 GB) and is left alone -- the budget is a ceiling, not a target, and
# raising it would only make packing slower for no accuracy the placements
# actually need. Internal on purpose, same as the membrane budget: it
# describes the machine, not the specimen.
_MAX_EXCLUSION_FIELD_VOXELS = 200_000_000


def _resolve_exclusion_field_grid(
    target_shape: tuple[int, int, int], voxel_size: float
) -> tuple[float, tuple[int, int, int], int]:
    """
    Coarsen ``(voxel_size, target_shape)`` for exclusion-field construction if
    the box exceeds ``_MAX_EXCLUSION_FIELD_VOXELS`` voxels.

    Safe to do because ``pack_hard_spheres_3d``'s own
    ``exclusion_distance_field``/``field_voxel_size`` mechanism is already
    documented to support (and trilinearly sample) a coarser grid than the
    box's own placement precision -- gap/radii here are tens of Angstrom,
    while ``voxel_size`` can be a small fraction of one; that docstring's own
    empirical finding ("a couple of Angstrom of bleed at field_voxel_size=5
    for gap=2, vanishing by field_voxel_size=2") already characterizes exactly
    this tradeoff. This just exploits a capability that was always
    available but never used before now (``field_voxel_size`` was always
    passed equal to ``voxel_size``).

    Returns
    -------
    field_voxel_size : float
        Coarsened voxel size, Angstrom (equals ``voxel_size`` if no
        coarsening was needed).
    field_shape : tuple of int
        Coarsened grid shape, ``ceil(target_shape / factor)`` per axis.
    factor : int
        Integer downsampling factor (1 if no coarsening was needed).
    """
    n = target_shape[0] * target_shape[1] * target_shape[2]
    if n <= _MAX_EXCLUSION_FIELD_VOXELS:
        return voxel_size, target_shape, 1
    factor = max(1, math.ceil((n / _MAX_EXCLUSION_FIELD_VOXELS) ** (1.0 / 3.0)))
    field_shape = tuple(math.ceil(s / factor) for s in target_shape)
    return voxel_size * factor, field_shape, factor


def _downsample_mask_maxpool(
    mask: torch.Tensor, factor: int, field_shape: tuple[int, int, int]
) -> torch.Tensor:
    """
    Coarsen a boolean mask by ``factor`` via max-pooling.

    ``ceil_mode=True`` so a coarse voxel is True if ANY of its
    constituent fine voxels is True -- a permissive/growing bias (the
    coarse "allowed" region can only be equal to or larger than the true
    fine one), matching the same direction of approximation
    ``pack_hard_spheres_3d``'s own coarse-field tolerance already accepts
    (see ``_resolve_exclusion_field_grid``), rather than a stricter
    erosion that risks losing thin true region.
    """
    pooled = F.max_pool3d(
        mask.to(torch.float32)[None, None],
        kernel_size=factor,
        stride=factor,
        ceil_mode=True,
    )[0, 0]
    assert tuple(pooled.shape) == field_shape
    return pooled > 0


def _diagnose_zero_placements(
    region_mask_field: torch.Tensor,
    exclusion_field: torch.Tensor,
    field_voxel_size: float,
    box: tuple[float, float, float],
    radius: float,
    gap: float,
    clip_axes: tuple[bool, bool, bool],
) -> tuple[int, float]:
    """
    Diagnose a 0-accepted `pack_hard_spheres_3d` call: how many field voxels
    are ACTUALLY viable for a sphere of this radius, and what the largest
    real clearance among them is.

    Exists because `exclusion_field[region_mask_field].max()` alone --
    clearance from the shell/exclusion source, ignoring the box wall -- can
    dramatically overstate how much room exists: it was reporting a
    misleadingly large number (found directly: 166 A "available", 72 A
    "needed", read as ample room) for a case with genuinely ZERO viable
    positions, because the voxels far enough from the shell were ALL too
    close to the box wall for this radius. `pack_hard_spheres_3d` itself
    already enforces the box-wall constraint (see its own `required_margin`
    check) -- this replicates just enough of that same logic, honoring
    `clip_axes`, to report a number a caller can actually act on instead of
    one that sends them looking in the wrong place.

    Parameters
    ----------
    region_mask_field : torch.Tensor
        Boolean, shape ``(Z, Y, X)`` -- same grid `exclusion_field` is on.
    exclusion_field : torch.Tensor
        Physical clearance to the nearest forbidden voxel, Angstrom, same
        shape as `region_mask_field`.
    field_voxel_size : float
        Voxel size of both fields, Angstrom.
    box : tuple of float
        ``(D, H, W)`` box extents in Angstrom (z, y, x) -- same convention
        `pack_hard_spheres_3d` takes.
    radius : float
        Sphere radius being diagnosed, Angstrom.
    gap : float
        Extra required clearance beyond touching, Angstrom.
    clip_axes : tuple of bool
        ``(z, y, x)`` -- True means only the CENTER needs to stay in-bounds
        on that axis (matching `pack_hard_spheres_3d`'s own parameter).

    Returns
    -------
    viable_voxels : int
        Voxels satisfying region membership, clearance, AND box containment
        all at once -- the true count `pack_hard_spheres_3d` was sampling
        from. Can be 0 even when `region_mask_field` and the raw clearance
        check both look generous.
    best_clearance_a : float
        The largest clearance among voxels that are at least IN the region
        and box-valid for this radius (ignoring the clearance requirement
        itself) -- 0.0 if no such voxel exists at all. Lets the warning
        report "this is the most room this species could ever get here",
        distinct from `exclusion_field`'s unconstrained max.
    """
    nz, ny, nx = region_mask_field.shape
    # exclusion_field is deliberately CPU-resident at the call site (see its
    # own comment there); region_mask_field can be on a different device
    # (self.device) since it's just a downsampled view of a GPU-resident
    # region classification. This diagnostic only runs on a zero-placement
    # cold path, so a one-time device copy here is fine.
    device = exclusion_field.device
    region_mask_field = region_mask_field.to(device)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    extent = (
        torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
        * field_voxel_size
    )
    origin = -0.5 * extent
    center_x = origin[0] + (xx.float() + 0.5) * field_voxel_size
    center_y = origin[1] + (yy.float() + 0.5) * field_voxel_size
    center_z = origin[2] + (zz.float() + 0.5) * field_voxel_size

    half_x, half_y, half_z = box[2] / 2, box[1] / 2, box[0] / 2
    margin_x = 0.0 if clip_axes[2] else radius
    margin_y = 0.0 if clip_axes[1] else radius
    margin_z = 0.0 if clip_axes[0] else radius
    within_box = (
        (center_x.abs() + margin_x <= half_x)
        & (center_y.abs() + margin_y <= half_y)
        & (center_z.abs() + margin_z <= half_z)
    )

    box_valid_region = region_mask_field & within_box
    best_clearance_a = (
        float(exclusion_field[box_valid_region].max())
        if bool(box_valid_region.any())
        else 0.0
    )
    viable = box_valid_region & (exclusion_field >= radius + gap)
    return int(viable.sum()), best_clearance_a


# Bytes/voxel for each accumulator tensor (volume: float32, instance_labels
# + membrane_labels: int32 -- all 4 bytes/voxel, so this is just a
# multiplier for how many of these coexist at once).
_ACCUMULATOR_BYTES_PER_VOXEL = 4
_ACCUMULATOR_N_TENSORS = 3  # volume, instance_labels, membrane_labels

# Maximum fraction of a CUDA device's CURRENTLY FREE memory the canvas is
# allowed to consume before recommend_accumulator_device falls back to
# CPU -- deliberately conservative (not "however much fits"), since
# rendering/rotation on that SAME device need real memory too, at the
# same time as the canvas exists, not before/after it.
_ACCUMULATOR_GPU_BUDGET_FRACTION = 0.5


def recommend_accumulator_device(
    device: str | torch.device,
    target_shape: tuple[int, int, int],
) -> torch.device:
    """
    Suggest an `accumulator_device`: `device` itself if the canvas
    (`target_shape`'s voxel count x `_ACCUMULATOR_N_TENSORS` same-
    sized tensors) fits within `_ACCUMULATOR_GPU_BUDGET_FRACTION` of that
    device's currently free memory, "cpu" otherwise. Trivially "cpu"
    whenever `device` isn't CUDA (or CUDA isn't available at all) --
    nothing to decouple from in that case.

    Parameters
    ----------
    device : str or torch.device
        The generator's own compute device (rendering/rotation/field
        generation) -- NOT necessarily the same as the returned
        accumulator device once this recommends "cpu".
    target_shape : tuple of int
        (Z, Y, X) voxels -- the shape every accumulator tensor will be.

    Returns
    -------
    torch.device
    """
    device_t = torch.device(device)
    if device_t.type != "cuda" or not torch.cuda.is_available():
        return torch.device("cpu")
    n_voxels = target_shape[0] * target_shape[1] * target_shape[2]
    estimated_bytes = n_voxels * _ACCUMULATOR_BYTES_PER_VOXEL * _ACCUMULATOR_N_TENSORS
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device_t)
    if estimated_bytes > _ACCUMULATOR_GPU_BUDGET_FRACTION * free_bytes:
        return torch.device("cpu")
    return device_t


def resolve_accumulator_device(
    device: str | torch.device,
    accumulator_device: str | torch.device | Literal["auto"] | None,
    target_shape: tuple[int, int, int],
) -> torch.device:
    """
    Normalize an `accumulator_device` config value -- `None` matches
    `device` (the original, one-device behaviour, unchanged default);
    `"auto"` resolves via `recommend_accumulator_device`; anything else
    (a concrete device string/`torch.device`) is used exactly as given.
    """
    if accumulator_device is None:
        return torch.device(device)
    if accumulator_device == "auto":
        return recommend_accumulator_device(device, target_shape)
    return torch.device(accumulator_device)


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
        return max(generator.sh_axes)
    return 0.5 * generator.swept_total_length + generator.swept_tube_radius


def _insert_shell_label(
    labels: torch.Tensor,
    shell_mask: torch.Tensor,
    instance_id: int,
    position_xyz: tuple[float, float, float],
    voxel_size: float,
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

    `shell_mask` is moved to `labels`' own device (not the other way
    around) -- see `_insert_volume_max`'s own docstring for why.
    """
    center_zyx = _position_to_center_index(
        position_xyz, tuple(labels.shape), voxel_size
    )
    bounds = clip_insert_bounds(center_zyx, shell_mask.shape, labels.shape)
    if bounds is None:
        return labels, False
    dst, src = bounds
    chunk = shell_mask[src].to(device=labels.device, dtype=labels.dtype) * instance_id
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
        Relative abundance weight among OTHER `ratio`-mode specs sharing
        the same `location` (species at different locations are packed
        independently, so ratios don't compare across locations). Ignored
        if `n_copies` is set. Default 1.0.
    n_copies : int, optional
        If set, place exactly this many instances of this species instead
        of ratio-weighted filling -- "target"/ground-truth semantics.
        Placed FIRST within this spec's `location`, before any
        `ratio`-mode species there (which then avoid the exact-count
        placements via an exclusion field, same mechanism used to avoid
        the membrane shell/filaments). Default None (ratio-weighted
        filler mode).
    """

    pdb_source: str
    location: Literal["cytosol", "lumen"] = "cytosol"
    ratio: float = 1.0
    n_copies: int | None = None

    def __post_init__(self) -> None:
        if self.n_copies is not None and self.n_copies <= 0:
            raise ValueError(
                f"TomogramProteinSpec({self.pdb_source!r}): n_copies must be "
                f"a positive int if set, got {self.n_copies}."
            )


@dataclass
class TomogramPlacement:
    """One placed cytosolic/lumen instance, for ground-truth bookkeeping.

    `role` is "target" for a `TomogramProteinSpec` placed via `n_copies`
    (exact count), "filler" for one placed via `ratio` -- `export_picks`
    excludes "filler" placements by default.
    """

    species_id: str
    location: str
    position_xyz: torch.Tensor
    rotation_matrix: torch.Tensor
    instance_id: int
    role: Literal["target", "filler"] = "target"


@dataclass
class TomogramBeadSpec:
    """
    One gold fiducial bead population to place -- built from real fcc gold
    atoms at bulk number density, so each bead carries lattice texture
    rather than being a uniform ball, and averages to gold's real mean
    inner potential (see `.._grid.BeadGenerator`). Every instance is an
    independent realisation, with its own crystal orientation. Scattered at
    an unrestricted ("any") location: fiducials sit in the ice itself, not
    a specific cytosol/lumen compartment, so there's no `location` field
    here the way `TomogramProteinSpec` has one. Placement still avoids the
    membrane shell and any already-placed filaments/other bead populations
    (see `TomogramSpecimenGenerator._stamp_beads`).

    Attributes
    ----------
    radius : float or [low, high]
        Bead radius, Angstrom. A ``[low, high]`` pair draws a fresh radius
        uniformly per instance -- real colloidal gold is not monodisperse
        (a "10 nm" prep typically spans roughly 8.5-11.5 nm). Radii are
        drawn *before* packing, so the placement's collision test uses each
        bead's own size. Same scalar-or-range spelling as
        `config.ParticleStackConfig`'s `dose`/`defocus`.
    count : int, optional
        Number of instances to place. Default 1.
    """

    radius: ScalarOrRange
    count: int = 1

    def __post_init__(self) -> None:
        self.radius_range = parse_scalar_or_range(self.radius)
        if self.radius_range[0] <= 0:
            raise ValueError(f"TomogramBeadSpec: radius must be > 0, got {self.radius}")
        if self.radius_range[1] < self.radius_range[0]:
            raise ValueError(
                f"TomogramBeadSpec: radius range must be [low, high], got {self.radius}"
            )
        if self.count <= 0:
            raise ValueError(f"TomogramBeadSpec: count must be > 0, got {self.count}")


@dataclass
class BeadPlacement:
    """One placed gold fiducial bead instance, for ground-truth bookkeeping."""

    radius: float
    position_xyz: torch.Tensor
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
        composite time -- its own `target_shape` need NOT match the
        owning `TomogramSpecimenGenerator`'s (typically shouldn't: leave it
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
        Shared voxel size, Angstrom -- must match every instance's own
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
    gap : float, optional
        Minimum clearance between placed spheres' surfaces, between a
        placed sphere and the membrane shell, AND between auto-placed
        membrane instances' own bounding spheres (all three reuse this one
        value). Default 5.0.
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
        `specter.pdb.DEFAULT_PDB_SAVEFOLDER` (`specter-data/pdb`, relative to
        the caller's cwd; see `config.default_pdb_cache_dir`).
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
        `target_shape`, dtype int32. Set after `generate()` runs.
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
        gap: float = 5.0,
        clip_axes: tuple[bool, bool, bool] = (False, False, False),
        region_density_threshold: float | None = None,
        region_max_passes: int = 300,
        min_transmembrane_spacing: float = 40.0,
        pdb_cache_dir: str = DEFAULT_PDB_SAVEFOLDER,
        parameterization: str = "shtyrov",
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
        self.gap = gap
        self.clip_axes = clip_axes
        self.region_density_threshold = region_density_threshold
        self.region_max_passes = region_max_passes
        self.min_transmembrane_spacing = min_transmembrane_spacing
        self.pdb_cache_dir = pdb_cache_dir
        self.parameterization = parameterization
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
        self.transmembrane_placements: list[TransmembranePlacement] = []
        self.placements: list[TomogramPlacement] = []
        self.instance_labels: torch.Tensor | None = None
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
            # Explicitly-positioned instances (to_composite, above) are NOT
            # checked against carbon either -- same existing gap as their
            # not being checked against each other (see this block's own
            # comment below); only auto-placement (this RSA solve) can
            # actually avoid it.
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
                f"Placing {len(auto_instances)} membrane instance(s)",
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
            n_dropped = len(auto_instances) - accepted_idx.numel()
            if n_dropped:
                warnings.warn(
                    f"TomogramSpecimenGenerator: {n_dropped}/{len(auto_instances)} "
                    "membrane instances with automatic (position_xyz=None) "
                    "placement did not fit without colliding (with each other, "
                    "the box walls, or the carbon film, if any) and were "
                    "dropped (never generated).",
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
        self.transmembrane_placements = []
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
                # to_composite starts as the explicitly-positioned instances,
                # then only accepted (position_xyz just set above) auto_instances
                # are appended to it.
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
                tm_placements = mi.generator.place_transmembrane(
                    min_spacing_a=self.min_transmembrane_spacing
                )
                offset = torch.tensor(mi.position_xyz, dtype=torch.float32)
                for tp in tm_placements:
                    tp.center_xyz = tp.center_xyz + offset

                local_volume = mi.generator.volume
                assert local_volume is not None
                if carbon_mask is not None:
                    # Auto-placed instances shouldn't reach here at all
                    # (their bounding sphere was already kept clear of
                    # carbon by the RSA exclusion field above), so this is
                    # a no-op for them in practice -- it's what actually
                    # protects explicitly-positioned instances (never
                    # collision-checked against anything, carbon included,
                    # see this loop's own leading comment), and is a real
                    # safety net either way if an irregular organelle's
                    # true shape extends past its own bounding-sphere
                    # approximation. Zeroes local_volume wherever it would
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
                                "membrane instance (explicit position_xyz or an "
                                "irregular shape exceeding its own bounding-"
                                "sphere estimate) that overlapped the carbon "
                                "film.",
                                stacklevel=2,
                            )
                            local_volume[src] = local_volume[src] * (~forbidden).to(
                                local_volume.dtype
                            )

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
                # is auto (None) -- membrane_scale already randomizes
                # contrast per instance (see MembraneGenerator's own
                # docstring), so a per-instance peak is arguably more
                # correct for per-instance shell LABELING anyway (a dim
                # instance's true shell isn't mislabeled as background just
                # because a brighter sibling set a higher global bar).
                # When region_density_threshold is explicitly set, it's
                # already an absolute density value (not a fraction, see
                # this same fallback below for self.regions), so using it
                # directly here is identical to a shared global threshold
                # -- no behaviour change in that case.
                if self.region_density_threshold is not None:
                    instance_threshold = self.region_density_threshold
                else:
                    local_peak = float(local_volume.max())
                    instance_threshold = 0.05 * local_peak if local_peak > 0 else 0.0
                shell_mask = (local_volume > instance_threshold).cpu()
                instance_shell_masks.append((mi, shell_mask))
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
                    "wins). Adjust position_xyz if this overlap wasn't intended.",
                    stacklevel=2,
                )
        self.membrane_labels = membrane_labels
        phase_done(
            f"Membranes ({len(instance_shell_masks)} instance(s))",
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

        instance_labels = torch.zeros(
            target_shape, dtype=torch.int32, device=self.accumulator_device
        )
        next_instance_id = 1

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
            _fetch_phase_start = phase_start(
                "Fetching PDB structures", disable=not self.progressbars
            )
            with TqdmProgress(
                transient=True, disable=not self.progressbars
            ) as progress:
                fetch_task = progress.add_task(
                    "Fetching PDB structures", total=len(unique_sources)
                )
                pdb_cache = build_pdb_cache_concurrently(
                    pdb_sources=unique_sources,
                    pdb_cache_dir=self.pdb_cache_dir,
                    max_workers=self.render_workers,
                    compute_atom_species=_wants_atom_species(self.parameterization),
                    on_result=lambda source: progress.update(
                        fetch_task, advance=1, description=f"Fetched {source}"
                    ),
                )
            phase_done(
                f"Fetched {len(unique_sources)} PDB structure(s)",
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
                region_mask = region_mask & ~obstacle_mask
            region_voxels = int(region_mask.sum())
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
                        savefolder=self.pdb_cache_dir,
                        verbose=False,
                        compute_atom_species=_wants_atom_species(self.parameterization),
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
                with status(
                    f"Packing {int(exact_radii.numel())} target instance(s) "
                    f"({location})",
                    disable=not self.progressbars,
                ):
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
                )
                phase_done(
                    f"  Target rendering ({location})",
                    _exact_render_start,
                    disable=not self.progressbars,
                )

                if coords.numel():
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

                pool_radii, pool_species_idx = draw_species_pool(
                    species_radii,
                    species_ratios,
                    self.occupancy_fraction,
                    region_volume_a3,
                    seed=self.seed,
                )

                _filler_pack_start = time.perf_counter()
                with status(
                    f"Packing filler instances ({location})",
                    disable=not self.progressbars,
                ):
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
                        # A TIGHT region (small fraction of the box, e.g. a
                        # vesicle lumen) can need many more consecutive
                        # misses than a "box is saturated" heuristic expects
                        # before finding a geometrically valid spot (see
                        # pack_hard_spheres_3d's own sampling_mask
                        # docstring) -- region_stall_patience exhausts
                        # region_max_passes there instead of bailing out
                        # early. An OPEN region (e.g. cytosol with no
                        # membrane, or with only a small organelle) instead
                        # gets pack_hard_spheres_3d's own fast default --
                        # it saturates quickly, so patiently retrying after
                        # that just burns time for near-zero extra density
                        # (see _TIGHT_REGION_FRACTION_THRESHOLD's own comment
                        # for a benchmark).
                        stall_patience=region_stall_patience,
                        clip_axes=self.clip_axes,
                    )
                phase_done(
                    f"  Filler packing ({location})",
                    _filler_pack_start,
                    disable=not self.progressbars,
                )
                if accepted_idx.numel() == 0:
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
        carbon_gen = CarbonFilmGenerator(
            voxel_size=voxel_size,
            parameterization=self.parameterization,
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
        return volume + film.density.to(volume.device)

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

        bead_gen = BeadGenerator(voxel_size=voxel_size, roughness=self.bead_roughness)
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
        return self._render_filament_instances(
            volume, instance_labels, next_instance_id, voxel_size, instances
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

        # One instance id per tube: `filament_id` already identifies it.
        tube_ids = sorted({inst.filament_id for inst in instances})
        id_of_tube = {
            tube_id: next_instance_id + i for i, tube_id in enumerate(tube_ids)
        }
        instance_ids = torch.tensor(
            [id_of_tube[inst.filament_id] for inst in instances], dtype=torch.int32
        )
        return self._render_filament_instances(
            volume,
            instance_labels,
            next_instance_id + len(tube_ids),
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
                    savefolder=self.pdb_cache_dir,
                    verbose=False,
                    compute_atom_species=_wants_atom_species(self.parameterization),
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
                        f"Placing {spec.pdb_source} ({n_instances} instances, "
                        f"{location}, {role})"
                    ),
                )
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
