"""Module-level geometry/placement helpers for TomogramSpecimenGenerator."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from ...arrays import clip_insert_bounds
from ..membrane import MembraneGenerator


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

# Voxel budget for packing_backend="shape"'s own occupancy grid, above which
# `packing_voxel_size=None` coarsens the collision grid automatically. One
# byte per voxel, so this is a ~1 GB host allocation.
#
# Deliberately well above _MAX_EXCLUSION_FIELD_VOXELS: a 200x1200x1200 box at
# 5 A is 288M voxels and must NOT trigger, since coarsening it would cost
# density on a configuration that already fits comfortably. What this is for
# is the regime where packing natively is not merely slow but impossible --
# a 1 A production box needs a 36 GB occupancy grid and a 20 GB rotation
# cache for a single 243 A species.
_MAX_PACKING_GRID_VOXELS = 1_000_000_000


def _resolve_exclusion_field_grid(
    target_shape: tuple[int, int, int], voxel_size: float
) -> tuple[float, tuple[int, int, int], int]:
    """
    Coarsen ``(voxel_size, target_shape)`` for exclusion-field construction if
    the box exceeds ``_MAX_EXCLUSION_FIELD_VOXELS`` voxels.

    Safe to do because ``pack_hard_spheres_3d``'s own
    ``exclusion_distance_field``/``field_voxel_size`` mechanism is already
    documented to support (and trilinearly sample) a coarser grid than the
    box's own placement precision -- gap/radii here are tens of Å,
    while ``voxel_size`` can be a small fraction of one; that docstring's own
    empirical finding ("a couple of Å of bleed at field_voxel_size=5
    for gap=2, vanishing by field_voxel_size=2") already characterizes exactly
    this tradeoff. This just exploits a capability that was always
    available but never used before now (``field_voxel_size`` was always
    passed equal to ``voxel_size``).

    Returns
    -------
    field_voxel_size : float
        Coarsened voxel size, Å (equals ``voxel_size`` if no
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
        Physical clearance to the nearest forbidden voxel, Å, same
        shape as `region_mask_field`.
    field_voxel_size : float
        Voxel size of both fields, Å.
    box : tuple of float
        ``(D, H, W)`` box extents in Å (z, y, x) -- same convention
        `pack_hard_spheres_3d` takes.
    radius : float
        Sphere radius being diagnosed, Å.
    gap : float
        Extra required clearance beyond touching, Å.
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
    instances are only collision-checked as bounding spheres, so an
    irregular shape extending past its own bounding-sphere estimate can
    still overlap another instance -- which instance "wins" that overlap
    must be a deliberate, deterministic choice).

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
