"""Module-level geometry/placement helpers for TomogramSpecimenGenerator."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from ...arrays import clip_insert_bounds
from ...crowding import insert_particles_into_micrograph
from ...rotations import rotate_volume
from ..membrane import MembraneGenerator, membrane_bounding_radius


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


def _insert_instance_labels(
    binarized: torch.Tensor,
    positions: torch.Tensor,
    voxel_size: float,
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
    positions_int = (positions / voxel_size).round().long()
    cz_center, cy_center, cx_center = Z // 2, Y // 2, X // 2

    # One host transfer per chunk, not three device syncs per instance --
    # same reasoning as `crowding._insert_all`'s own loop.
    positions_list = positions_int.cpu().tolist()
    for i, (px_i, py_i, pz_i) in enumerate(positions_list):
        cx_index = cx_center + px_i
        cy_index = cy_center + py_i
        cz_index = cz_center + pz_i
        bounds = clip_insert_bounds(
            (cz_index, cy_index, cx_index), (Zp, Yp, Xp), (Z, Y, X)
        )
        if bounds is None:
            continue
        dst, src = bounds
        chunk = binarized[i][src]
        labels[dst] = torch.where(chunk > 0, chunk, labels[dst])

    return labels


_INSTANCE_LABEL_REL_THRESHOLD = 0.01


def _insert_rotated_copies(
    template: torch.Tensor,
    theta: torch.Tensor,
    positions: torch.Tensor,
    instance_ids: torch.Tensor,
    volume: torch.Tensor,
    instance_labels: torch.Tensor,
    voxel_size: float,
    chunk_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate one template per instance, insert the copies into `volume` and
    their binarized footprints into `instance_labels`, a chunk at a time.

    Shared by the protein stage (`_render_species_pool`) and filament/
    microtubule rendering (`_render_filament_instances`). A voxel is
    labelled where the rotated copy exceeds `_INSTANCE_LABEL_REL_THRESHOLD`
    of the template's own peak.

    Parameters
    ----------
    template : torch.Tensor
        Potential template, shape (Z, Y, X), on the compute device.
    theta : torch.Tensor
        Affine matrices from `build_affine_matrix`, shape (N, 3, 4).
    positions : torch.Tensor
        Box-centred instance positions (x, y, z), A, shape (N, 3).
    instance_ids : torch.Tensor
        int32 instance ids, shape (N,).
    volume, instance_labels : torch.Tensor
        The shared canvas and its label volume.
    voxel_size : float
        Voxel size, A.
    chunk_size : int or None
        Instances rotated per batch; None rotates all N at once.

    Returns
    -------
    tuple of torch.Tensor
        The updated `volume` and `instance_labels`.
    """
    label_threshold = _INSTANCE_LABEL_REL_THRESHOLD * float(template.max())
    n_instances = positions.shape[0]
    step = chunk_size or n_instances
    for start in range(0, n_instances, step):
        end = min(start + step, n_instances)
        rotated = rotate_volume(template, theta[start:end], padding_mode="zeros")
        # Moved to the ACCUMULATOR's device (not necessarily the compute
        # device) right after the compute-heavy rotation -- only this small
        # per-chunk result crosses devices, never the shared canvas itself
        # (see `TomogramSpecimenGenerator`'s accumulator_device docstring).
        rotated = rotated.to(volume.device)
        volume = insert_particles_into_micrograph(
            rotated,
            positions[start:end],
            pixel_size=voxel_size,
            micrograph=volume,
        )
        binarized = (rotated > label_threshold).to(torch.int32) * instance_ids[
            start:end
        ].to(volume.device).view(-1, 1, 1, 1)
        instance_labels = _insert_instance_labels(
            binarized,
            positions[start:end],
            voxel_size=voxel_size,
            labels=instance_labels,
        )
    return volume, instance_labels


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

# Voxel budget for protein packing's own occupancy grid, above which
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
    return _coarsen_grid_to_budget(
        target_shape, voxel_size, _MAX_EXCLUSION_FIELD_VOXELS
    )


def _coarsen_grid_by(
    target_shape: tuple[int, int, int], voxel_size: float, factor: int
) -> tuple[float, tuple[int, int, int], int]:
    """Coarsen a grid by an integer ``factor``: ``(voxel, ceil(shape / factor), factor)``."""
    shape = tuple(math.ceil(s / factor) for s in target_shape)
    return voxel_size * factor, shape, factor  # type: ignore[return-value]


def _coarsen_grid_to_budget(
    target_shape: tuple[int, int, int], voxel_size: float, max_voxels: int
) -> tuple[float, tuple[int, int, int], int]:
    """
    Coarsen ``(voxel_size, target_shape)`` by the smallest integer factor
    that brings the voxel count to at most ``max_voxels`` (per-axis factor
    ``ceil((n / max_voxels) ** (1/3))``); returned unchanged, with factor 1,
    when the grid already fits.
    """
    n = target_shape[0] * target_shape[1] * target_shape[2]
    if n <= max_voxels:
        return voxel_size, target_shape, 1
    factor = max(1, math.ceil((n / max_voxels) ** (1.0 / 3.0)))
    return _coarsen_grid_by(target_shape, voxel_size, factor)


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


def _allowed_region_exclusion_field(
    allowed: torch.Tensor,
    target_shape: tuple[int, int, int],
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build the sampling mask and clearance field `pack_hard_spheres_3d`
    takes from a boolean ``allowed`` mask on the render grid.

    The grid is coarsened per `_resolve_exclusion_field_grid`, ``allowed``
    max-pooled onto it (`_downsample_mask_maxpool`), and the clearance is
    the Euclidean distance, in Å, from each allowed voxel to the nearest
    forbidden one.

    Parameters
    ----------
    allowed : torch.Tensor
        Boolean CPU mask, shape ``target_shape``; True where a centre may go.
    target_shape : tuple of int
        ``(Z, Y, X)`` render grid shape.
    voxel_size : float
        Render voxel size, Å.

    Returns
    -------
    allowed_field : torch.Tensor
        ``allowed`` on the (possibly coarsened) field grid.
    exclusion_field : torch.Tensor
        float32 clearance, Å, same shape as ``allowed_field``.
    field_voxel_size : float
        Voxel size of both fields, Å.
    """
    field_voxel_size, field_shape, field_factor = _resolve_exclusion_field_grid(
        target_shape, voxel_size
    )
    allowed_field = (
        _downsample_mask_maxpool(allowed, field_factor, field_shape)
        if field_factor > 1
        else allowed
    )
    exclusion_field = (
        torch.from_numpy(ndimage.distance_transform_edt(allowed_field.numpy())).float()
        * field_voxel_size
    )
    return allowed_field, exclusion_field, field_voxel_size


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
    return membrane_bounding_radius(
        generator.shape_backend,
        sh_axes=generator.sh_axes,
        swept_total_length=generator.swept_total_length,
        swept_tube_radius=generator.swept_tube_radius,
    )


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


def _insert_local_labels(
    labels: torch.Tensor,
    local_labels: torch.Tensor,
    id_offset: int,
    position_xyz: tuple[float, float, float],
    voxel_size: float,
) -> torch.Tensor:
    """Stamp a per-instance label volume rendered in one membrane
    instance's own local frame into the shared `labels` volume, shifted by
    `position_xyz` and with every id shifted by `id_offset`.

    The local ids are 1-based and local to their membrane
    (`MembraneGenerator.transmembrane_labels`), so `id_offset` is the
    shared counter's value before this instance's block was reserved.
    Same center-relative convention as `_insert_shell_label`, and
    FIRST-write-wins for the same reason: two membrane instances are only
    collision-checked as bounding spheres, so their rendered shapes -- and
    the proteins embedded in them -- can still overlap, and which one wins
    has to be deterministic rather than depend on iteration order.

    Parameters
    ----------
    labels : torch.Tensor
        Shared integer label volume, modified and returned.
    local_labels : torch.Tensor
        Integer labels in the membrane instance's own frame; 0 is empty.
    id_offset : int
        Added to every nonzero local id.
    position_xyz : tuple of float
        The membrane instance's own physical offset from the shared
        volume's center.
    voxel_size : float
        Angstroms per voxel.

    Returns
    -------
    torch.Tensor
        The updated `labels`.
    """
    center_zyx = _position_to_center_index(
        position_xyz, tuple(labels.shape), voxel_size
    )
    bounds = clip_insert_bounds(center_zyx, local_labels.shape, labels.shape)
    if bounds is None:
        return labels
    dst, src = bounds
    chunk = local_labels[src].to(device=labels.device, dtype=labels.dtype)
    chunk = torch.where(chunk > 0, chunk + id_offset, chunk)
    labels[dst] = torch.where(labels[dst] > 0, labels[dst], chunk)
    return labels
