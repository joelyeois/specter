"""Splat coordinate lists into voxel grids via soft (linear-interpolation) voxelization."""

from __future__ import annotations

import itertools
from typing import Sequence

import torch


def _normalize_voxel_size(
    voxel_size: float | Sequence[float] | torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """
    Broadcast a scalar or (dz, dy, dx) sequence voxel size to a (3,) tensor.
    """
    if isinstance(voxel_size, (int, float)):
        return torch.tensor([voxel_size] * 3, device=device)
    return torch.as_tensor(voxel_size, device=device)


def _ensure_batched_coords(coords: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Add a batch dimension to (N, 3) coordinates if missing.

    Returns
    -------
    coords : torch.Tensor
        Coordinates with shape (B, N, 3).
    was_unbatched : bool
        True if a batch dimension was added (caller should squeeze it back
        out of the result).
    """
    if coords.ndim == 2:
        return coords.unsqueeze(0), True
    return coords, False


def _linear_interp_offsets_and_weights(
    frac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    N-linear interpolation corner offsets and weights from fractional coordinates.

    Parameters
    ----------
    frac : torch.Tensor
        Fractional part of voxel coordinates, shape (..., D).

    Returns
    -------
    offsets : torch.Tensor
        Integer corner offsets, shape (2**D, D).
    weights : torch.Tensor
        Interpolation weight per corner, shape (..., 2**D).
    """
    d = frac.shape[-1]
    offsets = torch.tensor(
        list(itertools.product((0, 1), repeat=d)), device=frac.device
    )
    one_minus_frac = 1 - frac
    # per-dimension weight is frac where offset==1, else (1 - frac)
    per_dim_weight = torch.where(
        offsets.bool(), frac.unsqueeze(-2), one_minus_frac.unsqueeze(-2)
    )  # (..., 2**D, D)
    weights = per_dim_weight.prod(dim=-1)
    return offsets, weights


def _scatter_splat(
    volume: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    periodic: bool = False,
) -> None:
    """
    In-place accumulate `weights` into `volume` at `indices`.

    Parameters
    ----------
    volume : torch.Tensor
        Target grid, shape (d0, d1, ..., dk), modified in place.
    indices : torch.Tensor
        Integer indices, shape (..., k) matching `volume.ndim`.
    weights : torch.Tensor
        Weight per index, shape (...) matching `indices.shape[:-1]`.
    periodic : bool, optional
        If True, wrap out-of-bounds indices instead of discarding them.
    """
    grid_shape = torch.tensor(volume.shape, device=volume.device)
    if periodic:
        indices = indices % grid_shape
    else:
        in_bounds = ((indices >= 0) & (indices < grid_shape)).all(dim=-1)
        indices = indices[in_bounds]
        weights = weights[in_bounds]

    idx_tuple = tuple(indices[..., i].reshape(-1) for i in range(indices.shape[-1]))
    volume.index_put_(idx_tuple, weights.reshape(-1), accumulate=True)


def soft_voxelize_coordinates_into(
    volume: torch.Tensor,
    coords: torch.Tensor,
    voxel_size: float | Sequence[float] | torch.Tensor,
    periodic: bool = False,
) -> None:
    """
    Accumulate `coords` into a preallocated `volume` via trilinear
    splatting, in place.

    Uses the same centered-origin convention and interpolation as
    :func:`soft_voxelize_coordinates`, but never allocates a
    `volume`-shaped zero tensor itself -- callers preallocate `volume`
    once and can call this repeatedly (e.g. once per chunk of a much
    larger coordinate set) to keep peak memory bounded by the chunk size
    rather than the total coordinate count. Splatting disjoint chunks this
    way is exactly equivalent to a single call with every coordinate
    concatenated, since splatting is a linear (accumulating) operation --
    see :func:`_scatter_splat`.

    Parameters
    ----------
    volume : torch.Tensor
        Target grid, shape (nz, ny, nx), modified in place.
    coords : torch.Tensor
        Atomic coordinates, shape (N, 3).
    voxel_size : float or Sequence of float
        Voxel size. If float, assumes isotropic. If tuple, (dz, dy, dx).
    periodic : bool, optional
        If True, wrap out-of-bounds splat indices with periodic boundary
        conditions instead of discarding them. Default is False.
    """
    device = volume.device
    coords = coords.to(device)
    nz, ny, nx = volume.shape

    voxel_size_t = _normalize_voxel_size(voxel_size, device)
    coords_voxel = coords / voxel_size_t  # (N,3)

    # Shift coordinates so origin is at center
    origin = torch.tensor(
        [nx // 2, ny // 2, nz // 2], device=device, dtype=coords_voxel.dtype
    )
    coords_voxel_centered = coords_voxel + origin[None, :]  # (N,3)

    # Reorder to z,y,x
    coords_voxel_centered = coords_voxel_centered[..., [2, 1, 0]]

    coords_floor = torch.floor(coords_voxel_centered).long()  # (N,3)
    frac = coords_voxel_centered - coords_floor.float()

    offsets, weights = _linear_interp_offsets_and_weights(frac)  # (8,3), (N,8)
    indices = coords_floor.unsqueeze(-2) + offsets  # (N,8,3)

    _scatter_splat(volume, indices, weights, periodic=periodic)


def soft_voxelize_coordinates(
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    voxel_size: float | Sequence[float],
    device: str | torch.device | None = None,
    periodic: bool = False,
) -> torch.Tensor:
    """
    Differentiable 3D soft voxelization using trilinear splatting.

    Distributes each coordinate's contribution to surrounding voxels using
    trilinear interpolation for smooth, differentiable voxelization.

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates. Either (N, 3) for single volume or (B, N, 3)
        for batched volumes.
    grid_shape : tuple of int
        Shape of output grid (nz, ny, nx).
    voxel_size : float or Sequence of float
        Voxel size. If float, assumes isotropic. If tuple, (dz, dy, dx).
    device : str or torch.device, optional
        Device for tensors. Default is None (uses coords device).
    periodic : bool, optional
        If True, wrap out-of-bounds splat indices with periodic boundary
        conditions instead of discarding them. Default is False.

    Returns
    -------
    volume : torch.Tensor
        Soft voxelized volume. Shape (nz, ny, nx) if coords is (N, 3),
        or (B, nz, ny, nx) if coords is (B, N, 3).

    Notes
    -----
    Uses trilinear interpolation to distribute each atom's contribution
    among its 8 neighboring voxels, weighted by distance.
    """
    if device is None:
        device = coords.device
    coords = coords.to(device)
    coords, was_unbatched = _ensure_batched_coords(coords)

    B, N, _ = coords.shape
    nz, ny, nx = grid_shape

    voxel_size_t = _normalize_voxel_size(voxel_size, device)

    volume = torch.zeros(B, nz, ny, nx, device=device)
    for b in range(B):
        soft_voxelize_coordinates_into(
            volume[b], coords[b], voxel_size_t, periodic=periodic
        )

    if was_unbatched:
        volume = volume.squeeze(0)

    return volume


def soft_voxelize_xy_coordinates(
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    voxel_size: float | Sequence[float],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Semi-soft 3D voxelization with hard Z assignment and soft XY interpolation.

    Uses nearest-neighbor assignment along Z axis (hard) and bilinear
    interpolation in XY plane (soft) for faster computation than full
    trilinear interpolation.

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates. Either (N, 3) for single volume or (B, N, 3)
        for batched volumes.
    grid_shape : tuple of int
        Shape of output grid (nz, ny, nx).
    voxel_size : float or Sequence of float
        Voxel size. If float, assumes isotropic. If tuple, (dz, dy, dx).
    device : str or torch.device, optional
        Device for tensors. Default is None (uses coords device).

    Returns
    -------
    volume : torch.Tensor
        Semi-soft voxelized volume. Shape (nz, ny, nx) if coords is (N, 3),
        or (B, nz, ny, nx) if coords is (B, N, 3).

    Notes
    -----
    Faster than full trilinear but less smooth along Z. Good compromise
    for oriented structures like membranes.
    """
    if device is None:
        device = coords.device
    coords = coords.to(device)
    coords, was_unbatched = _ensure_batched_coords(coords)

    B, N, _ = coords.shape
    nz, ny, nx = grid_shape

    voxel_size_t = _normalize_voxel_size(voxel_size, device)
    origin = torch.tensor(
        [nx // 2, ny // 2, nz // 2], device=device, dtype=coords.dtype
    )

    volume = torch.zeros(B, nz, ny, nx, device=device)
    for b in range(B):
        coords_voxel_centered = coords[b] / voxel_size_t + origin[None, :]
        x, y, z = (
            coords_voxel_centered[:, 0],
            coords_voxel_centered[:, 1],
            coords_voxel_centered[:, 2],
        )

        # Hard Z assignment
        z_idx = torch.round(z).long()

        # Soft XY assignment (bilinear)
        xy_floor = torch.floor(torch.stack([y, x], dim=-1)).long()  # (N,2)
        frac_yx = torch.stack([y, x], dim=-1) - xy_floor.float()
        offsets, weights = _linear_interp_offsets_and_weights(frac_yx)  # (4,2), (N,4)
        yx_idx = xy_floor.unsqueeze(-2) + offsets  # (N,4,2)

        z_idx_full = z_idx[:, None].expand(-1, weights.shape[-1])  # (N,4)
        indices = torch.cat([z_idx_full.unsqueeze(-1), yx_idx], dim=-1)  # (N,4,3)

        _scatter_splat(volume[b], indices, weights)

    if was_unbatched:
        volume = volume.squeeze(0)

    return volume
