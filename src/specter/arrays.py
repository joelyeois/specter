from __future__ import annotations

import itertools
from typing import Literal, Sequence, overload

import torch
import torch.nn.functional as F

from .fft import fft2, ifft2


def kgrid_1d(n: int, dx: float, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Constructs a 1D k-grid.

    Parameters
    ----------
    n : int
        Number of pixels.
    dx : float
        Pixel size.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    k : torch.Tensor
        1D k-grid.
    """
    kx = torch.fft.fftfreq(n, dx, device=device)
    return kx


def kgrid_2d(
    n_xy: int | Sequence[int],
    d_xy: float | Sequence[float],
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs the kx, ky meshgrids.

    Parameters
    ----------
    n_xy : int or Sequence of int
        Number of pixels along x, y. If int, assumes nx=ny=n_xy.
    d_xy : float or Sequence of float
        Pixel length along x, y. If float, assumes dx=dy=d_xy.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    kx, ky : torch.Tensor
        1D frequency axes.
    KX, KY : torch.Tensor
        2D kx, ky meshgrids.
    """
    if isinstance(n_xy, int):
        nx = ny = n_xy
    else:
        nx, ny = n_xy

    if isinstance(d_xy, (int, float)) or (
        isinstance(d_xy, torch.Tensor) and d_xy.ndim == 0
    ):
        dx = dy = float(d_xy)
    else:
        dx, dy = d_xy
    kx = kgrid_1d(nx, dx, device=device)
    ky = kgrid_1d(ny, dy, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    return kx, ky, KX, KY


def kgrid_3d(
    n_xyz: int | Sequence[int],
    d_xyz: float | Sequence[float],
    device: str | torch.device = "cpu",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Constructs the kx, ky, kz meshgrids.

    Parameters
    ----------
    n_xyz : int or Sequence of int
        Number of pixels along x, y, z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or Sequence of float
        Pixel length along x, y, z. If float, assumes dx=dy=dz=d_xyz.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    kx, ky, kz : torch.Tensor
        1D frequency axes.
    KX, KY, KZ : torch.Tensor
        3D kx, ky, kz meshgrids.
    """
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    if isinstance(d_xyz, (int, float)) or (
        isinstance(d_xyz, torch.Tensor) and d_xyz.ndim == 0
    ):
        dx = dy = dz = float(d_xyz)
    else:
        dx, dy, dz = d_xyz
    kx = kgrid_1d(nx, dx, device=device)
    ky = kgrid_1d(ny, dy, device=device)
    kz = kgrid_1d(nz, dz, device=device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    return kx, ky, kz, KX, KY, KZ


def radial_kgrid_2d(
    n_xy: int | Sequence[int],
    d_xy: float | Sequence[float],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Construct 2D radial frequency grid.

    Parameters
    ----------
    n_xy : int or Sequence of int
        Number of pixels along x, y. If int, assumes nx=ny=n_xy.
    d_xy : float or Sequence of float
        Pixel length along x, y. If float, assumes dx=dy=d_xy.
    device : str or torch.device, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    K_radial : torch.Tensor
        2D radial frequencies, shape (ny, nx). Values represent the
        distance from the origin in frequency space.
    """
    _, _, KX, KY = kgrid_2d(n_xy, d_xy, device)
    return torch.sqrt(KX**2 + KY**2)


def radial_kgrid_3d(
    n_xyz: int | Sequence[int],
    d_xyz: float | Sequence[float],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Construct 3D radial frequency grid.

    Parameters
    ----------
    n_xyz : int or Sequence of int
        Number of pixels along x, y, z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or Sequence of float
        Pixel length along x, y, z. If float, assumes dx=dy=dz=d_xyz.
    device : str or torch.device, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    K_radial : torch.Tensor
        3D radial frequencies, shape (nz, ny, nx). Values represent the
        distance from the origin in 3D frequency space.
    """
    _, _, _, KX, KY, KZ = kgrid_3d(n_xyz, d_xyz, device)
    return torch.sqrt(KX**2 + KY**2 + KZ**2)


def grid_1d(
    n: int, dx: float, convention: str = "relion", device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Constructs a 1D grid. The coordinate grid convention only matters if the number of
    pixels is even.

    Parameters
    ----------
    n : int
        Number of pixels.
    dx : float
        Pixel size.
    convention : str, optional
        Determines location of origin (0). If 'relion', origin is located at
        index [n//2], which means coordinates are not symmetric about
        the origin for even number of pixels. If 'torch', forces coordinates to be
        symmetric about the origin, which means for even grids, there is no index
        for (0). Default is 'relion'.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    x : torch.Tensor
        1D coordinate array.
    """
    if convention == "relion":
        x = (torch.arange(n, device=device) - n // 2) * dx
    elif convention == "torch":
        x = (torch.arange(n, device=device) - (n - 1) / 2) * dx
    else:
        raise ValueError(
            f"Unknown convention '{convention}'. Must be 'relion' or 'torch'."
        )
    return x


def grid_2d(
    n_xy: int | Sequence[int],
    d_xy: float | Sequence[float],
    convention: str = "relion",
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs the xy coordinate arrays and meshgrids. Meshgrid indexing yields
    area[y_i, x_i] convention which matches cryo-EM software.

    The coordinate grid convention only matters if the number of pixels is even.

    Parameters
    ----------
    n_xy : int or Sequence of int
        Number of pixels along x, y. If int, assumes nx=ny=n_xy.
    d_xy : float or Sequence of float
        Pixel length along x, y. If float, assumes dx=dy=d_xy.
    convention : str, optional
        Determines location of origin (0,0). If 'relion', origin is located at
        index [ny//2, nx//2], which means coordinates are not symmetric about
        the origin for even number of pixels. If 'torch', forces coordinates to be
        symmetric about the origin, which means for even grids, there is no index
        for (0,0). Default is 'relion'.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    x, y : torch.Tensor
        1D x, y coordinates.
    X, Y : torch.Tensor
        2D x, y meshgrids.
    """
    if isinstance(n_xy, int):
        nx = ny = n_xy
    else:
        nx, ny = n_xy

    if isinstance(d_xy, (int, float)) or (
        isinstance(d_xy, torch.Tensor) and d_xy.ndim == 0
    ):
        dx = dy = float(d_xy)
    else:
        dx, dy = d_xy
    x = grid_1d(nx, dx, convention=convention, device=device)
    y = grid_1d(ny, dy, convention=convention, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return x, y, X, Y


def grid_3d(
    n_xyz: int | Sequence[int],
    d_xyz: float | Sequence[float],
    convention: str = "relion",
    device: str | torch.device = "cpu",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Constructs the xyz coordinate arrays and meshgrids. Meshgrid indexing yields
    vol[z_i, y_i, x_i] convention which matches cryo-EM software.

    The coordinate grid convention only matters if the number of pixels is even.

    Parameters
    ----------
    n_xyz : int or Sequence of int
        Number of pixels along x, y, z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or Sequence of float
        Pixel length along x, y, z. If float, assumes dx=dy=dz=d_xyz.
    convention : str, optional
        Determines location of origin (0,0,0). If 'relion', origin is located at
        index [nz//2, ny//2, nx//2], which means coordinates are not symmetric about
        the origin for even number of pixels. If 'torch', forces coordinates to be
        symmetric about the origin, which means for even grids, there is no index
        for (0,0,0). Default is 'relion'.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    x, y, z : torch.Tensor
        1D x, y, z coordinates.
    X, Y, Z : torch.Tensor
        3D x, y, z meshgrids.
    """
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    if isinstance(d_xyz, (int, float)) or (
        isinstance(d_xyz, torch.Tensor) and d_xyz.ndim == 0
    ):
        dx = dy = dz = float(d_xyz)
    else:
        dx, dy, dz = d_xyz
    x = grid_1d(nx, dx, convention, device)
    y = grid_1d(ny, dy, convention, device)
    z = grid_1d(nz, dz, convention, device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
    return x, y, z, X, Y, Z


def radial_grid_2d(
    n_xy: int | Sequence[int],
    d_xy: float | Sequence[float],
    convention: str = "relion",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Construct 2D radial coordinate grid.

    Parameters
    ----------
    n_xy : int or Sequence of int
        Number of pixels along x, y. If int, assumes nx=ny=n_xy.
    d_xy : float or Sequence of float
        Pixel length along x, y. If float, assumes dx=dy=d_xy.
    convention : str, optional
        Grid origin convention ('relion' or 'torch'). Default is 'relion'.
    device : str or torch.device, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    R : torch.Tensor
        2D radial distances from origin, shape (ny, nx).
    """
    _, _, X, Y = grid_2d(n_xy, d_xy, convention, device)
    return torch.sqrt(X**2 + Y**2)


def radial_grid_3d(
    n_xyz: int | Sequence[int],
    d_xyz: float | Sequence[float],
    convention: str = "relion",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Construct 3D radial coordinate grid.

    Parameters
    ----------
    n_xyz : int or Sequence of int
        Number of pixels along x, y, z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or Sequence of float
        Pixel length along x, y, z. If float, assumes dx=dy=dz=d_xyz.
    convention : str, optional
        Grid origin convention ('relion' or 'torch'). Default is 'relion'.
    device : str or torch.device, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    R : torch.Tensor
        3D radial distances from origin, shape (nz, ny, nx).
    """
    _, _, _, X, Y, Z = grid_3d(n_xyz, d_xyz, convention, device)
    return torch.sqrt(X**2 + Y**2 + Z**2)


def real_to_kgrid_3d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert real-space radial grid to frequency-space radial grid.

    Given a 3D real-space meshgrid magnitude R, returns the corresponding
    frequency-space radial grid KR for FFTs.

    Parameters
    ----------
    R : torch.Tensor
        3D tensor of radial distances in real space, shape (nx, ny, nz).

    Returns
    -------
    KR : torch.Tensor
        3D tensor of radial distances in Fourier space, shape (nx, ny, nz).

    Notes
    -----
    Supports non-cubic grids with different spacings along each axis.
    Assumes uniform spacing along each individual axis.
    """
    device = R.device

    # number of points along each axis
    nx, ny, nz = R.shape

    # compute spacing along each axis (assumes uniform spacing)
    dx = R[1, 0, 0] - R[0, 0, 0]
    dy = R[0, 1, 0] - R[0, 0, 0]
    dz = R[0, 0, 1] - R[0, 0, 0]

    # frequency axes
    kx = torch.fft.fftfreq(nx, dx, device=device)
    ky = torch.fft.fftfreq(ny, dy, device=device)
    kz = torch.fft.fftfreq(nz, dz, device=device)

    # 3D frequency grids
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    KR = torch.sqrt(KX**2 + KY**2 + KZ**2)

    return KR


def voxelize_coordinates(
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    voxel_size: Sequence[float],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convert 3D coordinates to a 3D binary occupancy grid.

    Creates a binary grid with 1s at the nearest voxel positions corresponding
    to the input coordinates, and 0s elsewhere.

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates in physical units (x, y, z), shape (N, 3).
    grid_shape : tuple of int
        Shape of output grid (nx, ny, nz).
    voxel_size : Sequence of float
        Voxel size (dx, dy, dz).
    device : str or torch.device, optional
        Device for tensors. Default is None (uses coords device).

    Returns
    -------
    grid : torch.Tensor
        Binary occupancy grid with shape (nz, ny, nx). 1s at voxel positions
        of atoms, 0s elsewhere.

    Notes
    -----
    Grid center is assumed to be at (nz//2, ny//2, nx//2).
    Uses nearest-neighbor assignment (hard voxelization).
    """
    device = device or coords.device
    coords = coords.to(device)
    nx, ny, nz = grid_shape  # number of voxels along x, y, z

    # Compute the center of the grid in voxel units
    center_voxel = torch.tensor([nx // 2, ny // 2, nz // 2], device=device)

    # Convert physical coordinates to voxel indices, shifting center
    indices = coords / torch.tensor(voxel_size, device=device)  # voxel units
    indices = indices + center_voxel  # shift so center is at middle voxel
    indices = torch.round(indices).long()

    # Mask atoms inside the grid
    mask = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < nx)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < ny)
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < nz)
    )
    indices = indices[mask]

    # Create empty grid (z, y, x)
    grid = torch.zeros((nz, ny, nx), device=device, dtype=torch.float32)

    # Insert ones at valid voxel indices
    grid.index_put_(
        (indices[:, 2], indices[:, 1], indices[:, 0]),
        torch.ones(indices.shape[0], device=device),
        accumulate=True,
    )

    return grid


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


@overload
def radial_profile_3d(
    data: torch.Tensor,
    center: tuple[float, float, float] | None = None,
    return_r: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def radial_profile_3d(
    data: torch.Tensor,
    center: tuple[float, float, float] | None = None,
    return_r: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def radial_profile_3d(
    data: torch.Tensor,
    center: tuple[float, float, float] | None = None,
    return_r: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the radial average (spherical average) of a 3D volume.

    Parameters
    ----------
    data : torch.Tensor
        3D tensor of shape (m, n, o).
    center : tuple of float, optional
        Center of the radial profile. Defaults to the integer center of the volume (m//2, n//2, o//2).
    return_r : bool, optional
        If True, also return the radius indices. Default is False.

    Returns
    -------
    radialprofile : torch.Tensor
        Radial average.
    r : torch.Tensor, optional
        Radius indices if return_r=True.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D tensor.")

    m, n, o = data.shape
    device = data.device

    # Default integer center
    if center is None:
        center = (m // 2, n // 2, o // 2)

    # Create coordinate grids relative to center
    z = torch.arange(m, device=device) - center[0]
    y = torch.arange(n, device=device) - center[1]
    x = torch.arange(o, device=device) - center[2]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Compute radial distances and integer bins
    r = torch.sqrt(xx**2 + yy**2 + zz**2)
    r_bin = r.round().long().flatten()
    data_flat = data.flatten()

    # Sum values per bin and count voxels per bin
    max_r = int(r_bin.max().item()) + 1
    sum_bin = torch.bincount(r_bin, weights=data_flat, minlength=max_r)
    count_bin = torch.bincount(r_bin, minlength=max_r)

    # Avoid division by zero
    radialprofile = sum_bin / count_bin.clamp(min=1)

    if return_r:
        return torch.arange(max_r, device=device), radialprofile
    else:
        return radialprofile


def compute_nps_1d(images: torch.Tensor) -> torch.Tensor:
    """
    Estimate the 1D radial noise power spectrum from a batch of images.

    Computed as the mean radial profile of ``|FFT2(images)|^2``, averaged
    over the batch. In low-SNR regimes (typical cryo-EM), the total power
    spectrum approximates the noise power spectrum.

    Parameters
    ----------
    images : torch.Tensor
        Batch of real images, shape (N, H, W).

    Returns
    -------
    nps_1d : torch.Tensor
        1D radial NPS, shape (R,), indexed by integer pixel radius from
        the DC component. R = max radial distance from center + 1.
    """
    N, H, W = images.shape

    # Power spectrum with DC shifted to center
    F_imgs = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))  # (N, H, W)
    mean_power = (F_imgs.abs() ** 2).mean(dim=0)  # (H, W)

    return radial_profile_2d(mean_power)


def compute_nps_2d(
    images: torch.Tensor,
    normalize: bool = True,
    zero_dc: bool = False,
) -> torch.Tensor:
    """
    Compute the 2D radial noise power spectrum from one or more images.

    Estimates the NPS as the mean power spectrum ``|FFT2(images)|^2``, radially
    averaged to enforce isotropy, then mapped back to 2D. Returns in rfft2
    half-plane format (H, W//2+1) for direct use in spectral-weighted losses.

    In low-SNR regimes (typical cryo-EM), the total power spectrum
    approximates the noise power spectrum. High-power (signal-dominated)
    frequencies will have large NPS values; low-power (noise-dominated)
    frequencies will have small values.

    The DC component (k=0) is always replaced by the value at k=1 to avoid
    the large mean-intensity spike dominating the spectrum. If zero_dc=True,
    the DC bin is set to zero instead, fully excluding it from any loss.

    Parameters
    ----------
    images : torch.Tensor
        Input images, shape (H, W) for a single image or (N, H, W) for a batch.
    normalize : bool, optional
        If True, normalize so the mean NPS value = 1. Default is True.
    zero_dc : bool, optional
        If True, set the DC bin to 0 rather than interpolating from k=1.
        Default is False.

    Returns
    -------
    nps_2d : torch.Tensor
        2D radial NPS, shape (H, W//2+1), matching torch.fft.rfft2 output.
    """
    if images.ndim == 2:
        images = images.unsqueeze(0)

    N, H, W = images.shape
    device = images.device

    # Mean power spectrum with DC shifted to center, shape (H, W)
    F_imgs = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
    mean_power = (F_imgs.abs() ** 2).mean(dim=0)

    # Radially average to 1D NPS
    nps_1d = radial_profile_2d(mean_power)

    # Handle DC bin: it is dominated by the squared mean intensity and carries
    # no structural information. Replace with k=1 value (smooth continuation)
    # or zero (full exclusion).
    nps_1d = nps_1d.clone()
    nps_1d[0] = 0.0 if zero_dc else nps_1d[1]

    # Map 1D NPS back to 2D in rfft2 half-plane layout
    kx_px = torch.fft.fftfreq(H, device=device) * H  # (H,)
    ky_px = torch.fft.rfftfreq(W, device=device) * W  # (W//2+1,)
    KX_px, KY_px = torch.meshgrid(kx_px, ky_px, indexing="ij")
    r_idx = torch.sqrt(KX_px**2 + KY_px**2).round().long().clamp(0, len(nps_1d) - 1)
    nps_2d = nps_1d[r_idx]  # (H, W//2+1)

    if normalize:
        nps_2d = nps_2d / nps_2d.mean().clamp(min=1e-10)

    return nps_2d


def compute_nps_3d(
    diff_volume: torch.Tensor,
    normalize: bool = True,
    zero_dc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the 3D noise power spectrum from a half-map difference volume.

    The difference (halfmap1 - halfmap2) cancels the signal, leaving 2x the
    noise. The NPS is estimated as ``|FFT3(diff)|^2 / 2``, radially averaged to
    enforce isotropy.

    Parameters
    ----------
    diff_volume : torch.Tensor
        Difference volume (halfmap1 - halfmap2), shape (D, H, W).
    normalize : bool, optional
        If True, normalize so the mean NPS value = 1. Default is True.
    zero_dc : bool, optional
        If True, set the DC bin to 0. Default is False.

    Returns
    -------
    nps_1d : torch.Tensor
        1D radial NPS, shape (R,), indexed by integer pixel radius.
    nps_3d : torch.Tensor
        3D radial NPS mapped to rfft3 half-plane layout (D, H, W//2+1),
        for direct use in spectral-weighted losses on volumes.
    """
    if diff_volume.ndim != 3:
        raise ValueError("Input must be a 3D volume (D, H, W).")
    D, H, W = diff_volume.shape
    device = diff_volume.device

    # Power spectrum of difference map, DC centered
    # Divide by 2 because diff = noise_1 - noise_2, so var(diff) = 2*var(noise)
    F = torch.fft.fftshift(torch.fft.fftn(diff_volume), dim=(-3, -2, -1))
    power = (F.abs() ** 2) / 2.0  # (D, H, W)

    # Build 3D radial index grid (centered)
    kx = torch.fft.fftfreq(D, device=device) * D  # (D,)
    ky = torch.fft.fftfreq(H, device=device) * H  # (H,)
    kz = torch.fft.fftfreq(W, device=device) * W  # (W,)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    # After fftshift, we need the centered radii
    KX_s = torch.fft.fftshift(KX)
    KY_s = torch.fft.fftshift(KY)
    KZ_s = torch.fft.fftshift(KZ)
    r_grid = torch.sqrt(KX_s**2 + KY_s**2 + KZ_s**2)  # (D, H, W)
    r_idx_full = r_grid.round().long()

    # Compute 1D radial NPS by averaging shells
    R = int(r_idx_full.max().item()) + 1
    nps_1d = torch.zeros(R, device=device)
    counts = torch.zeros(R, device=device)
    r_flat = r_idx_full.clamp(0, R - 1).reshape(-1)
    p_flat = power.reshape(-1)
    nps_1d.scatter_add_(0, r_flat, p_flat)
    counts.scatter_add_(0, r_flat, torch.ones_like(p_flat))
    nps_1d = nps_1d / counts.clamp(min=1)

    # Handle DC bin
    nps_1d = nps_1d.clone()
    nps_1d[0] = 0.0 if zero_dc else nps_1d[1]

    # Map 1D NPS back to rfft3 half-plane layout (D, H, W//2+1)
    kx_r = torch.fft.fftfreq(D, device=device) * D  # (D,)
    ky_r = torch.fft.fftfreq(H, device=device) * H  # (H,)
    kz_r = torch.fft.rfftfreq(W, device=device) * W  # (W//2+1,)
    KX_r, KY_r, KZ_r = torch.meshgrid(kx_r, ky_r, kz_r, indexing="ij")
    r_idx_rfft = torch.sqrt(KX_r**2 + KY_r**2 + KZ_r**2).round().long().clamp(0, R - 1)
    nps_3d = nps_1d[r_idx_rfft]  # (D, H, W//2+1)

    if normalize:
        mean_val = nps_3d.mean().clamp(min=1e-10)
        nps_1d = nps_1d / mean_val
        nps_3d = nps_3d / mean_val

    return nps_1d, nps_3d


@overload
def radial_profile_2d(
    data: torch.Tensor,
    center: tuple[float, float] | None = None,
    return_r: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def radial_profile_2d(
    data: torch.Tensor,
    center: tuple[float, float] | None = None,
    return_r: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def radial_profile_2d(
    data: torch.Tensor,
    center: tuple[float, float] | None = None,
    return_r: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the radial average (circular average) of a 2D image.

    Parameters
    ----------
    data : torch.Tensor
        2D tensor of shape (m, n).
    center : tuple of float, optional
        Center of the radial profile. Defaults to the integer center of the image (m//2, n//2).
    return_r : bool, optional
        If True, also return the radius indices. Default is False.

    Returns
    -------
    radialprofile : torch.Tensor
        Radial average.
    r : torch.Tensor, optional
        Radius indices if return_r=True.
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D tensor.")

    m, n = data.shape
    device = data.device

    # Default integer center
    if center is None:
        center = (m // 2, n // 2)

    # Create coordinate grids relative to center
    y = torch.arange(m, device=device) - center[0]
    x = torch.arange(n, device=device) - center[1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Compute radial distances and integer bins
    r = torch.sqrt(xx**2 + yy**2)
    r_bin = r.round().long().flatten()
    data_flat = data.flatten()

    # Sum values per bin and count pixels per bin
    max_r = int(r_bin.max().item()) + 1
    sum_bin = torch.bincount(r_bin, weights=data_flat, minlength=max_r)
    count_bin = torch.bincount(r_bin, minlength=max_r)

    # Avoid division by zero
    radialprofile = sum_bin / count_bin.clamp(min=1)

    if return_r:
        return torch.arange(max_r, device=device), radialprofile
    else:
        return radialprofile


def nearest_index(
    x_arr: torch.Tensor,
    y_arr: torch.Tensor,
    z_arr: torch.Tensor,
    x_coord: float,
    y_coord: float,
    z_coord: float,
) -> tuple[int, int, int]:
    """
    Find the nearest grid indices to specified coordinates.

    Parameters
    ----------
    x_arr : torch.Tensor
        1D array of x-coordinates defining the grid.
    y_arr : torch.Tensor
        1D array of y-coordinates defining the grid.
    z_arr : torch.Tensor
        1D array of z-coordinates defining the grid.
    x_coord : float
        Target x-coordinate.
    y_coord : float
        Target y-coordinate.
    z_coord : float
        Target z-coordinate.

    Returns
    -------
    xi : int
        Index in x_arr closest to x_coord.
    yi : int
        Index in y_arr closest to y_coord.
    zi : int
        Index in z_arr closest to z_coord.
    """
    xi = int(torch.argmin(torch.abs(x_arr - x_coord)).item())
    yi = int(torch.argmin(torch.abs(y_arr - y_coord)).item())
    zi = int(torch.argmin(torch.abs(z_arr - z_coord)).item())
    return xi, yi, zi


def ball3d(N: int, d: float) -> torch.Tensor:
    """
    Generates a 3D tensor with a filled-in ball,
    centered at the DC index corresponding to fftshift.

    Parameters
    ----------
    N : int
        Size of the 3D tensor (N x N x N).
    d : float
        Diameter of the ball.

    Returns
    -------
    ball : torch.Tensor
        3D tensor with ones inside the ball, zeros outside.
    """
    x = torch.arange(N)
    y = torch.arange(N)
    z = torch.arange(N)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    center = N // 2  # aligns with DC after fftshift
    r2 = (X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2

    radius = d / 2
    ball = (r2 <= radius**2).float()
    return ball


def disk2d(N: int, d: float) -> torch.Tensor:
    """
    Generate a 2D binary mask of a filled disk.

    Creates an NxN tensor with 1s inside a centered disk of diameter d
    and 0s outside. Origin is at index N//2 (consistent with fftshift convention).

    Parameters
    ----------
    N : int
        Size of the output grid in pixels (N x N).
    d : float
        Diameter of the disk in pixels.

    Returns
    -------
    disk : torch.Tensor
        Binary mask with shape (N, N). Values are 1.0 inside the disk
        and 0.0 outside.
    """
    x = torch.arange(N)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    center = N // 2
    r2 = (X - center) ** 2 + (Y - center) ** 2
    disk = (r2 <= (d / 2) ** 2).float()
    return disk


def fourier_crop(
    images: torch.Tensor,
    current_pixel_size: float,
    target_pixel_size: float,
) -> tuple[torch.Tensor, float]:
    """
    Resample 2D image(s) via Fourier-space center cropping.

    Downsamples from smaller pixel size (finer resolution) to larger
    pixel size (coarser resolution) by FFT → center crop → IFFT.
    Preserves integrated density (sum × pixel_size²) via physics-correct
    scaling.

    Parameters
    ----------
    images : torch.Tensor
        Shape (H, W) for single image or (N, H, W) for batch.
        Real-valued, any dtype.
    current_pixel_size : float
        Current pixel size (Å or any consistent unit).
    target_pixel_size : float
        Desired pixel size (must be ≥ current_pixel_size).

    Returns
    -------
    resampled : torch.Tensor
        Downsampled image(s), same shape as input but with smaller
        spatial dimensions.
    actual_pixel_size : float
        Achieved pixel size (may differ slightly due to rounding).

    Raises
    ------
    ValueError
        If target_pixel_size < current_pixel_size (upsampling not supported).

    Examples
    --------
    >>> img = torch.randn(512, 512)
    >>> resampled, actual_ps = fourier_crop(img, 0.5, 1.0)
    >>> resampled.shape  # downsampled to ~256x256
    torch.Size([256, 256])
    """
    if target_pixel_size < current_pixel_size:
        raise ValueError(
            f"target_pixel_size ({target_pixel_size}) must be >= "
            f"current_pixel_size ({current_pixel_size}). "
            "Upsampling not supported."
        )

    # Handle single vs batch
    if images.ndim == 2:
        images = images.unsqueeze(0)
        squeeze_output = True
    elif images.ndim == 3:
        squeeze_output = False
    else:
        raise ValueError(
            f"images must be 2D (H, W) or 3D (N, H, W), got shape {images.shape}"
        )

    N, H, W = images.shape

    # Compute scaling and output size
    scale = current_pixel_size / target_pixel_size

    if abs(scale - 1.0) < 1e-7:
        # No resampling needed
        result = images.squeeze(0) if squeeze_output else images
        return result, current_pixel_size

    output_size = int(round(H * scale))

    # FFT with DC at center
    images_f = fft2(images, shift=True)

    # Center crop in Fourier space
    start_idx = (H - output_size) // 2
    end_idx = start_idx + output_size
    images_f_cropped = images_f[:, start_idx:end_idx, start_idx:end_idx]

    # IFFT back to real space
    resampled = ifft2(images_f_cropped, shift=True).real

    # Scale to conserve integrated density: multiply by scale²
    scale_factor = scale**2
    resampled = resampled * scale_factor

    # Remove batch dimension if input was single image
    if squeeze_output:
        resampled = resampled.squeeze(0)

    # Compute actual achieved pixel size
    actual_pixel_size = current_pixel_size * H / output_size

    return resampled, actual_pixel_size


def downsample(
    images: torch.Tensor, bin_factor: int = 2, method: str = "fft"
) -> torch.Tensor:
    """
    Downsample images using FFT or average pooling.

    Parameters
    ----------
    images : torch.Tensor
        Input images.
    bin_factor : int, optional
        Binning factor. Default is 2.
    method : str, optional
        Downsampling method ('fft' or 'avgpool'). Default is 'fft'.

    Returns
    -------
    images_bin : torch.Tensor
        Downsampled images.
    """
    if method == "fft":
        N = images.shape[-1]
        n = N // bin_factor
        images_bin = ifft2(
            fft2(images, shift=True)[
                :,
                N // 2 - n // 2 : N // 2 - n // 2 + n,
                N // 2 - n // 2 : N // 2 - n // 2 + n,
            ],
            shift=True,
        ).real
    elif method == "avgpool":
        avgpool = torch.nn.AvgPool2d(bin_factor, stride=bin_factor)
        images_bin = avgpool(images) * bin_factor**2
    return images_bin


def centered_pad(X: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    """
    Pad a tensor to a target shape, symmetrically.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor.
    target_shape : Sequence of int
        Target shape for padding.

    Returns
    -------
    padded : torch.Tensor
        Padded tensor.
    """
    pad = []
    for size, tgt in zip(reversed(X.shape), reversed(target_shape)):
        diff = tgt - size
        pad.extend([diff // 2, diff - diff // 2])
    return F.pad(X, pad)


def pad_to_common_shape(
    A: torch.Tensor, B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad two tensors to a common shape.

    Parameters
    ----------
    A, B : torch.Tensor
        Input tensors.

    Returns
    -------
    padded_A, padded_B : torch.Tensor
        Padded tensors.
    """
    target = [max(a, b) for a, b in zip(A.shape, B.shape)]
    return centered_pad(A, target), centered_pad(B, target)


def compute_nz(base_nz: int, ice_thickness: float | None, pixel_size: float) -> int:
    """
    Number of Z slices given an optional ice thickness.

    Returns ``base_nz`` unchanged if ice thickness is None or smaller than the
    base volume depth; otherwise uses ice thickness to determine the depth.

    Parameters
    ----------
    base_nz : int
        Depth of the particle volume in slices.
    ice_thickness : float or None
        Total ice thickness in Å. None means no ice padding.
    pixel_size : float
        Voxel size in Å.

    Returns
    -------
    nz : int
        Number of Z slices to use.
    """
    if ice_thickness is None or ice_thickness < base_nz * pixel_size:
        return base_nz
    return int(ice_thickness // pixel_size)


def pad_volume(
    V: torch.Tensor,
    nxy: int,
    nz: int,
    ice_thickness: float | None,
    pad_fft: bool,
    xy_pad_mode: str = "constant",
) -> torch.Tensor:
    """
    Pad a potential volume in Z and/or XY.

    Z-padding extends the volume to ``nz`` slices (zeros added symmetrically)
    when ice thickness is set.  XY-padding adds ``nxy // 2`` pixels on each
    side for FFT antialiasing.

    Parameters
    ----------
    V : torch.Tensor
        Volume of shape (B, Z, Y, X).
    nxy : int
        Unpadded image size in pixels.
    nz : int
        Target Z depth after padding.
    ice_thickness : float or None
        If not None, Z-padding is applied.
    pad_fft : bool
        If True, XY-padding is applied.
    xy_pad_mode : str, optional
        Padding mode for XY axes. Default 'constant'.

    Returns
    -------
    V : torch.Tensor
        Padded volume.
    """
    if ice_thickness is not None:
        zpad_px = nz - nxy
        V = F.pad(
            V,
            (0, 0, 0, 0, zpad_px // 2, nz - zpad_px // 2 - V.shape[1]),
            mode="constant",
        )
    if pad_fft:
        V = F.pad(
            V,
            (nxy // 2, nxy // 2, nxy // 2, nxy // 2, 0, 0),
            mode=xy_pad_mode,
        )
    return V


def radial_symmetrize(
    data: torch.Tensor,
    center: float | None = None,
    output_ndim: int | None = None,
    return_size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Radially/spherically average a 2D or 3D tensor and map it back.

    Parameters
    ----------
    data : torch.Tensor
        (N, N) or (N, N, N) input tensor.
    center : float, optional
        Center used when computing the radial profile from the square/cubic
        input. Default is N // 2.
    output_ndim : {2, 3}, optional
        Output dimensionality. Defaults to match the input.
        If the input is 2D and ``output_ndim=3``, the 1D radial profile
        computed from the 2D image is mapped onto a volume via spherical
        distances from the same center.
    return_size : tuple[int, ...], optional
        Shape of the output tensor, e.g. ``(512, 256, 256)``.  The number of
        dimensions must match ``output_ndim`` (or ``data.ndim`` when
        ``output_ndim`` is not given).

        Each axis is scaled so that its own Nyquist pixel maps to the same
        radial-profile index as every other axis, i.e. the profile index for
        output pixel ``(iz, iy, ix)`` is

            ``sqrt((N/Oz*(iz-cz))^2 + (N/Oy*(iy-cy))^2 + (N/Ox*(ix-cx))^2)``

        This preserves equal Nyquist frequency along all axes, matching a
        real-space volume with uniform voxel size.  When ``return_size`` is
        omitted (or cubic) the formula reduces to plain Euclidean distance,
        identical to the previous behaviour.

    Returns
    -------
    torch.Tensor
        Radially/spherically symmetrised tensor with shape ``return_size`` (if
        provided) or ``(N,) * output_ndim`` otherwise.
    """
    if data.ndim not in (2, 3):
        raise ValueError("Input must be a 2D or 3D tensor.")
    N = data.shape[0]
    if not all(s == N for s in data.shape):
        raise ValueError(f"Input must be square/cubic, got shape {tuple(data.shape)}.")

    if output_ndim is None:
        output_ndim = data.ndim
    if output_ndim not in (2, 3):
        raise ValueError("output_ndim must be 2 or 3.")
    if data.ndim == 3 and output_ndim == 2:
        raise ValueError("Reducing a 3D input to a 2D output is not supported.")

    if return_size is not None and len(return_size) != output_ndim:
        raise ValueError(
            f"return_size has {len(return_size)} dimensions but output_ndim={output_ndim}."
        )

    device = data.device
    if center is None:
        center = N // 2

    if data.ndim == 2:
        radial_mean = radial_profile_2d(data, center=(center, center))
    else:
        radial_mean = radial_profile_3d(data, center=(center, center, center))

    out_shape: tuple[int, ...] = (
        return_size if return_size is not None else (N,) * output_ndim
    )
    out_centers = tuple(s // 2 for s in out_shape)
    # Per-axis scale: maps each axis's Nyquist pixel to the same profile index.
    scales = tuple(float(N) / s for s in out_shape)

    if output_ndim == 2:
        oh, ow = out_shape
        cy, cx = out_centers
        sy, sx = scales
        y, x = torch.meshgrid(
            torch.arange(oh, device=device),
            torch.arange(ow, device=device),
            indexing="ij",
        )
        r_int = torch.sqrt((sx * (x - cx)) ** 2 + (sy * (y - cy)) ** 2).round().long()
    else:
        oz, oy, ox = out_shape
        cz, cy, cx = out_centers
        sz, sy, sx = scales
        z, y, x = torch.meshgrid(
            torch.arange(oz, device=device),
            torch.arange(oy, device=device),
            torch.arange(ox, device=device),
            indexing="ij",
        )
        r_int = (
            torch.sqrt(
                (sz * (z - cz)) ** 2 + (sy * (y - cy)) ** 2 + (sx * (x - cx)) ** 2
            )
            .round()
            .long()
        )

    r_int = r_int.clamp(0, len(radial_mean) - 1)
    return radial_mean[r_int]


def clip_insert_bounds(
    center: Sequence[float],
    local_shape: Sequence[int],
    volume_shape: Sequence[int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    """
    Compute the destination/source slices for inserting a small array
    centered at `center` into a larger array, clipping at the boundaries.

    Shared low-level geometry for "stamp a local array into a big one at an
    arbitrary center, cropping whatever falls outside" -- used by both
    crowding.py (batched, centered coordinates, additive merge) and
    specimen/cryoet.py (one array at a time, absolute coordinates, max
    merge). Only the index arithmetic lives here; callers own the actual
    merge (`+=`, `torch.maximum`, ...) since that differs by use case.

    Parameters
    ----------
    center : sequence of float
        Index (one per axis) in the destination array where the local
        array's center voxel (``local_shape[i] // 2``) should land. Rounded
        to the nearest integer.
    local_shape : sequence of int
        Shape of the array being inserted.
    volume_shape : sequence of int
        Shape of the destination array.

    Returns
    -------
    tuple of (dst_slices, src_slices), or None
        `dst_slices`/`src_slices` are tuples of `slice`, one per axis, ready
        to index the destination/source arrays directly (e.g.
        ``volume[dst_slices] += local[src_slices]``). None if the local
        array falls entirely outside the volume's bounds.
    """
    dst_slices = []
    src_slices = []
    for c, lp, vp in zip(center, local_shape, volume_shape):
        half = lp // 2
        d0 = int(round(c)) - half
        d1 = d0 + lp
        d0_clip, d1_clip = max(d0, 0), min(d1, vp)
        if d1_clip <= d0_clip:
            return None
        dst_slices.append(slice(d0_clip, d1_clip))
        src_slices.append(slice(d0_clip - d0, lp - (d1 - d1_clip)))
    return tuple(dst_slices), tuple(src_slices)


def center_crop(
    x: torch.Tensor, size: int | tuple[int, ...], dim: int | Sequence[int] | None = None
) -> torch.Tensor:
    """
    Center crop a tensor along the specified axes (supports negative axes).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape
    size : int or tuple of int
        Desired crop size. If int, all axes in `dim` use the same size.
        If tuple, length must match number of axes in `dim`.
    dim : int, sequence of int, or None
        Axes along which to crop. Can be negative. Defaults to all dimensions.

    Returns
    -------
    torch.Tensor
        Center-cropped tensor
    """
    # default to all dimensions if not specified
    if dim is None:
        dim = list(range(x.ndim))
    # normalize dim to list
    elif isinstance(dim, int):
        dim = [dim]
    dim = [d + x.ndim if d < 0 else d for d in dim]  # handle negative axes

    # normalize size to list
    if isinstance(size, int):
        crop_size = [size] * len(dim)
    else:
        if len(size) != len(dim):
            raise ValueError(
                f"Length of size {len(size)} must match number of dims {len(dim)}"
            )
        crop_size = list(size)

    slices = [slice(None)] * x.ndim  # default: keep all elements
    for d, cs in zip(dim, crop_size):
        L = x.shape[d]
        if cs > L:
            raise ValueError(f"Crop size {cs} is larger than axis {d} length {L}")
        start = (L - cs) // 2
        slices[d] = slice(start, start + cs)

    return x[tuple(slices)]


def tile_volume_from_blocks(
    blocks: torch.Tensor,
    target_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Tile a bank of 3-D blocks into a larger volume with random augmentation per tile.

    Each placed tile receives an independent random roll, flip, and 90°-multiple
    rotation before insertion, breaking periodicity that would otherwise create
    visible seams at block boundaries.

    Parameters
    ----------
    blocks : torch.Tensor
        Pre-generated block bank, shape ``(N_blocks, S, S, S)``. Blocks must be cubic.
    target_shape : tuple of int
        Desired output shape ``(N_batch, A, B, C)``.

    Returns
    -------
    torch.Tensor
        Assembled volume cropped to ``target_shape``, shape ``(N_batch, A, B, C)``.
    """
    N_blocks, block_size, _, _ = blocks.shape
    N_batch, A, B, C = target_shape

    batch_volumes = []
    for _ in range(N_batch):
        n_a = (A + block_size - 1) // block_size
        n_b = (B + block_size - 1) // block_size
        n_c = (C + block_size - 1) // block_size

        tile_idx = torch.randint(0, N_blocks, (n_a, n_b, n_c))

        a_slices = []
        for i in range(n_a):
            b_slices = []
            for j in range(n_b):
                c_slices = []
                for k in range(n_c):
                    blk = blocks[tile_idx[i, j, k]].clone()

                    # Random roll along all three axes
                    shifts = (
                        int(torch.randint(0, block_size, (1,)).item()),
                        int(torch.randint(0, block_size, (1,)).item()),
                        int(torch.randint(0, block_size, (1,)).item()),
                    )
                    blk = torch.roll(blk, shifts=shifts, dims=(0, 1, 2))

                    # Random flip along each axis
                    for dim in (0, 1, 2):
                        if torch.rand(1).item() < 0.5:
                            blk = torch.flip(blk, dims=(dim,))

                    # Random 90° rotations in each plane
                    for d0, d1 in ((0, 1), (0, 2), (1, 2)):
                        k_rot = int(torch.randint(0, 4, (1,)).item())
                        blk = torch.rot90(blk, k=k_rot, dims=(d0, d1))

                    c_slices.append(blk)
                b_slices.append(torch.cat(c_slices, dim=2))
            a_slices.append(torch.cat(b_slices, dim=1))

        batch_volumes.append(torch.cat(a_slices, dim=0))

    return torch.stack(batch_volumes, dim=0)[:, :A, :B, :C]


def tile_volume_from_blocks_blended(
    blocks: torch.Tensor,
    target_shape: tuple[int, int, int, int],
    overlap_frac: float = 0.5,
    conserve_sum: bool = True,
) -> torch.Tensor:
    """
    Tile a bank of 3-D blocks into a larger volume using overlap-add blending.

    Same per-tile random roll/flip/90-degree-rotation augmentation as
    ``tile_volume_from_blocks``, but adjacent blocks are cross-faded with a
    raised-cosine taper over ``overlap_frac * block_size`` instead of being
    concatenated edge-to-edge. ``tile_volume_from_blocks`` abuts blocks with a
    hard edge; each block tiles seamlessly with itself (blocks are generated
    with periodic boundary conditions), but two different blocks placed side
    by side don't share matching boundary values, and that mismatch sits on
    the regular block-size lattice as an axis-aligned artifact in Fourier
    space. Blending removes the hard discontinuity at the cost of a small,
    local reduction in high-frequency content within the overlap band.

    Blending two uncorrelated blocks with weights that sum to 1 is unbiased
    in expectation, but any single realization tends to read slightly low:
    features that land in the transition band get scaled by w<1 on both
    sides with no compensating boost. If ``conserve_sum`` is set, each
    assembled volume is rescaled so its total exactly matches
    ``blocks.mean()`` times the number of output voxels — the expected total
    for that many voxels of the source block distribution.

    Parameters
    ----------
    blocks : torch.Tensor
        Pre-generated block bank, shape ``(N_blocks, S, S, S)``. Blocks must be cubic.
    target_shape : tuple of int
        Desired output shape ``(N_batch, A, B, C)``.
    overlap_frac : float, optional
        Fraction of the block size used as the crossfade region at each
        edge. 0.5 gives a classic constant-overlap-add (Hann) taper. Default
        is 0.5.
    conserve_sum : bool, optional
        If True (default), rescale each output volume so its total sum
        matches the expected total implied by the source blocks' mean.

    Returns
    -------
    torch.Tensor
        Assembled volume, shape ``(N_batch, A, B, C)``.
    """
    N_blocks, S, _, _ = blocks.shape
    N_batch, A, B, C = target_shape
    device = blocks.device
    ov = max(1, int(S * overlap_frac))
    stride = S - ov

    # Bin-centered sample points (not linspace's inclusive endpoints): avoids a
    # true-zero weight landing on a real voxel, which would zero out both sides
    # of a seam at once (degenerates badly for small ov, e.g. ov=1).
    theta = (torch.arange(ov, device=device) + 0.5) / ov * (torch.pi / 2)
    ramp = torch.sin(theta) ** 2
    w1d = torch.ones(S, device=device)
    w1d[:ov] = ramp
    w1d[S - ov :] = ramp.flip(0)
    window3d = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]

    def positions(total: int) -> list[int]:
        pos = [0]
        while pos[-1] + S < total:
            pos.append(pos[-1] + stride)
        return pos

    pa, pb, pc = positions(A), positions(B), positions(C)
    pad_shape = (pa[-1] + S, pb[-1] + S, pc[-1] + S)
    target_sum = blocks.mean() * (A * B * C)

    batch_volumes = []
    for _ in range(N_batch):
        acc = torch.zeros(pad_shape, device=device)
        wsum = torch.zeros(pad_shape, device=device)
        for i in pa:
            for j in pb:
                for k in pc:
                    idx = int(torch.randint(0, N_blocks, (1,)).item())
                    blk = blocks[idx].clone()

                    shifts = (
                        int(torch.randint(0, S, (1,)).item()),
                        int(torch.randint(0, S, (1,)).item()),
                        int(torch.randint(0, S, (1,)).item()),
                    )
                    blk = torch.roll(blk, shifts=shifts, dims=(0, 1, 2))

                    for dim in (0, 1, 2):
                        if torch.rand(1).item() < 0.5:
                            blk = torch.flip(blk, dims=(dim,))

                    for d0, d1 in ((0, 1), (0, 2), (1, 2)):
                        k_rot = int(torch.randint(0, 4, (1,)).item())
                        blk = torch.rot90(blk, k=k_rot, dims=(d0, d1))

                    acc[i : i + S, j : j + S, k : k + S] += blk * window3d
                    wsum[i : i + S, j : j + S, k : k + S] += window3d

        vol = acc / wsum.clamp_min(1e-8)
        vol = vol[:A, :B, :C]

        if conserve_sum:
            vol = vol * (target_sum / vol.sum().clamp_min(1e-8))

        batch_volumes.append(vol)

    return torch.stack(batch_volumes, dim=0)
