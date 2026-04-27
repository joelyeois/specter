from __future__ import annotations

from typing import Sequence

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

    if isinstance(d_xy, (int, float)):
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

    if isinstance(d_xyz, (int, float)):
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

    if isinstance(d_xy, (int, float)):
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

    if isinstance(d_xyz, (int, float)):
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

    # Handle non-batch input
    batched_input = True
    if coords.ndim == 2:  # (N,3) -> add batch dimension
        coords = coords.unsqueeze(0)
        batched_input = False

    B, N, _ = coords.shape
    nz, ny, nx = grid_shape
    values = torch.ones(B, N, device=device)

    # Convert physical coordinates to voxel units
    if isinstance(voxel_size, (int, float)):
        voxel_size = torch.tensor([voxel_size] * 3, device=device)
    else:
        voxel_size = torch.tensor(voxel_size, device=device)
    coords_voxel = coords / voxel_size  # (B,N,3)

    # Shift coordinates so origin is at center
    origin = torch.tensor(
        [nx // 2, ny // 2, nz // 2], device=device, dtype=coords_voxel.dtype
    )
    coords_voxel_centered = coords_voxel + origin[None, None, :]  # (B,N,3)

    # Reorder to z,y,x
    coords_voxel_centered = coords_voxel_centered[..., [2, 1, 0]]

    # Floor and fractional part
    coords_floor = torch.floor(coords_voxel_centered).long()  # (B,N,3)
    frac = coords_voxel_centered - coords_floor.float()
    dz, dy, dx = frac[..., 0], frac[..., 1], frac[..., 2]
    z0, y0, x0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]

    # 8 neighbor offsets
    offsets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=device,
    )

    z_idx = z0[..., None] + offsets[None, None, :, 0]
    y_idx = y0[..., None] + offsets[None, None, :, 1]
    x_idx = x0[..., None] + offsets[None, None, :, 2]

    # Trilinear weights
    w = (
        (
            (1 - dz)[..., None] * (1 - offsets[None, None, :, 0])
            + dz[..., None] * offsets[None, None, :, 0]
        )
        * (
            (1 - dy)[..., None] * (1 - offsets[None, None, :, 1])
            + dy[..., None] * offsets[None, None, :, 1]
        )
        * (
            (1 - dx)[..., None] * (1 - offsets[None, None, :, 2])
            + dx[..., None] * offsets[None, None, :, 2]
        )
    )
    w = w * values[..., None]

    # Initialize volume
    volume = torch.zeros(B, nz, ny, nx, device=device)

    # Scatter-add per batch
    for b in range(B):
        if periodic:
            volume[b].index_put_(
                (z_idx[b] % nz, y_idx[b] % ny, x_idx[b] % nx),
                w[b],
                accumulate=True,
            )
        else:
            mask = (
                (z_idx[b] >= 0)
                & (z_idx[b] < nz)
                & (y_idx[b] >= 0)
                & (y_idx[b] < ny)
                & (x_idx[b] >= 0)
                & (x_idx[b] < nx)
            )
            volume[b].index_put_(
                (z_idx[b][mask], y_idx[b][mask], x_idx[b][mask]),
                w[b][mask],
                accumulate=True,
            )

    # Remove batch dimension if input was non-batch
    if not batched_input:
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

    # Ensure batch dimension
    if coords.ndim == 2:  # (N,3)
        coords = coords.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, N, _ = coords.shape
    nz, ny, nx = grid_shape

    # Convert voxel size
    if isinstance(voxel_size, (int, float)):
        voxel_size = torch.tensor([voxel_size] * 3, device=device)
    else:
        voxel_size = torch.tensor(voxel_size, device=device)

    # Shift origin to center
    origin = torch.tensor(
        [nx // 2, ny // 2, nz // 2], device=device, dtype=coords.dtype
    )

    volumes = torch.zeros(B, nz, ny, nx, device=device)
    offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device)

    for b in range(B):
        batch_coords = coords[b]
        values = torch.ones(N, device=device)

        coords_voxel = batch_coords / voxel_size
        coords_voxel_centered = coords_voxel + origin[None, :]

        # Extract x, y, z safely without .T
        x = coords_voxel_centered[:, 0]
        y = coords_voxel_centered[:, 1]
        z = coords_voxel_centered[:, 2]

        # Hard Z assignment
        z_idx = torch.round(z).long()

        # XY floor + fractional for bilinear
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        dx = x - x0.float()
        dy = y - y0.float()

        # Neighbor indices for bilinear
        x_idx = x0[:, None] + offsets[None, :, 1]
        y_idx = y0[:, None] + offsets[None, :, 0]
        z_idx_full = z_idx[:, None].repeat(1, 4)

        # Bilinear weights
        w = (
            (1 - dx)[:, None] * (1 - offsets[None, :, 1])
            + dx[:, None] * offsets[None, :, 1]
        ) * (
            (1 - dy)[:, None] * (1 - offsets[None, :, 0])
            + dy[:, None] * offsets[None, :, 0]
        )
        w = w * values[:, None]

        # Mask out-of-bounds
        mask = (
            (z_idx_full >= 0)
            & (z_idx_full < nz)
            & (y_idx >= 0)
            & (y_idx < ny)
            & (x_idx >= 0)
            & (x_idx < nx)
        )
        z_idx_full = z_idx_full[mask]
        y_idx = y_idx[mask]
        x_idx = x_idx[mask]
        w = w[mask]

        # Scatter values into volume
        volumes[b].index_put_((z_idx_full, y_idx, x_idx), w, accumulate=True)

    if squeeze_output:
        return volumes[0]
    return volumes


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
    max_r = r_bin.max().item() + 1
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

    Computed as the mean radial profile of |FFT2(images)|^2, averaged
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

    Estimates the NPS as the mean power spectrum |FFT2(images)|^2, radially
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
    noise. The NPS is estimated as |FFT3(diff)|^2 / 2, radially averaged to
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
    R = r_idx_full.max().item() + 1
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
    max_r = r_bin.max().item() + 1
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
    xi = torch.argmin(torch.abs(x_arr - x_coord))
    yi = torch.argmin(torch.abs(y_arr - y_coord))
    zi = torch.argmin(torch.abs(z_arr - z_coord))
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


def radial_symmetrize(image: torch.Tensor, center: float | None = None) -> torch.Tensor:
    """
    Radially average a 2D image and map it back to a 2D image.

    Parameters
    ----------
    image : torch.Tensor
        (N, N) input image.
    center : float, optional
        Center of the radial average. Default is N // 2.

    Returns
    -------
    image_ring : torch.Tensor
        (N, N) radially averaged image.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D tensor.")
    N, M = image.shape
    if N != M:
        raise ValueError(f"Image must be square, got shape ({N}, {M}).")

    device = image.device

    if center is None:
        center = N // 2

    # Compute 1D radial profile using the shared helper
    radial_mean = radial_profile_2d(image, center=(center, center))

    # Build matching integer radius grid to map back to 2D
    y, x = torch.meshgrid(
        torch.arange(N, device=device), torch.arange(N, device=device), indexing="ij"
    )
    r_int = torch.sqrt((x - center) ** 2 + (y - center) ** 2).round().long()
    r_int = r_int.clamp(0, len(radial_mean) - 1)

    return radial_mean[r_int]


def center_crop(
    x: torch.Tensor, size: int | tuple[int, ...], dim: int | Sequence[int]
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
    dim : int or sequence of int
        Axes along which to crop. Can be negative.

    Returns
    -------
    torch.Tensor
        Center-cropped tensor
    """
    # normalize dim to list
    if isinstance(dim, int):
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
