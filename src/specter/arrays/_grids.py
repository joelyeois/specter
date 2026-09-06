"""Coordinate/frequency grid construction and simple filled-shape masks."""

from __future__ import annotations

from typing import Sequence

import torch
from specter.options import GridConvention


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
    convention: GridConvention = "relion",
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
    convention: GridConvention = "relion",
    device: str | torch.device = "cpu",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Constructs the xyz coordinate arrays and meshgrids. Meshgrid indexing yields
    volume[z_i, y_i, x_i] convention which matches cryo-EM software.

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
    convention: GridConvention = "relion",
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
    convention: GridConvention = "relion",
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


def real_to_kgrid_3d(
    n_xyz: int | Sequence[int],
    d_xyz: float | Sequence[float],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Construct the 3D frequency-space radial grid conjugate to a real-space grid.

    Parameters
    ----------
    n_xyz : int or Sequence of int
        Number of pixels along x, y, z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or Sequence of float
        Real-space pixel spacing along x, y, z (Å). If float, assumes
        dx=dy=dz=d_xyz.
    device : str or torch.device, optional
        Device for the tensor. Default is 'cpu'.

    Returns
    -------
    KR : torch.Tensor
        3D tensor of radial spatial frequencies, shape (nz, ny, nx).

    Notes
    -----
    Takes the grid shape and real-space spacing directly, rather than
    inferring spacing from a real-space radial-*distance* grid: for a
    genuine 3D radial magnitude `R = sqrt(x²+y²+z²)`, `R[1,0,0] - R[0,0,0]`
    does not equal the pixel spacing (adjacent radial magnitudes differ by
    more than one axis's step unless the other two axes sit exactly at
    zero, which `radial_grid_3d`'s conventions don't guarantee) — an
    earlier version of this function tried that and silently produced the
    wrong frequency grid for every caller.
    """
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    if isinstance(d_xyz, (int, float)):
        dx = dy = dz = float(d_xyz)
    else:
        dx, dy, dz = d_xyz

    # frequency axes
    kx = torch.fft.fftfreq(nx, dx, device=device)
    ky = torch.fft.fftfreq(ny, dy, device=device)
    kz = torch.fft.fftfreq(nz, dz, device=device)

    # 3D frequency grids (nz, ny, nx), matching radial_grid_3d's convention
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
    KR = torch.sqrt(KX**2 + KY**2 + KZ**2)

    return KR


def ball3d(n: int, d: float) -> torch.Tensor:
    """
    Generates a 3D tensor with a filled-in ball,
    centered at the DC index corresponding to fftshift.

    Parameters
    ----------
    n : int
        Size of the 3D tensor (n x n x n).
    d : float
        Diameter of the ball.

    Returns
    -------
    ball : torch.Tensor
        3D tensor with ones inside the ball, zeros outside.
    """
    x = torch.arange(n)
    y = torch.arange(n)
    z = torch.arange(n)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    center = n // 2  # aligns with DC after fftshift
    r2 = (X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2

    radius = d / 2
    ball = (r2 <= radius**2).float()
    return ball


def disk2d(n: int, d: float) -> torch.Tensor:
    """
    Generate a 2D binary mask of a filled disk.

    Creates an n x n tensor with 1s inside a centered disk of diameter d
    and 0s outside. Origin is at index n//2 (consistent with fftshift convention).

    Parameters
    ----------
    n : int
        Size of the output grid in pixels (n x n).
    d : float
        Diameter of the disk in pixels.

    Returns
    -------
    disk : torch.Tensor
        Binary mask with shape (n, n). Values are 1.0 inside the disk
        and 0.0 outside.
    """
    x = torch.arange(n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    center = n // 2
    r2 = (X - center) ** 2 + (Y - center) ** 2
    disk = (r2 <= (d / 2) ** 2).float()
    return disk
