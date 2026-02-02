import torch
from .fft_tools import fft2, ifft2


def kgrid_1d(n, dx, device="cpu"):
    """
    Constructs a 1D k-grid.

    Parameters
    ----------
    n : int
        Number of pixels.
    dx : float
        Pixel size.

    Returns
    -------
    k : 1d tensor
    """
    kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx, device=device))
    return kx


def kgrid_2d(n_xy, d_xy, device="cpu"):
    """Constructs the kx, ky meshgrids.

    Parameters
    ----------
    n_xy : int or array-like
        Number of pixels along x,y. If int, assumes nx=ny=n_xy.
    d_xy : float or array-like
        Pixel length along x,y. If float, assumes dx=dy=d_xy.

    Returns
    -------
    KX,KY : 2d tensors
        kx,ky meshgrids.
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


def kgrid_3d(n_xyz, d_xyz, device="cpu"):
    """Constructs the kx, ky, kz meshgrids.

    Parameters
    ----------
    n_xyz : int or array-like
        Number of pixels along x,y,z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or array-like
        Pixel length along x,y,z. If float, assumes dx=dy=dz=d_xyz.

    Returns
    -------
    KX, KY, KZ : 2d tensors
        kx,ky, kz meshgrids.
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


def radial_kgrid_2d(n_xy, d_xy, device="cpu"):
    """
    Construct 2D radial frequency grid.

    Parameters
    ----------
    n_xy : int or array-like
        Number of pixels along x,y. If int, assumes nx=ny=n_xy.
    d_xy : float or array-like
        Pixel length along x,y. If float, assumes dx=dy=d_xy.
    device : str, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    K_radial : torch.Tensor
        2D radial frequencies, shape (ny, nx). Values represent the
        distance from the origin in frequency space.
    """
    _, _, KX, KY = kgrid_2d(n_xy, d_xy, device)
    return torch.sqrt(KX**2 + KY**2)


def radial_kgrid_3d(n_xyz, d_xyz, device="cpu"):
    """
    Construct 3D radial frequency grid.

    Parameters
    ----------
    n_xyz : int or array-like
        Number of pixels along x,y,z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or array-like
        Pixel length along x,y,z. If float, assumes dx=dy=dz=d_xyz.
    device : str, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    K_radial : torch.Tensor
        3D radial frequencies, shape (nz, ny, nx). Values represent the
        distance from the origin in 3D frequency space.
    """
    _, _, _, KX, KY, KZ = kgrid_3d(n_xyz, d_xyz, device)
    return torch.sqrt(KX**2 + KY**2 + KZ**2)


def grid_1d(n, dx, convention="relion", device="cpu"):
    """
    Constructs a 1D grid. The coordinate grid convention only matters if the number of
    pixels is even.

    Parameters
    ----------
    n : int
        Number of pixels.
    dx : float
        Pixel size.
    convention : str
        Determines location of origin (0). If 'relion', origin is located at
        index [n//2], which means coordinates are not symmetric about
        the origin for even number of pixels. If 'torch', forces coordinates to be
        symmetric about the origin, which means for even grids, there is no index
        for (0).

    Returns
    -------
    x : 1d tensor
    """
    if convention == "relion":
        x = (torch.arange(n, device=device) - n // 2) * dx
    elif convention == "torch":
        x = (torch.arange(n, device=device) - (n - 1) / 2) * dx
    return x


def grid_2d(n_xy, d_xy, convention="relion", device="cpu"):
    """Constructs the xy coordinate arrays and meshgrids. Meshgrid indexing yields
    area[y_i, x_i] convention which matches cryo-EM software.

    The coordinate grid convention only matters if the number of pixels is even.

    Parameters
    ----------
    n_xy : int or array-like
        Number of pixels along x,y. If int, assumes nx=ny=n_xy.
    d_xy : float or array-like
        Pixel length along x,y. If float, assumes dx=dy=d_xy.
    convention : str
        Determines location of origin (0,0). If 'relion', origin is located at
        index [ny//2, nx//2], which means coordinates are not symmetric about
        the origin for even number of pixels. If 'torch', forces coordinates to be
        symmetric about the origin, which means for even grids, there is no index
        for (0,0).

    Returns
    -------
    x,y : 1d tensors
        x,y coordinates.
    X,Y : 2d tensors
        x,y meshgrids.
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


def grid_3d(n_xyz, d_xyz, convention="relion", device="cpu"):
    """Constructs the xyz coordinate arrays and meshgrids. Meshgrid indexing yields
    vol[z_i, y_i, x_i] convention which matches cryo-EM software.

    The coordinate grid convention only matters if the number of pixels is even.

    Parameters
    ----------
    n_xyz : int or array-like
        Number of pixels along x,y,z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or array-like
        Pixel length along x,y,z. If float, assumes dx=dy=dz=d_xyz.
    convention : str
        Determines location of origin (0,0,0). If 'relion', origin is located at
        index [nz//2, ny//2, nx//2], which means coordinates are not symmetric about
        the origin for even number of pixels. If 'torch', forces coordinates to be
        symmetric about the origin, which means for even grids, there is no index
        for (0,0,0).

    Returns
    -------
    x,y,z : 1d tensors
        x,y,z coordinates.
    X,Y,Z : 3d tensors
        x,y,z meshgrids.
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


def radial_grid_2d(n_xy, d_xy, convention="relion", device="cpu"):
    """
    Construct 2D radial coordinate grid.

    Parameters
    ----------
    n_xy : int or array-like
        Number of pixels along x,y. If int, assumes nx=ny=n_xy.
    d_xy : float or array-like
        Pixel length along x,y. If float, assumes dx=dy=d_xy.
    convention : str, optional
        Grid origin convention ('relion' or 'torch'). Default is 'relion'.
    device : str, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    R : torch.Tensor
        2D radial distances from origin, shape (ny, nx).
    """
    _, _, X, Y = grid_2d(n_xy, d_xy, convention, device)
    return torch.sqrt(X**2 + Y**2)


def radial_grid_3d(n_xyz, d_xyz, convention="relion", device="cpu"):
    """
    Construct 3D radial coordinate grid.

    Parameters
    ----------
    n_xyz : int or array-like
        Number of pixels along x,y,z. If int, assumes nx=ny=nz=n_xyz.
    d_xyz : float or array-like
        Pixel length along x,y,z. If float, assumes dx=dy=dz=d_xyz.
    convention : str, optional
        Grid origin convention ('relion' or 'torch'). Default is 'relion'.
    device : str, optional
        Device for tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    R : torch.Tensor
        3D radial distances from origin, shape (nz, ny, nx).
    """
    _, _, _, X, Y, Z = grid_3d(n_xyz, d_xyz, convention, device)
    return torch.sqrt(X**2 + Y**2 + Z**2)


def real_to_kgrid_3d(R):
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
    kx = torch.fft.fftshift(torch.fft.fftfreq(nx, dx, device=device))
    ky = torch.fft.fftshift(torch.fft.fftfreq(ny, dy, device=device))
    kz = torch.fft.fftshift(torch.fft.fftfreq(nz, dz, device=device))

    # 3D frequency grids
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    KR = torch.sqrt(KX**2 + KY**2 + KZ**2)

    return KR


def voxelize_coordinates(coords, grid_shape, voxel_size, device=None):
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
    voxel_size : tuple of float
        Voxel size (dx, dy, dz).
    device : torch.device, optional
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


# def soft_voxelize_coordinates(coords, grid_shape, voxel_size, device=None):
#     """
#     Differentiable 3D soft voxelization using trilinear splatting.

#     Args:
#         coords: (N,3) tensor of atomic coordinates in physical units (x, y, z)
#         grid_shape: tuple of ints (nz, ny, nx)
#         voxel_size: tuple of floats (dx, dy, dz)
#         device: optional, torch device

#     Returns
#     -------
#     volume : (nz, ny, nx) tensor
#         Differentiable soft voxelized volume
#     """
#     if device is None:
#         device = coords.device
#     coords = coords.to(device)
#     nz, ny, nx = grid_shape
#     N = coords.shape[0]

#     values = torch.ones(N, device=device)

#     # Convert physical coordinates to voxel units
#     if isinstance(voxel_size, (int, float)):
#         voxel_size = torch.tensor([voxel_size] * 3, device=device)
#     else:
#         voxel_size = torch.tensor(voxel_size, device=device)
#     coords_voxel = coords / voxel_size  # (N,3)

#     # Shift coordinates so origin (0,0,0) is at floor division center
#     origin = torch.tensor(
#         [nx // 2, ny // 2, nz // 2], device=device, dtype=coords_voxel.dtype
#     )
#     coords_voxel_centered = coords_voxel + origin[None, :]  # (N,3)

#     # Reorder coords to z, y, x for indexing
#     coords_voxel_centered = coords_voxel_centered[:, [2, 1, 0]]  # (N,3)

#     # Floor and fractional part for trilinear weights
#     coords_floor = torch.floor(coords_voxel_centered).long()  # (N,3)
#     frac = coords_voxel_centered - coords_floor.float()  # (N,3)
#     dz, dy, dx = frac[:, 0], frac[:, 1], frac[:, 2]
#     z0, y0, x0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]

#     # 8 neighbor offsets
#     offsets = torch.tensor(
#         [
#             [0, 0, 0],
#             [0, 0, 1],
#             [0, 1, 0],
#             [0, 1, 1],
#             [1, 0, 0],
#             [1, 0, 1],
#             [1, 1, 0],
#             [1, 1, 1],
#         ],
#         device=device,
#     )

#     # Compute neighbor indices (N,8)
#     z_idx = z0[:, None] + offsets[None, :, 0]
#     y_idx = y0[:, None] + offsets[None, :, 1]
#     x_idx = x0[:, None] + offsets[None, :, 2]

#     # Trilinear weights (N,8)
#     w = (
#         ((1 - dz)[:, None] * (1 - offsets[None, :, 0])
#             + dz[:, None] * offsets[None, :, 0])
#         * ((1 - dy)[:, None] * (1 - offsets[None, :, 1])
#             + dy[:, None] * offsets[None, :, 1])
#         * ((1 - dx)[:, None] * (1 - offsets[None, :, 2])
#             + dx[:, None] * offsets[None, :, 2])
#     )
#     w = w * values[:, None]

#     # Mask out-of-bounds
#     mask = (
#         (z_idx >= 0)
#         & (z_idx < nz)
#         & (y_idx >= 0)
#         & (y_idx < ny)
#         & (x_idx >= 0)
#         & (x_idx < nx)
#     )
#     z_idx = z_idx[mask]
#     y_idx = y_idx[mask]
#     x_idx = x_idx[mask]
#     w = w[mask]

#     # Scatter-add into volume
#     volume = torch.zeros(nz, ny, nx, device=device)
#     volume.index_put_((z_idx, y_idx, x_idx), w, accumulate=True)

#     return volume


def soft_voxelize_coordinates(coords, grid_shape, voxel_size, device=None):
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
    voxel_size : float or tuple of float
        Voxel size. If float, assumes isotropic. If tuple, (dz, dy, dx).
    device : torch.device, optional
        Device for tensors. Default is None (uses coords device).

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


def soft_voxelize_xy_coordinates(coords, grid_shape, voxel_size, device=None):
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
    voxel_size : float or tuple of float
        Voxel size. If float, assumes isotropic. If tuple, (dz, dy, dx).
    device : torch.device, optional
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


def radial_profile_3d(data, center=None, return_r=False):
    """
    Compute the radial average of a 3D tensor.

    Parameters
    ----------
    data : torch.Tensor
        3D tensor of shape (m, n, o)
    center : tuple of floats, optional
        Center of the radial profile. Defaults to geometric center.
    return_r : bool
        If True, also return the radius indices.

    Returns
    -------
    radialprofile : torch.Tensor
        Radial average.
    r : torch.Tensor, optional
        Radius indices if return_r=True
    """

    m, n, o = data.shape
    device = data.device

    if center is None:
        center = (0, 0, 0)

    # create coordinate grids
    x = torch.arange(n, device=device) - n // 2 + center[0]
    y = torch.arange(m, device=device) - m // 2 + center[1]
    z = torch.arange(o, device=device) - o // 2 + center[2]
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # compute distances
    r = torch.sqrt(xx**2 + yy**2 + zz**2)
    r = r.round().long()  # integer bins

    # flatten
    r_flat = r.flatten()
    data_flat = data.flatten()

    # sum per bin
    max_r = r_flat.max().item() + 1
    tbin = torch.bincount(r_flat, weights=data_flat, minlength=max_r)
    nr = torch.bincount(r_flat, minlength=max_r)

    radialprofile = tbin / nr

    if return_r:
        return torch.arange(max_r, device=device), radialprofile
    else:
        return radialprofile


# def soft_voxelize_xy_coordinates(coords, grid_shape, voxel_size, device=None):
#     """
#     Semi-soft 3D voxelization:
#     - hard assignment along z (nearest slice)
#     - bilinear interpolation in x and y.

#     Out-of-bounds neighbors are excluded.

#     Args:
#         coords: (N,3) tensor of atomic coordinates (x, y, z)
#         grid_shape: tuple of ints (nz, ny, nx)
#         voxel_size: float or tuple of floats (dx, dy, dz)
#         device: optional torch.device

#     Returns
#     -------
#     volume : (nz, ny, nx) tensor
#         Semi-soft voxelized volume
#     """
#     if device is None:
#         device = coords.device
#     coords = coords.to(device)
#     nz, ny, nx = grid_shape
#     N = coords.shape[0]

#     values = torch.ones(N, device=device)

#     # Convert to voxel units
#     if isinstance(voxel_size, (int, float)):
#         voxel_size = torch.tensor([voxel_size]*3, device=device)
#     else:
#         voxel_size = torch.tensor(voxel_size, device=device)
#     coords_voxel = coords / voxel_size

#     # Shift origin to center
#     origin = torch.tensor([nx//2, ny//2, nz//2], device=device, dtype=coords_voxel.dtype)
#     coords_voxel_centered = coords_voxel + origin[None, :]

#     # Separate coordinates
#     x, y, z = coords_voxel_centered.T

#     # Hard assign Z
#     z_idx = torch.round(z).long()

#     # XY floor + fractional part for bilinear weights
#     x0 = torch.floor(x).long()
#     y0 = torch.floor(y).long()
#     dx = x - x0.float()
#     dy = y - y0.float()

#     # 4 neighbor offsets for XY bilinear
#     offsets = torch.tensor([[0,0],[0,1],[1,0],[1,1]], device=device)

#     x_idx = x0[:, None] + offsets[None,:,1]
#     y_idx = y0[:, None] + offsets[None,:,0]
#     z_idx_full = z_idx[:, None].repeat(1,4)

#     # Bilinear weights
#     w = ((1-dx)[:, None]*(1-offsets[None,:,1]) + dx[:, None]*offsets[None,:,1]) \
#         * ((1-dy)[:, None]*(1-offsets[None,:,0]) + dy[:, None]*offsets[None,:,0])
#     w = w * values[:, None]

#     # Mask out-of-bounds
#     mask = (z_idx_full>=0)&(z_idx_full<nz) & (y_idx>=0)&(y_idx<ny) & (x_idx>=0)&(x_idx<nx)
#     z_idx_full = z_idx_full[mask]
#     y_idx = y_idx[mask]
#     x_idx = x_idx[mask]
#     w = w[mask]

#     # Scatter
#     volume = torch.zeros(nz, ny, nx, device=device)
#     volume.index_put_((z_idx_full, y_idx, x_idx), w, accumulate=True)

#     return volume


def nearest_index(x_arr, y_arr, z_arr, x_coord, y_coord, z_coord):
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


def ball3d(N, d):
    """
    Generates a 3D tensor with a filled-in ball,
    centered at the DC index corresponding to fftshift.
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


def downsample(images, bin_factor=2, method="fft"):
    if method == "fft":
        N = images.shape[-1]
        n = N // bin_factor
        images_bin = ifft2(
            fft2(images)[
                :,
                N // 2 - n // 2 : N // 2 - n // 2 + n,
                N // 2 - n // 2 : N // 2 - n // 2 + n,
            ]
        ).real
    elif method == "avgpool":
        avgpool = torch.nn.AvgPool2d(bin_factor, stride=bin_factor)
        images_bin = avgpool(images) * bin_factor**2
    return images_bin
