from . import atom
import torch
from scipy.special import kn
from tqdm.auto import tqdm
from .fft_tools import fftconvolve
import torch.nn.functional as F

def atomic_potential_2d(atomic_number, r_xy):
    """Returns the 2D projected atomic potential for a specific element given a
    2D grid of radial distances from the atom core. Kirkland C.20.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    r_xy : 2D tensor
        Distances from the atomic core in units of Ångstrom. r^2 = x^2 + y^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.

    Returns
    -------
    potential : tensor
        Atomic potential in units of V-Ångstrom, same shape as r_xy.
    """
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]
    c1 = 4 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi**2) * a0 * e

    # get scattering factors
    atom_params_dict = atom.get_atom_params_dict()
    P = torch.from_numpy(atom_params_dict[atomic_number]["params"])
    # tile scattering factors to match r_xy grid
    P = P[:, :, None, None].expand((4, 3) + r_xy.shape)

    s1 = c1 * torch.sum(P[0] * kn(0.0, 2 * torch.pi * r_xy * torch.sqrt(P[1])), 0)
    s2 = c2 * torch.sum(
        P[2] / P[3] * torch.exp(-(torch.pi**2) * (r_xy**2) / P[3]), 0
    )
    return s1 + s2


def atomic_potential_3d(atomic_number, r_xyz):
    """Returns the 3D atomic potential for a specific element and given a 3D grid
    of radial distances from the atom core. Kirkland C.19.

    Summing along the z-axes (or any other axes due to symmetry) should yield
    approximately the same results as atomic_potential_2d.

    Note: There is a singularity at r = 0 because the atomic nucleaus is essentially
    a point charge on this scale (~1e-5 Angstroms).

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    r_xyz : 3D tensor
        Distances from the atomic core in units of Ångstrom. r^2 = x^2 + y^2 + z^2.
        Assume equally spaced grid along x and y, i.e. dx = dy.

    Returns
    -------
    potential : tensor
        Atomic potential in units of V-Ångstrom, same shape as r_xyz.
    """
    device = r_xyz.device
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]
    c1 = 2 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

    # get scattering factors
    atom_params_dict = atom.get_atom_params_dict()
    P = torch.from_numpy(atom_params_dict[atomic_number]["params"])
    P = P.to(device)
    # tile scattering factors to match r_xy grid
    P = P[:, :, None, None, None].expand((4, 3) + r_xyz.shape)

    s1 = c1 * torch.sum(
        P[0] / r_xyz * torch.exp(-2 * torch.pi * r_xyz * torch.sqrt(P[1])), 0
    )
    s2 = c2 * torch.sum(
        P[2] * P[3] ** (-3 / 2) * torch.exp(-(torch.pi**2) * (r_xyz**2) / P[3]), 0
    )
    return s1 + s2

def atomic_potential_3d_fourier(atomic_number, k_xyz):
    """Returns the Fourier transformed 3D atomic potential for a specific element and 
    given a 3D grid of radial distances from the atom core. Kirkland C.15.

    Parameters
    ----------
    atomic_number : int
        Atomic number, Hyrdogen has number 1.
    k_xyz : 3D tensor
        Distances from the atomic core in units of Ångstrom. k^2 = kx^2 + ky^2 + kz^2.
        Assume equally spaced grid along kx and ky, i.e. dkx = dky.

    Returns
    -------
    potential : tensor
        Atomic potential in Fourier space in units of 1/V-Ångstrom, same shape as r_xyz.
    """
    device = k_xyz.device
    a0 = 0.529  # Bohr radius, [Angstrom]
    e = 14.4  # electron charge, [V-Angstrom]

    # get scattering factors
    atom_params_dict = atom.get_atom_params_dict()
    P = torch.from_numpy(atom_params_dict[atomic_number]["params"])
    P = P.to(device)
    # tile scattering factors to match r_xy grid
    P = P[:, :, None, None, None].expand((4, 3) + k_xyz.shape)

    s1 = torch.sum(P[0] / (k_xyz**2 + P[1]), 0)
    s2 = torch.sum(P[2] * torch.exp(-P[3] * k_xyz**2), 0)
    return (s1 + s2)


def nearest_index(x_arr, y_arr, z_arr, x_coord, y_coord, z_coord):
    xi = torch.argmin(torch.abs(x_arr - x_coord))
    yi = torch.argmin(torch.abs(y_arr - y_coord))
    zi = torch.argmin(torch.abs(z_arr - z_coord))
    return xi, yi, zi

def voxelize_atoms(coords, grid_shape, voxel_size, device=None):
    """
    Convert atomic coordinates to a 3D binary grid (1s at nearest voxel, zeros elsewhere),
    assuming the center of the grid is at (nz//2, ny//2, nx//2).

    Args:
        coords: (N,3) tensor of atomic coordinates in physical units (x, y, z)
        grid_shape: tuple of ints (nx, ny, nz)
        voxel_size: tuple of floats (dx, dy, dz)
        device: optional, torch device

    Returns:
        grid: (nz, ny, nx) tensor with 1s at voxel positions of atoms
    """
    device = device or coords.device
    coords = coords.to(device)
    nx, ny, nz = grid_shape  # number of voxels along x, y, z

    # Compute the center of the grid in voxel units
    center_voxel = torch.tensor([nx//2, ny//2, nz//2], device=device)

    # Convert physical coordinates to voxel indices, shifting center
    indices = coords / torch.tensor(voxel_size, device=device)  # voxel units
    indices = indices + center_voxel  # shift so center is at middle voxel
    indices = torch.round(indices).long()

    # Mask atoms inside the grid
    mask = (
        (indices[:,0] >= 0) & (indices[:,0] < nx) &
        (indices[:,1] >= 0) & (indices[:,1] < ny) &
        (indices[:,2] >= 0) & (indices[:,2] < nz)
    )
    indices = indices[mask]

    # Create empty grid (z, y, x)
    grid = torch.zeros((nz, ny, nx), device=device, dtype=torch.float32)

    # Insert ones at valid voxel indices
    grid.index_put_(
        (indices[:,2], indices[:,1], indices[:,0]),
        torch.ones(indices.shape[0], device=device),
        accumulate=True
    )

    return grid

def soft_voxelize_atoms(coords, grid_shape, voxel_size, device=None):
    """
    Differentiable 3D soft voxelization using trilinear splatting.

    Args:
        coords: (N,3) tensor of atomic coordinates in physical units (x, y, z)
        grid_shape: tuple of ints (nz, ny, nx)
        voxel_size: tuple of floats (dx, dy, dz)
        device: optional, torch device

    Returns
    -------
    volume : (nz, ny, nx) tensor
        Differentiable soft voxelized volume
    """
    if device is None:
        device = coords.device
    coords = coords.to(device)
    nz, ny, nx = grid_shape
    N = coords.shape[0]

    values = torch.ones(N, device=device)

    # Convert physical coordinates to voxel units
    if isinstance(voxel_size, (int,float)):
        voxel_size = torch.tensor([voxel_size]*3, device=device)
    else:
        voxel_size = torch.tensor(voxel_size, device=device)
    coords_voxel = coords / voxel_size  # (N,3)

    # Shift coordinates so origin (0,0,0) is at floor division center
    origin = torch.tensor([nx//2, ny//2, nz//2], device=device, dtype=coords_voxel.dtype)
    coords_voxel_centered = coords_voxel + origin[None,:]  # (N,3)

    # Reorder coords to z, y, x for indexing
    coords_voxel_centered = coords_voxel_centered[:, [2,1,0]]  # (N,3)

    # Floor and fractional part for trilinear weights
    coords_floor = torch.floor(coords_voxel_centered).long()  # (N,3)
    frac = coords_voxel_centered - coords_floor.float()       # (N,3)
    dz, dy, dx = frac[:,0], frac[:,1], frac[:,2]
    z0, y0, x0 = coords_floor[:,0], coords_floor[:,1], coords_floor[:,2]

    # 8 neighbor offsets
    offsets = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                            [1,0,0],[1,0,1],[1,1,0],[1,1,1]], device=device)

    # Compute neighbor indices (N,8)
    z_idx = z0[:,None] + offsets[None,:,0]
    y_idx = y0[:,None] + offsets[None,:,1]
    x_idx = x0[:,None] + offsets[None,:,2]

    # Trilinear weights (N,8)
    w = ((1-dz)[:,None]*(1-offsets[None,:,0]) + dz[:,None]*offsets[None,:,0]) * \
        ((1-dy)[:,None]*(1-offsets[None,:,1]) + dy[:,None]*offsets[None,:,1]) * \
        ((1-dx)[:,None]*(1-offsets[None,:,2]) + dx[:,None]*offsets[None,:,2])
    w = w * values[:,None]

    # Mask out-of-bounds
    mask = (z_idx>=0)&(z_idx<nz) & (y_idx>=0)&(y_idx<ny) & (x_idx>=0)&(x_idx<nx)
    z_idx = z_idx[mask]
    y_idx = y_idx[mask]
    x_idx = x_idx[mask]
    w = w[mask]

    # Scatter-add into volume
    volume = torch.zeros(nz, ny, nx, device=device)
    volume.index_put_((z_idx, y_idx, x_idx), w, accumulate=True)

    return volume

def coordinate_grid_3d(n_xyz, d_xyz, convention="relion"):
    """Constructs the xyz coordinate arrays and meshgrids. Meshgrid indexing yields
    vol[z_i, y_i, x_i] convention which matches cryo-EM software.

    The coordinate grid convention only matters if the number of pixels is even.

    Parameters
    ----------
    n_xyz : array-like
        Number of pixels along x,y,z: (nx, ny, nz)
    d_xyz : array-like
        Pixel length along x,y,z: (dx,dy,dz)
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
    dx, dy, dz = d_xyz
    nx, ny, nz = n_xyz
    if convention == "relion":
        x = (torch.arange(nx) - nx // 2) * dx
        y = (torch.arange(ny) - ny // 2) * dy
        z = (torch.arange(nz) - nz // 2) * dz
    elif convention == "torch":
        x = (torch.arange(nx) - (nx - 1) / 2) * dx
        y = (torch.arange(ny) - (ny - 1) / 2) * dy
        z = (torch.arange(nz) - (nz - 1) / 2) * dz
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
    return x, y, z, X, Y, Z


def coordinate_grid_2d(n_xy, d_xy, convention="relion"):
    """Constructs the xy coordinate arrays and meshgrids. Meshgrid indexing yields
    area[y_i, x_i] convention which matches cryo-EM software.

    The coordinate grid convention only matters if the number of pixels is even.

    Parameters
    ----------
    n_xy : array-like
        Number of pixels along x,y: (nx, ny)
    d_xy : array-like
        Pixel length along x,y: (dx,dy)
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
    dx, dy = d_xy
    nx, ny = n_xy
    if convention == "relion":
        x = (torch.arange(nx) - nx // 2) * dx
        y = (torch.arange(ny) - ny // 2) * dy
    elif convention == "torch":
        x = (torch.arange(nx) - (nx - 1) / 2) * dx
        y = (torch.arange(ny) - (ny - 1) / 2) * dy
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return x, y, X, Y


def build_potential_volume(
    atomic_numbers,
    centered_coords,
    n_xyz,
    d_xyz,
    atom_size_px=None,
    super_sampling_factor=4,
    convention="relion",
    method="3d",
    disable_tqdm=False,
):
    """Constructs volumetric potential from list of atomic elements and their
    respective coordinates.

    General strategy is:
    1. Compute potential of a single atom on a super-sampled grid (higher resolution
    than main volume but lesser pixels since the potential decays fast).
    2. Bin the potential down to main volume grid size and insert additively.

    Differences in methods:
    - 2D/3D uses either the 3D potential or projected 2D potential equation.
    - snapped-methods precomputes the potentials assuming the atom-core falls 
    exactly on a voxel (i.e. snaps all atoms to the nearest voxel). This yields 
    significantly faster computation as potentials only need to be calculated for
    individual element once.

    Note: 
    For 2D versions, assumes dx = dy, nx = ny. nz and dz are free to be different.
    For 3D version, assumes dx = dy = dz, nx = ny = nz.

    Parameters
    ----------
    atomic_numbers : 1d tensor
        Atomic numbers. Hydrogen is 1.
    centered_coords : 2d tensor
        xyz coordinates corresponding to each entry in atomic_numbers. Shape of
        (len(atomic_numbers), 3).
    n_xyz : array-like
        Number of pixels along x,y,z: (nx, ny, nz) of main volume.
    d_xyz : array-like
        Pixel length along x,y,z: (dx,dy,dz) of main volume.
    atom_size_px : int
        Number of main volume pixels to sufficiently represent an atom. If None,
        will assume 3A diameter per atom, and computed required number of pixels
        accordingly.
    super_sampling_factor : int
        The supersampling factor to compute the potentials. For example, if main
        volume has pixel size of 1A, then potentials will be first computed on a
        grid of 1A/super_sampling_factor before binning back to 1A pixels. Must be
        even to avoid singularity at 0, and larger than 4 (Kirkland's rule of thumb,
        Chapter 5, after Fig 5.15.).
    convention : str
        The origin convention for main volume only. The super-sampled grid for
        atomic potentials will always be even-valued and symmetric to avoid the
        singularity at 0.
    method : str
        '3d' - Does not snap atom to nearest voxel. Computes each atom's 3D
        potential individually based on the local super-sampled coordinate grid.

        'snapped-3d' - Assumes each atom snaps to the nearest voxel defined on a
        rectangular grid. Each 3D potential is first super-sampled on a finer grid
        before averaging the pixels to insert into the main volume.

        '2d' - Snaps atom only to nearest z-plane, but maintains it's x,y
        coordinates. Computes each atom's 2D potential individually based on the
        local super-sampled coordinate grid.

        'snapped-2d' - Assumes each atom snaps to the nearest voxel defined on a
        rectangular grid. Further assumes each atom can be represented by its
        projected 2D potential. This 2D potential is first super-sampled on a finer
        grid before averaging the pixels to insert into the main volume.

    Returns
    -------
    potential_volume : 3d tensor
        The sampled potential volume.
    """
    # create main volume coordinate system
    nx, ny, nz = n_xyz
    dx, dy, dz = d_xyz
    x, y, z, X, Y, Z = coordinate_grid_3d(
        (nx, ny, nz), (dx, dy, dz), convention=convention
    )

    # create super-sampled (ss) coordinate system
    if atom_size_px is None:
        # forces odd number to ensure central pixel exists.
        atom_size_px = int(torch.ceil(torch.tensor(3 / dx)) // 2 * 2 + 1)
    ssn = atom_size_px * super_sampling_factor
    ssdx = dx / super_sampling_factor

    if method == "3d" or method == "snapped-3d":
        sx, sy, sz, sX, sY, sZ = coordinate_grid_3d(
            (ssn, ssn, ssn), (ssdx, ssdx, ssdx), convention="torch"
        )
        if method == "snapped-3d":
            sR = torch.sqrt(sX**2 + sY**2 + sZ**2)
    elif method == "2d" or method == "snapped-2d":
        sx, sy, sX, sY = coordinate_grid_2d(
            (ssn, ssn), (ssdx, ssdx), convention="torch"
        )
        if method == "snapped-2d":
            sR = torch.sqrt(sX**2 + sY**2)

    # for binning super-sampled grids to main volume grid.
    avgpool2d = torch.nn.AvgPool2d(super_sampling_factor, stride=super_sampling_factor)
    avgpool3d = torch.nn.AvgPool3d(super_sampling_factor, stride=super_sampling_factor)

    # For snapped methods, compute unique element potentials on ss-grid and average
    # onto main volume grid
    if method == "snapped-2d":
        sampled_2dpot_dict = {}
        for an in torch.unique(atomic_numbers):
            pot = atomic_potential_2d(int(an), sR)
            sampled_2dpot_dict[int(an)] = avgpool2d(pot[None, None]).squeeze()
    elif method == "snapped-3d":
        sampled_3dpot_dict = {}
        for an in torch.unique(atomic_numbers):
            pot = atomic_potential_3d(int(an), sR)
            # note the multiplicative factor of dx to properly scale for
            # projection/multislice to match 2d version above.
            sampled_3dpot_dict[int(an)] = avgpool3d(pot[None, None]).squeeze() * dx

    # insert atomic potentials into main volume.
    potential_volume = torch.zeros(nz, ny, nx)
    occupancy = torch.zeros(nz, ny, nx, dtype=torch.bool)
    for an, cc in tqdm(zip(atomic_numbers, centered_coords), disable=disable_tqdm):
        xi, yi, zi = nearest_index(x, y, z, cc[0], cc[1], cc[2])

        # don't insert if bounding box of atom falls outside of main volume grid.
        if (
            (zi - atom_size_px // 2) < 0
            or zi - atom_size_px // 2 + atom_size_px > nz
            or (yi - atom_size_px // 2) < 0
            or yi - atom_size_px // 2 + atom_size_px > ny
            or (xi - atom_size_px // 2) < 0
            or xi - atom_size_px // 2 + atom_size_px > nx
        ):
            pass
        else:
            # update occupancy
            occupancy[zi, yi, xi] = True

            # insert atoms
            if method == "3d":
                # relative 3D origin of the atom w.r.t. neighbouring voxels.
                x_ro = cc[0] - x[xi]
                y_ro = cc[1] - y[yi]
                z_ro = cc[2] - z[zi]
                sR = torch.sqrt((sX - x_ro) ** 2 + (sY - y_ro) ** 2 + (sZ - z_ro) ** 2)
                sspot = atomic_potential_3d(int(an), sR)
                pot = avgpool3d(sspot[None, None]).squeeze() * dx

                potential_volume[
                    zi - atom_size_px // 2 : zi - atom_size_px // 2 + atom_size_px,
                    yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
                    xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
                ] += pot
            elif method == "snapped-3d":
                potential_volume[
                    zi - atom_size_px // 2 : zi - atom_size_px // 2 + atom_size_px,
                    yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
                    xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
                ] += sampled_3dpot_dict[int(an)]
            elif method == "2d":
                # relative 2D origin of the atom w.r.t. neighbouring voxels.
                x_ro = cc[0] - x[xi]
                y_ro = cc[1] - y[yi]
                sR = torch.sqrt((sX - x_ro) ** 2 + (sY - y_ro) ** 2)
                sspot = atomic_potential_2d(int(an), sR)
                pot = avgpool2d(sspot[None, None]).squeeze()

                potential_volume[
                    zi,
                    yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
                    xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
                ] += pot
            elif method == "snapped-2d":
                potential_volume[
                    zi,
                    yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
                    xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
                ] += sampled_2dpot_dict[int(an)]
    return potential_volume, occupancy

def build_potential_volume_fftconvolve(
    atomic_numbers,
    centered_coords,
    n_xyz,
    d_xyz,
    atom_size_px=None,
    super_sampling_factor=4,
    convention="relion",
    method="snapped-3d",
    compute_high_res=False,
    disable_tqdm=False,
):
    """Constructs volumetric potential from list of atomic elements and their
    respective coordinates. 

    General strategy is:
    1. Compute potentials for each unique element on a super-sampled grid (higher 
    resolution than main volume but lesser pixels since the potential decays fast).
    2. Calculate the potential contributions for each elemental species and sum.
    2. Bin the potential down to main volume grid size.

    Note: 
    For 2D versions, assumes dx = dy, nx = ny. nz and dz are free to be different.
    For 3D version, assumes dx = dy = dz, nx = ny = nz.

    Parameters
    ----------
    atomic_numbers : 1d tensor
        Atomic numbers. Hydrogen is 1.
    centered_coords : 2d tensor
        xyz coordinates corresponding to each entry in atomic_numbers. Shape of
        (len(atomic_numbers), 3).
    n_xyz : array-like
        Number of pixels along x,y,z: (nx, ny, nz) of main volume.
    d_xyz : array-like
        Pixel length along x,y,z: (dx,dy,dz) of main volume.
    atom_size_px : int
        Number of main volume pixels to sufficiently represent an atom. If None,
        will assume 3A diameter per atom, and computed required number of pixels
        accordingly.
    super_sampling_factor : int
        The supersampling factor to compute the potentials. For example, if main
        volume has pixel size of 1A, then potentials will be first computed on a
        grid of 1A/super_sampling_factor before binning back to 1A pixels. Must be
        even to avoid singularity at 0, and larger than 4 (Kirkland's rule of thumb,
        Chapter 5, after Fig 5.15.).
    convention : str
        The origin convention for main volume only. The super-sampled grid for
        atomic potentials will always be even-valued and symmetric to avoid the
        singularity at 0.
    method : str
        'snapped-3d' - Assumes each atom snaps to the nearest voxel defined on a
        rectangular grid. Each 3D potential is first super-sampled on a finer grid
        before averaging the pixels to insert into the main volume.

        'snapped-2d' - Assumes each atom snaps to the nearest voxel defined on a
        rectangular grid. Further assumes each atom can be represented by its
        projected 2D potential. This 2D potential is first super-sampled on a finer
        grid before averaging the pixels to insert into the main volume.

    Returns
    -------
    potential_volume : 3d tensor
        The sampled potential volume.
    """
    # create main volume coordinate system
    nx, ny, nz = n_xyz
    dx, dy, dz = d_xyz

    if compute_high_res:
        snx = nx * super_sampling_factor
        sny = ny * super_sampling_factor
        snz = nz * super_sampling_factor
        sdx = dx / super_sampling_factor
        sdy = dy / super_sampling_factor
        sdz = dz / super_sampling_factor
        x, y, z, X, Y, Z = coordinate_grid_3d(
            (snx, sny, snz), (sdx, sdy, sdz), convention=convention
        )
    else:
        x, y, z, X, Y, Z = coordinate_grid_3d(
            (nx, ny, nz), (dx, dy, dz), convention=convention
        )

    # create super-sampled (ss) coordinate system
    if atom_size_px is None:
        # forces odd number to ensure central pixel exists.
        atom_size_px = int(torch.ceil(torch.tensor(3 / dx)) // 2 * 2 + 1)
    ssn = atom_size_px * super_sampling_factor
    ssdx = dx / super_sampling_factor

    if method == "snapped-3d":
        sx, sy, sz, sX, sY, sZ = coordinate_grid_3d(
            (ssn, ssn, ssn), (ssdx, ssdx, ssdx), convention="torch"
        )
        sR = torch.sqrt(sX**2 + sY**2 + sZ**2)
    elif method == "snapped-2d":
        sx, sy, sX, sY = coordinate_grid_2d(
            (ssn, ssn), (ssdx, ssdx), convention="torch"
        )
        sR = torch.sqrt(sX**2 + sY**2)

    # for binning super-sampled grids to main volume grid.
    avgpool2d = torch.nn.AvgPool2d(super_sampling_factor, stride=super_sampling_factor)
    avgpool3d = torch.nn.AvgPool3d(super_sampling_factor, stride=super_sampling_factor)

    # For snapped methods, compute unique element potentials on ss-grid and average
    # onto main volume grid
    if method == "snapped-2d":
        sampled_2dpot_dict = {}
        for an in torch.unique(atomic_numbers):
            pot = atomic_potential_2d(int(an), sR)
            sampled_2dpot_dict[int(an)] = avgpool2d(pot[None, None]).squeeze()
    elif method == "snapped-3d":
        sampled_3dpot_dict = {}
        for an in torch.unique(atomic_numbers):
            pot = atomic_potential_3d(int(an), sR)
            # note the multiplicative factor of dx to properly scale for
            # projection/multislice to match 2d version above.
            sampled_3dpot_dict[int(an)] = avgpool3d(pot[None, None]).squeeze() * dx

    # insert atomic potentials into main volume.
    potential_volume = torch.zeros(nz, ny, nx)
    occupancy = torch.zeros(nz, ny, nx, dtype=torch.bool)
    atomic_potentials = {}

    pbar = tqdm(torch.unique(atomic_numbers), disable=disable_tqdm)
    for elem in pbar:
        pbar.set_description(f"Building element {atom.atom_symbol(int(elem))}")
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # populate elemental volume with delta function atoms
        # soft_voxelize_atoms is differentiable w.r.t. coordinates.
        temp_vol = soft_voxelize_atoms(centered_coords[atomic_indices].reshape(-1,3),
                                 grid_shape=(nz, ny, nx),
                                 voxel_size=(dz, dy, dx))
        occupancy = occupancy | (temp_vol > 0)

        # get potential kernel for this element
        pot = atomic_potential_3d(int(elem), sR)
    
        atomic_potentials[atom.atom_symbol(int(elem))] = pot
        atomic_potentials['ssdx'] = ssdx
        #convolve
        if compute_high_res:
            temp_vol = fftconvolve(temp_vol, pot, mode='same')
            potential_volume += avgpool3d(temp_vol[None, None]).squeeze() * dx
        else:
            pot = avgpool3d(pot[None, None]).squeeze() * dx
            potential_volume += fftconvolve(temp_vol, pot, mode='same')

    return potential_volume, occupancy, atomic_potentials
