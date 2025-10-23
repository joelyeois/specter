import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .array import (
    grid_1d,
    grid_2d,
    grid_3d,
    nearest_index,
    soft_voxelize_coordinates,
    voxelize_coordinates,
)
from .atom import (
    atom_symbol,
    kirkland_atomic_potential_2d,
    kirkland_atomic_potential_3d,
    lobato_atomic_potential_3d,
)
from .fft_tools import fftconvolve


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
    x, y, z, X, Y, Z = grid_3d(
        (nx, ny, nz), (dx, dy, dz), convention=convention
    )

    # create super-sampled (ss) coordinate system
    if atom_size_px is None:
        # forces odd number to ensure central pixel exists.
        atom_size_px = int(torch.ceil(torch.tensor(3 / dx)) // 2 * 2 + 1)
    ssn = atom_size_px * super_sampling_factor
    ssdx = dx / super_sampling_factor

    if method == "3d" or method == "snapped-3d":
        sx, sy, sz, sX, sY, sZ = grid_3d(
            (ssn, ssn, ssn), (ssdx, ssdx, ssdx), convention="torch"
        )
        if method == "snapped-3d":
            sR = torch.sqrt(sX**2 + sY**2 + sZ**2)
    elif method == "2d" or method == "snapped-2d":
        sx, sy, sX, sY = grid_2d(
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
            pot = kirkland_atomic_potential_2d(int(an), sR)
            sampled_2dpot_dict[int(an)] = avgpool2d(pot[None, None]).squeeze()
    elif method == "snapped-3d":
        sampled_3dpot_dict = {}
        for an in torch.unique(atomic_numbers):
            pot = kirkland_atomic_potential_3d(int(an), sR)
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
                sspot = kirkland_atomic_potential_3d(int(an), sR)
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
                sspot = kirkland_atomic_potential_2d(int(an), sR)
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
        x, y, z, X, Y, Z = grid_3d(
            (snx, sny, snz), (sdx, sdy, sdz), convention=convention
        )
    else:
        x, y, z, X, Y, Z = grid_3d(
            (nx, ny, nz), (dx, dy, dz), convention=convention
        )

    # create super-sampled (ss) coordinate system
    if atom_size_px is None:
        # forces odd number to ensure central pixel exists.
        atom_size_px = int(torch.ceil(torch.tensor(3 / dx)) // 2 * 2 + 1)
    ssn = atom_size_px * super_sampling_factor
    ssdx = dx / super_sampling_factor

    if method == "snapped-3d":
        sx, sy, sz, sX, sY, sZ = grid_3d(
            (ssn, ssn, ssn), (ssdx, ssdx, ssdx), convention="torch"
        )
        sR = torch.sqrt(sX**2 + sY**2 + sZ**2)
    elif method == "snapped-2d":
        sx, sy, sX, sY = grid_2d(
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
            pot = kirkland_atomic_potential_2d(int(an), sR)
            sampled_2dpot_dict[int(an)] = avgpool2d(pot[None, None]).squeeze()
    elif method == "snapped-3d":
        sampled_3dpot_dict = {}
        for an in torch.unique(atomic_numbers):
            pot = kirkland_atomic_potential_3d(int(an), sR)
            # note the multiplicative factor of dx to properly scale for
            # projection/multislice to match 2d version above.
            sampled_3dpot_dict[int(an)] = avgpool3d(pot[None, None]).squeeze() * dx

    # insert atomic potentials into main volume.
    potential_volume = torch.zeros(nz, ny, nx)
    occupancy = torch.zeros(nz, ny, nx, dtype=torch.bool)
    atomic_potentials = {}

    pbar = tqdm(torch.unique(atomic_numbers), disable=disable_tqdm)
    for elem in pbar:
        pbar.set_description(f"Building element {atom_symbol(int(elem))}")
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # populate elemental volume with delta function atoms
        # soft_voxelize_atoms is differentiable w.r.t. coordinates.
        temp_vol = soft_voxelize_coordinates(centered_coords[atomic_indices].reshape(-1,3),
                                 grid_shape=(nz, ny, nx),
                                 voxel_size=(dz, dy, dx))
        occupancy = occupancy | (temp_vol > 0)

        # get potential kernel for this element
        pot = kirkland_atomic_potential_3d(int(elem), sR)
    
        atomic_potentials[atom_symbol(int(elem))] = pot
        atomic_potentials['ssdx'] = ssdx
        #convolve
        if compute_high_res:
            temp_vol = fftconvolve(temp_vol, pot, mode='same')
            potential_volume += avgpool3d(temp_vol[None, None]).squeeze() * dx
        else:
            pot = avgpool3d(pot[None, None]).squeeze() * dx
            potential_volume += fftconvolve(temp_vol, pot, mode='same')

    return potential_volume, occupancy, atomic_potentials
