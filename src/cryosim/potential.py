import torch
import torch.nn.functional as F
from rich.progress import track, Progress
import torch.nn as nn
import lightning as L

from .array_utils import (
    grid_2d,
    grid_3d,
    nearest_index,
    radial_grid_2d,
    radial_grid_3d,
    soft_voxelize_coordinates,
    soft_voxelize_xy_coordinates,
)
from .atom import (
    atom_symbol,
    kirkland_atomic_potential_2d,
    kirkland_atomic_potential_3d,
    lobato_atomic_potential_2d,
    lobato_atomic_potential_3d,
    shryov_atomic_potential_3d
)
from .fft_tools import fftconvolve


def compute_supersampling_parameters(dx, width_atom=5.0, dx_atom=0.1):
    """
    Compute grid size, pixel spacing and supersampling factor for atomic potentials.

    Ensures the grid is fine enough to sample the potential (default 0.1 Å/pixel)
    and wide enough to cover the 'size' of an atom (default 5Å). Can be downsampled
    later using integer stride.

    Parameters
    ----------
    dx : float
        Desired final pixel size (Å) of the main volume.
    width_atom : float
        Size of an atom in Å.
    dx_atom : float
        Pixel size to sample the potential of an atom.

    Returns
    -------
    n_atom : int
        Number of pixels along each axis in the supersampled grid.
    ss_dx : float
        Pixel size of the supersampled grid.
    ssf : int
        Supersampling factor
    """
    if dx <= dx_atom:
        ssf = 1
        ss_dx = dx
        n_atom = int(width_atom / dx)
        # make even
        n_atom = n_atom + (n_atom % 2)
        return n_atom, ss_dx, ssf

    else:
        # Number of pixels at atom sampling
        n_atom = int(torch.ceil(torch.tensor(width_atom / dx_atom)))

        # Step 1: make divisible by ssf
        ssf = int(torch.round(torch.tensor(dx / dx_atom)))
        ss_dx = dx / ssf

        # Step 2: adjust n_atom to satisfy both evenness and divisibility
        # find the smallest even number divisible by ssf and >= n_atom
        while (n_atom % ssf != 0) or (n_atom % 2 != 0):
            n_atom += 1

        return n_atom, ss_dx, ssf


def build_potential_volume_fftconvolve_3d(
    atomic_numbers,
    centered_coords,
    n_xyz,
    dx,
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
    dx : float
        Voxel size of main volume.

    Returns
    -------
    potential_volume : 3d tensor
        The sampled potential volume.
    """
    # create main volume coordinate system
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    # create super-sampled (ss) coordinate system
    ssn, ssdx, ssf = compute_supersampling_parameters(dx)
    # set origina convention to torch to avoid singularity at origin.
    sR = radial_grid_3d(ssn, ssdx, convention="torch")

    # for binning super-sampled grids to main volume grid.
    avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)

    # insert atomic potentials into main volume.
    potential_volume = torch.zeros(nz, ny, nx)
    occupancy = torch.zeros(nz, ny, nx, dtype=torch.bool)
    atomic_potentials = {}

    for elem in track(
        torch.unique(atomic_numbers),
        description="Building elements",
        disable=disable_tqdm
    ):
        # Update the description dynamically per element
        track.description = f"Building element {atom_symbol(int(elem))}"
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # populate elemental volume with delta function atoms
        # soft_voxelize_atoms is differentiable w.r.t. coordinates.
        temp_vol = soft_voxelize_coordinates(centered_coords[atomic_indices].reshape(-1,3),
                                 grid_shape=(nz, ny, nx),
                                 voxel_size=dx)
        occupancy = occupancy | (temp_vol > 0)

        # get potential kernel for this element
        pot = kirkland_atomic_potential_3d(int(elem), sR)

        #convolve
        if ssf != 1:
            pot = avgpool3d(pot[None, None]) * dx
            pot = pot.squeeze(0).squeeze(0)
        potential_volume += fftconvolve(temp_vol, pot, mode='same')
    return potential_volume, sR, atomic_potentials


def build_potential_volume_fftconvolve_2d(
    atomic_numbers,
    centered_coords,
    n_xyz,
    dx,
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
    dx : float
        Pixel size of main volume.

    Returns
    -------
    potential_volume : 3d tensor
        The sampled potential volume.
    """
    # create main volume coordinate system
    if isinstance(n_xyz, int):
        nx = ny = nz = n_xyz
    else:
        nx, ny, nz = n_xyz

    # create super-sampled (ss) coordinate system
    ssn, ssdx, ssf = compute_supersampling_parameters(dx)
    sR = radial_grid_2d(ssn, ssdx, convention="torch")

    # for binning super-sampled grids to main volume grid.
    avgpool2d = torch.nn.AvgPool2d(ssf, stride=ssf)

    # insert atomic potentials into main volume.
    potential_volume = torch.zeros(nz, ny, nx)
    occupancy = torch.zeros(nz, ny, nx, dtype=torch.bool)
    atomic_potentials = {}

    for elem in track(
        torch.unique(atomic_numbers),
        description="Building elements",
    ):
        # Update the description dynamically per element
        track.description = f"Building element {atom_symbol(int(elem))}"
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # populate elemental volume with delta function atoms
        # soft_voxelize_atoms is differentiable w.r.t. coordinates.
        temp_vol = soft_voxelize_xy_coordinates(
            centered_coords[atomic_indices].reshape(-1,3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx
        )
        occupancy = occupancy | (temp_vol > 0)

        # get potential kernel for this element
        pot = kirkland_atomic_potential_2d(int(elem), sR)

        #convolve
        if ssf != 1:
            pot = avgpool2d(pot[None, None]) * dx
            pot = pot.squeeze(0).squeeze(0)

        #batch 2D convolve
        temp_vol_b = temp_vol.unsqueeze(1)   # (nz, 1, ny, nx)
        pot_b = pot.unsqueeze(0).unsqueeze(0)  # (1, 1, ky, kx)
        convolved = F.conv2d(temp_vol_b, pot_b, padding='same')
        potential_volume += convolved.squeeze(1)    # (nz, ny, nx)
    return potential_volume, sR, atomic_potentials


class PotentialBuilder(L.LightningModule):
    def __init__(
        self,
        n_xyz,
        dx,
        atomic_numbers,
        verbose=True,
        parameterization='kirkland',
        conv_backend='fftconvolve',
        trainable=False,
        mmcif_filepath=None,
    ):
        super().__init__()

        if isinstance(n_xyz, int):
            self.nx = self.ny = self.nz = n_xyz
        else:
            self.nx, self.ny, self.nz = n_xyz
        self.dx = dx
        self.verbose = verbose
        self.conv_backend = conv_backend
        self.mmcif_filepath = mmcif_filepath

        # create super-sampled (ss) coordinate system
        self.ssn, self.ssdx, self.ssf = compute_supersampling_parameters(dx)
        sR_2d = radial_grid_2d(self.ssn, self.ssdx, convention="torch")
        sR_3d = radial_grid_3d(self.ssn, self.ssdx, convention="torch")
        self.register_buffer('sR_2d', sR_2d)
        self.register_buffer('sR_3d', sR_3d)

        # create atomic potentials
        self.atomic_numbers = atomic_numbers
        self.unique_elements = torch.unique(atomic_numbers)
        atomic_potentials_2d = torch.empty(
            len(self.unique_elements),
            self.ssn//self.ssf,
            self.ssn//self.ssf
        )
        atomic_potentials_3d = torch.empty(
            len(self.unique_elements),
            self.ssn//self.ssf,
            self.ssn//self.ssf,
            self.ssn//self.ssf
        )
        self.register_buffer('atomic_potentials_2d', atomic_potentials_2d)
        self.register_buffer('atomic_potentials_3d', atomic_potentials_3d)

        self.parameterization = parameterization
        self.avgpool2d = torch.nn.AvgPool2d(self.ssf, stride=self.ssf)
        self.avgpool3d = torch.nn.AvgPool3d(self.ssf, stride=self.ssf)
        if parameterization in ('kirkland', 'lobato'):
            self.get_2d_atomic_potentials()
        self.get_3d_atomic_potentials()


    def get_2d_atomic_potentials(self, unique_elements=None):
        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            # update unique elements
            self.unique_elements = unique_elements
            self.atomic_potentials = torch.empty(
                len(unique_elements),
                self.ssn//self.ssf,
                self.ssn//self.ssf
            )

        # fetch potential kernels
        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == 'kirkland':
                pot = kirkland_atomic_potential_2d(int(elem), self.sR_2d)
            elif self.parameterization == 'lobato':
                pot = lobato_atomic_potential_2d(int(elem), self.sR_2d)

            if self.ssf != 1:
                pot = self.avgpool2d(pot[None, None]) * self.dx
                pot = pot.squeeze(0).squeeze(0)

            self.atomic_potentials_2d[i] = pot

    def get_3d_atomic_potentials(self, unique_elements=None):
        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            # update unique elements
            self.unique_elements = unique_elements
            self.atomic_potentials = torch.empty(
                len(unique_elements),
                self.ssn//self.ssf,
                self.ssn//self.ssf,
                self.ssn//self.ssf
            )

        # fetch potential kernels
        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == 'kirkland':
                pot = kirkland_atomic_potential_3d(int(elem), self.sR_3d)
            elif self.parameterization == 'lobato':
                pot = lobato_atomic_potential_3d(int(elem), self.sR_3d)
            elif self.parameterization == 'shryov':
                if self.mmcif_filepath is None:
                    raise ValueError(f"mmcif_filepath must be specified.")
                else:
                    pot = shryov_atomic_potential_3d(int(elem), self.sR_3d, self.mmcif_filepath)

            if self.ssf != 1:
                pot = self.avgpool3d(pot[None, None]) * self.dx
                pot = pot.squeeze(0).squeeze(0)

            self.atomic_potentials_3d[i] = pot


    def forward(self, coordinates, method='3d', conv_backend=None):
        if conv_backend is None:
            conv_backend = self.conv_backend
        coordinates = coordinates.to(self.device)
        self.method = method

        # Detect batch
        batched_input = True
        if coordinates.ndim == 2:  # (N,3) -> add batch dimension
            coordinates = coordinates.unsqueeze(0)
            batched_input = False
    
        B, N, _ = coordinates.shape
        
        # insert atomic potentials into main volume.
        potential_volume = torch.zeros((B, self.nz, self.ny, self.nx), device=self.device)
        self.occupancy = torch.zeros((B, self.nz, self.ny, self.nx), dtype=torch.bool)
        
        with Progress(transient=True) as progress:
            # Create a single task for the outer loop
            task = progress.add_task("Building element ...", total=len(self.unique_elements))
            for i, elem in enumerate(self.unique_elements):
                progress.update(task, description=f"Building element {atom_symbol(int(elem))}", advance=1)
                atomic_indices = torch.squeeze(torch.argwhere(self.atomic_numbers == elem))

                # Select atomic coordinates for this element
                coords_elem = coordinates[:, atomic_indices, :]  # (B, Nelem, 3)
                
                if method == '2d':
                    temp_vol = soft_voxelize_xy_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx
                    )
                    
                    # Flatten B and Z for conv2d
                    temp_vol_flat = temp_vol.reshape(-1, 1, self.ny, self.nx)  # (B*Z, 1, Y, X)
                    
                    # Kernel: (1, 1, ky, kx)
                    pot_b = self.atomic_potentials_2d[i].unsqueeze(0).unsqueeze(0)  # (1,1,ky,kx)
                    
                    # Perform conv2d
                    convolved_flat = F.conv2d(temp_vol_flat, pot_b, padding='same')  # (B*Z, 1, Y, X)
                    
                    # Reshape back to (B, Z, Y, X)
                    convolved = convolved_flat.reshape(B, self.nz, self.ny, self.nx)
                    
                    # Add to potential volume
                    potential_volume += convolved
                    
                elif method == '3d':
                    temp_vol = soft_voxelize_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx
                    )
    
                    #convolve
                    # Convolve 3D potentials per batch
                    if conv_backend == 'fftconvolve':
                        # for b in track(range(B), description='Convolving atoms', transient=True):
                        for b in range(B):
                            potential_volume[b] += fftconvolve(
                                temp_vol[b],
                                self.atomic_potentials_3d[i],
                                mode='same'
                            )

                    # using conv3d instead
                    elif conv_backend == 'conv3d':
                        vol_b = temp_vol.unsqueeze(1)  # (B,1,nz,ny,nx)
                        kernel = self.atomic_potentials_3d[i].unsqueeze(0).unsqueeze(0)  # (1,1,kz,ky,kx)
                        # Use conv3d with groups=1
                        convolved = F.conv3d(vol_b, kernel, padding='same')  # (B,1,nz,ny,nx)
                        potential_volume += convolved.squeeze(1)  # (B,nz,ny,nx)
                        
                    # potential_volume += fftconvolve(
                    #     temp_vol,
                    #     self.atomic_potentials_3d[i],
                    #     mode='same'
                    # )
    
                # Update occupancy, very slow.
                # self.occupancy |= (temp_vol.detach().cpu() > 0)
        if B == 1:
            potential_volume = potential_volume.squeeze(0)
        return potential_volume

################ OLD & SLOW ################
# def build_potential_volume(
#     atomic_numbers,
#     centered_coords,
#     n_xyz,
#     dx,
#     atom_size_px=None,
#     super_sampling_factor=4,
#     convention="relion",
#     method="3d",
#     disable_tqdm=False,
# ):
#     """Constructs volumetric potential from list of atomic elements and their
#     respective coordinates.

#     General strategy is:
#     1. Compute potential of a single atom on a super-sampled grid (higher resolution
#     than main volume but lesser pixels since the potential decays fast).
#     2. Bin the potential down to main volume grid size and insert additively.

#     Differences in methods:
#     - 2D/3D uses either the 3D potential or projected 2D potential equation.
#     - snapped-methods precomputes the potentials assuming the atom-core falls 
#     exactly on a voxel (i.e. snaps all atoms to the nearest voxel). This yields 
#     significantly faster computation as potentials only need to be calculated for
#     individual element once.

#     Note: 
#     For 2D versions, assumes dx = dy, nx = ny. nz and dz are free to be different.
#     For 3D version, assumes dx = dy = dz, nx = ny = nz.

#     Parameters
#     ----------
#     atomic_numbers : 1d tensor
#         Atomic numbers. Hydrogen is 1.
#     centered_coords : 2d tensor
#         xyz coordinates corresponding to each entry in atomic_numbers. Shape of
#         (len(atomic_numbers), 3).
#     n_xyz : array-like
#         Number of pixels along x,y,z: (nx, ny, nz) of main volume.
#     dx : array-like
#         Pixel length along x,y,z: (dx,dy,dz) of main volume.
#     atom_size_px : int
#         Number of main volume pixels to sufficiently represent an atom. If None,
#         will assume 3A diameter per atom, and computed required number of pixels
#         accordingly.
#     super_sampling_factor : int
#         The supersampling factor to compute the potentials. For example, if main
#         volume has pixel size of 1A, then potentials will be first computed on a
#         grid of 1A/super_sampling_factor before binning back to 1A pixels. Must be
#         even to avoid singularity at 0, and larger than 4 (Kirkland's rule of thumb,
#         Chapter 5, after Fig 5.15.).
#     convention : str
#         The origin convention for main volume only. The super-sampled grid for
#         atomic potentials will always be even-valued and symmetric to avoid the
#         singularity at 0.
#     method : str
#         '3d' - Does not snap atom to nearest voxel. Computes each atom's 3D
#         potential individually based on the local super-sampled coordinate grid.

#         'snapped-3d' - Assumes each atom snaps to the nearest voxel defined on a
#         rectangular grid. Each 3D potential is first super-sampled on a finer grid
#         before averaging the pixels to insert into the main volume.

#         '2d' - Snaps atom only to nearest z-plane, but maintains it's x,y
#         coordinates. Computes each atom's 2D potential individually based on the
#         local super-sampled coordinate grid.

#         'snapped-2d' - Assumes each atom snaps to the nearest voxel defined on a
#         rectangular grid. Further assumes each atom can be represented by its
#         projected 2D potential. This 2D potential is first super-sampled on a finer
#         grid before averaging the pixels to insert into the main volume.

#     Returns
#     -------
#     potential_volume : 3d tensor
#         The sampled potential volume.
#     """
#     # create main volume coordinate system
#     nx, ny, nz = n_xyz
#     x, y, z, X, Y, Z = grid_3d(
#         (nx, ny, nz), dx, convention=convention
#     )

#     # create super-sampled (ss) coordinate system
#     if atom_size_px is None:
#         # forces odd number to ensure central pixel exists.
#         atom_size_px = int(torch.ceil(torch.tensor(3 / dx)) // 2 * 2 + 1)
#     ssn = atom_size_px * super_sampling_factor
#     ssdx = dx / super_sampling_factor

#     if method == "3d" or method == "snapped-3d":
#         sx, sy, sz, sX, sY, sZ = grid_3d(
#             (ssn, ssn, ssn), (ssdx, ssdx, ssdx), convention="torch"
#         )
#         if method == "snapped-3d":
#             sR = torch.sqrt(sX**2 + sY**2 + sZ**2)
#     elif method == "2d" or method == "snapped-2d":
#         sx, sy, sX, sY = grid_2d(
#             (ssn, ssn), (ssdx, ssdx), convention="torch"
#         )
#         if method == "snapped-2d":
#             sR = torch.sqrt(sX**2 + sY**2)

#     # for binning super-sampled grids to main volume grid.
#     avgpool2d = torch.nn.AvgPool2d(super_sampling_factor, stride=super_sampling_factor)
#     avgpool3d = torch.nn.AvgPool3d(super_sampling_factor, stride=super_sampling_factor)

#     # For snapped methods, compute unique element potentials on ss-grid and average
#     # onto main volume grid
#     if method == "snapped-2d":
#         sampled_2dpot_dict = {}
#         for an in torch.unique(atomic_numbers):
#             pot = kirkland_atomic_potential_2d(int(an), sR)
#             sampled_2dpot_dict[int(an)] = avgpool2d(pot[None, None]).squeeze()
#     elif method == "snapped-3d":
#         sampled_3dpot_dict = {}
#         for an in torch.unique(atomic_numbers):
#             pot = kirkland_atomic_potential_3d(int(an), sR)
#             # note the multiplicative factor of dx to properly scale for
#             # projection/multislice to match 2d version above.
#             sampled_3dpot_dict[int(an)] = avgpool3d(pot[None, None]).squeeze() * dx

#     # insert atomic potentials into main volume.
#     potential_volume = torch.zeros(nz, ny, nx)
#     occupancy = torch.zeros(nz, ny, nx, dtype=torch.bool)
#     for an, cc in tqdm(zip(atomic_numbers, centered_coords), disable=disable_tqdm):
#         xi, yi, zi = nearest_index(x, y, z, cc[0], cc[1], cc[2])

#         # don't insert if bounding box of atom falls outside of main volume grid.
#         if (
#             (zi - atom_size_px // 2) < 0
#             or zi - atom_size_px // 2 + atom_size_px > nz
#             or (yi - atom_size_px // 2) < 0
#             or yi - atom_size_px // 2 + atom_size_px > ny
#             or (xi - atom_size_px // 2) < 0
#             or xi - atom_size_px // 2 + atom_size_px > nx
#         ):
#             pass
#         else:
#             # update occupancy
#             occupancy[zi, yi, xi] = True

#             # insert atoms
#             if method == "3d":
#                 # relative 3D origin of the atom w.r.t. neighbouring voxels.
#                 x_ro = cc[0] - x[xi]
#                 y_ro = cc[1] - y[yi]
#                 z_ro = cc[2] - z[zi]
#                 sR = torch.sqrt((sX - x_ro) ** 2 + (sY - y_ro) ** 2 + (sZ - z_ro) ** 2)
#                 sspot = kirkland_atomic_potential_3d(int(an), sR)
#                 pot = avgpool3d(sspot[None, None]).squeeze() * dx

#                 potential_volume[
#                     zi - atom_size_px // 2 : zi - atom_size_px // 2 + atom_size_px,
#                     yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
#                     xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
#                 ] += pot
#             elif method == "snapped-3d":
#                 potential_volume[
#                     zi - atom_size_px // 2 : zi - atom_size_px // 2 + atom_size_px,
#                     yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
#                     xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
#                 ] += sampled_3dpot_dict[int(an)]
#             elif method == "2d":
#                 # relative 2D origin of the atom w.r.t. neighbouring voxels.
#                 x_ro = cc[0] - x[xi]
#                 y_ro = cc[1] - y[yi]
#                 sR = torch.sqrt((sX - x_ro) ** 2 + (sY - y_ro) ** 2)
#                 sspot = kirkland_atomic_potential_2d(int(an), sR)
#                 pot = avgpool2d(sspot[None, None]).squeeze()

#                 potential_volume[
#                     zi,
#                     yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
#                     xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
#                 ] += pot
#             elif method == "snapped-2d":
#                 potential_volume[
#                     zi,
#                     yi - atom_size_px // 2 : yi - atom_size_px // 2 + atom_size_px,
#                     xi - atom_size_px // 2 : xi - atom_size_px // 2 + atom_size_px,
#                 ] += sampled_2dpot_dict[int(an)]
#     return potential_volume, occupancy