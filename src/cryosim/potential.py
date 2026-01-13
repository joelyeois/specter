import torch
import torch.nn.functional as F
from rich.progress import track, Progress
import lightning as L
import gemmi

from .array_utils import (
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
    shtyrov_atomic_potential_3d,
)
from .fft_tools import fftconvolve
import numpy as np


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
        disable=disable_tqdm,
    ):
        # Update the description dynamically per element
        track.description = f"Building element {atom_symbol(int(elem))}"
        atomic_indices = torch.squeeze(torch.argwhere(atomic_numbers == elem))

        # populate elemental volume with delta function atoms
        # soft_voxelize_atoms is differentiable w.r.t. coordinates.
        temp_vol = soft_voxelize_coordinates(
            centered_coords[atomic_indices].reshape(-1, 3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx,
        )
        occupancy = occupancy | (temp_vol > 0)

        # get potential kernel for this element
        pot = kirkland_atomic_potential_3d(int(elem), sR)

        # convolve
        if ssf != 1:
            pot = avgpool3d(pot[None, None]) * dx
            pot = pot.squeeze(0).squeeze(0)
        potential_volume += fftconvolve(temp_vol, pot, mode="same")
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
            centered_coords[atomic_indices].reshape(-1, 3),
            grid_shape=(nz, ny, nx),
            voxel_size=dx,
        )
        occupancy = occupancy | (temp_vol > 0)

        # get potential kernel for this element
        pot = kirkland_atomic_potential_2d(int(elem), sR)

        # convolve
        if ssf != 1:
            pot = avgpool2d(pot[None, None]) * dx
            pot = pot.squeeze(0).squeeze(0)

        # batch 2D convolve
        temp_vol_b = temp_vol.unsqueeze(1)  # (nz, 1, ny, nx)
        pot_b = pot.unsqueeze(0).unsqueeze(0)  # (1, 1, ky, kx)
        convolved = F.conv2d(temp_vol_b, pot_b, padding="same")
        potential_volume += convolved.squeeze(1)  # (nz, ny, nx)
    return potential_volume, sR, atomic_potentials


class PotentialBuilder(L.LightningModule):
    """
    Lightning module for building 3D electrostatic potential volumes from atomic coordinates.

    Computes potentials using supersampled atomic potential kernels and
    convolution, supporting multiple parameterizations (Kirkland, Lobato, Shtyrov).

    Parameters
    ----------
    n_xyz : int or tuple of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Pixel/voxel size in Å.
    atomic_numbers : torch.Tensor
        Atomic numbers of all atoms in structure.
    verbose : bool, optional
        Enable progress bars during computation. Default is True.
    parameterization : str, optional
        Atomic potential parameterization: 'kirkland', 'lobato', or 'shtyrov'.
        Default is 'kirkland'.
    conv_backend : str, optional
        Convolution backend: 'fftconvolve' or 'conv3d'. Default is 'fftconvolve'.
    trainable : bool, optional
        Whether parameters are trainable. Default is False.
    mmcif_filepath : str, optional
        Path to mmCIF file for Shtyrov parameterization. Default is None.

    Attributes
    ----------
    atomic_potentials_2d : torch.Tensor
        Precomputed 2D atomic potentials for each unique element.
    atomic_potentials_3d : torch.Tensor
        Precomputed 3D atomic potentials for each unique element.
    """

    def __init__(
        self,
        n_xyz,
        dx,
        atomic_numbers,
        verbose=True,
        parameterization="kirkland",
        conv_backend="fftconvolve",
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
        self.register_buffer("sR_2d", sR_2d)
        self.register_buffer("sR_3d", sR_3d)

        # create atomic potentials
        self.atomic_numbers = atomic_numbers
        self.unique_elements = torch.unique(atomic_numbers)
        atomic_potentials_2d = torch.empty(
            len(self.unique_elements), self.ssn // self.ssf, self.ssn // self.ssf
        )
        atomic_potentials_3d = torch.empty(
            len(self.unique_elements),
            self.ssn // self.ssf,
            self.ssn // self.ssf,
            self.ssn // self.ssf,
        )
        self.register_buffer("atomic_potentials_2d", atomic_potentials_2d)
        self.register_buffer("atomic_potentials_3d", atomic_potentials_3d)

        self.parameterization = parameterization
        self.avgpool2d = torch.nn.AvgPool2d(self.ssf, stride=self.ssf)
        self.avgpool3d = torch.nn.AvgPool3d(self.ssf, stride=self.ssf)
        if parameterization in ("kirkland", "lobato"):
            self.get_2d_atomic_potentials()
        self.get_3d_atomic_potentials()

    def get_2d_atomic_potentials(self, unique_elements=None):
        """
        Compute and cache 2D atomic potential kernels for unique elements.

        Parameters
        ----------
        unique_elements : torch.Tensor, optional
            Elements to compute potentials for. If None, uses all unique
            elements from initialization. Default is None.

        Notes
        -----
        Potentials are supersampled and downsampled to main grid resolution.
        Results are stored in `self.atomic_potentials_2d`.
        """
        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            # update unique elements
            self.unique_elements = unique_elements
            self.atomic_potentials = torch.empty(
                len(unique_elements), self.ssn // self.ssf, self.ssn // self.ssf
            )

        # fetch potential kernels
        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == "kirkland":
                pot = kirkland_atomic_potential_2d(int(elem), self.sR_2d)
            elif self.parameterization == "lobato":
                pot = lobato_atomic_potential_2d(int(elem), self.sR_2d)

            if self.ssf != 1:
                pot = self.avgpool2d(pot[None, None]) * self.dx
                pot = pot.squeeze(0).squeeze(0)

            self.atomic_potentials_2d[i] = pot

    def get_3d_atomic_potentials(self, unique_elements=None):
        """
        Compute and cache 3D atomic potential kernels for unique elements.

        Parameters
        ----------
        unique_elements : torch.Tensor, optional
            Elements to compute potentials for. If None, uses all unique
            elements from initialization. Default is None.

        Notes
        -----
        Potentials are supersampled and downsampled to main grid resolution.
        Results are stored in `self.atomic_potentials_3d`.
        Supports Kirkland, Lobato, and Shtyrov parameterizations.
        """
        if unique_elements is None:
            unique_elements = self.unique_elements
        else:
            # update unique elements
            self.unique_elements = unique_elements
            self.atomic_potentials = torch.empty(
                len(unique_elements),
                self.ssn // self.ssf,
                self.ssn // self.ssf,
                self.ssn // self.ssf,
            )

        # fetch potential kernels
        for i, elem in enumerate(self.unique_elements):
            if self.parameterization == "kirkland":
                pot = kirkland_atomic_potential_3d(int(elem), self.sR_3d)
            elif self.parameterization == "lobato":
                pot = lobato_atomic_potential_3d(int(elem), self.sR_3d)
            elif self.parameterization == "shtyrov":
                if self.mmcif_filepath is None:
                    raise ValueError("mmcif_filepath must be specified.")
                else:
                    pot = shtyrov_atomic_potential_3d(
                        int(elem), self.sR_3d, self.mmcif_filepath
                    )

            if self.ssf != 1:
                pot = self.avgpool3d(pot[None, None]) * self.dx
                pot = pot.squeeze(0).squeeze(0)

            self.atomic_potentials_3d[i] = pot

    def forward(self, coordinates, method="3d", conv_backend=None):
        """
        Build potential volume(s) from atomic coordinates.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates. Shape (N, 3) for single volume or (B, N, 3)
            for batch of volumes.
        method : str, optional
            Voxelization method: '2d' (soft XY, hard Z) or '3d' (trilinear).
            Default is '3d'.
        conv_backend : str, optional
            Convolution backend override. Default is None (uses self.conv_backend).

        Returns
        -------
        potential_volume : torch.Tensor
            Electrostatic potential volume(s). Shape (nz, ny, nx) for single
            input or (B, nz, ny, nx) for batched input.

        Notes
        -----
        Uses soft voxelization followed by convolution with precomputed
        atomic potential kernels. The 2d method is faster but less accurate.
        """
        if conv_backend is None:
            conv_backend = self.conv_backend
        coordinates = coordinates.to(self.device)
        self.method = method

        # Detect batch
        if coordinates.ndim == 2:  # (N,3) -> add batch dimension
            coordinates = coordinates.unsqueeze(0)

        B, N, _ = coordinates.shape

        # insert atomic potentials into main volume.
        potential_volume = torch.zeros(
            (B, self.nz, self.ny, self.nx), device=self.device
        )
        self.occupancy = torch.zeros((B, self.nz, self.ny, self.nx), dtype=torch.bool)

        with Progress(transient=True) as progress:
            # Create a single task for the outer loop
            task = progress.add_task(
                "Building element ...", total=len(self.unique_elements)
            )
            for i, elem in enumerate(self.unique_elements):
                progress.update(
                    task,
                    description=f"Building element {atom_symbol(int(elem))}",
                    advance=1,
                )
                atomic_indices = torch.squeeze(
                    torch.argwhere(self.atomic_numbers == elem)
                )

                # Select atomic coordinates for this element
                coords_elem = coordinates[:, atomic_indices, :]  # (B, Nelem, 3)

                if method == "2d":
                    temp_vol = soft_voxelize_xy_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx,
                    )

                    # Flatten B and Z for conv2d
                    temp_vol_flat = temp_vol.reshape(
                        -1, 1, self.ny, self.nx
                    )  # (B*Z, 1, Y, X)

                    # Kernel: (1, 1, ky, kx)
                    pot_b = (
                        self.atomic_potentials_2d[i].unsqueeze(0).unsqueeze(0)
                    )  # (1,1,ky,kx)

                    # Perform conv2d
                    convolved_flat = F.conv2d(
                        temp_vol_flat, pot_b, padding="same"
                    )  # (B*Z, 1, Y, X)

                    # Reshape back to (B, Z, Y, X)
                    convolved = convolved_flat.reshape(B, self.nz, self.ny, self.nx)

                    # Add to potential volume
                    potential_volume += convolved

                elif method == "3d":
                    temp_vol = soft_voxelize_coordinates(
                        coords_elem,
                        grid_shape=(self.nz, self.ny, self.nx),
                        voxel_size=self.dx,
                    )

                    # convolve
                    # Convolve 3D potentials per batch
                    if conv_backend == "fftconvolve":
                        # for b in track(range(B), description='Convolving atoms', transient=True):
                        for b in range(B):
                            potential_volume[b] += fftconvolve(
                                temp_vol[b], self.atomic_potentials_3d[i], mode="same"
                            )

                    # using conv3d instead
                    elif conv_backend == "conv3d":
                        vol_b = temp_vol.unsqueeze(1)  # (B,1,nz,ny,nx)
                        kernel = (
                            self.atomic_potentials_3d[i].unsqueeze(0).unsqueeze(0)
                        )  # (1,1,kz,ky,kx)
                        # Use conv3d with groups=1
                        convolved = F.conv3d(
                            vol_b, kernel, padding="same"
                        )  # (B,1,nz,ny,nx)
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


class GemmiPotentialBuilder:
    """
    Build electrostatic potential volumes using Gemmi library.

    Uses Gemmi's density calculator with Gaussian atomic form factors.
    Supports custom scattering factors from mmCIF files.

    Parameters
    ----------
    n_xyz : int or tuple of int
        Grid size (nx, ny, nz). If int, assumes cubic grid.
    dx : float
        Pixel/voxel size in Å.
    atomic_numbers : torch.Tensor, optional
        Atomic numbers of all atoms. Default is None.
    b_factor : float, optional
        Isotropic B-factor. Default is 20.0.

    Attributes
    ----------
    dencalc : gemmi.DensityCalculatorE
        Gemmi density calculator instance.
    translate_to_center : torch.Tensor
        Translation vector to center atoms in grid.
    c1 : float
        Scaling factor for electrostatic potential (2π*e*a₀).
    """

    def __init__(self, n_xyz, dx, atomic_numbers=None, b_factor=None):
        if isinstance(n_xyz, (int, float)):
            self.nx = self.ny = self.nz = n_xyz
        else:
            self.nx, self.ny, self.nz = n_xyz
        self.dx = dx
        self.b_factor = b_factor
        self.translate_to_center = torch.tensor(
            [[self.nx // 2 * dx, self.ny // 2 * dx, self.nz // 2 * dx]]
        )
        self.atomic_numbers = atomic_numbers

        # prepare density calculator
        self.dencalc = gemmi.DensityCalculatorE()

        # setup box size
        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,  # for a cubic cell
        )
        self.dencalc.grid.unit_cell = unit_cell
        self.dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        self.dencalc.grid.set_size(self.nx, self.ny, self.nz)

        # scaling prefactor
        a0 = 0.529  # Bohr radius, [Å]
        e = 14.4  # electron charge, [V·Å]
        self.c1 = 2 * torch.pi * e * a0

    def build_model(self, atom_coordinates, atom_elements):
        """
        Build Gemmi structure model from atomic coordinates and elements.

        Parameters
        ----------
        atom_coordinates : torch.Tensor or np.ndarray
            Atomic coordinates in Å, shape (N, 3).
        atom_elements : torch.Tensor or np.ndarray
            Atomic numbers, shape (N,).

        Returns
        -------
        model : gemmi.Model
            Gemmi model containing atoms within grid bounds.

        Notes
        -----
        Filters out atoms outside the grid boundaries.
        All atoms are assigned to chain A, residue 1.
        """
        # Create placeholder structure
        st = gemmi.Structure()
        model = st.add_model(gemmi.Model(1))
        chain = model.add_chain("A")  # add chain A
        res = chain.add_residue(gemmi.Residue())

        # Boolean mask for atoms inside the box
        mask = (
            (atom_coordinates[:, 0] >= 0.0)
            & (atom_coordinates[:, 0] <= self.nx * self.dx)
            & (atom_coordinates[:, 1] >= 0.0)
            & (atom_coordinates[:, 1] <= self.ny * self.dx)
            & (atom_coordinates[:, 2] >= 0.0)
            & (atom_coordinates[:, 2] <= self.nz * self.dx)
        )

        # Apply mask to coordinates and elements
        filtered_coords = atom_coordinates[mask]
        filtered_elements = atom_elements[mask]

        for i, (pos, z) in enumerate(zip(filtered_coords, filtered_elements), start=1):
            # if not (0. <= pos[0] <= self.nx * self.dx and 0. <= pos[1] <= self.ny * self.dx and 0. <= pos[2] <= self.nz * self.dx):
            #     continue
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
            atom.element = gemmi.Element(
                int(z)
            )  # Element constructed from atomic number
            atom.occ = 1.0
            atom.b_iso = self.b_factor
            atom.serial = i
            # Add atom to residue (Python API returns reference to added atom)
            res.add_atom(atom)
        return model

    def build_dencalc(self):
        """
        Build a fresh Gemmi density calculator with current grid settings.

        Returns
        -------
        dencalc : gemmi.DensityCalculatorE
            Configured density calculator.

        Notes
        -----
        Creates a new calculator to avoid state contamination between
        multiple potential calculations.
        """
        # prepare density calculator
        dencalc = gemmi.DensityCalculatorE()

        # setup box size
        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,  # for a cubic cell
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(self.nx, self.ny, self.nz)
        return dencalc

    def build_potential_from_custom_mmcif(self, mmcif_filepath):
        """
        Build potential using custom scattering factors from mmCIF file.

        Reads scattering factor coefficients from '_lmb_scat_coef' table
        in mmCIF file for high-accuracy potential calculations.

        Parameters
        ----------
        mmcif_filepath : str
            Path to mmCIF file containing structure and scattering factors.

        Returns
        -------
        potential : torch.Tensor
            Electrostatic potential volume, shape (nz, ny, nx).

        Notes
        -----
        Uses only the first atom from the structure and applies custom
        form factors from the mmCIF file. Atoms are recentered to grid center.
        """
        # Read the CIF structure file
        st = gemmi.read_structure(mmcif_filepath)

        # --- keep only the first atom ---
        # first_atom = st[0][0][0][0]
        # new_st = gemmi.Structure()
        # model = new_st.add_model(gemmi.Model(1))
        # chain = model.add_chain("A")
        # residue = chain.add_residue(gemmi.Residue())
        # residue.add_atom(first_atom.clone())
        # st = new_st
        # --------------------------------

        # Extract scattering factors from table
        block = gemmi.cif.read_file(mmcif_filepath).sole_block()
        ctable = block.find(
            "_lmb_scat_coef.",
            [
                "coef_a1",
                "coef_a2",
                "coef_a3",
                "coef_a4",
                "coef_a5",
                "coef_b1",
                "coef_b2",
                "coef_b3",
                "coef_b4",
                "coef_b5",
            ],
        )

        # restructure scattering factors
        coefs = np.empty((len(ctable), 10))
        for ind, row in enumerate(ctable):
            coefs[ind] = [float(field) for field in row]
        max_serial = max(cra.atom.serial for cra in st[0].all())
        custom_form_factors = np.zeros((max_serial + 1, 10))
        itable = block.find("_atom_site.", ["id", "scat_id"])
        for row in itable:
            serial, scat_id = row
            custom_form_factors[int(serial)] = coefs[int(scat_id)]
            # print(scat_id)
            # break
        gemmi.set_custom_form_factors(custom_form_factors)
        dencalc = gemmi.DensityCalculatorC()

        # Recenter atoms
        coords = np.array([cra.atom.pos for cra in st[0].all()])  # (N, 3)
        center_geom = coords.mean(axis=0)
        # Apply shift to all atoms
        for cra in st[0].all():
            translate = -np.asarray(
                center_geom.tolist()
            ) + self.translate_to_center.numpy().squeeze(0)
            cra.atom.pos += gemmi.Position(
                float(translate[0]), float(translate[1]), float(translate[2])
            )
            if self.b_factor is not None:
                cra.atom.b_iso = self.b_factor

        if self.b_factor is None:
            print(
                "Using default B-factor in mmcif file. Set b_factor to 0 if not intended."
            )

        # setup box size
        unit_cell = gemmi.UnitCell(
            self.nx * self.dx,
            self.ny * self.dx,
            self.nz * self.dx,
            90.0,
            90.0,
            90.0,  # for a cubic cell
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(self.nx, self.ny, self.nz)
        dencalc.put_model_density_on_grid(st[0])
        return self.c1 * torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    def _build_single_potential(self, coords_elements_tuple):
        """
        Build potential for a single set of coordinates (non-parallel).

        Parameters
        ----------
        coords_elements_tuple : tuple
            (coordinates, elements) where coordinates is (N, 3) and
            elements is (N,).

        Returns
        -------
        potential : torch.Tensor
            Potential volume with shape (nz, ny, nx).
        """
        coords, elements = coords_elements_tuple
        model = self.build_model(coords, elements)
        dencalc = self.build_dencalc()
        dencalc.put_model_density_on_grid(model)
        return torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    @staticmethod
    def _build_parallelizable_single_potential(args):
        """
        Build potential for parallel processing (static method).

        Parameters
        ----------
        args : tuple
            (coords, elements, nx, ny, nz, dx, b_factor) containing all
            necessary parameters for building potential.

        Returns
        -------
        potential : torch.Tensor
            Potential volume with shape (nz, ny, nx).

        Notes
        -----
        Static method to enable multiprocessing with spawn context.
        Creates fresh Gemmi objects to avoid pickling issues.
        """
        coords, elements, nx, ny, nz, dx, b_factor = args
        # Create placeholder structure
        st = gemmi.Structure()
        model = st.add_model(gemmi.Model(1))
        chain = model.add_chain("A")  # add chain A
        res = chain.add_residue(gemmi.Residue())

        # Boolean mask for atoms inside the box
        mask = (
            (coords[:, 0] >= 0.0)
            & (coords[:, 0] <= nx * dx)
            & (coords[:, 1] >= 0.0)
            & (coords[:, 1] <= ny * dx)
            & (coords[:, 2] >= 0.0)
            & (coords[:, 2] <= nz * dx)
        )

        # Apply mask to coordinates and elements
        filtered_coords = coords[mask]
        filtered_elements = elements[mask]

        for i, (pos, z) in enumerate(zip(filtered_coords, filtered_elements), start=1):
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
            atom.element = gemmi.Element(
                int(z)
            )  # Element constructed from atomic number
            atom.occ = 1.0
            atom.b_iso = b_factor
            atom.serial = i
            # Add atom to residue (Python API returns reference to added atom)
            res.add_atom(atom)

        # prepare density calculator
        dencalc = gemmi.DensityCalculatorE()

        # setup box size
        unit_cell = gemmi.UnitCell(
            nx * dx,
            ny * dx,
            nz * dx,
            90.0,
            90.0,
            90.0,  # for a cubic cell
        )
        dencalc.grid.unit_cell = unit_cell
        dencalc.grid.spacegroup = gemmi.SpaceGroup("P1")
        dencalc.grid.set_size(nx, ny, nz)
        dencalc.put_model_density_on_grid(model)
        return torch.as_tensor(dencalc.grid.array).transpose(0, 2)

    def build_potential(self, atom_coordinates, atomic_numbers=None, n_processes=None):
        """
        Build electrostatic potential volume from atomic coordinates.

        Parameters
        ----------
        atom_coordinates : torch.Tensor
            Atomic coordinates in Ų, shape (N, 3). Centered at origin.
        atomic_numbers : torch.Tensor, optional
            Atomic numbers, shape (N,). If None, uses self.atomic_numbers.
            Default is None.
        n_processes : int, optional
            Number of parallel processes. If None, runs serially.
            Default is None.

        Returns
        -------
        potential : torch.Tensor
            Electrostatic potential volume in Volt-Ångströms, shape (nz, ny, nx).

        Notes
        -----
        Coordinates are automatically translated to place origin at grid center.
        Parallel processing splits atoms across processes and sums results.
        Scaling factor c1 = 2π*e*a₀ is applied to match physical units.
        """
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers
        else:
            self.atomic_numbers = atomic_numbers
        # translate coordinates
        translated_coordinates = atom_coordinates + self.translate_to_center

        if n_processes is None:
            vol = self._build_single_potential((translated_coordinates, atomic_numbers))
            return self.c1 * vol

        else:
            # split atoms into roughly equal chunks
            chunks_coords = torch.split(translated_coordinates, n_processes)
            chunks_elements = torch.split(atomic_numbers, n_processes)
            # pack arguments into single tuples
            args_list = [
                (
                    chunks_coords[i],
                    chunks_elements[i],
                    self.nx,
                    self.ny,
                    self.nz,
                    self.dx,
                    self.b_factor,
                )
                for i in range(n_processes)
            ]

            import multiprocessing as mp

            with mp.get_context("spawn").Pool(processes=n_processes) as pool:
                results = pool.map(
                    self._build_parallelizable_single_potential, args_list
                )

            # sum volumes from all processes
            total_volume = torch.stack(results).sum(0)
            return self.c1 * total_volume
