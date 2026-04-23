from __future__ import annotations

import os
import time
import warnings
from typing import Any, Literal, Optional, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchinterp1d import interp1d

from .progress import track

from . import potential, rotations
from .arrays import (
    grid_3d,
    radial_grid_3d,
    radial_profile_3d,
    real_to_kgrid_3d,
    soft_voxelize_coordinates,
)
from .atom import kirkland_atomic_potential_3d, lobato_atomic_potential_3d
from .fft import fft3, fftconvolve
from .coords import radial_distribution_function
from specter.pdb import PDB

avogadro = 6.02214076e23
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / Å³]


def water_molecule_coordinates(
    bond_angle: float = 105.0, bond_length: float = 0.9572
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Return coordinates of a single H2O molecule with oxygen at the origin.

    Parameters
    ----------
    bond_angle : float, optional
        Bond angle between the O-H bonds in degrees. Default is 105.0.
    bond_length : float, optional
        O-H bond length in Å. Default is 0.9572.

    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers of the three atoms [8, 1, 1].
    coordinates : torch.Tensor
        xyz coordinates of O, H1, H2 with shape (3, 3).
    """
    O_xyz = torch.tensor([0.0, 0.0, 0.0])
    angle_rad = torch.tensor(bond_angle) / 180 * torch.pi
    y = bond_length * torch.cos(angle_rad / 2)
    x = bond_length * torch.sin(angle_rad / 2)
    H1_xyz = torch.tensor([x, y, 0.0])
    H2_xyz = torch.tensor([-x, y, 0.0])
    coordinates = torch.stack((O_xyz, H1_xyz, H2_xyz))
    atomic_numbers = np.array([8, 1, 1])
    return atomic_numbers, coordinates


def create_n_randomly_rotated_water_molecules(
    n: int, **kwargs: Any
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Create n randomly rotated water molecules.

    Parameters
    ----------
    n : int
        Number of molecules to create.
    **kwargs
        Passed to :func:`water_molecule_coordinates` (``bond_angle``, ``bond_length``).

    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers of all atoms, shape (3*n,).
    coordinates : torch.Tensor
        Atomic coordinates of all atoms, shape (3*n, 3).
    """
    quats = torch.stack([rotations.random_quaternion() for _ in range(n)])
    atomic_numbers, coords = water_molecule_coordinates(**kwargs)

    O_coordinates = coords[0].repeat(n, 1)
    H1_coordinates = rotations.rotate_coordinates(coords[1], quats)
    H2_coordinates = rotations.rotate_coordinates(coords[2], quats)

    all_coords = torch.zeros(n * 3, 3)
    all_coords[0::3] = O_coordinates
    all_coords[1::3] = H1_coordinates
    all_coords[2::3] = H2_coordinates

    all_atomic_numbers = np.array([8, 1, 1] * n)
    return all_atomic_numbers, all_coords


def volume_of_ice(
    n_xyz: Sequence[int], d_xyz: Sequence[float]
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Generate atomic coordinates for a volume of amorphous ice.

    Molecules are placed at random positions (overlaps possible) with random
    orientations. Oxygen is at the molecule centre; hydrogens are rotated randomly.

    Parameters
    ----------
    n_xyz : sequence of int
        Number of voxels (nx, ny, nz).
    d_xyz : sequence of float
        Voxel size in Å (dx, dy, dz).

    Returns
    -------
    atomic_numbers : np.ndarray
        Atomic numbers of all ice atoms.
    coordinates : torch.Tensor
        Coordinates of all ice atoms in Å.
    """
    nx, ny, nz = n_xyz
    dx, dy, dz = d_xyz
    total_vol = nx * ny * nz * dx * dy * dz
    n_molecules = int(ndensity_of_amorphous_ice * total_vol)

    x_ice = (torch.rand(n_molecules) - 0.5) * dx * nx
    y_ice = (torch.rand(n_molecules) - 0.5) * dy * ny
    z_ice = (torch.rand(n_molecules) - 0.5) * dz * nz
    centers = torch.repeat_interleave(
        torch.stack((x_ice, y_ice, z_ice), dim=1), 3, dim=0
    )

    atomic_numbers, coords = create_n_randomly_rotated_water_molecules(n_molecules)
    coords += centers
    return atomic_numbers, coords


def rfftn(array: torch.Tensor) -> torch.Tensor:
    """
    Compute N-dimensional real-input Fourier transform with centering.

    Wraps torch.fft.rfftn with FFT shifting to ensure the zero-frequency component
    is centered, handling the last dimension which is complex-valued differently.

    Parameters
    ----------
    array : torch.Tensor
        Input real-valued tensor.

    Returns
    -------
    fft : torch.Tensor
        Complex-valued tensor containing the Fourier coefficients.
        Zero frequency is centered.
    """
    return torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2),
    )


def torch_peak_local_max(
    image: torch.Tensor, min_distance: int = 1, num_peaks: int | None = None
) -> torch.LongTensor:
    """
    Find local maxima in batched 3D images and return fixed number of peaks per batch.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (B, D, H, W).
    min_distance : int, optional
        Minimum separation between peaks (voxels). Default is 1.
    num_peaks : int, optional
        Number of peaks to return per batch (must be <= total peaks in each batch).
        If None, uses the minimum number of peaks found in any batch item. Default is None.

    Returns
    -------
    peaks : torch.LongTensor
        Peak coordinates (z, y, x) for each batch. Shape (B, num_peaks, 3).
    """
    B, D, H, W = image.shape
    x = image.unsqueeze(1)  # (B, 1, D, H, W)
    k = 2 * min_distance + 1
    pooled = F.max_pool3d(x, kernel_size=k, stride=1, padding=min_distance)
    mask = (x == pooled).squeeze(1)  # (B, D, H, W)

    # Flatten spatial dims
    flat_mask = mask.view(B, -1)
    flat_image = image.view(B, -1)

    # Mask non-maxima
    flat_image_masked = flat_image.clone()
    flat_image_masked[~flat_mask] = -float("inf")

    if num_peaks is None:
        num_peaks = flat_mask.sum(dim=1).min().item()  # take min available peaks

    # Top-k per batch
    topk_vals, topk_idx = flat_image_masked.topk(num_peaks, dim=1)

    # Convert flat indices back to 3D coords
    z = topk_idx // (H * W)
    y = (topk_idx % (H * W)) // W
    x_ = topk_idx % W

    peaks = torch.stack([z, y, x_], dim=2)  # (B, num_peaks, 3)
    return peaks


class Icemaker(L.LightningModule):
    """
    Generates 3D ice volumes with water-like molecular structure.

    Based on molecular dynamics simulations. Provides methods to load simulation data, compute radial averages, and iteratively generate ice volumes that match a target Fourier amplitude kernel.

    Parameters
    ----------
    dx : float, optional
        Voxel size in Angstroms. Default is 0.5.
    n : int, optional
        Number of voxels in x and y dimensions. Default is 200.
    nz : float, optional
        Ice thickness in angstroms. If None, defaults to `n * dx`. Default is None.
    chunk_size : int, optional
        Size of chunks for processing large volumes. Default is None.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    parameterization : str, optional
        Atomic potential parameterization ('kirkland', 'lobato', 'shryov'). Default is 'kirkland'.
    min_distance : float, optional
        Minimum distance between water molecules in Angstroms. Default is 1.9.

    Attributes
    ----------
    ice_thickness : float
        Thickness of the ice slab in Angstroms.
    """

    def __init__(
        self,
        dx: float = 0.5,
        n: int = 200,
        nz: int | None = None,
        chunk_size: int | None = None,
        progressbars: bool = True,
        parameterization: str = "kirkland",
        min_distance: float = 1.9,
        correction_factor: float | None = None,
    ):
        super().__init__()

        # load 3D radial average of mdsim data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        self.saved_data_path = os.path.join(
            root_dir, "ice-data", "mdsim_f_radial_avg_400x400x400_0.25A.pt"
        )
        self.mdsim_dx = 0.25
        self.mdsim_n = 400
        self.mdsim_dk = 1 / self.mdsim_n / self.mdsim_dx
        self.get_mdsim_f_radial_avg(self.saved_data_path)
        self.chunk_size = chunk_size
        self.progressbars = progressbars
        self.parameterization = parameterization

        if nz is None:
            self.nz = n
        else:
            self.nz = nz
        self.dx = dx
        self.dk = 1 / n / dx
        self.n = n
        self.dv = dx**3
        self.nv = n**2 * self.nz
        self.v = self.dv * self.nv
        min_distance_vox = int(min_distance / dx)
        min_distance_actual = min_distance_vox * dx
        self.min_distance = min_distance
        self.correction_factor = correction_factor
        if correction_factor is None:
            if min_distance_actual == 0.0:
                # dx is coarser than min_distance — no voxel-level correction possible
                self.correction_factor = 1.0
            else:
                self.correction_factor = (min_distance / min_distance_actual) ** 3
        self.n_ice_molecules_theory = int(ndensity_of_amorphous_ice * self.v)
        self.n_ice_molecules = int(
            ndensity_of_amorphous_ice * self.v / self.correction_factor
        )

        # create k-space coordinates grid
        kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
        ky = kx
        kz = torch.fft.fftshift(torch.fft.fftfreq(self.nz, dx))
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        self.register_buffer("K", torch.sqrt(KX**2 + KY**2 + KZ**2))

        # pre-compute ice kernel for algorithm
        self.interpolate_mdsim_f_kernel()

        self.register_buffer("ice_kernel", self.create_ice_kernel())

    def get_mdsim(
        self,
        filepath: str,
        trim_size: int = 100,
        startframe: int = 10,
        endframe: int = 101,
    ) -> None:
        """
        Load MD simulation dump and convert atomic coordinates into voxel grid.

        Parameters
        ----------
        filepath : str
            Path to the MD simulation dump file.
        trim_size : int, optional
            Maximum half-size of the cube to retain around particle center. Default is 100.
        startframe : int, optional
            Frame index to start processing. Default is 10.
        endframe : int, optional
            Frame index to stop processing. Default is 101.
        """
        self.get_mdsim_file(filepath)
        mdsim_ice_deltas = []

        x, y, z, X, Y, Z = grid_3d(self.mdsim_n, self.mdsim_dx)

        self.mdsim_ice_coordinates = []
        for frame in track(
            self.mdsim_frame_indexes[startframe:endframe],
            disable=not self.progressbars,
        ):
            coordstart = frame + 9
            coords = self.get_coordinates_from_frame(coordstart)
            centered_coords = PDB.center_coordinates(coords)
            centered_coords = self.trim_coordinates(
                centered_coords, trim_size=trim_size
            )

            # mdsim_ice_delta = torch.zeros(self.mdsim_n, self.mdsim_n, self.mdsim_n)
            # for cc in centered_coords:
            #     xi, yi, zi = potential.nearest_index(x, y, z, cc[0], cc[1], cc[2])
            #     mdsim_ice_delta[zi, yi, xi] = 1

            # populate elemental volume with delta function atoms
            # soft_voxelize_atoms is differentiable w.r.t. coordinates.
            mdsim_ice_delta = soft_voxelize_coordinates(
                centered_coords.reshape(-1, 3),
                grid_shape=(self.mdsim_n, self.mdsim_n, self.mdsim_n),
                voxel_size=self.mdsim_dx,
            )

            mdsim_ice_deltas.append(mdsim_ice_delta)
            self.mdsim_ice_coordinates.append(centered_coords)
        self.mdsim_ice_deltas = torch.stack(mdsim_ice_deltas)

    def get_mdsim_file(self, filepath: str) -> None:
        """
        Read MD simulation dump file into memory and find timestep indices.

        Parameters
        ----------
        filepath : str
            Path to the MD simulation dump file.
        """
        with open(filepath) as f:
            self.lines = f.readlines()

        self.mdsim_frame_indexes = [
            i for i, x in track(enumerate(self.lines)) if x == "ITEM: TIMESTEP\n"
        ]

    def get_coordinates_from_frame(
        self,
        start_line_number: int,
        lines: list[str] | None = None,
        no_atoms: int = 128000,
    ) -> torch.Tensor:
        """
        Parse atom coordinates from a given frame in MD dump.

        Parameters
        ----------
        start_line_number : int
            Line number where atom coordinates start.
        lines : list of str, optional
            Pre-loaded file lines. If None, uses `self.lines`. Default is None.
        no_atoms : int, optional
            Number of atoms to read. Default is 128000.

        Returns
        -------
        coords : torch.Tensor
            Atomic coordinates (x, y, z) for the frame. Shape (no_atoms, 3).
        """
        if lines is None:
            lines = self.lines

        coords = torch.zeros(no_atoms, 3)
        for i, s in enumerate(lines[start_line_number : start_line_number + no_atoms]):
            id, typ, x, y, z = s.split()
            coords[i, 0] = float(x)
            coords[i, 1] = float(y)
            coords[i, 2] = float(z)
        return coords

    def trim_coordinates(
        self, coords: torch.Tensor, trim_size: float = 100
    ) -> torch.Tensor:
        """
        Trim coordinates to a cube of given size centered at origin.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates to trim. Shape (N, 3).
        trim_size : float, optional
            Side length of the cube to retain in Å. Default is 100.

        Returns
        -------
        trimmed_coords : torch.Tensor
            Coordinates within the cube. Shape (M, 3).
        """
        trimmed_coords = []
        for co in coords:
            if not (
                torch.abs(co[0]) > trim_size // 2
                or torch.abs(co[1]) > trim_size // 2
                or torch.abs(co[2]) > trim_size // 2
            ):
                trimmed_coords.append(co)
        trimmed_coords = torch.stack(trimmed_coords)
        return trimmed_coords

    def get_mdsim_averaged_f_kernel(
        self, filepath: str, source: Literal["dump", "torch"] = "torch"
    ) -> None:
        """
        Compute or load the 3D Fourier amplitude of the ice volume.

        Parameters
        ----------
        filepath : str
            Path to load precomputed Fourier amplitude tensor from, or MD dump path.
        source : str, optional
            - 'dump': Compute FFT from MD simulation dump.
            - 'torch': Load precomputed tensor from file.
            Default is 'torch'.
        """
        if source == "dump":
            self.get_mdsim(filepath, trim_size=100)
            self.mdsim_ice_deltas_f = []
            for mdsim_ice_delta in track(
                self.mdsim_ice_deltas, disable=not self.progressbars
            ):
                self.mdsim_ice_deltas_f.append(fft3(mdsim_ice_delta))
            self.mdsim_ice_deltas_f = torch.stack(self.mdsim_ice_deltas_f)
            self.mdsim_ice_deltas_f = torch.mean(
                torch.abs(self.mdsim_ice_deltas_f), dim=0
            )

        elif source == "torch":
            self.mdsim_ice_deltas_f = torch.load(filepath).to(self.device)

    def get_mdsim_f_radial_avg(self, saved_data_path: str | None = None) -> None:
        """
        Compute or load radial average of MD simulation Fourier amplitudes.

        Parameters
        ----------
        saved_data_path : str, optional
            Path to precomputed radial average. If None, compute from
            `self.mdsim_ice_deltas_f`. Default is None.
        """
        if saved_data_path is not None:
            mdsim_f_radial_avg = torch.load(saved_data_path)
            self.register_buffer("mdsim_f_radial_avg", mdsim_f_radial_avg)
        else:
            # compute 3D radial average of mdsim data
            self.mdsim_f_radial_avg = radial_profile_3d(self.mdsim_ice_deltas_f)
        mdsim_radial_k = torch.arange(len(self.mdsim_f_radial_avg)) * self.mdsim_dk
        self.register_buffer("mdsim_radial_k", mdsim_radial_k)

    def create_initial_ice_volume(self, batchsize: int = 1) -> torch.Tensor:
        """
        Create a random initial 3D ice volume with specified number of molecules.

        Parameters
        ----------
        batchsize : int, optional
            Number of ice volumes to generate. Default is 1.

        Returns
        -------
        ice_vol_init : torch.Tensor
            Binary tensor with 1 where ice molecules are placed.
            Shape (batchsize, nz, n, n).
        """

        # Preallocate batch tensor
        ice_vol_init = torch.zeros(
            batchsize, self.nz * self.n * self.n, device=self.device
        )

        # Randomly select indices for each batch volume
        idx = torch.randint(
            0, self.nz * self.n * self.n, (batchsize, self.n_ice_molecules)
        )

        # Scatter 1s at chosen indices
        batch_indices = (
            torch.arange(batchsize).unsqueeze(1).expand(-1, self.n_ice_molecules)
        )
        ice_vol_init[batch_indices, idx] = 1.0

        # Reshape to (B, nz, n, n)
        ice_vol_init = ice_vol_init.view(batchsize, self.nz, self.n, self.n)
        return ice_vol_init

    def _compute_interp_f_halfkernel(
        self, dx: float, n_ice_molecules: int
    ) -> torch.Tensor:
        """
        Compute the Fourier amplitude halfkernel for a given voxel size.

        Parameters
        ----------
        dx : float
            Voxel size in Angstroms.
        n_ice_molecules : int
            Number of ice molecules for this voxel size.

        Returns
        -------
        interp_f_halfkernel : torch.Tensor
            Half-kernel for use with rfftn, shape (nz, n, n//2 + 1).
        """
        # Create frequency grid for the given dx
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.n, dx))
        ky = kx
        kz = torch.fft.fftshift(torch.fft.fftfreq(self.nz, dx))
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        K = torch.sqrt(KX**2 + KY**2 + KZ**2)

        # Interpolate MD simulation data onto this grid
        interp = interp1d(
            self.mdsim_radial_k[1:], self.mdsim_f_radial_avg[1:], K.ravel()
        )

        # mdsim_f_radial_avg stores √S(k); scale to absolute amplitudes for this volume
        interp_f_kernel = interp.reshape(self.nz, self.n, self.n) * (
            n_ice_molecules**0.5
        )
        interp_f_kernel[self.nz // 2, self.n // 2, self.n // 2] = n_ice_molecules

        # Extract half kernel for rfftn
        return torch.flip(interp_f_kernel[:, :, : self.n // 2 + 1], dims=[2])

    def interpolate_mdsim_f_kernel(self) -> None:
        """
        Generate a 3D Fourier amplitude kernel for ice generation.

        Interpolates MD simulation radial averages to the current grid.
        Updates `self.interp_radial_k`, `self.interp_f_radial_avg`, `self.interp_f_kernel`,
        and `self.interp_f_halfkernel`.
        """

        # interpolate, exclude DC
        interp = interp1d(
            self.mdsim_radial_k[1:], self.mdsim_f_radial_avg[1:], self.K.ravel()
        )

        # mdsim_f_radial_avg stores √S(k); scale to absolute amplitudes for this volume
        interp_f_kernel = interp.reshape(self.nz, self.n, self.n) * (
            self.n_ice_molecules**0.5
        )
        interp_f_kernel[self.nz // 2, self.n // 2, self.n // 2] = self.n_ice_molecules
        self.register_buffer("interp_f_kernel", interp_f_kernel)

        # register half kernel for rfftn
        self.register_buffer(
            "interp_f_halfkernel",
            torch.flip(interp_f_kernel[:, :, : self.n // 2 + 1], dims=[2]),
        )

        # compute 3D radial average of interp data
        self.register_buffer("interp_f_radial_avg", radial_profile_3d(interp_f_kernel))
        self.register_buffer(
            "interp_radial_k", torch.arange(len(self.interp_f_radial_avg)) * self.dk
        )

    def generate_ice_deltas(
        self,
        niter: int = 5,
        min_distance: float | None = None,
        add_extra_molecules: bool = False,
        batchsize: int = 1,
        reduce_fraction: float = 1.0,
        dx: float | None = None,
    ) -> None:
        """
        Iteratively generate ice volume using Fourier amplitude kernel.

        Parameters
        ----------
        niter : int, optional
            Maximum number of iterations. Default is 5.
        min_distance : float, optional
            Minimum separation between molecules in Angstroms. If None, uses `self.min_distance`.
            Default is None.
        add_extra_molecules : bool, optional
            If True, randomly add extra molecules to satisfy density. Default is True.
        batchsize : int, optional
            Number of ice volumes to generate. Default is 1.
        reduce_fraction : float, optional
            Fraction of target number of molecules to initially target with peak finding.
            Default is 1.0.
        dx : float, optional
            Voxel size in Angstroms for interpreting the grid. If None, uses `self.dx`.
            Default is None. This affects the min_distance voxel calculation.

        Notes
        -----
        Updates `self.current_ice_vol`, `self.ice_coordinates`, `self.frob_norm`.
        """

        self.batchsize = batchsize

        # initialize
        ice_vol_init = self.create_initial_ice_volume(batchsize=batchsize)
        self.register_buffer("ice_vol_init", ice_vol_init)

        self.register_buffer("current_icedeltas", self.ice_vol_init.clone())
        self.niter = niter
        if min_distance is None:
            min_distance = self.min_distance
        if dx is None:
            dx = self.dx

        # Compute n_ice_molecules for the given dx
        dv = dx**3
        v = dv * self.nv
        min_distance_vox = int(min_distance / dx)
        min_distance_actual = min_distance_vox * dx
        correction_factor = (min_distance / min_distance_actual) ** 3
        n_ice_molecules = int(ndensity_of_amorphous_ice * v / correction_factor)

        self.frob_norm = []
        self.n_extra_atoms = []

        # Compute halfkernel for the given dx
        if dx != self.dx:
            interp_f_halfkernel = self._compute_interp_f_halfkernel(dx, n_ice_molecules)
        else:
            interp_f_halfkernel = self.interp_f_halfkernel

        for i in range(niter):
            ice_vol_f = rfftn(self.current_icedeltas)

            # amplitude multiplication
            ice_vol_f *= interp_f_halfkernel.unsqueeze(0)

            new_ice = torch.abs(self.irfftn(ice_vol_f))
            peaks = torch_peak_local_max(
                new_ice,
                num_peaks=int(n_ice_molecules * reduce_fraction),
                min_distance=int(min_distance / dx),
            )

            # ice_vol shape: (B, nz, n, n)
            self.register_buffer(
                "ice_vol",
                torch.zeros(batchsize, self.nz, self.n, self.n, device=self.device),
            )
            num_peaks = peaks.shape[1]  # must be fixed per batch

            # batch indices
            self.register_buffer(
                "batch_idx", torch.arange(batchsize).view(-1, 1).expand(-1, num_peaks)
            )

            # unpack coordinates
            z_idx = peaks[:, :, 0]
            y_idx = peaks[:, :, 1]
            x_idx = peaks[:, :, 2]

            # set ice voxels
            self.ice_vol[
                self.batch_idx.flatten(),
                z_idx.flatten(),
                y_idx.flatten(),
                x_idx.flatten(),
            ] = 1

            if add_extra_molecules:
                if peaks.shape[1] < n_ice_molecules:
                    n_extra = n_ice_molecules - peaks.shape[1]
                    self.n_extra_atoms.append(n_extra)

                    zero_idx = (self.ice_vol == 0).nonzero(as_tuple=False)
                    perm = torch.randperm(zero_idx.shape[0], device=self.device)
                    chosen = zero_idx[perm[:n_extra]]
                    self.ice_vol[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1

            mse = F.mse_loss(self.current_icedeltas.cpu(), self.ice_vol.cpu())
            self.frob_norm.append(mse)
            self.current_icedeltas = self.ice_vol
            if i > 1 and torch.isclose(self.frob_norm[-1], self.frob_norm[-2]):
                break
        self.n_peaks = peaks.shape[1]
        self.ice_coordinates = peaks.cpu()
        self.frob_norm = torch.tensor(self.frob_norm)
        self.n_extra_atoms = torch.tensor(self.n_extra_atoms)

    def create_ice_kernel(self, dx: float | None = None) -> torch.Tensor:
        """
        Create the atomic potential kernel for ice atoms.

        Uses the specified parameterization (Kirkland, Lobato, or Shryov) to generate
        the potential volume of a single water molecule (approximated as Oxygen).

        Returns
        -------
        pot : torch.Tensor
            Potential kernel volume, downsampled to simulation grid.
        """
        # create super-sampled (ss) coordinate system
        if dx is None:
            dx = self.dx
        ssn, ssdx, ssf = potential.compute_supersampling_parameters(dx)
        # set original convention to torch to avoid singularity at origin.
        sR = radial_grid_3d(ssn, ssdx, convention="torch")

        # for binning super-sampled grids to main volume grid.
        avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)

        if self.parameterization == "kirkland":
            pot = kirkland_atomic_potential_3d(8, sR)
        elif self.parameterization == "lobato":
            pot = lobato_atomic_potential_3d(8, sR)
        elif self.parameterization == "shtyrov":
            # from params_cat.json, 'O(HH)'
            params = torch.tensor(
                [
                    [0.3131, 0.8722],
                    [0.8102, 4.9669],
                    [0.9812, 14.1666],
                    [-0.5997, 64.1638],
                    [-0.1519, 121.3711],
                ]
            )
            # Separate columns: a_i, b_i
            a = (
                params[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )  # shape (3,1,1,1)
            b = params[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            k_xyz = real_to_kgrid_3d(sR)
            k2 = k_xyz**2
            k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

            s1_f = torch.sum(a * torch.exp(-b * k2 / 4), 0)
            dkx = k_xyz[1, 0, 0] - k_xyz[0, 0, 0]
            dky = k_xyz[0, 1, 0] - k_xyz[0, 0, 0]
            dkz = k_xyz[0, 0, 1] - k_xyz[0, 0, 0]
            pot = -torch.abs(fft3(s1_f, shift=True)) * dkx * dky * dkz  # need to negate
        else:
            raise ValueError(
                f"Unknown parameterization '{self.parameterization}'. "
                "Choose 'kirkland', 'lobato', or 'shtyrov'."
            )

        return avgpool3d(pot[None, None]).squeeze()

    def generate_ice(
        self, batchsize: int = 1, reduce_fraction: float = 1.0
    ) -> torch.Tensor:
        """
        Generate ice volumes by running the iterative algorithm and convolving.

        Parameters
        ----------
        batchsize : int, optional
            Number of ice volumes to generate. Default is 1.
        reduce_fraction : float, optional
            Fraction of target molecules for peak finding. Default is 1.0.

        Returns
        -------
        icecubes : torch.Tensor
            Generated ice potential volumes. Shape (batchsize, nz, n, n).
        """
        # run algorithms
        self.generate_ice_deltas(
            batchsize=batchsize,
            min_distance=self.min_distance,
            reduce_fraction=reduce_fraction,
        )

        # convolve with ice kernel
        self.register_buffer("icecubes", torch.zeros_like(self.current_icedeltas))

        # Batched convolution
        # ice_kernel shape: (nz, n, n) -> (1, nz, n, n) to match (B, nz, n, n)
        self.icecubes = fftconvolve(
            self.current_icedeltas,
            self.ice_kernel.unsqueeze(0),
            mode="same",
            axes=(-3, -2, -1),
        )
        return self.icecubes

    def irfftn(self, array: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse N-dimensional real Fourier transform with centering.

        Inverse of `rfftn`.

        Parameters
        ----------
        array : torch.Tensor
            Input complex-valued tensor (half-Hermitian).

        Returns
        -------
        real_array : torch.Tensor
            Real-valued output tensor.
        """
        return torch.fft.fftshift(
            torch.fft.irfftn(
                torch.fft.ifftshift(array, dim=(-3, -2)),
                s=(self.nz, self.n, self.n),
                dim=(-3, -2, -1),
            ),
            dim=(-3, -2, -1),
        )

    def generate_big_ice(
        self, shape: Sequence[int], num_unique: int = 8, bin_factor: int = 1
    ) -> torch.Tensor:
        """
        Generate a large ice volume by stitching smaller generated blocks.

        Handles boundary conditions and overlaps to ensure continuity.

        Parameters
        ----------
        shape : tuple of int
            Target shape (B, nz, ny, nx).
        bin_factor : int, optional
            Integer 3D binning factor applied to block deltas and kernel before
            assembly/convolution. Default is 1 (no binning).

        Returns
        -------
        big_ice : torch.Tensor
            Large ice volume.
        """
        if not isinstance(bin_factor, int) or bin_factor < 1:
            raise ValueError("bin_factor must be an integer >= 1")

        B, nz, ny, nx = shape
        if bin_factor > 1:
            target_shape = (
                B,
                int(torch.ceil(torch.as_tensor(nz) / bin_factor)),
                int(torch.ceil(torch.as_tensor(ny) / bin_factor)),
                int(torch.ceil(torch.as_tensor(nx) / bin_factor)),
            )
            print(
                f"bin_factor = {bin_factor}, outputing target shape of {target_shape}"
            )
        else:
            target_shape = shape

        print("Generating ice deltas.")
        self.generate_ice_deltas(batchsize=num_unique)
        icedeltas = self.current_icedeltas.cpu()

        # clean up faces of the ice blocks
        print("Replacing outer faces of ice cubes.")
        icedeltas = replace_outer_faces(icedeltas)

        if bin_factor > 1:
            pool_scale = bin_factor**3
            icedeltas = (
                F.avg_pool3d(
                    icedeltas.unsqueeze(1),
                    kernel_size=bin_factor,
                    stride=bin_factor,
                ).squeeze(1)
                * pool_scale
            )

        block_nz, block_ny, block_nx = icedeltas.shape[-3:]
        target_nz, target_ny, target_nx = target_shape[1:]
        num_z = int(torch.ceil(torch.as_tensor(target_nz) / block_nz))
        num_y = int(torch.ceil(torch.as_tensor(target_ny) / block_ny))
        num_x = int(torch.ceil(torch.as_tensor(target_nx) / block_nx))
        N_blocks = B * num_z * num_y * num_x
        num_blocks_per_B = num_z * num_y * num_x

        # assemble into big ice
        print("Assembling ice deltas into large volume.")
        big_ice = assemble_volume_randomized(icedeltas, target_shape)

        if bin_factor > 1:
            pool_scale = bin_factor**3
            conv_kernel = (
                F.avg_pool3d(
                    self.ice_kernel.unsqueeze(0).unsqueeze(0),
                    kernel_size=bin_factor,
                    stride=bin_factor,
                )
                .squeeze(0)
                .squeeze(0)
                * pool_scale
            )
        else:
            conv_kernel = self.ice_kernel

        # We can iterate over blocks, extract them, convolve, and put them back.
        # To vectorize, we can process 'chunk_size' blocks at a time.
        chunk_size = 32  # Adjust based on GPU memory

        for i in track(
            range(0, N_blocks, chunk_size),
            description="Ice convolution",
            transient=True,
            disable=not self.progressbars,
        ):
            # Identify which blocks belong to this chunk
            current_chunk_size = min(chunk_size, N_blocks - i)
            indices = torch.arange(i, i + current_chunk_size)

            ib = indices // num_blocks_per_B
            local_idx = indices % num_blocks_per_B
            iz = local_idx // (num_y * num_x)
            iy = (local_idx % (num_y * num_x)) // num_x
            ix = local_idx % num_x

            # Extract blocks
            # We have to loop to extract because they are not contiguous in memory
            # But this loop is over a small 'chunk_size' (e.g. 32), so it's fast.
            blocks = []
            for j in range(current_chunk_size):
                b, z, y, x = ib[j], iz[j], iy[j], ix[j]
                block = big_ice[
                    b,
                    z * block_nz : (z + 1) * block_nz,
                    y * block_ny : (y + 1) * block_ny,
                    x * block_nx : (x + 1) * block_nx,
                ]
                blocks.append(block)

            # Stack into a batch: (chunk_size, nz, n, n)
            batch = torch.stack(blocks).to(self.ice_kernel.device)

            # Convolve
            # ice_kernel shape: (nz, n, n) -> (1, nz, n, n)
            convolved_batch = fftconvolve(
                batch, conv_kernel.unsqueeze(0), mode="same", axes=(-3, -2, -1)
            )

            # Put back
            convolved_batch = (
                convolved_batch.cpu()
            )  # Move back to CPU if big_ice is on CPU
            for j in range(current_chunk_size):
                b, z, y, x = ib[j], iz[j], iy[j], ix[j]
                big_ice[
                    b,
                    z * block_nz : (z + 1) * block_nz,
                    y * block_ny : (y + 1) * block_ny,
                    x * block_nx : (x + 1) * block_nx,
                ] = convolved_batch[j]
        return big_ice[:B, :target_nz, :target_ny, :target_nx]

    def generate_big_ice_fast(
        self, shape: Sequence[int], num_unique: int = 8, bin_factor: int = 1
    ) -> torch.Tensor:
        """
        Generate a large ice volume from a bank of pre-convolved unique cubes.

        Workflow:
        1) generate `num_unique` ice delta cubes
        2) replace outer faces for boundary robustness
        3) convolve all unique cubes in one batch with `self.ice_kernel`
        4) bin down the convolved cubes (sum-binning if `bin_factor > 1`)
        5) assemble a large volume by randomized block tiling

        Parameters
        ----------
        shape : tuple of int
            Target shape (B, nz, ny, nx).
        num_unique : int, optional
            Number of unique cubes to generate. Default is 8.
        bin_factor : int, optional
            Integer 3D binning factor applied after convolution.
            Default is 1 (no binning).

        Returns
        -------
        big_ice : torch.Tensor
            Generated large ice volume.
        """
        if not isinstance(bin_factor, int) or bin_factor < 1:
            raise ValueError("bin_factor must be an integer >= 1")
        if not isinstance(num_unique, int) or num_unique < 1:
            raise ValueError("num_unique must be an integer >= 1")

        B, nz, ny, nx = shape
        if bin_factor > 1:
            target_shape = (
                B,
                int(torch.ceil(torch.as_tensor(nz) / bin_factor)),
                int(torch.ceil(torch.as_tensor(ny) / bin_factor)),
                int(torch.ceil(torch.as_tensor(nx) / bin_factor)),
            )
            print(
                f"bin_factor = {bin_factor}, outputing target shape of {target_shape}"
            )
            print(f"Recomputing ice kernal at {self.dx * bin_factor} A pixel size.")
            ice_kernel = (self.create_ice_kernel(self.dx * bin_factor)).to(
                self.ice_kernel.device
            )
        else:
            target_shape = shape
            ice_kernel = self.ice_kernel

        # 1) generate unique delta cubes
        self.generate_ice_deltas(batchsize=num_unique)
        icedeltas = self.current_icedeltas.cpu()

        # 2) clean cube faces
        icedeltas = replace_outer_faces(icedeltas)

        # 3) batched convolution of all unique cubes
        convolved = fftconvolve(
            icedeltas.to(ice_kernel.device),
            ice_kernel.unsqueeze(0),
            mode="same",
            axes=(-3, -2, -1),
        ).cpu()

        # 4) bin down convolved cubes
        if bin_factor > 1:
            pool_scale = bin_factor**3
            convolved = (
                F.avg_pool3d(
                    convolved.unsqueeze(1),
                    kernel_size=bin_factor,
                    stride=bin_factor,
                ).squeeze(1)
                * pool_scale
            )

        # 5) randomized assembly into requested binned shape
        big_ice = assemble_volume_randomized(convolved, target_shape)
        return big_ice[:B, : target_shape[1], : target_shape[2], : target_shape[3]]

    def generate_big_ice_interpolate(
        self, shape: Sequence[int], n_blocks: int = 8, algorithm_dx: float = 0.5
    ) -> torch.Tensor:
        """
        Generate a large ice volume by tiling interpolated blocks.

        Computes the ice algorithm at `algorithm_dx` resolution, then interpolates to self.dx
        (the user's target voxel size).

        Workflow:
        1) generate `n_blocks` ice delta cubes
        2) replace outer faces for boundary robustness
        3) convolve all unique cubes with kernel at algorithm_dx
        4) interpolate each convolved cube to self.dx
        5) assemble large volume using randomized block tiling

        Parameters
        ----------
        shape : tuple of int
            Target shape (nz, ny, nx) in pixels.
        n_blocks : int, optional
            Number of unique blocks to generate. Default is 8.
        algorithm_dx : float, optional
            Voxel size in Angstroms at which to run the ice generation algorithm.
            The algorithm is most stable at 0.5A. Default is 0.5.

        Returns
        -------
        big_ice : torch.Tensor
            Generated large ice volume of shape (1, nz, ny, nx) at self.dx voxel size.
        """
        if not isinstance(n_blocks, int) or n_blocks < 1:
            raise ValueError("n_blocks must be an integer >= 1")
        if algorithm_dx <= 0:
            raise ValueError("algorithm_dx must be positive")

        nz, ny, nx = shape
        target_shape = (1, nz, ny, nx)

        # Compute interpolated block size
        # Native block: 256 pixels at algorithm_dx covers 256*algorithm_dx A
        # Interpolated block covers same physical size at self.dx
        interpolated_block_size = int(
            torch.ceil(torch.as_tensor(256 * algorithm_dx / self.dx))
        )

        print(
            f"Generating {n_blocks} blocks at {algorithm_dx}A, interpolating to {self.dx}A"
        )
        print(f"Interpolated block size: {interpolated_block_size}^3 at {self.dx}A")

        # 1) generate unique delta cubes at algorithm_dx resolution
        self.generate_ice_deltas(batchsize=n_blocks, dx=algorithm_dx)
        icedeltas = self.current_icedeltas.cpu()

        # 2) clean cube faces
        icedeltas = replace_outer_faces(icedeltas)

        # 3) convolve at algorithm_dx resolution
        kernel = self.create_ice_kernel(dx=algorithm_dx).to(self.device)
        convolved = fftconvolve(
            icedeltas.to(kernel.device),
            kernel.unsqueeze(0),
            mode="same",
            axes=(-3, -2, -1),
        ).cpu()

        # 4) interpolate each block to self.dx
        interpolated_blocks = F.interpolate(
            convolved.unsqueeze(1),  # (n_blocks, 1, 256, 256, 256)
            size=(
                interpolated_block_size,
                interpolated_block_size,
                interpolated_block_size,
            ),
            mode="trilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # (n_blocks, interpolated_block_size, interpolated_block_size, interpolated_block_size)

        # 5) randomized assembly into requested shape
        big_ice = assemble_volume_randomized(interpolated_blocks, target_shape)
        return big_ice[:, : target_shape[1], : target_shape[2], : target_shape[3]]


def _wrap_coords(x: torch.Tensor, L: float) -> torch.Tensor:
    """Wrap coordinates into [-L/2, L/2)."""
    return (x + L / 2) % L - L / 2


class MCMCIcemaker:
    """
    Reverse Monte Carlo ice coordinate generator.

    Generates oxygen molecule positions by matching the O-O pair correlation
    function g(r) derived from an MD simulation frame via Metropolis MCMC.

    Two execution paths
    -------------------
    CPU  (device="cpu")
        Sequential Metropolis with a cell list; O(N) per sweep.
    GPU  (device="cuda")
        Fully vectorised parallel Metropolis; ~50-200× faster for N > 2000.
        g(r) is recomputed from scratch after each sweep to prevent drift.

    Parameters
    ----------
    n : int
        Number of voxels per axis. Box size is ``n * dx`` Å (cubic).
    dx : float
        Voxel size in Å.
    nz : int, optional
        Number of voxels along z. Defaults to ``n`` (cubic box).
    min_distance : float
        Hard-core O-O exclusion radius in Å.
    r_max : float, optional
        Maximum radius for g(r). Defaults to ``min(box/2 - 0.1, 14.0)`` Å.
    dr : float
        Histogram bin width for g(r) in Å.
    device : str or torch.device
        ``"cpu"`` for cell-list sequential path; ``"cuda"`` for GPU path.
    """

    def __init__(
        self,
        n: int = 200,
        dx: float = 0.5,
        nz: Optional[int] = None,
        min_distance: float = 2.0,
        r_max: Optional[float] = None,
        dr: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n = n
        self.dx = dx
        self.nz = nz if nz is not None else n
        self.box_size = n * dx  # cubic box (MCMC assumes isotropy)
        if r_max is None:
            r_max = min(self.box_size / 2 - 0.1, 14.0)
        assert (
            r_max <= self.box_size / 2
        ), "r_max must be <= box_size/2 for unambiguous PBC"

        self.min_distance = min_distance
        self.r_max = r_max
        self.dr = dr
        self.n_bins = int(r_max / dr)
        self.device = torch.device(device)

        self.n_molecules = int(ndensity_of_amorphous_ice * self.box_size**3)
        self.r_centers = (torch.arange(self.n_bins) + 0.5) * dr

        self.positions: Optional[torch.Tensor] = None
        self.gr_hist: Optional[torch.Tensor] = None
        self.gr_target: Optional[torch.Tensor] = None

        self._cell_size: float = r_max / 3.0
        self._nc: int = max(3, int(np.ceil(self.box_size / self._cell_size)))
        self._cell_list: dict[tuple[int, int, int], list[int]] = {}
        self._mol_cell: list[tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gr_norm(self, N: int) -> torch.Tensor:
        rho = N / self.box_size**3
        return N * rho * 4 * torch.pi * self.r_centers**2 * self.dr

    def _hist_from_dists(self, dists: torch.Tensor) -> torch.Tensor:
        mask = dists < self.r_max
        valid = dists[mask]
        if len(valid) == 0:
            return torch.zeros(self.n_bins)
        bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
        return torch.bincount(bins, minlength=self.n_bins).float()

    # ------------------------------------------------------------------
    # Cell list (CPU path)
    # ------------------------------------------------------------------

    def _pos_to_cell(self, pos: np.ndarray) -> tuple[int, int, int]:
        nc, cs = self._nc, self._cell_size
        return (int(pos[0] / cs) % nc, int(pos[1] / cs) % nc, int(pos[2] / cs) % nc)

    def _build_cell_list(self) -> None:
        self._cell_list = {}
        self._mol_cell = []
        pos_np = self.positions.cpu().numpy()
        for i, p in enumerate(pos_np):
            c = self._pos_to_cell(p)
            self._cell_list.setdefault(c, []).append(i)
            self._mol_cell.append(c)

    def _cell_neighbors(self, pos_np: np.ndarray, exclude: int) -> torch.Tensor:
        c = self._pos_to_cell(pos_np)
        nc = self._nc
        nbrs: list[int] = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                for ddz in (-1, 0, 1):
                    key = ((c[0] + ddx) % nc, (c[1] + ddy) % nc, (c[2] + ddz) % nc)
                    lst = self._cell_list.get(key)
                    if lst:
                        nbrs.extend(lst)
        return torch.tensor([i for i in nbrs if i != exclude], dtype=torch.long)

    def _cell_move(self, j: int, r_new_np: np.ndarray) -> None:
        old_c = self._mol_cell[j]
        new_c = self._pos_to_cell(r_new_np)
        if old_c != new_c:
            self._cell_list[old_c].remove(j)
            self._cell_list.setdefault(new_c, []).append(j)
            self._mol_cell[j] = new_c

    # ------------------------------------------------------------------
    # g(r) computation
    # ------------------------------------------------------------------

    def compute_gr(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute g(r) via full O(N²) all-pairs matrix (CPU).

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions, shape (N, 3).

        Returns
        -------
        gr : torch.Tensor
            Normalised g(r), shape (n_bins,).
        hist : torch.Tensor
            Raw pair count histogram, shape (n_bins,).
        """
        N = len(positions)
        d = positions.unsqueeze(0) - positions.unsqueeze(1)
        d -= self.box_size * torch.round(d / self.box_size)
        dists = torch.norm(d, dim=2)
        mask = (dists > 1e-6) & (dists < self.r_max)
        valid = dists[mask]
        bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
        hist = torch.bincount(bins, minlength=self.n_bins).float()
        return hist / self._gr_norm(N), hist

    def compute_gr_chunked(
        self, positions: torch.Tensor, chunk_size: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute g(r) in row-chunks.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions, shape (N, 3).
        chunk_size : int
            Number of rows per chunk.

        Returns
        -------
        gr : torch.Tensor
            Normalised g(r).
        hist : torch.Tensor
            Raw count histogram.
        """
        N = len(positions)
        dev = positions.device
        hist = torch.zeros(self.n_bins, device=dev)
        for b in range(0, N, chunk_size):
            end = min(b + chunk_size, N)
            B = end - b
            d = positions[b:end].unsqueeze(1) - positions.unsqueeze(0)
            d -= self.box_size * torch.round(d / self.box_size)
            dists = torch.norm(d, dim=2)
            dists[torch.arange(B, device=dev), torch.arange(b, end, device=dev)] = (
                float("inf")
            )
            mask = (dists > 1e-6) & (dists < self.r_max)
            valid = dists[mask]
            if len(valid):
                bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
                hist += torch.bincount(bins, minlength=self.n_bins).float()
        return hist / self._gr_norm(N).to(dev), hist

    # ------------------------------------------------------------------
    # Target g(r) from MD dump
    # ------------------------------------------------------------------

    def set_target_gr_from_md(
        self,
        filepath: str,
        frame_idx: int = 0,
        trim_size: float = 100.0,
        oxygen_type: int = 1,
        chunk_size: int = 1000,
    ) -> None:
        """
        Load one MD frame, extract oxygen positions, compute target g(r).

        Parameters
        ----------
        filepath : str
            Path to LAMMPS dump file.
        frame_idx : int
            Which timestep frame to use.
        trim_size : float
            Side length (Å) of cube trimmed around the MD box centre.
        oxygen_type : int
            Atom type index for oxygen.
        chunk_size : int
            Row block size for batched distance computation.
        """
        print(f"Parsing MD dump frame {frame_idx} for oxygen atoms …")
        with open(filepath) as f:
            lines = f.readlines()

        frame_starts = [i for i, ln in enumerate(lines) if ln == "ITEM: TIMESTEP\n"]
        fs = frame_starts[frame_idx]
        n_atoms = int(lines[fs + 3])
        coord_start = fs + 9

        oxygens = []
        for ln in lines[coord_start : coord_start + n_atoms]:
            parts = ln.split()
            if int(parts[1]) == oxygen_type:
                oxygens.append([float(parts[2]), float(parts[3]), float(parts[4])])

        coords = torch.tensor(oxygens, dtype=torch.float32)
        print(f"  {len(coords)} oxygen atoms found")
        coords -= coords.mean(0)
        half = trim_size / 2
        mask = (
            (coords[:, 0].abs() < half)
            & (coords[:, 1].abs() < half)
            & (coords[:, 2].abs() < half)
        )
        coords = coords[mask]
        N_O = len(coords)
        print(f"  {N_O} oxygens after trimming to {trim_size} Å cube")

        hist = torch.zeros(self.n_bins)
        for i in range(0, N_O, chunk_size):
            ci = coords[i : i + chunk_size]
            d = ci.unsqueeze(1) - coords.unsqueeze(0)
            dists = torch.norm(d, dim=2)
            for k in range(len(ci)):
                dists[k, i + k] = float("inf")
            mask2 = dists < self.r_max
            valid = dists[mask2]
            if len(valid):
                bins = torch.clamp((valid / self.dr).long(), 0, self.n_bins - 1)
                hist += torch.bincount(bins, minlength=self.n_bins).float()

        rho_O = N_O / trim_size**3
        gr_norm_md = N_O * rho_O * 4 * torch.pi * self.r_centers**2 * self.dr
        self.gr_target = hist / gr_norm_md
        print("  Target g(r) computed.")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_random(self, seed: int = 42) -> None:
        """
        Initialise via Random Sequential Addition with hard-core exclusion.

        Uses a cell list for O(N) placement with ``min_distance`` exclusion.

        Parameters
        ----------
        seed : int
            Random seed.
        """
        rng = np.random.default_rng(seed)
        N = self.n_molecules
        L = self.box_size
        md = self.min_distance

        cs = md
        nc = max(3, int(np.ceil(L / cs)))
        cs = L / nc

        placed = np.zeros((N, 3), dtype=np.float32)
        placed_count = 0
        cell_list: dict[tuple[int, int, int], list[int]] = {}

        def _cell(p: np.ndarray) -> tuple[int, int, int]:
            return (int(p[0] / cs) % nc, int(p[1] / cs) % nc, int(p[2] / cs) % nc)

        max_attempts = N * 200
        attempts = 0
        while placed_count < N and attempts < max_attempts:
            r = rng.random(3).astype(np.float32) * L
            c = _cell(r)
            ok = True
            for ddx in (-1, 0, 1):
                if not ok:
                    break
                for ddy in (-1, 0, 1):
                    if not ok:
                        break
                    for ddz in (-1, 0, 1):
                        key = ((c[0] + ddx) % nc, (c[1] + ddy) % nc, (c[2] + ddz) % nc)
                        nbrs = cell_list.get(key)
                        if not nbrs:
                            continue
                        for i in nbrs:
                            d = r - placed[i]
                            d -= L * np.round(d / L)
                            if float(np.dot(d, d)) < md * md:
                                ok = False
                                break
            if ok:
                placed[placed_count] = r
                cell_list.setdefault(c, []).append(placed_count)
                placed_count += 1
            attempts += 1

        if placed_count < N:
            print(
                f"  RSA warning: placed {placed_count}/{N} molecules "
                f"(saturation after {attempts} attempts)"
            )

        self.positions = torch.from_numpy(placed[:placed_count])
        self.n_molecules = placed_count
        pos_dev = self.positions.to(self.device)
        _, gr_dev = self.compute_gr_chunked(pos_dev, chunk_size=512)
        self.gr_hist = gr_dev.cpu()
        print(f"  Initialised with {self.n_molecules} molecules (RSA)")

    def init_from_icemaker(self, im: "Icemaker", batch_idx: int = 0) -> None:
        """
        Initialise from peak voxel coordinates of a completed
        :meth:`Icemaker.generate_ice_deltas` run.

        Parameters
        ----------
        im : Icemaker
            A completed Icemaker instance.
        batch_idx : int
            Which batch item to use.
        """
        vox = im.ice_coordinates[batch_idx].float()
        n_half = im.n // 2
        xyz = torch.zeros_like(vox)
        xyz[:, 0] = (vox[:, 2] - n_half) * im.dx + self.box_size / 2
        xyz[:, 1] = (vox[:, 1] - n_half) * im.dx + self.box_size / 2
        xyz[:, 2] = (vox[:, 0] - im.nz // 2) * im.dx + self.box_size / 2
        xyz = xyz % self.box_size
        self.positions = xyz
        self.n_molecules = len(self.positions)
        self._remove_pbc_violations()
        _, self.gr_hist = self.compute_gr(self.positions)
        print(f"  Initialised with {self.n_molecules} molecules (from Icemaker)")

    def _remove_pbc_violations(self) -> int:
        d = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
        d -= self.box_size * torch.round(d / self.box_size)
        dists = torch.norm(d, dim=2)
        dists.fill_diagonal_(float("inf"))
        violated = (dists < self.min_distance).any(dim=1)
        n_removed = int(violated.sum())
        if n_removed > 0:
            self.positions = self.positions[~violated]
            self.n_molecules = len(self.positions)
            print(
                f"  Removed {n_removed} PBC-violating molecules ({self.n_molecules} remain)"
            )
        return n_removed

    # ------------------------------------------------------------------
    # GPU parallel sweep
    # ------------------------------------------------------------------

    def _parallel_gpu_sweep(
        self, T: float, step_size: float, chunk_size: int = 256, n_subgroups: int = 1
    ) -> tuple[int, int, float]:
        N = self.n_molecules
        dev = self.device
        L = self.box_size
        dr = self.dr
        n_bins = self.n_bins

        pos = self.positions.to(dev)
        norm = self._gr_norm(N).to(dev)
        gr_target = self.gr_target.to(dev)
        gr_hist = self.gr_hist.to(dev)

        n_accepted_total = 0
        n_hard_ok_total = 0

        perm = torch.randperm(N, device=dev)
        group_size = (N + n_subgroups - 1) // n_subgroups

        for g in range(n_subgroups):
            idx = perm[g * group_size : (g + 1) * group_size]
            Bg = len(idx)
            if Bg == 0:
                continue

            pos_g = pos[idx]
            delta = step_size * (2 * torch.rand(Bg, 3, device=dev) - 1)
            r_new_g = (pos_g + delta) % L

            h_new_g = torch.zeros(Bg, n_bins, device=dev)
            h_old_g = torch.zeros(Bg, n_bins, device=dev)
            hard_ok_g = torch.ones(Bg, dtype=torch.bool, device=dev)

            for b in range(0, Bg, chunk_size):
                end = min(b + chunk_size, Bg)
                B = end - b
                row = torch.arange(B, device=dev)
                glob = idx[b:end]

                d = r_new_g[b:end].unsqueeze(1) - pos.unsqueeze(0)
                d -= L * torch.round(d / L)
                dn = torch.norm(d, dim=2)
                dn[row, glob] = float("inf")
                hard_ok_g[b:end] = dn.min(dim=1).values >= self.min_distance

                mask_n = dn < self.r_max
                bins_n = dn.div(dr).long().clamp(0, n_bins - 1)
                bins_n[~mask_n] = n_bins
                h_ext = torch.zeros(B, n_bins + 1, device=dev)
                h_ext.scatter_add_(1, bins_n, mask_n.float())
                h_new_g[b:end] = h_ext[:, :n_bins]

                d_o = pos_g[b:end].unsqueeze(1) - pos.unsqueeze(0)
                d_o -= L * torch.round(d_o / L)
                do = torch.norm(d_o, dim=2)
                do[row, glob] = float("inf")
                mask_o = do < self.r_max
                bins_o = do.div(dr).long().clamp(0, n_bins - 1)
                bins_o[~mask_o] = n_bins
                h_ext_o = torch.zeros(B, n_bins + 1, device=dev)
                h_ext_o.scatter_add_(1, bins_o, mask_o.float())
                h_old_g[b:end] = h_ext_o[:, :n_bins]

            delta_hist_g = 2.0 * (h_new_g - h_old_g)
            E_cur = torch.mean((gr_hist / norm - gr_target) ** 2)
            gr_prop = (gr_hist.unsqueeze(0) + delta_hist_g) / norm
            E_prop = torch.mean((gr_prop - gr_target) ** 2, dim=1)
            delta_E = E_prop - E_cur

            u = torch.rand(Bg, device=dev)
            log_acc = (-delta_E / T).clamp(min=-60.0)
            accept = hard_ok_g & ((delta_E <= 0) | (u < torch.exp(log_acc)))

            n_accepted_total += int(accept.sum())
            n_hard_ok_total += int(hard_ok_g.sum())

            acc_idx = idx[accept]
            pos[acc_idx] = r_new_g[accept]
            if accept.any():
                gr_hist = gr_hist + delta_hist_g[accept].sum(0)

        self.positions = pos
        _, self.gr_hist = self.compute_gr_chunked(pos, chunk_size)
        self.gr_hist = self.gr_hist.to(dev)
        E_new = torch.mean((self.gr_hist / norm - gr_target) ** 2).item()
        return n_accepted_total, n_hard_ok_total, E_new

    # ------------------------------------------------------------------
    # MCMC main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_sweeps: int = 200,
        step_size: float = 0.5,
        temperature: float = 1e-3,
        record_every: int = 10,
        anneal: bool = True,
        T_start: float = 1.0,
        T_end: float = 1e-4,
        chunk_size: int = 256,
        n_subgroups: int = 8,
    ) -> dict:
        """
        Run Metropolis MCMC sweeps targeting ``self.gr_target``.

        Parameters
        ----------
        n_sweeps : int
            Number of sweeps (each sweep ≈ N move attempts).
        step_size : float
            Uniform displacement half-width in Å.
        temperature : float
            Metropolis temperature when ``anneal=False``.
        record_every : int
            Diagnostic recording interval in sweeps.
        anneal : bool
            Log-linearly anneal temperature from ``T_start`` to ``T_end``.
        T_start, T_end : float
            Annealing schedule endpoints.
        chunk_size : int
            Row-block size for chunked distance computation (GPU path).
        n_subgroups : int
            Sequential sub-groups per GPU sweep. Higher values reduce the
            parallel-approximation error at the cost of speed.

        Returns
        -------
        history : dict
            Keys: ``'sweep'``, ``'energy'``, ``'acceptance'``,
            ``'acceptance_mc'``, ``'gr'``.
        """
        assert self.positions is not None, "Call init_random() first"
        assert self.gr_target is not None, "Call set_target_gr_from_md() first"

        N = self.n_molecules
        use_gpu = self.device.type != "cpu"
        self.positions = self.positions.to(self.device)
        self.gr_hist = self.gr_hist.to(self.device)

        norm = self._gr_norm(N).to(self.device)
        gr_target_dev = self.gr_target.to(self.device)
        E_current = torch.mean((self.gr_hist / norm - gr_target_dev) ** 2).item()

        if anneal:
            temps = np.exp(
                np.linspace(np.log(T_start), np.log(T_end), n_sweeps)
            ).tolist()
        else:
            temps = [temperature] * n_sweeps

        history: dict = {
            "sweep": [],
            "energy": [],
            "acceptance": [],
            "acceptance_mc": [],
            "gr": [],
        }

        if use_gpu:
            print(
                f"  GPU: chunk_size={chunk_size}, n_subgroups={n_subgroups}, device={self.device}"
            )
        else:
            self._build_cell_list()
            pos_np = self.positions.cpu().numpy()

        for sweep in range(n_sweeps):
            T = temps[sweep]

            if use_gpu:
                n_accepted, n_hard_ok, E_current = self._parallel_gpu_sweep(
                    T, step_size, chunk_size, n_subgroups
                )
                mc_rate = n_accepted / n_hard_ok if n_hard_ok > 0 else 0.0
                accept_rate = n_accepted / N
            else:
                n_accepted = 0
                n_mc_trials = 0
                n_mc_accepted = 0
                j_all = torch.randint(0, N, (N,)).tolist()
                delta_all = (step_size * (2 * torch.rand(N, 3) - 1)).numpy()
                u_all = np.random.rand(N)

                for i in range(N):
                    j = j_all[i]
                    r_old_np = pos_np[j].copy()
                    r_new_np = (r_old_np + delta_all[i]) % self.box_size

                    nbr_idx = self._cell_neighbors(r_new_np, j)
                    if len(nbr_idx) == 0:
                        n_mc_trials += 1
                        continue

                    nbr_pos = self.positions[nbr_idx]
                    d_new_vec = nbr_pos - torch.from_numpy(r_new_np)
                    d_new_vec -= self.box_size * torch.round(d_new_vec / self.box_size)
                    d_new = torch.norm(d_new_vec, dim=1)
                    if d_new.min().item() < self.min_distance:
                        continue

                    n_mc_trials += 1
                    nbr_old_idx = self._cell_neighbors(r_old_np, j)
                    nbr_old_pos = self.positions[nbr_old_idx]
                    d_old_vec = nbr_old_pos - torch.from_numpy(r_old_np)
                    d_old_vec -= self.box_size * torch.round(d_old_vec / self.box_size)
                    d_old = torch.norm(d_old_vec, dim=1)

                    h_add = self._hist_from_dists(d_new)
                    h_rem = self._hist_from_dists(d_old)
                    delta_hist = 2.0 * (h_add - h_rem)
                    gr_proposed = (self.gr_hist + delta_hist) / norm.cpu()
                    E_proposed = float(torch.mean((gr_proposed - self.gr_target) ** 2))
                    delta_E = E_proposed - E_current

                    if delta_E < 0 or (T > 0 and u_all[i] < np.exp(-delta_E / T)):
                        self.positions[j] = torch.from_numpy(r_new_np)
                        pos_np[j] = r_new_np
                        self.gr_hist = self.gr_hist + delta_hist
                        E_current = E_proposed
                        n_accepted += 1
                        n_mc_accepted += 1
                        self._cell_move(j, r_new_np)

                mc_rate = n_mc_accepted / n_mc_trials if n_mc_trials > 0 else 0.0
                accept_rate = n_accepted / N

            if sweep % record_every == 0:
                gr_now = (self.gr_hist / norm).cpu().clone()
                history["sweep"].append(sweep)
                history["energy"].append(E_current)
                history["acceptance"].append(accept_rate)
                history["acceptance_mc"].append(mc_rate)
                history["gr"].append(gr_now)
                print(
                    f"  sweep {sweep:4d}/{n_sweeps}  E={E_current:.5f}  "
                    f"accept={accept_rate:.2f}  mc_accept={mc_rate:.2f}  T={T:.2e}"
                )

        self.positions = self.positions.cpu()
        self.gr_hist = self.gr_hist.cpu()
        return history

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def voxelize(self) -> torch.Tensor:
        """
        Voxelize positions onto the ``(nz, n, n)`` grid.

        Returns
        -------
        grid : torch.Tensor
            Soft-voxelized density, shape (nz, n, n).
        """
        assert self.positions is not None, "No positions — call init_random() first"
        centred = self.positions.cpu() - self.box_size / 2
        return soft_voxelize_coordinates(
            centred, grid_shape=(self.nz, self.n, self.n), voxel_size=self.dx
        )

    def generate_ice(
        self,
        batchsize: int = 1,
        n_sweeps: int = 200,
        step_size: float = 0.5,
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Run ``batchsize`` independent MCMC chains and return voxelized densities.

        Requires :meth:`set_target_gr_from_md` to have been called first.

        Parameters
        ----------
        batchsize : int
            Number of independent ice volumes to generate.
        n_sweeps : int
            MCMC sweeps per volume.
        step_size : float
            Displacement half-width in Å.
        seed : int
            Base random seed; each chain uses ``seed + i``.

        Returns
        -------
        ice : torch.Tensor
            Voxelized oxygen densities, shape (batchsize, nz, n, n).
        """
        assert (
            self.gr_target is not None
        ), "Call set_target_gr_from_md() before generate_ice()"
        results = []
        for i in range(batchsize):
            self.init_random(seed=seed + i)
            self.run(n_sweeps=n_sweeps, step_size=step_size)
            results.append(self.voxelize())
        return torch.stack(results)


class GradientSKIcemaker:
    """
    Differentiable S(k)-matching ice coordinate generator.

    Generates oxygen molecule positions by matching the target 3D Fourier
    amplitude |F(k)| from MD simulations via L-BFGS gradient descent on
    continuously-valued atomic positions.

    Forward pass (fully differentiable):
        positions → soft_voxelize_coordinates → FFT → radial |F(k)| → MSE → backward()

    Cost per step: O(N log N) vs O(N²) per MCMC sweep.

    Parameters
    ----------
    n : int
        Number of voxels per axis (x, y).
    dx : float
        Voxel size in Å.
    nz : int, optional
        Number of voxels along z. Defaults to ``n``.
    min_distance : float
        Hard-core exclusion radius in Å used only during RSA initialisation.
    device : str or torch.device
        Computation device.
    """

    def __init__(
        self,
        n: int = 200,
        dx: float = 0.5,
        nz: Optional[int] = None,
        min_distance: float = 2.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.min_distance = min_distance

        _im = Icemaker(n=n, dx=dx, nz=nz)
        self.n = _im.n
        self.nz = _im.nz
        self.dx = _im.dx
        self.box_x = self.n * self.dx
        self.box_y = self.n * self.dx
        self.box_z = self.nz * self.dx
        self._ice_kernel: torch.Tensor = _im.ice_kernel.cpu()

        self.n_molecules = int(
            ndensity_of_amorphous_ice * self.box_x * self.box_y * self.box_z
        )

        K_flat = _im.K.cpu().ravel()
        interp_vals = interp1d(
            _im.mdsim_radial_k[1:].cpu(),
            _im.mdsim_f_radial_avg[1:].cpu(),
            K_flat,
        )
        f_kernel = interp_vals.reshape(self.nz, self.n, self.n).float()
        f_kernel = f_kernel * (self.n_molecules**0.5)
        f_kernel[self.nz // 2, self.n // 2, self.n // 2] = float(self.n_molecules)
        self.f_target: torch.Tensor = f_kernel.to(self.device)

        self.dk: float = _im.dk
        self.f_target_radial: torch.Tensor = radial_profile_3d(f_kernel.cpu())

        r_bins = (_im.K.cpu() / self.dk).round().long().flatten()
        n_rbins = int(r_bins.max().item()) + 1
        bin_count = torch.bincount(r_bins, minlength=n_rbins).float().clamp(min=1)
        self._r_bins: torch.Tensor = r_bins.to(self.device)
        self._bin_count: torch.Tensor = bin_count.to(self.device)
        self._n_rbins: int = n_rbins
        self._f_target_rad_1d: torch.Tensor = self.f_target_radial[:n_rbins].to(
            self.device
        )

        self.positions: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_random(self, seed: int = 42) -> None:
        """
        Initialise from uniform random positions (no exclusion).

        Parameters
        ----------
        seed : int
            Random seed.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        pos = torch.rand(self.n_molecules, 3, generator=gen)
        pos[:, 0] = (pos[:, 0] - 0.5) * self.box_x
        pos[:, 1] = (pos[:, 1] - 0.5) * self.box_y
        pos[:, 2] = (pos[:, 2] - 0.5) * self.box_z
        self.positions = pos

    def init_rsa(self, seed: int = 42) -> None:
        """
        Initialise via RSA with hard-core exclusion.

        Delegates to :class:`MCMCIcemaker` for O(N) cell-list RSA.

        Parameters
        ----------
        seed : int
            Random seed.
        """
        rmc = MCMCIcemaker(
            n=self.n,
            dx=self.dx,
            nz=self.nz,
            min_distance=self.min_distance,
            device="cpu",
        )
        rmc.n_molecules = self.n_molecules
        rmc.init_random(seed=seed)
        raw = rmc.positions.float()
        self.positions = raw - torch.tensor(
            [self.box_x / 2, self.box_y / 2, self.box_z / 2]
        )
        self.n_molecules = rmc.n_molecules

    # ------------------------------------------------------------------
    # Differentiable S(k) loss
    # ------------------------------------------------------------------

    def _sk_loss(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Radial-profile MSE between |FFT(voxelized positions)| and the target.

        Differentiable w.r.t. ``pos`` through soft voxelization, FFT, and
        scatter-based radial average.

        Parameters
        ----------
        pos : torch.Tensor
            Positions (N, 3) in Å, centered at origin.

        Returns
        -------
        loss : torch.Tensor
            Scalar radial-profile MSE.
        f_amp : torch.Tensor
            |F(k)| in shifted frequency space, shape (nz, n, n).
        """
        vox = soft_voxelize_coordinates(
            pos,
            grid_shape=(self.nz, self.n, self.n),
            voxel_size=self.dx,
            device=self.device,
        )
        f_amp = torch.abs(fft3(vox, shift=True))
        bin_sum = torch.zeros(self._n_rbins, device=self.device)
        bin_sum.scatter_add_(0, self._r_bins, f_amp.flatten())
        sim_radial = bin_sum / self._bin_count
        loss = torch.mean((sim_radial - self._f_target_rad_1d) ** 2)
        return loss, f_amp

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_steps: int = 50,
        lr: float = 1.0,
        optimizer: str = "lbfgs",
        record_every: int = 5,
    ) -> dict:
        """
        Pure gradient descent on the S(k) loss — no Langevin noise.

        L-BFGS (default) converges in ~50 steps vs ~500 for Adam on this smooth
        radial-profile loss; the strong-Wolfe line search adapts step sizes
        automatically so the ``lr`` parameter rarely needs tuning.

        Parameters
        ----------
        n_steps : int
            Optimizer outer iterations. For L-BFGS each outer step may call
            the closure up to 10 times for line search.
        lr : float
            Step size. Keep at 1.0 for L-BFGS; use ~0.01 for Adam.
        optimizer : str
            ``'lbfgs'`` (default) or ``'adam'``.
        record_every : int
            Diagnostic recording interval.

        Returns
        -------
        history : dict
            Keys: ``'step'``, ``'loss'``, ``'radial_profile'``.
        """
        assert self.positions is not None, "Call init_random() or init_rsa() first"

        pos = self.positions.to(self.device).clone().requires_grad_(True)
        history: dict[str, list] = {"step": [], "loss": [], "radial_profile": []}
        t0 = time.time()

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [pos],
                lr=lr,
                max_iter=10,
                history_size=20,
                line_search_fn="strong_wolfe",
            )
            last_f_amp: list[torch.Tensor] = [torch.empty(0)]

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss, f_amp = self._sk_loss(pos)
                loss.backward()
                last_f_amp[0] = f_amp.detach()
                return loss

            for step in range(n_steps):
                loss = opt.step(closure)
                with torch.no_grad():
                    pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
                    pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
                    pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
                if step % record_every == 0:
                    loss_val = (
                        loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                    )
                    with torch.no_grad():
                        rad = radial_profile_3d(last_f_amp[0].cpu())
                    history["step"].append(step)
                    history["loss"].append(loss_val)
                    history["radial_profile"].append(rad)
                    print(f"  step {step:3d}/{n_steps}  loss={loss_val:.6f}")

        else:  # adam
            opt_adam = torch.optim.Adam([pos], lr=lr)
            for step in range(n_steps):
                opt_adam.zero_grad()
                loss, f_amp = self._sk_loss(pos)
                loss.backward()
                opt_adam.step()
                with torch.no_grad():
                    pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
                    pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
                    pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
                if step % record_every == 0:
                    with torch.no_grad():
                        rad = radial_profile_3d(f_amp.detach().cpu())
                    history["step"].append(step)
                    history["loss"].append(loss.item())
                    history["radial_profile"].append(rad)
                    print(f"  step {step:4d}/{n_steps}  loss={loss.item():.6f}")

        print(f"  Finished in {time.time()-t0:.1f}s")
        self.positions = pos.detach().cpu()
        return history

    def sample(
        self,
        n_steps: int = 200,
        lr: float = 0.05,
        noise: float = 0.02,
        record_every: int = 50,
    ) -> dict:
        """
        Langevin sampling around a converged structure.

        Combines a gradient step (keeps structure near the S(k) target) with
        Gaussian noise injection (provides positional diversity). Call
        :meth:`optimize` first.

        Parameters
        ----------
        n_steps : int
            Number of Langevin steps.
        lr : float
            Adam learning rate.
        noise : float
            Gaussian noise std per step in Å.
        record_every : int
            Diagnostic recording interval.

        Returns
        -------
        history : dict
            Keys: ``'step'``, ``'loss'``, ``'radial_profile'``.
        """
        assert self.positions is not None, "Call optimize() or init_* first"

        pos = self.positions.to(self.device).clone().requires_grad_(True)
        opt = torch.optim.Adam([pos], lr=lr)
        history: dict[str, list] = {"step": [], "loss": [], "radial_profile": []}
        t0 = time.time()

        for step in range(n_steps):
            opt.zero_grad()
            loss, f_amp = self._sk_loss(pos)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pos.data.add_(noise * torch.randn_like(pos))
                pos.data[:, 0] = _wrap_coords(pos.data[:, 0], self.box_x)
                pos.data[:, 1] = _wrap_coords(pos.data[:, 1], self.box_y)
                pos.data[:, 2] = _wrap_coords(pos.data[:, 2], self.box_z)
            if step % record_every == 0:
                with torch.no_grad():
                    rad = radial_profile_3d(f_amp.detach().cpu())
                history["step"].append(step)
                history["loss"].append(loss.item())
                history["radial_profile"].append(rad)
                print(
                    f"  step {step:4d}/{n_steps}  loss={loss.item():.6f}  noise={noise:.4f}"
                )

        print(f"  Finished in {time.time()-t0:.1f}s")
        self.positions = pos.detach().cpu()
        return history

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def voxelize(self) -> torch.Tensor:
        """
        Voxelize the current positions onto the ``(nz, n, n)`` grid.

        Returns
        -------
        grid : torch.Tensor
            Soft-voxelized density, shape (nz, n, n).
        """
        assert self.positions is not None, "No positions — call init_* first"
        with torch.no_grad():
            return soft_voxelize_coordinates(
                self.positions.cpu(),
                grid_shape=(self.nz, self.n, self.n),
                voxel_size=self.dx,
            )

    def generate_ice_deltas(
        self,
        batchsize: int = 1,
        n_steps: int = 50,
        lr: float = 1.0,
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Run ``batchsize`` independent L-BFGS optimisations and return soft-voxelized
        ice position volumes, without convolving with scattering potentials.

        Each volume uses a fresh RSA initialisation with seed ``seed + i``,
        then L-BFGS optimisation. Result is stored as ``self.current_icedeltas``.

        Parameters
        ----------
        batchsize : int
            Number of independent ice volumes.
        n_steps : int
            L-BFGS outer iterations per volume.
        lr : float
            L-BFGS initial step size (line search adapts it automatically).
        seed : int
            Base random seed; volume ``i`` uses ``seed + i``.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape (batchsize, nz, n, n).
        """
        results = []
        for i in range(batchsize):
            # self.init_rsa(seed=seed + i)
            self.init_random(seed=seed + i)
            self.optimize(
                n_steps=n_steps, lr=lr, optimizer="lbfgs", record_every=n_steps
            )
            results.append(self.voxelize())
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_ice(
        self,
        batchsize: int = 1,
        n_steps: int = 50,
        lr: float = 1.0,
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Run ``batchsize`` independent L-BFGS optimisations and return convolved
        ice potential volumes.

        Each volume uses a fresh RSA initialisation with seed ``seed + i``,
        then L-BFGS, then convolution with the Kirkland O potential kernel.

        Parameters
        ----------
        batchsize : int
            Number of independent ice volumes.
        n_steps : int
            L-BFGS outer iterations per volume.
        lr : float
            L-BFGS initial step size (line search adapts it automatically).
        seed : int
            Base random seed; volume ``i`` uses ``seed + i``.

        Returns
        -------
        ice : torch.Tensor
            Convolved ice potential volumes, shape (batchsize, nz, n, n).
        """
        self.generate_ice_deltas(batchsize=batchsize, n_steps=n_steps, lr=lr, seed=seed)
        return fftconvolve(
            self.current_icedeltas,
            self._ice_kernel.unsqueeze(0),
            mode="same",
            axes=(-3, -2, -1),
        )

    @staticmethod
    def assemble_tiles(
        tile_positions: list[torch.Tensor],
        tile_box: tuple[float, float, float],
        nx: int,
        ny: int,
        nz: int,
    ) -> torch.Tensor:
        """
        Assemble ``nx × ny × nz`` independently-optimised tiles into one volume.

        Each tile is a unique set of atom positions (optimised with a different
        seed), so there is no periodicity artifact.

        Parameters
        ----------
        tile_positions : list of torch.Tensor
            Exactly ``nx * ny * nz`` position tensors, each shape ``(N, 3)``,
            centered at the origin in tile-box coordinates. Order is x-fastest
            (C order).
        tile_box : (Lx, Ly, Lz)
            Physical size of one tile in Å.
        nx, ny, nz : int
            Number of tiles along each axis.

        Returns
        -------
        positions : torch.Tensor
            All assembled positions, shape ``(nx*ny*nz*N, 3)``, in a box
            centered at the origin with extent ``(nx*Lx, ny*Ly, nz*Lz)`` Å.
        """
        n_tiles = nx * ny * nz
        assert (
            len(tile_positions) == n_tiles
        ), f"Expected {n_tiles} tiles, got {len(tile_positions)}"
        Lx, Ly, Lz = tile_box
        assembled: list[torch.Tensor] = []
        for idx, (ix, iy, iz) in enumerate(
            (ix, iy, iz) for iz in range(nz) for iy in range(ny) for ix in range(nx)
        ):
            offset = torch.tensor(
                [
                    (ix - (nx - 1) / 2.0) * Lx,
                    (iy - (ny - 1) / 2.0) * Ly,
                    (iz - (nz - 1) / 2.0) * Lz,
                ]
            )
            assembled.append(tile_positions[idx] + offset)
        return torch.cat(assembled, dim=0)


class MDSimDump:
    """
    Lazy reader and analyser for LAMMPS dump files from amorphous ice MD simulations.

    Indexes frame start positions on construction, then parses and trims
    coordinates on demand. The box geometric centre (from the dump header's
    box-bound lines) is used to centre each frame before trimming to a cubic
    region of side ``trim_size``.

    Parameters
    ----------
    filepath : str
        Path to the LAMMPS dump file.
    n : int, optional
        Number of voxels per axis for soft-voxelization. Default 200.
    dx : float, optional
        Voxel size in Å. Default 0.5.
    trim_size : float, optional
        Side length in Å of the cubic region extracted around the box centre.
        Default 100.0.

    Attributes
    ----------
    n_frames : int
        Total number of timestep frames found in the dump.
    coordinates : list of torch.Tensor or None
        Set by :meth:`get_coordinates`; one tensor per requested frame,
        each of shape ``(N_i, 3)`` in Å (N_i may vary between frames as
        atoms drift across the trim boundary).
    """

    # LAMMPS dump header is exactly 9 lines per frame:
    #   0: ITEM: TIMESTEP
    #   1: <timestep>
    #   2: ITEM: NUMBER OF ATOMS
    #   3: <n_atoms>
    #   4: ITEM: BOX BOUNDS ...
    #   5: xlo xhi
    #   6: ylo yhi
    #   7: zlo zhi
    #   8: ITEM: ATOMS id type x y z
    #   9+: atom data
    _HEADER_LINES = 9

    def __init__(
        self,
        filepath: str,
        n: int = 200,
        dx: float = 0.5,
        trim_size: float = 100.0,
    ) -> None:
        self.n = n
        self.dx = dx
        self.trim_size = trim_size

        fov = n * dx
        if fov > trim_size:
            warnings.warn(
                f"Field of view {fov:.1f} Å (n={n}, dx={dx} Å) exceeds "
                f"trim_size={trim_size:.1f} Å. Voxelized volumes will contain "
                "zero-padded regions outside the trimmed coordinate cube.",
                stacklevel=2,
            )

        with open(filepath) as f:
            self._lines = f.readlines()

        self._frame_starts: list[int] = [
            i for i, ln in enumerate(self._lines) if ln == "ITEM: TIMESTEP\n"
        ]
        self.n_frames: int = len(self._frame_starts)
        self._box_center: torch.Tensor = self._parse_box_center(0)
        self.coordinates: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_box_center(self, frame_idx: int) -> torch.Tensor:
        """Return (cx, cy, cz) from the box-bound lines of one frame.

        Handles both orthogonal (``lo hi``) and triclinic (``lo hi tilt``)
        LAMMPS dump formats by using only the first two tokens per line.
        """
        fs = self._frame_starts[frame_idx]
        center = []
        for offset in (5, 6, 7):
            parts = self._lines[fs + offset].split()
            lo, hi = float(parts[0]), float(parts[1])
            center.append((lo + hi) / 2.0)
        return torch.tensor(center, dtype=torch.float32)

    def _trim_frame(self, frame_idx: int) -> torch.Tensor:
        """
        Parse, centre, and trim one frame's coordinates.

        Reads the atom lines for ``frame_idx``, subtracts the box centre
        (parsed from the dump header), then keeps only atoms within
        ±trim_size/2 along every axis.

        Parameters
        ----------
        frame_idx : int
            Index into the frame list (0-based).

        Returns
        -------
        coords : torch.Tensor
            Trimmed coordinates, shape ``(N_trimmed, 3)``, in Å relative to
            the box centre.
        """
        fs = self._frame_starts[frame_idx]
        n_atoms = int(self._lines[fs + 3])
        coord_start = fs + self._HEADER_LINES

        lines_block = self._lines[coord_start : coord_start + n_atoms]
        coords_np = np.array([ln.split()[2:5] for ln in lines_block], dtype=np.float32)
        coords = torch.from_numpy(np.ascontiguousarray(coords_np))
        coords -= self._box_center

        half = self.trim_size / 2.0
        mask = (
            (coords[:, 0].abs() <= half)
            & (coords[:, 1].abs() <= half)
            & (coords[:, 2].abs() <= half)
        )
        return coords[mask]

    def _resolve_frames(
        self,
        frames: int | Sequence[int] | torch.Tensor | None,
    ) -> list[int]:
        """
        Resolve the ``frames`` argument to a sorted list of frame indices.

        ``None`` returns all frames from index 10 onwards (frames 0–9 are
        pre-steady-state). A warning is raised if any index < 10 is requested.
        """
        if frames is None:
            return list(range(10, self.n_frames))

        if isinstance(frames, int):
            frame_list = [frames]
        elif isinstance(frames, torch.Tensor):
            frame_list = frames.tolist()
        else:
            frame_list = list(frames)

        if any(f < 10 for f in frame_list):
            warnings.warn(
                "Frames 0–9 are typically pre-steady-state and may not represent "
                "equilibrium amorphous ice. Consider using frames ≥ 10.",
                stacklevel=3,
            )

        out_of_range = [f for f in frame_list if f < 0 or f >= self.n_frames]
        if out_of_range:
            raise IndexError(
                f"Frame indices {out_of_range} are out of range "
                f"(dump has {self.n_frames} frames, indices 0–{self.n_frames - 1})."
            )

        return frame_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coordinates(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Parse and return trimmed coordinates for the specified frames.

        Results are stored as ``self.coordinates`` for subsequent calls to
        :meth:`get_voxels`, :meth:`compute_sk_3d`, and :meth:`compute_rdf`.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            Frame indices to retrieve. ``None`` (default) uses all frames
            from index 10 onwards (frames 0–9 are pre-steady-state).

        Returns
        -------
        coordinates : list of torch.Tensor
            One tensor per frame, each of shape ``(N_i, 3)`` in Å centred
            at the box origin. ``N_i`` may differ between frames as atoms
            drift across the trim boundary.
        """
        frame_list = self._resolve_frames(frames)
        self.coordinates = [self._trim_frame(i) for i in frame_list]
        return self.coordinates

    def get_voxels(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Soft-voxelize trimmed coordinates onto a cubic ``(n, n, n)`` grid.

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        voxels : torch.Tensor
            Shape ``(F, n, n, n)``, one voxelized cube per frame.
        """
        coords_list = self.get_coordinates(frames)
        voxels = [
            soft_voxelize_coordinates(
                c,
                grid_shape=(self.n, self.n, self.n),
                voxel_size=self.dx,
            )
            for c in coords_list
        ]
        return torch.stack(voxels)

    def compute_sk_3d(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the mean Fourier amplitude |FFT3| averaged over frames.

        Equivalent to the square root of the structure factor S(k) ensemble-
        averaged over the requested frames:  mean_f( |FFT3(vox_f)| ).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        sk_3d : torch.Tensor
            Shape ``(n, n, n)``, DC component at centre (fftshift applied).
        """
        voxels = self.get_voxels(frames)
        return torch.mean(torch.abs(fft3(voxels, shift=True)), dim=0)

    def compute_sk_radial(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spherically-averaged structure factor S(k).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.

        Returns
        -------
        k : torch.Tensor
            Spatial frequency values in Å⁻¹.
        sk_radial : torch.Tensor
            Radially-averaged S(k).
        """
        sk3d = self.compute_sk_3d(frames)
        r_bins, sk_radial = radial_profile_3d(sk3d, return_r=True)
        dk = 1.0 / (self.n * self.dx)
        k = r_bins.float() * dk
        return k, sk_radial

    def compute_rdf(
        self,
        frames: int | Sequence[int] | torch.Tensor | None = None,
        dr: float = 0.5,
        r_max: float | None = None,
        chunk_size: int | None = None,
        approximate: bool = False,
        n_samples: int = 1_000_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the radial distribution function g(r) averaged over frames.

        Delegates to :func:`specter.coords.radial_distribution_function` with
        ``volume = trim_size³`` and per-frame number density.  No periodic
        boundary conditions are applied (coordinates are from a trimmed,
        non-periodic cube).

        Parameters
        ----------
        frames : int or sequence of int or Tensor or None, optional
            See :meth:`get_coordinates`.
        dr : float, optional
            Bin width in Å. Default 0.5.
        r_max : float, optional
            Maximum radius in Å. Defaults to ``trim_size`` (the side length
            of the trimmed cube).
        chunk_size : int or None, optional
            Row block size for chunked exact mode. If ``None``, uses
            ``torch.pdist`` (full O(N²) — exact but high memory for large N).
        approximate : bool, optional
            Use random-pair-sampling approximation. Default False.
        n_samples : int, optional
            Pairs drawn in approximate mode. Default 1 000 000.

        Returns
        -------
        r : torch.Tensor
            Bin-centre radii in Å.
        g_r : torch.Tensor
            g(r) averaged across the requested frames.
        """
        if r_max is None:
            r_max = self.trim_size

        coords_list = self.get_coordinates(frames)
        volume = self.trim_size**3

        gr_sum: torch.Tensor | None = None
        r_out: torch.Tensor | None = None
        for coords in coords_list:
            r, gr = radial_distribution_function(
                coords,
                volume=volume,
                dr=dr,
                r_max=r_max,
                chunk_size=chunk_size,
                approximate=approximate,
                n_samples=n_samples,
            )
            if gr_sum is None:
                gr_sum = gr
                r_out = r
            else:
                gr_sum = gr_sum + gr

        assert gr_sum is not None and r_out is not None
        return r_out, gr_sum / len(coords_list)

    def __repr__(self) -> str:
        return (
            f"MDSimDump(n_frames={self.n_frames}, n={self.n}, "
            f"dx={self.dx} Å, trim_size={self.trim_size} Å)"
        )


class NaiveIcemaker(L.LightningModule):
    """
    Creates ice through random molecule placement.

    Calculates the number of ice molecules based on amorphous ice density,
    randomly places them, and convolves with a scattering kernel.

    Parameters
    ----------
    dx : float
        Pixel size in angstroms.
    n : int
        Number of pixels in xy-axis. Assumes a square field-of-view.
    nz : float, optional
        Thickness of ice in Angstroms. If None, defaults to `n * dx`. Default is None.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    """

    def __init__(
        self, dx: float, n: int, nz: int | None = None, progressbars: bool = True
    ):
        super().__init__()

        self.dx = dx
        self.n = n

        if nz is None:
            self.nz = n
        else:
            self.nz = nz

        self.dv = dx**3  # voxel volume
        self.nv = n**2 * self.nz  # number of voxels
        self.total_vol = self.nv * self.dv  # total volume
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * self.total_vol)
        self.register_buffer("ice_kernel", self.create_ice_kernel())

        self.progressbars = progressbars

    def create_initial_ice_volume(self) -> torch.Tensor:
        """
        Create initial ice volume with randomly placed molecules.

        Returns
        -------
        ice_vol_init : torch.Tensor
            Binary tensor with 1s at molecule locations. Shape (nz, n, n).
        """
        ice_idx = torch.randint(0, self.nv, (self.n_ice_molecules,))

        ice_vol_init = torch.zeros(self.nv, device=self.device)
        ice_vol_init[ice_idx] = 1
        ice_vol_init = ice_vol_init.reshape(self.nz, self.n, self.n)  # z, y, x
        return ice_vol_init

    def create_ice_kernel(self, sn: int = 28) -> torch.Tensor:
        """
        Create atomic potential kernel for Oxygen using Kirkland parameterization.

        Parameters
        ----------
        sn : int, optional
            Size of the supersampled grid kernel. Default is 28.

        Returns
        -------
        pot : torch.Tensor
            Potential kernel volume.
        """
        # sample a 28x28 grid to represent kernel first.
        # 4xbin down to 7x7, centerd on atom origin
        sx = (torch.arange(sn) - (sn - 1) / 2) * self.dx / 4
        sZ, sY, sX = torch.meshgrid(sx, sx, sx, indexing="ij")
        sR = torch.sqrt(sX**2 + sY**2 + sZ**2)

        # see specter for details.
        a0 = 0.529  # Bohr radius, [Angstrom]
        e = 14.4  # electron charge, [V-Angstrom]
        c1 = 2 * (torch.pi**2) * a0 * e
        c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

        # P params for Oxygen. See Kirkland Appendix C.
        P = torch.tensor(
            [
                [3.39969204e-001, 3.81570280e-001, 3.07570172e-001, 3.81571436e-001],
                [1.30369072e-001, 1.91919745e001, 8.83326058e-002, 7.60635525e-001],
                [1.96586700e-001, 2.07401094e000, 9.96220028e-004, 3.03266869e-002],
            ]
        )
        P = P.T
        # tile scattering factors to match r_xy grid
        P = P[:, :, None, None, None].expand((4, 3) + sR.shape)

        s1 = c1 * torch.sum(
            P[0] / sR * torch.exp(-2 * torch.pi * sR * torch.sqrt(P[1])), 0
        )
        s2 = c2 * torch.sum(
            P[2] * P[3] ** (-3 / 2) * torch.exp(-(torch.pi**2) * (sR**2) / P[3]), 0
        )
        pot = s1 + s2

        avgpool3d = torch.nn.AvgPool3d(4, stride=4)
        return avgpool3d(pot[None, None]).squeeze()

    def generate_ice(
        self, batchsize: int = 1, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """
        Generate ice volumes.

        Parameters
        ----------
        batchsize : int, optional
            Number of ice volumes to generate. Default is 1.

        Returns
        -------
        icecubes : torch.Tensor
            Generated ice potential volumes. Shape (batchsize, nz, n, n).
        """
        if device is None:
            device = self.device
        icecubes = torch.zeros(batchsize, self.nz, self.n, self.n, device=device)
        for i in range(batchsize):
            self.icedeltas = self.create_initial_ice_volume()
            self.icecube = fftconvolve(self.icedeltas, self.ice_kernel, mode="same")
            icecubes[i] = self.icecube
        return icecubes


def assemble_volume_randomized(
    blocks: torch.Tensor, target_shape: tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Tile a batch of cubic blocks into a large volume with random augmentation per tile.

    Each tile receives an independent random roll, flip, and 90°-multiple rotation
    before being placed, so the assembled volume has no visible periodic seams.

    Parameters
    ----------
    blocks : torch.Tensor
        Bank of pre-generated blocks, shape (N_blocks, S, S, S). Blocks must be cubic.
    target_shape : tuple of int
        Desired output shape (N_batch, A, B, C).

    Returns
    -------
    torch.Tensor
        Assembled volume, shape (N_batch, A, B, C).
    """
    N_blocks, block_size, _, _ = blocks.shape
    N_batch, A, B, C = target_shape

    batch_volumes = []

    for b_idx in range(N_batch):
        # Compute number of blocks along each axis
        n_x = (A + block_size - 1) // block_size
        n_y = (B + block_size - 1) // block_size
        n_z = (C + block_size - 1) // block_size

        # Random indices for selecting blocks
        idx = torch.randint(0, N_blocks, (n_x, n_y, n_z))

        x_slices = []
        for i in range(n_x):
            y_slices = []
            for j in range(n_y):
                z_slices = []
                for k in range(n_z):
                    b = blocks[idx[i, j, k]]

                    # Random roll along all axes
                    shift_x = torch.randint(0, block_size, (1,)).item()
                    shift_y = torch.randint(0, block_size, (1,)).item()
                    shift_z = torch.randint(0, block_size, (1,)).item()
                    b = torch.roll(
                        b, shifts=(shift_x, shift_y, shift_z), dims=(0, 1, 2)
                    )

                    # Random flip along any axis
                    if torch.rand(1) < 0.5:
                        b = torch.flip(b, dims=(0,))
                    if torch.rand(1) < 0.5:
                        b = torch.flip(b, dims=(1,))
                    if torch.rand(1) < 0.5:
                        b = torch.flip(b, dims=(2,))

                    # Random rotation along any plane
                    k_rot_xy = torch.randint(0, 4, (1,)).item()
                    b = torch.rot90(b, k=k_rot_xy, dims=(0, 1))
                    k_rot_xz = torch.randint(0, 4, (1,)).item()
                    b = torch.rot90(b, k=k_rot_xz, dims=(0, 2))
                    k_rot_yz = torch.randint(0, 4, (1,)).item()
                    b = torch.rot90(b, k=k_rot_yz, dims=(1, 2))

                    z_slices.append(b)
                y_slices.append(torch.cat(z_slices, dim=2))
            x_slices.append(torch.cat(y_slices, dim=1))
        full_volume = torch.cat(x_slices, dim=0)

        # Crop to target shape
        batch_volumes.append(full_volume)

    # Stack all volumes along batch dimension
    return torch.stack(batch_volumes, dim=0)


def replace_outer_faces(tensors: torch.Tensor) -> torch.Tensor:
    """
    Replace the 6 outer faces of a batch of 3D tensors with random inner slices.

    Prevents boundary discontinuities when tiling ice blocks by filling each
    face with a copy of a randomly chosen interior slice.

    Parameters
    ----------
    tensors : torch.Tensor
        Batch of 3D volumes, shape (N, D, H, W). Modified in-place.

    Returns
    -------
    torch.Tensor
        Modified tensors with outer faces replaced.
    """
    N, D, H, W = tensors.shape

    if D <= 2 or H <= 2 or W <= 2:
        raise ValueError("Tensor too small to have inner slices")

    # Random inner indices for each tensor in the batch
    z_idx = torch.randint(1, D - 1, (N,))
    y_idx = torch.randint(1, H - 1, (N,))
    x_idx = torch.randint(1, W - 1, (N,))

    batch_idx = torch.arange(N)

    # Front/back faces
    tensors[batch_idx, 0, :, :] = tensors[batch_idx, z_idx, :, :]
    tensors[batch_idx, -1, :, :] = tensors[batch_idx, z_idx, :, :]

    # Top/bottom faces
    tensors[batch_idx, :, 0, :] = tensors[batch_idx, :, y_idx, :]
    tensors[batch_idx, :, -1, :] = tensors[batch_idx, :, y_idx, :]

    # Left/right faces
    tensors[batch_idx, :, :, 0] = tensors[batch_idx, :, :, x_idx]
    tensors[batch_idx, :, :, -1] = tensors[batch_idx, :, :, x_idx]

    return tensors
