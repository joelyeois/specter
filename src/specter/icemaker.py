from __future__ import annotations

import os
from typing import Literal, Sequence

import torch
import lightning as L
import torch.nn.functional as F
from .progress import track
from torchinterp1d import interp1d

from . import potential
from .array_utils import (
    grid_3d,
    radial_grid_3d,
    radial_profile_3d,
    real_to_kgrid_3d,
    soft_voxelize_coordinates,
)
from .atom import kirkland_atomic_potential_3d, lobato_atomic_potential_3d
from .fft_tools import fft3, fftconvolve
from specter.pdbtools import PDB

avogadro = 6.02214076e23
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / Å³]


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
        # Assuming the repo structure is src/specter/icemaker.py and ice-data/ is at root
        # root is up 2 levels from current_dir
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        self.saved_data_path = os.path.join(
            root_dir, "ice-data", "mdsim_f_radial_avg_400x400x400_0.25A.pt"
        )
        self.mdsim_dx = 0.25
        self.mdsim_n = 400
        self.min_distance = min_distance

        self.mdsim_dk = 1 / self.mdsim_n / self.mdsim_dx
        self.get_mdsim_f_radial_avg(self.saved_data_path)
        self.chunk_size = chunk_size
        self.progressbars = progressbars
        self.parameterization = parameterization

        # self.ice_thickness = ice_thickness
        # if ice_thickness is None or ice_thickness < n * dx:
        #     self.nz = n
        #     if ice_thickness is not None and ice_thickness < n * dx:
        #         print(
        #             "Ice thickness smaller than particle size. Using minimum thickness."
        #         )
        #     self.ice_thickness = n * dx
        # else:
        #     self.nz = int(ice_thickness // dx)
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
            disable=not (self.progressbars),
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
                self.mdsim_ice_deltas, disable=not (self.progressbars)
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

        # Build kernel and set DC value
        interp_f_kernel = interp.reshape(self.nz, self.n, self.n)
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

        # replace DC value
        interp_f_kernel = interp.reshape(self.nz, self.n, self.n)
        interp_f_kernel[self.nz // 2, self.n // 2, self.n // 2] = self.n_ice_molecules
        # interp_f_kernel[self.nz // 2, self.n // 2, self.n // 2] = self.n_ice_molecules / self.nv
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
        add_extra_molecules: bool = True,
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

            # Add extra molecules to satisfy density. But this leads to bad results.
            # if add_extra_molecules:
            #     if len(peaks) < self.n_ice_molecules:
            #         n_extra = self.n_ice_molecules - len(peaks)
            #         self.n_extra_atoms.append(n_extra)

            #         # Find all empty locations
            #         zero_idx = (self.ice_vol == 0).nonzero(as_tuple=False)

            #         # Randomly choose n_extra of them
            #         perm = torch.randperm(zero_idx.shape[0], device=self.device)
            #         chosen = zero_idx[perm[:n_extra]]

            #         # Mark them as filled
            #         self.ice_vol[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1

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
        elif self.parameterization == "shryov":
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

        return avgpool3d(pot[None, None]).squeeze() * dx

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
            disable=not (self.progressbars),
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

    def generate_big_ice_old(self, shape: Sequence[int]) -> torch.Tensor:
        """
        Generate a large ice volume by stitching smaller generated blocks.

        Handles boundary conditions and overlaps to ensure continuity.

        Parameters
        ----------
        shape : tuple of int
            Target shape (B, nz, ny, nx).

        Returns
        -------
        big_ice : torch.Tensor
            Large ice volume.
        """
        B, nz, ny, nx = shape
        num_z = int(torch.ceil(torch.as_tensor(nz) / self.nz))
        num_y = int(torch.ceil(torch.as_tensor(ny) / self.n))
        num_x = int(torch.ceil(torch.as_tensor(nx) / self.n))
        N = B * num_z * num_y * num_x
        num_blocks_per_B = num_z * num_y * num_x

        # generate N ice cubes
        if self.chunk_size is None:
            ices = self.generate_ice(N)

            # assemble ice
            # ices shape: (N, nz, n, n)
            # Reshape to (B, num_z, num_y, num_x, nz, n, n)
            ices = ices.view(B, num_z, num_y, num_x, self.nz, self.n, self.n)

            # Permute to (B, num_z, nz, num_y, n, num_x, n)
            ices = ices.permute(0, 1, 4, 2, 5, 3, 6)

            # Reshape to (B, num_z*nz, num_y*n, num_x*n)
            big_ice = ices.reshape(B, num_z * self.nz, num_y * self.n, num_x * self.n)

            # trim
            big_ice = big_ice[:B, :nz, :ny, :nx]
        else:
            # generate batch of ice positions
            big_ice = torch.empty(B, num_z * self.nz, num_y * self.n, num_x * self.n)

            # for start in tqdm(range(0, N, self.chunk_size), desc='Generate ice positions', leave=False):
            for start in track(
                range(0, N, self.chunk_size),
                description="Generate ice positions",
                transient=True,
                disable=not (self.progressbars),
            ):
                end = min(start + self.chunk_size, N)
                batchsize = end - start

                # initialize & generate
                self.generate_ice_deltas(batchsize=batchsize)

                # get current batch (move to CPU if needed)
                batch_icedeltas = (
                    self.current_icedeltas.cpu()
                )  # shape (batchsize, self.nz, self.n, self.n)

                # directly insert into big_ice
                # We can vectorize this insertion too if we are careful, but chunking makes it tricky.
                # However, the inner loop over batchsize is slow.

                # Calculate indices for the whole batch
                global_indices = torch.arange(start, end)
                ib = global_indices // num_blocks_per_B
                local_idx = global_indices % num_blocks_per_B

                iz = local_idx // (num_y * num_x)
                iy = (local_idx % (num_y * num_x)) // num_x
                ix = local_idx % num_x

                # This part is hard to fully vectorize without advanced indexing which might be slow on CPU for large tensors
                # But we can at least remove the python loop
                for b in range(batchsize):
                    big_ice[
                        ib[b],
                        iz[b] * self.nz : (iz[b] + 1) * self.nz,
                        iy[b] * self.n : (iy[b] + 1) * self.n,
                        ix[b] * self.n : (ix[b] + 1) * self.n,
                    ] = batch_icedeltas[b]

        # resolve boundary conflicts where ice are too near
        min_distance_px = int(self.min_distance / self.dx)
        for ib in range(B):
            big_ice[ib] = clean_block_boundaries(
                big_ice[ib], (self.nz, self.n, self.n), min_distance_px, self.dx
            )

        # perform batchwise fft
        # We process the volume in batches of blocks to avoid high memory usage.
        # big_ice is (B, num_z*nz, num_y*n, num_x*n)

        # Total number of blocks
        N_blocks = B * num_z * num_y * num_x

        # We can iterate over blocks, extract them, convolve, and put them back.
        # To vectorize, we can process 'chunk_size' blocks at a time.
        chunk_size = 32  # Adjust based on GPU memory

        for i in track(
            range(0, N_blocks, chunk_size),
            description="Ice convolution",
            transient=True,
            disable=not (self.progressbars),
        ):
            # Identify which blocks belong to this chunk
            current_chunk_size = min(chunk_size, N_blocks - i)
            indices = torch.arange(i, i + current_chunk_size)

            # Map linear index to (ib, iz, iy, ix)
            # Note: The order depends on how we want to traverse.
            # Previously we filled big_ice with:
            # ib = global_idx // num_blocks_per_B
            # local_idx = global_idx % num_blocks_per_B
            # iz = local_idx // (num_y * num_x) ...
            # Let's stick to that mapping to be consistent, although for convolution it doesn't matter
            # as long as we cover everything.

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
                    z * self.nz : (z + 1) * self.nz,
                    y * self.n : (y + 1) * self.n,
                    x * self.n : (x + 1) * self.n,
                ]
                blocks.append(block)

            # Stack into a batch: (chunk_size, nz, n, n)
            batch = torch.stack(blocks).to(self.ice_kernel.device)

            # Convolve
            # ice_kernel shape: (nz, n, n) -> (1, nz, n, n)
            convolved_batch = fftconvolve(
                batch, self.ice_kernel.unsqueeze(0), mode="same", axes=(-3, -2, -1)
            )

            # Put back
            convolved_batch = (
                convolved_batch.cpu()
            )  # Move back to CPU if big_ice is on CPU
            for j in range(current_chunk_size):
                b, z, y, x = ib[j], iz[j], iy[j], ix[j]
                big_ice[
                    b,
                    z * self.nz : (z + 1) * self.nz,
                    y * self.n : (y + 1) * self.n,
                    x * self.n : (x + 1) * self.n,
                ] = convolved_batch[j]

        return big_ice[:B, :nz, :ny, :nx]

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

        # self.ice_thickness = ice_thickness
        # if ice_thickness is None or ice_thickness < n * dx:
        #     self.nz = n
        #     if ice_thickness is not None and ice_thickness < n * dx:
        #         print(
        #             "Ice thickness smaller than particle size. Using minimum thickness."
        #         )
        #     self.ice_thickness = n * dx
        # else:
        #     self.nz = int(ice_thickness // dx)
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
        # slowest, without duplicates
        # ice_idx = np.random.choice(self.n**3, self.n_ice_molecules, replace=False)

        # second fastest, without duplicates
        # ice_idx = torch.randperm(self.n**3)
        # ice_idx = ice_idx[:self.n_ice_molecules]

        # fastest, with duplicates
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
        return avgpool3d(pot[None, None]).squeeze() * self.dx

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


def remove_deltas_based_on_density(
    slab: torch.Tensor, expected_number: int | None = None, dx: float | None = None
) -> torch.Tensor:
    """
    Randomly remove delta functions from a slab to match expected density.

    Parameters
    ----------
    slab : torch.Tensor
        Input slab (binary tensor).
    expected_number : int, optional
        Expected number of particles. If None, calculated from `dx` and density.
    dx : float, optional
        Voxel size in Angstroms. Required if `expected_number` is None.

    Returns
    -------
    slab : torch.Tensor
        Processed slab with entries removed.
    """
    if expected_number is None:
        if dx is None:
            raise ValueError("dx must be specified.")
        else:
            expected_number = int(slab.numel() * dx**3 * ndensity_of_amorphous_ice)

    # Step 1: Get indices of all 1s
    ones_indices = (slab == 1).nonzero(as_tuple=False)  # shape: [num_ones, 3]
    current_number = len(ones_indices)
    # Step 2: Randomly pick N indices
    if current_number > expected_number:
        N = current_number - expected_number
        selected_idx = ones_indices[torch.randperm(len(ones_indices))[:N]]

        # Step 3: Set selected positions to 0
        slab[selected_idx[:, 0], selected_idx[:, 1], selected_idx[:, 2]] = 0
        return slab
    else:
        return slab


def clean_block_boundaries(
    bigblock: torch.Tensor,
    shape: tuple,
    min_dist: int,
    dx: float,
) -> torch.Tensor:
    """
    Remove excess ice density at block boundaries after stitching.

    Parameters
    ----------
    bigblock : torch.Tensor
        Large stitched ice volume.
    shape : tuple
        Shape of individual blocks (d, h, w).
    min_dist : int
        Minimum distance in pixels to check around boundaries.
    dx : float
        Voxel size in Angstroms.

    Returns
    -------
    bigblock : torch.Tensor
        Cleaned ice volume.
    """
    D, H, W = bigblock.shape
    d, h, w = shape  # block sizes

    nD = D // d
    nH = H // h
    nW = W // w

    # Depth boundaries
    for bd in range(1, nD):
        start = bd * d - min_dist * 2
        end = bd * d + min_dist * 2
        start = max(start, 0)
        end = min(end, D)
        slab = bigblock[start:end, :, :]
        bigblock[start:end, :, :] = remove_deltas_based_on_density(slab, dx=dx)

    # Height boundaries
    for bh in range(1, nH):
        start = bh * h - min_dist * 2
        end = bh * h + min_dist * 2
        start = max(start, 0)
        end = min(end, H)
        slab = bigblock[:, start:end, :]
        bigblock[:, start:end, :] = remove_deltas_based_on_density(slab, dx=dx)

    # Width boundaries
    for bw in range(1, nW):
        start = bw * w - min_dist * 2
        end = bw * w + min_dist * 2
        start = max(start, 0)
        end = min(end, W)
        slab = bigblock[:, :, start:end]
        bigblock[:, :, start:end] = remove_deltas_based_on_density(slab, dx=dx)

    return bigblock


def assemble_volume_randomized(blocks: torch.Tensor, target_shape):
    """
    Assemble a batch of 3D volumes from small blocks randomly, with random roll, flip, and rotation.

    Args:
        blocks: torch.Tensor of shape (N_blocks, 256, 256, 256)
        target_shape: tuple (N_batch, A, B, C) specifying output batch size and volume shape

    Returns:
        batch_volume: torch.Tensor of shape (N_batch, A, B, C)
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
    Fully vectorized across batch (N < 20 is small, so fine).

    Args:
        tensors: torch.Tensor of shape (N, D, H, W)

    Returns:
        tensors: modified in-place
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
