from __future__ import annotations

import os
import warnings

import lightning as L
import torch
import torch.nn.functional as F

from ..arrays import radial_profile_3d
from ..fft import fftconvolve
from ._helpers import ndensity_of_amorphous_ice, rfftn, torch_peak_local_max
from ._kernels import (
    MDSIM_DX,
    MDSIM_N,
    build_atomic_potential_kernel,
    ice_kspace_radial_grid,
    interpolate_target_kernel,
    load_mdsim_f_radial_avg,
)


class APIcemaker(L.LightningModule):
    """
    Generates 3D ice volumes with water-like molecular structure.

    Uses an alternating-projection iterative Fourier amplitude matching
    algorithm driven by MD simulation structure factors. Provides methods
    to load simulation data, compute radial averages, and iteratively
    generate ice volumes that match a target Fourier amplitude kernel.

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
    mdsim_target_path : str, optional
        Path to a precomputed radial-average |F(k)| target ``.pt`` file, in the
        same format as the bundled default (a 1D tensor indexed by k-bin on a
        400x400x400, 0.25 Å grid). If None, uses the bundled
        ``mdsim_f_radial_avg_400x400x400_0.25A.pt``. Default is None.

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
        mdsim_target_path: str | None = None,
    ):
        super().__init__()

        # load 3D radial average of mdsim data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        self.saved_data_path = mdsim_target_path or os.path.join(
            root_dir, "ice-data", "mdsim_f_radial_avg_400x400x400_0.25A.pt"
        )
        self.mdsim_dx = MDSIM_DX
        self.mdsim_n = MDSIM_N
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
        self.register_buffer("K", ice_kspace_radial_grid(n, self.nz, dx))

        # pre-compute ice kernel for algorithm
        self.interpolate_mdsim_f_kernel()

        self.register_buffer("ice_kernel", self.create_ice_kernel())

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
            mdsim_radial_k, mdsim_f_radial_avg = load_mdsim_f_radial_avg(
                saved_data_path
            )
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
        K = ice_kspace_radial_grid(self.n, self.nz, dx)
        return interpolate_target_kernel(
            K, self.mdsim_radial_k, self.mdsim_f_radial_avg, n_ice_molecules, half=True
        )

    def interpolate_mdsim_f_kernel(self) -> None:
        """
        Generate a 3D Fourier amplitude kernel for ice generation.

        Interpolates MD simulation radial averages to the current grid.
        Updates `self.interp_radial_k`, `self.interp_f_radial_avg`, `self.interp_f_kernel`,
        and `self.interp_f_halfkernel`.
        """
        interp_f_kernel = interpolate_target_kernel(
            self.K, self.mdsim_radial_k, self.mdsim_f_radial_avg, self.n_ice_molecules
        )
        self.register_buffer("interp_f_kernel", interp_f_kernel)

        # register half kernel for rfftn
        self.register_buffer(
            "interp_f_halfkernel",
            interpolate_target_kernel(
                self.K,
                self.mdsim_radial_k,
                self.mdsim_f_radial_avg,
                self.n_ice_molecules,
                half=True,
            ),
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

        self.current_icedeltas: torch.Tensor
        self.register_buffer("current_icedeltas", self.ice_vol_init.clone())
        self.niter = niter
        if min_distance is None:
            min_distance = self.min_distance
        if dx is None:
            dx = self.dx
        if dx > 1.5:
            warnings.warn(
                f"APIcemaker.generate_ice_deltas at dx={dx:.2f} Å is above the "
                "~1.5 Å pixel-size limit at which peak-finding can still "
                f"resolve the minimum O-O separation (min_distance={min_distance:.2f} Å) "
                "— two neighboring oxygens can collapse onto the same voxel, "
                "degrading ice quality. Prefer method='gd' (GradientSKIcemaker) "
                "at coarse pixel sizes; it doesn't rely on voxel-level peak "
                "finding.",
                stacklevel=2,
            )

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
        if dx is None:
            dx = self.dx
        return build_atomic_potential_kernel(dx, self.parameterization)

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
