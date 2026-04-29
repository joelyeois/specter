from __future__ import annotations

import lightning as L
import torch

from ..fft import fftconvolve
from ._helpers import ndensity_of_amorphous_ice


class RandomIcemaker(L.LightningModule):
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
