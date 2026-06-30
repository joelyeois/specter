from __future__ import annotations

from typing import Optional

import lightning as L
import torch

from ..arrays import soft_voxelize_coordinates
from ..fft import fftconvolve
from ._helpers import ndensity_of_amorphous_ice


class RandomIcemaker(L.LightningModule):
    """
    Creates ice through random molecule placement.

    Calculates the number of ice molecules based on amorphous ice density,
    randomly places them as continuous Angstrom coordinates, soft-voxelizes
    onto the grid, then convolves with a scattering kernel.

    Two-stage pipeline (mirrors :class:`GradientSKIcemaker` and
    :class:`MCMCIcemaker`):

    1. :meth:`init_random` — draw ``n_ice_molecules`` uniform positions in
       ``[-box/2, box/2]`` Å.  Stored as ``self.positions`` with columns
       ``(x, y, z)``.
    2. :meth:`voxelize` — trilinear-splat positions onto ``(nz, n, n)``
       via :func:`~specter.arrays.soft_voxelize_coordinates`.

    Parameters
    ----------
    dx : float
        Pixel size in angstroms.
    n : int
        Number of pixels in xy-axis. Assumes a square field-of-view.
    nz : int, optional
        Number of pixels along z. If None, defaults to ``n``.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    """

    def __init__(
        self, dx: float, n: int, nz: int | None = None, progressbars: bool = True
    ):
        super().__init__()

        self.dx = dx
        self.n = n
        self.nz = n if nz is None else nz

        self.box_x: float = n * dx
        self.box_y: float = n * dx
        self.box_z: float = self.nz * dx

        self.dv = dx**3
        self.nv = n**2 * self.nz
        self.total_vol = self.nv * self.dv
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * self.total_vol)
        self.register_buffer("ice_kernel", self.create_ice_kernel())

        self.progressbars = progressbars
        self.positions: Optional[torch.Tensor] = None
        self.current_icedeltas: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Coordinate generation
    # ------------------------------------------------------------------

    def init_random(self) -> None:
        """
        Draw molecule positions uniformly at random.

        Positions are stored in ``self.positions`` as a float tensor of shape
        ``(n_ice_molecules, 3)`` with columns ``(x, y, z)`` in Angstroms,
        centered at the origin: ``x ∈ [-box_x/2, box_x/2]``, etc.
        """
        pos = torch.rand(self.n_ice_molecules, 3)
        pos[:, 0] = (pos[:, 0] - 0.5) * self.box_x
        pos[:, 1] = (pos[:, 1] - 0.5) * self.box_y
        pos[:, 2] = (pos[:, 2] - 0.5) * self.box_z
        self.positions = pos

    # ------------------------------------------------------------------
    # Voxelization
    # ------------------------------------------------------------------

    def voxelize(self) -> torch.Tensor:
        """
        Soft-voxelize ``self.positions`` onto the ``(nz, n, n)`` grid.

        Uses trilinear splatting via
        :func:`~specter.arrays.soft_voxelize_coordinates` with periodic
        boundary conditions, consistent with :class:`GradientSKIcemaker` and
        :class:`MCMCIcemaker`.

        Returns
        -------
        grid : torch.Tensor
            Soft-voxelized density, shape ``(nz, n, n)``.
        """
        assert self.positions is not None, "No positions — call init_random() first"
        return soft_voxelize_coordinates(
            self.positions.cpu(),
            grid_shape=(self.nz, self.n, self.n),
            voxel_size=self.dx,
            periodic=True,
        )

    # ------------------------------------------------------------------
    # Ice kernel
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_ice_deltas(self, batchsize: int = 1) -> torch.Tensor:
        """
        Generate soft-voxelized ice position volumes, without convolution.

        Each volume uses a fresh :meth:`init_random` + :meth:`voxelize` call.
        The last set of coordinates is left in ``self.positions`` and the full
        batch is stored as ``self.current_icedeltas``.

        Parameters
        ----------
        batchsize : int, optional
            Number of independent ice volumes. Default is 1.

        Returns
        -------
        icedeltas : torch.Tensor
            Soft-voxelized ice position volumes, shape ``(batchsize, nz, n, n)``.
        """
        results = []
        for _ in range(batchsize):
            self.init_random()
            results.append(self.voxelize())
        self.current_icedeltas = torch.stack(results)
        return self.current_icedeltas

    def generate_ice(
        self, batchsize: int = 1, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """
        Generate ice potential volumes.

        Runs :meth:`generate_ice_deltas` then convolves each volume with the
        Kirkland oxygen potential kernel. ``self.positions`` holds the last
        batch's coordinates; ``self.current_icedeltas`` holds the pre-convolution
        grids.

        Parameters
        ----------
        batchsize : int, optional
            Number of ice volumes to generate. Default is 1.
        device : torch.device or str, optional
            Target device for the output. Default is ``self.device``.

        Returns
        -------
        icecubes : torch.Tensor
            Generated ice potential volumes. Shape ``(batchsize, nz, n, n)``.
        """
        if device is None:
            device = self.device
        self.generate_ice_deltas(batchsize=batchsize)
        assert self.current_icedeltas is not None
        icecubes = fftconvolve(
            self.current_icedeltas.to(device),
            self.ice_kernel.unsqueeze(0).to(device),
            mode="same",
            axes=(-3, -2, -1),
        )
        return icecubes
