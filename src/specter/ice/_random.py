from __future__ import annotations

from typing import Optional

import lightning as L
import torch

from ..arrays import soft_voxelize_coordinates
from ..fft import fftconvolve
from ._energy import MLBOP
from ._helpers import ndensity_of_amorphous_ice
from ._kernels import build_water_kernel


class RandomIcemaker(L.LightningModule):
    """
    Creates ice through random molecule placement.

    Calculates the number of ice molecules based on amorphous ice density,
    randomly places them as continuous Å coordinates, soft-voxelizes
    onto the grid, then convolves with a scattering kernel.

    Two-stage pipeline (mirrors :class:`GradientSKIcemaker`):

    1. :meth:`init_random` — draw ``n_ice_molecules`` uniform positions in
       ``[-box/2, box/2]`` Å.  Stored as ``self.positions`` with columns
       ``(x, y, z)``.
    2. :meth:`voxelize` — trilinear-splat positions onto ``(nz, n, n)``
       via :func:`~specter.arrays.soft_voxelize_coordinates`.

    Parameters
    ----------
    dx : float
        Pixel size in Å.
    n : int
        Number of pixels in xy-axis. Assumes a square field-of-view.
    nz : int, optional
        Number of pixels along z. If None, defaults to ``n``.
    parameterization : str, optional
        Atomic scattering-factor parameterization for the ice kernel:
        ``'kirkland'``, ``'lobato'``, or ``'shtyrov'``. Default
        ``'kirkland'``: Shtyrov fits bonded species of BIOMOLECULES over
        0.011-0.62 1/A, so bulk ice is out of its domain and its k=0 limit
        (which is what a mean inner potential is) extrapolates below the
        fitted range. Kirkland, Lobato and Peng are per-element and valid at
        k=0, and agree with each other there; see `build_water_kernel`.
    progressbars : bool, optional
        Whether to show progress bars. Default is True.
    """

    method: str = "random"

    def __init__(
        self,
        dx: float,
        n: int,
        nz: int | None = None,
        parameterization: str = "kirkland",
        progressbars: bool = True,
    ):
        super().__init__()

        self.dx = dx
        self.n = n
        self.nz = n if nz is None else nz
        self.parameterization = parameterization

        self.box_x: float = n * dx
        self.box_y: float = n * dx
        self.box_z: float = self.nz * dx

        self.dv = dx**3
        self.nv = n**2 * self.nz
        self.total_volume = self.nv * self.dv
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * self.total_volume)
        self.register_buffer(
            "ice_kernel",
            build_water_kernel(self.dx, self.parameterization),
            persistent=False,
        )

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
        ``(n_ice_molecules, 3)`` with columns ``(x, y, z)`` in Å,
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
        boundary conditions, consistent with :class:`GradientSKIcemaker`.

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
        whole-water-molecule kernel (`build_water_kernel`). ``self.positions`` holds the last
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mlbop_energy(self, pbc: bool = True) -> dict[str, float]:
        """
        Score the last-generated positions against the ML-BOP potential.

        See :meth:`specter.ice._energy.MLBOP.compute_energy`. Defaults to
        periodic boundaries (``pbc=True``): a generated block is itself the
        full periodic cell it was placed in, unlike e.g.
        :class:`~specter.ice.MDSimDump`'s hard-edge-trimmed MD frames.

        Parameters
        ----------
        pbc : bool, optional
            Whether to treat the block as periodic with box lengths
            ``(box_x, box_y, box_z)``. Default is True.

        Returns
        -------
        dict[str, float]
            See :meth:`specter.ice._energy.MLBOP.compute_energy` for the
            fields returned.
        """
        assert self.positions is not None, "No positions — call init_random() first"
        box = (self.box_x, self.box_y, self.box_z)
        model = MLBOP(device=self.positions.device)
        with torch.no_grad():
            result = model.compute_energy(self.positions, box_size=box, pbc=pbc)
        return {k: v.item() for k, v in result.items()}
