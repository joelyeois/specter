from __future__ import annotations

import torch
import lightning as L

from .crowding import CrowdWithDuplicates
from .ice import IceBank


class TomogramGenerator(L.LightningModule):
    """
    Generates a 3D scattering potential volume (tomogram) by populating it with
    molecules and ice.

    This class modularizes the volume generation process, allowing it to be used
    independently of the imaging simulators. It combines template potentials
    (e.g., proteins from coordinates), crowding molecules, and amorphous ice.

    Parameters
    ----------
    pixel_size : float
        Voxel size in Å.
    nz : int
        Number of slices in Z.
    nxy : int
        Number of pixels in X and Y.
    scattering_potential : torch.Tensor, optional
        A template potential (e.g., proteins) to embed. Shape (Z, Y, X).
    crowd_min_distance : float, optional
        Minimum distance between crowding molecules in Å.
    crowd_max_distance_z : float, optional
        Range in Z where crowding molecules are placed.
    ice_model : str, optional
        Ice generation algorithm: ``'ap'`` (alternating projections), ``'gd'``
        (gradient descent), or ``'mcmc'`` (Metropolis Monte Carlo).
    num_unique_icecubes : int, optional
        Number of unique ice cubes pre-built into the ``IceBank``. Default 8.
    icecube_size : int, optional
        XY and Z side length in voxels of each ice cube in the bank.
        Cubes are tiled to fill the full volume. Default 256.
    ice_thickness : float, optional
        Thickness of the ice layer in Å.
    water_air_interface : bool, optional
        Whether to account for water-air interface in crowding and ice.
    progressbars : bool, optional
        Whether to show progress bars.
    chunk_size : int, optional
        Chunk size for parallel processing.
    """

    def __init__(
        self,
        pixel_size: float,
        nz: int,
        nxy: int,
        scattering_potential: torch.Tensor | None = None,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        num_unique_icecubes: int = 8,
        icecube_size: int = 256,
        water_air_interface: bool = True,
        progressbars: bool = True,
        chunk_size: int | None = None,
        save_clean_exitwaves: bool = False,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.nz = nz
        self.nxy = nxy
        self.scattering_potential = scattering_potential
        self.crowd_min_distance = crowd_min_distance
        self.crowd_max_distance_z = crowd_max_distance_z
        self.ice_model = ice_model
        self.ice_thickness = ice_thickness
        self.water_air_interface = water_air_interface
        self.progressbars = progressbars
        self.chunk_size = chunk_size
        self.save_clean_exitwaves = save_clean_exitwaves

        if self.crowd_min_distance is not None and scattering_potential is not None:
            self.crowd = CrowdWithDuplicates(
                scattering_potential,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=nxy,
                nz_out=nz,
                max_distance_z=self.crowd_max_distance_z,
                progressbars=progressbars,
                chunk_size=chunk_size,
                water_air_interface=water_air_interface,
            )
        else:
            self.crowd = None

        if self.ice_model is not None:
            if self.ice_model not in ("ap", "gd", "mcmc", "random"):
                raise ValueError(
                    f"Unknown ice_model '{self.ice_model}'. Choose 'ap', 'gd', 'mcmc', or 'random'."
                )
            self.icemaker = IceBank(
                n=icecube_size,
                dx=pixel_size,
                nz=icecube_size,
                method=self.ice_model,
            )
            self.icemaker.build(num_unique=num_unique_icecubes)
        else:
            self.icemaker = None

    def generate(self) -> torch.Tensor:
        """
        Generate the populated 3D volume.

        Returns
        -------
        V : torch.Tensor
            Populated 3D volume of shape (1, Z, Y, X).
        """
        device = self.device
        V = torch.zeros(1, self.nz, self.nxy, self.nxy, device=device)

        # 1. Add crowd
        if self.crowd is not None:
            with torch.no_grad():
                V_crowd = self.crowd()
                if not isinstance(V_crowd, float):
                    V = V + V_crowd.to(device)

        # Hold a reference to V before ice is added (V + ice creates a new tensor,
        # so this costs no extra memory).
        if self.save_clean_exitwaves and self.icemaker is not None:
            self.clean_V = V

        # 2. Add ice
        if self.icemaker is not None:
            with torch.no_grad():
                ice = self.icemaker.generate_big_ice(V.shape).to(device)
                icemask = (V < 0.05 * V.max()).to(V.dtype)
                V = V + ice * icemask

        return V
