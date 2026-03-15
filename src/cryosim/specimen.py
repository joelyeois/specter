from __future__ import annotations

import torch
import lightning as L

from .crowding import CrowdWithDuplicates
from .icemaker import Icemaker, NaiveIcemaker


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
        Name of the ice model ('iterative' or 'randomchoice').
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
        water_air_interface: bool = True,
        progressbars: bool = True,
        chunk_size: int | None = None,
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
            if self.ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(
                    dx=pixel_size, n=nxy, nz=nz, progressbars=progressbars
                )
            elif self.ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=min(nxy, 256),  # Icemaker usually works with power-of-2 blocks
                    dx=pixel_size,
                    nz=min(nz, 256),
                    chunk_size=chunk_size,
                    progressbars=progressbars,
                )
            else:
                self.icemaker = None
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

        # 2. Add ice
        if self.icemaker is not None:
            with torch.no_grad():
                if self.ice_model == "randomchoice":
                    ice = self.icemaker.generate_ice(device=device)
                else:  # iterative
                    ice = self.icemaker.generate_big_ice_fast(V.shape).to(device)

                # Combine with mask to avoid overwriting dense objects
                # Using 10V as threshold for 'dense'
                icemask = (V < 10.0).to(V.dtype)
                V = V + ice * icemask

        return V
