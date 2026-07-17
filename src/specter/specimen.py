from __future__ import annotations

import torch
import lightning as L

from .crowding import CrowdWithDuplicates
from .ice import IceBank
from .progress import status


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
        (gradient descent), or ``'mcmc'`` (Metropolis Monte Carlo). Ignored
        when ``icemaker`` is provided.
    num_unique_icecubes : int, optional
        Number of unique ice cubes pre-built into the ``IceBank``. Default 8.
        Ignored when ``icemaker`` is provided.
    icecube_size : int, optional
        XY and Z side length in voxels of each ice cube in the bank.
        Cubes are tiled to fill the full volume. Default 256. Ignored when
        ``icemaker`` is provided.
    icemaker : IceBank, optional
        A pre-built :class:`~specter.ice.IceBank` instance to reuse across
        multiple ``TomogramGenerator`` instances. When supplied, ``ice_model``,
        ``num_unique_icecubes``, and ``icecube_size`` are all ignored. The
        bank must already be built (i.e. :meth:`IceBank.build` has been
        called or will be triggered lazily on first use).
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
        icemaker: IceBank | None = None,
        water_air_interface: bool = True,
        progressbars: bool = True,
        chunk_size: int | None = None,
        move_to_cpu: bool = True,
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
        self.move_to_cpu = move_to_cpu
        self.save_clean_exitwaves = save_clean_exitwaves

        self.crowd: CrowdWithDuplicates | None
        if self.crowd_min_distance is not None and scattering_potential is not None:
            crowd_max_distance_z_ang = (
                crowd_max_distance_z
                if crowd_max_distance_z is not None
                else nz * pixel_size
            )
            self.crowd = CrowdWithDuplicates(
                scattering_potential,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=nxy,
                nz_out=nz,
                max_distance_z=crowd_max_distance_z_ang,
                max_distance_xy=nxy * pixel_size,
                progressbars=progressbars,
                chunk_size=chunk_size,
                water_air_interface=water_air_interface,
                move_to_cpu=move_to_cpu,
            )
        else:
            self.crowd = None

        self.icemaker: IceBank | None
        if icemaker is not None:
            self.icemaker = icemaker
            self.ice_model = icemaker.method
        elif self.ice_model is not None:
            if self.ice_model not in ("ap", "gd", "mcmc", "random"):
                raise ValueError(
                    f"Unknown ice_model '{self.ice_model}'. Choose 'ap', 'gd', 'mcmc', or 'random'."
                )
            self.icemaker = IceBank(
                n=icecube_size,
                dx=pixel_size,
                nz=icecube_size,
                method=self.ice_model,
                num_unique=num_unique_icecubes,
            )
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
        # Assemble on CPU when move_to_cpu is set — avoids holding two copies of the
        # full micrograph volume in VRAM simultaneously (crowd accumulator + V).
        assembly_device = torch.device("cpu") if self.move_to_cpu else device
        V = torch.zeros(1, self.nz, self.nxy, self.nxy, device=assembly_device)

        # 1. Add crowd
        if self.crowd is not None:
            with torch.no_grad():
                V_crowd = self.crowd()
                if not isinstance(V_crowd, float):
                    V = V + V_crowd.to(assembly_device)

        # Hold a reference to V before ice is added (V + ice creates a new tensor,
        # so this costs no extra memory).
        if self.save_clean_exitwaves and self.icemaker is not None:
            self.clean_V = V

        # 2. Add ice
        if self.icemaker is not None:
            with torch.no_grad():
                with status("Tiling ice volume", disable=not self.progressbars):
                    ice = self.icemaker.generate_big_ice(
                        (V.shape[0], V.shape[1], V.shape[2], V.shape[3])
                    ).to(assembly_device)
                with status("Applying ice mask", disable=not self.progressbars):
                    icemask = (V < 0.05 * V.max()).to(V.dtype)
                    V = V + ice * icemask

        return V
