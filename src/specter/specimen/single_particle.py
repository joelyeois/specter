from __future__ import annotations

import torch
import lightning as L

from ..crowding import CrowdWithDuplicates
from ..ice import (
    IceBank,
    IceProfile,
    RandomIcemaker,
    blend_ice_into_volume,
    resolve_icemaker,
)
from ..progress import status


class MicrographSpecimenGenerator(L.LightningModule):
    """
    Generates a 3D scattering potential volume by populating it with many
    duplicate copies of ONE particle template plus amorphous ice.

    This class modularizes the volume generation process, allowing it to be used
    independently of the imaging simulators. It combines a single template
    potential (e.g., a protein from coordinates), crowding duplicates of that
    one template, and amorphous ice.

    Single-species only, unlike `specter.specimen.tomogram.
    TomogramSpecimenGenerator` (the `specter build tomogram` backend), which
    places any number of distinct species. The two also differ in what their
    placement step actually optimizes for, matching their different end
    goals:

    - Here, particle placement (`~specter.crowding.CrowdWithDuplicates`) is
      built for realistic SINGLE-PARTICLE MICROGRAPH statistics -- besides
      Poisson-disk minimum-distance packing, it supports an optional
      `water_air_interface` bias that skews the Z-distribution of placed
      copies toward the ice's top/bottom surfaces (real particles
      preferentially adsorb there during vitrification; see
      `~specter.crowding.filter_by_z_density`), since a plausible
      through-ice distribution matters for particle-picking-style
      benchmarking.
    - `TomogramSpecimenGenerator`'s protein placement (RSA hard-sphere
      packing) instead optimizes purely for DENSITY -- packing each region
      (cytosol/lumen) as densely as `occupancy_fraction` allows, uniformly
      throughout it. There is no equivalent distributional shaping there
      (no water-air-interface bias or similar) -- crowding realism there
      comes from region-gating against real membrane geometry, not from
      shaping any one species' own spatial statistics.

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
        Ice generation algorithm: ``'gd'`` (samples from the pre-generated
        :class:`~specter.ice.IceBank` cache) or ``'random'`` (instant, cheap
        :class:`~specter.ice.RandomIcemaker` placement -- no realism, useful
        as a fast baseline/smoke test). Ignored when ``icemaker`` is provided.
    ice_cache_dir : str, optional
        Directory of cached ice configs for ``ice_model='gd'`` (see
        :func:`specter.ice.build_ice_cache`). Defaults to the bundled
        ``ice_data/ice_cache``. Ignored for other ``ice_model`` values or
        when ``icemaker`` is provided.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across multiple
        ``MicrographSpecimenGenerator`` instances. When supplied, ``ice_model`` and
        ``ice_cache_dir`` are both ignored.
    ice_thickness : float, optional
        Thickness of the ice layer in Å. Ignored when ``ice_profile`` is given,
        which carries its own thickness.
    ice_profile : IceProfile, optional
        Laterally varying ice thickness (wedge, meniscus, tilted slab). Ice is
        confined to the profile instead of filling the box, and particle
        placement is gated on each column's own slab. The caller is
        responsible for having sized ``nz`` from
        :meth:`~specter.ice.IceProfile.required_nz`. Default None: a uniform
        slab filling the box.
    ice_relax_steps : int, optional
        Forwarded to :meth:`~specter.ice.IceBank.generate_big_ice` when
        ``ice_model='gd'`` (or an ``IceBank`` ``icemaker``): number of local
        MLBOP relaxation steps used to heal tile seams. Default 0 (no
        relaxation). Ignored for ``RandomIcemaker``.
    water_air_interface : bool, optional
        Whether to account for water-air interface in crowding and ice.
    sigma_frac : float, optional
        Forwarded to ``CrowdWithDuplicates``. Only used when
        ``water_air_interface=True``.
    peak_amplitude : float, optional
        Forwarded to ``CrowdWithDuplicates``. Only used when
        ``water_air_interface=True``.
    baseline : float, optional
        Forwarded to ``CrowdWithDuplicates``. Only used when
        ``water_air_interface=True``.
    progressbars : bool, optional
        Whether to show progress bars.
    chunk_size : int, optional
        Crowding duplicate volumes rotated per batch, forwarded to
        ``CrowdWithDuplicates``. Default 1; see that class for why raising it
        trades memory for nothing.
    packing_backend : {'poisson_disk', 'shape'}, optional
        Forwarded to ``CrowdWithDuplicates``. ``'shape'`` collides the
        template's real rotated footprint (via
        ``~specter.specimen.packing.pack_shapes_3d``) instead of
        bounding-sphere-exclusion Poisson-disk sampling, reaching
        substantially higher crowding density -- measured on a 512x512x256
        A benchmark of this class's own placement, 8.6x more instances and
        8.6x the occupied volume fraction at the same ``crowd_min_distance``
        and box. See ``CrowdWithDuplicates``'s own docstring for the
        mechanism, including how it handles ``ice_profile`` confinement and
        ``water_air_interface`` adsorption. Requires ``atom_coordinates``.
        Default ``'poisson_disk'``.
    atom_coordinates : torch.Tensor, optional
        The template's real atomic coordinates -- ``PDB.coordinates`` --
        required (and used only) when ``packing_backend='shape'``.
    packing_gap : float, optional
        Forwarded to ``CrowdWithDuplicates`` as ``gap``. Shape backend only.
    n_orientations : int, optional
        Forwarded to ``CrowdWithDuplicates``. Shape backend only.
    packing_max_retries : int, optional
        Forwarded to ``CrowdWithDuplicates``. Shape backend only.
    packing_stall_patience : int, optional
        Forwarded to ``CrowdWithDuplicates``. Shape backend only.
    packing_seed : int, optional
        Forwarded to ``CrowdWithDuplicates``. Shape backend only.
    n_candidates : int, optional
        Forwarded to ``CrowdWithDuplicates``. Shape backend only.
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
        ice_profile: IceProfile | None = None,
        ice_cache_dir: str | None = None,
        icemaker: IceBank | RandomIcemaker | None = None,
        ice_relax_steps: int = 0,
        ice_parameterization: str = "kirkland",
        water_air_interface: bool = True,
        sigma_frac: float = 0.05,
        peak_amplitude: float = 1.0,
        baseline: float = 0.1,
        progressbars: bool = True,
        chunk_size: int = 1,
        move_to_cpu: bool = True,
        save_clean_exitwaves: bool = False,
        packing_backend: str = "poisson_disk",
        atom_coordinates: torch.Tensor | None = None,
        packing_gap: float = 0.0,
        n_orientations: int = 256,
        packing_max_retries: int = 1500,
        packing_stall_patience: int = 5000,
        packing_seed: int | None = None,
        n_candidates: int | None = None,
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
        self.ice_profile = ice_profile
        self.ice_relax_steps = ice_relax_steps
        self.water_air_interface = water_air_interface
        self.progressbars = progressbars
        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.save_clean_exitwaves = save_clean_exitwaves
        self.packing_backend = packing_backend
        self.atom_coordinates = atom_coordinates

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
                packing_backend=packing_backend,
                atom_coordinates=atom_coordinates,
                gap=packing_gap,
                n_orientations=n_orientations,
                packing_max_retries=packing_max_retries,
                packing_stall_patience=packing_stall_patience,
                packing_seed=packing_seed,
                n_candidates=n_candidates,
                max_distance_z=crowd_max_distance_z_ang,
                max_distance_xy=nxy * pixel_size,
                progressbars=progressbars,
                chunk_size=chunk_size,
                water_air_interface=water_air_interface,
                sigma_frac=sigma_frac,
                peak_amplitude=peak_amplitude,
                baseline=baseline,
                move_to_cpu=move_to_cpu,
                ice_profile=ice_profile,
            )
        else:
            self.crowd = None

        self.icemaker: IceBank | RandomIcemaker | None = resolve_icemaker(
            self.ice_model,
            pixel_size,
            nxy=nxy,
            nz=nz,
            ice_cache_dir=ice_cache_dir,
            icemaker=icemaker,
            parameterization=ice_parameterization,
        )
        if icemaker is not None:
            self.ice_model = icemaker.method

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
                    # Adopted, not added into the zeros above. `crowd()` already
                    # returns a full canvas, so `V = V + V_crowd` allocated a
                    # third one to hold a sum whose other operand was all zeros
                    # -- 33.6 GB each at micrograph_size.
                    V = V_crowd.to(assembly_device).reshape(V.shape)
                    del V_crowd

        # Hold the pre-ice volume for the clean exit wave. This DOES cost a
        # whole canvas: the blend below writes in place precisely to avoid
        # allocating one, so there is no new tensor to keep instead.
        keep_clean = self.save_clean_exitwaves and self.icemaker is not None
        if keep_clean:
            self.clean_V = V.clone()

        # 2. Add ice
        if self.icemaker is not None:
            with torch.no_grad():
                with status("Tiling ice volume", disable=not self.progressbars):
                    V = blend_ice_into_volume(
                        V,
                        self.icemaker,
                        self.pixel_size,
                        relax_steps=self.ice_relax_steps,
                        profile=self.ice_profile,
                        inplace=True,
                    )

        return V
