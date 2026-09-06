from __future__ import annotations

import torch
import lightning as L

from ..arrays import compute_nz
from ..crowding import CrowdWithDuplicates
from ..ice import (
    IceBank,
    IceProfile,
    RandomIcemaker,
    blend_ice_into_volume,
    resolve_icemaker,
)
from ..progress import status
from ..settings import Crowding, Ice, Packing


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
    template : torch.Tensor, optional
        The particle potential to embed, shape (Z, Y, X). None gives an
        empty (ice-only) specimen, for which ``nz`` is required.
    pixel_size : float
        Voxel size in Å.
    nxy : int
        Number of voxels in X and Y.
    nz : int, optional
        Number of slices in Z. Defaults to the depth the ice asks for: the
        thickest column of ``ice.profile``, else ``ice.thickness`` (at least
        the template's own depth), else the template's depth.
    crowding : Crowding, optional
        How duplicates of the template are packed into the volume. Default
        ``Crowding()``, which places none.
    packing : Packing, optional
        The collision backend for those duplicates. The ``"shape"`` backend
        needs ``atom_coordinates``. Default ``Packing()``, Poisson-disk.
    atom_coordinates : torch.Tensor, optional
        The template's real atomic coordinates (``PDB.coordinates``),
        required by ``packing.backend="shape"``.
    ice : Ice, optional
        The amorphous ice: model, thickness (or a laterally varying
        ``profile``, in which case particle placement is gated on each
        column's own slab), library, seam relaxation and scattering factors.
        Default ``Ice()``, no ice.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across generators. When
        supplied, ``ice.model`` and ``ice.cache_dir`` are ignored.
    move_to_cpu : bool, optional
        Assemble the volume on the host rather than the compute device: a
        micrograph canvas is tens of GB at 4096 px, and the imager streams
        it slice by slice when it does not fit. Default True.
    progressbars : bool, optional
        Whether to show progress bars.
    save_clean_exitwaves : bool, optional
        Keep the pre-ice volume as ``clean_V`` (a whole extra canvas), for
        an imager that wants the ice-free exit wave. Default False.
    """

    def __init__(
        self,
        template: torch.Tensor | None,
        pixel_size: float,
        nxy: int,
        nz: int | None = None,
        crowding: Crowding = Crowding(),
        packing: Packing = Packing(),
        atom_coordinates: torch.Tensor | None = None,
        ice: Ice = Ice(),
        icemaker: IceBank | RandomIcemaker | None = None,
        move_to_cpu: bool = True,
        progressbars: bool = True,
        save_clean_exitwaves: bool = False,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.nxy = nxy
        self.template = template
        self.crowding = crowding
        self.packing = packing
        self.atom_coordinates = atom_coordinates
        self.ice = ice
        self.ice_model = ice.model
        self.ice_profile: IceProfile | None = ice.profile
        self.ice_relax_steps = ice.relax_steps
        self.progressbars = progressbars
        self.move_to_cpu = move_to_cpu
        self.save_clean_exitwaves = save_clean_exitwaves

        if nz is None:
            if template is None:
                raise ValueError("nz is required when there is no template.")
            base_nz = int(template.shape[0])
            nz = (
                ice.profile.required_nz(nxy, pixel_size, base_nz)
                if ice.profile is not None
                else compute_nz(base_nz, ice.thickness, pixel_size)
            )
        self.nz = int(nz)
        # The ice's own thickness, not the box depth: the two differ when a
        # profile leaves part of the box empty.
        self.ice_thickness = (
            float(ice.profile.thickness(nxy, pixel_size).mean())
            if ice.profile is not None
            else self.nz * pixel_size
        )

        self.crowd: CrowdWithDuplicates | None
        if crowding.min_distance is not None and template is not None:
            self.crowd = CrowdWithDuplicates(
                template,
                pixel_size,
                crowding.min_distance,
                nxy_out=nxy,
                nz_out=self.nz,
                packing_backend=packing.backend,
                atom_coordinates=atom_coordinates,
                gap=packing.gap,
                n_orientations=packing.n_orientations,
                packing_max_retries=packing.max_retries,
                packing_stall_patience=packing.stall_patience,
                packing_seed=packing.seed,
                n_candidates=packing.n_candidates,
                max_distance_z=(
                    crowding.max_distance_z
                    if crowding.max_distance_z is not None
                    else self.nz * pixel_size
                ),
                max_distance_xy=(
                    crowding.max_distance_xy
                    if crowding.max_distance_xy is not None
                    else nxy * pixel_size
                ),
                method=crowding.method,
                n_points=(
                    crowding.n_points if crowding.n_points is not None else torch.inf
                ),
                seed=crowding.seed,
                progressbars=progressbars,
                chunk_size=crowding.chunk_size,
                water_air_interface=crowding.water_air_interface,
                sigma_frac=crowding.sigma_frac,
                peak_amplitude=crowding.peak_amplitude,
                baseline=crowding.baseline,
                move_to_cpu=move_to_cpu,
                ice_profile=self.ice_profile,
            )
        else:
            self.crowd = None

        self.icemaker: IceBank | RandomIcemaker | None = resolve_icemaker(
            self.ice_model,
            pixel_size,
            nxy=nxy,
            nz=self.nz,
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=ice.parameterization,
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
        shape = (1, self.nz, self.nxy, self.nxy)
        V: torch.Tensor | None = None

        # 1. Add crowd
        if self.crowd is not None:
            with torch.no_grad():
                V_crowd = self.crowd()
                if not isinstance(V_crowd, float):
                    # `crowd()` already returns a full canvas; adopting it
                    # rather than adding it into a zeroed one saves a whole
                    # touched canvas (33.6 GB at micrograph_size).
                    V = V_crowd.to(assembly_device).reshape(shape)
                    del V_crowd
        if V is None:
            V = torch.zeros(shape, device=assembly_device)

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
