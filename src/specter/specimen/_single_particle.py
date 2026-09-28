"""
`MicrographSpecimenGenerator`: a micrograph specimen from duplicates of one
particle template plus amorphous ice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
from ..ice._blend import AbsorptionTarget
from ..potential import potential_occupancy_slabs, template_occupancy_reference
from ..progress import status
from ..settings import Crowding, Ice, Packing


@dataclass(frozen=True)
class AbsorptionRates:
    """
    Absorption potentials, in volts, for a specimen's two materials.

    What :meth:`MicrographSpecimenGenerator.assemble` needs to build the
    mean-free-path absorption field: each voxel receives
    ``o * specimen + (1 - o) * w * solvent``, with ``o`` the occupancy the
    ice blend reads and ``w`` the ice's presence.

    Attributes
    ----------
    specimen : float
        Absorption potential of fully occupied material. Equal to `solvent`
        when the specimen absorbs at the ice's rate.
    solvent : float
        Absorption potential of the ice. Ignored when there is no ice, since
        vacuum does not absorb.
    """

    specimen: float
    solvent: float


@dataclass(frozen=True)
class AssembledSpecimen:
    """
    One assembled micrograph specimen.

    Attributes
    ----------
    volume : torch.Tensor
        The scattering potential with its ice, ``(1, Z, Y, X)``.
    absorption : torch.Tensor or None
        The mean-free-path absorption field, same shape, or None when none
        was requested or nothing absorbs.
    """

    volume: torch.Tensor
    absorption: torch.Tensor | None = None


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
    molecular_mass : float, optional
        Mass of the template's molecule in daltons, hydrogens included
        (:func:`~specter.potential.molecular_mass_from_atoms`). Sets how much
        ice each copy displaces: exactly its own volume at 0.73 cm^3/g,
        whatever scattering factors rendered it
        (:func:`~specter.potential.template_occupancy_reference`). Default None
        reads occupancy against the fixed
        :data:`~specter.potential.FULL_OCCUPANCY_POTENTIAL_V`.
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
        molecular_mass: float | None = None,
    ):
        super().__init__()
        self.molecular_mass = molecular_mass
        self._occupancy_reference_V: float | None = None
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

    def _occupancy_reference(self) -> float:
        """
        The potential at which a voxel of the template reads as full, V.

        Solved once, on first use, by
        :func:`~specter.potential.template_occupancy_reference`: every copy
        in the micrograph is the same template, so one reference serves them
        all. With no template there is nothing to displace ice, and the fixed
        reference is returned.
        """
        if self._occupancy_reference_V is None:
            self._occupancy_reference_V = template_occupancy_reference(
                self.template,
                self.pixel_size,
                self.molecular_mass if self.template is not None else None,
            )
        return self._occupancy_reference_V

    def generate(self) -> torch.Tensor:
        """
        Generate the populated 3D volume.

        Returns
        -------
        V : torch.Tensor
            Populated 3D volume of shape (1, Z, Y, X).
        """
        return self.assemble().volume

    def assemble(
        self,
        absorption: AbsorptionRates | None = None,
        damage: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ice_filter: Callable[[torch.Tensor], None] | None = None,
    ) -> AssembledSpecimen:
        """
        Generate the populated volume, and the fields read off its dry form.

        The specimen exists without its ice only inside this method: the
        blend writes the ice into the assembled canvas in place. Everything
        that must be read off the DRY specimen is therefore done here.

        Parameters
        ----------
        absorption : AbsorptionRates, optional
            Build the mean-free-path absorption field alongside the volume.
            With ice it is written by the blend itself, a slab at a time from
            the occupancy that weights the ice (see
            :class:`~specter.ice._blend.AbsorptionTarget`), so it costs one
            extra canvas and no extra blur. Without ice only the specimen
            term remains, and nothing is built unless the specimen absorbs.
            Default None.
        damage : callable, optional
            Applied to the dry specimen potential before the ice is added,
            in place, returning it: the dose envelope on the specimen
            (:func:`~specter.potential.apply_dose_damage`). The ice is still
            weighted by the occupancy of the UNDAMAGED specimen, and the
            absorption field is read off it too, since how much room a
            molecule takes up does not depend on its radiation history. With
            ice this costs a second canvas while the blend runs: the ice is
            blended into a copy of the undamaged specimen, and the difference
            is added back once the specimen has been damaged. Default None.
        ice_filter : callable, optional
            Applied in place to the unweighted ice canvas before it is
            blended: the solvent-exposure filter. Default None.

        Returns
        -------
        AssembledSpecimen
            The volume and, when requested, its absorption field.
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

        field: torch.Tensor | None = None
        target: AbsorptionTarget | None = None
        if absorption is not None and self.icemaker is not None:
            field = torch.empty_like(V)
            target = AbsorptionTarget(field, absorption.specimen, absorption.solvent)
        elif absorption is not None and absorption.specimen > 0.0:
            # No ice: vacuum surrounds the specimen and only it absorbs.
            field = self._specimen_absorption(V, absorption.specimen)

        if self.icemaker is None:
            if damage is not None:
                with torch.no_grad():
                    V = damage(V)
            return AssembledSpecimen(V, field)

        # 2. Add ice
        with torch.no_grad():
            with status("Tiling ice volume", disable=not self.progressbars):
                if damage is None:
                    V = self._blend(V, target, ice_filter, inplace=True)
                else:
                    # The ice is weighted by the undamaged specimen's
                    # occupancy, so it is blended into a copy of it and the
                    # difference, which is the weighted ice alone, is added to
                    # the specimen once that has been damaged.
                    solvent = self._blend(V, target, ice_filter, inplace=False)
                    solvent.sub_(V)
                    V = damage(V)
                    V.add_(solvent)
                    del solvent

        return AssembledSpecimen(V, field)

    def _blend(
        self,
        V: torch.Tensor,
        absorption: AbsorptionTarget | None,
        ice_filter: Callable[[torch.Tensor], None] | None,
        inplace: bool,
    ) -> torch.Tensor:
        """This specimen's ice blend, :func:`~specter.ice.blend_ice_into_volume`."""
        assert self.icemaker is not None
        return blend_ice_into_volume(
            V,
            self.icemaker,
            self.pixel_size,
            full_potential=self._occupancy_reference(),
            relax_steps=self.ice_relax_steps,
            profile=self.ice_profile,
            inplace=inplace,
            absorption=absorption,
            ice_filter=ice_filter,
        )

    def _specimen_absorption(self, V: torch.Tensor, specimen: float) -> torch.Tensor:
        """
        ``occupancy * specimen`` over a specimen with no ice around it.

        Read a z-slab at a time on the icemaker-free compute device, with the
        same occupancy reference the blend would use, and written into a
        field on `V`'s device.
        """
        field = torch.empty_like(V)
        chunk = max(1, 2**24 // (self.nxy * self.nxy))
        with torch.no_grad():
            for z0, z1, occ in potential_occupancy_slabs(
                V,
                self.pixel_size,
                chunk,
                full_potential=self._occupancy_reference(),
                device=self.device,
            ):
                field[:, z0:z1] = (occ * specimen).to(field.device)
        return field
