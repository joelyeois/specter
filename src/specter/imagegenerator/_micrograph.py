"""
`MicrographGenerator`: images a whole specimen volume, or a
`MicrographSpecimenGenerator`'s output, as a micrograph.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import torch

from specter import logger

from ._base import BaseImager, RemovalMFPs, mfp_absorption_field, pad_volume
from ..ice import (
    IceBank,
    RandomIcemaker,
    blend_ice_into_volume,
    resolve_icemaker,
)
from ..ice._blend import AbsorptionTarget
from ..ice._profile import IceProfile
from ..potential import absorption_potential, aperture_lowpass
from ..progress import status
from ..scattering import IterativeScattering
from ..settings import Camera, Envelopes, Ice, Optics, Propagation
from ..specimen import MicrographSpecimenGenerator
from ..specimen._single_particle import AbsorptionRates


class MicrographGenerator(BaseImager):
    """
    Images a full specimen volume as a micrograph.

    The specimen is either a pre-assembled volume or a
    `MicrographSpecimenGenerator`, which builds one from a particle template
    with crowding and ice and can rebuild it for every micrograph
    (`regenerate_specimen`). Scattering is performed slice-by-slice using
    ``IterativeScattering``.

    Parameters
    ----------
    specimen : MicrographSpecimenGenerator or torch.Tensor
        What is imaged. A `MicrographSpecimenGenerator` carries its own
        crowding, packing and ice, and ``ice``/``icemaker`` below must then be
        left unset. A tensor is a pre-assembled volume of shape (1, Z, Y, X)
        -- e.g. the output of
        :func:`~specter.pipelines.build_tomogram_generator`/`specter build
        tomogram` -- imaged as is, except that ``ice``/``icemaker`` blend ice
        into it once, at construction, wherever it has little existing
        scattering potential (the same masking rule as ``ImageGenerator``'s
        ``solvate()``).
    micrograph_size : int or tuple[int, int]
        Output image size in pixels (must be square).
    pixel_size : float
        Pixel size in Å.
    ctf_params : dict[str, torch.Tensor] or None
        Per-micrograph CTF parameters; each value is a 1-D tensor of length
        n. Required unless ``optics`` is ``None``.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per micrograph in e⁻/Å². Scalar, or a
        1-D tensor of length n giving a separate dose for each micrograph.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices, shape (n, 2, 2).
    propagation : Propagation, optional
        How the exit wave is computed. Default ``Propagation()``.
    optics : Optics, optional
        The aberration stage; ``None`` skips it. Default ``Optics()``.
    envelopes : Envelopes, optional
        Coherence and radiation-damage envelopes. Default ``Envelopes()``.
    camera : Camera, optional
        The detector chain. Default ``Camera()``.
    ice : Ice, optional
        Ice blended into a tensor ``specimen`` at construction. ``thickness``
        is ignored, since the volume's Z extent is fixed; a ``profile`` still
        confines the ice. Default ``Ice()``, no ice.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker for that blend. When supplied, ``ice.model`` and
        ``ice.cache_dir`` are ignored.
    slice_batchsize : int, optional
        Number of Z slices propagated together in ``IterativeScattering``.
        Default 1.
    progressbars : bool, optional
        Show progress bars. Default True.
    verbose : bool, optional
        Emit debug-level log messages. Default True.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Default 0.0.
    potential_scale : float or torch.Tensor, optional
        Multiplier applied to the potential before scattering. Default 1.0.
    save_clean_exitwaves : bool, optional
        Also compute the exit wave of the ice-free specimen
        (``clean_exitwaves``). Needs a `MicrographSpecimenGenerator` built
        with ``save_clean_exitwaves=True``. Default False.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.

    Notes
    -----
    Under ``Propagation(absorption_model="inelastic_mfp")`` absorption is
    modelled as in the particle generators: the imaginary potential is the
    field of :func:`~specter.potential.inelastic_absorption_potential`, read
    off the DRY specimen with the occupancy that also weights its ice, with
    mean free paths resolved by
    :func:`~specter.imagegenerator._base.resolve_removal_mfps`. The ice exists
    separately from the specimen only while it is being blended, so the
    field is written by the blend itself, a slab at a time from the same
    occupancy read (:class:`~specter.ice._blend.AbsorptionTarget`); for a
    `MicrographSpecimenGenerator` that happens in
    :meth:`~specter.specimen.MicrographSpecimenGenerator.assemble`, once per
    specimen. The field is propagated alongside the potential by
    :meth:`~specter.scattering.IterativeScattering.multislice_absorptive`,
    and requires ``scattering_model="multislice"``.

    Without a specimen mean free path, and with ice filling the box, the
    field is uniform and no field is built. The beam is untilted, so every
    slice of the box is full of material, including the reflect-padded
    margin under ``pad_fft``, and a constant absorption factorises out of
    each slice's transmission function: the exit wave is multiplied by
    ``exp(-sigma * t * V_ab)`` for a box ``t`` thick, which is exact for the
    multislice recursion. An :class:`~specter.ice.IceProfile` leaves vacuum
    in part of the box and needs the field. With no ice and no specimen mean
    free path nothing absorbs, as in the particle path. An objective aperture
    low-passes each slice of the specimen volume with
    :func:`~specter.potential.aperture_lowpass`, in place, as the particle
    path does in its beam frame.

    The field is a second canvas the size of the specimen. It is placed on
    the compute device together with the volume, or both are streamed from
    the host when they do not fit.
    """

    def __init__(
        self,
        specimen: MicrographSpecimenGenerator | torch.Tensor,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any] | None,
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        propagation: Propagation = Propagation(),
        optics: Optics | None = Optics(),
        envelopes: Envelopes = Envelopes(),
        camera: Camera = Camera(),
        ice: Ice = Ice(),
        icemaker: IceBank | RandomIcemaker | None = None,
        slice_batchsize: int = 1,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        potential_scale: float | torch.Tensor = 1.0,
        save_clean_exitwaves: bool = False,
        bfactor: float | torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if isinstance(micrograph_size, int):
            nxy = micrograph_size
        elif (
            isinstance(micrograph_size, (tuple, list))
            and micrograph_size[0] == micrograph_size[1]
        ):
            nxy = micrograph_size[0]
        else:
            raise ValueError("micrograph_size must have same dimensions in x and y.")

        if (
            propagation.absorption_model == "inelastic_mfp"
            and propagation.scattering_model != "multislice"
        ):
            raise ValueError(
                f"{type(self).__name__}'s absorption_model='inelastic_mfp' "
                "requires scattering_model='multislice': the absorption field "
                "is propagated by IterativeScattering.multislice_absorptive."
            )
        self.pad_fft = propagation.pad_fft
        self.pad_nxy = nxy + (nxy // 2) * 2 if self.pad_fft else nxy

        volume: torch.Tensor | None
        specimen_gen: MicrographSpecimenGenerator | None = None
        if isinstance(specimen, MicrographSpecimenGenerator):
            if ice.model is not None or icemaker is not None:
                raise ValueError(
                    "A MicrographSpecimenGenerator carries its own ice; pass "
                    "`ice`/`icemaker` to it rather than to MicrographGenerator."
                )
            if specimen.nxy != nxy or specimen.pixel_size != pixel_size:
                raise ValueError(
                    f"The specimen is {specimen.nxy} px at {specimen.pixel_size} A "
                    f"but the micrograph is {nxy} px at {pixel_size} A."
                )
            volume = None
            specimen_gen = specimen
            self.nz = specimen.nz
            self.ice = specimen.ice
            self.ice_model = specimen.ice_model
            self.ice_profile = specimen.ice_profile
            self.ice_thickness = specimen.ice_thickness
            self.move_to_cpu = specimen.move_to_cpu
        elif isinstance(specimen, torch.Tensor):
            if specimen.ndim != 4:
                raise ValueError(
                    "specimen must be a (1, Z, Y, X) volume; wrap a particle "
                    "template in MicrographSpecimenGenerator to build one."
                )
            volume = specimen
            self.nz = volume.shape[1]
            self.ice = ice
            self.ice_model = ice.model
            self.ice_profile = ice.profile
            # Ice thickness, not box depth: the two are the same only when
            # the ice fills the box, which a profile breaks.
            self.ice_thickness = (
                float(ice.profile.thickness(nxy, pixel_size).mean())
                if ice.profile is not None
                else self.nz * pixel_size
            )
            self.move_to_cpu = False
        else:
            raise TypeError(
                "specimen must be a MicrographSpecimenGenerator or a volume "
                f"tensor, not {type(specimen).__name__}."
            )

        super().__init__(
            pixel_size=pixel_size,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            nxy=nxy,
            nz=self.nz,
            pad_nxy=self.pad_nxy,
            propagation=propagation,
            optics=optics,
            envelopes=envelopes,
            camera=camera,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=progressbars,
            verbose=verbose,
            coincidence_radius=coincidence_radius,
            potential_scale=potential_scale,
            bfactor=bfactor,
        )

        # A submodule can only be attached after Module.__init__.
        if specimen_gen is not None:
            self.specimen_gen = specimen_gen

        # Nothing below consumes ``kwargs``: a key that reaches here is a
        # misspelling or a setting that has moved into one of the settings
        # groups (``Propagation``, ``Camera``, ``Envelopes``, ...). Swallowing
        # it silently turned ``dose_envelope=True`` and, earlier,
        # ``bfactor_envelope=`` into no-ops, so it is an error.
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s) "
                f"{sorted(kwargs)}; settings now live in the Propagation, Camera, "
                "Envelopes, Optics, Ice and TiltGeometry groups"
            )

        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"],
            shift=(
                self.ice_profile.entry_face_shift(self.nxy, pixel_size)
                if self.ice_profile is not None
                else None
            ),
        )

        self._init_optics()
        self.slice_batchsize = slice_batchsize
        self._warned_volume_on_host = False
        self._warned_clean_volume_on_host = False
        self.iterative_scattering = IterativeScattering(
            self.pad_nxy,
            self.pixel_size,
            self.voltage,
            scattering_model=self.scattering_model,
            klim=self.klim,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        self.save_clean_exitwaves = save_clean_exitwaves
        # Like `volume` in TiltSeriesGenerator, kept out of the registered
        # buffers so that `.to(device)` does not upload a second canvas
        # unconditionally; `_ensure_volume_placed` places the two together.
        self.absorption_potential: torch.Tensor | None = None
        self._uniform_absorption_V = 0.0
        self._absorption_rates: AbsorptionRates | None = None

        if specimen_gen is not None:
            self._init_absorption(
                specimen_gen.icemaker is not None, specimen_gen.ice_profile
            )

        if volume is not None:
            volume_icemaker = resolve_icemaker(
                ice.model,
                pixel_size,
                nxy=volume.shape[-1],
                nz=volume.shape[-3],
                ice_cache_dir=ice.cache_dir,
                icemaker=icemaker,
                parameterization=ice.parameterization,
            )
            self._init_absorption(volume_icemaker is not None, ice.profile)
            rates = self._absorption_rates
            field: torch.Tensor | None = None
            if rates is not None and volume_icemaker is None:
                # Vacuum around the specimen: only the specimen absorbs.
                with torch.no_grad():
                    field = mfp_absorption_field(
                        volume,
                        pixel_size,
                        cast(RemovalMFPs, self._removal_mfps),
                        has_solvent=False,
                    )
            dry = volume
            if volume_icemaker is not None:
                if self.verbose:
                    logger.info(f"Adding ice to volume using {ice.model} model")
                target = None
                if rates is not None:
                    field = torch.empty_like(volume)
                    target = AbsorptionTarget(field, rates.specimen, rates.solvent)
                with (
                    torch.no_grad(),
                    status("Tiling ice volume", disable=not self.progressbars),
                ):
                    volume = blend_ice_into_volume(
                        volume,
                        volume_icemaker,
                        pixel_size,
                        relax_steps=ice.relax_steps,
                        profile=ice.profile,
                        absorption=target,
                    )
            # In place only on a canvas this class allocated, never on the
            # caller's tensor.
            volume = self._apply_aperture(volume, inplace=volume is not dry)
            self.register_buffer("volume", volume)
            self.absorption_potential = field

    def _init_absorption(self, has_ice: bool, profile: IceProfile | None) -> None:
        """
        Decide how the mean-free-path absorption is applied, if at all.

        Sets ``_absorption_rates`` when a field has to be built (see
        :meth:`~specter.specimen.MicrographSpecimenGenerator.assemble`), or
        ``_uniform_absorption_V`` when a scalar serves, which the class Notes
        explain. Neither under ``absorption_model="alpha"``.

        Parameters
        ----------
        has_ice : bool
            Whether ice is blended into the specimen.
        profile : IceProfile or None
            The ice's lateral profile; one leaves vacuum in the box.
        """
        removal = self._removal_mfps
        if removal is None:
            return
        solvent = absorption_potential(removal.removal("solvent"), self.voltage)
        if removal.inelastic_specimen is not None:
            self._absorption_rates = AbsorptionRates(
                specimen=absorption_potential(
                    removal.removal("specimen"), self.voltage
                ),
                solvent=solvent if has_ice else 0.0,
            )
        elif has_ice and profile is None:
            self._uniform_absorption_V = solvent
        elif has_ice:
            self._absorption_rates = AbsorptionRates(specimen=solvent, solvent=solvent)

    def _apply_aperture(
        self, volume: torch.Tensor, inplace: bool = True
    ) -> torch.Tensor:
        """
        Low-pass the specimen at the objective aperture, in place by default.

        Scattering beyond the aperture is charged as absorption
        (``RemovalMFPs.removal``), so the share of it the grid carries is
        filtered out of the elastic potential, as the particle path does. The
        beam is untilted, so the volume's slices are the beam frame's. A
        no-op without an aperture, and at voxel sizes coarse enough that it
        lies outside Nyquist.
        """
        if self.objective_aperture is None:
            return volume
        nxy = volume.shape[-1] * volume.shape[-2]
        with torch.no_grad():
            return aperture_lowpass(
                volume,
                self.pixel_size,
                self.objective_aperture,
                self.voltage,
                max_slices_per_chunk=max(1, min(64, 2**26 // nxy)),
                out=volume if inplace else None,
            )

    def _generate_volume(self) -> None:
        if self.verbose:
            logger.info(
                "Generating specimen volume (this may take a while for large micrographs)"
            )
        assembled = self.specimen_gen.assemble(absorption=self._absorption_rates)
        self.volume = assembled.volume
        self.absorption_potential = assembled.absorption
        if self.move_to_cpu:
            self.volume = self.volume.cpu()
            if self.absorption_potential is not None:
                self.absorption_potential = self.absorption_potential.cpu()
        self.volume = self._apply_aperture(self.volume)

    def regenerate_specimen(self) -> None:
        """
        Regenerate the specimen volume with fresh ice and crowding placement.

        Allows multiple independent micrographs from a single model instance
        without reinstantiating.

        Raises
        ------
        RuntimeError
            If the model was constructed with a pre-built ``volume`` (no
            ``specimen_gen`` available).
        """
        if not hasattr(self, "specimen_gen"):
            raise RuntimeError(
                "regenerate_specimen() requires the model to have been constructed "
                "with a MicrographSpecimenGenerator, not a pre-built volume."
            )
        self._generate_volume()

    def _ensure_volume_placed(self) -> None:
        """
        Move ``self.volume`` onto the compute device if it fits; keep it on the
        host and stream it if not.

        `MicrographSpecimenGenerator` assembles the specimen with
        ``move_to_cpu=True``, so the volume starts on the host. Uploading it is
        worth doing when it fits -- the scattering then reads it without a
        per-slice host transfer -- but at ``micrograph_size`` it often does not:
        the default config's 500 x 4096 x 4096 canvas is 33.5 GB, so an
        unconditional upload OOMs the very device ``move_to_cpu`` just
        moved the volume off.

        `IterativeScattering.multislice` accepts an off-device volume and
        streams it a slice at a time, so falling back costs per-slice
        transfers rather than the run. `TiltSeriesGenerator` inherits this
        and streams windowed blocks per z-chunk instead (see its ``volume``
        docstring). The warning is a `warnings.warn` rather than a
        `verbose`-gated print because the pipelines construct these
        generators with ``verbose=False``, so a print would never reach a
        CLI user whose run had silently taken the slow path.

        A no-op once the volume has settled on a device.

        With an absorption field the two are placed together or not at all:
        the propagator reads both slice by slice and needs them on one
        device.
        """
        if self.absorption_potential is None:
            if self.volume.device == self.device:
                return
            placed = self._place_on_device(self.volume, "specimen volume")
            if placed is not None:
                self.volume = placed
            return
        if (
            self.volume.device == self.device
            and self.absorption_potential.device == self.device
        ):
            return
        try:
            pair = (
                self.volume.to(self.device),
                self.absorption_potential.to(self.device),
            )
        except torch.cuda.OutOfMemoryError:
            self.volume = self.volume.cpu()
            self.absorption_potential = self.absorption_potential.cpu()
            torch.cuda.empty_cache()
            if not self._warned_volume_on_host:
                warnings.warn(
                    f"{type(self).__name__}: the specimen volume and its "
                    f"absorption field do not fit on {self.device} together; "
                    "streaming both from the host.",
                    stacklevel=2,
                )
                self._warned_volume_on_host = True
        else:
            self.volume, self.absorption_potential = pair

    def _place_on_device(self, V: torch.Tensor, what: str) -> torch.Tensor | None:
        """
        Upload `V` to the compute device, or return None if it does not fit.

        The one upload path for both the specimen volume and the ice-free
        ``clean_V``, so neither can OOM where the other streams from the
        host. Warns once per `what` on the fallback; see
        :meth:`_ensure_volume_placed` for why that is a `warnings.warn`.

        Parameters
        ----------
        V : torch.Tensor
            The volume to upload.
        what : str
            ``"specimen volume"`` or ``"clean specimen volume"``, naming it
            in the warning.

        Returns
        -------
        torch.Tensor or None
            `V` on ``self.device``, or None when the upload ran out of memory.
        """
        try:
            return V.to(self.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            flag = (
                "_warned_volume_on_host"
                if what == "specimen volume"
                else "_warned_clean_volume_on_host"
            )
            if not getattr(self, flag):
                setattr(self, flag, True)
                gb = V.numel() * V.element_size() / 1e9
                warnings.warn(
                    f"{type(self).__name__}: the {what} ({gb:.1f} GB) "
                    f"does not fit on {self.device}; keeping it in host memory "
                    "and streaming it slice by slice instead. The result is "
                    "unchanged and GPU memory stays bounded regardless of "
                    "micrograph_size, but each slice now costs a host-to-device "
                    "transfer. Reduce micrograph_size or ice_thickness to keep "
                    "the volume on the device.",
                    stacklevel=3,
                )
            return None

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate micrograph images for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated micrographs.
        """
        if not hasattr(self, "volume"):
            self._generate_volume()
        batchsize = len(idx) if isinstance(idx, torch.Tensor) else 1
        self._ensure_volume_placed()
        # Every image in the batch sees the same specimen at the same pose, so
        # with a unit potential scale the B exit waves are identical: the
        # volume is propagated once and the exit wave expanded to B before
        # the per-image CTF and detector (whose noise draws are unchanged).
        # The expansion is materialised: MKL's CPU FFT rejects a stride-0
        # batch, and the B exit waves were materialised before anyway.
        n_propagate = 1 if self._potential_scale_is_unity else batchsize

        def to_batch(exitwave: torch.Tensor) -> torch.Tensor:
            if exitwave.shape[0] == batchsize:
                return exitwave
            return exitwave.expand(batchsize, -1, -1).contiguous()

        V = self.volume.expand(n_propagate, -1, -1, -1)
        V = pad_volume(V, self.nxy, self.nz, None, self.pad_fft, xy_pad_mode="reflect")
        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1).to(V.device)
        # Skipped when every scale is 1 (the default), since `V * scale` is a
        # second full copy of a volume that may be tens of GB.
        if not self._potential_scale_is_unity:
            V = V * scale

        if (
            self.save_clean_exitwaves
            and hasattr(self, "specimen_gen")
            and hasattr(self.specimen_gen, "clean_V")
        ):
            clean = self.specimen_gen.clean_V
            if clean.device != self.device:
                placed = self._place_on_device(clean, "clean specimen volume")
                if placed is not None:
                    clean = placed
            V_clean = clean.expand(n_propagate, -1, -1, -1)
            V_clean = pad_volume(
                V_clean, self.nxy, self.nz, None, self.pad_fft, xy_pad_mode="reflect"
            )
            if not self._potential_scale_is_unity:
                V_clean = V_clean * scale.to(V_clean.device)
            self.clean_exitwaves = self.iterative_scattering(
                V_clean, pose=0, slice_batchsize=self.slice_batchsize
            )
            self.clean_exitwaves = to_batch(self.clean_exitwaves)

        if self.absorption_potential is None:
            self.exitwaves = self.iterative_scattering(
                V, pose=0, slice_batchsize=self.slice_batchsize
            )
            if self._uniform_absorption_V:
                # A constant absorption factorises out of every slice's
                # transmission function (see the class Notes).
                self.exitwaves = self.exitwaves * math.exp(
                    -self.iterative_scattering.sigma
                    * self.pixel_size
                    * V.shape[1]
                    * self._uniform_absorption_V
                )
        else:
            field = pad_volume(
                self.absorption_potential.expand(n_propagate, -1, -1, -1),
                self.nxy,
                self.nz,
                None,
                self.pad_fft,
                xy_pad_mode="reflect",
            )
            self.exitwaves = self.iterative_scattering(
                V,
                pose=0,
                slice_batchsize=self.slice_batchsize,
                absorption_source=field,
            )
            del field
        self.exitwaves = to_batch(self.exitwaves)

        self.detector_waves = self._aberrate(self.exitwaves, self._ctf_batch(idx))

        dose_batch = self.dose_per_angstrom[idx]
        cr_batch = self.coincidence_radius[idx]
        anisomag = None if self.anisomag is None else self.anisomag[idx]
        return self.detector(
            self.detector_waves, dose_batch, cr_batch, anisomag, nxy=self.nxy
        )
