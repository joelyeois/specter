"""
`Field` (a micrograph-scale specimen recipe) and `Sample` (one drawn volume).

A particle `Specimen` is drawn afresh for every image. A micrograph's specimen
is a field of copies through a slab of ice, drawn once and imaged as it lies,
so the two steps are separate: :meth:`Field.draw` makes a `Sample`, and
:meth:`Sample.scatter` propagates the beam through it slice by slice. The
assembly itself is :class:`~specter.specimen.MicrographSpecimenGenerator`'s.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import torch

from ..aberrations._functions import defocus_midplane_shift
from ..arrays import pad_volume
from ..fft import RadialSpectrum, apply_radial_envelope_
from ..ice import (
    IceBank,
    IceProfile,
    RandomIcemaker,
    blend_ice_into_volume,
    resolve_icemaker,
)
from ..ice._blend import AbsorptionTarget
from ..potential import absorption_potential, aperture_lowpass, dose_damage_envelope
from ..random_seed import derive_seed, seeded_rng
from ..scattering import IterativeScattering
from ..settings import Camera, Crowding, Ice, Optics, Packing, Propagation
from ..specimen import MicrographSpecimenGenerator
from ..specimen._single_particle import AbsorptionRates
from . import _stages as stages
from ._data import Exposure, Poses, settings_group
from ._waves import ExitWave

if TYPE_CHECKING:
    from ._tilt import TiltSeries

__all__ = ["Field", "Sample"]


class _DoseSeries:
    """
    A fixed specimen kept in the form each exposure's damage is made from.

    The dose envelope on the specimen damages the dry specimen and not its
    ice, and a pre-assembled volume is imaged at more than one exposure. It
    is therefore held as the 3D spectrum of the dry specimen and the ice it
    was blended with, from which the damaged volume for an exposure is
    rendered, in place, with one inverse transform.
    """

    def __init__(
        self, dry: torch.Tensor, volume: torch.Tensor, voxel_size: float
    ) -> None:
        self.spectrum = RadialSpectrum(dry[0], voxel_size)
        self.solvent = None if volume is dry else volume - dry
        self.rendered: tuple[float, float] | None = None

    def render(
        self,
        volume: torch.Tensor,
        exposure: Exposure,
        dose: float,
        pre_exposure: float,
        voltage: float,
    ) -> None:
        """Write the specimen damaged for one exposure into `volume`, once."""
        key = (dose, pre_exposure)
        if self.rendered == key:
            return
        envelope = dose_damage_envelope(
            dose,
            pre_exposure,
            weighted=exposure.weighted,
            voltage=voltage,
            n_frames=exposure.n_frames,
            frame_weights=exposure.weights,
            frame_weights_max_frequency=exposure.weights_max_frequency,
        )
        with torch.no_grad():
            if self.spectrum.device != volume.device:
                self.spectrum.to(volume.device)
            self.spectrum.filter_into(volume[0], envelope)
            if self.solvent is not None:
                volume.add_(self.solvent.to(volume.device))
        self.rendered = key


@dataclass(frozen=True)
class Sample:
    """
    One drawn specimen volume, imaged by every view made of it.

    Attributes
    ----------
    volume : torch.Tensor
        The potential with its ice, ``(1, Z, Y, X)``, in volts.
    voxel_size : float
        In Angstrom.
    absorption : torch.Tensor, optional
        The mean-free-path absorption field, same shape, propagated beside
        the potential.
    uniform_absorption : float
        A spatially constant absorption potential in volts, applied as one
        factor on the exit wave instead of a field. Default 0.
    profile : IceProfile, optional
        The ice's lateral profile; the defocus refers to its scattering
        centroid, which differs from the box's midplane only when the ice's
        mid-plane is not level.
    dry : torch.Tensor, optional
        The same volume before its ice was added, for the clean reference.
    dose : float, optional
        The dose the sample was drawn for, when its damage or its ice depend
        on the exposure; it then refuses to be imaged at another.
    damaged, solvent_filtered : bool
        Whether the exposure has already acted on the specimen or its ice.
    """

    volume: torch.Tensor
    voxel_size: float
    absorption: torch.Tensor | None = None
    uniform_absorption: float = 0.0
    profile: IceProfile | None = None
    dry: torch.Tensor | None = field(default=None, repr=False)
    dose: float | None = None
    damaged: bool = False
    solvent_filtered: bool = False
    series: _DoseSeries | None = field(default=None, repr=False)

    def prepare(
        self,
        ice: Ice | None = None,
        *,
        voltage: float,
        icemaker: IceBank | RandomIcemaker | None = None,
        exposure: Exposure | None = None,
        damaged: bool = False,
        objective_aperture: float | None = None,
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> Sample:
        """
        Make a pre-assembled specimen ready to image untilted.

        The untilted counterpart of :meth:`tilt_series`: ice is blended into
        the dry volume once, wherever it has little existing scattering
        potential, the mean-free-path field is read off the dry specimen,
        and with `damaged` the specimen, not its ice, is damaged afresh for
        each exposure :meth:`scatter` is given.

        Parameters
        ----------
        ice : Ice, optional
            The ice; its ``thickness`` is ignored, since the volume's Z
            extent is fixed, and a ``profile`` confines it.
        voltage : float
            Accelerating voltage in kV; sets the mean free paths.
        icemaker : IceBank or RandomIcemaker, optional
            A prebuilt icemaker for the blend.
        exposure : Exposure, optional
            The one dose every image shares, required by
            ``Ice(motion_variance=...)``, which filters the ice once.
        damaged : bool, optional
            Damage the specimen per exposure. Default False.
        objective_aperture : float, optional
            Aperture semi-angle in mrad; requires
            ``absorption_model="inelastic_mfp"``.
        propagation : Propagation, optional
            How the exit wave is computed, or its fields as keywords
            (``scattering_model``, ``alpha``, ``absorption_model``,
            ``oversample``, ``pad_fft``, ...); the absorption fields decide
            the field.

        Returns
        -------
        Sample
        """
        p = settings_group(Propagation, propagation, settings)
        ice = ice if ice is not None else Ice()
        volume = self.volume
        maker = resolve_icemaker(
            ice.model,
            self.voxel_size,
            nxy=volume.shape[-1],
            nz=volume.shape[-3],
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=ice.parameterization,
        )
        has_ice = maker is not None
        rates, uniform = _absorption(p, objective_aperture, voltage, has_ice, ice)
        field_: torch.Tensor | None = None
        if rates is not None and not has_ice:
            # Vacuum around the specimen: only the specimen absorbs.
            removal = stages.resolve_removal_mfps(
                p, Optics(objective_aperture=objective_aperture), voltage
            )
            assert removal is not None
            with torch.no_grad():
                field_ = stages.mfp_absorption_field(
                    volume, self.voxel_size, removal, has_solvent=False
                )
        dry = volume
        filters_solvent = has_ice and ice.motion_variance is not None
        dose = None
        if has_ice:
            target = None
            if rates is not None:
                field_ = torch.empty_like(volume)
                target = AbsorptionTarget(field_, rates.specimen, rates.solvent)
            ice_filter = None
            if filters_solvent:
                if exposure is None:
                    raise ValueError("Ice(motion_variance=...) needs an exposure")
                doses = exposure.doses(1)
                dose = float(doses[0])
                ice_filter = stages.solvent_exposure_filter(
                    ice,
                    Camera(n_frames=exposure.n_frames),
                    self.voxel_size,
                    dose,
                    exposure.weights,
                    exposure.weights_max_frequency,
                )
            with torch.no_grad():
                volume = blend_ice_into_volume(
                    volume,
                    cast(IceBank | RandomIcemaker, maker),
                    self.voxel_size,
                    relax_steps=ice.relax_steps,
                    profile=ice.profile,
                    absorption=target,
                    ice_filter=ice_filter,
                )

        def lowpass(v: torch.Tensor, inplace: bool) -> torch.Tensor:
            if objective_aperture is None:
                return v
            area = v.shape[-1] * v.shape[-2]
            with torch.no_grad():
                return aperture_lowpass(
                    v,
                    self.voxel_size,
                    objective_aperture,
                    voltage,
                    max_slices_per_chunk=max(1, min(64, 2**26 // area)),
                    out=v if inplace else None,
                )

        # In place only on a canvas this method allocated, never the caller's.
        volume = lowpass(volume, inplace=has_ice)
        series = None
        if damaged:
            if volume is dry:
                # Each exposure is rendered into the volume: never the
                # caller's tensor.
                volume = volume.clone()
                dry = volume
            series = _DoseSeries(
                lowpass(dry, inplace=False) if has_ice else volume,
                volume,
                self.voxel_size,
            )
        return replace(
            self,
            volume=volume,
            absorption=field_,
            uniform_absorption=uniform,
            profile=ice.profile,
            dose=dose,
            damaged=damaged,
            solvent_filtered=filters_solvent,
            series=series,
        )

    @classmethod
    def from_volume(cls, volume: torch.Tensor, voxel_size: float) -> Sample:
        """
        A pre-assembled specimen, ``(Z, Y, X)`` or ``(1, Z, Y, X)``, such as
        the volume `specter build tomogram` writes.
        """
        return cls(volume if volume.ndim == 4 else volume[None], voxel_size)

    def tilt_series(
        self, poses: Poses, exposure: Exposure, **settings: Any
    ) -> TiltSeries:
        """
        This sample prepared for a set of tilted views; see `TiltSeries`.

        The sample must be dry: the series blends its own ice, before any
        padding, from its ``ice`` setting.
        """
        from ._tilt import TiltSeries

        return TiltSeries(self.volume, self.voxel_size, poses, exposure, **settings)

    @property
    def nxy(self) -> int:
        """Lateral size, in voxels."""
        return self.volume.shape[-1]

    @property
    def nz(self) -> int:
        """Depth, in slices."""
        return self.volume.shape[-3]

    def clean(self) -> Sample:
        """The volume before its ice was added, with no absorption or damage."""
        if self.dry is None:
            raise ValueError("the sample was drawn without keep_dry=True")
        return Sample(self.dry, self.voxel_size, profile=self.profile)

    def placed(self, device: torch.device | str) -> Sample:
        """
        The sample on `device` when it fits there, else on the host.

        The volume and its absorption field move together or not at all:
        the propagator reads them slice by slice side by side. A sample left
        on the host is streamed a slice at a time, which costs per-slice
        transfers rather than the run, and warns.
        """
        device = torch.device(device)
        if self.volume.device == device and (
            self.absorption is None or self.absorption.device == device
        ):
            return self
        try:
            volume = self.volume.to(device)
            absorption = None if self.absorption is None else self.absorption.to(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gb = self.volume.numel() * self.volume.element_size() / 1e9
            warnings.warn(
                f"the specimen volume ({gb:.1f} GB) and the fields paired with "
                f"it do not fit on {device}; streaming them from the host "
                "slice by slice. The result is unchanged; each slice costs a "
                "host-to-device transfer.",
                stacklevel=3,
            )
            return replace(
                self,
                volume=self.volume.cpu(),
                absorption=None if self.absorption is None else self.absorption.cpu(),
            )
        return replace(self, volume=volume, absorption=absorption)

    def scatter(
        self,
        *,
        voltage: float,
        ids: torch.Tensor | None = None,
        exposure: Exposure | None = None,
        slice_batchsize: int = 1,
        device: torch.device | str | None = None,
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> ExitWave:
        """
        Propagate the beam through the sample, untilted, once per image.

        Parameters
        ----------
        voltage : float
            Accelerating voltage in kV.
        ids : torch.Tensor, optional
            The images' indices in their run; their count is the number of
            exit waves. Default one image, index 0.
        exposure : Exposure, optional
            The images' exposure, one dose for all of them; must match the
            dose the sample was drawn for, when it was drawn for one, and is
            required to render a damage series (:meth:`prepare`).
        slice_batchsize : int, optional
            Slices propagated together. Default 1.
        device : torch.device or str, optional
            Where to compute; the sample is placed there if it fits and
            streamed from the host if not. Default the volume's device.
        propagation : Propagation, optional
            How the exit wave is computed, or its fields as keywords
            (``scattering_model``, ``alpha``, ``absorption_model``,
            ``oversample``, ``pad_fft``, ...).

        Returns
        -------
        ExitWave
            On the propagation grid.

        Notes
        -----
        Every image sees the same volume at the same pose, so with a unit
        potential scale the beam is propagated once and the exit wave
        repeated per image.
        """
        p = settings_group(Propagation, propagation, settings)
        ids = torch.zeros(1, dtype=torch.long) if ids is None else ids
        n = len(ids)
        dose = None
        if exposure is not None:
            doses = exposure.doses(n)
            if not torch.allclose(doses, doses[0].expand_as(doses)):
                raise ValueError(
                    "every image of a sample's batch must share one dose: they "
                    "are made from the one volume"
                )
            dose = float(doses[0])
        if self.dose is not None and dose is not None and dose != self.dose:
            raise ValueError(
                f"The sample was drawn for a dose of {self.dose} e-/A^2 and cannot "
                f"be imaged at {dose}: its damage and its ice depend on the "
                "exposure. Draw another."
            )
        device = torch.device(device) if device is not None else self.volume.device
        sample = self.placed(device)
        if self.series is not None:
            if exposure is None or dose is None:
                raise ValueError("a damaged sample needs the exposure it is imaged at")
            self.series.render(sample.volume, exposure, dose, 0.0, voltage)
        nxy, nz = self.nxy, self.nz
        # Opposite edges of a profile that varies across the field differ,
        # and the unpadded FFT joins them: a 200-800 A wedge was 16% off
        # within 200 px of its edges without padding, 0.6% with it.
        pad_fft = p.pads(
            self.profile is not None
            and (self.profile.mode != "flat" or self.profile.tilt != 0.0)
        )
        pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy
        # Every view of one sample sees the same specimen, so it is propagated
        # once and the wave shared between the images.
        V = pad_volume(sample.volume, nxy, nz, None, pad_fft, xy_pad_mode="reflect")
        scattering = IterativeScattering(
            pad_nxy,
            self.voxel_size,
            voltage,
            scattering_model=p.scattering_model,
            klim=p.klim,
            oversample=p.oversample,
            crop=False,
            alpha=p.alpha,
            progressbars=False,
        ).to(device)
        if sample.absorption is None:
            wave = scattering(V, pose=0, slice_batchsize=slice_batchsize)
            if self.uniform_absorption:
                # A constant absorption factorises out of every slice's
                # transmission function.
                wave = wave * math.exp(
                    -scattering.sigma
                    * self.voxel_size
                    * V.shape[1]
                    * self.uniform_absorption
                )
        else:
            absorption = pad_volume(
                sample.absorption,
                nxy,
                nz,
                None,
                pad_fft,
                xy_pad_mode="reflect",
            )
            wave = scattering(
                V, pose=0, slice_batchsize=slice_batchsize, absorption_source=absorption
            )
            del absorption
        if wave.shape[0] != n:
            # Materialised: MKL's CPU FFT rejects a stride-0 batch.
            wave = wave.expand(n, -1, -1).contiguous()
        linear = p.scattering_model == "ctf"
        has_depth = p.scattering_model not in ("projection", "ctf")
        shift = defocus_midplane_shift(nz, self.voxel_size)
        if self.profile is not None:
            # The defocus refers to the ice's scattering centroid, and the
            # wave leaves the exit face: the high-z face when the beam enters
            # at slice 0 (negative Ewald sign, the default), the low one
            # otherwise.
            toward_exit = 1.0 if p.ews_curvature_sign == "negative" else -1.0
            shift += toward_exit * self.profile.centroid(nxy, self.voxel_size)
        return ExitWave(
            array=wave,
            pixel_size=self.voxel_size,
            voltage=voltage,
            nxy=nxy,
            pad_nxy=pad_nxy,
            oversample=1 if linear else p.oversample,
            defocus_shift=shift if has_depth else 0.0,
            ids=ids,
            linear=linear,
            alpha=p.alpha,
            damaged=self.damaged,
            solvent_filtered=self.solvent_filtered,
        )


@dataclass(frozen=True)
class Field:
    """
    A micrograph-scale specimen: copies of a template through a slab of ice.

    A recipe, drawn into a `Sample` by :meth:`draw`, once per micrograph.

    Attributes
    ----------
    template : torch.Tensor or None
        The particle potential, ``(Z, Y, X)``; None for ice alone.
    voxel_size : float
        In Angstrom.
    nxy : int
        Lateral size of the field, in voxels.
    nz : int, optional
        Depth in slices; by default what the ice asks for (see
        :class:`~specter.specimen.MicrographSpecimenGenerator`).
    molecular_mass : float, optional
        The template's mass in daltons; sets how much ice each copy displaces.
    crowding, packing : Crowding, Packing
        How the copies are placed; set by :meth:`pack`.
    atom_coordinates : torch.Tensor, optional
        The template's atoms, for ``packing.backend="shape"``.
    ice : Ice
        Set by :meth:`solvate`.
    icemaker : IceBank or RandomIcemaker, optional
        The ice library, resolved once by :meth:`solvate`.
    damaged : bool
        Set by :meth:`damage`.
    on_cpu : bool
        Assemble on the host: a 4096-px field is tens of GB. Default True.
    """

    template: torch.Tensor | None
    voxel_size: float
    nxy: int
    nz: int | None = None
    molecular_mass: float | None = None
    crowding: Crowding = Crowding()
    packing: Packing = Packing()
    atom_coordinates: torch.Tensor | None = field(default=None, repr=False)
    ice: Ice = Ice()
    icemaker: IceBank | RandomIcemaker | None = field(default=None, repr=False)
    damaged: bool = False
    on_cpu: bool = True

    def pack(
        self,
        crowding: Crowding | None = None,
        *,
        packing: Packing | None = None,
        atom_coordinates: torch.Tensor | None = None,
        **settings: Any,
    ) -> Field:
        """Place copies of the template through the field; keywords are `Crowding`'s."""
        return replace(
            self,
            crowding=settings_group(Crowding, crowding, settings),
            packing=packing if packing is not None else self.packing,
            atom_coordinates=(
                atom_coordinates
                if atom_coordinates is not None
                else self.atom_coordinates
            ),
        )

    def solvate(
        self,
        ice: Ice | None = None,
        *,
        icemaker: IceBank | RandomIcemaker | None = None,
        **settings: Any,
    ) -> Field:
        """Fill the field with amorphous ice; keywords are `Ice`'s."""
        ice = settings_group(Ice, ice, settings)
        probe = self._engine(ice, icemaker=icemaker)
        if probe.icemaker is None:
            raise ValueError("solvate() needs an ice model ('gd' or 'random')")
        return replace(self, ice=ice, icemaker=probe.icemaker)

    def damage(self) -> Field:
        """Apply each exposure's dose envelope to the specimen, before the ice."""
        return replace(self, damaged=True)

    def draw(
        self,
        *,
        voltage: float,
        exposure: Exposure | None = None,
        seed: int | None = None,
        index: int = 0,
        keep_dry: bool = False,
        objective_aperture: float | None = None,
        device: torch.device | str = "cpu",
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> Sample:
        """
        Assemble one realisation of the field.

        Parameters
        ----------
        voltage : float
            Accelerating voltage in kV; sets the mean free paths.
        exposure : Exposure, optional
            The one exposure the sample is drawn for, required by
            :meth:`damage` and by ``Ice(motion_variance=...)``.
        seed : int, optional
            Run seed; draw `index` comes from ``(seed, index, specimen
            stage)``, so the n-th micrograph of a seeded run is the same
            whatever came before it. Default None, the global RNG.
        index : int, optional
            Which draw of the run this is. Default 0.
        keep_dry : bool, optional
            Keep the volume before its ice, for :meth:`Sample.clean`; a whole
            extra canvas. Default False.
        objective_aperture : float, optional
            Aperture semi-angle in mrad, charged as absorption; requires
            ``absorption_model="inelastic_mfp"``.
        device : torch.device or str, optional
            Where the work is done; the volume itself is assembled on the
            host when ``on_cpu`` is set. Default "cpu".
        propagation : Propagation, optional
            How the exit wave is computed, or its fields as keywords
            (``scattering_model``, ``alpha``, ``absorption_model``,
            ``oversample``, ``pad_fft``, ...); the absorption fields decide
            which field is built here, from the dry specimen.

        Returns
        -------
        Sample
        """
        p = settings_group(Propagation, propagation, settings)
        if objective_aperture is not None and p.absorption_model != "inelastic_mfp":
            raise ValueError(
                "objective_aperture requires absorption_model='inelastic_mfp'"
            )
        if p.absorption_model == "inelastic_mfp" and p.scattering_model != "multislice":
            raise ValueError(
                "absorption_model='inelastic_mfp' requires "
                "scattering_model='multislice': the absorption field is "
                "propagated by IterativeScattering.multislice_absorptive"
            )
        engine = self._engine(self.ice, keep_dry=keep_dry).to(device)
        has_ice = engine.icemaker is not None
        rates, uniform = _absorption(p, objective_aperture, voltage, has_ice, self.ice)

        damage = None
        ice_filter = None
        dose = None
        filters_solvent = has_ice and self.ice.motion_variance is not None
        if self.damaged or filters_solvent:
            if exposure is None:
                raise ValueError(
                    "damage() and Ice(motion_variance=...) need an exposure"
                )
            doses = exposure.doses(1)
            dose = float(doses[0])
            if self.damaged:
                envelope = dose_damage_envelope(
                    dose,
                    0.0,
                    weighted=exposure.weighted,
                    voltage=voltage,
                    n_frames=exposure.n_frames,
                    frame_weights=exposure.weights,
                    frame_weights_max_frequency=exposure.weights_max_frequency,
                )

                def damage(V: torch.Tensor) -> torch.Tensor:
                    for item in V:
                        apply_radial_envelope_(item, self.voxel_size, envelope)
                    return V

            ice_filter = stages.solvent_exposure_filter(
                self.ice,
                Camera(n_frames=exposure.n_frames),
                self.voxel_size,
                dose,
                exposure.weights,
                exposure.weights_max_frequency,
            )

        def assemble() -> Any:
            return engine.assemble(
                absorption=rates, damage=damage, ice_filter=ice_filter
            )

        if seed is None:
            assembled = assemble()
        else:
            # The crowd and ice draws are on the CPU; the shape packer's
            # NumPy stream is seeded through `packing_seed` (mixed with the
            # one the caller set, if any), restored after.
            draw_seed = derive_seed(seed, index, stages.SEED_STAGE_SPECIMEN)
            crowd = engine.crowd
            packing_seed = None if crowd is None else crowd.packing_seed
            try:
                if crowd is not None:
                    crowd.packing_seed = stages.packer_seed(draw_seed, packing_seed)
                with seeded_rng(draw_seed):
                    assembled = assemble()
            finally:
                if crowd is not None:
                    crowd.packing_seed = packing_seed
        volume, absorption = assembled.volume, assembled.absorption
        dry = getattr(engine, "clean_V", None) if keep_dry else None
        if dry is volume and objective_aperture is not None:
            # With no ice and no damage the volume is its own clean form, and
            # the aperture below filters the volume in place.
            dry = dry.clone()
        if self.on_cpu:
            volume = volume.cpu()
            absorption = None if absorption is None else absorption.cpu()
        if objective_aperture is not None:
            area = volume.shape[-1] * volume.shape[-2]
            with torch.no_grad():
                volume = aperture_lowpass(
                    volume,
                    self.voxel_size,
                    objective_aperture,
                    voltage,
                    max_slices_per_chunk=max(1, min(64, 2**26 // area)),
                    out=volume,
                )
        return Sample(
            volume,
            self.voxel_size,
            absorption=absorption,
            uniform_absorption=uniform,
            profile=self.ice.profile,
            dry=dry,
            dose=dose,
            damaged=self.damaged,
            solvent_filtered=filters_solvent,
        )

    def _engine(
        self,
        ice: Ice,
        *,
        icemaker: IceBank | RandomIcemaker | None = None,
        keep_dry: bool = False,
    ) -> MicrographSpecimenGenerator:
        return MicrographSpecimenGenerator(
            self.template,
            self.voxel_size,
            self.nxy,
            nz=self.nz,
            crowding=self.crowding,
            packing=self.packing,
            atom_coordinates=self.atom_coordinates,
            ice=ice,
            icemaker=icemaker if icemaker is not None else self.icemaker,
            move_to_cpu=self.on_cpu,
            progressbars=False,
            save_clean_exitwaves=keep_dry,
            molecular_mass=self.molecular_mass,
        )


def _absorption(
    p: Propagation,
    objective_aperture: float | None,
    voltage: float,
    has_ice: bool,
    ice: Ice,
) -> tuple[AbsorptionRates | None, float]:
    """
    How the mean-free-path absorption is applied: as a field built from the
    rates while the ice is blended, or as one uniform potential, or neither.

    A constant absorption factorises out of each slice's transmission, so
    with ice filling the box and no specimen mean free path no field is
    built. A profile leaves vacuum in part of the box and needs the field.
    """
    removal = stages.resolve_removal_mfps(
        p, Optics(objective_aperture=objective_aperture), voltage
    )
    if removal is None:
        return None, 0.0
    solvent = absorption_potential(removal.removal("solvent"), voltage)
    if removal.inelastic_specimen is not None:
        specimen = absorption_potential(removal.removal("specimen"), voltage)
        return AbsorptionRates(specimen, solvent if has_ice else 0.0), 0.0
    if has_ice and ice.profile is None:
        return None, solvent
    if has_ice:
        return AbsorptionRates(solvent, solvent), 0.0
    return None, 0.0
