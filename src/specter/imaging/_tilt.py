"""
`TiltSeries`: one specimen volume prepared for a set of tilted views.

A tilt series images the same specimen many times, so the work that does not
depend on the tilt is done once: the ice is blended in, the mean-free-path
field read off the dry specimen, and the volume padded for the coverage the
largest tilt needs and tapered at its edges. What does depend on the tilt is
re-rendered per view: the specimen's damage after the dose of the tilts
before it, and the ice's decorrelation under its own dose.
:meth:`TiltSeries.scatter` then propagates one tilt and returns its exit
wave, which the lens and camera stages image like any other.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import roma
import torch

from specter import logger

from .. import rotations
from .. import tilt as tilt_geometry
from ..aberrations._functions import defocus_midplane_shift
from ..fft import RadialSpectrum
from ..ice import IceBank, RandomIcemaker, blend_ice_into_volume, resolve_icemaker
from ..ice._exposure import solvent_exposure_envelope
from ..potential import aperture_lowpass_isotropic, dose_damage_envelope
from ..progress import status
from ..random_seed import derive_seed, seeded_rng
from ..scattering import IterativeScattering
from ..settings import Camera, Ice, Optics, Propagation, TiltGeometry
from . import _stages as stages
from ._data import Exposure, Poses, settings_group
from ._waves import ExitWave

__all__ = [
    "TiltCoverage",
    "TiltSeries",
    "fit_volume_to_tilt",
    "tilt_defocus_shift",
    "tilt_theta",
]


@dataclass(frozen=True)
class TiltCoverage:
    """
    The XY a tilt range needs, and what the volume it was given allows.

    Attributes
    ----------
    max_tilt_angle_deg : float
        The largest tilt of the series.
    recommended_nxy_for_max_tilt : int
        The exact geometric minimum XY for that tilt.
    edge_margin : int
        Extra pixels added on each side of that minimum.
    max_allowed_tilt_deg_for_volume : float
        The largest tilt the given volume covers without padding.
    max_allowed_nxy : int
        The largest image the given volume covers at the series' largest tilt.
    """

    max_tilt_angle_deg: float
    recommended_nxy_for_max_tilt: int
    edge_margin: int
    max_allowed_tilt_deg_for_volume: float
    max_allowed_nxy: int


def fit_volume_to_tilt(
    volume: torch.Tensor,
    desired_nxy: int,
    quaternions: torch.Tensor,
    edge_margin: int,
    taper_width: int,
    z_taper_width: int,
    pad_volume: bool,
    log: bool = True,
) -> tuple[torch.Tensor, TiltCoverage]:
    """
    Size a volume for a tilt range: reflect-pad its XY to the coverage the
    range needs when asked, then apply the edge tapers.

    The geometric minimum leaves no interpolation slack for the crop's edge
    pixels, so `edge_margin` is added before the taper, which is a separate,
    purely cosmetic apron. ``log=False`` repeats the padding silently, for a
    tilt's ice.

    Returns
    -------
    volume : torch.Tensor
        The padded, tapered volume.
    coverage : TiltCoverage
        What the range needs and what the input allowed.
    """
    max_tilt = tilt_geometry.infer_max_tilt_from_inputs(
        angles=None, quaternions=quaternions
    )
    nz_input = int(volume.shape[-3])
    available_nxy = int(min(volume.shape[-2], volume.shape[-1]))
    required_nxy = tilt_geometry.estimate_required_nxy(
        desired_nxy=desired_nxy, nz=nz_input, max_tilt_angle_deg=max_tilt
    )
    required_nxy_padded = required_nxy + 2 * int(edge_margin)
    target_nxy = required_nxy_padded + 2 * taper_width
    coverage = TiltCoverage(
        max_tilt_angle_deg=float(max_tilt),
        recommended_nxy_for_max_tilt=required_nxy,
        edge_margin=int(edge_margin),
        max_allowed_tilt_deg_for_volume=tilt_geometry.estimate_max_allowed_tilt_deg(
            desired_nxy=desired_nxy, nz=nz_input, available_nxy=available_nxy
        ),
        max_allowed_nxy=tilt_geometry.estimate_max_allowed_nxy(
            available_nxy=available_nxy, nz=nz_input, max_tilt_angle_deg=max_tilt
        ),
    )

    if available_nxy < target_nxy:
        if pad_volume:
            volume = tilt_geometry.pad_volume_xy_for_tilt(
                volume, target_nxy, available_nxy
            )
            if log:
                logger.info(
                    "Volume XY too small for requested tilt coverage"
                    + (" and taper" if taper_width > 0 else "")
                    + f"; padded (reflect) from {available_nxy} to "
                    f"{volume.shape[-1]} px in XY (micrograph_size={desired_nxy}, "
                    f"max tilt {coverage.max_tilt_angle_deg:.2f} deg, required "
                    f">={required_nxy} + edge_margin {edge_margin})."
                )
        elif log:
            logger.info(
                "Input volume XY may be too small for the requested tilt "
                f"coverage; proceeding anyway (pad_volume=False): volume "
                f"{tuple(volume.shape)}, required >={required_nxy}, max "
                f"allowed tilt ~{coverage.max_allowed_tilt_deg_for_volume:.2f} deg."
            )

    if taper_width > 0 or z_taper_width > 0:
        volume = tilt_geometry.apply_volume_cosine_taper(
            volume, taper_xy=int(taper_width), taper_z=int(z_taper_width)
        )
    return volume, coverage


def tilt_theta(
    quaternion: torch.Tensor, shift: torch.Tensor, nxy: int, voxel_size: float
) -> torch.Tensor:
    """
    The affine sampling matrix of one tilt, ``(1, 3, 4)``.

    Parameters
    ----------
    quaternion : torch.Tensor
        ``(1, 4)``, roma's convention.
    shift : torch.Tensor
        ``(1, 2)`` in-plane shift in Angstrom.
    nxy : int
        Lateral size of the volume being sampled.
    voxel_size : float
        In Angstrom.
    """
    R = roma.unitquat_to_rotmat(quaternion)
    if R.ndim == 2:
        R = R.unsqueeze(0)
    T = rotations.translations_angstrom_to_torch(shift, nxy, voxel_size)
    return rotations.build_affine_matrix(R, T)


def tilt_defocus_shift(
    volume_shape: tuple[int, ...],
    theta: torch.Tensor,
    voxel_size: float,
    scattering_model: str,
) -> float:
    """
    The defocus shift for a tilt's exit wave: minus the distance from its
    exit face to the plane its nominal defocus refers to.

    Multislice propagates the tilted column's depth rather than the
    volume's, so the wave leaves half the tilted depth below the specimen's
    centre rather than half the volume's; the shift is negative for the same
    reason as `defocus_midplane_shift`'s. Zero for the models with no depth.
    """
    if scattering_model in ("projection", "ctf"):
        return 0.0
    nz = volume_shape[-3]
    nz_tilt = tilt_geometry.nz_tilt_for_pose(volume_shape, theta)
    return defocus_midplane_shift(nz, voxel_size) - (nz_tilt - nz) * voxel_size / 2.0


class _TiltSolvent:
    """
    The ice of a tilt series, rendered for one tilt's exposure at a time.

    The solvent-exposure filter acts on the unweighted ice canvas, before it
    is weighted by the free fraction of each voxel and padded for the tilt
    range. The canvas is therefore kept as its 3D spectrum and the free
    fraction beside it, so the ice for a dose costs one inverse transform,
    one product and the padding.
    """

    def __init__(
        self,
        spectrum: RadialSpectrum,
        free: torch.Tensor,
        motion_variance: float,
        n_frames: int,
        weights: torch.Tensor | None,
        weights_max_frequency: float | None,
        fit: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.spectrum = spectrum
        self.free = free
        self.motion_variance = motion_variance
        self.n_frames = n_frames
        self.weights = weights
        self.weights_max_frequency = weights_max_frequency
        self.fit = fit

    @property
    def device(self) -> torch.device:
        """Where the spectrum and the free fraction live."""
        return self.spectrum.device

    def to(self, device: torch.device | str) -> _TiltSolvent:
        """Move the spectrum and the free fraction to `device`, in place."""
        free = self.free.to(device)
        self.spectrum.to(device)
        self.free = free
        return self

    def render(self, dose: float) -> torch.Tensor:
        """The weighted, padded ice for an exposure of `dose` e-/A^2."""
        envelope = solvent_exposure_envelope(
            dose,
            self.motion_variance,
            self.n_frames,
            self.weights,
            self.weights_max_frequency,
        )
        ice = torch.empty(
            self.spectrum.shape, device=self.device, dtype=self.spectrum.dtype
        )
        self.spectrum.filter_into(ice, envelope)
        ice.mul_(self.free[0])
        return self.fit(ice[None])


class TiltSeries:
    """
    One specimen volume prepared for a set of tilted views.

    Built once per series by :meth:`~specter.imaging.Sample.tilt_series`;
    :meth:`scatter` then propagates one tilt at a time. Holds state between
    tilts (the exposure the volume was last rendered for, where the volume
    lives), so unlike the other image-formation objects it is not frozen.

    Parameters
    ----------
    volume : torch.Tensor
        The dry specimen, ``(1, Z, Y, X)``, at its true Z extent: the
        defocus is measured from its midplane to its Z face.
    voxel_size : float
        In Angstrom.
    poses : Poses
        One pose per tilt, in acquisition order.
    exposure : Exposure
        One exposure per tilt (:meth:`Exposure.tilt_series`).
    voltage : float
        Accelerating voltage in kV.
    micrograph_size : int
        Size of each tilt image, in pixels.
    ice : Ice, optional
        Ice blended into the volume before any padding; ``thickness`` is
        ignored, since the volume's Z extent is the specimen's.
    icemaker : IceBank or RandomIcemaker, optional
        A prebuilt icemaker for that blend.
    damaged : bool, optional
        Damage the specimen, not its ice, per tilt: tilt ``i`` sees the dry
        specimen under the envelope of its own dose after the tilts before
        it, plus the ice. Default False.
    tilt : TiltGeometry, optional
        Tilt axis and the XY/Z cosine tapers.
    edge_margin : int, optional
        Pixels added on each side of the geometric coverage. Default 8.
    pad_volume : bool, optional
        Reflect-pad the volume's XY when it is too small for the coverage.
        Default True.
    fft_pad_margin : int, optional
        Zero-filled headroom of the multislice canvas under ``pad_fft``,
        which a tilt series takes by default. Zeros rather than reflection,
        since a tilted slice's flanks are vacuum rather than continuing ice,
        and reflecting would fabricate density there. A uniform slab at 60
        degrees images 16% off flat at the frame edge with no margin, 1.0%
        at 16 px and 0.3% at 32 px: the oversampled slice interpolation rings
        from the frame join in proportion to one over the margin. Default 32.
    absorption : torch.Tensor, optional
        An explicit nonnegative imaginary potential matching `volume`; it
        supplies all absorption, and the mean-free-path settings are not
        used to build one. Requires multislice with ``alpha=0``.
    seed : int, optional
        Run seed for the ice draw, ``(seed, 0, ice stage)``.
    objective_aperture : float, optional
        Aperture semi-angle in mrad, charged as absorption; requires
        ``absorption_model="inelastic_mfp"``.
    log : bool, optional
        Log the padding and the ice blend. Default True.
    propagation : Propagation, optional
        How the exit wave is computed, or its fields as keywords
        (``scattering_model``, ``alpha``, ``absorption_model``,
        ``oversample``, ``pad_fft``, ...).

    Notes
    -----
    See the class Notes of the generator this replaces for the measurements
    behind the padding, the margins and the absorption field, which is a
    constant over the volume's box rather than a scalar: a tilted slice
    extends past the rotated slab into vacuum, where nothing may absorb.

    With `damaged`, the per-tilt damage envelope is built from the critical
    exposure of Grant & Grigorieff [1]_.

    References
    ----------
    .. [1] Grant, T., & Grigorieff, N. (2015). Measuring the optimal exposure for
       single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6.
       *eLife*, 4, e06980. https://doi.org/10.7554/eLife.06980
    """

    def __init__(
        self,
        volume: torch.Tensor,
        voxel_size: float,
        poses: Poses,
        exposure: Exposure,
        *,
        voltage: float,
        micrograph_size: int,
        ice: Ice = Ice(),
        icemaker: IceBank | RandomIcemaker | None = None,
        damaged: bool = False,
        tilt: TiltGeometry = TiltGeometry(),
        edge_margin: int = 8,
        pad_volume: bool = True,
        fft_pad_margin: int = 32,
        absorption: torch.Tensor | None = None,
        seed: int | None = None,
        objective_aperture: float | None = None,
        log: bool = True,
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> None:
        p = settings_group(Propagation, propagation, settings)
        mfp = p.absorption_model == "inelastic_mfp"
        if objective_aperture is not None and not mfp:
            raise ValueError(
                "objective_aperture requires absorption_model='inelastic_mfp': "
                "under 'alpha' the fitted amplitude contrast already stands in "
                "for aperture loss"
            )
        if (mfp or absorption is not None) and (
            p.scattering_model != "multislice" or p.alpha != 0
        ):
            raise ValueError(
                "a tilt series' absorption field (absorption_model="
                "'inelastic_mfp' or an explicit absorption) requires "
                "multislice and alpha=0"
            )
        if absorption is not None and (
            absorption.shape != volume.shape
            or absorption.dtype != volume.dtype
            or absorption.device != volume.device
            or not absorption.is_floating_point()
            or not torch.isfinite(absorption).all()
            or (absorption < 0).any()
        ):
            raise ValueError(
                "absorption must be finite, nonnegative, real and match the "
                "volume's shape, dtype and device"
            )
        self.voxel_size = voxel_size
        self.voltage = voltage
        self.poses = poses
        self.exposure = exposure
        self.nxy = micrograph_size
        self.propagation = p
        self.damaged = damaged
        n_tilts = len(poses)
        self.doses = exposure.doses(n_tilts)
        self.pre_exposure = (
            exposure.pre_exposure
            if exposure.pre_exposure is not None
            else torch.zeros(n_tilts)
        )

        volume_icemaker = resolve_icemaker(
            ice.model,
            voxel_size,
            nxy=volume.shape[-1],
            nz=volume.shape[-3],
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=ice.parameterization,
        )
        has_ice = volume_icemaker is not None
        # Read off the DRY specimen, before ice is blended into it.
        removal = stages.resolve_removal_mfps(
            p, Optics(objective_aperture=objective_aperture), voltage
        )
        if absorption is None and removal is not None:
            if removal.inelastic_specimen is not None or has_ice:
                with torch.no_grad():
                    absorption = stages.mfp_absorption_field(
                        volume, voxel_size, removal, has_solvent=has_ice
                    )
        self.filters_solvent = has_ice and ice.motion_variance is not None
        per_tilt_solvent = self.filters_solvent and not bool(
            torch.all(self.doses == self.doses[0])
        )
        captured: list[RadialSpectrum] = []
        ice_filter: Callable[[torch.Tensor], None] | None = None
        if per_tilt_solvent:

            def capture(canvas: torch.Tensor) -> None:
                # Keep the unweighted ice as its spectrum and let the blend
                # weight a canvas of ones: what it adds is then the free
                # fraction of every voxel.
                captured.append(RadialSpectrum(canvas[0], voxel_size))
                canvas.fill_(1.0)

            ice_filter = capture
        elif self.filters_solvent:
            # Every tilt has the same dose, so one filtered ice serves all.
            ice_filter = stages.solvent_exposure_filter(
                ice,
                Camera(n_frames=exposure.n_frames),
                voxel_size,
                float(self.doses[0]),
                exposure.weights,
                exposure.weights_max_frequency,
            )
        dry = volume if damaged else None
        blended = volume
        if has_ice:
            seeded = (
                contextlib.nullcontext()
                if seed is None
                else seeded_rng(derive_seed(seed, 0, stages.SEED_STAGE_ICE))
            )
            if log:
                logger.info(f"Adding ice to volume using {ice.model} model")
            with (
                seeded,
                torch.no_grad(),
                status("Tiling ice volume", disable=not log),
            ):
                blended = blend_ice_into_volume(
                    volume,
                    cast(IceBank | RandomIcemaker, volume_icemaker),
                    voxel_size,
                    relax_steps=ice.relax_steps,
                    ice_filter=ice_filter,
                )
        free: torch.Tensor | None = None
        if per_tilt_solvent:
            # The specimen is padded without its ice, which each tilt adds.
            free = blended.sub_(volume)
            del blended
        else:
            volume = blended

        def fit(v: torch.Tensor, log: bool = log) -> torch.Tensor:
            fitted, coverage = fit_volume_to_tilt(
                v,
                micrograph_size,
                poses.quaternions,
                edge_margin,
                tilt.taper_width,
                tilt.z_taper_width,
                pad_volume,
                log=log,
            )
            self.coverage = coverage
            return fitted

        def lowpass(v: torch.Tensor) -> torch.Tensor:
            # Isotropic in 3D, so that it commutes with every tilt.
            if objective_aperture is None:
                return v
            with torch.no_grad():
                return aperture_lowpass_isotropic(
                    v, voxel_size, objective_aperture, voltage
                )

        volume = lowpass(fit(volume))
        if dry is not None:
            dry = lowpass(fit(dry, log=False))
        if absorption is not None:
            absorption = fit(absorption, log=False)
        self.nz = volume.shape[-3]

        self._solvent_series: _TiltSolvent | None = None
        if per_tilt_solvent:
            assert free is not None and ice.motion_variance is not None
            self._solvent_series = _TiltSolvent(
                captured[0],
                free,
                ice.motion_variance,
                exposure.n_frames or 1,
                exposure.weights,
                exposure.weights_max_frequency,
                lambda v: lowpass(fit(v, log=False)),
            )
            del free, captured

        # The damage series: the dry specimen kept as its spectrum, damaged
        # per tilt with one inverse transform, plus the ice it was blended
        # with, which was weighted by the undamaged specimen's occupancy.
        self._spectrum: RadialSpectrum | None = None
        self._solvent: torch.Tensor | None = None
        self._rendered: tuple[float, float] | None = None
        if dry is not None:
            if volume is dry or volume.data_ptr() == dry.data_ptr():
                # Rendered into for every tilt: never the caller's tensor.
                volume = volume.clone()
            with torch.no_grad():
                self._spectrum = RadialSpectrum(dry[0], voxel_size)
                if has_ice and not per_tilt_solvent:
                    self._solvent = volume - dry
            del dry
        self.volume = volume
        self.absorption = absorption
        # With a per-tilt solvent and no damage, the padded dry specimen is
        # kept beside the volume to add each tilt's ice to.
        self._solvent_dose: float | None = None
        self._tilt_dry: torch.Tensor | None = None
        if self._solvent_series is not None and self._spectrum is None:
            self._tilt_dry = self.volume
            self.volume = torch.empty_like(self._tilt_dry)
            self._render_solvent(float(self.doses[0]))

        self.scattering = IterativeScattering(
            micrograph_size,
            voxel_size,
            voltage,
            scattering_model=p.scattering_model,
            klim=p.klim,
            oversample=p.oversample,
            crop=False,
            alpha=p.alpha,
            progressbars=False,
            roi_padding_mode="zeros",
            pad_fft=p.pads(True),
            fft_pad_margin=fft_pad_margin,
            # The margin carries on through the transfer function; the image
            # is cropped to the field at the detector.
            keep_pad=True,
        )
        self._warned_on_host = False

    def __len__(self) -> int:
        return len(self.poses)

    # ------------------------------------------------------------------ #
    # Per-tilt rendering                                                   #
    # ------------------------------------------------------------------ #

    def _render_solvent(self, dose: float) -> None:
        """Put the ice for an exposure of `dose` into the volume, once per dose."""
        if self._solvent_dose == dose:
            return
        assert self._solvent_series is not None
        with torch.no_grad():
            solvent = self._solvent_series.render(dose).to(self.volume.device)
            if self._tilt_dry is None:
                self._solvent = solvent
                self._rendered = None
            else:
                torch.add(self._tilt_dry, solvent, out=self.volume)
            del solvent
        self._solvent_dose = dose

    def _render_exposure(self, dose: float, pre_exposure: float) -> None:
        """Write the specimen damaged for one exposure into the volume, once."""
        key = (dose, pre_exposure)
        if self._rendered == key:
            return
        spectrum = cast(RadialSpectrum, self._spectrum)
        envelope = dose_damage_envelope(
            dose,
            pre_exposure,
            weighted=self.exposure.weighted,
            voltage=self.voltage,
            n_frames=self.exposure.n_frames,
            frame_weights=self.exposure.weights,
            frame_weights_max_frequency=self.exposure.weights_max_frequency,
        )
        with torch.no_grad():
            try:
                spectrum.filter_into(self.volume[0], envelope)
            except torch.cuda.OutOfMemoryError:
                self._to_host()
                spectrum.filter_into(self.volume[0], envelope)
            if self._solvent is not None:
                self.volume.add_(self._solvent)
        self._rendered = key

    # ------------------------------------------------------------------ #
    # Placement                                                            #
    # ------------------------------------------------------------------ #

    def _paired(self) -> list[str]:
        names = ["volume"]
        for n in ("absorption", "_solvent", "_tilt_dry"):
            if getattr(self, n) is not None:
                names.append(n)
        return names

    def place(self, device: torch.device | str) -> None:
        """
        Move the volume and every field paired with it onto `device`, or
        keep them all on the host, streamed per slice, if they do not fit.
        """
        device = torch.device(device)
        names = self._paired()
        spectrum = self._spectrum
        settled = all(getattr(self, n).device == device for n in names) and (
            spectrum is None or spectrum.device == device
        )
        if not settled:
            try:
                moved = [getattr(self, n).to(device) for n in names]
                moved_spectrum = (
                    None if spectrum is None else spectrum.spectrum.to(device)
                )
            except torch.cuda.OutOfMemoryError:
                self._to_host()
            else:
                for n, t in zip(names, moved, strict=True):
                    setattr(self, n, t)
                if spectrum is not None and moved_spectrum is not None:
                    spectrum.spectrum = moved_spectrum
                    spectrum._work = None
        series = self._solvent_series
        if series is not None and series.device != self.volume.device:
            try:
                series.to(self.volume.device)
            except torch.cuda.OutOfMemoryError:
                series.to("cpu")
                torch.cuda.empty_cache()
        # The propagator runs on the compute device whether or not the volume
        # fitted there: off the device it fetches geometry-bounded blocks
        # per z-chunk, so device memory stays bounded by the query geometry.
        self.scattering.to(device)

    def _to_host(self) -> None:
        for n in self._paired():
            setattr(self, n, getattr(self, n).cpu())
        if self._spectrum is not None:
            self._spectrum.to("cpu")
        if self._solvent_series is not None:
            self._solvent_series.to("cpu")
        torch.cuda.empty_cache()
        if not self._warned_on_host:
            warnings.warn(
                "the specimen volume and the fields paired with it do not fit "
                "on the device together; streaming them from the host.",
                stacklevel=3,
            )
            self._warned_on_host = True

    # ------------------------------------------------------------------ #
    # Scattering                                                           #
    # ------------------------------------------------------------------ #

    def scatter(
        self,
        i: int,
        *,
        series: int = 0,
        slice_batchsize: int = 1,
        device: torch.device | str | None = None,
    ) -> ExitWave:
        """
        Propagate tilt `i` and return its exit wave.

        Parameters
        ----------
        i : int
            The tilt, in acquisition order.
        series : int, optional
            Which series of the run this is; the noise is drawn from
            ``(seed, series, detector stage, i)``. Default 0.
        slice_batchsize : int, optional
            Slices propagated together. Default 1.
        device : torch.device or str, optional
            Where to compute; the volume is placed there if it fits.
            Default the volume's device.

        Returns
        -------
        ExitWave
            With the defocus referenced to the tilted column's midplane.
        """
        device = torch.device(device) if device is not None else self.volume.device
        self.place(device)
        if self._solvent_series is not None:
            self._render_solvent(float(self.doses[i]))
        if self._spectrum is not None:
            self._render_exposure(float(self.doses[i]), float(self.pre_exposure[i]))
        volume = self.volume

        theta = tilt_theta(
            self.poses.quaternions[i].unsqueeze(0).to(device),
            self.poses.shifts[i].unsqueeze(0).to(device),
            self.volume.shape[-1],
            self.voxel_size,
        )
        if self.absorption is None:
            wave = self.scattering(volume, theta, slice_batchsize=slice_batchsize)
        else:
            wave = self.scattering.multislice_absorptive(
                volume, self.absorption, theta, slice_batchsize=slice_batchsize
            )
        p = self.propagation
        linear = p.scattering_model == "ctf"
        shift = tilt_defocus_shift(
            tuple(volume.shape), theta, self.voxel_size, p.scattering_model
        )
        return ExitWave(
            array=wave,
            pixel_size=self.voxel_size,
            voltage=self.voltage,
            nxy=self.nxy,
            pad_nxy=self.scattering.output_nxy,
            oversample=1 if linear else p.oversample,
            defocus_shift=shift,
            ids=torch.tensor([series]),
            tilt=i,
            linear=linear,
            alpha=p.alpha,
            damaged=self.damaged,
            solvent_filtered=self.filters_solvent,
        )
