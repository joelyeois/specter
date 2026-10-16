"""
The forward model's settings, grouped by the physics they describe.

A generator is configured with a few small frozen dataclasses rather than
one keyword per knob, so that each group is declared, documented and
validated once and passed around as a unit. The groups follow the stages of
image formation, and which of them a class accepts says what it models:

- `Propagation`: how the exit wave is computed from the potential. The one
  group the simulator and the reconstruction share in full, since its
  conventions (which defocus origin, whether amplitude contrast is applied)
  must agree on both sides.
- `Optics`: the aberration stage, i.e. the engine that turns per-image CTF
  parameters into a transfer function, and the phase plate. ``None`` skips
  the stage and yields the bare exit wave.
- `Envelopes`: the partial-coherence and radiation-damage envelopes. An
  experimental dataset gives no way to determine these, so the
  reconstruction does not accept them: the high-frequency loss they cause
  is absorbed into the reconstructed volume.
- `Camera`: the detector chain, likewise forward-only.
- `TiltGeometry`: the tilt axis and edge tapers a tilt series is imaged
  with, shared by `imaging.TiltSeries` and `TomogramReconstructor`.
- `Ice`: the amorphous ice a specimen is embedded in, forward-only.
- `Crowding` and `Packing`: how duplicates of a particle template are
  placed around it, and which collision test decides where they fit.

Per-image data (poses, CTF parameters, dose, coincidence radius, potential
scale) stays as direct tensor arguments, and the two scalars that define
the sampling of the experiment, ``pixel_size`` and ``voltage``, are direct
arguments too, since every stage consumes them.

These are the Python-API surface. The TOML configs and CLI flags stay flat
(one field per setting); a pipeline builds the groups from a config with
:func:`bundle_from_config`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from specter.options import (
    AbsorptionModel,
    DetectorModel,
    DoseEnvelopeTarget,
    EwaldSphereSign,
    IceModel,
    NoiseModel,
    PoseMode,
    ScatteringFactors,
    ScatteringModel,
    TemporalCoherence,
    TiltAxis,
)

AberrationBackend = Literal["legacy", "torch_ctf"]

__all__ = [
    "AberrationBackend",
    "Camera",
    "Crowding",
    "Envelopes",
    "Ice",
    "Optics",
    "Packing",
    "Propagation",
    "TiltGeometry",
    "bundle_from_config",
]


#: Chromatic aberration coefficient of the Titan Krios objective lens, 2.7 mm,
#: in Angstrom: the forward simulators' default (`Envelopes.cc`), so a
#: simulated image carries the temporal-coherence loss every real one does.
DEFAULT_CC_ANGSTROM = 2.7e7


@dataclass(frozen=True)
class Propagation:
    """
    How the exit wave is computed from the scattering potential.

    Parameters
    ----------
    scattering_model : ScatteringModel
        Wave-propagation model. Default ``"multislice"``, the most accurate.
    alpha : float
        Amplitude contrast ratio, dimensionless in ``[0, 1]``. Applied to the
        potential before propagation for the wave models, and inside the
        transfer function for ``scattering_model="ctf"``, whose exit wave is
        a real projection. Default 0.0. Ignored, and required to be zero, when
        ``absorption_model="inelastic_mfp"``.
    absorption_model : AbsorptionModel
        Where the imaginary part of the potential comes from. ``"alpha"``
        (default) scales the real potential by `alpha`, tying absorption to
        every atomic cusp. ``"inelastic_mfp"`` derives it per material from a
        measured mean free path instead -- see
        :func:`~specter.potential.inelastic_absorption_potential`, which also
        records why a fitted `alpha` is the wrong quantity for this job.
        Supported by every forward simulation (particles, micrographs and
        tilt series). TomogramReconstructor rejects this model until
        its iterative path carries explicit material information.
    inelastic_mfp_solvent : float or None
        Inelastic mean free path of the embedding medium, in Angstrom. Used
        only when ``absorption_model="inelastic_mfp"``, and only where the
        medium is actually present: with no icemaker there is vacuum around
        the specimen, not ice, so the solvent term is dropped. Default None
        takes the value for amorphous ice at the imager's voltage from
        :func:`~specter.potential.ice_inelastic_mfp`: measured at 300 kV
        (3950 A) and 120 kV (2030 A), and elsewhere estimated from those two
        with a warning (200 kV: 3040 A +/- 7 %; 100 kV: 1730 A +/- 20 %).
    inelastic_mfp_specimen : float or None
        Inelastic mean free path of the specimen material, in Angstrom.
        Default None gives the specimen the solvent's value, so it absorbs
        exactly like the medium it displaces and carries bulk attenuation but
        no absorption *contrast*. Pass
        :data:`~specter.potential.INELASTIC_MFP_PROTEIN_A` for protein at
        300 kV, noting the uncertainty and the voltage caveat documented
        there.
    ews_curvature_sign : EwaldSphereSign
        Sign of the Ewald-sphere curvature. Which sign describes a given
        dataset is a property of the instrument's image geometry rather than of
        a software convention, so a reconstruction should try both; a
        simulation and the reconstruction of its output must use the same one.
        Default ``"negative"``.
    klim : float, optional
        Kirkland bandlimit as a fraction of Nyquist (``0.66`` prevents FFT
        aliasing at the cost of the highest frequencies). Default None, no
        bandlimit.
    oversample : int
        Grid refinement for the multislice recursion and the image: each
        slice is interpolated onto an xy grid this many times finer (adding
        no content), so the product of transmission function and wave has
        room to exist instead of wrapping at the volume's Nyquist; the
        transfer function is applied and the intensity formed on that grid
        too, since ``|psi|^2`` has twice the wave's bandwidth, and the
        detector samples the intensity to the pixel grid. The Fourier-space counterpart of
        `pad_fft`: that pads real space (a wider frame, same pixels, finer
        k-sampling, same Nyquist) against wrap at the frame edge; this pads
        Fourier space (same frame, finer pixels, same k-sampling, higher
        Nyquist) against wrap at Nyquist. Neither does the other's job, and
        the aliasing below was measured with `pad_fft` on. At 1 the wrapped content costs
        0.2-1.1% of the scattered power (thin to 2000 A ice, 6bdf at
        1 A/px) and 1.9% at Nyquist; at 2 the error is at the float32 noise
        floor and no bandlimit is needed, for about 3x the multislice time.
        Rytov and projection alias the same way through their one final
        exponential, and take it on the fine grid too, at the cost of one
        fine-grid exponential per image; first Born and the CTF model are
        linear and untouched. Default 2.
    pad_fft : bool or None
        Give the propagation canvas FFT headroom so the multislice
        recursion's circular convolution cannot wrap at the frame edge: the
        real-space counterpart of `oversample`, see there. The FFT treats the
        frame as periodic, which is harmless while its opposite edges look
        alike and wrong when they do not. None (the default) therefore takes
        the imaging path's own default, :meth:`pads`: off for a particle box
        and for a micrograph of level, uniform ice, on for a tilt series
        (whose two edges cross a tilted slab at different depths) and for a
        micrograph whose ice profile varies across the field.
    rotate_mode : PoseMode
        How a pose is applied: ``"real"`` or ``"fourier"`` rotate the volume
        into the beam frame with trilinear interpolation, in real or Fourier
        space; ``"surface"`` leaves the volume unrotated and samples its
        transform on the posed Ewald surface
        (`specter.scattering.surface_slice_sum`), with no interpolation. The
        last is available for the single-scatter models (``"projection"``,
        ``"firstborn"``, ``"rytov"``, ``"ctf"``) and, in `Specimen.scatter`,
        only for an undressed template: crowding, ice, damage and per-image
        atom rendering all need the rotated volume. Default ``"real"``.
    """

    scattering_model: ScatteringModel = "multislice"
    alpha: float = 0.0
    absorption_model: AbsorptionModel = "alpha"
    inelastic_mfp_solvent: float | None = None
    inelastic_mfp_specimen: float | None = None
    ews_curvature_sign: EwaldSphereSign = "negative"
    klim: float | None = None
    oversample: int = 2
    pad_fft: bool | None = None
    rotate_mode: PoseMode = "real"

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha={self.alpha} must be in [0, 1]")
        if self.absorption_model == "inelastic_mfp":
            # The two are alternative routes to the same imaginary potential,
            # so allowing both would absorb twice.
            if self.alpha != 0.0:
                raise ValueError(
                    f"absorption_model='inelastic_mfp' with alpha={self.alpha}: "
                    "the two are alternative routes to the imaginary potential "
                    "and would double-count. Set alpha=0.0 (note that a "
                    ".cs/.star file's amplitude contrast overrides the config's "
                    "alpha, so it has to be zeroed at the source)."
                )
            # scattering_model="ctf" has a real projection for an exit wave and
            # applies amplitude contrast at the lens instead (see
            # `ExitWave.apply_ctf`), so a complex potential has nowhere to go.
            if self.scattering_model == "ctf":
                raise ValueError(
                    "absorption_model='inelastic_mfp' is not available with "
                    "scattering_model='ctf', whose exit wave is a real "
                    "projection and which applies amplitude contrast inside "
                    "the transfer function. Use a wave model, or alpha."
                )
            for name in ("inelastic_mfp_solvent", "inelastic_mfp_specimen"):
                value = getattr(self, name)
                if value is not None and value <= 0.0:
                    raise ValueError(f"{name}={value} must be positive")
        if self.klim is not None and not 0.0 < self.klim <= 1.0:
            raise ValueError(
                f"klim={self.klim} must be in (0, 1] (a fraction of Nyquist)"
            )
        if not isinstance(self.oversample, int) or self.oversample < 1:
            raise ValueError(f"oversample={self.oversample!r} must be an integer >= 1")
        if self.rotate_mode == "surface":
            if self.scattering_model not in ("projection", "firstborn", "rytov", "ctf"):
                raise ValueError(
                    f"rotate_mode='surface' with scattering_model="
                    f"{self.scattering_model!r}: only the single-scatter models "
                    "(projection, firstborn, rytov, ctf) are one sample of the "
                    "volume's transform; multislice and kinematic need the "
                    "rotated volume."
                )
            if self.absorption_model != "alpha":
                raise ValueError(
                    "rotate_mode='surface' needs absorption_model='alpha': the "
                    "mean-free-path absorption field is read off the rotated volume."
                )

    def pads(self, default: bool) -> bool:
        """`pad_fft`, or the imaging path's `default` when it is None."""
        return default if self.pad_fft is None else self.pad_fft


@dataclass(frozen=True)
class Optics:
    """
    The aberration stage: the CTF engine and the phase plate.

    Per-image aberration terms (defocus, astigmatism, Cs, beam tilt, trefoil,
    tetrafoil, B-factor) are data and travel in the ``ctf_params`` dict; this
    group holds only what is shared by every image.

    Parameters
    ----------
    aberration_backend : {"legacy", "torch_ctf"}
        Which engine computes the transfer function. ``"legacy"`` (default)
        is :class:`specter.aberrations.Aberration`; ``"torch_ctf"`` is
        :class:`specter.ctf.LegacyAberrationAdapter`, verified term by term
        against it and the only backend with a phase-plate model.
    lpp_params : dict, optional
        Laser-phase-plate settings in :class:`specter.ctf.CTFParameters`
        native units. Requires ``aberration_backend="torch_ctf"``. Default
        None, no phase plate.
    objective_aperture : float, optional
        Objective aperture semi-angle in milliradians (a 70 um aperture on a
        Krios is ~12 mrad). Electrons scattered elastically beyond it leave
        the image; the loss is applied as an absorption rate per material
        from the scattering cross section
        (:func:`~specter.potential.aperture_mfp_ice`), since no practical grid
        carries that scattering itself, and the potential is low-passed at
        the aperture so the grid's own share is not counted twice. Requires
        ``Propagation(absorption_model="inelastic_mfp")``: the ``alpha``
        model's fitted constant already stands in for this loss. Default
        None, no aperture.
    """

    aberration_backend: AberrationBackend = "legacy"
    lpp_params: dict[str, float] | None = None
    objective_aperture: float | None = None

    def __post_init__(self) -> None:
        if self.objective_aperture is not None and self.objective_aperture <= 0.0:
            raise ValueError(
                f"objective_aperture={self.objective_aperture} must be positive "
                "(milliradians), or None for no aperture"
            )
        if self.lpp_params is not None and self.aberration_backend != "torch_ctf":
            raise ValueError(
                "lpp_params requires aberration_backend='torch_ctf' -- "
                "aberrations.Aberration has no laser-phase-plate model."
            )


@dataclass(frozen=True)
class Envelopes:
    """
    Partial-coherence and radiation-damage envelopes on the transfer function.

    Every envelope is off by default, and the reconstruction does not accept
    this group at all: none of these can be determined from a dataset, so
    assuming values for them would inject prior information the data does
    not contain.

    Parameters
    ----------
    convergence_angle : float, optional
        Beam convergence semi-angle in mrad, for the spatial-coherence (Cs)
        envelope. Default None, envelope off.
    cc : float, optional
        Chromatic aberration coefficient in Angstrom, for the temporal
        coherence loss from the beam's energy spread. Default
        :data:`DEFAULT_CC_ANGSTROM` (2.7 mm, Titan Krios),
        on, since every real image carries it: with it off, a 300 kV image at
        1 A/px has 15% too much power in the shell nearest Nyquist (8.7x at
        120 kV). An energy filter does not remove it, since the source's
        0.3-0.7 eV spread passes any slit. None or 0 disables it. The inverse
        (`Reconstructor`) takes no `Envelopes` and runs without it.
    energy_spread : float
        FWHM of the beam energy spread in eV, as a source is quoted, used
        by the Cc envelope (which converts it to a standard deviation).
        Default 0.7.
    deltaV_V : float
        Relative high-voltage instability, rms, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I : float
        Relative objective-lens current instability, rms, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope : bool
        Apply the Grant & Grigorieff (2015) radiation-damage envelope over
        each image's exposure. Default False.
    dose_envelope_target : DoseEnvelopeTarget
        Where the envelope acts. ``"transfer_function"`` (default) filters the
        whole image, solvent included. ``"specimen"`` damages the specimen's
        own 3D potential before the ice is added, so the solvent keeps its
        structure: its water ring does not fade under exposure, and it loses
        coherence between frames instead (``Ice.motion_variance``). Supported
        by every forward simulation; a micrograph or tilt series damages its specimen
        before, or apart from, the ice blended into it.
    temporal_coherence : TemporalCoherence
        How the focal spread `cc` causes is applied, when `cc` is set.
        ``"integrated"`` (default) forms the image as the incoherent sum of
        ``|psi|^2`` over a Gauss-Hermite quadrature of the defocus spread,
        which is exact; ``"envelope"`` multiplies the transfer function by
        the Cc envelope, exact only for the part of the image linear in the
        scattered wave. Against the exact sum the envelope is 0.4% low in
        mean intensity and 0.7% low in low-frequency power in 2000 A of ice
        at 300 kV (0.7 eV, Cc 2.7 mm), and 5% low in mean at 120 kV; the
        integration costs one inverse FFT per node (16 at 300 kV, 1 A/px).
        The linear model (``scattering_model="ctf"``) is linear in the
        transfer function and keeps the envelope either way.
    """

    convergence_angle: float | None = None
    cc: float | None = DEFAULT_CC_ANGSTROM
    energy_spread: float = 0.7
    deltaV_V: float = 0.06e-6
    deltaI_I: float = 0.01e-6
    dose_envelope: bool = False
    dose_envelope_target: DoseEnvelopeTarget = "transfer_function"
    temporal_coherence: TemporalCoherence = "integrated"

    def __post_init__(self) -> None:
        if self.temporal_coherence not in ("integrated", "envelope"):
            raise ValueError(
                f"temporal_coherence={self.temporal_coherence!r} must be "
                "'integrated' or 'envelope'"
            )
        if self.dose_envelope_target not in ("transfer_function", "specimen"):
            raise ValueError(
                f"dose_envelope_target={self.dose_envelope_target!r} must be "
                "'transfer_function' or 'specimen'"
            )


@dataclass(frozen=True)
class Camera:
    """
    The detector chain: MTF and DQE(0), shot noise, and frame fractionation.

    Parameters
    ----------
    detector_model : DetectorModel, optional
        Bundled detector whose MTF and DQE(0) are applied. Default None, an
        ideal counter with no blur.
    noise_model : NoiseModel, optional
        ``"poisson"`` (default) draws shot noise from the expected counts;
        None (or ``"none"``) returns the expected image.
    dose_weights_path : str or None, optional
        Path to a ``(n_frames, n_bins)`` ``.npy`` of the exposure filter's own
        per-frame, per-frequency weights -- CryoSPARC's
        ``refm_empirical_dw.npy``, say. The frames are then summed in Fourier
        space under those weights instead of equally, which leaves the signal
        untouched and multiplies the noise power by ``sum(w^2)/n_frames``.
        That rising floor is what a signal-preserving dose weighting leaves in
        real data and what `Envelopes.dose_envelope` cannot express, since it
        attenuates the signal and leaves the noise white. Requires
        ``n_frames``. Default None.
    dose_weights_max_frequency : float or None, optional
        Frequency, in 1/Angstrom, of the last bin of `dose_weights_path`.
        Default None derives it from the job's own files via
        :func:`~specter.io.load_dose_weights`, which is the reliable route:
        the pixel size alone does not determine it (CryoSPARC's weights array
        carries twice the radial sampling of the FCC beside it, so its axis
        runs to twice Nyquist at the same pixel size).
    n_frames : int, optional
        Movie frames the dose is fractionated into, which sets how
        coincidence loss saturates. Default None, a single frame.
    detector_pixel_size : float or None, optional
        Pixel size, in Angstrom, at which the movies were recorded, when the
        image was resampled afterwards (a particle stack Fourier-cropped or
        binned from its micrographs). The detector MTF belongs to the physical
        pixel, so it is evaluated there and read off at the image's
        frequencies; see :func:`~specter.detectors.detector_mtf`. Default None,
        the image pixel is the physical pixel.
    """

    detector_model: DetectorModel | None = None
    noise_model: NoiseModel | None = "poisson"
    n_frames: int | None = None
    dose_weights_path: str | None = None
    dose_weights_max_frequency: float | None = None
    detector_pixel_size: float | None = None

    def __post_init__(self) -> None:
        if self.detector_pixel_size is not None and self.detector_pixel_size <= 0:
            raise ValueError(
                f"detector_pixel_size={self.detector_pixel_size} must be positive"
            )
        if self.detector_model == "none":
            object.__setattr__(self, "detector_model", None)
        if self.noise_model == "none":
            object.__setattr__(self, "noise_model", None)
        if self.n_frames is not None and self.n_frames < 1:
            raise ValueError(f"n_frames={self.n_frames} must be at least 1")


@dataclass(frozen=True)
class TiltGeometry:
    """
    The tilt-series conventions the forward and inverse tilt models share.

    Parameters
    ----------
    tilt_axis : TiltAxis
        Axis the specimen tilts about, ``"x"`` or ``"y"``. Decides which
        image dimension shrinks at high tilt. Default ``"x"``.
    taper_width : int
        Cosine taper in pixels at the XY edges of the volume. Default 0.
    z_taper_width : int
        Cosine taper in pixels at the top and bottom of the volume, so a
        hard Z edge does not become an in-plane discontinuity once tilted.
        Default 0.
    """

    tilt_axis: TiltAxis = "x"
    taper_width: int = 0
    z_taper_width: int = 0

    def __post_init__(self) -> None:
        axis = str(self.tilt_axis).lower()
        if axis not in ("x", "y"):
            raise ValueError(f"tilt_axis={self.tilt_axis!r} must be 'x' or 'y'")
        object.__setattr__(self, "tilt_axis", axis)
        if self.taper_width < 0 or self.z_taper_width < 0:
            raise ValueError("taper_width and z_taper_width must be non-negative")


@dataclass(frozen=True)
class Ice:
    """
    The amorphous-ice model a specimen is embedded in.

    Parameters
    ----------
    model : IceModel, optional
        ``"gd"`` draws from the bundled `IceBank` library of optimised ice,
        ``"random"`` places molecules at random (instant, no structure).
        Default None (or ``"none"``): no ice.
    thickness : float, optional
        Ice thickness in Angstrom. For a particle stack this sizes the
        column the particle sits in (None: the particle box's own depth);
        for a micrograph it sizes the specimen slab. A tilt series takes
        its thickness from the volume it is given and ignores this.
    profile : IceProfile, optional
        Laterally varying thickness (wedge, meniscus, tilted slab) for a
        micrograph, replacing the uniform ``thickness``. Default None.
    cache_dir : str, optional
        `IceBank` library directory for ``model="gd"``. Default None, the
        bundled ``ice_data/ice_cache``.
    relax_steps : int
        Local MLBOP relaxation steps healing the seams between tiled
        `IceBank` crops. Default 0.
    parameterization : ScatteringFactors
        Atomic scattering factors for the water kernel. Default
        ``"kirkland"``; ice is a bulk material, outside the Shtyrov fits'
        domain (see ``bulk_scattering_factors`` in the configs).
    motion_variance : float, optional
        McMullan et al.'s sigma0^2: per-axis beam-induced displacement of the
        water in A^2 per e-/A^2, as measured from the decorrelation of the
        3.7 A ring (0.38 for their 300 kV exposure). When set, the ice's
        fluctuation is filtered to what survives the summed exposure, frame
        weights included (:func:`~specter.ice.apply_solvent_exposure`, relaxed
        coherence model); its mean is kept. Requires the dose envelope, if
        on, to act on the specimen, or the solvent would lose its structure
        twice. A tilt series filters its one ice volume at the dose of one
        tilt, which must then be the same on every tilt. Default None:
        frozen ice.
    """

    model: IceModel | None = None
    thickness: float | None = None
    profile: Any = None
    cache_dir: str | None = None
    relax_steps: int = 0
    parameterization: ScatteringFactors = "kirkland"
    motion_variance: float | None = None

    def __post_init__(self) -> None:
        if self.model == "none":
            object.__setattr__(self, "model", None)
        if self.motion_variance is not None and self.motion_variance < 0:
            raise ValueError(
                f"motion_variance={self.motion_variance} must be non-negative"
            )
        if self.thickness is not None and self.thickness < 0:
            raise ValueError(f"thickness={self.thickness} must be non-negative")
        if self.relax_steps < 0:
            raise ValueError(f"relax_steps={self.relax_steps} must be non-negative")


@dataclass(frozen=True)
class Crowding:
    """
    How duplicates of a particle template are packed around it.

    Parameters
    ----------
    min_distance : float, optional
        Minimum centre-to-centre distance between duplicates, in Angstrom.
        Default None: no crowding at all.
    max_distance_z : float, optional
        Depth of the slab duplicates are placed in, in Angstrom. Default
        None: the template's own depth for a particle stack (a contrast knob
        like ice thickness must not deepen the neighbour slab), the whole
        box for a micrograph specimen.
    max_distance_xy : float, optional
        Lateral extent duplicates are placed over, in Angstrom. Default
        None: the box.
    method : {"2d", "3d"}
        Poisson-disk sampling dimensionality. Default ``"3d"``.
    n_points : int, optional
        Cap on the number of duplicates. Default None, fill the volume.
    seed : {"origin", "random"}
        Where the Poisson-disk sampling starts. Default ``"origin"``.
    chunk_size : int
        Duplicates rotated per batch. Default 1, both the cheapest and the
        fastest setting.
    water_air_interface : bool
        Bias the duplicates' z distribution toward the ice surfaces, as
        particles adsorb there during vitrification. Default False.
    sigma_frac, peak_amplitude, baseline : float
        The shape of that bias, see
        :func:`specter.crowding.filter_by_z_density`.
    """

    min_distance: float | None = None
    max_distance_z: float | None = None
    max_distance_xy: float | None = None
    method: Literal["2d", "3d"] = "3d"
    n_points: int | None = None
    seed: Literal["origin", "random"] = "origin"
    chunk_size: int = 1
    water_air_interface: bool = False
    sigma_frac: float = 0.05
    peak_amplitude: float = 1.0
    baseline: float = 0.1

    def __post_init__(self) -> None:
        if self.min_distance is not None and self.min_distance <= 0:
            raise ValueError(f"min_distance={self.min_distance} must be positive")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size={self.chunk_size} must be at least 1")


@dataclass(frozen=True)
class Packing:
    """
    The collision test a `Crowding` placement uses.

    Parameters
    ----------
    backend : {"poisson_disk", "shape"}
        ``"poisson_disk"`` (default) rejects on bounding spheres;
        ``"shape"`` collides the template's real rotated footprint and packs
        several times denser, see :func:`specter.specimen.packing.pack_shapes_3d`.
    gap : float
        Extra clearance between footprints in Angstrom (shape backend).
        Default 0.
    n_orientations : int
        Rotations cached per template (shape backend). Default 256.
    max_retries : int
        Rejected attempts before the packing stops; sets the density reached
        (shape backend). Default 1500.
    stall_patience : int
        Consecutive rejections tolerated before giving up (shape backend).
        Default 5000.
    seed : int, optional
        RNG seed for the packing (shape backend). In a seeded run it is mixed
        into each image's own packing seed. Default None.
    n_candidates : int, optional
        Candidate positions drawn (shape backend). Default None, estimated.
    """

    backend: Literal["poisson_disk", "shape"] = "poisson_disk"
    gap: float = 0.0
    n_orientations: int = 256
    max_retries: int = 1500
    stall_patience: int = 5000
    seed: int | None = None
    n_candidates: int | None = None


_B = TypeVar("_B")


def bundle_from_config(
    bundle_cls: type[_B], config: Any, *, prefix: str = "", **overrides: Any
) -> _B:
    """
    Build a settings group from a flat config's fields of the same names.

    Parameters
    ----------
    bundle_cls : type
        One of the dataclasses in this module.
    config : object
        A config dataclass. Fields the config does not have keep the group's
        own default.
    prefix : str, optional
        Prepended to each field name to find it on the config: `Ice`'s
        ``thickness`` is a config's ``ice_thickness``. Default "".
    **overrides
        Values that take precedence over the config's, for a field whose
        config spelling differs (a unit conversion, a ``"none"`` sentinel, or
        a value read from a data file).

    Returns
    -------
    bundle_cls
    """
    values = {
        f.name: getattr(config, prefix + f.name)
        for f in dataclasses.fields(bundle_cls)  # type: ignore[arg-type]
        if hasattr(config, prefix + f.name)
    }
    values.update(overrides)
    return bundle_cls(**values)
