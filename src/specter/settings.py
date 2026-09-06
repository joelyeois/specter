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
    DetectorModel,
    EwaldSphereSign,
    NoiseModel,
    RotateMode,
    ScatteringModel,
)

AberrationBackend = Literal["legacy", "torch_ctf"]

__all__ = [
    "AberrationBackend",
    "Camera",
    "Envelopes",
    "Optics",
    "Propagation",
    "bundle_from_config",
]


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
        a real projection. Default 0.0.
    ews_curvature_sign : EwaldSphereSign
        Sign of the Ewald-sphere curvature, matching CryoSPARC's convention.
        Default ``"negative"``.
    klim : float, optional
        Kirkland bandlimit as a fraction of Nyquist (``0.66`` prevents FFT
        aliasing at the cost of the highest frequencies). Default None, no
        bandlimit.
    pad_fft : bool
        Give the propagation canvas FFT headroom so the multislice
        recursion's circular convolution cannot wrap at the frame edge.
        Default False.
    rotate_mode : RotateMode
        How a volume is rotated into the beam frame: trilinear in real space
        or in Fourier space. Default ``"real"``.
    """

    scattering_model: ScatteringModel = "multislice"
    alpha: float = 0.0
    ews_curvature_sign: EwaldSphereSign = "negative"
    klim: float | None = None
    pad_fft: bool = False
    rotate_mode: RotateMode = "real"

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha={self.alpha} must be in [0, 1]")
        if self.klim is not None and not 0.0 < self.klim <= 1.0:
            raise ValueError(
                f"klim={self.klim} must be in (0, 1] (a fraction of Nyquist)"
            )


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
    """

    aberration_backend: AberrationBackend = "legacy"
    lpp_params: dict[str, float] | None = None

    def __post_init__(self) -> None:
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
        Chromatic aberration coefficient in Angstrom, for the temporal-
        coherence envelope. Default None, envelope off.
    energy_spread : float
        FWHM of the beam energy spread in eV, used by the Cc envelope.
        Default 0.7.
    deltaV_V : float
        Relative high-voltage instability, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I : float
        Relative objective-lens current instability, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope : bool
        Apply the Grant & Grigorieff (2015) radiation-damage envelope over
        each image's exposure. Default False.
    """

    convergence_angle: float | None = None
    cc: float | None = None
    energy_spread: float = 0.7
    deltaV_V: float = 0.06e-6
    deltaI_I: float = 0.01e-6
    dose_envelope: bool = False


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
    n_frames : int, optional
        Movie frames the dose is fractionated into, which sets how
        coincidence loss saturates. Default None, a single frame.
    """

    detector_model: DetectorModel | None = None
    noise_model: NoiseModel | None = "poisson"
    n_frames: int | None = None

    def __post_init__(self) -> None:
        if self.detector_model == "none":
            object.__setattr__(self, "detector_model", None)
        if self.noise_model == "none":
            object.__setattr__(self, "noise_model", None)
        if self.n_frames is not None and self.n_frames < 1:
            raise ValueError(f"n_frames={self.n_frames} must be at least 1")


_B = TypeVar("_B")


def bundle_from_config(bundle_cls: type[_B], config: Any, **overrides: Any) -> _B:
    """
    Build a settings group from a flat config's fields of the same names.

    Parameters
    ----------
    bundle_cls : type
        One of the dataclasses in this module.
    config : object
        A config dataclass. Fields the config does not have keep the group's
        own default.
    **overrides
        Values that take precedence over the config's, for a field whose
        config spelling differs (a unit conversion, a ``"none"`` sentinel, or
        a value read from a data file).

    Returns
    -------
    bundle_cls
    """
    values = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(bundle_cls)  # type: ignore[arg-type]
        if hasattr(config, f.name)
    }
    values.update(overrides)
    return bundle_cls(**values)
