"""
The enumerated option vocabularies, each spelled once.

Every string-valued switch that a class, function or config field accepts
is one of these ``Literal`` aliases, so the class that implements a choice,
the config dataclass that exposes it and the CLI ``Choice`` built from that
dataclass all agree on the spelling by construction. A config field may use
a narrower ``Literal`` than the class it feeds when a command deliberately
exposes a subset (``ParticleStackConfig.scattering_model`` omits the
reconstruction-only ``rytov``); it must never use a wider one.
"""

from __future__ import annotations

from typing import Literal

#: Wave-propagation model, see :class:`specter.scattering.Scattering`.
ScatteringModel = Literal[
    "multislice", "rytov", "firstborn", "kinematic", "projection", "ctf"
]

#: Sign of the Ewald-sphere curvature, matching CryoSPARC's convention.
EwaldSphereSign = Literal["negative", "positive"]

#: Detector noise model. ``"none"`` (or ``None``) leaves the image noiseless.
NoiseModel = Literal["poisson", "none"]

#: Bundled detector MTF/DQE models, see :mod:`specter.detectors`.
DetectorModel = Literal[
    "none",
    "perfect",
    "k3_300kv",
    "k3_200kv",
    "k2_300kv",
    "falcon4i_300kv",
    "falcon4i_200kv",
]

#: Amorphous-ice generator: the bundled ``IceBank`` library, a
#: ``RandomIcemaker``, or no ice.
IceModel = Literal["gd", "random", "none"]

#: Optimiser for ``GradientSKIcemaker.optimize``.
IceOptimizer = Literal["lbfgs", "adam"]

#: Atomic scattering-factor parameterization for structures with bond
#: topology (``shtyrov``) or without (``kirkland``, ``lobato``).
ScatteringFactors = Literal["kirkland", "lobato", "shtyrov"]

#: What :func:`specter.potential.build_atomic_potential_kernel` can sample:
#: the three above plus the per-element Peng ``c4322`` factors Shtyrov
#: typing falls back to.
KernelParameterization = Literal["kirkland", "lobato", "shtyrov", "peng"]

#: How ``PotentialBuilder.forward`` rasterizes atoms.
PotentialMethod = Literal["analytic", "2d", "3d"]

#: Convolution backend for kernel-based potentials; ``"auto"`` picks by size.
ConvBackend = Literal["fftconvolve", "conv3d", "auto"]

#: Boundary handling of a kernel convolution: zero-padded or wrapped.
ConvBoundary = Literal["linear", "periodic"]

#: Output size of :func:`specter.fft.fftconvolve`, as in ``scipy.signal``.
ConvMode = Literal["full", "same", "valid"]

#: ``torch.nn.functional.pad`` modes accepted for XY padding.
PadMode = Literal["constant", "reflect", "replicate", "circular"]

#: Aberration image-formation model: ``|CTF(psi)|^2`` or the linearised CTF.
AberrationModel = Literal["nonlinear", "linear"]

#: Whether a volume is rotated in real or Fourier space.
RotateMode = Literal["real", "fourier"]

#: Where the rotation centre of an even-sized grid sits.
GridOrigin = Literal["relion", "center"]

#: Coordinate-grid convention for even-sized k-space and real-space grids.
GridConvention = Literal["relion", "torch"]

#: ``torch.nn.functional.grid_sample`` padding modes.
GridSamplePadding = Literal["zeros", "border", "reflection"]

#: Organic membrane shape generator, see :mod:`specter.specimen.membrane`.
ShapeBackend = Literal["spherical_harmonics", "swept_spline"]

#: Which edge of the frame a carbon-film hole is placed against.
CarbonEdgeSide = Literal["random", "left", "right", "top", "bottom"]

#: Axis a tilt series is tilted about.
TiltAxis = Literal["x", "y"]

#: Learning-rate scheduler for a reconstruction, by ``torch.optim`` name.
Scheduler = Literal[
    "LambdaLR", "OneCycleLR", "CosineAnnealingWarmRestarts", "MultiplicativeLR"
]

__all__ = [
    "AberrationModel",
    "CarbonEdgeSide",
    "ConvBackend",
    "ConvBoundary",
    "ConvMode",
    "DetectorModel",
    "EwaldSphereSign",
    "GridConvention",
    "GridOrigin",
    "GridSamplePadding",
    "IceModel",
    "IceOptimizer",
    "KernelParameterization",
    "NoiseModel",
    "PadMode",
    "PotentialMethod",
    "RotateMode",
    "ScatteringFactors",
    "ScatteringModel",
    "Scheduler",
    "ShapeBackend",
    "TiltAxis",
]
