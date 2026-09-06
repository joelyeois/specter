"""
Wave propagation through a scattering potential.

`Scattering` takes a batch of already-rotated potentials and computes each
exit wave whole; `IterativeScattering` samples one volume slice by slice
under a pose, which is what a micrograph, a tilt series and the tomogram
reconstruction need. Both follow Kirkland's multislice formalism and offer
the same models (``multislice``, ``rytov``, ``firstborn``, ``kinematic``,
``projection``, ``ctf``), built from the kernels in ``_kernels``.
"""

from ..constants import energy_to_wavelength, interaction_parameter
from ._batch import Scattering
from ._iterative import IterativeScattering
from ._kernels import (
    absorption_factor,
    bandlimit_mask,
    frequency_grid,
    fresnel_propagator,
    phase_scale,
)

__all__ = [
    "IterativeScattering",
    "Scattering",
    "absorption_factor",
    "bandlimit_mask",
    "energy_to_wavelength",
    "frequency_grid",
    "fresnel_propagator",
    "interaction_parameter",
    "phase_scale",
]
