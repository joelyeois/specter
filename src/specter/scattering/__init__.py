"""
Wave propagation through a scattering potential.

`Scattering` takes a batch of already-rotated potentials and computes each
exit wave whole; `IterativeScattering` samples one volume slice by slice
under a pose, which is what a micrograph, a tilt series and the tomogram
reconstruction need. Both follow Kirkland's multislice formalism [1]_ and
offer the same models (``multislice``, ``rytov``, ``firstborn``,
``kinematic``, ``projection``, ``ctf``), built from the kernels in
``_kernels``. The multislice method is that of Cowley & Moodie [2]_.
`surface_slice_sum` forms the single-scatter models' slice sum from the
UNROTATED volume instead, by sampling its transform on the posed Ewald
surface, with no interpolation.

References
----------
.. [1] Kirkland, E. J. (2010). *Advanced computing in electron microscopy*
   (2nd ed.). Springer. https://doi.org/10.1007/978-1-4419-6533-2
.. [2] Cowley, J. M., & Moodie, A. F. (1957). The scattering of electrons by
   atoms and crystals. I. A new theoretical approach. *Acta
   Crystallographica*, 10(10), 609–619.
   https://doi.org/10.1107/S0365110X57002194
"""

from ..constants import energy_to_wavelength, interaction_parameter
from ._batch import Scattering
from ._iterative import IterativeScattering
from ._surface import SURFACE_MODELS, surface_slice_sum
from ._kernels import (
    absorption_factor,
    bandlimit_mask,
    frequency_grid,
    fresnel_propagator,
    phase_scale,
)

__all__ = [
    "SURFACE_MODELS",
    "IterativeScattering",
    "Scattering",
    "absorption_factor",
    "bandlimit_mask",
    "energy_to_wavelength",
    "frequency_grid",
    "fresnel_propagator",
    "interaction_parameter",
    "phase_scale",
    "surface_slice_sum",
]
