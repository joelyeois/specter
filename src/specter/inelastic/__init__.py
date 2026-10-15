"""Himes-inspired, exposure-resolved zero-loss simulation.

Measured spectra and source coefficients are explicit inputs. The optional
Drude spectrum and Brownian motion are labeled development approximations.

References
----------
.. [1] Himes, B., & Grigorieff, N. (2021). Cryo-TEM simulations of amorphous
   radiation-sensitive samples using multislice wave propagation. *IUCrJ*,
   8(6), 943–953. https://doi.org/10.1107/S2052252521008538
"""

from ._damage import PotentialDoseDamage
from ._exposure import BrownianCoordinates, ExposureStep, exposure_steps
from ._spectrum import PlasmonFilter
from ._specimen import AtomicPlasmonSpecimen, PotentialState
from ._forward import FrozenPlasmonForward, FrozenPlasmonResult

__all__ = [
    "PotentialDoseDamage",
    "AtomicPlasmonSpecimen",
    "BrownianCoordinates",
    "ExposureStep",
    "exposure_steps",
    "PlasmonFilter",
    "PotentialState",
    "FrozenPlasmonForward",
    "FrozenPlasmonResult",
]
