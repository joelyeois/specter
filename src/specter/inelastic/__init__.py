"""Himes-inspired, exposure-resolved zero-loss simulation.

Measured spectra and source coefficients are explicit inputs. The optional
Drude spectrum and Brownian motion are labeled development approximations.
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
