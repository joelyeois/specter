"""
Default values that several layers spell as their own default.

A value here is the default of a parameter that a library function, a
settings group and a config field each accept, so the layers cannot drift
apart. This module imports only `typing`, so the config layer can use it without
importing torch.
"""

from __future__ import annotations

from typing import Final

#: FWHM of the beam energy spread, eV, for the Cc envelope.
ENERGY_SPREAD_EV: Final = 0.7

#: Relative high-voltage instability, for the Cc envelope.
DELTA_V_V: Final = 0.06e-6

#: Relative objective-lens current instability, for the Cc envelope.
DELTA_I_I: Final = 0.01e-6

#: Chromatic aberration coefficient of the Titan Krios objective, mm.
CC_MM: Final = 2.7

#: Learning-rate scheduler of a reconstruction.
SCHEDULER: Final = "LambdaLR"

#: Decay rate of the ``"LambdaLR"`` schedule.
LR_DECAY: Final = 0.1

#: Zero-filled margin on each side of a padded multislice canvas, pixels.
FFT_PAD_MARGIN: Final = 32
