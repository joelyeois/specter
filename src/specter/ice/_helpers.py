"""
Small helpers for the ice generators: the number density of amorphous ice
and a few array utilities.
"""

from __future__ import annotations

import torch

# The water constants live in `specter.constants`, which `potential` imports
# without importing `ice`; re-exported here for the ice generators.
from ..constants import (  # noqa: F401
    avogadro,
    density_of_amorphous_ice,
    molar_mass_of_water,
    ndensity_of_amorphous_ice,
)


def rfftn(array: torch.Tensor) -> torch.Tensor:
    """
    Compute N-dimensional real-input Fourier transform with centering.

    Wraps torch.fft.rfftn with FFT shifting to ensure the zero-frequency component
    is centered, handling the last dimension which is complex-valued differently.

    Parameters
    ----------
    array : torch.Tensor
        Input real-valued tensor.

    Returns
    -------
    fft : torch.Tensor
        Complex-valued tensor containing the Fourier coefficients.
        Zero frequency is centered.
    """
    return torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2),
    )
