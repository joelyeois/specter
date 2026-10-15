"""
Small helpers for the ice generators: the number density of amorphous ice
and a few array utilities.
"""

from __future__ import annotations


import torch
from scipy import constants as _sc


avogadro = _sc.Avogadro
# The one ice density in the package. RandomIcemaker and GradientSKIcemaker
# fill their boxes to it, so the bundled IceBank library is at it too (527,178
# molecules per 256 A cube), and `potential.aperture_mfp_ice` reads it for its
# number density. The LDA-80K MD frame the S(k) target is computed from is
# slightly less dense, 0.932 g/cm3 (64,000 molecules in a 127.12 A cell).
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / Å³]


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
