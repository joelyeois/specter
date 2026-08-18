"""Fourier-space envelope functions.

Pure functions of frequency-grid tensors and physical parameters — no
dependence on a class instance. Each function returns a real-valued,
multiplicative amplitude attenuation (an "envelope") in the range (0, 1]
that can be multiplied into a transfer function. Math ported from
teamtomo's torch_fourier_filter.envelopes, adapted to specter's existing
unit conventions (wavelength/cs/defocus/cc in Å, angles in mrad).

Notes
-----
.. [1] T. Grant and N. Grigorieff, "Measuring the optimal exposure for
   single particle cryo-EM using a 2.6 A reconstruction of rotavirus
   VP6," eLife 4, e06980 (2015).
"""

from __future__ import annotations

import torch


def b_envelope(k2: torch.Tensor, bfactor: torch.Tensor) -> torch.Tensor:
    """
    Calculate an isotropic B-factor envelope.

    Parameters
    ----------
    k2 : torch.Tensor
        Squared frequency magnitude grid, in 1/Å².
    bfactor : torch.Tensor
        B-factor (temperature factor), in Å².

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope, exp(-B * k^2 / 4).
    """
    return torch.exp(-bfactor * k2 / 4)


def cs_envelope(
    k: torch.Tensor,
    wavelength: float,
    cs: torch.Tensor,
    defocus: torch.Tensor,
    convergence_angle: float,
) -> torch.Tensor:
    """
    Calculate a spatial-coherence envelope from finite beam convergence.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Å.
    wavelength : float
        Electron wavelength, in Å.
    cs : torch.Tensor
        Spherical aberration coefficient, in Å.
    defocus : torch.Tensor
        Defocus, in Å.
    convergence_angle : float
        Beam convergence semi-angle, in milliradians.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope due to partial spatial coherence.
    """
    half_angle = convergence_angle / 1000  # mrad -> rad
    return torch.exp(
        -((torch.pi * half_angle / wavelength) ** 2)
        * (cs * wavelength**3 * k**3 + wavelength * defocus * k) ** 2
    )


def cc_envelope(
    k2: torch.Tensor,
    wavelength: float,
    cc: float,
    voltage: float,
    energy_spread: float,
    deltaV_V: float,
    deltaI_I: float,
) -> torch.Tensor:
    """
    Calculate a temporal-coherence envelope from chromatic aberration.

    Parameters
    ----------
    k2 : torch.Tensor
        Squared frequency magnitude grid, in 1/Å².
    wavelength : float
        Electron wavelength, in Å.
    cc : float
        Chromatic aberration coefficient, in Å.
    voltage : float
        Accelerating voltage, in Volts.
    energy_spread : float
        FWHM of the beam energy spread, in eV.
    deltaV_V : float
        Relative high-voltage instability.
    deltaI_I : float
        Relative objective-lens current instability.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope due to partial temporal coherence.
    """
    focus_spread = cc * (
        ((energy_spread / voltage) ** 2 + deltaV_V**2 + (2 * deltaI_I) ** 2) ** 0.5
    )
    return torch.exp(-0.5 * (torch.pi * wavelength * focus_spread * k2) ** 2)


def dose_envelope(
    k: torch.Tensor,
    dose: torch.Tensor,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
) -> torch.Tensor:
    """
    Calculate a Grant & Grigorieff (2015) cumulative-dose envelope.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Å.
    dose : torch.Tensor
        Cumulative electron dose (fluence), in e⁻/Å².
    a : float, optional
        Fitted parameter from Grant & Grigorieff (2015). Default 0.245.
    b : float, optional
        Fitted parameter from Grant & Grigorieff (2015). Default -1.665.
    c : float, optional
        Fitted parameter from Grant & Grigorieff (2015). Default 2.81.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope; exactly 1 where ``dose < c``.
    """
    fluence_env = torch.exp(-(dose - c) / (a * torch.pow(k, b)))
    return torch.where(dose < c, torch.ones_like(fluence_env), fluence_env)
