from __future__ import annotations


import torch
from scipy import constants as _sc


def rest_mass_energy() -> float:
    """
    Rest mass energy of the electron.

    Returns
    -------
    float
        Electron rest mass energy in eV, derived from the CODATA electron
        mass energy equivalent (~511.0e3 eV).
    """
    return _sc.physical_constants["electron mass energy equivalent in MeV"][0] * 1.0e6


def hc() -> float:
    """
    Planck constant times the speed of light.

    Returns
    -------
    float
        hc in eV·Å, derived from CODATA h, c, e (~12.398e3 eV·Å).
    """
    return _sc.h * _sc.c / _sc.e * 1.0e10


def energy_to_wavelength(voltage: float) -> float:
    """
    Convert electron beam accelerating voltage to de Broglie wavelength.

    Uses the relativistic formula for electron wavelength calculation.

    Parameters
    ----------
    voltage : float
        Electron beam accelerating voltage in kV.

    Returns
    -------
    wavelength : float
        De Broglie wavelength in Å.

    Notes
    -----
    Uses the relativistic formula:
    λ = hc / sqrt(E * (E + 2*m_e*c²))
    where E is the electron's kinetic energy (numerically equal to the
    accelerating voltage in eV) and m_e*c² = 511 keV (rest mass energy of
    electron).
    """
    ev = voltage * 1.0e3  # [eV]
    return hc() / (ev * (ev + 2.0 * rest_mass_energy())) ** 0.5


def interaction_parameter(voltage: float) -> float:
    """
    Calculate the electron-specimen interaction parameter.

    Computes the interaction constant σ for electron scattering, following
    Kirkland Eq. (5.6) [1]_.

    Parameters
    ----------
    voltage : float
        Electron beam accelerating voltage in kV.

    Returns
    -------
    sigma : float
        Interaction parameter in units of 1/(Å·V).

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Eq. (5.6), Springer US, Boston, MA, 2010.
    """
    w = energy_to_wavelength(voltage)
    ev = voltage * 1e3
    m = rest_mass_energy()
    return 2.0 * torch.pi / (w * ev) * ((ev + m) / (ev + 2.0 * m))
