"""Low-level aberration phase functions.

Pure functions of frequency-grid tensors and physical parameters — no
dependence on a class instance. Each function returns the phase contribution
(in radians) of one aberration term to the total wavefront aberration
function chi(k).

Notes
-----
.. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
   US, Boston, MA, 2010).
.. [2] P. A. Penczek, "Image Restoration in Cryo-Electron Microscopy" in
   Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35-72.
"""

from __future__ import annotations

import torch


def cs(k: torch.Tensor, wavelength: float, cs: torch.Tensor) -> torch.Tensor:
    """
    Calculate spherical aberration phase contribution.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Angstrom.
    wavelength : float
        Electron wavelength, in Angstrom.
    cs : torch.Tensor
        Spherical aberration coefficient, in Angstrom.

    Returns
    -------
    chi_cs : torch.Tensor
        Phase contribution from spherical aberration.
    """
    return torch.pi / 2 * wavelength**3 * k**4 * cs


def defocus(
    k2: torch.Tensor,
    radian: torch.Tensor,
    wavelength: float,
    dfu: torch.Tensor,
    dfv: torch.Tensor,
    dfang: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate defocus phase contribution including astigmatism.

    Parameters
    ----------
    k2 : torch.Tensor
        Squared frequency magnitude grid, in 1/Angstrom^2.
    radian : torch.Tensor
        Frequency angle grid, in radians.
    wavelength : float
        Electron wavelength, in Angstrom.
    dfu : torch.Tensor
        Defocus along first axis, in Angstrom.
    dfv : torch.Tensor
        Defocus along second axis, in Angstrom.
    dfang : torch.Tensor
        Astigmatism angle, in degrees.

    Returns
    -------
    chi_defocus : torch.Tensor
        Phase contribution from defocus and astigmatism.
    """
    dfang = dfang / 180 * torch.pi
    df = 0.5 * (dfu + dfv + (dfv - dfu) * torch.cos(2 * (radian + dfang)))
    return -torch.pi * wavelength * k2 * df


def beamtilt(
    k2: torch.Tensor,
    kxx: torch.Tensor,
    kyy: torch.Tensor,
    wavelength: float,
    cs: torch.Tensor,
    tiltx: torch.Tensor,
    tilty: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate beam tilt phase contribution.

    Parameters
    ----------
    k2 : torch.Tensor
        Squared frequency magnitude grid, in 1/Angstrom^2.
    kxx : torch.Tensor
        Frequency grid component along the first axis, in 1/Angstrom.
    kyy : torch.Tensor
        Frequency grid component along the second axis, in 1/Angstrom.
    wavelength : float
        Electron wavelength, in Angstrom.
    cs : torch.Tensor
        Spherical aberration coefficient, in Angstrom.
    tiltx : torch.Tensor
        Beam tilt in x direction, in radians.
    tilty : torch.Tensor
        Beam tilt in y direction, in radians.

    Returns
    -------
    chi_tilt : torch.Tensor
        Phase contribution from beam tilt.
    """
    tilts = torch.sin(tilty) * kxx + torch.sin(tiltx) * kyy
    return -2 * torch.pi * wavelength**2 * cs * k2 * tilts


def trefoil(
    k: torch.Tensor,
    radian: torch.Tensor,
    trefoil1: torch.Tensor,
    trefoil2: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate trefoil (3-fold astigmatism) phase contribution.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Angstrom.
    radian : torch.Tensor
        Frequency angle grid, in radians.
    trefoil1 : torch.Tensor
        First trefoil component.
    trefoil2 : torch.Tensor
        Second trefoil component.

    Returns
    -------
    chi_trefoil : torch.Tensor
        Phase contribution from trefoil aberration.
    """
    return trefoil1 * k**3 * torch.sin(3 * radian) + trefoil2 * k**3 * torch.cos(
        3 * radian
    )


def tetrafoil(
    tetrafoil1: torch.Tensor,
    tetrafoil2: torch.Tensor,
    tetrafoil3: torch.Tensor,
    tetrafoil4: torch.Tensor,
):
    """
    Calculate tetrafoil (4-fold astigmatism) phase contribution.

    Parameters
    ----------
    tetrafoil1 : torch.Tensor
        First tetrafoil component.
    tetrafoil2 : torch.Tensor
        Second tetrafoil component.
    tetrafoil3 : torch.Tensor
        Third tetrafoil component.
    tetrafoil4 : torch.Tensor
        Fourth tetrafoil component.

    Returns
    -------
    chi_tetrafoil : torch.Tensor or None
        Phase contribution from tetrafoil aberration. Currently not implemented.
    """
    pass


def phaseshift(
    phaseshift: torch.Tensor, k: torch.Tensor, n_pixels: int, aberration_model: str
) -> torch.Tensor:
    """
    Calculate phase shift contribution (e.g., from Volta phase plate).

    Parameters
    ----------
    phaseshift : torch.Tensor
        Phase shift value in radians.
    k : torch.Tensor
        Frequency magnitude grid, in 1/Angstrom. Used only for its shape.
    n_pixels : int
        Number of pixels along each axis of the grid.
    aberration_model : str
        Aberration model in use; 'holography' or 'ctf'.

    Returns
    -------
    chi_phaseshift : torch.Tensor
        Phase shift contribution. For holography model, DC component is
        set to zero to maintain Fourier optics validity.
    """
    if aberration_model == "holography":
        phaseshift = phaseshift * torch.ones_like(k)
        # phaseshift must be zero at DC for Fourier optics
        phaseshift[:, 0, 0] = 0
    return -phaseshift
