"""Low-level aberration phase functions.

Pure functions of frequency-grid tensors and physical parameters — no
dependence on a class instance. Each function returns the phase contribution
(in radians) of one aberration term to the total wavefront aberration
function chi(k), following the conventions of Kirkland [1]_ and
Penczek [2]_.

Notes
-----
.. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
   US, Boston, MA, 2010).
.. [2] P. A. Penczek, "Image Restoration in Cryo-Electron Microscopy" in
   Methods in Enzymology (Academic Press Inc., 2010) vol. 482, pp. 35-72.
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
    KY: torch.Tensor,
    KX: torch.Tensor,
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
    KY : torch.Tensor
        Frequency grid component along the first (row/y) axis, in 1/Angstrom.
    KX : torch.Tensor
        Frequency grid component along the second (column/x) axis, in
        1/Angstrom.
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
    tilts = torch.sin(tilty) * KY + torch.sin(tiltx) * KX
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
    k: torch.Tensor,
    radian: torch.Tensor,
    tetrafoil1: torch.Tensor,
    tetrafoil2: torch.Tensor,
    tetrafoil3: torch.Tensor,
    tetrafoil4: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate tetrafoil phase contribution.

    Covers the full 4th-order, non-rotationally-symmetric aberration
    subspace: secondary astigmatism (Zernike n=4, m=+-2, the
    ``tetrafoil1``/``tetrafoil2`` terms) and true 4-fold tetrafoil
    (n=4, m=+-4, ``tetrafoil3``/``tetrafoil4``). Spherical aberration
    (n=4, m=0) is handled separately by :func:`cs`.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Angstrom.
    radian : torch.Tensor
        Frequency angle grid, in radians.
    tetrafoil1 : torch.Tensor
        Coefficient of the k^4 cos(2*radian) (secondary astigmatism) term.
    tetrafoil2 : torch.Tensor
        Coefficient of the k^4 sin(2*radian) (secondary astigmatism) term.
    tetrafoil3 : torch.Tensor
        Coefficient of the k^4 cos(4*radian) (tetrafoil) term.
    tetrafoil4 : torch.Tensor
        Coefficient of the k^4 sin(4*radian) (tetrafoil) term.

    Returns
    -------
    chi_tetrafoil : torch.Tensor
        Phase contribution from tetrafoil aberration.
    """
    k4 = k**4
    return (
        tetrafoil1 * k4 * torch.cos(2 * radian)
        + tetrafoil2 * k4 * torch.sin(2 * radian)
        + tetrafoil3 * k4 * torch.cos(4 * radian)
        + tetrafoil4 * k4 * torch.sin(4 * radian)
    )


def phaseshift(
    phaseshift: torch.Tensor,
    k: torch.Tensor,
    n_pixels: int,
    aberration_model: str,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculate phase shift contribution (e.g., from Volta phase plate),
    plus the amplitude-contrast phase offset for the CTF model.

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
    alpha : torch.Tensor, optional
        Amplitude contrast ratio. Only meaningful for the 'ctf' model,
        where the exit wave is real-valued and has no complex/absorptive
        component of its own -- amplitude contrast has to be represented
        as this k-independent chi offset instead (``-acos(alpha)``,
        matching CryoSPARC's convention: ``phase_shift - arccos(amp_contrast)``).
        None (default) omits the term entirely, matching the 'holography'
        model (where amplitude contrast is instead baked into the exit
        wave upstream via ``scattering.complex_potential``, so adding it
        again here would double-count it).

    Returns
    -------
    chi_phaseshift : torch.Tensor
        Phase shift contribution. For holography model, DC component is
        set to zero to maintain Fourier optics validity.
    """
    if aberration_model == "ctf" and alpha is not None:
        phaseshift = phaseshift - torch.acos(alpha)
    if aberration_model == "holography":
        phaseshift = phaseshift * torch.ones_like(k)
        # phaseshift must be zero at DC for Fourier optics
        phaseshift[:, 0, 0] = 0
    return -phaseshift


def defocus_midplane_shift(nz: int, pixel_size: float) -> float:
    """
    Å shift between a volume's entry face and its midplane.

    Multislice evaluates the exit wave's phase at the centre of the
    volume's Z extent, but defocus values follow the CryoSPARC/RELION
    convention of being measured from the specimen's entry face. Subtract
    this shift from ``dfu``/``dfv`` before building a transfer function for
    a multislice-propagated exit wave; add it back to recover the original
    entry-face convention (e.g. before exporting to a STAR/`.cs` file).
    Not needed for the ``'projection'``/``'ctf'`` scattering models, which
    have no Z extent to offset from.

    Parameters
    ----------
    nz : int
        Number of Z slices in the (possibly padded) simulation volume.
    pixel_size : float
        Voxel size in Å.

    Returns
    -------
    float
        Shift in Å.
    """
    return (nz * pixel_size) / 2
