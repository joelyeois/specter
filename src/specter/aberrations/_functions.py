"""Low-level aberration phase functions.

Pure functions of frequency-grid tensors and physical parameters — no
dependence on a class instance. Each function returns the phase contribution
(in radians) of one aberration term to the total wavefront aberration
function chi(k), following the conventions of Kirkland [1]_ and
Penczek [2]_.

References
----------
.. [1] Kirkland, E. J. (2010). *Advanced computing in electron microscopy*
   (2nd ed.). Springer. https://doi.org/10.1007/978-1-4419-6533-2
.. [2] Penczek, P. A. (2010). Image restoration in cryo-electron microscopy.
   *Methods in Enzymology*, 482, 35–72.
   https://doi.org/10.1016/S0076-6879(10)82002-6
"""

from __future__ import annotations

import torch
from specter.options import AberrationModel, ScatteringModel


def cs(k: torch.Tensor, wavelength: float, cs: torch.Tensor) -> torch.Tensor:
    """
    Calculate spherical aberration phase contribution.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Å.
    wavelength : float
        Electron wavelength, in Å.
    cs : torch.Tensor
        Spherical aberration coefficient, in Å.

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
        Squared frequency magnitude grid, in 1/Å².
    radian : torch.Tensor
        Frequency angle grid, in radians.
    wavelength : float
        Electron wavelength, in Å.
    dfu : torch.Tensor
        Defocus along first axis, in Å.
    dfv : torch.Tensor
        Defocus along second axis, in Å.
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
        Squared frequency magnitude grid, in 1/Å².
    KY : torch.Tensor
        Frequency grid component along the first (row/y) axis, in 1/Å.
    KX : torch.Tensor
        Frequency grid component along the second (column/x) axis, in
        1/Å.
    wavelength : float
        Electron wavelength, in Å.
    cs : torch.Tensor
        Spherical aberration coefficient, in Å.
    tiltx : torch.Tensor
        Beam tilt in x direction, in radians.
    tilty : torch.Tensor
        Beam tilt in y direction, in radians.

    Returns
    -------
    chi_tilt : torch.Tensor
        Phase contribution from beam tilt.

    References
    ----------
    .. [1] Zemlin, F., Weiss, K., Schiske, P., Kunath, W., & Herrmann, K.-H.
       (1978). Coma-free alignment of high resolution electron microscopes with
       the aid of optical diffractograms. *Ultramicroscopy*, 3, 49–60.
       https://doi.org/10.1016/S0304-3991(78)80006-0
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
        Frequency magnitude grid, in 1/Å.
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

    References
    ----------
    .. [1] Zivanov, J., Nakane, T., & Scheres, S. H. W. (2020). Estimation of
       high-order aberrations and anisotropic magnification from cryo-EM data
       sets in *RELION*-3.1. *IUCrJ*, 7(2), 253–267.
       https://doi.org/10.1107/S2052252520000081
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
        Frequency magnitude grid, in 1/Å.
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

    References
    ----------
    .. [1] Zivanov, J., Nakane, T., & Scheres, S. H. W. (2020). Estimation of
       high-order aberrations and anisotropic magnification from cryo-EM data
       sets in *RELION*-3.1. *IUCrJ*, 7(2), 253–267.
       https://doi.org/10.1107/S2052252520000081
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
    aberration_model: AberrationModel,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculate phase shift contribution (e.g., from Volta phase plate),
    plus the amplitude-contrast phase offset for the linear model.

    Parameters
    ----------
    phaseshift : torch.Tensor
        Phase shift value in radians.
    k : torch.Tensor
        Frequency magnitude grid, in 1/Å. Used only for its shape.
    n_pixels : int
        Number of pixels along each axis of the grid.
    aberration_model : str
        Aberration model in use; 'nonlinear' or 'linear'.
    alpha : torch.Tensor, optional
        Amplitude contrast ratio. Only meaningful for the 'linear' model,
        where the exit wave is real-valued and has no complex/absorptive
        component of its own -- amplitude contrast has to be represented
        as this k-independent chi offset instead (``-acos(alpha)``,
        matching CryoSPARC's convention: ``phase_shift - arccos(amp_contrast)``).
        None (default) omits the term entirely, matching the 'nonlinear'
        model (where amplitude contrast is instead baked into the exit
        wave upstream via ``potential.apply_amplitude_contrast``, so adding it
        again here would double-count it).

    Returns
    -------
    chi_phaseshift : torch.Tensor
        Phase shift contribution. For the nonlinear model, DC component is
        set to zero to maintain Fourier optics validity.

    References
    ----------
    .. [1] Danev, R., Buijsse, B., Khoshouei, M., Plitzko, J. M., & Baumeister,
       W. (2014). Volta potential phase plate for in-focus phase contrast
       transmission electron microscopy. *Proceedings of the National Academy
       of Sciences*, 111(44), 15635–15640.
       https://doi.org/10.1073/pnas.1418377111
    .. [2] Erickson, H. P., & Klug, A. (1971). Measurement and compensation of
       defocusing and aberrations by Fourier processing of electron
       micrographs. *Philosophical Transactions of the Royal Society of
       London. B, Biological Sciences*, 261(837), 105–118.
       https://doi.org/10.1098/rstb.1971.0040
    """
    if aberration_model == "linear" and alpha is not None:
        phaseshift = phaseshift - torch.acos(alpha)
    if aberration_model == "nonlinear":
        phaseshift = phaseshift * torch.ones_like(k)
        # phaseshift must be zero at DC for Fourier optics
        phaseshift[:, 0, 0] = 0
    return -phaseshift


def defocus_midplane_shift(nz: int, pixel_size: float) -> float:
    """
    Å between a volume's exit face and its midplane.

    A propagating scattering model returns the wave where it leaves the
    volume, at the exit face, while a defocus value refers to the specimen's
    scattering centroid: the plane whose defocus a CTF fit of the image
    reports, which for a particle centred in its box, or a slab of uniform
    ice, is the box's midplane. Subtracting this shift from ``dfu``/``dfv``
    before building a transfer function for the exit wave makes the nominal
    defocus the midplane's. It is the defocus a CryoSPARC/RELION file
    records, so nothing is added back on export. Not needed for the
    ``'projection'``/``'ctf'`` scattering models, which have no Z extent.
    Pinned by
    ``tests/test_aberrations_functions.py::test_nominal_defocus_is_the_midplane_defocus``.

    The shift is *negative*, minus half the depth. An underfocused lens
    images a plane upstream of the specimen, so a scatterer further
    downstream sits further from it, and the exit face is the high-defocus
    end: the transfer function applied to the exit wave needs ``T / 2`` more
    underfocus than the midplane's, and subtracting this shift adds it. Under
    the default negative Ewald sign the beam enters at slice 0, so defocus
    *increases* with z, and for a particle at centred coordinate ``z_i`` (the
    convention :func:`specter.crowding.insert_particles_into_micrograph`
    uses) the effective defocus is ``df_ref + z_i``, where ``df_ref`` is the
    nominal defocus, at ``z = 0``. Measured, not assumed: a scatterer
    placed 96 A below the midplane produces an image matching the same
    scatterer at the midplane imaged at ``dfu - 96`` to within an RMS of
    5.9e-4, against 1.6 for the unshifted comparison and 3.1 for the
    opposite sign. Pinned by
    ``tests/test_aberrations_functions.py::test_defocus_increases_with_z``.

    That relation is what a per-particle defocus column in a STAR/`.cs`
    export has to use, and a sign error in it is invisible in the images.

    The simulator (the exit waves of :class:`~specter.imaging.Specimen`,
    :class:`~specter.imaging.Sample` and :class:`~specter.imaging.TiltSeries`
    carry it as ``defocus_shift``) and the tomogram inverse
    (:class:`~specter.ghostbuster.TomogramReconstructor`) both take the
    shift from here; a mismatch between the two images the specimen at
    different defocus, which ``tests/test_forward_model_parity.py`` pins.

    Parameters
    ----------
    nz : int
        Number of Z slices in the (possibly padded) simulation volume.
    pixel_size : float
        Pixel size in Å.

    Returns
    -------
    float
        Shift in Å.

    References
    ----------
    .. [1] Balakrishnan, D., Chee, S. W., Baraissov, Z., Bosman, M., Mirsaidov,
       U., & Loh, N. D. (2023). Single-shot, coherent, pop-out 3D metrology.
       *Communications Physics*, 6(1), 321.
       https://doi.org/10.1038/s42005-023-01431-6
    """
    return -(nz * pixel_size) / 2


def aberration_model_for_scattering(
    scattering_model: ScatteringModel,
) -> AberrationModel:
    """
    The ``aberration_model`` implied by a given ``scattering_model``.

    ``"ctf"`` is the only ``scattering_model`` whose exit wave has no
    complex/absorptive component of its own -- ``Scattering.ctf()`` returns
    a real-valued projected potential, which needs the ``"linear"``
    aberration model (real output, weak-phase-object image formation).
    Every other ``scattering_model`` produces a complex exit wave from full
    wave-optics propagation, matching ``"nonlinear"``. Not user-configurable
    independently of ``scattering_model``: the two must agree, or the
    aberration/detector stage misinterprets the exit wave it's given.

    Parameters
    ----------
    scattering_model : str
        The scattering model in use (``"multislice"``, ``"rytov"``,
        ``"firstborn"``, ``"projection"``, or ``"ctf"``).

    Returns
    -------
    str
        ``"linear"`` if ``scattering_model == "ctf"``, else ``"nonlinear"``.
    """
    return "linear" if scattering_model == "ctf" else "nonlinear"
