"""Fourier-space envelope functions.

Pure functions of frequency-grid tensors and physical parameters — no
dependence on a class instance. Each function returns a real-valued,
multiplicative amplitude attenuation (an "envelope") in the range (0, 1]
that can be multiplied into a transfer function. Math ported from
teamtomo's torch_fourier_filter.envelopes, adapted to specter's existing
unit conventions (wavelength/cs/defocus/cc in Å, angles in mrad).

References
----------
.. [1] Grant, T., & Grigorieff, N. (2015). Measuring the optimal exposure for
   single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6.
   *eLife*, 4, e06980. https://doi.org/10.7554/eLife.06980
.. [2] torch-fourier-filter, ``torch_fourier_filter.envelopes``. teamtomo.
   https://github.com/teamtomo/torch-fourier-filter
"""

from __future__ import annotations

import math
from typing import TypeVar

import torch

from ..constants import rest_mass_energy


#: A Gaussian's FWHM over its standard deviation, ``2 sqrt(2 ln 2)``.
FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


_Voltage = TypeVar("_Voltage", float, torch.Tensor)


def _inverse_beta_squared(voltage_kv: _Voltage) -> _Voltage:
    """``1 / beta^2 = gamma^2 / (gamma^2 - 1)`` at an accelerating voltage in kV."""
    gamma2 = (1.0 + voltage_kv * 1.0e3 / rest_mass_energy()) ** 2
    return gamma2 / (gamma2 - 1.0)


#: Radiation damage per incident electron at 100 keV over that at 300 keV,
#: 1.57 +/- 0.03: the mean ratio of the critical exposures of paraffin and
#: purple-membrane 2D crystals at 80 K, from the fading of their diffraction
#: spots at 9, 5 and 4 A (Peet, Henderson & Russo 2019, Table 1). The elastic
#: cross section of carbon rises by 2.01 +/- 0.05 over the same step and the
#: inelastic one by 1.51 +/- 0.1, which the damage follows.
DAMAGE_RATIO_100_TO_300_KV = 1.57

#: Intercept ``a`` of the damage per incident electron, taken as proportional
#: to ``a + 1/beta^2``. Peet, Henderson & Russo (2019) measured the inelastic
#: cross section of carbon to be linear in ``1/beta^2`` over 100-300 keV, with
#: a slope and intercept that differ from the elastic cross section's, and the
#: damage to follow the inelastic cross section. ``a`` is solved here so that
#: the 100/300 keV ratio equals ``DAMAGE_RATIO_100_TO_300_KV`` exactly: 1.2695.
DAMAGE_INTERCEPT = (
    _inverse_beta_squared(100.0)
    - DAMAGE_RATIO_100_TO_300_KV * _inverse_beta_squared(300.0)
) / (DAMAGE_RATIO_100_TO_300_KV - 1.0)


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

    References
    ----------
    .. [1] Rosenthal, P. B., & Henderson, R. (2003). Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy. *Journal of Molecular
       Biology*, 333(4), 721–745. https://doi.org/10.1016/j.jmb.2003.07.013
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
        Beam convergence semi-angle, in milliradians: the 1/e half-angle of a
        Gaussian source distribution ``exp(-(theta / alpha)^2)``, Frank's
        ``alpha_0``, which is ``sqrt(2)`` times the distribution's standard
        deviation.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope due to partial spatial coherence.

    Notes
    -----
    The Gaussian average of ``exp(i grad(chi) . theta)`` over source angles
    of standard deviation ``sigma`` is ``exp(-sigma^2 |grad chi|^2 / 2)``.
    With ``chi = -pi lambda df k^2 + (pi / 2) Cs lambda^3 k^4`` (defocus
    positive for underfocus, as in `_functions`), ``grad chi = 2 pi (Cs
    lambda^3 k^3 - df lambda k)``: for underfocus the two terms partly
    cancel. Writing ``alpha = sqrt(2) sigma`` gives the form below.

    References
    ----------
    .. [1] Frank, J. (1973). The envelope of electron microscopic transfer
       functions for partially coherent illumination. *Optik*, 38, 519–536.
       Reprinted in J. Frank, *Single-particle cryo-electron microscopy*
       (pp. 51–68), World Scientific, 2018.
       https://doi.org/10.1142/9789813234864_0003
    """
    half_angle = convergence_angle / 1000  # mrad -> rad
    return torch.exp(
        -((torch.pi * half_angle / wavelength) ** 2)
        * (cs * wavelength**3 * k**3 - wavelength * defocus * k) ** 2
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
        FWHM of the beam energy spread, in eV, the figure a source is quoted
        with (0.7 eV for a Schottky FEG, 0.3 for a cold FEG).
    deltaV_V : float
        Relative high-voltage instability, rms.
    deltaI_I : float
        Relative objective-lens current instability, rms.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope due to partial temporal coherence.

    Notes
    -----
    The envelope is the Gaussian average of the defocus phase
    ``exp(-i pi lambda df k^2)`` over a defocus distribution of standard
    deviation ``sigma_f``: ``exp(-(pi lambda sigma_f k^2)^2 / 2)``, with
    ``sigma_f = Cc sqrt((sigma_E / E)^2 + (dV/V)^2 + (2 dI/I)^2)``. The
    energy spread enters as a standard deviation, so the quoted FWHM is
    divided by ``2 sqrt(2 ln 2)`` first. The reference implementation this
    was ported from (teamtomo's ``torch_fourier_filter.envelopes``) has the
    same Gaussian form and the same FWHM input, fed straight in; so did this
    port until 2026-10-14, which made the exponent 5.5x too large (0.62
    instead of 0.92 at 2 A for Cc = 2.7 mm, 0.7 eV, 300 kV). Kirkland's 1/e
    width form is the same Gaussian with ``Delta = sqrt(2) sigma_f``.
    ``tests/test_aberrations_envelopes.py`` pins this against an explicit
    weighted integration over defocus.

    References
    ----------
    .. [1] Hanszen, K. J., & Trepte, L. (1971). Der Einfluß von Strom- und
       Spannungsschwankungen sowie der Energiebreite der Strahlelektronen auf
       Kontrastübertragung und Auflösung des Elektronenmikroskopes. *Optik*,
       32, 519–538.
    .. [2] Wade, R. H., & Frank, J. (1977). Electron microscope transfer
       functions for partially coherent axial illumination and chromatic
       defocus spread. *Optik*, 49, 81–92.
    .. [3] Kirkland, E. J. (2010). *Advanced computing in electron microscopy*
       (2nd ed.). Springer. https://doi.org/10.1007/978-1-4419-6533-2
    """
    focus_spread = focal_spread_sigma(cc, voltage, energy_spread, deltaV_V, deltaI_I)
    return torch.exp(-0.5 * (torch.pi * wavelength * focus_spread * k2) ** 2)


def focal_spread_sigma(
    cc: float,
    voltage: float,
    energy_spread: float,
    deltaV_V: float,
    deltaI_I: float,
) -> float:
    """
    Standard deviation of the defocus spread chromatic aberration causes.

    ``sigma_f = Cc sqrt((sigma_E / E)^2 + (dV/V)^2 + (2 dI/I)^2)``, with the
    quoted FWHM energy spread converted to its standard deviation. The one
    definition both the Cc envelope and the explicit focal-spread
    integration (:class:`~specter.aberrations.FocalSpread`) use.

    Parameters
    ----------
    cc : float
        Chromatic aberration coefficient, in Angstrom.
    voltage : float
        Accelerating voltage, in volts.
    energy_spread : float
        FWHM of the beam energy spread, in eV.
    deltaV_V : float
        Relative high-voltage instability, rms.
    deltaI_I : float
        Relative objective-lens current instability, rms.

    Returns
    -------
    float
        ``sigma_f`` in Angstrom.

    References
    ----------
    .. [1] Hanszen, K. J., & Trepte, L. (1971). Der Einfluß von Strom- und
       Spannungsschwankungen sowie der Energiebreite der Strahlelektronen auf
       Kontrastübertragung und Auflösung des Elektronenmikroskopes. *Optik*,
       32, 519–538.
    """
    sigma_e = energy_spread / FWHM_TO_SIGMA
    return cc * ((sigma_e / voltage) ** 2 + deltaV_V**2 + (2 * deltaI_I) ** 2) ** 0.5


def critical_exposure(
    k: torch.Tensor,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    voltage: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Grant & Grigorieff (2015) critical exposure :math:`N_e(k) = a k^b + c`.

    The exposure, in e⁻/Å², after which the diffracted *intensity* at
    spatial frequency ``k`` has fallen to 1/e of its undamaged value.
    ``c`` is part of the fit, not an onset dose: damage starts at zero
    exposure at every frequency. At ``k = 0`` the value is ``inf``.

    The fit was measured at 300 kV. At another accelerating voltage the
    critical exposure is divided by the damage per incident electron
    relative to 300 kV. That ratio follows the inelastic cross section, not
    the elastic one: Peet et al. [2]_ measured the damage to 2D crystals as
    1.57 +/- 0.03 times higher at 100 keV than at 300 keV, against
    2.01 +/- 0.05 for elastic scattering (the :math:`1/\\beta^2` law) and
    1.51 +/- 0.1 for inelastic scattering, which they found linear in
    :math:`1/\\beta^2` with a different slope and intercept from the
    elastic cross section. The damage per electron is therefore taken as
    proportional to :math:`a + 1/\\beta^2`, with ``a = DAMAGE_INTERCEPT``
    (1.27) solved from the measured ratio, so that

    .. math::

        N_e(k; V) = N_e(k; 300\\,\\text{kV})
        \\frac{a + 1/\\beta^2(300\\,\\text{kV})}{a + 1/\\beta^2(V)}.

    The factor is 1 at 300 kV, 0.637 at 100 kV (the measured
    :math:`1/1.57`, which RELION also applies [3]_) and 0.877 at 200 kV.
    The damage ratio is measured at 100 kV only, so the 200 kV value is an
    interpolation, and outside 100-300 kV the form is an extrapolation.
    The 0.80 that RELION, cisTEM and MotionCor3 apply at 200 kV is a
    convention: it is the elastic ratio
    :math:`\\beta^2(200\\,\\text{kV})/\\beta^2(300\\,\\text{kV})`, and
    Grant & Grigorieff's "about 25% lower" at 200 kV [1]_ rests on an
    elastic cross section as well. For a 200 kV pre-exposure series,
    Danev et al. [4]_ found the fit scaled by 0.75 predicted a worse
    resolution than was reached, and taking the exposure at 82% of its
    measured value brought the prediction close to it, an effective factor
    of about 0.91 (0.75/0.82).

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Å.
    a, b, c : float, optional
        Fitted parameters from Grant & Grigorieff (2015) [1]_, for 300 kV.
    voltage : float or torch.Tensor, optional
        Accelerating voltage in kV. None (default) means 300 kV. A 1-D
        tensor is taken as one value per image and broadcast as (B, 1, 1).

    Returns
    -------
    torch.Tensor
        Critical exposure in e⁻/Å², same shape as ``k`` (times the batch).

    References
    ----------
    .. [1] Grant, T., & Grigorieff, N. (2015). Measuring the optimal exposure for
       single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6.
       *eLife*, 4, e06980. https://doi.org/10.7554/eLife.06980
    .. [2] Peet, M. J., Henderson, R., & Russo, C. J. (2019). The energy
       dependence of contrast and damage in electron cryomicroscopy of
       biological molecules. *Ultramicroscopy*, 203, 125–131.
       https://doi.org/10.1016/j.ultramic.2019.02.007
    .. [3] Naydenova, K., McMullan, G., Peet, M. J., Lee, Y., Edwards, P. C.,
       Chen, S., Leahy, E., Scotcher, S., Henderson, R., & Russo, C. J.
       (2019). CryoEM at 100 keV: a demonstration and prospects. *IUCrJ*,
       6(6), 1086–1098. https://doi.org/10.1107/S2052252519012612
    .. [4] Danev, R., Yanagisawa, H., Yamashita, K., Eisenstein, F., &
       Kikkawa, M. (2026). Atomic resolution cryo-EM at 200 keV. *IUCrJ*,
       13(4), 343–353. https://doi.org/10.1107/S2052252526004100
    """
    ne = a * torch.pow(k, b) + c
    if voltage is None:
        return ne
    # In float64, so that 300 kV gives a factor of exactly 1.
    v = torch.as_tensor(voltage, dtype=torch.float64, device=ne.device)
    if v.ndim == 1:
        v = v.view(-1, 1, 1)
    damage_300 = DAMAGE_INTERCEPT + _inverse_beta_squared(300.0)
    factor = damage_300 / (DAMAGE_INTERCEPT + _inverse_beta_squared(v))
    return ne * factor.to(ne.dtype)


def dose_envelope(
    k: torch.Tensor,
    dose: torch.Tensor,
    pre_exposure: torch.Tensor | float = 0.0,
    weighted: bool = True,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    voltage: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Signal amplitude envelope of an image accumulated over an exposure.

    Grant & Grigorieff (2015) [1]_ measured that the signal-to-noise ratio
    at frequency ``k`` decays as :math:`\\exp(-N / N_e(k))` with exposure
    ``N``, so a frame recorded at cumulative exposure ``N`` carries
    :math:`q(k, N) = \\exp(-N / 2 N_e(k))` of its undamaged amplitude. An
    image is a sum of such frames, and its envelope is the average of
    ``q`` over the exposure interval it spans, not ``q`` evaluated at the
    end of it. Two sums occur in practice:

    - ``weighted=True``: an exposure-filtered (dose-weighted) movie sum,
      :math:`\\sum_i q_i F_i / \\sqrt{\\sum_i q_i^2}`, which is what motion
      correction produces for single-particle data. Frames carry equal
      Poisson noise, and the normalisation keeps the noise power of the
      sum equal to an unweighted sum's, so at equal noise the signal
      envelope is :math:`\\sqrt{\\tfrac{1}{D}\\int_{N_0}^{N_0+D} e^{-N/N_e}\\,dN}`.
    - ``weighted=False``: a plain sum of the frames, whose envelope is
      :math:`\\tfrac{1}{D}\\int_{N_0}^{N_0+D} e^{-N/2N_e}\\,dN`. This is the
      form for a single tilt of a tilt series (a short exposure ``D`` after
      a pre-exposure ``N_0`` from the earlier tilts) and for movies summed
      without an exposure filter.

    Both reduce to :math:`\\exp(-N_0 / 2N_e)` as ``D -> 0`` and to 1 at
    ``k = 0``. At 40 e⁻/Å² from zero pre-exposure the weighted form gives
    0.58, 0.46 and 0.35 at 10, 6.7 and 3.7 Å, the plain sum 0.53, 0.39 and
    0.24. The noise is *not* attenuated: the simulator draws it at the
    full dose, which is exactly the normalisation the two forms assume.

    Parameters
    ----------
    k : torch.Tensor
        Frequency magnitude grid, in 1/Å.
    dose : torch.Tensor
        Electron dose (fluence) accumulated *in this image*, in e⁻/Å².
        Broadcast against ``k``.
    pre_exposure : torch.Tensor or float, optional
        Exposure the specimen had already received when this image
        started, in e⁻/Å². Zero (default) for a single-particle movie;
        the summed dose of the earlier tilts for a tilt image.
    weighted : bool, optional
        Whether the image is an exposure-filtered sum (True, default) or a
        plain one (False). See above.
    a, b, c : float, optional
        Critical-exposure fit parameters, see :func:`critical_exposure`.
    voltage : float or torch.Tensor, optional
        Accelerating voltage in kV, which rescales the critical exposure by
        the measured voltage dependence of radiation damage (0.637 at
        100 kV, 0.877 at 200 kV; see :func:`critical_exposure`). None means
        300 kV.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope in (0, 1], broadcast shape of ``k`` and ``dose``.

    References
    ----------
    .. [1] Grant, T., & Grigorieff, N. (2015). Measuring the optimal exposure for
       single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6.
       *eLife*, 4, e06980. https://doi.org/10.7554/eLife.06980
    """
    ne = critical_exposure(k, a, b, c, voltage)
    pre = torch.as_tensor(pre_exposure, dtype=ne.dtype, device=ne.device)
    dose = torch.as_tensor(dose, dtype=ne.dtype, device=ne.device)
    # exp(-s/x) - exp(-e/x) == -exp(-s/x) * expm1(-D/x): stable as x -> inf.
    if weighted:
        integral = -ne * torch.exp(-pre / ne) * torch.expm1(-dose / ne)
        env = torch.sqrt(integral / dose)
    else:
        integral = (
            -2.0 * ne * torch.exp(-pre / (2.0 * ne)) * torch.expm1(-dose / (2.0 * ne))
        )
        env = integral / dose
    # k == 0 has ne == inf, where the expression above is 0 * inf; the
    # limit is exp(-pre / 2 ne) -> 1. dose == 0 is the same limit.
    limit = torch.exp(-pre / (2.0 * ne))
    return torch.where(torch.isfinite(env), env, limit.expand_as(env))
