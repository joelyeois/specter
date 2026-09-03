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

    The fit was measured at 300 kV. Damage per electron scales with the
    inelastic cross section, which goes as :math:`1/\\beta^2`, so at
    another accelerating voltage the critical exposure is scaled by
    :math:`\\beta^2(V) / \\beta^2(300\\,\\text{kV})`: 0.80 at 200 kV (the
    factor RELION and MotionCor2 apply) and 0.50 at 100 kV.

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
    """
    ne = a * torch.pow(k, b) + c
    if voltage is None:
        return ne
    v = torch.as_tensor(voltage, dtype=ne.dtype, device=ne.device)
    if v.ndim == 1:
        v = v.view(-1, 1, 1)
    m0c2 = 510.998950  # keV
    beta2 = 1.0 - 1.0 / (1.0 + v / m0c2) ** 2
    beta2_300 = 1.0 - 1.0 / (1.0 + 300.0 / m0c2) ** 2
    return ne * (beta2 / beta2_300)


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
        :math:`\\beta^2(V)/\\beta^2(300\\,\\text{kV})` (see
        :func:`critical_exposure`). None means 300 kV.

    Returns
    -------
    envelope : torch.Tensor
        Amplitude envelope in (0, 1], broadcast shape of ``k`` and ``dose``.
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
