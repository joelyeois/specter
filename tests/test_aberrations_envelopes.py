from __future__ import annotations

import math

import pytest
import torch

from specter.aberrations import Aberration
from specter.aberrations._envelopes import (
    b_envelope,
    cc_envelope,
    cs_envelope,
    dose_envelope,
)


def test_static_envelopes_match_their_closed_forms():
    """Pin the three parameter-driven envelopes at 300 kV (lambda = 0.0197 A).

    Each is exp of a negative quantity, so "1 at DC and falling with k" holds
    for any coefficient and pins nothing. What these functions can actually
    get wrong is the coefficient: the 1/4 on the B-factor, the mrad -> rad
    conversion on the convergence semi-angle, the factor 2 on the lens-current
    instability, and the 1/2 in the Cc exponent. The values below are computed
    by hand from the intermediate physical quantities, at k = 0.1, 0.3 and
    0.5 1/A.
    """
    k = torch.tensor([0.1, 0.3, 0.5])
    wavelength = 0.0197

    # exp(-B k^2 / 4), B = 50 A^2.
    assert torch.allclose(
        b_envelope(k**2, torch.tensor(50.0)),
        torch.tensor([0.8825, 0.3247, 0.0439]),
        atol=1e-4,
    )

    # exp(-(pi alpha / lambda)^2 (Cs lambda^3 k^3 + dz lambda k)^2), with the
    # semi-angle in radians: 0.1 mrad -> 1e-4.
    assert torch.allclose(
        cs_envelope(
            k,
            wavelength=wavelength,
            cs=torch.tensor(2.7e7),
            defocus=torch.tensor(10000.0),
            convergence_angle=0.1,
        ),
        torch.tensor([0.9041, 0.3452, 0.0197]),
        atol=1e-4,
    )

    # exp(-(pi lambda d k^2)^2 / 2) with a focus spread
    # d = Cc sqrt((dE/V)^2 + (dV/V)^2 + (2 dI/I)^2) = 87.258 A. The two
    # instabilities are 1 ppm rather than an instrument's real 0.01-0.06, so
    # that all three terms contribute: at realistic values the energy spread
    # is five orders up and the sum pins only itself (dropping the factor 2
    # on dI/I would move the focus spread by 0.003%, against 16% here).
    assert torch.allclose(
        cc_envelope(
            k**2,
            wavelength=wavelength,
            cc=2.7e7,
            voltage=3.0e5,
            energy_spread=0.7,
            deltaV_V=1e-6,
            deltaI_I=1e-6,
        ),
        torch.tensor([0.9985, 0.8886, 0.4020]),
        atol=1e-4,
    )


def test_dose_envelope_dc_is_one():
    k = torch.tensor([0.0])
    for weighted in (True, False):
        result = dose_envelope(k, dose=torch.tensor(50.0), weighted=weighted)
        assert torch.allclose(result, torch.ones_like(result))
        result = dose_envelope(
            k, dose=torch.tensor(3.0), pre_exposure=60.0, weighted=weighted
        )
        assert torch.allclose(result, torch.ones_like(result))


def test_dose_envelope_matches_grant_grigorieff_2015_integrals():
    """Pin the two closed forms at 40 e/A^2 against a numerical quadrature.

    Ne = a k^b + c is the exposure at which the diffracted *intensity* has
    fallen to 1/e (Grant & Grigorieff 2015, eLife 4:e06980), so a frame at
    exposure N carries exp(-N / 2Ne) of its amplitude. The envelope of an
    image is that decay averaged over the exposure it spans: for a plain sum
    the mean of exp(-N/2Ne); for an exposure-filtered sum with weights
    q = exp(-N/2Ne) and noise-preserving normalisation, sqrt of the mean of
    exp(-N/Ne). Neither is exp evaluated at the final dose, which is what the
    function computed before 2026-09-03 (exp(-(D - c) / (a k^b)): 4e-8 at
    3.7 A, where the physics says 0.24-0.35).
    """
    k = torch.tensor([0.1, 0.149, 0.27])  # 10, 6.7 and 3.7 A
    ne = 0.245 * k**-1.665 + 2.81
    dose = 40.0
    n = torch.linspace(0.0, dose, 20001)[:, None]
    plain_ref = torch.exp(-n / (2 * ne)).mean(0)
    weighted_ref = torch.sqrt(torch.exp(-n / ne).mean(0))
    plain = dose_envelope(k, torch.tensor(dose), weighted=False)
    weighted = dose_envelope(k, torch.tensor(dose), weighted=True)
    assert torch.allclose(plain, plain_ref, rtol=1e-3)
    assert torch.allclose(weighted, weighted_ref, rtol=1e-3)
    # The values quoted in the docstring, so a silent parameter change shows.
    assert torch.allclose(plain, torch.tensor([0.535, 0.389, 0.245]), atol=2e-3)
    assert torch.allclose(weighted, torch.tensor([0.577, 0.462, 0.353]), atol=2e-3)
    # Optimal weighting recovers more signal than a plain sum, everywhere.
    assert torch.all(weighted > plain)


def test_dose_envelope_pre_exposure_short_interval_limit():
    """A short exposure after a pre-exposure N0 tends to exp(-N0 / 2Ne)."""
    k = torch.tensor([0.05, 0.1, 0.2, 0.3])
    ne = 0.245 * k**-1.665 + 2.81
    pre = 30.0
    expected = torch.exp(-pre / (2 * ne))
    for weighted in (True, False):
        short = dose_envelope(
            k, dose=torch.tensor(1e-3), pre_exposure=pre, weighted=weighted
        )
        assert torch.allclose(short, expected, rtol=1e-3)
        zero = dose_envelope(
            k, dose=torch.tensor(0.0), pre_exposure=pre, weighted=weighted
        )
        assert torch.allclose(zero, expected, rtol=1e-6)
    # A tilt series is a plain sum over [N0, N0 + D]: the mean of the decay.
    n = torch.linspace(pre, pre + 3.0, 20001)[:, None]
    ref = torch.exp(-n / (2 * ne)).mean(0)
    tilt = dose_envelope(k, dose=torch.tensor(3.0), pre_exposure=pre, weighted=False)
    assert torch.allclose(tilt, ref, rtol=1e-3)


def test_critical_exposure_scales_with_beta_squared():
    """300 kV is the fit itself; 200 kV gives the conventional 0.80; 100 kV 0.50."""
    from specter.aberrations._envelopes import critical_exposure

    k = torch.tensor([0.05, 0.1, 0.2, 0.3])
    ne300 = critical_exposure(k)
    assert torch.allclose(critical_exposure(k, voltage=300.0), ne300)
    assert torch.allclose(
        critical_exposure(k, voltage=200.0) / ne300,
        torch.full_like(k, 0.802),
        atol=2e-3,
    )
    assert torch.allclose(
        critical_exposure(k, voltage=100.0) / ne300,
        torch.full_like(k, 0.498),
        atol=2e-3,
    )
    # A per-image voltage tensor broadcasts as (B, 1, 1) against a 2-D grid.
    grid = k.view(1, -1).expand(4, -1)
    per_image = critical_exposure(grid, voltage=torch.tensor([300.0, 100.0]))
    assert per_image.shape == (2, 4, 4)
    assert torch.allclose(per_image[0], ne300.expand(4, -1))
    assert torch.allclose(
        per_image[1] / per_image[0], torch.full((4, 4), 0.498), atol=2e-3
    )
    # The envelope uses the scaled Ne, so 100 kV damages more than 300 kV.
    e100 = dose_envelope(k, torch.tensor(40.0), voltage=100.0)
    e300 = dose_envelope(k, torch.tensor(40.0), voltage=300.0)
    ne100 = critical_exposure(k, voltage=100.0)
    ref = torch.sqrt(ne100 * (1 - torch.exp(-40.0 / ne100)) / 40.0)
    assert torch.allclose(e100, ref, rtol=1e-5)
    assert torch.all(e100 < e300)


def test_dose_envelope_decreases_with_frequency_and_dose():
    k = torch.tensor([0.05, 0.1, 0.2, 0.3])
    result = dose_envelope(k, dose=torch.tensor(50.0))
    assert torch.all(result[:-1] >= result[1:])
    assert result[-1] < result[0]
    # Damage starts at zero exposure: there is no threshold below which the
    # envelope is exactly 1.
    small = dose_envelope(k, dose=torch.tensor(1.0))
    assert torch.all(small < 1.0)
    assert torch.all(small > result)


def test_aberration_dose_envelope_uses_pre_exposure_and_weighting():
    """The transfer function applies the weighted form by default, and the
    plain interval form when `dose_weighted=False` with a pre_exposure key."""
    ctf_params = {
        "dfu": torch.tensor([5000.0]),
        "dose": torch.tensor([3.0]),
        "pre_exposure": torch.tensor([30.0]),
    }
    weighted = Aberration(
        16,
        pixel_size=1.0,
        voltage=300.0,
        aberration_model="nonlinear",
        dose_envelope=True,
    )
    tilt = Aberration(
        16,
        pixel_size=1.0,
        voltage=300.0,
        aberration_model="nonlinear",
        dose_envelope=True,
        dose_weighted=False,
    )
    plain = Aberration(16, pixel_size=1.0, voltage=300.0, aberration_model="nonlinear")
    base = torch.abs(plain.transfer_function({"dfu": ctf_params["dfu"]}))
    for ab, w in ((weighted, True), (tilt, False)):
        expected = base * dose_envelope(
            ab.k, ctf_params["dose"].view(-1, 1, 1), pre_exposure=30.0, weighted=w
        )
        got = torch.abs(ab.transfer_function(ctf_params))
        assert torch.allclose(got, expected, atol=1e-6)
    # Without a pre_exposure key the exposure starts at zero.
    got = torch.abs(
        weighted.transfer_function(
            {"dfu": ctf_params["dfu"], "dose": ctf_params["dose"]}
        )
    )
    expected = base * dose_envelope(weighted.k, ctf_params["dose"].view(-1, 1, 1))
    assert torch.allclose(got, expected, atol=1e-6)


def test_aberration_bfactor_matches_inline_formula():
    ab = Aberration(8, pixel_size=1.0, voltage=300.0, aberration_model="nonlinear")
    ctf_params = {
        "dfu": torch.tensor([5000.0]),
        "bfactor": torch.tensor([50.0]),
    }
    transfer = ab.transfer_function(ctf_params)
    expected_envelope = torch.exp(-50.0 * ab.k2 / 4)
    assert torch.allclose(torch.abs(transfer), expected_envelope, atol=1e-6)


def test_aberration_convergence_angle_matches_closed_form():
    """Pin the spatial-coherence envelope through the transfer function.

    What the call site has to get right, beyond attenuating high frequencies
    at all: the k grid built from ``pixel_size``, the wavelength from
    ``voltage``, the semi-angle handed over in milliradians, and the *mean*
    of dfu and dfv as the defocus -- which is why the two differ here.
    """
    n_pixels, pixel_size = 32, 1.2
    ab = Aberration(
        n_pixels,
        pixel_size=pixel_size,
        voltage=300.0,
        aberration_model="nonlinear",
        convergence_angle=0.1,
    )
    assert ab.wavelength == pytest.approx(0.019687, abs=1e-6)  # 1.969 pm at 300 kV
    ctf_params = {
        "dfu": torch.tensor([9000.0]),
        "dfv": torch.tensor([11000.0]),
        "cs": torch.tensor([2.7e7]),
    }

    freq = torch.fft.fftfreq(n_pixels, pixel_size)
    k = torch.sqrt(freq[:, None] ** 2 + freq[None, :] ** 2)
    lam, half_angle, defocus = ab.wavelength, 0.1e-3, 10000.0
    expected = torch.exp(
        -((torch.pi * half_angle / lam) ** 2)
        * (2.7e7 * lam**3 * k**3 + lam * defocus * k) ** 2
    )
    got = torch.abs(ab.transfer_function(ctf_params)).squeeze(0)
    assert torch.allclose(got, expected, atol=1e-6)
    # Nyquist is damped to 0.2% here, so the comparison spans the whole range
    # rather than agreeing on a grid where the envelope is everywhere ~1.
    assert expected.min() < 0.01


def test_aberration_cc_matches_closed_form():
    """Pin the temporal-coherence envelope through the transfer function.

    The call site owns the kV -> V conversion on the accelerating voltage
    (the energy spread it divides is in eV) and the instrument defaults
    ``energy_spread``/``deltaV_V``/``deltaI_I``, which together set the
    focus spread.
    """
    n_pixels, pixel_size = 32, 1.2
    ab = Aberration(
        n_pixels,
        pixel_size=pixel_size,
        voltage=300.0,
        aberration_model="nonlinear",
        cc=2.7e7,
    )
    freq = torch.fft.fftfreq(n_pixels, pixel_size)
    k2 = freq[:, None] ** 2 + freq[None, :] ** 2
    # d = Cc sqrt((dE/V)^2 + (dV/V)^2 + (2 dI/I)^2), with V in volts.
    focus_spread = 2.7e7 * math.sqrt(
        (0.7 / 300e3) ** 2 + 0.06e-6**2 + (2 * 0.01e-6) ** 2
    )
    assert focus_spread == pytest.approx(63.02, abs=0.01)
    expected = torch.exp(-0.5 * (torch.pi * ab.wavelength * focus_spread * k2) ** 2)
    got = torch.abs(ab.transfer_function({"dfu": torch.tensor([10000.0])})).squeeze(0)
    assert torch.allclose(got, expected, atol=1e-6)
    assert expected.min() < 0.5


def test_aberration_dose_envelope_false_is_unchanged():
    ab_off = Aberration(8, pixel_size=1.0, voltage=300.0, aberration_model="nonlinear")
    ab_on = Aberration(
        8,
        pixel_size=1.0,
        voltage=300.0,
        aberration_model="nonlinear",
        dose_envelope=False,
    )
    ctf_params = {"dfu": torch.tensor([5000.0]), "dose": torch.tensor([50.0])}
    assert torch.allclose(
        ab_off.transfer_function({"dfu": torch.tensor([5000.0])}),
        ab_on.transfer_function(ctf_params),
    )


def test_aberration_dose_envelope_attenuates_high_frequency():
    ab = Aberration(
        64,
        pixel_size=1.0,
        voltage=300.0,
        aberration_model="nonlinear",
        dose_envelope=True,
    )
    ctf_params = {"dfu": torch.tensor([5000.0]), "dose": torch.tensor([50.0])}
    transfer = ab.transfer_function(ctf_params)
    magnitude = torch.abs(transfer).squeeze(0)
    assert magnitude[0, 1] > magnitude[0, 20]


def test_ctf_batch_includes_dose():
    from specter.imagegenerator._base import BaseImager

    gen = BaseImager(
        pixel_size=1.0,
        voltage=300.0,
        dose_per_angstrom=torch.tensor([40.0, 60.0]),
        nxy=16,
        nz=16,
        ctf_params={"dfu": torch.tensor([5000.0, 6000.0])},
        progressbars=False,
        verbose=False,
    )
    batch = gen._ctf_batch(torch.tensor([0, 1]))
    assert "dose" in batch
    assert torch.allclose(batch["dose"], torch.tensor([40.0, 60.0]))
