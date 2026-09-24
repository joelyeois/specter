"""
The solvent exposure filter: how much ice structure survives a summed movie.

Self-contained checks of `specter.ice._exposure`: exact limits of the Gaussian
model, the relaxed model against direct quadrature and dense covariances, and
behaviour on grids from 0.25 to 10 A.
"""

import math

import numpy as np
import pytest
import torch

from specter.ice import (
    apply_solvent_exposure,
    solvent_coherence,
    solvent_decorrelation_rate,
    solvent_exposure_power,
)


def _dense(k, D, n, weights, model, sub=400):
    """a^T C a from an explicit sub-sampled covariance of the instantaneous amplitudes."""
    a = weights / weights.sum()
    ts = (np.arange(n * sub) + 0.5) * (D / n) / sub
    lag = torch.tensor(np.abs(ts[:, None] - ts[None, :]))
    cov = solvent_coherence(k, lag, 0.38, model).numpy()
    fw = np.repeat(a, sub) / sub
    return fw @ cov @ fw


@pytest.mark.parametrize("model", ["gaussian", "relaxed"])
def test_weighted_exposure_matches_dense_covariance(model):
    rng = np.random.default_rng(9)
    n, D = 12, 50.0
    w = rng.uniform(0.1, 1, n)
    k = torch.tensor([0.1], dtype=torch.float64)
    actual = solvent_exposure_power(
        k, D, 0.38, n, torch.tensor(w[:, None]), 1.0, model=model
    )
    assert float(actual) == pytest.approx(_dense(k, D, n, w, model), rel=2e-4)


@pytest.mark.parametrize("model", ["gaussian", "relaxed"])
def test_uniform_exposure_is_independent_of_frame_count(model):
    """Equal weights: the continuous average over the exposure, whatever n."""
    k = torch.tensor([0.01, 0.1, 0.27, 0.6], dtype=torch.float64)
    one = solvent_exposure_power(k, 50, 0.38, 1, model=model)
    for n in [7, 40, 200]:
        torch.testing.assert_close(
            solvent_exposure_power(k, 50, 0.38, n, model=model),
            one,
            rtol=1e-9,
            atol=1e-12,
        )


def test_no_motion_or_no_dose_keeps_everything():
    k = torch.tensor([0.0, 0.05, 0.27, 0.6], dtype=torch.float64)
    for model in ("gaussian", "relaxed"):
        torch.testing.assert_close(
            solvent_exposure_power(k, 50, 0, 40, model=model), torch.ones_like(k)
        )
        torch.testing.assert_close(
            solvent_exposure_power(k, 0, 0.38, 40, model=model), torch.ones_like(k)
        )


def test_gaussian_matches_mcmullan_closed_form():
    """Equal weights: 2 (Z + e^-Z - 1) / Z^2 with Z = 2 pi^2 k^2 sigma0^2 D (their Eq. 15)."""
    k = torch.tensor([0.05, 0.27, 0.6], dtype=torch.float64)
    Z = 2 * math.pi**2 * k.square() * 0.38 * 50
    expected = 2 * (Z + torch.exp(-Z) - 1) / Z.square()
    torch.testing.assert_close(
        solvent_exposure_power(k, 50, 0.38, 40, model="gaussian"), expected
    )


@pytest.mark.parametrize("k", [0.002, 0.05, 1 / 3.7, 2.5])
def test_relaxed_exposure_matches_direct_quadrature(k):
    """
    From ice that barely decorrelates over the exposure to ice that
    decorrelates within a small fraction of a frame.
    """
    D = 50.0
    kk = torch.tensor([k], dtype=torch.float64)
    t = np.concatenate([[0.0], np.geomspace(1e-9, D, 200001)])
    rho = solvent_coherence(kk, torch.tensor(t), 0.38).numpy()
    brute = 2 / D**2 * np.trapezoid((D - t) * rho, t)
    assert float(solvent_exposure_power(kk, D, 0.38, 40)) == pytest.approx(
        brute, rel=1e-4
    )


def test_models_share_the_correlation_area_at_the_water_ring():
    """sigma0^2 is measured at 3.7 A; the relaxed curve is fixed to it there."""
    k = torch.tensor([1 / 3.7], dtype=torch.float64)
    torch.testing.assert_close(
        solvent_decorrelation_rate(k, 0.38, "relaxed"),
        solvent_decorrelation_rate(k, 0.38, "gaussian"),
        rtol=1e-9,
        atol=0.0,
    )


def test_relaxed_ice_forgets_long_wavelengths_faster():
    k = torch.tensor([1 / 20, 1 / 10], dtype=torch.float64)
    assert torch.all(
        solvent_decorrelation_rate(k, 0.38)
        > 10 * solvent_decorrelation_rate(k, 0.38, "gaussian")
    )
    kept_relaxed = solvent_exposure_power(k, 303.0, 0.38, 141)
    kept_gaussian = solvent_exposure_power(k, 303.0, 0.38, 141, model="gaussian")
    assert torch.all(kept_relaxed < kept_gaussian / 5)


def test_relaxed_coherence_is_compressed_not_exponential():
    """Against an exponential of the same area: higher at short lags, lower at long."""
    k = torch.tensor([1 / 10, 1 / 3.7], dtype=torch.float64)
    gamma = solvent_decorrelation_rate(k, 0.38)
    for lag, above in [(0.2, True), (8.0, False)]:
        rho = solvent_coherence(k, lag / gamma, 0.38)
        expo = math.exp(-lag)
        assert torch.all(rho > expo) if above else torch.all(rho < expo)


def test_relaxed_rate_edges_are_diffusive_and_continuous():
    from specter.ice._exposure import _RELAXED_K

    lo, hi = _RELAXED_K[0], _RELAXED_K[-1]
    k = torch.tensor(
        [
            1e-6,
            lo / 4,
            lo / 2,
            lo * (1 - 1e-9),
            lo * (1 + 1e-9),
            0.3,
            hi * (1 - 1e-9),
            hi * (1 + 1e-9),
            2 * hi,
            4 * hi,
        ],
        dtype=torch.float64,
    )
    one = solvent_decorrelation_rate(k, 1.0)
    assert float(one[0]) < 1e-6
    assert one[2] / one[1] == pytest.approx(4.0)
    assert one[4] == pytest.approx(float(one[3]), rel=1e-6)
    assert one[7] == pytest.approx(float(one[6]), rel=1e-6)
    assert one[9] / one[8] == pytest.approx(4.0)
    torch.testing.assert_close(solvent_decorrelation_rate(k, 0.38), 0.38 * one)
    with pytest.raises(ValueError, match="unknown coherence model"):
        solvent_decorrelation_rate(k, 1.0, "brownian")  # type: ignore[arg-type]


def test_filter_preserves_mean_and_sets_projected_power():
    torch.manual_seed(4)
    x = torch.rand(1, 16, 16, 16) + 2
    original = x.clone()
    apply_solvent_exposure(x, 1.0, 50.0, 0.38, 40, None, None)
    torch.testing.assert_close(x.mean(), original.mean(), rtol=1e-6, atol=1e-6)
    k = torch.hypot(torch.fft.fftfreq(16)[:, None], torch.fft.rfftfreq(16)[None, :])
    expected = (
        torch.fft.rfft2(original.sum(1))
        * solvent_exposure_power(k, 50, 0.38, 40).sqrt()
    )
    torch.testing.assert_close(
        torch.fft.rfft2(x.sum(1)), expected, atol=0.004, rtol=0.001
    )


@pytest.mark.parametrize("pixel_size", [0.25, 1.0, 10.0])
def test_filter_is_well_behaved_from_fine_to_coarse_sampling(pixel_size):
    k = torch.linspace(0.0, math.sqrt(3) / (2 * pixel_size), 400, dtype=torch.float64)
    kept = solvent_exposure_power(k, 303.0, 0.38, 141)
    assert torch.isfinite(kept).all()
    assert torch.all((kept >= 0) & (kept <= 1))
    assert kept[0] == 1.0
    torch.manual_seed(1)
    x = torch.rand(1, 24, 24, 24) + 2
    original = x.clone()
    apply_solvent_exposure(x, pixel_size, 303.0, 0.38, 141, None, None)
    assert torch.isfinite(x).all()
    torch.testing.assert_close(x.mean(), original.mean(), rtol=1e-6, atol=1e-6)
