"""Independent event-counting checks for detector-noise calibration."""

import numpy as np
import pytest

from specter.match._event_covariance import (
    event_noise_power,
    estimate_detector_noise,
    exclusion_cell_efficiency,
)


def test_noise_power_matches_bernoulli_cell_events():
    rng = np.random.default_rng(876)
    q, side = 0.35, 2.7
    k = np.array([0.0, 0.1, 0.3, 0.6])
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    # Independent occupied cells, each localizing one event uniformly.
    # Cell origins affect means, but not each cell's independent variance.
    occupied = rng.random((16000, 16)) < q
    positions = rng.uniform(-side / 2, side / 2, (16000, 16, 2))
    power = []
    for frequency in k:
        directional = []
        for theta in angles:
            projection = positions @ np.array([np.cos(theta), np.sin(theta)])
            F = (occupied * np.exp(-2j * np.pi * frequency * projection)).sum(1)
            directional.append(np.mean(abs(F - F.mean()) ** 2) / (16 * q))
        power.append(np.mean(directional))
    np.testing.assert_allclose(power, event_noise_power(k, q, side), rtol=0.025)


def test_exclusion_dqe_matches_poisson_response_derivative():
    for lam in [0.001, 0.1, 0.5, 1.0]:
        q = -np.expm1(-lam)
        step = 1e-6
        derivative = (np.exp(-(lam - step)) - np.exp(-(lam + step))) / (2 * step)
        expected = lam * derivative**2 / (q * (1 - q))
        assert exclusion_cell_efficiency(q) == pytest.approx(expected, rel=1e-7)
    assert exclusion_cell_efficiency(0) == 1


def test_event_calibration_rejects_validation_movies(tmp_path):
    p = tmp_path / "withheld.npz"
    np.savez(p, split="validation")
    with pytest.raises(ValueError, match="calibration"):
        estimate_detector_noise([p], 0.3)


@pytest.mark.parametrize(
    "q,side,sigma", [(-0.1, 2, 0), (1, 2, 0), (0.2, 0, 0), (0.2, 2, -1)]
)
def test_invalid_event_parameters(q, side, sigma):
    with pytest.raises(ValueError):
        event_noise_power(np.array([0.1]), q, side, sigma)
