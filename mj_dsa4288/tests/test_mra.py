"""Smoke and correctness tests for the MRA toolkit (run: pytest mj_dsa4288/tests -v)."""

from __future__ import annotations

import pytest
import torch

from mj_dsa4288.mra.em import em_mra
from mj_dsa4288.mra.invariants import (
    bispectrum_1d,
    power_spectrum,
    third_moment_tensor_1d,
    third_moment_tensor_1d_bruteforce,
)
from mj_dsa4288.mra.jennrich import homojen, jennrich
from mj_dsa4288.mra.metrics import align_to, reconstruction_snr, rho
from mj_dsa4288.mra.model import (
    cyclic_shift,
    inverse_cyclic_shift,
    sample_mra,
    sigma_for_snr,
    snr_of,
)
from mj_dsa4288.mra.template import synthetic_template, template_from_volume


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_cyclic_shift_convention_and_inverse() -> None:
    theta = torch.arange(6.0)
    out = cyclic_shift(theta, torch.tensor([0, 2]))
    assert torch.equal(out[1], torch.tensor([2.0, 3, 4, 5, 0, 1]))  # theta[j + l]
    back = inverse_cyclic_shift(out, torch.tensor([0, 2]))
    assert torch.equal(back[1], theta)
    img = torch.arange(12.0).reshape(3, 4)
    out2 = cyclic_shift(img, torch.tensor([[1, 1]]))
    assert out2[0, 0, 0] == img[1, 1]
    assert torch.equal(inverse_cyclic_shift(out2, torch.tensor([[1, 1]]))[0], img)


def test_snr_conventions() -> None:
    theta = synthetic_template((32,), generator=_gen())
    assert torch.isclose(theta.norm(), torch.tensor(1.0))
    sigma = sigma_for_snr(theta, 4.0, "paper")
    assert sigma == pytest.approx(0.5)
    assert snr_of(theta, sigma, "paper") == pytest.approx(4.0)
    assert snr_of(theta, sigma, "pixel") < 4.0  # per-pixel SNR is ~d times smaller


def test_rho_is_shift_invariant() -> None:
    theta = synthetic_template((16, 16), generator=_gen())
    shifted = cyclic_shift(theta, torch.tensor([[3, 5]]))[0]
    assert rho(shifted, theta) < 1e-5
    assert rho(torch.rot90(shifted, 1), theta, rotations=True) < 1e-5
    assert torch.allclose(align_to(shifted, theta), theta, atol=1e-6)


def test_third_moment_tensor_matches_bruteforce() -> None:
    d, n, sigma = 5, 3, 0.7
    y = torch.randn(n, d, generator=_gen(1))
    fast = third_moment_tensor_1d(y, sigma)
    slow = third_moment_tensor_1d_bruteforce(y, sigma)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_invariants_are_unbiased() -> None:
    theta = synthetic_template((12,), generator=_gen(2))
    sigma = 1.0
    y, _, _ = sample_mra(theta, 200_000, sigma, generator=_gen(3))
    ps = power_spectrum(y, sigma)
    ps_true = torch.fft.fft(theta, norm="ortho").abs().pow(2)
    assert torch.allclose(ps, ps_true, atol=0.05)
    tf = torch.fft.fft(theta)
    k = torch.arange(12)
    b_true = tf[:, None] * tf[None, :] * tf[(k[:, None] + k[None, :]) % 12].conj()
    b_est = bispectrum_1d(y, sigma)
    assert (b_est - b_true).abs().max() < 0.15 * b_true.abs().max() + 1.0


def test_jennrich_recovers_components_noiseless() -> None:
    d = 7
    theta = synthetic_template((d,), generator=_gen(4))
    y = cyclic_shift(theta, torch.arange(d))  # all shifts, no noise
    est = homojen(y, sigma=0.0, generator=_gen(5))
    assert rho(est, theta) < 1e-3
    comps = jennrich(third_moment_tensor_1d(y, 0.0), d, generator=_gen(6))
    assert comps.shape == (d, d)


def test_em_recovers_signal_at_high_snr_1d_and_2d() -> None:
    theta = synthetic_template((24,), generator=_gen(7))
    sigma = sigma_for_snr(theta, 50.0)
    y, _, _ = sample_mra(theta, 400, sigma, generator=_gen(8))
    res = em_mra(y, sigma, n_iter=60, generator=_gen(9))
    assert reconstruction_snr(res.theta, theta) > 50
    assert res.log_likelihood[-1] >= res.log_likelihood[0] - 1e-3

    img = synthetic_template((12, 12), generator=_gen(10))
    sigma2 = sigma_for_snr(img, 50.0)
    y2, _, _ = sample_mra(img, 300, sigma2, generator=_gen(11), rotations=True)
    res2 = em_mra(y2, sigma2, n_iter=60, rotations=True, generator=_gen(12))
    assert reconstruction_snr(res2.theta, img, rotations=True) > 30


def test_em_error_decreases_with_more_samples() -> None:
    theta = synthetic_template((16,), generator=_gen(13))
    sigma = sigma_for_snr(theta, 1.0)
    errs = []
    for n in (200, 3200):
        y, _, _ = sample_mra(theta, n, sigma, generator=_gen(14))
        errs.append(rho(em_mra(y, sigma, n_iter=80, init=theta).theta, theta))
    assert errs[1] < errs[0]


def test_specter_projection_template_smoke() -> None:
    """Physics-free single-view projection through SPECTER's ImageGenerator."""
    pytest.importorskip("specter.imagegenerator", reason="SPECTER venv not synced")
    n = 16
    z, yy, xx = torch.meshgrid(*[torch.arange(n) - n / 2] * 3, indexing="ij")
    blob = torch.exp(-((z + 2) ** 2 + yy**2 + (xx - 3) ** 2) / 6.0)
    blob = blob + torch.exp(-((z - 3) ** 2 + (yy + 4) ** 2 + xx**2) / 4.0)
    theta = template_from_volume(blob, pixel_size=2.0)
    assert theta.shape == (n, n)
    assert torch.isfinite(theta).all()
    assert torch.isclose(theta.norm(), torch.tensor(1.0))
    # the projection of a Gaussian blob peaks where the blob is, not at the box edge
    peak = torch.unravel_index(theta.argmax(), theta.shape)
    assert 2 < int(peak[0]) < n - 2 and 2 < int(peak[1]) < n - 2
