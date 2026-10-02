"""Smoke and correctness tests for the MRA image-generation toolkit (run: pytest mj_dsa4288/tests -v)."""

from __future__ import annotations

import pytest
import torch

from mj_dsa4288.mra.model import (
    cyclic_shift,
    inverse_cyclic_shift,
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
