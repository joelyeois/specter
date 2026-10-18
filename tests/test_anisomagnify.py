"""
Anisotropic magnification is a resampling of each image through its own spectrum.

The reference is the analytic warp of a smooth blob, ``img(A (x - c) + c)``
about ``c = (N - 1) / 2``, with the matrix read in ``affine_grid``'s
normalised coordinates as before: the warp must reproduce it to the NUFFT's
accuracy, keep that convention on a non-square image, preserve the amplitude
of a frequency next to Nyquist (which a bilinear resampling damps), and be
its own adjoint's transpose so a reconstruction's gradient is right.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from specter.microscope import Detector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
A_PLAIN = torch.tensor([[1.0045, 0.0008], [0.0009, 0.9955]], dtype=torch.float64)


def _detector() -> Detector:
    return Detector(pixel_size=1.0)


def _blob(h: int, w: int, A: torch.Tensor | None = None) -> torch.Tensor:
    """A smooth off-centre blob, optionally warped analytically by ``A`` (normalised coordinates)."""
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )
    c = torch.tensor([(w - 1) / 2, (h - 1) / 2], dtype=torch.float64)
    u = torch.stack([xx, yy], dim=-1) - c
    if A is not None:
        D = torch.tensor([w / 2, h / 2], dtype=torch.float64)
        u = u @ (D[:, None] * A / D[None, :]).T
    x, y = (u + c).unbind(-1)
    return 10 + torch.exp(-((x - 0.62 * w) ** 2 / 60 + (y - 0.41 * h) ** 2 / 25))


def test_identity_returns_the_image() -> None:
    img = _blob(64, 64)[None].float().to(DEVICE)
    out = _detector().anisomagnify(img, torch.eye(2)[None].to(DEVICE))
    assert torch.allclose(out, img, atol=1e-5 * float(img.abs().max()))


@pytest.mark.parametrize("shape", [(96, 96), (80, 112)])
def test_matches_the_analytic_warp(shape: tuple[int, int]) -> None:
    h, w = shape
    img = _blob(h, w)[None].to(DEVICE)
    out = _detector().anisomagnify(img, A_PLAIN[None].to(DEVICE)).cpu()[0]
    truth = _blob(h, w, A_PLAIN)
    fluct = truth - truth.mean()
    assert float((out - truth).norm() / fluct.norm()) < 1e-5


def test_keeps_the_convention_of_the_bilinear_warp_it_replaced() -> None:
    """Same matrix meaning and centre as affine_grid + grid_sample, on a blob smooth enough to agree."""
    img = _blob(96, 96)[None].float()
    theta = torch.zeros(1, 2, 3)
    theta[:, :, :2] = A_PLAIN.float()
    grid = F.affine_grid(theta, [1, 1, 96, 96], align_corners=False)
    bilinear = F.grid_sample(
        img[:, None], grid, align_corners=False, padding_mode="border"
    )[:, 0]
    out = (
        _detector().anisomagnify(img.to(DEVICE), A_PLAIN[None].float().to(DEVICE)).cpu()
    )
    fluct = bilinear - bilinear.mean()
    assert float((out - bilinear).norm() / fluct.norm()) < 5e-3


def test_preserves_amplitude_next_to_nyquist() -> None:
    """A 0.453 cycle/px fringe comes out at full amplitude; bilinear resampling damps it up to ~6x."""
    n, cycles = 128, 58  # a whole number of periods, so the image is periodic
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    img = (10 + torch.cos(2 * math.pi * cycles / n * xx))[None].to(DEVICE)
    out = _detector().anisomagnify(img, A_PLAIN[None].to(DEVICE)).cpu()[0]
    c = (n - 1) / 2
    u = torch.stack([xx - c, yy - c], dim=-1) @ A_PLAIN.T + c
    truth = 10 + torch.cos(2 * math.pi * cycles / n * u[..., 0])
    centre = (slice(n // 4, 3 * n // 4),) * 2  # away from where the warp wraps
    assert float((out - truth)[centre].abs().max()) < 1e-3


def test_is_linear_with_a_consistent_adjoint() -> None:
    """<W x, g> == <x, W^T g>: the gradient a reconstruction takes through the warp is the true adjoint."""
    g = torch.Generator().manual_seed(0)
    x = (
        torch.randn(2, 48, 48, generator=g, dtype=torch.float64)
        .to(DEVICE)
        .requires_grad_(True)
    )
    r = torch.randn(2, 48, 48, generator=g, dtype=torch.float64).to(DEVICE)
    A = torch.stack([A_PLAIN, A_PLAIN.T]).to(DEVICE)
    y = _detector().anisomagnify(x, A)
    (gx,) = torch.autograd.grad((y * r).sum(), x)
    lhs, rhs = float((y * r).sum()), float((x * gx).sum())
    assert abs(lhs - rhs) < 1e-8 * max(abs(lhs), 1.0)


def test_a_batch_warps_each_image_by_its_own_matrix() -> None:
    """Images sharing a matrix are transformed together; the result is per image regardless."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 40, 40, generator=g, dtype=torch.float64).to(DEVICE)
    A = torch.stack(
        [A_PLAIN, A_PLAIN.T, A_PLAIN, torch.eye(2, dtype=torch.float64)]
    ).to(DEVICE)
    det = _detector()
    together = det.anisomagnify(x, A)
    alone = torch.cat([det.anisomagnify(x[i : i + 1], A[i : i + 1]) for i in range(4)])
    assert torch.allclose(together, alone, atol=1e-10)
