"""Tests for `specter.filters`."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from specter.filters import gaussian_blur3d


def _blur_reference(V: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """The cudnn conv3d, one-axis-at-a-time formulation this replaced."""
    r = max(1, int(round(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, device=V.device, dtype=V.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    kernel = kernel / kernel.sum()
    lead = V.shape[:-3]
    out = V.reshape(-1, 1, *V.shape[-3:])
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = -1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), kernel.view(*shape))
    return out.reshape(*lead, *V.shape[-3:])


@pytest.mark.parametrize("shape", [(7, 9, 11), (2, 6, 8, 5), (2, 1, 5, 5, 5)])
def test_gaussian_blur3d_matches_conv3d_reference(shape):
    torch.manual_seed(0)
    V = torch.rand(*shape, dtype=torch.float64) * 7
    got = gaussian_blur3d(V, 2.0 / 0.731)
    want = _blur_reference(V, 2.0 / 0.731)
    assert got.shape == V.shape
    assert torch.allclose(got, want, atol=1e-12, rtol=0)
