"""
The Fourier resampler that carries the propagation grid, tested against what
it claims to be: the band-limited interpolant of the coarse samples, inverted
exactly by the crop, and linear, so that a real field stays real.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter.scattering._kernels import fourier_crop_xy, fourier_upsample_xy


def _field(n: int, factor: int) -> torch.Tensor:
    """
    A band-limited field of the coarse ``n x n`` grid, sampled ``factor``
    times finer.

    Two plane waves below the coarse Nyquist and, on an even grid, a cosine
    at the Nyquist frequency itself along each axis, the term that the shared
    Nyquist bin exists for. Coordinates are in coarse pixels, so ``factor=1``
    gives the coarse samples.
    """
    t = torch.arange(n * factor, dtype=torch.float64) / factor
    y, x = torch.meshgrid(t, t, indexing="ij")
    field = torch.cos(2 * math.pi * (2 * x + y) / n + 0.3)
    field = field + 0.5 * torch.sin(2 * math.pi * (-3 * x + 2 * y) / n + 1.1)
    if n % 2 == 0:
        field = field + 0.25 * torch.cos(math.pi * x) + 0.25 * torch.cos(math.pi * y)
    return field


@pytest.mark.parametrize("n", [8, 9])
@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_upsampling_is_band_limited_interpolation(
    n: int, factor: int, dtype: torch.dtype
) -> None:
    """
    The fine grid holds the analytic field, Nyquist cosine included.

    The complex copy is the case that needs the shared Nyquist bin: a real
    input returns the real part of its transform, which recovers the cosine
    even from a one-sided Nyquist coefficient, while a complex input keeps
    the spurious imaginary part.
    """
    fine = fourier_upsample_xy(_field(n, 1).to(dtype), factor)
    assert torch.allclose(fine, _field(n, factor).to(dtype), atol=1e-12)


@pytest.mark.parametrize("n", [8, 9])
@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_cropping_inverts_upsampling(n: int, factor: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, n, n, dtype=dtype, generator=generator)
    back = fourier_crop_xy(fourier_upsample_xy(x, factor), n)
    assert torch.allclose(back, x, atol=1e-12)


@pytest.mark.parametrize("n", [8, 9])
@pytest.mark.parametrize("factor", [2, 3])
def test_upsampling_is_linear_and_keeps_a_real_field_real(n: int, factor: int) -> None:
    """
    ``up(x + i y) == up(x) + i up(y)`` for real `x` and `y`.

    A real input returns the real part of its transform, so the identity
    holds only if that real part is the whole result: an even grid's Nyquist
    bin, put on one side of the padded spectrum alone, gives a real field an
    imaginary part, and an absorptive wave a different interpolation from a
    plain one.
    """
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, n, n, dtype=torch.float64, generator=generator)
    y = torch.randn(2, n, n, dtype=torch.float64, generator=generator)
    combined = fourier_upsample_xy(torch.complex(x, y), factor)
    separate = fourier_upsample_xy(x, factor) + 1j * fourier_upsample_xy(y, factor)
    assert torch.allclose(combined, separate, atol=1e-12)
    imaginary = fourier_upsample_xy(x.to(torch.complex128), factor).imag
    assert float(imaginary.abs().max()) < 1e-12
