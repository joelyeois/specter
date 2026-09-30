"""
Tests for specter.fft.
"""

import pytest
import torch

from specter.fft import (
    fftconvolve,
    fourier_shell_correlation,
    spatial_convolve2d_same,
    spatial_convolve3d_same,
)


def test_fsc_of_a_volume_with_itself_is_one():
    torch.manual_seed(0)
    volume = torch.randn(16, 16, 16)
    k, fsc = fourier_shell_correlation(volume, volume, pixel_size=1.0)

    assert k.shape == fsc.shape
    assert torch.allclose(fsc, torch.ones_like(fsc), atol=1e-5)


def test_fsc_frequency_axis_is_physical():
    """Shell spacing is 1/(N * pixel_size), and shells run past Nyquist out to
    the corners of the Fourier cube (sqrt(3) * Nyquist)."""
    n, pixel_size = 16, 2.0
    volume = torch.randn(n, n, n)
    k, _ = fourier_shell_correlation(volume, volume, pixel_size=pixel_size)

    assert k[0] == 0.0
    assert k[1] == pytest.approx(1.0 / (n * pixel_size))
    nyquist = 1.0 / (2.0 * pixel_size)
    assert k[-1] == pytest.approx(3.0**0.5 * nyquist, rel=0.05)


@pytest.mark.parametrize(
    ("shape_a", "shape_b", "match"),
    [
        ((8, 12, 16), (8, 12, 16), "cubic"),
        ((8, 8, 8), (16, 16, 16), "same shape"),
    ],
    ids=["non_cubic", "mismatched_shapes"],
)
def test_fsc_rejects_invalid_shapes(shape_a, shape_b, match: str):
    """Shells are binned by voxel-index radius, which only matches physical
    frequency for a cubic box. A non-cubic volume must raise rather than
    return a silently wrong k axis. Two volumes of different shapes must
    raise too."""
    with pytest.raises(ValueError, match=match):
        fourier_shell_correlation(
            torch.randn(*shape_a), torch.randn(*shape_b), pixel_size=1.0
        )


@pytest.mark.parametrize(
    ("seed", "volume_shape", "kernel_shape"),
    [
        (0, (2, 12, 14, 16), (5, 5, 5)),
        (1, (1, 10, 11, 13), (4, 4, 4)),
        (2, (3, 8, 8, 8), (1, 3, 3, 3)),
    ],
    ids=["odd_kernel", "even_kernel", "unbatched_kernel_dim"],
)
def test_spatial_convolve3d_same_matches_fftconvolve(seed, volume_shape, kernel_shape):
    """spatial_convolve3d_same must match fftconvolve's 'same' mode for odd
    kernels, even kernels, and a kernel carrying a leading singleton batch dim.

    Even-length kernel axes make fftconvolve's 'same'-mode crop asymmetric
    (see `_centered`); spatial_convolve3d_same must reproduce that exact
    centering, not just a plausible-looking 'same' output."""
    torch.manual_seed(seed)
    volume = torch.randn(*volume_shape)
    kernel = torch.randn(*kernel_shape)

    direct = spatial_convolve3d_same(volume, kernel)
    batched_kernel = kernel if kernel.ndim == 4 else kernel.unsqueeze(0)
    fft = fftconvolve(volume, batched_kernel, mode="same", axes=(-3, -2, -1))

    assert direct.shape == volume.shape
    assert torch.allclose(direct, fft, atol=1e-4)


def test_spatial_convolve3d_same_chunked_matches_unchunked():
    """Forces the Z-chunked path (via a tiny _max_spatial_elements override
    standing in for the real ~2**30 cuDNN-safety threshold) and checks it's
    identical to a single whole-volume conv3d call -- i.e. chunking with
    halo context reproduces the same result as an unchunked call, not just
    a plausible approximation."""
    torch.manual_seed(3)
    volume = torch.randn(2, 17, 6, 5)
    kernel = torch.randn(4, 3, 3)

    unchunked = spatial_convolve3d_same(volume, kernel)
    chunked = spatial_convolve3d_same(volume, kernel, _max_spatial_elements=40)

    assert torch.allclose(chunked, unchunked, atol=1e-5)


@pytest.mark.parametrize("ky", [3, 4, 5, 6])
def test_spatial_convolve2d_same_matches_fftconvolve(ky):
    """
    The 2D direct convolution must reproduce fftconvolve exactly.

    F.conv2d(padding="same"), which this replaced, pads asymmetrically for an
    even-sized kernel and computes cross-correlation rather than convolution.
    Both discrepancies matter here, so the kernel is deliberately asymmetric:
    a radially symmetric one (like an atomic potential) would hide the missing
    flip and expose only the padding.
    """
    torch.manual_seed(0)
    images = torch.rand(3, 20, 22)
    kernel = torch.rand(ky, ky + 1)

    expected = torch.stack(
        [fftconvolve(images[b], kernel, mode="same") for b in range(len(images))]
    )
    torch.testing.assert_close(
        spatial_convolve2d_same(images, kernel), expected, atol=1e-5, rtol=1e-5
    )
