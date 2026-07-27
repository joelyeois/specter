"""
Tests for specter.fft.
"""

import torch

from specter.fft import fftconvolve, spatial_convolve3d_same


def test_spatial_convolve3d_same_matches_fftconvolve_odd_kernel():
    torch.manual_seed(0)
    volume = torch.randn(2, 12, 14, 16)
    kernel = torch.randn(5, 5, 5)

    direct = spatial_convolve3d_same(volume, kernel)
    fft = fftconvolve(volume, kernel.unsqueeze(0), mode="same", axes=(-3, -2, -1))

    assert direct.shape == volume.shape
    assert torch.allclose(direct, fft, atol=1e-4)


def test_spatial_convolve3d_same_matches_fftconvolve_even_kernel():
    """Even-length kernel axes make fftconvolve's 'same'-mode crop
    asymmetric (see `_centered`); spatial_convolve3d_same must reproduce
    that exact centering, not just a plausible-looking 'same' output."""
    torch.manual_seed(1)
    volume = torch.randn(1, 10, 11, 13)
    kernel = torch.randn(4, 4, 4)

    direct = spatial_convolve3d_same(volume, kernel)
    fft = fftconvolve(volume, kernel.unsqueeze(0), mode="same", axes=(-3, -2, -1))

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


def test_spatial_convolve3d_same_accepts_unbatched_kernel_dim():
    torch.manual_seed(2)
    volume = torch.randn(3, 8, 8, 8)
    kernel = torch.randn(1, 3, 3, 3)

    direct = spatial_convolve3d_same(volume, kernel)
    fft = fftconvolve(volume, kernel, mode="same", axes=(-3, -2, -1))

    assert torch.allclose(direct, fft, atol=1e-4)
