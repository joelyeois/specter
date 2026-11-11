"""
Band-limited resampling of a square ``(..., Y, X)`` grid in Fourier space:
zero-padding the spectrum onto a finer grid, and cropping it back.
"""

from __future__ import annotations

import torch


def _split_nyquist(X: torch.Tensor, index: int, n: int, dim: int) -> torch.Tensor:
    """
    Share an even grid's single Nyquist bin between ``-N/2`` and ``+N/2``.

    After ``fftshift``, an even-length axis holds its Nyquist frequency once,
    at the negative end. Zero-padding that spectrum leaves the bin on one
    side only, and the interpolant of a real signal then acquires an
    imaginary part (and ``up(i x) != i up(x)``). Halving the bin onto both
    ends restores Hermitian symmetry, so the interpolation is linear and
    real-preserving, and the matching crop (:func:`_merge_nyquist`) inverts
    it exactly.
    """
    if n % 2:
        return X
    half = X.narrow(dim, index, 1) * 0.5
    X = X.clone()
    X.narrow(dim, index, 1).copy_(half)
    X.narrow(dim, index + n, 1).copy_(half)
    return X


def _merge_nyquist(X: torch.Tensor, index: int, n: int, dim: int) -> torch.Tensor:
    """Inverse of :func:`_split_nyquist`: fold ``+N/2`` back onto ``-N/2``."""
    if n % 2:
        return X
    X = X.clone()
    X.narrow(dim, index, 1).add_(X.narrow(dim, index + n, 1))
    return X


def fourier_upsample_xy(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Resample ``(..., Y, X)`` onto a grid `factor` times finer, adding no content.

    The spectrum is zero-padded about its centre, so the fine-grid samples
    are the band-limited interpolant of the coarse ones: every frequency the
    coarse grid holds is kept at its value and nothing is added above the
    coarse Nyquist. Values (not integrals) are preserved. An even grid's
    Nyquist bin is shared between ``-N/2`` and ``+N/2``, which keeps the
    operation linear and a real input real; :func:`fourier_crop_xy` inverts
    it exactly.

    Parameters
    ----------
    x : torch.Tensor
        Real or complex, any leading shape, square trailing grid.
    factor : int
        Integer upsampling factor, 1 returns `x` itself.

    Returns
    -------
    torch.Tensor
        Same dtype as `x`, trailing grid ``factor`` times larger.
    """
    if factor == 1:
        return x
    n = x.shape[-1]
    m = n * factor
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    # Align the DC bins (index n//2 and m//2 after fftshift) for either parity.
    pad = m // 2 - n // 2
    X = torch.nn.functional.pad(X, (pad, m - n - pad, pad, m - n - pad))
    X = _split_nyquist(_split_nyquist(X, pad, n, -2), pad, n, -1)
    fine = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))) * factor**2
    return fine.real if not x.is_complex() else fine


def fourier_crop_xy(psi: torch.Tensor, n: int) -> torch.Tensor:
    """
    Crop ``(..., M, M)`` complex `psi` to its central ``n x n`` spectrum.

    The inverse of :func:`fourier_upsample_xy`: content beyond the coarse
    Nyquist is discarded, the values of the retained band are preserved, and
    an even output grid's Nyquist bin collects both ``+-N/2`` fine bins.

    Parameters
    ----------
    psi : torch.Tensor
        Field on the fine grid.
    n : int
        Output grid size; ``psi.shape[-1]`` returns `psi` itself.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n, n)``.
    """
    m = psi.shape[-1]
    if m == n:
        return psi
    X = torch.fft.fftshift(torch.fft.fft2(psi), dim=(-2, -1))
    lo = m // 2 - n // 2
    X = _merge_nyquist(_merge_nyquist(X, lo, n, -2), lo, n, -1)
    X = X[..., lo : lo + n, lo : lo + n]
    out = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))) * (n / m) ** 2
    return out.real if not psi.is_complex() else out
