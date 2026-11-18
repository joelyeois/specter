"""
Band-limited resampling of a square ``(..., Y, X)`` grid in Fourier space:
zero-padding the spectrum onto a finer grid, and cropping it back.
"""

from __future__ import annotations

import functools

import torch


def _even_nyquist(n: int) -> bool:
    return n % 2 == 0


@functools.lru_cache(maxsize=16)
def _nyquist_weights(
    n: int, columns: int, scale: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    ``(n, columns)`` weights that share an even grid's Nyquist bins, times `scale`.

    An even-length axis holds its Nyquist frequency once, at index ``n // 2``
    (the negative end). Zero-padding that spectrum leaves the bin on one side
    only, and the interpolant of a real signal then acquires an imaginary
    part (and ``up(i x) != i up(x)``). Halving the bin, and writing it at
    both ``-N/2`` and ``+N/2`` (:func:`_pad_middle`), restores Hermitian
    symmetry, so the interpolation is linear and real-preserving, and the
    matching crop (:func:`_crop_middle`) inverts it exactly. `columns` is
    ``n`` for a full spectrum and ``n // 2 + 1`` for a half spectrum, whose
    Nyquist column is the same index.
    """
    w = torch.full((n, columns), scale, device=device, dtype=dtype)
    if _even_nyquist(n):
        w[n // 2, :] *= 0.5
        w[:, n // 2] *= 0.5
    return w


def _pad_middle(X: torch.Tensor, n: int, m: int, dim: int) -> torch.Tensor:
    """
    Zero-pad an unshifted spectrum axis from `n` to `m` bins at its high end.

    The non-negative frequencies stay at the start and the negative ones move
    to the end, with no ``fftshift`` round trip. An even `n` writes its
    Nyquist bin at both ``+N/2`` and ``-N/2``; the caller has halved it.
    """
    top = n // 2 + 1 if _even_nyquist(n) else (n + 1) // 2
    bottom = n - (n + 1) // 2
    shape = list(X.shape)
    shape[dim] = m - top - bottom
    zeros = X.new_zeros(shape)
    return torch.cat(
        [X.narrow(dim, 0, top), zeros, X.narrow(dim, n - bottom, bottom)], dim
    )


def _crop_middle(X: torch.Tensor, n: int, m: int, dim: int) -> torch.Tensor:
    """Inverse of :func:`_pad_middle`: keep `n` of `m` bins, folding ``+N/2`` onto ``-N/2``."""
    if not _even_nyquist(n):
        return torch.cat(
            [X.narrow(dim, 0, (n + 1) // 2), X.narrow(dim, m - n // 2, n // 2)], dim
        )
    h = n // 2
    nyquist = X.narrow(dim, m - h, 1) + X.narrow(dim, h, 1)
    return torch.cat(
        [X.narrow(dim, 0, h), nyquist, X.narrow(dim, m - h + 1, h - 1)], dim
    )


def fourier_upsample_xy(
    x: torch.Tensor, factor: int, *, half_spectrum: bool = False
) -> torch.Tensor:
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
    half_spectrum : bool, default False
        Resample a real `x` through its half spectrum (``rfft2`` /
        ``irfft2``), so that nothing complex of the fine grid's size is
        formed. It is the same interpolant at a third of the forward cost and
        a fifth of the forward-and-backward cost, and agrees with the full
        transform to float rounding, not bit for bit. The rounding of the
        real transforms also depends on the batch size, which the full
        transform's does not. Ignored for a complex `x`.

    Returns
    -------
    torch.Tensor
        Same dtype as `x`, trailing grid ``factor`` times larger.
    """
    if factor == 1:
        return x
    n = x.shape[-1]
    m = n * factor
    real = not x.is_complex()
    if real and half_spectrum:
        # The columns are padded by the inverse transform itself.
        # contiguous(): MKL's real transform rejects an expanded (stride-0) batch.
        X = torch.fft.rfft2(x.contiguous())
        X = X * _nyquist_weights(n, n // 2 + 1, float(factor**2), x.device, x.dtype)
        return torch.fft.irfft2(_pad_middle(X, n, m, -2), s=(m, m))
    X = torch.fft.fft2(x)
    X = X * _nyquist_weights(n, n, 1.0, x.device, x.real.dtype)
    fine = torch.fft.ifft2(_pad_middle(_pad_middle(X, n, m, -2), n, m, -1))
    fine = fine * factor**2
    return fine.real if real else fine


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
    X = torch.fft.fft2(psi)
    X = _crop_middle(_crop_middle(X, n, m, -2), n, m, -1)
    out = torch.fft.ifft2(X) * (n / m) ** 2
    return out.real if not psi.is_complex() else out
