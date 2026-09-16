"""
Method of invariants (Perry et al. 2019, Section 2.2 and eq. (3)).

Unbiased estimators of the shift-invariant moment tensors of ``theta``:

* ``T^(1)``: the mean of the signal,
* ``T^(2)``: the autocorrelation, whose Fourier transform is the power spectrum,
* ``T^(3)``: the third moment tensor, whose Fourier transform is the bispectrum.

The bias corrections assume white Gaussian noise with *known* ``sigma``.
Note that eq. (3) of the paper omits the ``sigma^2`` factor in the
``3 sym(y (x) I)`` correction; the proof of Lemma 1 makes clear it must be
``3 sigma^2 sym(...)``, which is what is implemented here.
"""

from __future__ import annotations

import torch


def mean_invariant(y: torch.Tensor) -> float:
    """Estimate the mean of ``theta`` from observations ``y`` (batch first)."""
    return float(y.mean())


def power_spectrum(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Unbiased estimate of the power spectrum ``|theta_hat_k|^2`` (ortho-normalised FFT).

    Parameters
    ----------
    y : torch.Tensor
        Observations, ``[n, d]`` or ``[n, H, W]``.
    sigma : float
        Known noise standard deviation.

    Returns
    -------
    torch.Tensor
        ``mean_i |FFT(y_i)|^2 - sigma^2``, same trailing shape as one sample.
    """
    dims = tuple(range(1, y.ndim))
    yf = torch.fft.fftn(y, dim=dims, norm="ortho")
    return yf.abs().pow(2).mean(0) - sigma**2


def bispectrum_1d(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Unbiased estimate of the bispectrum ``B(k1, k2) = Y_k1 Y_k2 conj(Y_{k1+k2})``.

    Uses the unnormalised FFT ``Y_k = sum_j theta_j exp(-2 pi i jk/d)``.  For
    white Gaussian noise the only biased entries are those with ``k1 = 0``,
    ``k2 = 0`` or ``k1 + k2 = 0 (mod d)``, each biased by ``d^2 sigma^2 mu``
    where ``mu`` is the mean of ``theta`` (estimated from the data, which keeps
    the estimator linear in the noise and hence unbiased).

    Parameters
    ----------
    y : torch.Tensor
        Observations ``[n, d]``.
    sigma : float
        Known noise standard deviation.

    Returns
    -------
    torch.Tensor
        Complex ``[d, d]`` bispectrum estimate.
    """
    n, d = y.shape
    yf = torch.fft.fft(y, dim=1)
    k = torch.arange(d, device=y.device)
    k12 = (k[:, None] + k[None, :]) % d
    b = torch.zeros((d, d), dtype=yf.dtype, device=y.device)
    for start in range(0, n, 4096):  # bounded memory: [batch, d, d]
        chunk = yf[start : start + 4096]
        b += (chunk[:, :, None] * chunk[:, None, :] * chunk[:, k12].conj()).sum(0)
    b /= n
    mu = mean_invariant(y)
    mask = (k[:, None] == 0).float() + (k[None, :] == 0).float() + (k12 == 0).float()
    return b - d**2 * sigma**2 * mu * mask


def third_moment_tensor_1d(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Unbiased estimate of ``T^(3)(theta) = (1/d) sum_l (R_l theta)^{(x)3}`` (eq. (2)/(3)).

    Computed through the bispectrum: averaging ``y^{(x)3}`` over all cyclic
    shifts gives a tensor that depends only on index differences,
    ``T[a, b, c] = C[b - a, c - a]``, where the triple correlation ``C`` is the
    inverse 2-D FFT of the bispectrum divided by ``d``.

    Parameters
    ----------
    y : torch.Tensor
        Observations ``[n, d]``.
    sigma : float
        Known noise standard deviation.

    Returns
    -------
    torch.Tensor
        Real ``[d, d, d]`` third-moment tensor estimate.
    """
    d = y.shape[1]
    b = bispectrum_1d(y, sigma)
    c = torch.fft.ifft2(b).real / d  # C[s, t] = (1/d) sum_a y_a y_{a+s} y_{a+t}
    a = torch.arange(d, device=y.device)
    s = (a[None, :, None] - a[:, None, None]) % d  # b - a
    t = (a[None, None, :] - a[:, None, None]) % d  # c - a
    return c[s, t]


def third_moment_tensor_1d_bruteforce(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Literal implementation of eq. (3) (with the sigma^2 fix). For tests only."""
    n, d = y.shape
    eye = torch.eye(d, device=y.device)
    t3 = torch.zeros((d, d, d), device=y.device)
    for i in range(n):
        for ell in range(d):
            v = torch.roll(y[i], -ell)
            t3 += torch.einsum("a,b,c->abc", v, v, v)
            sym = (
                torch.einsum("a,bc->abc", v, eye)
                + torch.einsum("b,ac->abc", v, eye)
                + torch.einsum("c,ab->abc", v, eye)
            ) / 3
            t3 -= 3 * sigma**2 * sym
    return t3 / (n * d)
