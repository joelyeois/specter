"""
Shift-invariant error metrics and slope fitting.

``rho`` is the paper's distance ``rho(theta, tau) = min_l ||theta - R_l tau||_2``
(Section 3.1), extended to 2-D shifts and, optionally, 90-degree rotations.
"""

from __future__ import annotations

import numpy as np
import torch


def _correlations(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``c[l] = sum_j a[j] b[j + l]`` for all cyclic shifts ``l`` (1-D or 2-D)."""
    af = torch.fft.fftn(a)
    bf = torch.fft.fftn(b)
    return torch.fft.ifftn(af.conj() * bf).real


def _rot_candidates(tau: torch.Tensor, rotations: bool) -> list[torch.Tensor]:
    if rotations and tau.ndim == 2:
        return [torch.rot90(tau, k) for k in range(4)]
    return [tau]


def align_to(
    estimate: torch.Tensor, reference: torch.Tensor, rotations: bool = False
) -> torch.Tensor:
    """
    Cyclically shift (and optionally rotate) ``estimate`` to best match ``reference``.

    Parameters
    ----------
    estimate : torch.Tensor
        Signal ``[d]`` or image ``[H, W]``.
    reference : torch.Tensor
        Same shape as ``estimate``.
    rotations : bool, optional
        Also search over rotations by multiples of 90 degrees (2-D only).

    Returns
    -------
    torch.Tensor
        The transformed ``estimate`` minimising ``||reference - R estimate||``.
    """
    best: torch.Tensor | None = None
    best_val = -float("inf")
    for cand in _rot_candidates(estimate, rotations):
        c = _correlations(reference, cand)  # c[l] = <reference, R_l cand>
        val, flat = c.max(), int(c.argmax())
        if float(val) > best_val:
            best_val = float(val)
            idx = np.unravel_index(flat, cand.shape)
            best = torch.roll(
                cand, shifts=tuple(-int(i) for i in idx), dims=tuple(range(cand.ndim))
            )
    assert best is not None
    return best


def rho(
    estimate: torch.Tensor, reference: torch.Tensor, rotations: bool = False
) -> float:
    """Shift-invariant distance ``min_l ||reference - R_l estimate||_2``."""
    aligned = align_to(estimate, reference, rotations=rotations)
    return float((aligned - reference).norm())


def reconstruction_snr(
    estimate: torch.Tensor, reference: torch.Tensor, rotations: bool = False
) -> float:
    """
    Reconstruction SNR ``||theta||^2 / rho(theta_tilde, theta)^2``.

    This is the ratio of signal energy to residual error energy after optimal
    alignment. Sigworth (1998, Fig. 2C) normalises differently by constants,
    which shifts the curve vertically but leaves the log-log slopes unchanged.
    """
    err = rho(estimate, reference, rotations=rotations)
    return float(reference.pow(2).sum()) / max(err**2, 1e-30)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of ``log10(y)`` against ``log10(x)``."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def two_regime_slopes(
    x: np.ndarray, y: np.ndarray, split: float
) -> tuple[float, float]:
    """
    Log-log slopes below and above ``split`` (the paper predicts 3 then 1).

    Parameters
    ----------
    x : np.ndarray
        Data SNR values.
    y : np.ndarray
        Reconstruction SNR (or ``1/n_required``) values.
    split : float
        SNR at which to split the two regimes.

    Returns
    -------
    tuple[float, float]
        ``(low_snr_slope, high_snr_slope)``; ``nan`` if a regime has < 2 points.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = x < split, x >= split
    low = fit_loglog_slope(x[lo], y[lo])[0] if lo.sum() >= 2 else float("nan")
    high = fit_loglog_slope(x[hi], y[hi])[0] if hi.sum() >= 2 else float("nan")
    return low, high
