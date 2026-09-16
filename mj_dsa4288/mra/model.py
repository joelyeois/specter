"""
The MRA forward model of Perry et al. (2019), eq. (1):  y_i = R_{l_i} theta + sigma xi_i.

``R_l`` is a cyclic shift, ``(R_l theta)[j] = theta[(j + l) mod d]``, and
``xi_i ~ N(0, I)``.  Everything here works for 1-D signals (the paper's
setting, ``theta`` of shape ``[d]``) and for 2-D images (Sigworth's setting,
``theta`` of shape ``[H, W]``), where the shift is a 2-D cyclic shift and an
optional latent 90-degree rotation can be switched on.
"""

from __future__ import annotations

from typing import Literal

import torch

SNRConvention = Literal["paper", "pixel"]


def cyclic_shift(theta: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """
    Apply a per-sample cyclic shift to a 1-D signal or 2-D image.

    Parameters
    ----------
    theta : torch.Tensor
        Signal of shape ``[d]`` or image of shape ``[H, W]``.
    shifts : torch.Tensor
        Integer shifts, shape ``[n]`` (1-D) or ``[n, 2]`` (2-D, row then column).

    Returns
    -------
    torch.Tensor
        Shifted copies, shape ``[n, d]`` or ``[n, H, W]``, with
        ``out[i, j] = theta[(j + shifts[i]) mod d]``.
    """
    shifts = shifts.to(theta.device)
    if theta.ndim == 1:
        d = theta.shape[0]
        idx = (torch.arange(d, device=theta.device)[None, :] + shifts[:, None]) % d
        return theta[idx]
    if theta.ndim == 2:
        h, w = theta.shape
        iy = (torch.arange(h, device=theta.device)[None, :] + shifts[:, 0:1]) % h
        ix = (torch.arange(w, device=theta.device)[None, :] + shifts[:, 1:2]) % w
        return theta[iy[:, :, None], ix[:, None, :]]
    raise ValueError("theta must be 1-D or 2-D")


def inverse_cyclic_shift(y: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Undo :func:`cyclic_shift` sample by sample (``y`` has a leading batch dim)."""
    if y.ndim == 2:
        d = y.shape[1]
        idx = (torch.arange(d, device=y.device)[None, :] - shifts[:, None]) % d
        return torch.gather(y, 1, idx)
    h, w = y.shape[1:]
    iy = (torch.arange(h, device=y.device)[None, :] - shifts[:, 0:1]) % h
    ix = (torch.arange(w, device=y.device)[None, :] - shifts[:, 1:2]) % w
    b = torch.arange(y.shape[0], device=y.device)[:, None, None]
    return y[b, iy[:, :, None], ix[:, None, :]]


def signal_power(theta: torch.Tensor, convention: SNRConvention = "paper") -> float:
    """
    Signal power under a given SNR convention.

    ``"paper"`` uses ``||theta||_2^2`` (Perry et al.: SNR = ||theta||^2 / sigma^2,
    with ||theta|| = 1 so SNR = 1/sigma^2).  ``"pixel"`` uses the per-pixel
    variance of ``theta`` (Sigworth 1998: signal variance over noise variance).
    The two differ by roughly a factor ``d``; pick one and report it.
    """
    if convention == "paper":
        return float(theta.pow(2).sum())
    if convention == "pixel":
        return float(theta.var(unbiased=False))
    raise ValueError(f"unknown SNR convention {convention!r}")


def sigma_for_snr(
    theta: torch.Tensor, snr: float, convention: SNRConvention = "paper"
) -> float:
    """Noise standard deviation that gives ``snr`` for ``theta`` under ``convention``."""
    return (signal_power(theta, convention) / snr) ** 0.5


def snr_of(
    theta: torch.Tensor, sigma: float, convention: SNRConvention = "paper"
) -> float:
    """SNR of ``theta`` corrupted by white noise of standard deviation ``sigma``."""
    return signal_power(theta, convention) / sigma**2


def sample_mra(
    theta: torch.Tensor,
    n: int,
    sigma: float,
    generator: torch.Generator | None = None,
    rotations: bool = False,
    max_shift: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw ``n`` samples from the MRA model with uniform latent shifts.

    Parameters
    ----------
    theta : torch.Tensor
        Clean signal ``[d]`` or image ``[H, W]``.
    n : int
        Number of observations.
    sigma : float
        Noise standard deviation (per coordinate).
    generator : torch.Generator, optional
        RNG for reproducibility.
    rotations : bool, optional
        For 2-D templates, also apply a latent rotation by a multiple of 90
        degrees before the shift (Sigworth's 2-D MRA). Default False.
    max_shift : int, optional
        If given, shifts are uniform in ``[-max_shift, max_shift]`` rather than
        over the whole period. Useful for the non-cyclic Phase B comparison.

    Returns
    -------
    y : torch.Tensor
        Observations ``[n, d]`` or ``[n, H, W]``.
    shifts : torch.Tensor
        Latent shifts ``[n]`` or ``[n, 2]``.
    rots : torch.Tensor
        Latent rotation index in ``{0, 1, 2, 3}`` per sample (all zero if
        ``rotations`` is False or ``theta`` is 1-D).
    """
    dev = theta.device
    shape = theta.shape
    ndim = theta.ndim
    if max_shift is None:
        shifts = torch.stack(
            [torch.randint(0, s, (n,), generator=generator) for s in shape], dim=-1
        )
    else:
        shifts = torch.randint(
            -max_shift, max_shift + 1, (n, ndim), generator=generator
        )
    shifts = shifts.to(dev)
    rots = torch.zeros(n, dtype=torch.long, device=dev)
    if rotations and ndim == 2:
        rots = torch.randint(0, 4, (n,), generator=generator).to(dev)
        clean = torch.empty((n, *shape), device=dev, dtype=theta.dtype)
        for k in range(4):
            mask = rots == k
            if mask.any():
                clean[mask] = cyclic_shift(torch.rot90(theta, k), shifts[mask])
    else:
        clean = cyclic_shift(theta, shifts.squeeze(-1) if ndim == 1 else shifts)
    noise = torch.randn(clean.shape, generator=generator).to(dev)
    y = clean + sigma * noise
    return y, (shifts.squeeze(-1) if ndim == 1 else shifts), rots
