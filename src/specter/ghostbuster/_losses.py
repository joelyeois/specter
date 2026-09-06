"""
Image-domain losses for reconstruction: normalised cross-correlation, MSE,
and the noise-weighted variants with their running noise estimate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..arrays import compute_nps_2d

# ---------------------------------------------------------------------------
# Image-domain loss functions used by Reconstructor._compute_loss
# ---------------------------------------------------------------------------


def ncc_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Per-image normalized cross-correlation loss in MSE-equivalent units.

    Computes ``var(target) * (1 - NCC)`` per image.  The ``var(target)``
    factor makes the loss dimensionally equivalent to MSE: at the optimum
    (low-SNR regime) both quantities converge to the noise variance
    ``σ²``, so the same learning rate can be used without retuning.

    Parameters
    ----------
    pred : torch.Tensor
        Simulated images, shape ``(B, H, W)``.
    target : torch.Tensor
        Experimental images, shape ``(B, H, W)``.
    eps : float
        Small constant added to the denominator for numerical stability.

    Returns
    -------
    torch.Tensor
        Per-image loss, shape ``(B,)``. Callers reduce over the batch
        (optionally weighting by a per-particle scale first).

    Notes
    -----
    NCC is invariant to multiplicative and additive intensity rescaling,
    which makes it robust to gain-reference errors and forward-model scale
    mismatches that inflate ordinary MSE.
    """
    p = pred.flatten(1)  # (B, N)
    t = target.flatten(1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    ncc = (p_c * t_c).sum(dim=1) / (
        p_c.norm(dim=1) * t_c.norm(dim=1) + eps
    )  # (B,), range [-1, 1]
    # var(target) per image: scales (1 - NCC) into MSE-equivalent units
    var_t = (t_c**2).mean(dim=1)  # (B,)
    return var_t * (1.0 - ncc)


def mse_loss(
    out: torch.Tensor,
    images: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Plain real-space MSE, optionally weighted by a 2D mask.

    Parameters
    ----------
    out : torch.Tensor
        Simulated images, shape ``(B, H, W)``.
    images : torch.Tensor
        Experimental images, shape ``(B, H, W)``.
    w : torch.Tensor
        Per-particle loss weight, shape ``(B,)``.
    mask : torch.Tensor, optional
        Per-image weighting mask, shape ``(B, H, W)`` (e.g. a projected 2D
        FSC mask). ``None`` disables masking.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    mse = F.mse_loss(images, out, reduction="none")
    if mask is not None:
        mse = mse * mask
    return (w[:, None, None] * mse).mean()


def nps_weighted_loss(
    out: torch.Tensor,
    images: torch.Tensor,
    nps_weight: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    MSE weighted by the noise power spectrum in Fourier space.

    Parameters
    ----------
    out : torch.Tensor
        Simulated images, shape ``(B, H, W)``.
    images : torch.Tensor
        Experimental images, shape ``(B, H, W)``.
    nps_weight : torch.Tensor
        Per-frequency weight, shape ``(H, W//2+1)``.
    w : torch.Tensor
        Per-particle loss weight, shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    images_f = torch.fft.rfft2(images)
    out_f = torch.fft.rfft2(out)
    H, W = images.shape[-2:]
    # Divide by H*W so that a flat (normalised) NPS weight gives the
    # same loss magnitude as real-space MSE (Parseval equivalence).
    residual = nps_weight * (images_f - out_f).abs() ** 2
    return torch.mean(w[:, None, None] * residual) / (H * W)


def update_sigma2(
    sigma2_k: torch.Tensor, residuals: torch.Tensor, momentum: float
) -> torch.Tensor:
    """Update the per-shell noise variance sigma^2(k) from real-space residuals.

    Mirrors the RELION noise model: the unexplained residuals are
    radially-averaged in Fourier space (via ``compute_nps_2d``) to estimate
    sigma^2(k) per shell, then an EMA smooths the estimate across batches.

    sigma^2(k) is normalised by its mean after each update so that the
    relative spectral weighting adapts while the loss magnitude stays
    stable (comparable to the nps_weight and plain-MSE modes).

    Parameters
    ----------
    sigma2_k : torch.Tensor
        Current per-shell noise variance estimate, shape ``(H, W//2+1)``.
    residuals : torch.Tensor
        Real-space residuals (``images - out``), shape ``(B, H, W)``.
    momentum : float
        EMA momentum in ``[0, 1)``.

    Returns
    -------
    torch.Tensor
        Updated sigma^2(k), same shape as ``sigma2_k``.
    """
    with torch.no_grad():
        # Raw per-shell power spectrum of residuals, shape (H, W//2+1)
        new_sigma2 = compute_nps_2d(
            residuals.detach(), normalize=False, zero_dc=False
        ).clamp(min=1e-10)
        # EMA update
        sigma2_k = momentum * sigma2_k + (1 - momentum) * new_sigma2
        # Normalise by mean so that a flat sigma^2 gives uniform weights
        # (i.e. loss magnitude remains comparable to real-space MSE).
        sigma2_k = sigma2_k / sigma2_k.mean().clamp(min=1e-10)
    return sigma2_k


def noise_weighted_loss(
    out: torch.Tensor,
    images: torch.Tensor,
    sigma2_k: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """RELION-style loss: weight residuals by 1/sigma^2(k).

    Callers must update ``sigma2_k`` (via :func:`update_sigma2`) before
    calling this function; gradient flows only through the residuals
    (M-step), not through ``sigma2_k`` itself (E-step, no_grad).

    Parameters
    ----------
    out : torch.Tensor
        Simulated images, shape ``(B, H, W)``.
    images : torch.Tensor
        Experimental images, shape ``(B, H, W)``.
    sigma2_k : torch.Tensor
        Per-shell noise variance, shape ``(H, W//2+1)``.
    w : torch.Tensor
        Per-particle loss weight, shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    images_f = torch.fft.rfft2(images)
    out_f = torch.fft.rfft2(out)
    H, W = images.shape[-2:]
    residual = (images_f - out_f).abs() ** 2 / sigma2_k.detach()
    return torch.mean(w[:, None, None] * residual) / (H * W)
