"""
The image-domain loss for reconstruction: the mean squared error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(out: torch.Tensor, images: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Real-space mean squared error with a per-particle weight.

    Parameters
    ----------
    out : torch.Tensor
        Simulated images, shape ``(B, H, W)``.
    images : torch.Tensor
        Experimental images, shape ``(B, H, W)``.
    w : torch.Tensor
        Per-particle loss weight, shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    mse = F.mse_loss(images, out, reduction="none")
    return (w[:, None, None] * mse).mean()
