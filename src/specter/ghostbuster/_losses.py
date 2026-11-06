"""
The image-domain loss for reconstruction: the mean squared error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(out: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """
    Real-space mean squared error.

    Parameters
    ----------
    out : torch.Tensor
        Simulated images, shape ``(B, H, W)``.
    images : torch.Tensor
        Experimental images, shape ``(B, H, W)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return F.mse_loss(images, out, reduction="none").mean()
