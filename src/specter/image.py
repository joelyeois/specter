"""
Image-level utilities for particle stacks.
"""

from __future__ import annotations

import torch
from typing import Literal


def normalize_particles(
    particles: torch.Tensor,
    mask_diameter: float | None = None,
    mode: Literal["mean", "median"] = "mean",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized per-particle normalization of a 2D particle stack.

    Parameters
    ----------
    particles : torch.Tensor, shape (N, H, W)
        Stack of N 2D particles.
    mask_diameter : float, optional
        Diameter of circular mask in pixels to exclude from mean/std calculation.
        If None, uses circular mask with diameter of image.
    mode : str
        'mean' or 'median' for centering. Default is 'mean'.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    normalized : torch.Tensor, shape (N, H, W)
        Normalized particle stack.
    means : torch.Tensor, shape (N,)
        Mean or median of each particle used for normalization.
    stds : torch.Tensor, shape (N,)
        Standard deviation of each particle used for normalization.
    """
    N, H, W = particles.shape
    device = particles.device

    # Build mask
    if mask_diameter is None:
        radius = W / 2
    else:
        radius = mask_diameter / 2
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    cy, cx = H // 2, W // 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2) >= radius**2  # True for pixels to include

    # Flatten masked pixels
    mask_flat = mask.view(-1)
    particles_flat = particles.view(N, -1)[:, mask_flat]  # (N, masked_pixels)

    # Compute mean / median per particle
    if mode == "mean":
        means = particles_flat.mean(dim=1)
    elif mode == "median":
        means = particles_flat.median(dim=1).values
    else:
        raise ValueError("mode must be 'mean' or 'median'")

    # Compute std per particle
    stds = particles_flat.std(dim=1, unbiased=False)

    # Normalize
    normalized = (particles - means[:, None, None]) / (stds[:, None, None] + eps)

    return normalized, means, stds
