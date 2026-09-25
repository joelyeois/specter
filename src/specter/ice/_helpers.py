"""
Small helpers for the ice generators: the number density of amorphous ice
and a few array utilities.
"""

from __future__ import annotations


import torch
import torch.nn.functional as F
from scipy import constants as _sc


avogadro = _sc.Avogadro
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / Å³]


def rfftn(array: torch.Tensor) -> torch.Tensor:
    """
    Compute N-dimensional real-input Fourier transform with centering.

    Wraps torch.fft.rfftn with FFT shifting to ensure the zero-frequency component
    is centered, handling the last dimension which is complex-valued differently.

    Parameters
    ----------
    array : torch.Tensor
        Input real-valued tensor.

    Returns
    -------
    fft : torch.Tensor
        Complex-valued tensor containing the Fourier coefficients.
        Zero frequency is centered.
    """
    return torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2),
    )


def torch_peak_local_max(
    image: torch.Tensor, min_distance: int = 1, n_peaks: int | None = None
) -> torch.Tensor:
    """
    Find local maxima in batched 3D images and return fixed number of peaks per batch.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (B, D, H, W).
    min_distance : int, optional
        Minimum separation between peaks (voxels). Default is 1.
    n_peaks : int, optional
        Number of peaks to return per batch (must be <= total peaks in each batch).
        If None, uses the minimum number of peaks found in any batch item. Default is None.

    Returns
    -------
    peaks : torch.LongTensor
        Peak coordinates (z, y, x) for each batch. Shape (B, n_peaks, 3).
    """
    B, D, H, W = image.shape
    x = image.unsqueeze(1)  # (B, 1, D, H, W)
    k = 2 * min_distance + 1
    pooled = F.max_pool3d(x, kernel_size=k, stride=1, padding=min_distance)
    mask = (x == pooled).squeeze(1)  # (B, D, H, W)

    # Flatten spatial dims
    flat_mask = mask.view(B, -1)
    flat_image = image.view(B, -1)

    # Mask non-maxima
    flat_image_masked = flat_image.clone()
    flat_image_masked[~flat_mask] = -float("inf")

    if n_peaks is None:
        n_peaks = int(flat_mask.sum(dim=1).min().item())  # take min available peaks

    # Top-k per batch
    _, topk_idx = flat_image_masked.topk(n_peaks, dim=1)

    # Convert flat indices back to 3D coords
    z = topk_idx // (H * W)
    y = (topk_idx % (H * W)) // W
    x_ = topk_idx % W

    peaks = torch.stack([z, y, x_], dim=2)  # (B, n_peaks, 3)
    return peaks
