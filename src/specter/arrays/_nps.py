"""Noise power spectrum (NPS) estimation from images and half-map differences."""

from __future__ import annotations

import torch

from ._profiles import radial_profile_2d


def compute_nps_1d(images: torch.Tensor) -> torch.Tensor:
    """
    Estimate the 1D radial noise power spectrum from a batch of images.

    Computed as the mean radial profile of ``|FFT2(images)|^2``, averaged
    over the batch. In low-SNR regimes (typical cryo-EM), the total power
    spectrum approximates the noise power spectrum.

    Parameters
    ----------
    images : torch.Tensor
        Batch of real images, shape (N, H, W).

    Returns
    -------
    nps_1d : torch.Tensor
        1D radial NPS, shape (R,), indexed by integer pixel radius from
        the DC component. R = max radial distance from center + 1.
    """
    N, H, W = images.shape

    # Power spectrum with DC shifted to center
    F_imgs = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))  # (N, H, W)
    mean_power = (F_imgs.abs() ** 2).mean(dim=0)  # (H, W)

    return radial_profile_2d(mean_power)


def compute_nps_2d(
    images: torch.Tensor,
    normalize: bool = True,
    zero_dc: bool = False,
) -> torch.Tensor:
    """
    Compute the 2D radial noise power spectrum from one or more images.

    Estimates the NPS as the mean power spectrum ``|FFT2(images)|^2``, radially
    averaged to enforce isotropy, then mapped back to 2D. Returns in rfft2
    half-plane format (H, W//2+1) for direct use in spectral-weighted losses.

    In low-SNR regimes (typical cryo-EM), the total power spectrum
    approximates the noise power spectrum. High-power (signal-dominated)
    frequencies will have large NPS values; low-power (noise-dominated)
    frequencies will have small values.

    The DC component (k=0) is always replaced by the value at k=1 to avoid
    the large mean-intensity spike dominating the spectrum. If zero_dc=True,
    the DC bin is set to zero instead, fully excluding it from any loss.

    Parameters
    ----------
    images : torch.Tensor
        Input images, shape (H, W) for a single image or (N, H, W) for a batch.
    normalize : bool, optional
        If True, normalize so the mean NPS value = 1. Default is True.
    zero_dc : bool, optional
        If True, set the DC bin to 0 rather than interpolating from k=1.
        Default is False.

    Returns
    -------
    nps_2d : torch.Tensor
        2D radial NPS, shape (H, W//2+1), matching torch.fft.rfft2 output.
    """
    if images.ndim == 2:
        images = images.unsqueeze(0)

    N, H, W = images.shape
    device = images.device

    # Mean power spectrum with DC shifted to center, shape (H, W)
    F_imgs = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
    mean_power = (F_imgs.abs() ** 2).mean(dim=0)

    # Radially average to 1D NPS
    nps_1d = radial_profile_2d(mean_power)

    # Handle DC bin: it is dominated by the squared mean intensity and carries
    # no structural information. Replace with k=1 value (smooth continuation)
    # or zero (full exclusion).
    nps_1d = nps_1d.clone()
    nps_1d[0] = 0.0 if zero_dc else nps_1d[1]

    # Map 1D NPS back to 2D in rfft2 half-plane layout
    kx_px = torch.fft.fftfreq(H, device=device) * H  # (H,)
    ky_px = torch.fft.rfftfreq(W, device=device) * W  # (W//2+1,)
    KX_px, KY_px = torch.meshgrid(kx_px, ky_px, indexing="ij")
    r_idx = torch.sqrt(KX_px**2 + KY_px**2).round().long().clamp(0, len(nps_1d) - 1)
    nps_2d = nps_1d[r_idx]  # (H, W//2+1)

    if normalize:
        nps_2d = nps_2d / nps_2d.mean().clamp(min=1e-10)

    return nps_2d


def compute_nps_3d(
    diff_volume: torch.Tensor,
    normalize: bool = True,
    zero_dc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the 3D noise power spectrum from a half-map difference volume.

    The difference (halfmap1 - halfmap2) cancels the signal, leaving 2x the
    noise. The NPS is estimated as ``|FFT3(diff)|^2 / 2``, radially averaged to
    enforce isotropy.

    Parameters
    ----------
    diff_volume : torch.Tensor
        Difference volume (halfmap1 - halfmap2), shape (D, H, W).
    normalize : bool, optional
        If True, normalize so the mean NPS value = 1. Default is True.
    zero_dc : bool, optional
        If True, set the DC bin to 0. Default is False.

    Returns
    -------
    nps_1d : torch.Tensor
        1D radial NPS, shape (R,), indexed by integer pixel radius.
    nps_3d : torch.Tensor
        3D radial NPS mapped to rfft3 half-plane layout (D, H, W//2+1),
        for direct use in spectral-weighted losses on volumes.
    """
    if diff_volume.ndim != 3:
        raise ValueError("Input must be a 3D volume (D, H, W).")
    D, H, W = diff_volume.shape
    device = diff_volume.device

    # Power spectrum of difference map, DC centered
    # Divide by 2 because diff = noise_1 - noise_2, so var(diff) = 2*var(noise)
    F = torch.fft.fftshift(torch.fft.fftn(diff_volume), dim=(-3, -2, -1))
    power = (F.abs() ** 2) / 2.0  # (D, H, W)

    # Build 3D radial index grid (centered)
    kx = torch.fft.fftfreq(D, device=device) * D  # (D,)
    ky = torch.fft.fftfreq(H, device=device) * H  # (H,)
    kz = torch.fft.fftfreq(W, device=device) * W  # (W,)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    # After fftshift, we need the centered radii
    KX_s = torch.fft.fftshift(KX)
    KY_s = torch.fft.fftshift(KY)
    KZ_s = torch.fft.fftshift(KZ)
    r_grid = torch.sqrt(KX_s**2 + KY_s**2 + KZ_s**2)  # (D, H, W)
    r_idx_full = r_grid.round().long()

    # Compute 1D radial NPS by averaging shells
    R = int(r_idx_full.max().item()) + 1
    nps_1d = torch.zeros(R, device=device)
    counts = torch.zeros(R, device=device)
    r_flat = r_idx_full.clamp(0, R - 1).reshape(-1)
    p_flat = power.reshape(-1)
    nps_1d.scatter_add_(0, r_flat, p_flat)
    counts.scatter_add_(0, r_flat, torch.ones_like(p_flat))
    nps_1d = nps_1d / counts.clamp(min=1)

    # Handle DC bin
    nps_1d = nps_1d.clone()
    nps_1d[0] = 0.0 if zero_dc else nps_1d[1]

    # Map 1D NPS back to rfft3 half-plane layout (D, H, W//2+1)
    kx_r = torch.fft.fftfreq(D, device=device) * D  # (D,)
    ky_r = torch.fft.fftfreq(H, device=device) * H  # (H,)
    kz_r = torch.fft.rfftfreq(W, device=device) * W  # (W//2+1,)
    KX_r, KY_r, KZ_r = torch.meshgrid(kx_r, ky_r, kz_r, indexing="ij")
    r_idx_rfft = torch.sqrt(KX_r**2 + KY_r**2 + KZ_r**2).round().long().clamp(0, R - 1)
    nps_3d = nps_1d[r_idx_rfft]  # (D, H, W//2+1)

    if normalize:
        mean_val = nps_3d.mean().clamp(min=1e-10)
        nps_1d = nps_1d / mean_val
        nps_3d = nps_3d / mean_val

    return nps_1d, nps_3d
