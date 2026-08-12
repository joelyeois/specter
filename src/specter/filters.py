from __future__ import annotations

import torch
import numpy as np
from skimage.filters import butterworth

from .fft import fftn, ifftn


def butter(images: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Applies butterworth filter to 2D images.

    Ref: 'https://discuss.cryosparc.com/t/inspect-raw-images-of-particles-of-certain-2-3d-classes/12261/8'

    Parameters
    ----------
    images : torch.Tensor or np.ndarray
        Can be a 2D single image (size, size) or a batch of 2D images (N, size, size).

    Returns
    -------
    filtered : torch.Tensor or np.ndarray
        Butterworth filtered images.
    """

    istensor = False
    n = images.shape[-1]
    if torch.is_tensor(images):
        istensor = True
        images = images.numpy()

    channel_axis = 0 if images.ndim == 3 else None

    filtered = butterworth(
        images,
        cutoff_frequency_ratio=6 / n,
        high_pass=False,
        order=1,
        channel_axis=channel_axis,
    )

    if istensor:
        return torch.from_numpy(filtered)
    else:
        return filtered


def cosine_taper_window(
    n: int,
    taper_px: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    1-D cosine window of length ``n`` with a ``taper_px``-wide fade at each end.

    Parameters
    ----------
    n : int
        Length of the window.
    taper_px : int
        Width of the fade at each end, in samples. Clamped to ``n // 2``.
        A value <= 0 returns an all-ones window.
    device : torch.device or str, optional
        Device for the output tensor.
    dtype : torch.dtype, optional
        Dtype for the output tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``, values in ``[0, 1]``, 1 away from the edges.
    """
    win = torch.ones(n, device=device, dtype=dtype)
    taper_px = min(taper_px, n // 2)
    if taper_px <= 0:
        return win
    ramp = 0.5 * (
        1
        - torch.cos(
            torch.pi * torch.linspace(0, 1, taper_px, device=device, dtype=dtype)
        )
    )
    win[:taper_px] = ramp
    win[-taper_px:] = ramp.flip(0)
    return win


def apply_bfactor(
    volume: torch.Tensor, pixel_size: float, bfactor: float
) -> torch.Tensor:
    """
    Apply B-factor blurring to a 3D scattering potential volume.

    Applies temperature factor (B-factor) blurring in Fourier space using
    the formula exp(-B/4 * k²), which simulates thermal motion effects
    on atomic scattering amplitudes.

    Parameters
    ----------
    volume : torch.Tensor
        3D scattering potential volume with shape (n, n, n). Assumed to be
        cubic and real-valued.
    pixel_size : float
        The pixel size in Å.
    bfactor : float
        B-factor (temperature factor). Higher values increase blurring.
        If bfactor=0.0, returns the original volume unchanged.

    Returns
    -------
    newvolume : torch.Tensor
        B-factor blurred volume with same shape as input. Returns real-valued
        tensor if input is real, complex tensor if input is complex.

    Notes
    -----
    The B-factor is applied in Fourier space as:
    F_blurred(k) = F(k) * exp(-B/4 * k²)
    where k is the spatial frequency magnitude.
    """
    if bfactor == 0.0:
        return volume

    kx = torch.fft.fftfreq(volume.shape[-1], pixel_size, device=volume.device)
    KZ, KY, KX = torch.meshgrid(kx, kx, kx, indexing="ij")
    k2 = KZ**2 + KY**2 + KX**2
    newvolume = ifftn(fftn(volume) * torch.exp(-bfactor / 4 * k2))

    if torch.is_complex(volume):
        return newvolume
    return torch.real(newvolume)


def chimera_gaussian_sigma_to_bfactor(
    sigma: float | torch.Tensor,
) -> float | torch.Tensor:
    """
    Convert ChimeraX Gaussian width to B-factor.

    Converts the Gaussian standard deviation (sigma) used in ChimeraX
    into the equivalent crystallographic B-factor.

    Parameters
    ----------
    sigma : float or torch.Tensor
        Gaussian width (standard deviation) in Å.

    Returns
    -------
    bfactor : float or torch.Tensor
        B-factor, calculated as 8π²σ².

    Notes
    -----
    The relationship between Gaussian width and B-factor is:
    B = 8π²σ²

    This conversion is useful when matching blurring parameters between
    ChimeraX visualization and cryo-EM simulation tools.
    """
    bfactor = 8 * torch.pi**2 * sigma**2
    return bfactor
