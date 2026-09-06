from __future__ import annotations

import torch
import torch.nn.functional as F
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


def gaussian_blur3d(
    V: torch.Tensor, sigma_vox: float, pad_mode: str = "replicate"
) -> torch.Tensor:
    """
    Separable Gaussian blur over the last three axes.

    The kernel is truncated at ``round(3 * sigma_vox)`` voxels each side.
    Each pass is a ``conv1d`` along the tensor's LAST (contiguous) axis, the
    other two axes being brought there by a transpose and a copy. A cuDNN
    ``conv3d`` with a ``(1, 1, 2r+1)``-shaped kernel computes the same thing
    but runs ~2x slower on the slabs `blend_ice_into_volume` and
    `ParticleGeneratorBase.solvate` hand it (0.36 vs 0.64 ms per 1024^2
    slice on an L40), and the blur is the largest single cost of solvation
    at a 512-pixel box. The two agree to float rounding (~2e-6 on a 7 V
    field), since only the summation order differs.

    Parameters
    ----------
    V : torch.Tensor
        Shape ``(..., Z, Y, X)``.
    sigma_vox : float
        Gaussian width in voxels. Non-positive returns ``V`` unchanged.
    pad_mode : str, optional
        How the faces are extended, any mode ``torch.nn.functional.pad``
        accepts for a 3-D tensor. ``"replicate"`` (default) suits a field
        that continues past the box, such as a specimen's occupancy;
        ``"constant"`` (zeros) suits a density that ends inside it.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``V``.
    """
    if sigma_vox <= 0:
        return V
    r = max(1, int(round(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, device=V.device, dtype=V.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    weight = (kernel / kernel.sum()).view(1, 1, -1)

    lead = V.shape[:-3]
    Z, Y, X = V.shape[-3:]
    nd = V.ndim

    # x axis, already contiguous
    t = F.conv1d(F.pad(V.reshape(-1, 1, X), (r, r), mode=pad_mode), weight)
    out = t.reshape(*lead, Z, Y, X)
    # y axis
    t = out.transpose(-1, -2).contiguous().reshape(-1, 1, Y)
    t = F.conv1d(F.pad(t, (r, r), mode=pad_mode), weight)
    out = t.reshape(*lead, Z, X, Y).transpose(-1, -2)
    # z axis
    t = out.permute(*range(nd - 3), nd - 2, nd - 1, nd - 3).contiguous()
    t = F.conv1d(F.pad(t.reshape(-1, 1, Z), (r, r), mode=pad_mode), weight)
    out = t.reshape(*lead, Y, X, Z).permute(*range(nd - 3), nd - 1, nd - 3, nd - 2)
    return out.contiguous()
