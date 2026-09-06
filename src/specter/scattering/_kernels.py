"""
The pieces of the propagation physics both scattering classes share.
"""

from __future__ import annotations

import torch

# Number of z-slices whose transmission functions are evaluated per batched
# torch.exp call in Scattering.multislice. Purely a performance knob: output
# and gradients are bitwise identical for any value, so it is deliberately not
# exposed as a constructor argument.
#
# 8 is the knee of the measured curve. Going 1 -> 8 is worth 16-60% (box
# 64-256, batch 1-16); going 8 -> 64 is <= 11% and inside run-to-run noise,
# while forward peak memory grows linearly in the chunk size (+3% at 8, +118%
# unchunked). Backward peak memory is flat in it, since autograd retains every
# slice's transmission function either way. Measured on an L40.
_MULTISLICE_SLICE_CHUNK = 8


#: Complex elements per z-chunk of the single-scatter slice sums
#: (:meth:`Scattering._fourier_slice_sum`): 2**23 is 64 MB of complex64, ten
#: slices of a 512-box at batch 3. Measured on an L40 at that size, rytov
#: forward + backward: 51 ms at 2**23, 89 ms at 2**25 and 2**27, 141 ms
#: unchunked -- a chunk that fits the L2 cache wins outright, since every
#: pass here is memory-bound.
_SLICE_SUM_CHUNK_ELEMENTS = 2**23


def frequency_grid(nxy: int, pixel_size: float) -> torch.Tensor:
    """
    The radial spatial frequency ``|k|`` of an ``(nxy, nxy)`` FFT grid.

    Native (unshifted) FFT order, DC at index 0, in 1/Angstrom.

    Parameters
    ----------
    nxy : int
        Grid size in pixels.
    pixel_size : float
        Pixel size in Angstrom.

    Returns
    -------
    torch.Tensor
        Shape ``(nxy, nxy)``.
    """
    kx = torch.fft.fftfreq(nxy, pixel_size)
    kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
    return torch.sqrt(kxx**2 + kyy**2)


def fresnel_propagator(
    k2: torch.Tensor, wavelength: float, distance: float | torch.Tensor
) -> torch.Tensor:
    """
    The Fresnel free-space propagator ``exp(i pi lambda d k^2)``.

    Parameters
    ----------
    k2 : torch.Tensor
        Squared spatial frequency, in 1/Angstrom^2, any shape.
    wavelength : float
        Electron wavelength in Angstrom.
    distance : float or torch.Tensor
        Propagation distance in Angstrom; a tensor broadcasts against ``k2``.

    Returns
    -------
    torch.Tensor
        Complex, the broadcast shape of ``distance`` and ``k2``.
    """
    return torch.exp(1j * torch.pi * wavelength * distance * k2)


def bandlimit_mask(
    k: torch.Tensor, pixel_size: float, klim: float | None
) -> torch.Tensor | int:
    """
    Kirkland's anti-aliasing bandlimit as a binary mask on ``k``.

    Built directly against ``k`` in native (unshifted) FFT order rather
    than via ``disk2d()``, whose disk is centred at index ``n // 2``
    (fftshift convention) and would keep the Nyquist corner while masking
    out DC if multiplied straight into an unshifted spectrum. Binary
    (0.0/1.0) on purpose: the propagation loops fold it into the propagator
    once, a reassociation that is exact only for a binary mask.

    Parameters
    ----------
    k : torch.Tensor
        Radial frequency, see :func:`frequency_grid`.
    pixel_size : float
        Pixel size in Angstrom.
    klim : float, optional
        Cut-off as a fraction of Nyquist. None returns the Python int 1,
        which multiplies as a no-op.

    Returns
    -------
    torch.Tensor or int
        Shape ``(1, nxy, nxy)`` float32, or 1.
    """
    if klim is None:
        return 1
    k_nyquist = 1.0 / (2.0 * pixel_size)
    return (k <= klim * k_nyquist).to(torch.float32)[None, ...]


def absorption_factor(V: torch.Tensor, alpha: float) -> complex:
    """
    The complex scalar :func:`~specter.potential.apply_amplitude_contrast`
    multiplies a real potential by, ``sqrt(1 - alpha^2) + i alpha``, or 1
    for a `V` that is already complex (and so carries it).

    For a model that is linear in V, or applies the factor inside an
    elementwise function, this is the scalar to fold in rather than
    materialising the complex volume.

    Parameters
    ----------
    V : torch.Tensor
        The potential the factor would apply to.
    alpha : float
        Amplitude-contrast ratio.

    Returns
    -------
    complex
    """
    if V.is_complex() or alpha == 0:
        return 1.0
    return (1 - alpha**2) ** 0.5 + 1j * alpha


def phase_scale(V: torch.Tensor, sigma: float, dz: float, alpha: float) -> complex:
    """
    The scalar ``i sigma dz`` times :func:`absorption_factor`, which turns
    a slice's potential into its phase in every single-scatter model.

    Parameters
    ----------
    V : torch.Tensor
        The potential the scalar applies to, real or complex.
    sigma : float
        Interaction parameter, see :func:`specter.constants.interaction_parameter`.
    dz : float
        Slice thickness in Angstrom.
    alpha : float
        Amplitude-contrast ratio.

    Returns
    -------
    complex
    """
    return 1j * sigma * dz * absorption_factor(V, alpha)
