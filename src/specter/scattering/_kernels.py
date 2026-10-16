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
    The Fresnel free-space propagator ``exp(-i pi lambda d k^2)``.

    The paraxial free-space propagator of the multislice method; see [1]_,
    Ch. 6. Its sign pairs with the transmission function ``exp(+i sigma V)``
    and with the transfer function's defocus term: propagating a wave a
    distance ``d`` downstream is the same as overfocusing it by ``d``, since
    an underfocused lens images a plane upstream of the specimen. Until
    2026-10-16 the sign was ``+``, the conjugate, which gave the right
    single-scattering image but walked each slice the wrong way in depth.

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

    References
    ----------
    .. [1] Kirkland, E. J. (2010). *Advanced computing in electron microscopy*
       (2nd ed.). Springer. https://doi.org/10.1007/978-1-4419-6533-2
    """
    return torch.exp(-1j * torch.pi * wavelength * distance * k2)


def bandlimit_mask(
    k: torch.Tensor, pixel_size: float, klim: float | None
) -> torch.Tensor | int:
    """
    Kirkland's anti-aliasing bandlimit [1]_ as a binary mask on ``k``.

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

    References
    ----------
    .. [1] Kirkland, E. J. (2010). *Advanced computing in electron microscopy*
       (2nd ed.). Springer. https://doi.org/10.1007/978-1-4419-6533-2
    """
    if klim is None:
        return 1
    k_nyquist = 1.0 / (2.0 * pixel_size)
    return (k <= klim * k_nyquist).to(torch.float32)[None, ...]


def _split_nyquist(X: torch.Tensor, index: int, n: int, dim: int) -> torch.Tensor:
    """
    Share an even grid's single Nyquist bin between ``-N/2`` and ``+N/2``.

    After ``fftshift``, an even-length axis holds its Nyquist frequency once,
    at the negative end. Zero-padding that spectrum leaves the bin on one
    side only, and the interpolant of a real signal then acquires an
    imaginary part (and ``up(i x) != i up(x)``). Halving the bin onto both
    ends restores Hermitian symmetry, so the interpolation is linear and
    real-preserving, and the matching crop (:func:`_merge_nyquist`) inverts
    it exactly.
    """
    if n % 2:
        return X
    half = X.narrow(dim, index, 1) * 0.5
    X = X.clone()
    X.narrow(dim, index, 1).copy_(half)
    X.narrow(dim, index + n, 1).copy_(half)
    return X


def _merge_nyquist(X: torch.Tensor, index: int, n: int, dim: int) -> torch.Tensor:
    """Inverse of :func:`_split_nyquist`: fold ``+N/2`` back onto ``-N/2``."""
    if n % 2:
        return X
    X = X.clone()
    X.narrow(dim, index, 1).add_(X.narrow(dim, index + n, 1))
    return X


def fourier_upsample_xy(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Resample ``(..., Y, X)`` onto a grid `factor` times finer, adding no content.

    The spectrum is zero-padded about its centre, so the fine-grid samples
    are the band-limited interpolant of the coarse ones: every frequency the
    coarse grid holds is kept at its value and nothing is added above the
    coarse Nyquist. Values (not integrals) are preserved. An even grid's
    Nyquist bin is shared between ``-N/2`` and ``+N/2``, which keeps the
    operation linear and a real input real; :func:`fourier_crop_xy` inverts
    it exactly.

    Parameters
    ----------
    x : torch.Tensor
        Real or complex, any leading shape, square trailing grid.
    factor : int
        Integer upsampling factor, 1 returns `x` itself.

    Returns
    -------
    torch.Tensor
        Same dtype as `x`, trailing grid ``factor`` times larger.
    """
    if factor == 1:
        return x
    n = x.shape[-1]
    m = n * factor
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    # Align the DC bins (index n//2 and m//2 after fftshift) for either parity.
    pad = m // 2 - n // 2
    X = torch.nn.functional.pad(X, (pad, m - n - pad, pad, m - n - pad))
    X = _split_nyquist(_split_nyquist(X, pad, n, -2), pad, n, -1)
    fine = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))) * factor**2
    return fine.real if not x.is_complex() else fine


def fourier_crop_xy(psi: torch.Tensor, n: int) -> torch.Tensor:
    """
    Crop ``(..., M, M)`` complex `psi` to its central ``n x n`` spectrum.

    The inverse of :func:`fourier_upsample_xy`: content beyond the coarse
    Nyquist is discarded, the values of the retained band are preserved, and
    an even output grid's Nyquist bin collects both ``+-N/2`` fine bins.

    Parameters
    ----------
    psi : torch.Tensor
        Field on the fine grid.
    n : int
        Output grid size; ``psi.shape[-1]`` returns `psi` itself.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n, n)``.
    """
    m = psi.shape[-1]
    if m == n:
        return psi
    X = torch.fft.fftshift(torch.fft.fft2(psi), dim=(-2, -1))
    lo = m // 2 - n // 2
    X = _merge_nyquist(_merge_nyquist(X, lo, n, -2), lo, n, -1)
    X = X[..., lo : lo + n, lo : lo + n]
    out = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))) * (n / m) ** 2
    return out.real if not psi.is_complex() else out


def oversampled_exp(
    phase: torch.Tensor, factor: int, crop: bool = True
) -> torch.Tensor:
    """
    ``exp(phase)`` evaluated on a grid `factor` times finer, cropped back
    unless `crop` is False.

    The exponential of a band-limited field has harmonics beyond the field's
    own band; evaluated on the field's grid they wrap. This is the one
    operation the single-exponential models (Rytov, projection) share with
    the per-slice transmission of multislice, and it aliases at the same
    level: 1e-2 of the scattered power in 2000 A of ice, 1e-3 in thin ice,
    for either. `factor` 1 is a plain ``torch.exp``.

    Parameters
    ----------
    phase : torch.Tensor
        Complex ``(..., n, n)`` exponent, band-limited to its grid.
    factor : int
        Grid refinement, see :class:`~specter.scattering.Scattering`.
    crop : bool, optional
        Crop the result back to `phase`'s grid (default), or return it on
        the fine grid for the image to be formed there.

    Returns
    -------
    torch.Tensor
        ``(..., n, n)`` complex, or ``(..., factor n, factor n)`` uncropped.
    """
    if factor == 1:
        return torch.exp(phase)
    fine = torch.exp(fourier_upsample_xy(phase, factor))
    return fourier_crop_xy(fine, phase.shape[-1]) if crop else fine


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
