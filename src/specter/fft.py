from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len


def fft2(
    array: torch.Tensor,
    dim: tuple[int, int] | Sequence[int] = (-1, -2),
    shift: bool = False,
) -> torch.Tensor:
    """
    Compute 2D fast Fourier transform.

    Parameters
    ----------
    array : torch.Tensor
        Input array.
    dim : tuple of int or Sequence of int, optional
        Dimensions along which to compute the FFT. Default is (-1, -2).
    shift : bool, optional
        If True, applies fftshift before and after the FFT to center
        zero-frequency component. Default is False.

    Returns
    -------
    result : torch.Tensor
        2D Fourier transform of input.
    """
    if shift:
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(array, dim=dim), dim=dim), dim=dim
        )
    return torch.fft.fft2(array, dim=dim)


def ifft2(
    array: torch.Tensor,
    dim: tuple[int, int] | Sequence[int] = (-1, -2),
    shift: bool = False,
) -> torch.Tensor:
    """
    Compute 2D inverse fast Fourier transform.

    Parameters
    ----------
    array : torch.Tensor
        Input array in Fourier space.
    dim : tuple of int or Sequence of int, optional
        Dimensions along which to compute the inverse FFT. Default is (-1, -2).
    shift : bool, optional
        If True, applies fftshift before and after the inverse FFT to center
        zero-frequency component. Default is False.

    Returns
    -------
    result : torch.Tensor
        2D inverse Fourier transform of input.
    """
    if shift:
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(array, dim=dim), dim=dim), dim=dim
        )
    return torch.fft.ifft2(array, dim=dim)


def fft3(
    array: torch.Tensor,
    dim: tuple[int, int, int] | Sequence[int] = (-1, -2, -3),
    shift: bool = False,
) -> torch.Tensor:
    """
    Compute 3D fast Fourier transform.

    Parameters
    ----------
    array : torch.Tensor
        Input 3D array.
    dim : tuple of int or Sequence of int, optional
        Dimensions along which to compute the FFT. Default is (-1, -2, -3).
    shift : bool, optional
        If True, applies fftshift before and after the FFT to center
        zero-frequency component. Default is False.

    Returns
    -------
    result : torch.Tensor
        3D Fourier transform of input.
    """
    return fftn(array, dim=list(dim), shift=shift)


def ifft3(
    array: torch.Tensor,
    dim: tuple[int, int, int] | Sequence[int] = (-1, -2, -3),
    shift: bool = False,
) -> torch.Tensor:
    """
    Compute 3D inverse fast Fourier transform.

    Parameters
    ----------
    array : torch.Tensor
        Input 3D array in Fourier space.
    dim : tuple of int or Sequence of int, optional
        Dimensions along which to compute the inverse FFT. Default is (-1, -2, -3).
    shift : bool, optional
        If True, applies fftshift before and after the inverse FFT to center
        zero-frequency component. Default is False.

    Returns
    -------
    result : torch.Tensor
        3D inverse Fourier transform of input.
    """
    return ifftn(array, dim=list(dim), shift=shift)


def fftn(
    array: torch.Tensor, dim: Sequence[int] | None = None, shift: bool = False
) -> torch.Tensor:
    """
    Compute N-dimensional fast Fourier transform.

    Parameters
    ----------
    array : torch.Tensor
        Input N-dimensional array.
    dim : Sequence of int or None, optional
        Dimensions along which to compute the FFT. If None, computes FFT
        over all dimensions. Default is None.
    shift : bool, optional
        If True, applies fftshift before and after the FFT to center
        zero-frequency component. Default is False.

    Returns
    -------
    result : torch.Tensor
        N-dimensional Fourier transform of input.
    """
    if shift:
        return torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(array, dim=dim), dim=dim), dim=dim
        )
    return torch.fft.fftn(array, dim=dim)


def ifftn(
    array: torch.Tensor, dim: Sequence[int] | None = None, shift: bool = False
) -> torch.Tensor:
    """
    Compute N-dimensional inverse fast Fourier transform.

    Parameters
    ----------
    array : torch.Tensor
        Input N-dimensional array in Fourier space.
    dim : Sequence of int or None, optional
        Dimensions along which to compute the inverse FFT. If None, computes
        inverse FFT over all dimensions. Default is None.
    shift : bool, optional
        If True, applies fftshift before and after the inverse FFT to center
        zero-frequency component. Default is False.

    Returns
    -------
    result : torch.Tensor
        N-dimensional inverse Fourier transform of input.
    """
    if shift:
        return torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(array, dim=dim), dim=dim), dim=dim
        )
    return torch.fft.ifftn(array, dim=dim)


def fftconvolve(
    in1: torch.Tensor,
    in2: torch.Tensor,
    mode: str = "full",
    axes: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """From scipy fftconvolve.

    Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : torch.Tensor
        First input.
    in2 : torch.Tensor
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : torch.Tensor
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    """

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.numel() == 0 or in2.numel() == 0:  # empty arrays
        return torch.tensor([])

    s1 = in1.shape
    s2 = in2.shape

    if axes is None:
        axes = [i for i in range(len(in1.shape))]  # assume ndim convolution.
    elif isinstance(axes, int):
        axes = [axes]

    # Handle negative axes
    axes = [a + in1.ndim if a < 0 else a for a in axes]

    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(in1.ndim)
    ]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


# Conservative margin under the ~2**31 (INT32_MAX) per-sample spatial-size
# limit that trips PyTorch/cuDNN's "large non-batch-splittable
# convolutions" warning on cuDNN < 9.3 without the v8 API. Past that point
# PyTorch silently falls back to an unfold/im2col-based convolution whose
# memory cost is O(output_elements * kernel_elements) -- a few voxels of
# kernel turning into a >1 TB allocation attempt for a volume the size
# IceBank.generate_big_ice can request. Chunking below this keeps every
# individual conv3d call on the cheap, direct cuDNN path regardless of the
# cuDNN version actually installed.
_CUDNN_SAFE_CONV3D_SPATIAL_ELEMENTS = 2**30


def _conv3d_same_core(
    volume: torch.Tensor, weight: torch.Tensor, kz: int, ky: int, kx: int
) -> torch.Tensor:
    """'same'-mode conv3d (see :func:`spatial_convolve3d_same`) for a chunk
    small enough to hand to a single ``F.conv3d`` call directly."""
    full = F.conv3d(volume.unsqueeze(1), weight, padding=(kz - 1, ky - 1, kx - 1))
    full = full.squeeze(1)
    z0, y0, x0 = (kz - 1) // 2, (ky - 1) // 2, (kx - 1) // 2
    z, y, x = volume.shape[-3:]
    return full[:, z0 : z0 + z, y0 : y0 + y, x0 : x0 + x]


def spatial_convolve3d_same(
    volume: torch.Tensor,
    kernel: torch.Tensor,
    _max_spatial_elements: int = _CUDNN_SAFE_CONV3D_SPATIAL_ELEMENTS,
) -> torch.Tensor:
    """
    Direct (non-FFT) 'same'-mode 3D convolution of a batched volume with a
    single kernel shared across the batch, matching ``fftconvolve(volume,
    kernel, mode="same", axes=(-3, -2, -1))`` exactly -- including its
    centered-crop convention for even-sized kernel axes (see
    :func:`_centered`) -- but via ``torch.nn.functional.conv3d`` instead of
    an FFT.

    FFT convolution pays the cost of transforming the *entire* volume into
    (complex-valued, ~2x memory) frequency space regardless of kernel size.
    Direct convolution's cost instead scales with the kernel, not the
    volume, so for a kernel that is tiny relative to the volume -- e.g. a
    ~4-15 voxel atomic potential kernel (:func:`specter.ice._kernels.
    build_atomic_potential_kernel`) convolved against an ice volume with
    billions of voxels in :meth:`specter.ice.IceBank.generate_big_ice` --
    this avoids an OOM that plain ``fftconvolve`` hits on volumes too large
    to hold several complex-valued copies of at once. ``conv3d`` computes
    cross-correlation (no kernel flip), so the kernel is flipped here to
    recover true convolution.

    When a single sample's spatial size (``Z*Y*X``) exceeds
    ``_max_spatial_elements``, the volume is convolved in Z-chunks instead
    of one ``conv3d`` call -- see ``_CUDNN_SAFE_CONV3D_SPATIAL_ELEMENTS``.
    Each chunk is extended by ``kz - 1`` voxels of real neighboring context
    on either side (clipped at the true volume boundary, where zero-padding
    is correct anyway) before convolving, so its cropped core exactly
    matches what a single whole-volume call would have produced -- chunking
    changes only memory/compute path, never the result.

    Parameters
    ----------
    volume : torch.Tensor
        Shape (B, Z, Y, X).
    kernel : torch.Tensor
        Shape (Z', Y', X') or (1, Z', Y', X'); the same kernel is applied
        to every volume in the batch.

    Returns
    -------
    torch.Tensor
        Convolved volume, shape (B, Z, Y, X).
    """
    if kernel.ndim == 4:
        kernel = kernel.squeeze(0)
    kz, ky, kx = kernel.shape
    weight = (
        kernel.flip([0, 1, 2])
        .to(device=volume.device, dtype=volume.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    z, y, x = volume.shape[-3:]
    if z * y * x <= _max_spatial_elements:
        return _conv3d_same_core(volume, weight, kz, ky, kx)

    halo = kz - 1
    chunk_z = max(1, _max_spatial_elements // (y * x))
    out = torch.empty_like(volume)
    for z0 in range(0, z, chunk_z):
        z1 = min(z0 + chunk_z, z)
        ctx0 = max(0, z0 - halo)
        ctx1 = min(z, z1 + halo)
        chunk_out = _conv3d_same_core(volume[:, ctx0:ctx1], weight, kz, ky, kx)
        out[:, z0:z1] = chunk_out[:, z0 - ctx0 : z0 - ctx0 + (z1 - z0)]
    return out


def spatial_convolve2d_same(
    images: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Direct (non-FFT) 'same'-mode 2D convolution of a batch of images with a
    single shared kernel, matching ``fftconvolve(images, kernel, mode="same",
    axes=(-2, -1))`` exactly -- including its centered-crop convention for
    even-sized kernel axes (see :func:`_centered`).

    The 2D twin of :func:`spatial_convolve3d_same`, and the reason it exists
    is the same: ``F.conv2d(..., padding="same")`` pads *asymmetrically* when
    a kernel axis is even, which offsets the result half a pixel relative to
    the centered convention every other potential path in specter uses. Even
    kernels are the common case, not the exotic one -- 20 of the 36 pixel
    sizes between 0.5 and 4.0 Å produce one (see
    :func:`specter.potential.compute_supersampling_parameters`).

    No chunking counterpart to :func:`spatial_convolve3d_same`'s is needed:
    the 2D path convolves single ``(ny, nx)`` slices, which stay far below
    the cuDNN spatial-size limit that motivates chunking in 3D.

    ``conv2d`` computes cross-correlation (no kernel flip), so the kernel is
    flipped here to recover true convolution. Atomic potential kernels are
    radially symmetric, making the flip a no-op for them, but this is a
    general-purpose convolution and should behave like one.

    Parameters
    ----------
    images : torch.Tensor
        Shape (B, Y, X).
    kernel : torch.Tensor
        Shape (Y', X') or (1, Y', X'); applied to every image in the batch.

    Returns
    -------
    torch.Tensor
        Convolved images, shape (B, Y, X).
    """
    if kernel.ndim == 3:
        kernel = kernel.squeeze(0)
    ky, kx = kernel.shape
    weight = (
        kernel.flip([0, 1])
        .to(device=images.device, dtype=images.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    full = F.conv2d(images.unsqueeze(1), weight, padding=(ky - 1, kx - 1)).squeeze(1)
    y0, x0 = (ky - 1) // 2, (kx - 1) // 2
    y, x = images.shape[-2:]
    return full[:, y0 : y0 + y, x0 : x0 + x]


def _freq_domain_conv(
    in1: torch.Tensor,
    in2: torch.Tensor,
    axes: list[int],
    shape: list[int],
    calc_fast_len: bool = False,
) -> torch.Tensor:
    """From scipy.signal._signaltools

    Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : torch.Tensor
        First input.
    in2 : torch.Tensor
        Second input. Should have the same number of dimensions as `in1`.
    axes : list of int
        Axes over which to compute the FFTs.
    shape : list of int
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : torch.Tensor
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.
    """
    if not len(axes):
        return in1 * in2

    complex_result = torch.is_complex(in1) or torch.is_complex(in2)

    fshape: list[int]
    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = torch.fft.rfftn, torch.fft.irfftn
    else:
        fft, ifft = torch.fft.fftn, torch.fft.ifftn

    sp1 = fft(in1, fshape, dim=axes)
    sp2 = fft(in2, fshape, dim=axes)

    if torch.is_grad_enabled() and (in1.requires_grad or in2.requires_grad):
        ret = ifft(sp1 * sp2, fshape, dim=axes)
    else:
        # The product is a third spectrum-sized complex array held while both
        # inputs' spectra are still alive; for a 512^3 ice volume each is
        # 0.6 GB, and the transient sum was the peak of the whole particle
        # forward pass. Multiply into sp1 and release sp2 before the inverse.
        sp1.mul_(sp2)
        del sp2
        ret = ifft(sp1, fshape, dim=axes)
        del sp1

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret


def _centered(
    arr: torch.Tensor, newshape: Sequence[int] | torch.Tensor
) -> torch.Tensor:
    """From scipy.signal._signaltools"""
    # Return the center newshape portion of the array.
    newshape = torch.as_tensor(newshape)
    currshape = torch.tensor(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def fourier_shell_correlation(
    volume1: torch.Tensor,
    volume2: torch.Tensor,
    pixelsize: float = 1.0,
    res_cutoff: float | None = None,
    randomise_phases_beyond: float | None = None,
    return_real: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Fourier Shell Correlation (FSC) between two 3D volumes.

    Parameters
    ----------
    volume1, volume2 : torch.Tensor
        Real-space 3D volumes of shape (N, N, N).
    pixelsize : float, optional
        Pixel size in Å (or consistent units). Default is 1.
    res_cutoff : float | None, optional
        High-resolution cutoff. Frequencies beyond 1/res_cutoff are zeroed.
    randomise_phases_beyond : float | None, optional
        Resolution limit beyond which phases are randomised, for
        phase-randomisation FSC (gold-standard artefact validation).
    return_real : bool, optional
        If True (default), return real-part FSC (standard definition).
        If False, return the magnitude of the complex cross-correlation per shell.

    Returns
    -------
    k : torch.Tensor
        Spatial frequency axis (units: 1/pixelsize). Shells run out to the
        *corners* of the Fourier cube, so the axis extends past Nyquist by a
        factor of sqrt(3); those trailing shells are sampled from the corner
        directions alone. Pass ``k_max=1/(2*pixelsize)`` to `plots.fsc_resolution`
        to keep a crossing in that aliased tail out of a reported resolution.
    fsc : torch.Tensor
        FSC curve, one value per Fourier shell.

    Raises
    ------
    ValueError
        If the volumes differ in shape, or are not cubic. Shells are binned by
        voxel-index radius, which coincides with physical frequency only when
        every axis has the same length; a non-cubic volume would otherwise
        return a silently wrong ``k`` axis.
    """
    # Lazy import to avoid circular dependency (arrays imports fft)
    from .arrays import radial_profile_3d

    if volume1.shape != volume2.shape:
        raise ValueError(
            f"Volumes must have the same shape, got {tuple(volume1.shape)} "
            f"and {tuple(volume2.shape)}."
        )
    if volume1.ndim != 3 or len(set(volume1.shape)) != 1:
        raise ValueError(
            "fourier_shell_correlation requires cubic (N, N, N) volumes, got "
            f"{tuple(volume1.shape)}."
        )

    n = volume1.shape[0]
    device = volume1.device

    volume1_f = fft3(volume1, shift=True)
    volume2_f = fft3(volume2, shift=True)

    # Apply resolution cutoff and/or phase randomisation if requested.
    # k-grid is fftshifted to match the shifted FFT output (DC at center).
    if res_cutoff is not None or randomise_phases_beyond is not None:
        kx = torch.fft.fftshift(torch.fft.fftfreq(n, pixelsize, device=device))
        kxx, kyy, kzz = torch.meshgrid(kx, kx, kx, indexing="ij")
        k_mag = torch.sqrt(kxx**2 + kyy**2 + kzz**2)

        if res_cutoff is not None:
            mask = k_mag > (1.0 / res_cutoff)
            volume1_f = volume1_f.clone()
            volume2_f = volume2_f.clone()
            volume1_f[mask] = 0.0
            volume2_f[mask] = 0.0

        if randomise_phases_beyond is not None:
            rand_mask = k_mag > (1.0 / randomise_phases_beyond)
            rand_phase1 = torch.rand(n, n, n, device=device) * (2 * torch.pi)
            rand_phase2 = torch.rand(n, n, n, device=device) * (2 * torch.pi)
            phase1 = torch.angle(volume1_f)
            phase2 = torch.angle(volume2_f)
            phase1[rand_mask] = rand_phase1[rand_mask]
            phase2[rand_mask] = rand_phase2[rand_mask]
            volume1_f = volume1_f.abs() * torch.exp(1j * phase1)
            volume2_f = volume2_f.abs() * torch.exp(1j * phase2)

    # Cross-spectrum F1 * conj(F2) and power spectra.
    cross = volume1_f * volume2_f.conj()
    norm1 = volume1_f.real**2 + volume1_f.imag**2  # |F1|^2, avoids sqrt in .abs()
    norm2 = volume2_f.real**2 + volume2_f.imag**2

    # Radial averages — DC is at (n//2, n//2, n//2), matching the default center.
    cross_r = radial_profile_3d(cross.real)
    norm1_r = radial_profile_3d(norm1)
    norm2_r = radial_profile_3d(norm2)

    if return_real:
        num = cross_r
    else:
        cross_i = radial_profile_3d(cross.imag)
        num = torch.sqrt(cross_r**2 + cross_i**2)

    denom = torch.sqrt(norm1_r * norm2_r).clamp(min=1e-10)
    fsc = num / denom

    # Frequency axis: shell index × frequency spacing (1 / (N * pixelsize))
    dk = 1.0 / (n * float(pixelsize))
    k = torch.arange(len(fsc), device=device) * dk

    return k, fsc


def _apply_conv_mode(
    ret: torch.Tensor,
    s1: Sequence[int],
    s2: Sequence[int],
    mode: str,
    axes: list[int],
) -> torch.Tensor:
    """From scipy.signal._signaltools

    Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : torch.Tensor
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of int
        Axes over which to compute the convolution.

    Returns
    -------
    ret : torch.Tensor
        A copy of `res`, sliced to the correct size for the given `mode`.
    """
    if mode == "full":
        return ret.clone()
    elif mode == "same":
        return _centered(ret, s1).clone()
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
            for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid).clone()
    else:
        raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")
