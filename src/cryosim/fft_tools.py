from __future__ import annotations

from typing import Sequence

import torch
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
    if shift:
        return torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(array, dim=dim), dim=dim), dim=dim
        )
    return torch.fft.fftn(array, dim=dim)


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
    if shift:
        return torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(array, dim=dim), dim=dim), dim=dim
        )
    return torch.fft.ifftn(array, dim=dim)


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

    ret = ifft(sp1 * sp2, fshape, dim=axes)

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


def _apply_conv_mode(
    ret: torch.Tensor, s1: list[int], s2: list[int], mode: str, axes: list[int]
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
