import torch
from scipy.fft import next_fast_len

def fftconvolve(in1, in2, mode="full"):
    """ From scipy fftconvolve.
    
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
    in1 : torch.tensor
        First input.
    in2 : torch.tensor
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
    out : array
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
    axes = [i for i in range(len(in1.shape))] #assume ndim convolution.

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)

def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """ From scipy.signal._signaltools
    
    Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : torch.tensor
        First input.
    in2 : torch.tensor
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : torch.tensor
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    if not len(axes):
        return in1 * in2

    complex_result = (torch.is_complex(in1) or torch.is_complex(in2))

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

def _centered(arr, newshape):
    """From scipy.signal._signaltools"""
    # Return the center newshape portion of the array.
    newshape = torch.as_tensor(newshape)
    currshape = torch.tensor(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _apply_conv_mode(ret, s1, s2, mode, axes):
    """ From scipy.signal._signaltools
    
    Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : torch.tensor
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
    if mode == "full":
        return ret.clone()
    elif mode == "same":
        return _centered(ret, s1).clone()
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid).clone()
    else:
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")