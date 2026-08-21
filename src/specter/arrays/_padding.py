"""Padding, cropping, resampling, and radial symmetrization of images/volumes."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from ..fft import fft2, ifft2
from ._profiles import radial_profile_2d, radial_profile_3d


def fourier_crop(
    images: torch.Tensor,
    current_pixel_size: float,
    target_pixel_size: float,
) -> tuple[torch.Tensor, float]:
    """
    Resample 2D image(s) via Fourier-space center cropping.

    Downsamples from smaller pixel size (finer resolution) to larger
    pixel size (coarser resolution) by FFT → center crop → IFFT.
    Preserves integrated density (sum × pixel_size²) via physics-correct
    scaling.

    Parameters
    ----------
    images : torch.Tensor
        Shape (H, W) for single image or (N, H, W) for batch.
        Real-valued, any dtype.
    current_pixel_size : float
        Current pixel size (Å or any consistent unit).
    target_pixel_size : float
        Desired pixel size (must be ≥ current_pixel_size).

    Returns
    -------
    resampled : torch.Tensor
        Downsampled image(s), same shape as input but with smaller
        spatial dimensions.
    actual_pixel_size : float
        Achieved pixel size (may differ slightly due to rounding).

    Raises
    ------
    ValueError
        If target_pixel_size < current_pixel_size (upsampling not supported).

    Examples
    --------
    >>> img = torch.randn(512, 512)
    >>> resampled, actual_ps = fourier_crop(img, 0.5, 1.0)
    >>> resampled.shape  # downsampled to ~256x256
    torch.Size([256, 256])
    """
    if target_pixel_size < current_pixel_size:
        raise ValueError(
            f"target_pixel_size ({target_pixel_size}) must be >= "
            f"current_pixel_size ({current_pixel_size}). "
            "Upsampling not supported."
        )

    # Handle single vs batch
    if images.ndim == 2:
        images = images.unsqueeze(0)
        squeeze_output = True
    elif images.ndim == 3:
        squeeze_output = False
    else:
        raise ValueError(
            f"images must be 2D (H, W) or 3D (N, H, W), got shape {images.shape}"
        )

    N, H, W = images.shape

    # Compute scaling and output size
    scale = current_pixel_size / target_pixel_size

    if abs(scale - 1.0) < 1e-7:
        # No resampling needed
        result = images.squeeze(0) if squeeze_output else images
        return result, current_pixel_size

    output_size = int(round(H * scale))

    # FFT with DC at center
    images_f = fft2(images, shift=True)

    # Center crop in Fourier space
    start_idx = (H - output_size) // 2
    end_idx = start_idx + output_size
    images_f_cropped = images_f[:, start_idx:end_idx, start_idx:end_idx]

    # IFFT back to real space
    resampled = ifft2(images_f_cropped, shift=True).real

    # Scale to conserve integrated density: multiply by scale²
    scale_factor = scale**2
    resampled = resampled * scale_factor

    # Remove batch dimension if input was single image
    if squeeze_output:
        resampled = resampled.squeeze(0)

    # Compute actual achieved pixel size
    actual_pixel_size = current_pixel_size * H / output_size

    return resampled, actual_pixel_size


def downsample(
    images: torch.Tensor, bin_factor: int = 2, method: str = "fft"
) -> torch.Tensor:
    """
    Downsample images using FFT or average pooling.

    Parameters
    ----------
    images : torch.Tensor
        Input images.
    bin_factor : int, optional
        Binning factor. Default is 2.
    method : str, optional
        Downsampling method ('fft' or 'avgpool'). Default is 'fft'.

    Returns
    -------
    images_bin : torch.Tensor
        Downsampled images.
    """
    if method == "fft":
        N = images.shape[-1]
        n = N // bin_factor
        images_bin = ifft2(
            fft2(images, shift=True)[
                :,
                N // 2 - n // 2 : N // 2 - n // 2 + n,
                N // 2 - n // 2 : N // 2 - n // 2 + n,
            ],
            shift=True,
        ).real
    elif method == "avgpool":
        avgpool = torch.nn.AvgPool2d(bin_factor, stride=bin_factor)
        images_bin = avgpool(images) * bin_factor**2
    return images_bin


def centered_pad(data: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    """
    Pad a tensor to a target shape, symmetrically.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor.
    target_shape : Sequence of int
        Target shape for padding.

    Returns
    -------
    padded : torch.Tensor
        Padded tensor.
    """
    pad = []
    for size, tgt in zip(reversed(data.shape), reversed(target_shape)):
        diff = tgt - size
        pad.extend([diff // 2, diff - diff // 2])
    return F.pad(data, pad)


def pad_to_common_shape(
    data_a: torch.Tensor, data_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad two tensors to a common shape.

    Parameters
    ----------
    data_a, data_b : torch.Tensor
        Input tensors.

    Returns
    -------
    padded_a, padded_b : torch.Tensor
        Padded tensors.
    """
    target = [max(a, b) for a, b in zip(data_a.shape, data_b.shape)]
    return centered_pad(data_a, target), centered_pad(data_b, target)


def compute_nz(base_nz: int, ice_thickness: float | None, pixel_size: float) -> int:
    """
    Number of Z slices given an optional ice thickness.

    Returns ``base_nz`` unchanged if ice thickness is None or smaller than the
    base volume depth; otherwise uses ice thickness to determine the depth.

    Parameters
    ----------
    base_nz : int
        Depth of the particle volume in slices.
    ice_thickness : float or None
        Total ice thickness in Å. None means no ice padding.
    pixel_size : float
        Voxel size in Å.

    Returns
    -------
    nz : int
        Number of Z slices to use.
    """
    if ice_thickness is None or ice_thickness < base_nz * pixel_size:
        return base_nz
    return int(ice_thickness // pixel_size)


def pad_volume(
    volume: torch.Tensor,
    nxy: int,
    nz: int,
    ice_thickness: float | None,
    pad_fft: bool,
    xy_pad_mode: str = "constant",
) -> torch.Tensor:
    """
    Pad a potential volume in Z and/or XY.

    Z-padding extends the volume to ``nz`` slices (zeros added symmetrically)
    when ice thickness is set.  XY-padding adds ``nxy // 2`` pixels on each
    side for FFT antialiasing.

    Parameters
    ----------
    volume : torch.Tensor
        Volume of shape (B, Z, Y, X).
    nxy : int
        Unpadded image size in pixels.
    nz : int
        Target Z depth after padding.
    ice_thickness : float or None
        If not None, Z-padding is applied.
    pad_fft : bool
        If True, XY-padding is applied.
    xy_pad_mode : str, optional
        Padding mode for XY axes. Default 'constant'.

    Returns
    -------
    volume : torch.Tensor
        Padded volume.
    """
    if ice_thickness is not None:
        zpad_px = nz - nxy
        volume = F.pad(
            volume,
            (0, 0, 0, 0, zpad_px // 2, nz - zpad_px // 2 - volume.shape[1]),
            mode="constant",
        )
    if pad_fft:
        volume = F.pad(
            volume,
            (nxy // 2, nxy // 2, nxy // 2, nxy // 2, 0, 0),
            mode=xy_pad_mode,
        )
    return volume


def radial_symmetrize(
    data: torch.Tensor,
    center: float | None = None,
    output_ndim: int | None = None,
    return_size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Radially/spherically average a 2D or 3D tensor and map it back.

    Parameters
    ----------
    data : torch.Tensor
        (N, N) or (N, N, N) input tensor.
    center : float, optional
        Center used when computing the radial profile from the square/cubic
        input. Default is N // 2.
    output_ndim : {2, 3}, optional
        Output dimensionality. Defaults to match the input.
        If the input is 2D and ``output_ndim=3``, the 1D radial profile
        computed from the 2D image is mapped onto a volume via spherical
        distances from the same center.
    return_size : tuple[int, ...], optional
        Shape of the output tensor, e.g. ``(512, 256, 256)``.  The number of
        dimensions must match ``output_ndim`` (or ``data.ndim`` when
        ``output_ndim`` is not given).

        Each axis is scaled so that its own Nyquist pixel maps to the same
        radial-profile index as every other axis, i.e. the profile index for
        output pixel ``(iz, iy, ix)`` is

            ``sqrt((N/Oz*(iz-cz))^2 + (N/Oy*(iy-cy))^2 + (N/Ox*(ix-cx))^2)``

        This preserves equal Nyquist frequency along all axes, matching a
        real-space volume with uniform voxel size.  When ``return_size`` is
        omitted (or cubic) the formula reduces to plain Euclidean distance,
        identical to the previous behaviour.

    Returns
    -------
    torch.Tensor
        Radially/spherically symmetrised tensor with shape ``return_size`` (if
        provided) or ``(N,) * output_ndim`` otherwise.
    """
    if data.ndim not in (2, 3):
        raise ValueError("Input must be a 2D or 3D tensor.")
    N = data.shape[0]
    if not all(s == N for s in data.shape):
        raise ValueError(f"Input must be square/cubic, got shape {tuple(data.shape)}.")

    if output_ndim is None:
        output_ndim = data.ndim
    if output_ndim not in (2, 3):
        raise ValueError("output_ndim must be 2 or 3.")
    if data.ndim == 3 and output_ndim == 2:
        raise ValueError("Reducing a 3D input to a 2D output is not supported.")

    if return_size is not None and len(return_size) != output_ndim:
        raise ValueError(
            f"return_size has {len(return_size)} dimensions but output_ndim={output_ndim}."
        )

    device = data.device
    if center is None:
        center = N // 2

    if data.ndim == 2:
        radial_mean = radial_profile_2d(data, center=(center, center))
    else:
        radial_mean = radial_profile_3d(data, center=(center, center, center))

    out_shape: tuple[int, ...] = (
        return_size if return_size is not None else (N,) * output_ndim
    )
    out_centers = tuple(s // 2 for s in out_shape)
    # Per-axis scale: maps each axis's Nyquist pixel to the same profile index.
    scales = tuple(float(N) / s for s in out_shape)

    if output_ndim == 2:
        oh, ow = out_shape
        cy, cx = out_centers
        sy, sx = scales
        y, x = torch.meshgrid(
            torch.arange(oh, device=device),
            torch.arange(ow, device=device),
            indexing="ij",
        )
        r_int = torch.sqrt((sx * (x - cx)) ** 2 + (sy * (y - cy)) ** 2).round().long()
    else:
        oz, oy, ox = out_shape
        cz, cy, cx = out_centers
        sz, sy, sx = scales
        z, y, x = torch.meshgrid(
            torch.arange(oz, device=device),
            torch.arange(oy, device=device),
            torch.arange(ox, device=device),
            indexing="ij",
        )
        r_int = (
            torch.sqrt(
                (sz * (z - cz)) ** 2 + (sy * (y - cy)) ** 2 + (sx * (x - cx)) ** 2
            )
            .round()
            .long()
        )

    r_int = r_int.clamp(0, len(radial_mean) - 1)
    return radial_mean[r_int]


def coarse_occupancy_mask(
    volume: torch.Tensor, coarse_factor: int, full_threshold: float = 1.0
) -> torch.Tensor:
    """
    Downsample `(volume > 0)` into a coarse grid (`coarse_factor` voxels
    per axis per coarse cell) marking which coarse cells are "practically
    full" -- their occupied VOXEL FRACTION is at or above `full_threshold`.

    `full_threshold` matters a lot in practice: for spherical/irregular
    (non-space-filling) particles, a coarse cell can have a handful of
    stray never-touched voxels essentially forever (e.g. the corners
    between packed spheres), so requiring literal 100% occupancy
    (`full_threshold=1.0`) means cells almost never get marked full at
    all -- the coarse grid then provides close to zero speedup, since
    "not full" ends up meaning almost the whole grid, even once a region
    is genuinely too crowded for more content. A lower threshold (e.g.
    0.6-0.8) marks a cell full once it's merely MOSTLY occupied, which is
    what actually predicts "nothing more meaningfully fits here" for
    realistic particle shapes. This only affects candidate-search/fill
    SPEED, never correctness downstream: a cell wrongly marked full just
    gets skipped (a completeness cost, occasionally missing a spot that
    genuinely still fits or still has a little real free space), while
    any fine-grained exact test downstream is unaffected and remains
    authoritative either way.

    Vectorized via ``avg_pool3d`` (one pooling pass over the whole volume)
    rather than a Python loop over coarse cells -- cheap regardless of
    volume size.

    Volume dimensions not evenly divisible by `coarse_factor` are padded
    with "occupied" (1.0) rather than "free" -- conservative: an edge cell
    might be marked full when it actually has a little real free space
    beyond the volume's own edge, which only costs a few skipped edge
    candidates, never a false "not full" that could mask a real overlap.

    Parameters
    ----------
    volume : torch.Tensor
        Shape (nz, ny, nx).
    coarse_factor : int
        Coarse-grid downsample factor per axis.
    full_threshold : float, optional
        Occupied-voxel-fraction (within one coarse cell) at or above which
        the cell is marked full. Default 1.0 (literal 100%).

    Returns
    -------
    torch.Tensor
        Bool, shape ``(ceil(nz/coarse_factor), ceil(ny/coarse_factor),
        ceil(nx/coarse_factor))``.
    """
    occ = (volume > 0).float()
    nz, ny, nx = occ.shape
    cf = coarse_factor
    pad_z, pad_y, pad_x = (-nz) % cf, (-ny) % cf, (-nx) % cf
    if pad_z or pad_y or pad_x:
        occ = F.pad(occ, (0, pad_x, 0, pad_y, 0, pad_z), value=1.0)
    pooled = F.avg_pool3d(occ.unsqueeze(0).unsqueeze(0), kernel_size=cf, stride=cf)
    return pooled[0, 0] >= full_threshold - 1e-6


def clip_insert_bounds(
    center: Sequence[float],
    local_shape: Sequence[int],
    volume_shape: Sequence[int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    """
    Compute the destination/source slices for inserting a small array
    centered at `center` into a larger array, clipping at the boundaries.

    Shared low-level geometry for "stamp a local array into a big one at an
    arbitrary center, cropping whatever falls outside" -- used by
    crowding.py (batched, centered coordinates, additive merge) and
    specimen/tomogram/generator.py (one array at a time, absolute
    coordinates, max merge). Only the index arithmetic lives here; callers
    own the actual merge (`+=`, `torch.maximum`, ...) since that differs by
    use case.

    Parameters
    ----------
    center : sequence of float
        Index (one per axis) in the destination array where the local
        array's center voxel (``local_shape[i] // 2``) should land. Rounded
        to the nearest integer.
    local_shape : sequence of int
        Shape of the array being inserted.
    volume_shape : sequence of int
        Shape of the destination array.

    Returns
    -------
    tuple of (dst_slices, src_slices), or None
        `dst_slices`/`src_slices` are tuples of `slice`, one per axis, ready
        to index the destination/source arrays directly (e.g.
        ``volume[dst_slices] += local[src_slices]``). None if the local
        array falls entirely outside the volume's bounds.
    """
    dst_slices = []
    src_slices = []
    for c, lp, vp in zip(center, local_shape, volume_shape):
        half = lp // 2
        d0 = int(round(c)) - half
        d1 = d0 + lp
        d0_clip, d1_clip = max(d0, 0), min(d1, vp)
        if d1_clip <= d0_clip:
            return None
        dst_slices.append(slice(d0_clip, d1_clip))
        src_slices.append(slice(d0_clip - d0, lp - (d1 - d1_clip)))
    return tuple(dst_slices), tuple(src_slices)


def center_crop(
    data: torch.Tensor,
    size: int | tuple[int, ...],
    dim: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Center crop a tensor along the specified axes (supports negative axes).

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of arbitrary shape
    size : int or tuple of int
        Desired crop size. If int, all axes in `dim` use the same size.
        If tuple, length must match number of axes in `dim`.
    dim : int, sequence of int, or None
        Axes along which to crop. Can be negative. Defaults to all dimensions.

    Returns
    -------
    torch.Tensor
        Center-cropped tensor
    """
    # default to all dimensions if not specified
    if dim is None:
        dim = list(range(data.ndim))
    # normalize dim to list
    elif isinstance(dim, int):
        dim = [dim]
    dim = [d + data.ndim if d < 0 else d for d in dim]  # handle negative axes

    # normalize size to list
    if isinstance(size, int):
        crop_size = [size] * len(dim)
    else:
        if len(size) != len(dim):
            raise ValueError(
                f"Length of size {len(size)} must match number of dims {len(dim)}"
            )
        crop_size = list(size)

    slices = [slice(None)] * data.ndim  # default: keep all elements
    for d, cs in zip(dim, crop_size):
        L = data.shape[d]
        if cs > L:
            raise ValueError(f"Crop size {cs} is larger than axis {d} length {L}")
        start = (L - cs) // 2
        slices[d] = slice(start, start + cs)

    return data[tuple(slices)]
