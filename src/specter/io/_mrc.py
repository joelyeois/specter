"""
Whole-file MRC reading and writing: an array and its voxel size.

Every file specter writes has a zero header origin and the standard axis
order (columns x, rows y, sections z), and says whether it is a volume or a
stack of images. The origin stays zero because nothing specter writes has a
frame other than its own box, whose centre is voxel ``n // 2``: a map and a
reconstruction of it then overlay in a viewer, as do maps from CryoSPARC and
RELION, which write the same.

Stacks too large to hold, written slab by slab or read through a memory map
(`io._reorder`, `pipelines._tomogram`, `io._images`), use mrcfile directly.
"""

from __future__ import annotations

import os
import warnings

import mrcfile
import numpy as np
import torch


def write_mrc(
    path: str | os.PathLike[str],
    data: torch.Tensor | np.ndarray,
    voxel_size: float | None = None,
    *,
    stack: bool | None = None,
) -> None:
    """
    Write an array to an MRC file as float32, replacing any file there.

    Parameters
    ----------
    path : str or path-like
        Output ``.mrc``/``.mrcs`` path.
    data : torch.Tensor or numpy.ndarray
        A volume ``(Z, Y, X)`` or an image stack ``(N, Y, X)``; a tensor is
        detached and moved to the host.
    voxel_size : float, optional
        Voxel (pixel) size in Å for the header. None leaves the header's
        default of 1, which no reader can tell from a real 1 Å.
    stack : bool, optional
        Whether a 3-D array is a stack of images rather than a volume, which
        the header records (space group 0, one section per cell) and viewers
        act on. None (default) goes by the extension: a stack for ``.mrcs``,
        a volume otherwise. A stack of one image is left unmarked, so that it
        reads back with its leading axis.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    with mrcfile.new(os.fspath(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        if stack is None:
            stack = os.fspath(path).endswith(".mrcs")
        # A stack of one stays untagged: with space group 0 and one section
        # the format has no way to say "stack", and readers (mrcfile among
        # them) return a bare 2-D image, dropping the leading axis.
        if stack and mrc.data.ndim == 3 and mrc.data.shape[0] > 1:
            mrc.set_image_stack()
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


def read_mrc(path: str | os.PathLike[str]) -> tuple[torch.Tensor, float]:
    """
    Read a whole MRC file into a float32 tensor, with its voxel size.

    Opened permissively, so a file with a non-standard header (as some
    packages write) still loads.

    Parameters
    ----------
    path : str or path-like
        Input ``.mrc``/``.mrcs`` path.

    Returns
    -------
    data : torch.Tensor
        The file's array, float32, on the host.
    voxel_size : float
        The header's x voxel size in Å.

    Warns
    -----
    UserWarning
        When the header declares a non-standard axis order, or voxel sizes
        that differ between axes; the array is returned as stored in both
        cases.
    """
    with mrcfile.open(os.fspath(path), permissive=True) as mrc:
        axes = tuple(int(mrc.header[k]) for k in ("mapc", "mapr", "maps"))
        # Only a header that names a real, different order: a zeroed or
        # otherwise unreadable one is what `permissive` is for.
        if sorted(axes) == [1, 2, 3] and axes != (1, 2, 3):
            warnings.warn(
                f"{os.fspath(path)}: the MRC header gives the axis order "
                f"(mapc, mapr, maps) = {axes}, not the standard (1, 2, 3). "
                "The array is read as stored and is not transposed, so its "
                "axes are not (Z, Y, X); rewrite the file in the standard "
                "order first.",
                stacklevel=2,
            )
        size = mrc.voxel_size
        # The z size of a stack of images means nothing, and programs write
        # anything there; an axis the header leaves at zero is unset.
        in_plane = mrc.data.ndim < 3 or mrc.is_image_stack()
        sizes = [float(size.x), float(size.y)] + ([] if in_plane else [float(size.z)])
        sizes = [s for s in sizes if s > 0]
        if sizes and max(sizes) - min(sizes) > 1e-3 * max(sizes):
            warnings.warn(
                f"{os.fspath(path)}: the MRC header gives unequal voxel sizes "
                f"(x, y, z) = ({size.x:g}, {size.y:g}, {size.z:g}) A. specter "
                f"works on cubic voxels and takes the x size, {size.x:g} A, for "
                "every axis, which distorts the object; resample the file to "
                "cubic voxels first.",
                stacklevel=2,
            )
        data = torch.from_numpy(np.array(mrc.data, dtype=np.float32))
        return data, float(size.x)
