"""
Whole-file MRC reading and writing: an array and its voxel size.

Stacks too large to hold, written slab by slab or read through a memory map
(`io._reorder`, `pipelines._tomogram`, `io._images`), use mrcfile directly.
"""

from __future__ import annotations

import os

import mrcfile
import numpy as np
import torch


def write_mrc(
    path: str | os.PathLike[str],
    data: torch.Tensor | np.ndarray,
    voxel_size: float | None = None,
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
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    with mrcfile.new(os.fspath(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
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
    """
    with mrcfile.open(os.fspath(path), permissive=True) as mrc:
        data = torch.from_numpy(np.array(mrc.data, dtype=np.float32))
        return data, float(mrc.voxel_size.x)
