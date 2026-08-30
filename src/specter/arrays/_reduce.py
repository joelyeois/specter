"""Memory-bounded reductions over volume-sized tensors."""

from __future__ import annotations

import torch

#: Elements per slab in :func:`count_nonzero_chunked`. 8M keeps the int64
#: promotion torch performs inside ``sum`` to 64 MB regardless of input size.
_COUNT_SLAB_ELEMENTS = 8_000_000


def count_nonzero_chunked(
    tensor: torch.Tensor, slab_elements: int = _COUNT_SLAB_ELEMENTS
) -> int:
    """
    Count nonzero entries without promoting the whole tensor at once.

    ``mask.sum()`` on a bool tensor promotes every element to int64 before
    reducing, so counting the True voxels of a 300x1200x1200 mask -- 0.40
    GiB as bool -- transiently allocates 3.22 GiB. ``torch.count_nonzero``
    measures the same 3.22 GiB, and ``sum(dtype=torch.int32)`` only halves
    it. Profiled on a default ``specter build tomogram``, one such call was
    the single largest allocation in the entire run.

    Reducing in slabs bounds that promotion to `slab_elements` at a time,
    for one Python-level iteration and device sync per slab. Worth it only
    on volume-sized masks: a per-instance footprint is cheaper to sum
    directly, and this would add syncs for nothing.

    Parameters
    ----------
    tensor : torch.Tensor
        Any dtype; bool masks are the intended case. Reshaped to 1D, which
        is a view for the contiguous tensors this is meant for.
    slab_elements : int, optional
        Elements reduced per step. Default :data:`_COUNT_SLAB_ELEMENTS`.

    Returns
    -------
    int
        Number of nonzero entries.
    """
    if slab_elements <= 0:
        raise ValueError(f"slab_elements must be positive, got {slab_elements}")
    flat = tensor.reshape(-1)
    total = 0
    for start in range(0, flat.numel(), slab_elements):
        chunk = flat[start : start + slab_elements]
        total += int((chunk != 0).sum(dtype=torch.int64))
    return total
