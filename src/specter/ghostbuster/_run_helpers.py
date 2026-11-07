"""
Device resolution for the two Ghostbuster pipelines.
"""

from __future__ import annotations

from typing import Sequence

import torch

# ---------------------------------------------------------------------------
# Shared device resolution for Ghostbuster/TomogramGhostbuster run()/test_run()
# ---------------------------------------------------------------------------


def resolve_device(device: int | Sequence[int] | str) -> tuple[bool, int | list[int]]:
    """
    Split a ``device`` argument into a GPU/CPU decision and the device ids.

    ``"cpu"`` is the only spelling that forces the CPU; every other value
    targets the GPU when CUDA is available and silently falls back to the CPU
    when it is not. Without the explicit ``"cpu"`` case, ``run(device=...)``
    had no way to express "run on the CPU" on a machine that has a GPU, since
    the decision was taken from ``torch.cuda.is_available()`` alone.

    Parameters
    ----------
    device : int or sequence of int or str
        A GPU index, a sequence of GPU indices for data-parallel training, or the
        string ``"cpu"``.

    Returns
    -------
    use_gpu : bool
        Whether to target a GPU.
    device : int or list of int
        The device ids, unchanged except that ``"cpu"`` becomes ``0`` (which
        the fit ignores once ``use_gpu`` is False).
    """
    if isinstance(device, str):
        if device != "cpu":
            raise ValueError(
                f"device={device!r} is invalid: the only string accepted is "
                "'cpu'; pass a GPU index (0) or a list of them ([0, 1]) to "
                "target GPUs."
            )
        return False, 0
    return torch.cuda.is_available(), (
        device if isinstance(device, int) else list(device)
    )
