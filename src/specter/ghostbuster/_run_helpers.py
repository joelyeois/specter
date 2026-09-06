"""
Device resolution and Lightning `Trainer` construction for the two
Ghostbuster pipelines.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
import torch

# ---------------------------------------------------------------------------
# Shared Trainer construction for Ghostbuster/TomogramGhostbuster run()/test_run()
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
        A GPU index, a sequence of GPU indices for multi-GPU DDP, or the
        string ``"cpu"``.

    Returns
    -------
    use_gpu : bool
        Whether to target a GPU, as :func:`build_trainer` expects it.
    device : int or list of int
        The device ids, unchanged except that ``"cpu"`` becomes ``0`` (which
        ``build_trainer`` ignores once ``use_gpu`` is False).
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


def build_trainer(
    use_gpu: bool,
    device: int | Sequence[int],
    max_epochs: int,
    precision: str,
    callbacks: list[Any] | None = None,
) -> L.Trainer:
    """
    Build a Lightning ``Trainer`` for a Ghostbuster run, with GPU/CPU dispatch.

    Parameters
    ----------
    use_gpu : bool
        Whether to target a GPU device (typically ``torch.cuda.is_available()``).
    device : int or sequence of int
        GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to train
        across multiple GPUs via Lightning DDP (``strategy="ddp"``).
        A single-element sequence behaves like a bare int. Ignored when
        ``use_gpu`` is False.
    max_epochs : int
        Number of training epochs.
    precision : str
        Requested precision (e.g. ``"16-mixed"``, ``"32"``). Falls back to
        ``"32"`` automatically when ``use_gpu`` is False.
    callbacks : list, optional
        Additional Lightning callbacks.

    Returns
    -------
    L.Trainer
        Configured trainer with no logger and no checkpointing (Ghostbuster
        manages its own volume/metrics output via ``run_dir``). Multi-GPU
        runs are gradient-synchronised (DDP all-reduce) every step, so the
        volume and any other trainable parameters stay identical across
        replicas -- but file/console output is emitted rank-0-only (see the
        ``is_global_zero`` guards in ``Reconstructor``/``TomogramReconstructor``
        fit hooks) to avoid every replica racing to write the same files.
    """
    device_list = [device] if isinstance(device, int) else list(device)
    multi_gpu = use_gpu and len(device_list) > 1
    # The progress bar reads the logged loss with `.item()` on every
    # refresh, a device sync; at the default refresh of every step that
    # serialised the CPU's kernel launches behind the GPU. Every tenth step
    # is the same bar to a human.
    callbacks = [TQDMProgressBar(refresh_rate=10), *(callbacks or [])]
    return L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=device_list if use_gpu else 1,
        strategy="ddp" if multi_gpu else "auto",
        max_epochs=max_epochs,
        precision=cast(Any, precision if use_gpu else "32"),
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
