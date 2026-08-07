from __future__ import annotations

from typing import Any, Sequence, cast

import lightning as L

# ---------------------------------------------------------------------------
# Shared Trainer construction for Ghostbuster/TomogramGhostbuster run()/test_run()
# ---------------------------------------------------------------------------


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
    return L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=device_list if use_gpu else 1,
        strategy="ddp" if multi_gpu else "auto",
        max_epochs=max_epochs,
        precision=cast(Any, precision if use_gpu else "32"),
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks or [],
    )
