from __future__ import annotations

from typing import Any, cast

import lightning as L

# ---------------------------------------------------------------------------
# Shared Trainer construction for Ghostbuster/TomogramGhostbuster run()/test_run()
# ---------------------------------------------------------------------------


def build_trainer(
    use_gpu: bool,
    device: int,
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
    device : int
        GPU index. Ignored when ``use_gpu`` is False.
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
        manages its own volume/metrics output via ``run_dir``).
    """
    return L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=[device] if use_gpu else 1,
        max_epochs=max_epochs,
        precision=cast(Any, precision if use_gpu else "32"),
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks or [],
    )
