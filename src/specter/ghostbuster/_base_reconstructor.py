from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import torch

from ._helpers import _apply_kmask_inplace, _build_epoch_metrics, _log_current_lr

# ---------------------------------------------------------------------------
# Shared LightningModule scaffolding for Reconstructor and TomogramReconstructor
# ---------------------------------------------------------------------------


class _BaseReconstructor(L.LightningModule):
    """
    Shared optimisation-loop scaffolding for volume-reconstruction modules.

    Both :class:`~specter.ghostbuster.Reconstructor` and
    :class:`~specter.ghostbuster.TomogramReconstructor` refine a volume
    parameter ``V`` with manual optimisation and need the same LR schedule,
    step counting, per-batch k-mask projection, and metrics logging. This
    base class holds that shared behaviour so each subclass only implements
    its own forward model and loss.

    Subclasses must set ``self.lr_decay``, ``self.lr``, ``self.kmask``,
    ``self.V``, ``self._run_dir``, and the ``log_total_loss``/
    ``log_norm_loss``/``log_sparsity_loss``/``log_lrs`` lists before these
    methods are used.
    """

    lr_decay: float
    lr: float | None
    kmask: torch.Tensor | None
    V: torch.Tensor
    _run_dir: Path | None
    log_lrs: list[float]
    log_total_loss: list[torch.Tensor]
    log_norm_loss: list[torch.Tensor]
    log_sparsity_loss: list[torch.Tensor]

    def reciprocal_lr_scheduler(self, *args: Any) -> float:
        """
        Reciprocal-square-root decay schedule: ``1 / (1 + decay * step^0.5)``.

        Returns
        -------
        float
            LR multiplier at the current global step.
        """
        return 1 / (1 + self.lr_decay * self.global_step**0.5)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Record the current learning rate before each batch."""
        _log_current_lr(self.trainer, self.lr, self.log_lrs)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Apply the Fourier-space k-mask to V after each gradient update."""
        _apply_kmask_inplace(self.V, self.kmask)

    def num_training_steps_per_epoch(self) -> int:
        """
        Return the number of optimizer steps per training epoch.

        Returns
        -------
        int
            Steps per epoch, accounting for gradient accumulation.
        """
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        assert self.trainer.train_dataloader is not None
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches

        return num_steps

    def num_training_steps(self) -> int:
        """
        Return the total number of optimizer steps in the configured run.

        Returns
        -------
        int
            Total scheduler steps across all epochs.
        """
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs < 1:
            raise ValueError("OneCycleLR requires a positive trainer.max_epochs.")

        return self.num_training_steps_per_epoch() * max_epochs

    def _metrics_path_suffix(self) -> str:
        """Filename suffix for saved metrics. Override for e.g. halfset labelling."""
        return ""

    def _save_metrics(
        self, include_loss_std: bool = False, include_lr_min: bool = False
    ) -> None:
        """Save training metrics (loss, lr) to JSON."""
        if self._run_dir is None or not self.log_total_loss:
            return

        suffix = self._metrics_path_suffix()
        metrics_path = self._run_dir / f"metrics{suffix}.json"
        meta = _build_epoch_metrics(
            self.log_total_loss,
            self.log_norm_loss,
            self.log_sparsity_loss,
            self.log_lrs,
            self.current_epoch,
            include_loss_std=include_loss_std,
            include_lr_min=include_lr_min,
        )
        metrics_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metrics → {metrics_path}")
