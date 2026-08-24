from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    def _optimizers_list(self) -> list[Any]:
        """Normalise ``self.optimizers()`` (Lightning returns one or a list) to a list."""
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        return list(opts)

    def _step_schedulers(self) -> None:
        """
        Step every LR scheduler once, for manual optimisation.

        No-op under automatic optimisation or when no scheduler is
        configured (``self.lr is None``). ``ReduceLROnPlateau`` needs a
        metric argument this single-step call can't supply, but
        ``_build_lr_scheduler`` never constructs one -- encountering one
        here would mean a caller bug, not a normal runtime condition.
        """
        if self.automatic_optimization:
            return
        sch = self.lr_schedulers()
        if not sch:
            return
        if not isinstance(sch, (list, tuple)):
            sch = [sch]
        for s in sch:
            if isinstance(s, ReduceLROnPlateau):
                raise TypeError(
                    "ReduceLROnPlateau is not supported by this manual "
                    "scheduler step loop; _build_lr_scheduler never "
                    "constructs one."
                )
            s.step()

    def _gather_for_logging(self, value: torch.Tensor) -> torch.Tensor:
        """
        Reduce a per-rank scalar to a single global value, for logging only.

        Under multi-GPU (DDP), each rank computes ``value`` from a different
        local batch -- averaging across ranks here (matching DDP's own
        gradient all-reduce semantics) makes the logged value reflect the
        true global-batch loss, not just one rank's local view. A no-op
        (returns ``value`` unchanged) outside DDP, or when no ``Trainer`` is
        attached at all (e.g. calling ``_compute_loss``/``training_step``
        directly without going through ``Trainer.fit()``) -- checks the
        private ``_trainer`` attribute directly rather than the ``trainer``
        property, since that property *raises* (rather than returning
        ``None``) when unattached.

        Only ever call this on a value being logged/recorded. Never use its
        result in place of the original ``value`` for ``backward()`` -- DDP
        already synchronises gradients correctly via its own hook, and
        averaging the loss a second time here before backward would double
        that averaging and silently change the effective learning rate.
        """
        if self._trainer is None or self.trainer.world_size <= 1:
            return value.detach()
        # self.all_gather's signature also accepts/returns dict/list/tuple
        # (for gathering nested structures); we only ever pass a Tensor, so
        # the result is always a Tensor too.
        gathered = cast(torch.Tensor, self.all_gather(value.detach()))
        return gathered.mean()

    def _save_metrics(
        self,
        include_loss_std: bool = False,
        include_lr_min: bool = False,
        write: bool = True,
    ) -> dict[str, Any] | None:
        """Build training metrics (loss, lr) and, by default, save them to JSON.

        Rank-0-only under multi-GPU (DDP) to avoid every replica racing to
        write the same file. ``log_total_loss``/``log_norm_loss`` are
        gathered/averaged across ranks at the point they're recorded (see
        ``_gather_for_logging``), so under DDP the saved metrics reflect the
        true global-batch loss, not just rank 0's local view.
        ``log_sparsity_loss`` needs no such gathering -- it depends only on
        the already-synced volume ``V``, identical on every rank already.
        ``log_lrs`` likewise needs none -- the learning rate is a
        deterministic function of step count, not of data.

        Parameters
        ----------
        write : bool
            Write ``metrics<suffix>.json`` to ``self._run_dir``. ``Reconstructor``
            passes ``False`` and folds the returned dict into a single
            ``job.json`` instead, rather than one metrics file per halfset.
        """
        if (
            self._run_dir is None
            or not self.log_total_loss
            or not self.trainer.is_global_zero
        ):
            return None

        meta = _build_epoch_metrics(
            self.log_total_loss,
            self.log_norm_loss,
            self.log_sparsity_loss,
            self.log_lrs,
            self.current_epoch,
            include_loss_std=include_loss_std,
            include_lr_min=include_lr_min,
        )
        if not write:
            return meta

        suffix = self._metrics_path_suffix()
        metrics_path = self._run_dir / f"metrics{suffix}.json"
        metrics_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metrics → {metrics_path}")
        return meta
