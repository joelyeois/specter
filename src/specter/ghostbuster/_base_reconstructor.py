"""
`_BaseReconstructor`: the training scaffolding `Reconstructor` and
`TomogramReconstructor` share -- optimiser and scheduler construction, the
manual optimisation step, k-space masking after each update, run-directory
handling and metrics logging. The loop driving it is `_training.fit`.
"""

from __future__ import annotations

from specter import logger

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from ._helpers import (
    _apply_kmask_inplace,
    _build_epoch_metrics,
    _build_lr_scheduler,
    _kmask_half_spectrum,
    _log_current_lr,
)
from ._training import TrainingState, all_reduce_sum
from .._device_module import DeviceModule
from ..io import write_mrc
from specter.options import Scheduler

# ---------------------------------------------------------------------------
# Shared scaffolding for Reconstructor and TomogramReconstructor
# ---------------------------------------------------------------------------


class _BaseReconstructor(DeviceModule):
    """
    Shared optimisation-loop scaffolding for volume-reconstruction modules.

    Both :class:`~specter.ghostbuster.Reconstructor` and
    :class:`~specter.ghostbuster.TomogramReconstructor` refine a volume
    parameter ``V`` with manual optimisation and need the same LR schedule,
    step counting, per-batch k-mask projection, and metrics logging. This
    base class holds that shared behaviour so each subclass only implements
    its own forward model and loss.

    Subclasses must set ``self.lr_decay``, ``self.lr``, ``self.scheduler``,
    ``self.kmask``, ``self.V``, ``self.voxel_size``, ``self._run_dir``, and
    call `_init_loss_logs` before these methods are used.
    """

    lr_decay: float
    lr: float | None
    scheduler: Scheduler
    kmask: torch.Tensor | None
    V: torch.Tensor
    voxel_size: float
    _run_dir: Path | None
    log_lrs: list[float]
    log_total_loss: list[torch.Tensor]
    hparams: dict[str, Any]
    #: The fit in progress or last run, set by `_training.fit`; None before.
    _training: TrainingState | None = None

    def _save_hyperparameters(
        self,
        init: Callable[..., None],
        init_locals: dict[str, Any],
        ignore: Sequence[str],
    ) -> None:
        """
        Record ``init``'s arguments, as passed, in ``self.hparams``.

        Parameters
        ----------
        init : callable
            The ``__init__`` calling this, whose signature names the arguments.
        init_locals : dict
            That call's ``locals()``.
        ignore : sequence of str
            Arguments left out (tensors and paths that are saved otherwise).
        """
        self.hparams = {
            name: init_locals[name]
            for name, p in inspect.signature(init).parameters.items()
            if name != "self"
            and name not in ignore
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }

    @property
    def current_epoch(self) -> int:
        """The epoch being trained, from 0; 0 outside a fit."""
        return self._training.epoch if self._training is not None else 0

    @property
    def global_step(self) -> int:
        """Optimiser steps taken, one per optimiser per batch; 0 outside a fit."""
        return self._training.global_step if self._training is not None else 0

    @property
    def is_global_zero(self) -> bool:
        """Whether this process is rank 0 (or the only one): the one that writes."""
        return self._training is None or self._training.rank == 0

    def _init_loss_logs(self) -> None:
        """Start the per-step loss and learning-rate histories."""
        self.log_lrs = []
        self.log_total_loss = []

    def _volume_optimizer(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        """
        The AdamW optimiser on ``V`` and its LR scheduler, as the lists
        `configure_optimizers` returns; both empty when ``lr`` is None.

        Not ``fused=True``: it is ~8 ms faster per step on a 512^3 volume,
        but Lightning's 16-mixed plugin, which this loop's ``GradScaler``
        replaced, handed it grads it rejected ("params, grads, exp_avgs, and
        exp_avg_sqs must have same dtype, device, and layout"); not
        re-measured since.
        """
        if self.lr is None:
            return [], []
        optimizer = AdamW([self.V], lr=self.lr, weight_decay=0.0)
        lr_scheduler = _build_lr_scheduler(
            self.scheduler,
            optimizer,
            self.lr,
            self.reciprocal_lr_scheduler,
            self.n_training_steps_per_epoch,
            self.n_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def _batch_total(self, n_local: int) -> int:
        """
        Particles in the step's whole batch, of which this rank holds `n_local`.

        `n_local` itself on one device. Under several ranks, the global batch
        the loop split (`TrainingState.global_batch_size`): a rank's loss,
        a mean over its share, is weighted by ``n_local / total`` so that the
        ranks' summed gradients are the global batch's mean.
        """
        state = self._training
        if state is None or state.world_size <= 1:
            return n_local
        return state.global_batch_size

    def _idle_step(self) -> torch.Tensor:
        """
        The step of a rank whose share of the batch is empty.

        It contributes no gradient and no loss but joins the step's
        collectives, which every rank must, and steps its optimisers on the
        summed gradient as the others do. Returns the batch's logged loss.
        """
        state = self._training
        assert state is not None, "_idle_step runs inside `_training.fit`"
        self._zero_grad()
        zero = torch.zeros((), device=self.V.device)
        if state.scaler is not None:
            # The scaler sets its scale on its first `scale`; a rank whose
            # first share is empty would otherwise reach `step` without one.
            state.scaler.scale(zero)
        logged = self._gather_for_logging(zero)
        self.log_total_loss.append(logged)
        self._step()
        return logged

    def _optimise(self, loss: torch.Tensor) -> None:
        """
        One manual optimisation step on ``loss``: zero every optimiser,
        backpropagate (scaled under ``"16-mixed"``), sum the gradients
        across ranks, step every optimiser, then the schedulers.

        The scaler is updated after each optimiser's step and ``global_step``
        advances once per optimiser, as under Lightning's manual optimisation.
        `_zero_grad`, `_backward` and `_step` are its three parts, for a
        caller that accumulates the gradient of several backwards first.
        """
        self._zero_grad()
        self._backward(loss)
        self._step()

    def _zero_grad(self) -> None:
        """Zero the gradients of every optimiser's parameters."""
        state = self._training
        assert state is not None, "_zero_grad runs inside `_training.fit`"
        for opt in state.optimizers:
            opt.zero_grad()

    def _backward(self, loss: torch.Tensor) -> None:
        """
        Add ``loss``'s gradient to the parameters' (scaled under ``"16-mixed"``).

        Every backward of one step goes through the same scaler, whose scale
        changes only in `_step`, so the accumulated gradient unscales as one.
        """
        state = self._training
        assert state is not None, "_backward runs inside `_training.fit`"
        (state.scaler.scale(loss) if state.scaler is not None else loss).backward()

    def _step(self) -> None:
        """
        Sum the accumulated gradients across ranks, step every optimiser
        (updating the scaler after each), then the schedulers.
        """
        state = self._training
        assert state is not None, "_step runs inside `_training.fit`"
        if state.world_size > 1:
            self._sum_gradients()
        for opt in state.optimizers:
            if state.scaler is not None:
                state.scaler.step(opt)
                state.scaler.update()
            else:
                opt.step()
            state.global_step += 1
        self._step_schedulers()

    def _sum_gradients(self) -> None:
        """
        Sum every trainable parameter's gradient across the ranks, in place.

        Each rank's loss is weighted by its share of the global batch
        (`_batch_total`), so the sum is the global batch's gradient. Every
        rank must reduce the same tensors in the same order, but a rank
        whose share was empty has no gradient at all; it reduces zeros. A
        parameter no rank has a gradient for keeps none, as on one device,
        where the optimiser then skips it rather than stepping on a zero.
        Under ``"16-mixed"`` the gradients are scaled by the same factor on
        every rank, so the scaler's overflow check on the sum is one
        decision for all of them.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        present = torch.tensor(
            [float(p.grad is not None) for p in params], device=self.V.device
        )
        dist.all_reduce(present)
        for p, count in zip(params, present.tolist(), strict=True):
            if count == 0:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)

    def _make_run_dir(self) -> bool:
        """
        Create ``run_dir`` and its ``epochs/`` folder if this rank writes.

        Returns False when there is no run directory or this is not the
        global-zero rank: under multi-GPU every replica would
        otherwise race to create and write the same files.
        """
        if self._run_dir is None or not self.is_global_zero:
            return False
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "epochs").mkdir(exist_ok=True)
        return True

    def _volume_cpu(self) -> torch.Tensor:
        """``V`` detached, on the host, in float32, for saving and FSCs."""
        return self.V.detach().cpu().float()

    def _save_volume(self, v: torch.Tensor, path: Path) -> None:
        """Write ``v`` as an MRC at this module's voxel size."""
        write_mrc(path, v, self.voxel_size)

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
        schedulers = self._training.schedulers if self._training is not None else []
        _log_current_lr(schedulers, self.lr, self.log_lrs)

    def on_train_batch_end(self, batch: Any, batch_idx: int) -> None:
        """Apply the Fourier-space k-mask to V after each gradient update."""
        _apply_kmask_inplace(self.V, self._kmask_half())

    def _kmask_half(self) -> torch.Tensor | None:
        """``self.kmask`` as an unshifted half spectrum, built once per mask
        tensor and device (see :func:`_kmask_half_spectrum`)."""
        kmask = self.kmask
        if kmask is None:
            return None
        key = (kmask.data_ptr(), kmask.device, tuple(kmask.shape))
        cached = getattr(self, "_kmask_half_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, _kmask_half_spectrum(kmask))
            self._kmask_half_cache = cached
        return cached[1]

    def _fit_state(self) -> TrainingState:
        """The fit in progress; the step counts below need one."""
        if self._training is None:
            raise RuntimeError(
                "The schedule's step count is known only inside a fit "
                "(`specter.ghostbuster._training.fit`)."
            )
        return self._training

    def n_training_steps_per_epoch(self) -> int:
        """
        Return the number of optimizer steps per training epoch.

        Returns
        -------
        int
            Steps per epoch: this rank's batches.
        """
        return self._fit_state().batches_per_epoch

    def n_training_steps(self) -> int:
        """
        Return the total number of optimizer steps in the configured run.

        Returns
        -------
        int
            Total scheduler steps across all epochs.
        """
        max_epochs = self._fit_state().max_epochs
        if max_epochs < 1:
            raise ValueError("OneCycleLR requires a positive max_epochs.")
        return self.n_training_steps_per_epoch() * max_epochs

    def _metrics_path_suffix(self) -> str:
        """Filename suffix for saved metrics. Override for e.g. halfset labelling."""
        return ""

    def _step_schedulers(self) -> None:
        """
        Step every LR scheduler once.

        No-op when no scheduler is configured (``self.lr is None``).
        ``ReduceLROnPlateau`` needs a metric argument this single-step call
        can't supply, but ``_build_lr_scheduler`` never constructs one --
        encountering one here would mean a caller bug, not a normal runtime
        condition.
        """
        assert self._training is not None
        for s in self._training.schedulers:
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

        Under several ranks, each rank's ``value`` is its share-weighted
        loss (see `_batch_total`), so the sum across ranks is the global
        batch's loss, as the gradient all-reduce in `_step` sums the
        gradients. A no-op (returns ``value`` detached) on one device, or
        outside a fit (e.g. calling ``_compute_loss`` directly).

        Only ever call this on a value being logged/recorded. Never use its
        result in place of the original ``value`` for ``backward()`` --
        `_step` already sums the gradients across ranks, and summing the
        loss here before backward would count every rank's share again.
        """
        if self._training is None or self._training.world_size <= 1:
            return value.detach()
        return all_reduce_sum(value.detach().clone(), self._training.world_size)

    def _save_metrics(
        self,
        include_loss_std: bool = False,
        include_lr_min: bool = False,
        write: bool = True,
    ) -> dict[str, Any] | None:
        """Build training metrics (loss, lr) and, by default, save them to JSON.

        Rank-0-only under multi-GPU to avoid every replica racing to
        write the same file. ``log_total_loss`` is
        summed across ranks at the point it is recorded (see
        ``_gather_for_logging``), so under multi-GPU the saved metrics
        reflect the true global-batch loss, not just rank 0's local view.
        ``log_lrs`` needs no such gathering -- the learning rate is a
        deterministic function of step count, not of data.

        Parameters
        ----------
        include_loss_std : bool
            If True, include the per-epoch loss standard deviation.
        include_lr_min : bool
            If True, include the per-epoch minimum learning rate.
        write : bool
            Write ``metrics<suffix>.json`` to ``self._run_dir``. ``Reconstructor``
            passes ``False`` and folds the returned dict into a single
            ``job.json`` instead, rather than one metrics file per halfset.
        """
        if (
            self._run_dir is None
            or not self.log_total_loss
            or not self.is_global_zero
            or self._training is None
        ):
            return None

        meta = _build_epoch_metrics(
            self.log_total_loss,
            self.log_lrs,
            self._training.batches_per_epoch,
            include_loss_std=include_loss_std,
            include_lr_min=include_lr_min,
        )
        if not write:
            return meta

        suffix = self._metrics_path_suffix()
        metrics_path = self._run_dir / f"metrics{suffix}.json"
        metrics_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved metrics → {metrics_path}")
        return meta
