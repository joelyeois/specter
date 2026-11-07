"""
The loop both reconstructors train under: epochs over a loader, the manual
optimisation step under optional mixed precision, and data parallelism over
several GPUs, one process per device.

It replaced Lightning's ``Trainer`` on 2026-11-07 and keeps its behaviour
wherever a reconstruction can see it:

* hooks run in Lightning's order, each callback's before the model's, and
  the model and its optimisers are moved to the CPU *before* ``on_fit_end``;
* ``global_step`` counts optimiser steps, one per optimiser per batch, which
  is what the reciprocal LR schedule decays with;
* ``"16-mixed"`` is ``torch.autocast`` around ``training_step`` (backward and
  optimiser step included) with a ``GradScaler`` updated after every
  optimiser's step, ``"bf16-mixed"`` the autocast alone, and both fall back
  to ``"32"`` on the CPU;
* several GPUs split each epoch with a ``DistributedSampler`` (seed 0,
  reshuffled every epoch) and average the gradients across ranks after each
  backward, as DDP's all-reduce does.

Unlike Lightning's launcher, which re-ran the whole calling script once per
extra GPU, only the fit is replicated: this process is rank 0 and returns
the trained model, and each other rank is a spawned process that receives
the model and the loader's dataset by pickling (CPU tensors through shared
memory) and exits when the fit ends. Callbacks run on rank 0 only.
"""

from __future__ import annotations

import contextlib
import socket
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from ._run_helpers import resolve_device

#: Autocast dtype of each supported precision; None trains in float32.
_AUTOCAST: dict[str, torch.dtype | None] = {
    "16-mixed": torch.float16,
    "bf16-mixed": torch.bfloat16,
    "32": None,
    "32-true": None,
}

#: Progress-bar loss refresh interval, in batches. Reading the loss is a
#: device sync; every tenth step is the same bar to a human and keeps the
#: CPU's kernel launches ahead of the GPU.
_PROGRESS_LOSS_EVERY = 10

_M = TypeVar("_M", bound=nn.Module)


@dataclass
class TrainingState:
    """
    What a model being fitted can ask of its loop.

    Attributes
    ----------
    max_epochs : int
        Epochs in the fit.
    batches_per_epoch : int
        Batches each epoch takes on this rank.
    rank, world_size : int
        This process's place among the ranks, and how many there are.
    optimizers, schedulers : list
        What the model's ``configure_optimizers`` returned.
    scaler : torch.amp.GradScaler or None
        The loss scaler under ``"16-mixed"``.
    epoch : int
        The current epoch, from 0.
    global_step : int
        Optimiser steps taken so far, one per optimiser per batch.
    """

    max_epochs: int
    batches_per_epoch: int
    rank: int = 0
    world_size: int = 1
    optimizers: list[torch.optim.Optimizer] = field(default_factory=list)
    schedulers: list[Any] = field(default_factory=list)
    scaler: torch.amp.GradScaler | None = None
    epoch: int = 0
    global_step: int = 0


def fit(
    model: _M,
    loader: DataLoader[Any],
    *,
    max_epochs: int,
    device: int | Sequence[int] | str = 0,
    precision: str = "32",
    callbacks: Sequence[Any] | None = None,
    progress: bool = True,
) -> _M:
    """
    Train ``model`` on ``loader`` for ``max_epochs`` epochs.

    Parameters
    ----------
    model : nn.Module
        A reconstructor: it provides ``configure_optimizers``,
        ``training_step(batch, batch_idx)`` and any of the ``on_fit_start``,
        ``on_train_batch_start``, ``on_train_batch_end``,
        ``on_train_epoch_end``, ``on_train_end`` and ``on_fit_end`` hooks,
        and reads the loop through a ``_training`` attribute set here.
    loader : DataLoader
        The training batches. With several GPUs, its dataset, batch size,
        worker count and shuffling are re-read into one loader per rank.
    max_epochs : int
        Number of epochs.
    device : int or sequence of int or {"cpu"}
        A GPU index, several (data-parallel, one process each), or ``"cpu"``;
        as for :meth:`~specter.ghostbuster.Ghostbuster.run`, a GPU request
        falls back to the CPU when CUDA is unavailable.
    precision : {"16-mixed", "bf16-mixed", "32", "32-true"}
        Training precision; forced to ``"32"`` on the CPU.
    callbacks : sequence, optional
        Objects with any of the hook methods above, each taking the model
        (and, for the batch hooks, ``batch, batch_idx``); called before the
        model's own hook, on rank 0 only.
    progress : bool
        Show a progress bar per epoch.

    Returns
    -------
    nn.Module
        ``model``, trained, on the CPU (rank 0's copy under several GPUs).
    """
    if precision not in _AUTOCAST:
        raise ValueError(f"precision={precision!r} is not one of {sorted(_AUTOCAST)}.")
    use_gpu, ids = resolve_device(device)
    devices = [ids] if isinstance(ids, int) else list(ids)
    if not use_gpu:
        precision = "32"
    if use_gpu and len(devices) > 1:
        _fit_distributed(
            model, loader, devices, max_epochs, precision, callbacks, progress
        )
    else:
        target = torch.device("cuda", devices[0]) if use_gpu else torch.device("cpu")
        _train(model, loader, target, max_epochs, precision, callbacks, progress)
    return model


def _call(hook: str, model: nn.Module, callbacks: Sequence[Any], *args: Any) -> None:
    """Run ``hook`` on every callback that has it, then on the model."""
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if method is not None:
            method(model, *args)
    method = getattr(model, hook, None)
    if method is not None:
        method(*args)


def _to_device(batch: Any, device: torch.device) -> Any:
    """``batch``'s tensors on ``device``, its list or tuple structure kept."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=device.type == "cuda")
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device) for b in batch)
    return batch


def _train(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    max_epochs: int,
    precision: str,
    callbacks: Sequence[Any] | None,
    progress: bool,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """The fit itself, on one device; one rank of several or the only one."""
    from ..progress import ProgressManager

    callbacks = list(callbacks or []) if rank == 0 else []
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model.to(device)
    model.train()
    if world_size > 1:
        # Every rank builds the same model; rank 0's is the one trained.
        for tensor in [*model.parameters(), *model.buffers()]:
            dist.broadcast(tensor.data, src=0)

    state = TrainingState(max_epochs, len(loader), rank, world_size)
    model._training = state
    optimizers, schedulers = model.configure_optimizers()
    state.optimizers, state.schedulers = list(optimizers), list(schedulers)
    dtype = _AUTOCAST[precision]
    if precision == "16-mixed":
        state.scaler = torch.amp.GradScaler("cuda")

    _call("on_fit_start", model, callbacks)
    _call("on_train_start", model, callbacks)
    manager = ProgressManager()
    for epoch in range(max_epochs):
        state.epoch = epoch
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        pbar, pos = (
            manager.get_pbar(None, f"Epoch {epoch}", total=len(loader))
            if progress and rank == 0
            else (None, None)
        )
        try:
            for batch_idx, batch in enumerate(loader):
                batch = _to_device(batch, device)
                _call("on_train_batch_start", model, callbacks, batch, batch_idx)
                autocast = (
                    torch.autocast(device.type, dtype=dtype)
                    if dtype is not None
                    else contextlib.nullcontext()
                )
                with autocast:
                    loss = model.training_step(batch, batch_idx)
                _call("on_train_batch_end", model, callbacks, batch, batch_idx)
                if pbar is not None:
                    if batch_idx % _PROGRESS_LOSS_EVERY == 0:
                        pbar.set_postfix(train_loss=f"{float(loss):.4g}", refresh=False)
                    pbar.update(1)
        finally:
            if pbar is not None and pos is not None:
                pbar.close()
                manager.release(pos)
        _call("on_train_epoch_end", model, callbacks)
    _call("on_train_end", model, callbacks)

    for optimizer in state.optimizers:
        for param_state in optimizer.state.values():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    param_state[key] = value.cpu()
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _call("on_fit_end", model, callbacks)


def _distributed_loader(
    loader: DataLoader[Any], rank: int, world_size: int
) -> DataLoader[Any]:
    """``loader`` with its batches split across ranks, as Lightning re-made it."""
    sampler: DistributedSampler[Any] = DistributedSampler(
        loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=isinstance(loader.sampler, RandomSampler),
        seed=0,
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        drop_last=loader.drop_last,
        pin_memory=loader.pin_memory,
    )


def _free_port() -> int:
    """A TCP port nothing on this host is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rank(
    rank: int,
    devices: list[int],
    port: int,
    model: nn.Module,
    loader: DataLoader[Any],
    max_epochs: int,
    precision: str,
) -> None:
    """One rank of a data-parallel fit: join the group, train, leave."""
    world_size = len(devices)
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", devices[rank]),
    )
    try:
        _train(
            model,
            _distributed_loader(loader, rank, world_size),
            torch.device("cuda", devices[rank]),
            max_epochs,
            precision,
            None,
            False,
            rank,
            world_size,
        )
    finally:
        dist.destroy_process_group()


def _fit_distributed(
    model: nn.Module,
    loader: DataLoader[Any],
    devices: list[int],
    max_epochs: int,
    precision: str,
    callbacks: Sequence[Any] | None,
    progress: bool,
) -> None:
    """Train on several GPUs: this process as rank 0, a spawned one per other."""
    import torch.multiprocessing as mp

    port = _free_port()
    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(
            target=_rank,
            args=(rank, devices, port, model, loader, max_epochs, precision),
        )
        for rank in range(1, len(devices))
    ]
    for worker in workers:
        worker.start()
    finished = False
    try:
        torch.cuda.set_device(devices[0])
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=len(devices),
            device_id=torch.device("cuda", devices[0]),
        )
        try:
            _train(
                model,
                _distributed_loader(loader, 0, len(devices)),
                torch.device("cuda", devices[0]),
                max_epochs,
                precision,
                callbacks,
                progress,
                0,
                len(devices),
            )
        finally:
            dist.destroy_process_group()
        finished = True
    finally:
        for worker in workers:
            if not finished:
                # A rank stuck in a collective rank 0 will never join.
                worker.terminate()
            worker.join()
    failed = [w.exitcode for w in workers if w.exitcode != 0]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(workers)} worker rank(s) exited with "
            f"code(s) {failed}; see their output above."
        )


def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """``tensor`` averaged over every rank, in place; unchanged on one."""
    if world_size > 1:
        dist.all_reduce(tensor)
        tensor /= world_size
    return tensor
