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
* several devices take the same steps as one. The loader's batch size is
  the global batch: rank 0 draws each epoch's particle order from the
  loader's own sampler, exactly as a one-device fit draws it, and sends it
  to every rank, which takes its contiguous share of each global batch
  (`_ShardedBatchSampler`). Each rank weights its loss by its share of the
  batch and the gradients are summed across ranks, so a step's gradient is
  the global batch's, as gradient accumulation (``micro_batchsize``) forms
  it. Lightning's ``DistributedSampler`` instead shuffled per rank and made
  the batch size per device, so N devices took different steps from one.

Unlike Lightning's launcher, which re-ran the whole calling script once per
extra GPU, only the fit is replicated: this process is rank 0 and returns
the trained model, and each other rank is a spawned process that receives
the model and the loader's dataset by pickling (CPU tensors through shared
memory) and exits when the fit ends (`run_on_ranks`). Callbacks run on
rank 0 only.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from ._run_helpers import torch_devices

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
_T = TypeVar("_T")

#: How long a rank waits on a collective, and on the others joining, before
#: the group fails. Rank 0 writes each epoch's volume and figures while the
#: other ranks wait at the next step's all-reduce, so this bounds that too.
_GROUP_TIMEOUT = datetime.timedelta(minutes=60)

#: How often rank 0 checks that the other ranks are alive, in seconds.
_LIVENESS_POLL = 1.0

#: Store key a spawned rank sets just before it joins the group.
_JOINING_KEY = "specter/joining/{rank}"


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
    global_batch_size : int
        Particles in the current batch over every rank; set under several
        ranks only, where a rank's own batch is its share of this one.
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
    global_batch_size: int = 0


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
        The training batches. With several devices its batch size is the
        global batch, split across the ranks, and its sampler is drawn on
        rank 0 alone, so the ranks take the steps one device would; its
        dataset, worker count and collate function are re-read into one
        loader per rank.
    max_epochs : int
        Number of epochs.
    device : int or sequence of int or {"cpu"} or sequence of "cpu"
        A GPU index, several (data-parallel, one process each), or ``"cpu"``;
        as for :meth:`~specter.ghostbuster.Ghostbuster.run`, a GPU request
        falls back to the CPU when CUDA is unavailable. Several ``"cpu"``
        entries run that many CPU ranks over ``gloo`` (see `torch_devices`).
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
    devices = torch_devices(device)
    if devices[0].type == "cpu":
        precision = "32"
    if len(devices) > 1:
        run_on_ranks(
            _fit_rank,
            devices,
            (model, loader, max_epochs, precision),
            rank0_kwargs=dict(callbacks=callbacks, progress=progress),
        )
    else:
        _train(model, loader, devices[0], max_epochs, precision, callbacks, progress)
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
            # A collective takes contiguous tensors only, and a parameter
            # read from a .cs file can be a strided view of its table. Its
            # values are kept; nothing holds the old storage yet, since the
            # optimisers are built below.
            if not tensor.data.is_contiguous():
                tensor.data = tensor.data.contiguous()
            dist.broadcast(tensor.data, src=0)

    state = TrainingState(max_epochs, len(loader), rank, world_size)
    model._training = state
    sharded = (
        loader.batch_sampler
        if isinstance(loader.batch_sampler, _ShardedBatchSampler)
        else None
    )
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
        pbar, pos = (
            manager.get_pbar(None, f"Epoch {epoch}", total=len(loader))
            if progress and rank == 0
            else (None, None)
        )
        try:
            for batch_idx, batch in enumerate(loader):
                if sharded is not None:
                    state.global_batch_size = sharded.global_size(batch_idx)
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
        if world_size > 1:
            # Every rank applied the same summed gradient, so the replicas
            # agree; a nondeterministic kernel in an epoch-end hook (the
            # NUFFT behind exact symmetrisation spreads with atomics) could
            # still part them by rounding, which this resets each epoch.
            for tensor in model.parameters():
                dist.broadcast(tensor.data, src=0)
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


class _EmptySafeCollate:
    """
    A collate function that also collates an empty batch.

    A rank's share of a global batch smaller than the number of ranks is
    empty, and ``default_collate`` cannot collate nothing. An empty batch is
    collated as one sample would be and cut to length zero, so a rank with
    nothing to do still gets tensors of the right shape and dtype.
    """

    def __init__(self, collate_fn: Callable[[list[Any]], Any], dataset: Any) -> None:
        self.collate_fn = collate_fn
        self.dataset = dataset

    def __call__(self, samples: list[Any]) -> Any:
        if samples:
            return self.collate_fn(samples)
        return _empty_like(self.collate_fn([self.dataset[0]]))


def _empty_like(batch: Any) -> Any:
    """``batch`` with every tensor cut to length zero, its structure kept."""
    if isinstance(batch, torch.Tensor):
        return batch[:0]
    if isinstance(batch, (list, tuple)):
        return type(batch)(_empty_like(b) for b in batch)
    return batch


class _ShardedBatchSampler(Sampler[list[int]]):
    """
    This rank's share of each global batch of a one-device loader.

    Rank 0 draws an epoch's order from ``sampler``, the loader's own, at the
    point a one-device loader would draw it, and sends it to every rank;
    the order is cut into global batches of ``batch_size``, and each rank
    takes its contiguous share of each, ``tensor_split`` across the ranks
    (sizes equal to within one, so a short last batch can leave a rank
    none). Every rank therefore steps once per global batch, and the global
    batches are the one-device loader's batches, particle for particle.

    Parameters
    ----------
    sampler : Sampler
        The one-device loader's sampler. Only rank 0 iterates it; a
        ``RandomSampler`` draws its seed from rank 0's global stream, as it
        would on one device.
    batch_size : int
        The global batch.
    drop_last : bool
        Whether a short last global batch is dropped.
    rank, world_size : int
        This rank and the number of ranks.
    device : torch.device
        This rank's device, which the order is broadcast through.
    """

    def __init__(
        self,
        sampler: Iterable[int],
        batch_size: int,
        drop_last: bool,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        self.sampler = sampler
        self.n = len(sampler)  # type: ignore[arg-type]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return -(-self.n // self.batch_size)

    def global_size(self, batch_idx: int) -> int:
        """Particles in global batch `batch_idx`, over every rank."""
        return min(self.batch_size, self.n - batch_idx * self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.empty(self.n, dtype=torch.int64, device=self.device)
        if self.rank == 0:
            order.copy_(torch.as_tensor(list(self.sampler), dtype=torch.int64))
        dist.broadcast(order, src=0)
        order = order.cpu()
        for i in range(len(self)):
            batch = order[i * self.batch_size : (i + 1) * self.batch_size]
            yield batch.tensor_split(self.world_size)[self.rank].tolist()


def _sharded_loader(
    loader: DataLoader[Any], rank: int, world_size: int, device: torch.device
) -> DataLoader[Any]:
    """
    ``loader`` with each of its batches split across the ranks.

    Raises
    ------
    ValueError
        If ``loader`` has no batch size: a loader given its own batch
        sampler has no global batch to split.
    """
    if loader.batch_size is None:
        raise ValueError(
            "a data-parallel fit needs a loader built with batch_size; a custom "
            "batch_sampler has no global batch to split across the ranks."
        )
    return DataLoader(
        loader.dataset,
        batch_sampler=_ShardedBatchSampler(
            loader.sampler,
            loader.batch_size,
            loader.drop_last,
            rank,
            world_size,
            device,
        ),
        num_workers=loader.num_workers,
        collate_fn=_EmptySafeCollate(
            loader.collate_fn or default_collate, loader.dataset
        ),
        pin_memory=loader.pin_memory,
    )


def _fit_rank(
    rank: int,
    devices: list[torch.device],
    model: nn.Module,
    loader: DataLoader[Any],
    max_epochs: int,
    precision: str,
    callbacks: Sequence[Any] | None = None,
    progress: bool = False,
) -> None:
    """One rank of a data-parallel fit, inside its process group."""
    if rank > 0 and devices[rank].type == "cpu":
        # Sending the model to this rank put its CPU tensors in shared
        # memory, shared with rank 0's, and moving to the CPU copies
        # nothing: without a copy, every CPU rank would step the same map.
        model = copy.deepcopy(model)
    _train(
        model,
        _sharded_loader(loader, rank, len(devices), devices[rank]),
        devices[rank],
        max_epochs,
        precision,
        callbacks,
        progress,
        rank,
        len(devices),
    )


def _join_group(rank: int, devices: list[torch.device], store: dist.Store) -> None:
    """Join the default process group as `rank`: NCCL on GPUs, gloo on the CPU."""
    device = devices[rank]
    if device.type == "cuda":
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl",
            store=store,
            rank=rank,
            world_size=len(devices),
            timeout=_GROUP_TIMEOUT,
            device_id=device,
        )
    else:
        dist.init_process_group(
            "gloo",
            store=store,
            rank=rank,
            world_size=len(devices),
            timeout=_GROUP_TIMEOUT,
        )


def _worker_rank(
    target: Callable[..., Any],
    rank: int,
    devices: list[torch.device],
    port: int,
    args: tuple[Any, ...],
) -> None:
    """A spawned rank: join rank 0's group, run `target`, leave."""
    store = dist.TCPStore(
        "127.0.0.1", port, len(devices), is_master=False, timeout=_GROUP_TIMEOUT
    )
    # Rank 0 waits for this, checking meanwhile that the rank is alive.
    store.set(_JOINING_KEY.format(rank=rank), "1")
    _join_group(rank, devices, store)
    try:
        target(rank, devices, *args)
    finally:
        dist.destroy_process_group()


class _RankFailed(RuntimeError):
    """A spawned rank died before or while the group ran; raised by rank 0."""


def _exit_description(code: int) -> str:
    """``exit code 3``, or ``killed by SIGKILL`` for a death by signal."""
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


def _failed_rank(workers: Sequence[Any]) -> tuple[int, int] | None:
    """The first spawned rank that has exited with an error, and its code."""
    for rank, worker in enumerate(workers, start=1):
        if worker.exitcode not in (None, 0):
            return rank, worker.exitcode
    return None


def _await_joining(
    store: dist.TCPStore, workers: Sequence[Any], devices: list[torch.device]
) -> None:
    """
    Wait until every spawned rank is about to join, failing at once if one dies.

    Without this, rank 0 would wait in the group's rendezvous for a rank that
    died first (an unpicklable argument, an import error, a device already
    full), and fail only when `_GROUP_TIMEOUT` ran out.
    """
    pending = set(range(1, len(devices)))
    while pending:
        for rank in list(pending):
            # Non-blocking; present at runtime, absent from torch's stubs.
            if store.check([_JOINING_KEY.format(rank=rank)]):  # type: ignore[attr-defined]
                pending.discard(rank)
        failed = _failed_rank(workers)
        if failed is not None and failed[0] in pending:
            rank, code = failed
            raise _RankFailed(
                f"rank {rank} ({devices[rank]}) died before joining the process "
                f"group ({_exit_description(code)}); see its output above."
            )
        if pending:
            time.sleep(_LIVENESS_POLL / 10)


def _abort_group() -> None:
    """
    Shut down the default group's communicators, so that a collective waiting
    on a rank that has died returns with an error instead of waiting out
    `_GROUP_TIMEOUT`.

    NCCL does not notice a dead peer: the collective spins on the device
    until the group's timeout. Gloo does notice, from the closed connection,
    and needs nothing.
    """
    if not dist.is_initialized():
        return
    group = dist.distributed_c10d._get_default_group()
    try:
        backend = group._get_backend(torch.device("cuda"))
    except RuntimeError:
        return  # no NCCL backend: gloo
    shutdown = getattr(backend, "abort", None) or getattr(backend, "_shutdown", None)
    if shutdown is not None:
        shutdown()


class _Watchdog(threading.Thread):
    """Rank 0's watch on the spawned ranks while the group runs."""

    def __init__(self, workers: Sequence[Any]) -> None:
        super().__init__(daemon=True, name="specter-rank-watchdog")
        self.workers = workers
        self.failed: tuple[int, int] | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(_LIVENESS_POLL):
            failed = _failed_rank(self.workers)
            if failed is not None:
                self.failed = failed
                _abort_group()
                return

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


def run_on_ranks(
    target: Callable[..., _T],
    devices: Sequence[torch.device],
    args: tuple[Any, ...],
    rank0_kwargs: dict[str, Any] | None = None,
) -> _T:
    """
    Run ``target(rank, devices, *args)`` once per device, one process each.

    This process is rank 0 and returns its result; every other rank is a
    spawned process that receives `args` by pickling (CPU tensors through
    shared memory) and exits when `target` returns. Each call makes its own
    process group, so two calls from two processes (the halves of a
    gold-standard run) never share one. The group's store is bound to a
    port the operating system picks before any rank is spawned, so two
    groups started at the same moment cannot race for one port.

    A rank that dies fails the call within seconds rather than at the
    group's timeout: one that dies before joining is noticed while rank 0
    waits for the others to join, and one that dies later by a watchdog
    thread, which shuts the group's communicators down so that rank 0's
    pending collective returns.

    Parameters
    ----------
    target : callable
        A module-level function, so the spawned ranks can unpickle it.
    devices : sequence of torch.device
        One per rank; GPUs join over NCCL, CPUs over gloo.
    args : tuple
        Further arguments for every rank.
    rank0_kwargs : dict, optional
        Keyword arguments for rank 0 only: what cannot or need not be sent
        to the other ranks, such as callbacks.

    Returns
    -------
    object
        Rank 0's result.

    Raises
    ------
    RuntimeError
        If any spawned rank dies or exits with a nonzero code, naming the
        first that did.
    """
    import torch.multiprocessing as mp

    devices = list(devices)
    # Not waiting for the workers here: they are spawned below, once the
    # port is known, and join in `_join_group`.
    store = dist.TCPStore(
        "127.0.0.1",
        0,
        len(devices),
        is_master=True,
        timeout=_GROUP_TIMEOUT,
        wait_for_workers=False,
    )
    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(target=_worker_rank, args=(target, rank, devices, store.port, args))
        for rank in range(1, len(devices))
    ]
    for worker in workers:
        worker.start()
    watchdog = _Watchdog(workers)
    finished = False
    try:
        _await_joining(store, workers, devices)
        watchdog.start()
        _join_group(0, devices, store)
        try:
            result = target(0, devices, *args, **(rank0_kwargs or {}))
        finally:
            dist.destroy_process_group()
        finished = True
    except Exception as error:
        dead = watchdog.failed or _failed_rank(workers)
        if dead is not None and not isinstance(error, _RankFailed):
            rank, code = dead
            raise _RankFailed(
                f"rank {rank} ({devices[rank]}) died ({_exit_description(code)}) "
                "while the others were running; see its output above."
            ) from error
        raise
    finally:
        if watchdog.is_alive():
            watchdog.stop()
        for worker in workers:
            if not finished:
                # A rank stuck in a collective rank 0 will never join.
                worker.terminate()
            worker.join()
    # A rank that died while rank 0's target ran to the end still fails the
    # call: a collective in flight when it died can complete on stale data.
    dead = watchdog.failed or _failed_rank(workers)
    if dead is not None:
        rank, code = dead
        raise _RankFailed(
            f"rank {rank} ({devices[rank]}) died ({_exit_description(code)}) "
            "while the others were running; see its output above."
        )
    return result


def all_reduce_sum(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """``tensor`` summed over every rank, in place; unchanged on one."""
    if world_size > 1:
        dist.all_reduce(tensor)
    return tensor
