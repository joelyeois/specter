"""
Batch sizes predicted from a measured peak of device memory.

A pass that runs out of device memory inside PyTorch raises
``torch.cuda.OutOfMemoryError`` and can be redone with a smaller batch. One
that runs out inside cuFINUFFT cannot: the library allocates with its own
``cudaMalloc``, outside PyTorch's caching allocator, and a failed allocation
there leaves the CUDA context unusable (the next call fails with
``cudaErrorInvalidValue``), so no retry recovers. The batch size is therefore
chosen before the first pass, from the memory the work is measured to need
at a few small batch sizes, rather than discovered by failing.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from specter import logger

from .. import nufft

#: Share of the free device memory a predicted batch may use. The remainder
#: covers what the probe cannot see: allocator fragmentation over a long
#: solve, cuFINUFFT's short-lived spreading buffers, and another process on
#: the same device.
SAFETY = 0.85

#: How far PyTorch's cache can grow over steady training beyond the peak of
#: one step taken from an emptied cache: blocks freed by one step are cached
#: in sizes the next does not reuse. Measured at up to 1.41x the predicted
#: peak over Adam steps at a 320 box (rytov, real rotation, anisomag, D2),
#: and back to 1.0x when the cache is emptied before each step.
CACHE_GROWTH = 1.5

#: Seconds between samples of the device's memory use during a probe.
_SAMPLE_INTERVAL = 0.001


@dataclass
class BatchPrediction:
    """
    A batch size and the measurements it was predicted from.

    Attributes
    ----------
    batchsize : int
        Largest batch whose predicted peak fits ``safety * free``.
    fixed : float
        Bytes the work needs whatever the batch: the intercept of the fit.
    per_particle : float
        Bytes each further particle adds: the slope of the fit.
    free : float
        Free device memory before the probe, in bytes.
    peaks : dict of int to float
        Measured peak above the starting use, in bytes, per probed batch.
    """

    batchsize: int
    fixed: float
    per_particle: float
    free: float
    peaks: dict[int, float]

    def predicted_peak(self, batchsize: int) -> float:
        """Bytes above the starting use a batch of `batchsize` is predicted to need."""
        return self.fixed + self.per_particle * batchsize

    def cache_may_overfill(self, batchsize: int, safety: float = SAFETY) -> bool:
        """
        Whether PyTorch's cache, left to grow over many batches of
        `batchsize`, may pass ``safety`` times the free memory
        (`CACHE_GROWTH`); emptying it before each batch prevents that.
        """
        return CACHE_GROWTH * self.predicted_peak(batchsize) > safety * self.free

    def steady_batchsize(self, safety: float = SAFETY) -> int:
        """
        The largest batch the cache cannot overfill (`cache_may_overfill`),
        at most `batchsize`; 0 when even one particle can.

        A training loop that takes this batch never needs to empty the cache.
        """
        if self.per_particle <= 0:
            return self.batchsize
        room = safety * self.free / CACHE_GROWTH - self.fixed
        return max(0, min(self.batchsize, int(room // self.per_particle)))


def _process_memory(device: torch.device) -> Callable[[], int] | None:
    """
    A reader of this process's memory on `device`, from NVML; None without one.

    NVML counts every allocation the process holds on the device, cuFINUFFT's
    included, and nothing another process holds. It is unavailable without
    the ``nvidia-ml-py`` package or the driver's library, and in a container
    whose process IDs NVML does not see.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
    except pynvml.NVMLError:  # no driver library, or a device NVML cannot see
        return None
    pid = os.getpid()

    def read() -> int:
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == pid and process.usedGpuMemory is not None:
                return int(process.usedGpuMemory)
        raise LookupError(pid)

    try:
        read()
    except (LookupError, pynvml.NVMLError):
        return None
    return read


class _PeakMemory:
    """
    The peak device memory a block uses, above what was in use when it began.

    A background thread samples this process's use: NVML's per-process count
    when NVML can see this process, so another job on the same GPU does not
    enter it, and the device's use (``torch.cuda.mem_get_info``) otherwise.
    Either includes cuFINUFFT's allocations, which PyTorch's statistics do
    not. PyTorch's own peak, read exactly from its allocator, is a floor: its
    caching allocator holds a peak until the cache is emptied, so a sampler
    rarely misses it, but the floor keeps a short spike from going uncounted.
    Adding the two peaks instead double-counts: the sampled use less PyTorch's
    reserved memory spikes by gigabytes as PyTorch empties its cache, which
    cuFINUFFT's plan construction does, and NVML has not yet caught up.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak = 0.0
        self._stop = threading.Event()
        self._top = 0
        self._read = _process_memory(device) or self._device_used

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._top = max(self._top, self._read())
            self._stop.wait(_SAMPLE_INTERVAL)

    def _device_used(self) -> int:
        free, total = torch.cuda.mem_get_info(self.device)
        return int(total - free)

    def __enter__(self) -> _PeakMemory:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start = self._top = self._read()
        self._start_reserved = torch.cuda.memory_reserved(self.device)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        torch.cuda.synchronize(self.device)
        self._stop.set()
        self._thread.join()
        self._top = max(self._top, self._read())
        sampled = self._top - self._start
        torch_peak = torch.cuda.max_memory_reserved(self.device) - self._start_reserved
        self.peak = float(max(sampled, torch_peak))


def _free_bytes(device: torch.device) -> float:
    """Free memory on `device`, as the driver reports it."""
    return float(torch.cuda.mem_get_info(device)[0])


def _release(device: torch.device) -> None:
    """Return cached memory to the driver: NUFFT plans and PyTorch's free blocks."""
    nufft.clear_plan_cache()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def fit_batchsize(
    peaks: dict[int, float], free: float, maximum: int, safety: float = SAFETY
) -> BatchPrediction:
    """
    The largest batch a line through measured peaks fits in ``safety * free``.

    Parameters
    ----------
    peaks : dict of int to float
        Peak bytes above the starting use, per probed batch size. The line
        runs through the two largest; with one probe, the batch is 1.
    free : float
        Free device bytes before the probes.
    maximum : int
        Largest batch allowed.
    safety : float, optional
        Share of `free` the prediction may fill. Default `SAFETY`.

    Returns
    -------
    BatchPrediction
        Clamped to ``[1, maximum]``: a batch of one runs even when predicted
        not to fit, and fails as it would have.
    """
    if len(peaks) < 2:
        (peak,) = peaks.values()
        return BatchPrediction(1, peak, 0.0, free, peaks)
    lo, hi = sorted(peaks)[-2:]
    # A further particle adds at least something; a measured slope at or
    # below zero is noise, and a line through it would predict any batch.
    per_particle = max((peaks[hi] - peaks[lo]) / (hi - lo), 0.01 * peaks[hi] / hi, 1.0)
    # Nor does the work need less than nothing at no particles, which a line
    # through two probes of almost no memory can say.
    fixed = max(peaks[hi] - per_particle * hi, 0.0)
    batch = int((free * safety - fixed) // per_particle)
    return BatchPrediction(
        max(1, min(batch, maximum)), fixed, per_particle, free, peaks
    )


def predict_batchsize(
    run: Callable[[int], Any],
    device: torch.device,
    maximum: int,
    safety: float = SAFETY,
    label: str = "",
) -> BatchPrediction:
    """
    The largest batch whose work fits in device memory, from measured probes.

    ``run(b)`` is called once at ``b = 1`` to warm up, unmeasured, then at
    ``b = 1`` and ``b = 2``, each from a released device (no NUFFT plans,
    PyTorch's cache emptied), and its peak use above the starting use is
    measured. A line through the two predicts a first batch; a further probe
    at a quarter of it, when that is at least 4, replaces the line by one
    through 2 and that batch. The result is the largest batch whose
    prediction stays within ``safety`` times the free memory, clamped to
    ``[1, maximum]`` (`fit_batchsize`).

    Parameters
    ----------
    run : callable
        Does the work of one batch of the given size, including every
        evaluation whose memory has to fit at once: a gradient pass and the
        line search's passes, for the converged solver. Must leave nothing
        allocated that the next call does not reuse.
    device : torch.device
        A CUDA device. Elsewhere `maximum` is returned unprobed.
    maximum : int
        Largest batch allowed, usually the number of particles.
    safety : float, optional
        Share of the free memory the prediction may fill. Default `SAFETY`.
    label : str, optional
        Prefixed to the logged prediction.

    Returns
    -------
    BatchPrediction

    Notes
    -----
    Memory is not linear in the batch at one or two particles: buffers whose
    size does not follow the batch, and the caching allocator's block sizes,
    make the first particle's step unrepresentative. At a 320 box the step
    from one particle to two was 0.50 GiB for first Born against 0.61 GiB per
    particle from 16 to 32, and a line through 1 and 2 put 32 particles 16%
    low; through 2 and 16 it was within 3%. The third probe costs one batch
    of a quarter of the final size, and is safe even if the first line is
    badly wrong.
    """
    if maximum < 1:
        raise ValueError(f"maximum={maximum} must be at least 1")
    if device.type != "cuda":
        return BatchPrediction(maximum, 0.0, 0.0, 0.0, {})
    # An unmeasured first call: what it builds once and keeps (the rotation's
    # frequency grid, cuFFT plans) is then part of the starting use, as it is
    # through the rest of the run, not the first probe's peak alone.
    run(1)
    _release(device)
    free = _free_bytes(device)
    peaks: dict[int, float] = {}

    def probe(b: int) -> None:
        with _PeakMemory(device) as meter:
            run(b)
        peaks[b] = meter.peak
        _release(device)

    probe(1)
    if maximum > 1:
        probe(2)
    third = fit_batchsize(peaks, free, maximum, safety).batchsize // 4
    if third >= 4:
        probe(third)
    prediction = fit_batchsize(peaks, free, maximum, safety)
    logger.info(
        f"{label + ': ' if label else ''}batch size {prediction.batchsize}, "
        f"predicted peak {prediction.predicted_peak(prediction.batchsize) / 2**30:.1f} "
        f"of {free / 2**30:.1f} GiB free ({prediction.fixed / 2**30:.2f} GiB fixed, "
        f"{prediction.per_particle / 2**30:.3f} GiB per particle)"
    )
    return prediction
