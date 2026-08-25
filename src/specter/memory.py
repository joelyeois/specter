"""
Memory-aware batch sizing for the particle-stack forward model.

`batchsize` controls how many particles are pushed through
`ImageGenerator.forward` at once. It is purely a memory/throughput knob --
it changes nothing about the physics -- but picking it well needs three
numbers a user has no reason to know: how many working copies of the padded
volume the multislice pipeline holds at peak, how big the padded volume
actually is (which is *not* `n_pixels**3`), and how much of the device is
free right now. `recommend_batchsize` computes all three.

Peak-memory model
-----------------
Peak allocation is linear in the batch size, with the per-particle slope set
by the FFT-padded volume and a batch-independent workspace set by the
unpadded template volume::

    peak ~= A * B * (nz * pad_nxy**2 * 4)   # B padded volumes in flight
          + W * (nxy**3 * 4)                # template, rotator, ice crops
          + C                               # CUDA context, model buffers

Constants were fit to measured `torch.cuda.max_memory_allocated` for real
`run_particle_stack` runs on an NVIDIA L40 (multislice + IceBank ice +
`pad_fft=True`, the default path), sweeping `n_pixels` over 128/192/256/384
and `batchsize` over 1/2/4/8:

===========  ====  ============  =============
n_pixels      B  measured GiB  predicted GiB
===========  ====  ============  =============
128             1          0.49           0.55
128             8          1.20           1.49
256             1          1.71           2.10
256             4          4.89           5.32
384             1          5.78           6.32
384             2          9.23           9.95
===========  ====  ============  =============

The fit deliberately overestimates everywhere (8-22% margin over the sweep),
since the cost of guessing low is a slower run while the cost of guessing
high is an OOM crash. Cheaper configurations -- `pad_fft=False`,
`ice_model="none"`, non-multislice scattering -- all measured *below* the
prediction too, so one conservative constant covers the whole matrix rather
than needing a per-model table.
"""

from __future__ import annotations

import math
import os
from typing import Literal

import psutil
import torch

__all__ = [
    "MAX_AUTO_BATCHSIZE",
    "available_memory_bytes",
    "estimate_peak_bytes",
    "recommend_batchsize",
    "resolve_batchsize",
]

#: Working copies of the FFT-padded per-particle volume held at peak. Fit
#: slope of peak-vs-batchsize; 4.09-4.23 measured, rounded up.
_PADDED_COPIES_PER_PARTICLE = 4.3

#: Batch-independent workspace, in copies of the *unpadded* template volume
#: (`nxy**3`): the built potential, the rotator's working buffer and the ice
#: crops drawn from the bank. Fit from the peak-vs-batchsize intercept across
#: box sizes.
_TEMPLATE_COPIES = 11.2

#: Everything that scales with neither: CUDA context, model buffers, k-grids.
_FIXED_OVERHEAD_BYTES = 350_000_000

#: float32 -- the dtype the potential/ice volumes are built in.
_BYTES_PER_ELEMENT = 4

#: Fraction of *currently free* device memory an auto batch may target. GPUs
#: get the larger share because `mem_get_info` is an exact, instantaneous
#: reading of the whole device (other processes' usage included); the CPU
#: number is softer -- `available` counts reclaimable page cache, and the host
#: is also holding the accumulated output stack and any DataLoader workers.
_CUDA_SAFETY_FRACTION = 0.8
_CPU_SAFETY_FRACTION = 0.5

#: Extra headroom applied when the CUDA caching allocator is running WITHOUT
#: expandable segments. The model above is fit against *allocated* bytes, but
#: what has to fit on the card is what the allocator *reserves*, and with the
#: default segment allocator those differ: measured on the particle pipeline at
#: box 256 / pad 512, reserved ran 1.3-1.6x allocated, reaching 1.29x of this
#: model's own prediction at batch 30 (28.7 GB allocated, 45.9 GB reserved) --
#: which is how `batchsize="auto"` came to pick a batch that died on its second
#: forward pass with ~19 GB "reserved but unallocated" going spare.
#:
#: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` mostly closes that gap
#: (0.86x of prediction at the same batch), and the `specter` CLI sets it -- so
#: this allowance exists for library callers who have not, and is skipped when
#: the variable is already set.
_FRAGMENTATION_HEADROOM = 1.3

#: Ceiling on an auto-chosen batch. Throughput has flattened well before this
#: (the fixed per-batch overheads are amortized within a few particles), and
#: the fit above was only measured out to B=8 -- so on a large GPU with a
#: small box, stop rather than extrapolate.
MAX_AUTO_BATCHSIZE = 64


def _expandable_segments_enabled() -> bool:
    """
    Whether the CUDA allocator is configured to use expandable segments.

    Read from the environment rather than from torch, which exposes no query
    for it. Only ever used to decide how much fragmentation headroom
    :func:`recommend_batchsize` should leave, so a false negative costs a
    smaller batch, never a crash.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:true" in conf.lower().replace(" ", "")


def estimate_peak_bytes(batchsize: int, nxy: int, nz: int, pad_nxy: int) -> int:
    """
    Estimate peak memory for one forward pass of the particle pipeline.

    Parameters
    ----------
    batchsize : int
        Particles per forward pass.
    nxy : int
        Unpadded box size in pixels (``config.n_pixels``).
    nz : int
        Number of Z slices, from :func:`specter.arrays.compute_nz`.
    pad_nxy : int
        FFT-padded XY size (``nxy + 2 * (nxy // 2)`` when ``pad_fft``, else
        ``nxy``).

    Returns
    -------
    int
        Estimated peak bytes, deliberately biased high -- see the module
        docstring for the measured basis.
    """
    padded_volume = nz * pad_nxy**2 * _BYTES_PER_ELEMENT
    template_volume = nxy**3 * _BYTES_PER_ELEMENT
    return int(
        _PADDED_COPIES_PER_PARTICLE * batchsize * padded_volume
        + _TEMPLATE_COPIES * template_volume
        + _FIXED_OVERHEAD_BYTES
    )


def available_memory_bytes(device: str | torch.device) -> int:
    """
    Free memory on `device`, right now.

    Parameters
    ----------
    device : str or torch.device
        Any device string the pipeline accepts (``"cpu"``, ``"cuda"``,
        ``"cuda:1"``).

    Returns
    -------
    int
        Free bytes: `torch.cuda.mem_get_info` for CUDA (device-wide, so
        memory held by *other* processes on a shared GPU is already
        excluded), `psutil.virtual_memory().available` for CPU.

    Notes
    -----
    The CPU reading is the *host's*, not a cgroup/container limit -- under
    Slurm's ``--mem`` or inside a container this will over-report, and a
    memory-constrained job there should pin `batchsize` to an integer
    rather than rely on ``"auto"``.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free)
    return int(psutil.virtual_memory().available)


def recommend_batchsize(
    nxy: int,
    nz: int,
    pad_nxy: int,
    device: str | torch.device,
    n_particles: int | None = None,
) -> int:
    """
    Largest batch size expected to fit in the free memory on `device`.

    Inverts :func:`estimate_peak_bytes` against a safety-discounted reading
    of free memory, then clamps into ``[1, min(n_particles,
    MAX_AUTO_BATCHSIZE)]``.

    Parameters
    ----------
    nxy : int
        Unpadded box size in pixels.
    nz : int
        Number of Z slices.
    pad_nxy : int
        FFT-padded XY size.
    device : str or torch.device
        Device the forward model will run on. For multi-GPU runs, pass the
        *smallest*-free device: every rank builds the same-sized batch.
    n_particles : int, optional
        Total particles in the run. A batch never needs to exceed this.

    Returns
    -------
    int
        At least 1 -- if even a single particle is predicted not to fit,
        this still returns 1 (with the run then free to OOM honestly)
        rather than refusing to start on what is, after all, an estimate.
    """
    free_bytes = available_memory_bytes(device)
    is_cuda = torch.device(device).type == "cuda"
    fraction = _CUDA_SAFETY_FRACTION if is_cuda else _CPU_SAFETY_FRACTION
    budget = free_bytes * fraction
    if is_cuda and not _expandable_segments_enabled():
        budget /= _FRAGMENTATION_HEADROOM

    per_particle = _PADDED_COPIES_PER_PARTICLE * nz * pad_nxy**2 * _BYTES_PER_ELEMENT
    overhead = estimate_peak_bytes(0, nxy, nz, pad_nxy)
    batchsize = int(math.floor((budget - overhead) / per_particle))

    ceiling = (
        MAX_AUTO_BATCHSIZE
        if n_particles is None
        else min(MAX_AUTO_BATCHSIZE, n_particles)
    )
    return max(1, min(batchsize, ceiling))


def resolve_batchsize(
    batchsize: int | Literal["auto"],
    nxy: int,
    nz: int,
    pad_nxy: int,
    device: str | torch.device,
    n_particles: int | None = None,
) -> int:
    """
    Normalize a config `batchsize`: ``"auto"`` sizes to the device, an int
    passes through unchanged.

    Parameters
    ----------
    batchsize : int or "auto"
        The configured value.
    nxy, nz, pad_nxy : int
        Box geometry -- see :func:`recommend_batchsize`.
    device : str or torch.device
        Device the forward model will run on.
    n_particles : int, optional
        Total particles in the run.

    Returns
    -------
    int
    """
    if batchsize == "auto":
        return recommend_batchsize(nxy, nz, pad_nxy, device, n_particles)
    return int(batchsize)
