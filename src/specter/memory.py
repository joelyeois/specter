"""
Memory-aware batch sizing for the particle-stack forward model.

`batchsize` controls how many particles are pushed through
`ImageGenerator.forward` at once. It is purely a memory/throughput knob --
it changes nothing about the physics -- but picking it well needs three
numbers a user has no reason to know: how many working copies of the padded
volume the multislice pipeline holds at peak, how big the padded volume
actually is (which is *not* `n_pixels**3`), and how much of the device is
free right now. `recommend_batchsize` computes all three.

Fitting the device is a bound, not a target
-------------------------------------------
A larger batch only pays until one forward pass already saturates the device,
and at the default config a single particle does: measured on an L40, box 256
is flat in batch size to 16 and then 17% SLOWER at 32, for 21x the memory.
Smaller boxes do benefit (~1.9x at box 64). So the batch is capped by work
per pass -- `_SATURATION_PADDED_VOXELS` -- as well as by memory, and "auto"
routinely returns far less than would fit. The per-batch figures behind both
numbers are tabulated at `_SATURATION_PADDED_VOXELS`; quote them from there
rather than restating them, so there is one place to correct after a
re-measurement.

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

#: Fraction of *currently free* device memory an auto batch may target.
#:
#: The CUDA number carries two things. `mem_get_info` is an exact reading of the
#: whole device, so the margin is not for that -- it is because the model above
#: is fit against *allocated* bytes while what has to fit is what the caching
#: allocator *reserves*, and the allocator's reserved footprint was measured at
#: 1.3-1.6x its allocated one on this pipeline. A batch sized against the
#: allocated figure alone is the mistake that let `"auto"` pick one which then
#: OOM'd on its second forward pass with ~19 GB "reserved but unallocated".
#:
#: The CPU number is softer for a different reason: `available` counts
#: reclaimable page cache, and the host is also holding the accumulated output
#: stack and any DataLoader workers.
_CUDA_SAFETY_FRACTION = 0.6
_CPU_SAFETY_FRACTION = 0.5

#: Padded voxels one forward pass may cover before batching stops paying.
#:
#: Batching amortizes per-call overhead, but only until a single pass already
#: saturates the device; past that a larger batch costs memory linearly and
#: buys nothing. Measured on an L40 (dev/perf-bench/bench_batchsize_
#: throughput.py, best of 3), per-particle speed against batch size:
#:
#:   batch:              1     2     4     8    16    32    64
#:   box  64 (1.05M/ptl)  1.00  1.40  1.76  1.93  1.84  1.91  1.62
#:   box 128 (8.4M/ptl)   1.00  1.42  1.75  1.88  1.73  1.77  1.62
#:   box 256 (67.1M/ptl)  1.00  1.04  0.97  1.01  0.96  0.83    --
#:
#: The 256 box -- the shipped config -- is flat: one particle already saturates
#: the device, so batching returns nothing and batch 32 is 17% SLOWER for 21x
#: the memory (1.5 GB at batch 1, 30.8 GB at 32). Smaller boxes gain ~1.9x.
#:
#: This budget is what stops the big box from batching up: it allows 2 there,
#: which is the measured optimum, against 19 and 152 for the 128 and 64 boxes
#: (both then cut to 8 by MAX_AUTO_BATCHSIZE, also the measured optimum).
_SATURATION_PADDED_VOXELS = 160_000_000

#: Ceiling on an auto-chosen batch. The measured optimum is 8 at every box size
#: above, and by 64 throughput has fallen back ~16% -- so this is the top of the
#: curve, not a "diminishing returns, could go higher" bound.
MAX_AUTO_BATCHSIZE = 8


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
    Largest batch size worth running on `device`, given the free memory there.

    The smaller of two bounds, not a memory calculation alone:

    - *What fits.* :func:`estimate_peak_bytes` inverted against a
      safety-discounted reading of free memory (`_CUDA_SAFETY_FRACTION` /
      `_CPU_SAFETY_FRACTION`).
    - *What is worth batching.* `_SATURATION_PADDED_VOXELS` divided by the
      padded voxels one particle covers. Past the point where a single
      forward pass already saturates the device, a larger batch costs memory
      in proportion to its size and returns nothing.

    The result is then clamped into ``[1, min(n_particles,
    MAX_AUTO_BATCHSIZE)]``.

    Which bound binds depends on the box, and at the shipped particle config
    it is the saturation one: a 256-pixel box with ``pad_fft`` covers 67M
    padded voxels per particle and yields 2 on any device, well under what
    would fit. Small boxes reach `MAX_AUTO_BATCHSIZE` instead.

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
    fraction = (
        _CUDA_SAFETY_FRACTION
        if torch.device(device).type == "cuda"
        else _CPU_SAFETY_FRACTION
    )
    budget = free_bytes * fraction

    per_particle = _PADDED_COPIES_PER_PARTICLE * nz * pad_nxy**2 * _BYTES_PER_ELEMENT
    overhead = estimate_peak_bytes(0, nxy, nz, pad_nxy)
    batchsize = int(math.floor((budget - overhead) / per_particle))

    # Past saturation a bigger batch is not faster, so this is a ceiling on
    # useful work, applied alongside the memory bound rather than instead of it.
    saturation = max(1, _SATURATION_PADDED_VOXELS // (nz * pad_nxy**2))

    ceiling = (
        MAX_AUTO_BATCHSIZE
        if n_particles is None
        else min(MAX_AUTO_BATCHSIZE, n_particles)
    )
    return max(1, min(batchsize, saturation, ceiling))


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
