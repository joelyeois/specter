"""
Memory-aware batch sizing for the particle-stack forward model.

`batchsize` controls how many particles are pushed through
`Specimen.scatter` at once. It is purely a memory/throughput knob --
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
Peak allocation is linear in the batch size. The per-particle slope has a
part set by the FFT-padded volume (the padded canvas itself and the ice
convolution's working copies) and a part set by the unpadded template
volume (the template rotation's sampling grid, three floats per voxel, and
its output); a batch-independent workspace is set by the template again::

    peak ~= A * B * (nz * pad_nxy**2 * 4)   # padded canvas + ice FFT copies
          + G * B * (nxy**3 * 4)            # rotation grid + rotated template
          + W * (nxy**3 * 4)                # template, ice source, k-grids
          + C                               # CUDA context, model buffers

Constants were fit to measured `torch.cuda.max_memory_allocated` on an
NVIDIA L40 (6bdf at 1.0 A, multislice + IceBank ice + `pad_fft=True`, the
default path), sweeping
`n_pixels` over 128/256/384/512 and `batchsize` over 1/2/3/4/8. The peak is
the solvation stage at every point:

===========  ====  ============  =============
n_pixels      B  measured GiB  predicted GiB
===========  ====  ============  =============
128             1          0.38           0.41
128             8          0.72           0.76
256             1          0.86           0.98
256             4          1.91           2.18
256             8          3.19           3.78
384             1          2.39           2.52
384             4          5.88           6.57
512             1          5.48           5.53
512             3         11.12          11.93
===========  ====  ============  =============

The fit deliberately overestimates everywhere (1-18% margin over the sweep),
since the cost of guessing low is a slower run while the cost of guessing
high is an OOM crash. Cheaper configurations -- `pad_fft=False`,
`ice_model="none"`, non-multislice scattering -- all measured *below* the
prediction too, so one conservative constant covers the whole matrix rather
than needing a per-model table. Re-fit the constants after any change to
what the forward pass holds at peak; a fit against an older, heavier
forward pass over-predicts by 1.4-2.4x.
"""

from __future__ import annotations

import math
from typing import Literal

import psutil
import torch

from .scattering._kernels import _MULTISLICE_SLICE_CHUNK

__all__ = [
    "MAX_AUTO_BATCHSIZE",
    "available_memory_bytes",
    "estimate_peak_bytes",
    "recommend_batchsize",
    "resolve_batchsize",
]

#: Working copies of the FFT-padded per-particle volume held at peak: the
#: canvas itself plus the ice's rFFT working copies, which are of the unpadded
#: box (one quarter of the canvas each).
_PADDED_COPIES_PER_PARTICLE = 0.6

#: The ice path's workspace, in copies of the part of the unpadded ice column
#: that extends BEYOND the box, ``(nz - nxy) * nxy**2``. A box-deep column is
#: a single cached config and sits on the volume terms alone (the thin-ice
#: sweep above); a taller one is tiled, convolved periodically with complex
#: transforms of deltas and kernel, and blended slab by slab, and until
#: 2026-10-14 none of that was modelled: a 2000 A column at 384 px measured
#: 10.1 GB against a 5.0 GB estimate. Re-measured across thickness and batch
#: (dev/thick_ice_memory_probe.py): one more particle costs 0.8-1.3 extra
#: column copies on top of the padded-canvas term (the per-item ice draw and
#: its blend), and the batch pays a further ~8 once (6.6-9.3 across 256-512 px
#: at 500-1000 A; ~4 where the convolution is chunked above 2^28 voxels, so
#: that case over-estimates by about a third, on the safe side).
_ICE_COLUMN_COPIES_PER_PARTICLE = 1.2
_ICE_COLUMN_COPIES = 8.0

#: Per-particle workspace in copies of the *unpadded* template volume
#: (`nxy**3`): the template rotation's (nz, ny, nx, 3) sampling grid and its
#: rotated output, built for the whole batch at once.
_TEMPLATE_COPIES_PER_PARTICLE = 4.0

#: Batch-independent workspace, in copies of the *unpadded* template volume:
#: the built potential, the crowd's alias of it, the ice source positions and
#: the aberration k-grids. Fit from the peak-vs-batchsize intercept across
#: box sizes.
_TEMPLATE_COPIES = 4.0

#: Everything that scales with neither: CUDA context, model buffers, k-grids.
_FIXED_OVERHEAD_BYTES = 350_000_000

#: float32 -- the dtype the potential/ice volumes are built in.
_BYTES_PER_ELEMENT = 4

#: Complex64 copies of one multislice chunk (`_MULTISLICE_SLICE_CHUNK` slices
#: of the propagation grid) live at once: the interpolated slices, their
#: transmission functions and the FFT scratch. Three matches the measured
#: 0.17 GB/particle at pad 512, oversample 2 (3 x 8 x 1024^2 x 8 B = 0.2 GB).
_FINE_GRID_COMPLEX_COPIES = 3

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
#: buys nothing. Measured on an L40 (best of 3), per-particle speed against
#: batch size:
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


def estimate_peak_bytes(
    batchsize: int,
    nxy: int,
    nz: int,
    pad_nxy: int,
    oversample: int = 1,
    ice: bool = True,
) -> int:
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
    oversample : int, optional
        `Propagation.oversample`: the multislice recursion runs on an xy
        grid this many times finer. Default 1.
    ice : bool, optional
        Whether the run solvates (an icemaker is set), which adds the ice
        path's own working set. Default True.

    Returns
    -------
    int
        Estimated peak bytes, deliberately biased high -- see the module
        docstring for the measured basis.
    """
    padded_volume = nz * pad_nxy**2 * _BYTES_PER_ELEMENT
    template_volume = nxy**3 * _BYTES_PER_ELEMENT
    # The multislice chunk's complex temporaries on the fine grid, over what
    # the fitted per-particle copies already hold at oversample=1: measured
    # +0.17 GB per particle at a 256 box (pad 512, oversample 2), batch 4
    # going 1.82 -> 2.50 GB with the volume terms unchanged.
    fine_grid = (
        _FINE_GRID_COMPLEX_COPIES
        * _MULTISLICE_SLICE_CHUNK
        * (oversample**2 - 1)
        * pad_nxy**2
        * 2
        * _BYTES_PER_ELEMENT
    )
    ice_column = max(nz - nxy, 0) * nxy**2 * _BYTES_PER_ELEMENT if ice else 0
    return int(
        _PADDED_COPIES_PER_PARTICLE * batchsize * padded_volume
        + _TEMPLATE_COPIES_PER_PARTICLE * batchsize * template_volume
        + _TEMPLATE_COPIES * template_volume
        + batchsize * fine_grid
        + (_ICE_COLUMN_COPIES + _ICE_COLUMN_COPIES_PER_PARTICLE * batchsize)
        * ice_column
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
    oversample: int = 1,
    ice: bool = True,
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

    overhead = estimate_peak_bytes(0, nxy, nz, pad_nxy, oversample, ice)
    per_particle = estimate_peak_bytes(1, nxy, nz, pad_nxy, oversample, ice) - overhead
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
