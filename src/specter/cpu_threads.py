"""
Cap PyTorch's intra-op CPU thread pool around loops of small tensor ops.

PyTorch sizes its CPU thread pool from the whole machine, and every CPU
tensor op pays a fixed synchronisation cost proportional to that pool. On a
tensor of a few hundred elements that cost is the whole op: on a 128-core
host, `specter.coords.poisson_disk_neighbors_3d` took 49 s at the default
128 threads and 0.24 s at 4, for bit-identical coordinates, and the same
overhead sat on every other CPU loop of the micrograph specimen build
(template potential 17.7 s against 4.5 s at 16 threads, tile placement,
particle insertion). Ops on big tensors do want the full pool, so the cap
is scoped to the loops that are known to be small-op bound rather than set
process-wide.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

#: Threads a small-op loop is capped to. Above this the per-op sync cost
#: dominates a small op; below it a loop that mixes in the odd real op (a
#: 64 MB slice add per particle in `insert_particles_into_micrograph`) loses
#: a little parallelism. 8 keeps both within a few percent of their optimum.
SMALL_OP_THREADS = 8


@contextlib.contextmanager
def limited_cpu_threads(max_threads: int = SMALL_OP_THREADS) -> Iterator[None]:
    """
    Run the block with at most `max_threads` intra-op CPU threads.

    Only ever lowers the count; the previous value is restored on exit, so
    a caller that already runs under a tighter cap (the test suite under
    xdist, for one) is unaffected.

    Parameters
    ----------
    max_threads : int, optional
        Thread ceiling for the block. Default :data:`SMALL_OP_THREADS`.
    """
    previous = torch.get_num_threads()
    if previous <= max_threads:
        yield
        return
    torch.set_num_threads(max_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
