"""
Shared helper for doing multiple per-species units of work concurrently.

`specter build tomogram` (membrane mode especially) does two things once
per PDB species -- fetch/parse the structure (`PDB`, network + gemmi,
CPU/IO-bound) and render its `PotentialBuilder` potential template (a torch
op whose actual compute happens in C++/CUDA, releasing the GIL) --
historically both one species at a time. Neither holds the GIL for long, so
a plain `ThreadPoolExecutor` gives real wall-clock overlap across species
without any multiprocessing/pickling complexity. Same pattern
`specimen.cryoet.CryoETSpecimenGenerator` already uses for its own 2-way
membrane/protein overlap. Measured directly on a 161-species production-
scale tomogram: PDB fetch/parse alone was ~45% of total wall time when it
ran serially, outside this helper's reach -- worth parallelizing in its own
right, not just template rendering.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import torch

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def resolve_render_devices(
    device: str | torch.device,
    render_devices: Sequence[str | torch.device] | None,
) -> list[torch.device]:
    """
    Normalize the device pool used for parallel per-species rendering.

    Parameters
    ----------
    device : str or torch.device
        The generator's own primary device -- used as the sole pool member
        when `render_devices` is not given.
    render_devices : sequence of str or torch.device, optional
        An explicit pool to round-robin concurrent species across (e.g.
        multiple GPUs). Default None.

    Returns
    -------
    list[torch.device]
        Always at least one entry.
    """
    if not render_devices:
        return [torch.device(device)]
    return [torch.device(d) for d in render_devices]


def build_templates_concurrently(
    keys: Sequence[K],
    build_one: Callable[[K, torch.device], V],
    devices: Sequence[torch.device],
    max_workers: int,
) -> dict[K, V]:
    """
    Build one result per key, optionally concurrently.

    Despite the name, `build_one` isn't required to return a potential
    template specifically -- it's also reused for parallel PDB fetch/parse
    (returning a `PDB` object), which has no device dimension at all
    (`devices` is just a fixed single-CPU pool in that case, ignored by
    `build_one`).

    Parameters
    ----------
    keys : sequence
        One entry per unit of work -- also used as the returned dict's
        keys (e.g. `pdb_source` strings or species indices).
    build_one : callable
        ``build_one(key, device) -> value``. Must be safe to call from
        multiple threads at once; constructing a fresh `PotentialBuilder`/
        `PDB` per call (the existing pattern at every call site) already
        satisfies this.
    devices : sequence of torch.device
        Round-robined across `keys` in declaration order, so multiple
        devices (e.g. multi-GPU) can process different keys at the same
        time.
    max_workers : int
        Number of keys processed concurrently. ``<= 1`` (or a single key)
        skips the thread pool entirely and runs fully serially, matching
        the pre-parallel behaviour exactly.

    Returns
    -------
    dict
        `key -> value`. An exception from any `build_one` call propagates
        once every submitted call has finished (matching serial semantics:
        the caller sees the same exception it would have seen running one
        at a time).
    """
    if max_workers <= 1 or len(keys) <= 1:
        return {
            key: build_one(key, devices[i % len(devices)]) for i, key in enumerate(keys)
        }
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(build_one, key, devices[i % len(devices)]): key
            for i, key in enumerate(keys)
        }
        return {futures[fut]: fut.result() for fut in futures}
