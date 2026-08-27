"""
Shared helpers for doing multiple per-species units of work concurrently.

`specter build tomogram` (membrane mode especially) does two things once
per PDB species -- fetch/parse the structure (`PDB`, network + gemmi) and
render its `PotentialBuilder` potential template (a torch op whose actual
compute happens in C++/CUDA, releasing the GIL) -- historically both one
species at a time.

These two need DIFFERENT concurrency mechanisms, not the same one, despite
looking like the same shape of problem:

- `PotentialBuilder.forward` releases the GIL for its real compute, so
  `build_templates_concurrently`'s plain `ThreadPoolExecutor` gives real
  wall-clock overlap across species with no multiprocessing/pickling
  complexity.
- `PDB.__init__`'s structure parsing (Biopython/gemmi) does NOT release
  the GIL for most of its work -- confirmed the hard way, not assumed:
  threading it gave zero wall-clock speedup on a 161-species production
  run despite dispatching correctly (summed per-thread durations DID
  scale with worker count, proving the threads were genuinely trying to
  run concurrently -- they just couldn't, GIL-serialized). `PDB` objects
  themselves pickle cheaply (~10ms to serialize, ~70ms to deserialize for
  a ~1600-atom structure -- negligible next to ~1-2s of parse time), so
  `build_pdb_cache_concurrently` uses real OS processes
  (`ProcessPoolExecutor`, spawn context -- matches `potential.py`'s own
  `build_potential(n_processes=...)` precedent, and is required rather
  than `fork` for CUDA safety once any CUDA context exists in the parent)
  instead. Measured directly: PDB fetch/parse alone was ~45% of total wall
  time on a 161-species production-scale tomogram, more than rendering
  itself -- worth fixing properly, not just parallelizing template
  rendering and assuming this half comes along for free.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal, TypeVar

import torch

from ..devices import parse_device

if TYPE_CHECKING:
    from ..pdb import PDB

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

# build_pdb_cache_concurrently falls back to serial when the process pool is
# not predicted to pay for itself. Spawn re-imports specter's whole dependency
# chain, torch/lightning included, in every worker, and that startup is a fixed
# cost the parsing has to earn back.
#
# This used to be a threshold on the NUMBER of unique sources (25), which is the
# wrong criterion: the cost being amortized is total parse WORK, and that varies
# ~40x across the structures a single tomogram config loads (2REC, 1,818 atoms,
# 0.67s; 6N4V, 236,760 atoms, 26s). A count threshold is wrong in both
# directions -- it pooled 25 tiny structures that would have been faster serial,
# and ran 24 large ones serially at a measured 2.8x loss. `configs/tomogram.toml`
# sat one source above it.
#
# Parse cost tracks the structure file's size on disk closely enough to decide
# on (dev/perf-bench/calibrate_pdb_cost.py, 27 structures spanning 0.26-25.6 MB:
# r = 0.974), and that size is one os.stat away, known before any parsing.
_PDB_PARSE_FIXED_S = 0.45
_PDB_PARSE_S_PER_MB = 0.65

# Measured spawn cost of the pool itself, as the residual between each pool run
# and its own perfect-scheduling floor, max(longest task, total work / workers)
# -- 6-8s across N = 2..24 (dev/perf-bench/bench_pool_crossover.py), not the
# ~13-15s previously assumed. Taking the top of that range keeps the estimate
# conservative, i.e. biased toward staying serial.
_POOL_SPAWN_OVERHEAD_S = 8.0

# Used when a source isn't cached yet and its size is therefore unknown. Those
# are network downloads, which overlap across processes regardless of parse
# cost, so guessing high is the safe direction.
_UNKNOWN_SOURCE_MB = 5.0

# The pool has to win by this much before it is worth taking, rather than merely
# breaking even. The estimator is good enough to rank and to size (r = 0.974)
# but individual structures land within about a factor of three of it, so a
# predicted saving of a second or two is noise -- and the downside of guessing
# wrong is the whole spawn cost, while the upside is that same second.
_POOL_REQUIRED_MARGIN = 0.75

# Measured sweet spot for build_templates_concurrently's THREAD pool
# specifically (dev/sweep_render_configs.py, full production-scale sweep,
# see [[project_render_workers_sweep_results]]): workers=8 was the fastest
# of {1,4,8,16,32,64} at N=161 species, and device choice (single-GPU,
# multi-GPU, CPU-only) barely mattered at that worker count (all within
# 2.5% of each other) -- so this is a single hardware-agnostic number, not
# something that needs tuning per device count. workers>=16 measured a
# ~3x SLOWDOWN from thread/GPU contention, on GPU *and* CPU configs alike
# -- so 8 is both the fastest measured value and a safe ceiling, not a
# "diminishing returns, could still go higher" one.
RECOMMENDED_MAX_RENDER_WORKERS = 8


def recommend_render_workers(n_species: int) -> int:
    """
    Suggest a `render_workers` value for a job with `n_species` unique PDB
    species to render/fetch, per `RECOMMENDED_MAX_RENDER_WORKERS`'s own
    measured basis. Never recommends more workers than there are species
    (no possible benefit past that) or fewer than 1.

    Parameters
    ----------
    n_species : int
        Number of unique species this job will process -- e.g.
        ``len(transmembrane_specs)`` or ``len(protein_specs)``, whichever
        pool is being sized (a job can reasonably want a different value
        for each if their counts differ a lot).

    Returns
    -------
    int
    """
    return max(1, min(n_species, RECOMMENDED_MAX_RENDER_WORKERS))


def recommend_render_devices() -> list[str] | None:
    """
    Suggest a `render_devices` pool: every visible CUDA GPU, or `None`
    (falls back to the caller's own `device`, typically CPU) if none are
    visible. Device CHOICE was measured to barely matter at
    `RECOMMENDED_MAX_RENDER_WORKERS` (single-GPU/multi-GPU/CPU-only all
    within 2.5% of each other) -- this just hands back whatever hardware
    is actually there rather than trying to be clever about which subset
    to use.

    Returns
    -------
    list[str] or None
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return None
    return [f"cuda:{i}" for i in range(n_gpus)]


def parse_device_pool(device: str) -> tuple[str, list[str] | None]:
    """
    Parse a single `device` config/CLI value into a primary device and an
    optional multi-device pool for concurrent per-species rendering.

    Lets config/CLI callers set one field instead of two separate
    `device`/`render_devices` values. A scalar (`"cpu"`, `"cuda"`,
    `"cuda:0"`, or a bare GPU index like `"0"`) means "everything on this
    one device" -- unchanged from the pre-merge `device`-only behaviour.
    A comma-separated list of GPU indices (`"0,1,2"`) pools those GPUs for
    per-species rendering, with the first as the primary device for
    everything else (membrane/filament generation, rotation, accumulator
    sizing). Shares `specter.devices.parse_device` with every other
    consumer of a `device` setting, so the accepted spellings cannot drift
    from theirs.

    Parameters
    ----------
    device : str

    Returns
    -------
    primary : str
        Device for everything outside per-species rendering.
    pool : list[str] or None
        Pass straight through as `resolve_render_devices`'s own
        `render_devices` argument. None means no pooling: render on
        `primary` alone.
    """
    spec = parse_device(device)
    if spec.is_multi:
        return spec.primary, list(spec.devices)
    return spec.primary, None


def resolve_render_workers(
    render_workers: int | Literal["auto"],
    n_species: int,
) -> int:
    """
    Normalize a `render_workers` config value -- `"auto"` resolves via
    `recommend_render_workers`, anything else passes through unchanged.
    """
    if render_workers == "auto":
        return recommend_render_workers(n_species)
    return render_workers


def resolve_render_devices(
    device: str | torch.device,
    render_devices: Sequence[str | torch.device] | Literal["auto"] | None,
) -> list[torch.device]:
    """
    Normalize the device pool used for parallel per-species rendering.

    Parameters
    ----------
    device : str or torch.device
        The generator's own primary device -- used as the sole pool member
        when `render_devices` is not given (or `"auto"` finds no GPUs).
    render_devices : sequence of str or torch.device, "auto", or None, optional
        An explicit pool to round-robin concurrent species across (e.g.
        multiple GPUs). `"auto"` resolves via `recommend_render_devices`.
        Default None.

    Returns
    -------
    list[torch.device]
        Always at least one entry.
    """
    if render_devices == "auto":
        render_devices = recommend_render_devices()
    if not render_devices:
        return [torch.device(device)]
    return [torch.device(d) for d in render_devices]


def build_templates_concurrently(
    keys: Sequence[K],
    build_one: Callable[[K, torch.device], V],
    devices: Sequence[torch.device],
    max_workers: int,
    on_result: Callable[[K], None] | None = None,
) -> dict[K, V]:
    """
    Build one result per key, optionally concurrently, on background
    THREADS -- see module docstring for why this is the right mechanism
    for `PotentialBuilder` rendering specifically, and the wrong one for
    PDB fetch/parse (`build_pdb_cache_concurrently` below).

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
    on_result : callable, optional
        ``on_result(key)``, called once per key as soon as its result is
        available -- in COMPLETION order when concurrent (via
        `as_completed`, not submission order, so a progress bar driven by
        this actually reflects wall-clock progress instead of stalling on
        whichever future happens to be first in the dict), in `keys`
        order when serial. Meant for a caller-side progress tick (e.g.
        `specter.progress.TqdmProgress`); exceptions from it propagate
        immediately, same as `build_one`'s own.

    Returns
    -------
    dict
        `key -> value`. An exception from any `build_one` call propagates
        once every submitted call has finished (matching serial semantics:
        the caller sees the same exception it would have seen running one
        at a time).
    """
    if max_workers <= 1 or len(keys) <= 1:
        results = {}
        for i, key in enumerate(keys):
            results[key] = build_one(key, devices[i % len(devices)])
            if on_result is not None:
                on_result(key)
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(build_one, key, devices[i % len(devices)]): key
            for i, key in enumerate(keys)
        }
        results = {}
        for fut in as_completed(futures):
            key = futures[fut]
            results[key] = fut.result()
            if on_result is not None:
                on_result(key)
        return results


def _estimated_parse_seconds(pdb_source: str, pdb_cache_dir: str) -> float:
    """
    Predict how long `PDB(pdb_source)` will take, from the file's size on disk.

    Used for two decisions, neither of which needs better than order-of-
    magnitude accuracy: whether a process pool will pay for its spawn cost at
    all, and which sources to submit to it first.

    Parameters
    ----------
    pdb_source : str
        A 4-character accession code or a path to a structure file.
    pdb_cache_dir : str
        Where downloaded structures are cached, checked for an accession code.

    Returns
    -------
    float
        Estimated seconds. Falls back to `_UNKNOWN_SOURCE_MB`'s worth for a
        source that isn't on disk yet (an un-fetched accession code), since its
        real cost is a network download whose size can't be known first.
    """
    import os

    candidates = [pdb_source]
    if pdb_cache_dir:
        candidates += [
            os.path.join(pdb_cache_dir, f"{pdb_source}-assembly{n}.cif")
            for n in range(1, 4)
        ]
        candidates += [
            os.path.join(pdb_cache_dir, f"{pdb_source}{ext}")
            for ext in (".cif", ".pdb")
        ]
    megabytes = _UNKNOWN_SOURCE_MB
    for path in candidates:
        try:
            if os.path.isfile(path):
                megabytes = os.path.getsize(path) / 1e6
                break
        except OSError:
            continue
    return _PDB_PARSE_FIXED_S + _PDB_PARSE_S_PER_MB * megabytes


def _process_pool_is_worth_it(estimates: list[float], max_workers: int) -> bool:
    """
    Decide whether spawning a process pool beats parsing these sources serially.

    Compares the serial total against the pool's best case -- perfect
    scheduling, so the longest single task or an even split across workers,
    whichever binds -- plus the pool's own measured spawn cost. Both sides use
    the same estimator, so a systematic bias in it largely cancels.

    Parameters
    ----------
    estimates : list of float
        Per-source estimated parse seconds, from `_estimated_parse_seconds`.
    max_workers : int
        Worker count the pool would be given.

    Returns
    -------
    bool
    """
    if not estimates or max_workers <= 1:
        return False
    serial = sum(estimates)
    pooled = max(max(estimates), serial / max_workers) + _POOL_SPAWN_OVERHEAD_S
    return pooled < _POOL_REQUIRED_MARGIN * serial


def _fetch_one_pdb(args: tuple[str, str, bool, bool | str, str | None]) -> "PDB":
    """
    Worker target for `build_pdb_cache_concurrently` -- a plain top-level
    function, not a closure/lambda, because `ProcessPoolExecutor` pickles
    the callable to send it to each worker process, and closures/lambdas
    aren't picklable. Imports `PDB` locally rather than at module level so
    this file doesn't force a `specter.pdb` import (and everything it
    drags in) on every caller of the thread-based helpers above, most of
    which never touch PDB fetching at all.
    """
    from ..pdb import PDB, suppress_monomer_library_warning

    # The parent already reported a missing monomer library once, on every
    # worker's behalf (see build_pdb_cache_concurrently). Each spawned worker
    # starts with fresh module state, so without this the same
    # environment-level fact is reported once per worker.
    suppress_monomer_library_warning()

    (
        pdb_source,
        pdb_cache_dir,
        compute_atom_species,
        readd_hydrogens,
        monomer_library_path,
    ) = args
    return PDB(
        pdb_source,
        pdb_cache_dir=pdb_cache_dir,
        verbose=False,
        compute_atom_species=compute_atom_species,
        readd_hydrogens=readd_hydrogens,
        monomer_library_path=monomer_library_path,
    )


def build_pdb_cache_concurrently(
    pdb_sources: Sequence[str],
    pdb_cache_dir: str,
    max_workers: int,
    on_result: Callable[[str], None] | None = None,
    compute_atom_species: bool = False,
    readd_hydrogens: bool | str = "auto",
    monomer_library_path: str | None = None,
) -> dict[str, "PDB"]:
    """
    Fetch/parse multiple PDB sources concurrently across OS PROCESSES (not
    threads) -- see module docstring for why threading this specific step
    doesn't work.

    Parameters
    ----------
    pdb_sources : sequence of str
        `pdb_source` values (PDB IDs or local file paths) to load. Duplicates
        are only fetched once (first occurrence order preserved in the
        returned dict's construction, though dict key order isn't itself
        load-bearing for callers).
    pdb_cache_dir : str
        Passed through to every `PDB(...)` call as `pdb_cache_dir`.
    max_workers : int
        Number of PDB sources fetched/parsed concurrently. ``<= 1``, or
        fewer than `_MIN_SOURCES_FOR_PROCESS_POOL` unique sources, skips
        process-pool startup entirely and runs fully serially in-process --
        matches the pre-parallel behaviour exactly for a trivially small
        job, and avoids the measured net LOSS from paying spawn overhead
        without enough work to amortize it (see that constant's own
        comment for the numbers).
    on_result : callable, optional
        ``on_result(pdb_source)``, called once per unique source as soon
        as its `PDB` is available -- in COMPLETION order when the process
        pool runs (via `as_completed`, not submission order), in
        `unique_sources` order when serial. Meant for a caller-side
        progress tick; exceptions from it propagate immediately.

    Returns
    -------
    dict[str, PDB]
        `pdb_source -> PDB`, one entry per unique input source.

    Notes
    -----
    Uses the `"spawn"` multiprocessing start method (like
    `potential.py`'s own `PotentialBuilder.build_potential(n_processes=...)`)
    rather than `"fork"`: fork-after-CUDA-init is unsafe, and by the time
    this runs a caller may already have touched CUDA elsewhere in the same
    process (e.g. a previous tomogram's species rendering, in a multi-
    tomogram run). `spawn` re-imports whatever module is `__main__` in
    each worker process -- callers driving this from a standalone script
    (not through the `specter` CLI, which already guards its own
    entry point) MUST guard their own top-level code with
    ``if __name__ == "__main__":``, or that top-level code re-runs inside
    every worker process too (a standard Python multiprocessing pitfall,
    not specific to this function).
    """
    unique_sources = list(dict.fromkeys(pdb_sources))
    estimates = [
        _estimated_parse_seconds(source, pdb_cache_dir) for source in unique_sources
    ]
    if not _process_pool_is_worth_it(estimates, max_workers):
        from ..pdb import PDB

        results = {}
        for source in unique_sources:
            results[source] = PDB(
                source,
                pdb_cache_dir=pdb_cache_dir,
                verbose=False,
                compute_atom_species=compute_atom_species,
                readd_hydrogens=readd_hydrogens,
                monomer_library_path=monomer_library_path,
            )
            if on_result is not None:
                on_result(source)
        return results

    # Submit the most expensive structures FIRST (longest-processing-time
    # first). Callers hand these in sorted alphabetically, which on the default
    # tomogram config puts the two priciest -- 6N4V and 6QZP, 26s and 18s of a
    # 90s serial total -- at positions 21 and 22 of 24. Starting the longest
    # task last makes the makespan (total / workers) + longest instead of
    # max(total / workers, longest): measured 35.6s -> 31.5s against a 26s floor
    # set by 6N4V alone.
    submission_order = [
        source
        for _, source in sorted(
            zip(estimates, unique_sources), key=lambda pair: -pair[0]
        )
    ]
    if compute_atom_species:
        # Report it here, once, rather than letting each worker discover it:
        # the workers are silenced (see _fetch_one_pdb) precisely so this is
        # the only place it can come from.
        from ..pdb import _warn_missing_monomer_library, _resolve_monomer_library_path

        if _resolve_monomer_library_path(monomer_library_path) is None:
            _warn_missing_monomer_library()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _fetch_one_pdb,
                (
                    source,
                    pdb_cache_dir,
                    compute_atom_species,
                    readd_hydrogens,
                    monomer_library_path,
                ),
            ): source
            for source in submission_order
        }
        results = {}
        for fut in as_completed(futures):
            source = futures[fut]
            results[source] = fut.result()
            if on_result is not None:
                on_result(source)
        return results
