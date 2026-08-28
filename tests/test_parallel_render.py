"""Tests for specter.specimen._parallel_render -- the concurrency helpers
`specter build tomogram` (membrane mode) uses to fetch/parse multiple PDB
species (processes, build_pdb_cache_concurrently) and render their
PotentialBuilder templates (threads, build_templates_concurrently). The
thread-pool tests use dummy build functions (pure/stateless helper); the
process-pool tests use real PDB codes resolved from the tracked
fixtures in tests/test_data/ (never the network) since correctness
there depends on real pickling across a process boundary, which a dummy
function wouldn't exercise (covered end-to-end otherwise by
tests/test_tomogram_pipeline_membrane.py and tests/test_membrane_generator.py).

NOTE: build_pdb_cache_concurrently uses spawn-context multiprocessing --
safe here because pytest's own entry point is already __main__-guarded
(see that function's own docstring for why this matters at all)."""

from __future__ import annotations

import threading
import time

import pytest
from pathlib import Path

from specter.config import default_pdb_cache_dir
import torch

import specter.specimen._parallel_render as parallel_render_module
from specter.specimen import CRYOETSIM_PARTICLE_TABLE
from specter.specimen._parallel_render import (
    RECOMMENDED_MAX_RENDER_WORKERS,
    build_pdb_cache_concurrently,
    build_templates_concurrently,
    recommend_render_workers,
    resolve_render_devices,
    resolve_render_workers,
)


def test_resolve_render_devices_defaults_to_single_device():
    assert resolve_render_devices("cpu", None) == [torch.device("cpu")]
    assert resolve_render_devices("cpu", []) == [torch.device("cpu")]


def test_resolve_render_devices_honors_explicit_pool():
    devices = resolve_render_devices("cpu", ["cpu", "cpu"])
    assert devices == [torch.device("cpu"), torch.device("cpu")]


def test_build_templates_concurrently_serial_matches_keys():
    calls = []

    def build_one(key: int, device: torch.device) -> torch.Tensor:
        calls.append(key)
        return torch.tensor([key])

    result = build_templates_concurrently(
        keys=[0, 1, 2],
        build_one=build_one,
        devices=[torch.device("cpu")],
        max_workers=1,
    )
    assert calls == [0, 1, 2]  # max_workers=1 runs strictly in order
    assert {k: v.item() for k, v in result.items()} == {0: 0, 1: 1, 2: 2}


def test_build_templates_concurrently_parallel_produces_same_result():
    def build_one(key: int, device: torch.device) -> torch.Tensor:
        time.sleep(0.01)
        return torch.tensor([key])

    result = build_templates_concurrently(
        keys=[0, 1, 2, 3],
        build_one=build_one,
        devices=[torch.device("cpu")],
        max_workers=4,
    )
    assert {k: v.item() for k, v in result.items()} == {0: 0, 1: 1, 2: 2, 3: 3}


def test_build_templates_concurrently_actually_overlaps():
    barrier = threading.Barrier(3, timeout=5)

    def build_one(key: int, device: torch.device) -> torch.Tensor:
        barrier.wait()  # only returns once all 3 threads are inside at once
        return torch.tensor([key])

    result = build_templates_concurrently(
        keys=[0, 1, 2],
        build_one=build_one,
        devices=[torch.device("cpu")],
        max_workers=3,
    )
    assert len(result) == 3


def test_build_templates_concurrently_round_robins_devices():
    seen_devices = []

    def build_one(key: int, device: torch.device) -> torch.Tensor:
        seen_devices.append((key, device))
        return torch.tensor([key])

    build_templates_concurrently(
        keys=[0, 1, 2, 3],
        build_one=build_one,
        devices=[torch.device("cpu"), torch.device("meta")],
        max_workers=1,
    )
    assert seen_devices == [
        (0, torch.device("cpu")),
        (1, torch.device("meta")),
        (2, torch.device("cpu")),
        (3, torch.device("meta")),
    ]


def test_build_pdb_cache_concurrently_below_threshold_skips_process_pool(monkeypatch):
    """
    A handful of small structures is parsed serially, not in a process pool.

    Three test structures are a few seconds of parsing between them, far less
    than the pool's own measured spawn cost, so paying it would be a net loss --
    see `_process_pool_is_worth_it`.
    """

    def _boom(*args, **kwargs):
        raise AssertionError(
            "ProcessPoolExecutor must not be used when the estimated parse "
            "work cannot repay its spawn cost -- see _process_pool_is_worth_it"
        )

    monkeypatch.setattr(parallel_render_module, "ProcessPoolExecutor", _boom)
    cache = build_pdb_cache_concurrently(
        pdb_sources=["1mbo", "1bxn", "1a6m"],
        pdb_cache_dir=str(Path(__file__).parent / "test_data"),
        max_workers=8,
    )
    assert set(cache) == {"1mbo", "1bxn", "1a6m"}


# Enough real structures for `_process_pool_is_worth_it` to choose the pool.
_TABLE_SOURCES = [d["code"] for d in CRYOETSIM_PARTICLE_TABLE[:27]]


def _table_sources_cached() -> bool:
    """Whether every above-threshold source is already downloaded.

    These two tests need 27 real structures (~32 MB) to make the process pool
    the chosen path, which is too much to commit to git, so
    they are opt-in on a populated cache. Guarding on it keeps a clean clone
    from silently fetching 32 MB from RCSB in the middle of a test run.
    """
    cache = Path(default_pdb_cache_dir())
    return all((cache / f"{code}-assembly1.cif").is_file() for code in _TABLE_SOURCES)


_requires_table_sources = pytest.mark.skipif(
    not _table_sources_cached(),
    reason="needs the 27 CRYOETSIM_PARTICLE_TABLE structures in the PDB cache",
)


@_requires_table_sources
def test_build_pdb_cache_concurrently_parallel_matches_serial():
    # Enough estimated parse work to choose the pool, so this genuinely
    # exercises the process-pool path (see the below-threshold test above for
    # the fallback-to-serial case).
    sources = _TABLE_SOURCES
    serial = build_pdb_cache_concurrently(
        pdb_sources=sources, pdb_cache_dir=default_pdb_cache_dir(), max_workers=1
    )
    parallel = build_pdb_cache_concurrently(
        pdb_sources=sources, pdb_cache_dir=default_pdb_cache_dir(), max_workers=4
    )
    assert set(parallel) == set(serial) == set(sources)
    for source in sources:
        # PDB objects pickled across the process boundary should carry the
        # same data as an in-process (serial) parse.
        assert parallel[source].max_diameter == pytest.approx(
            serial[source].max_diameter
        )
        assert (
            parallel[source].atomic_numbers.shape == serial[source].atomic_numbers.shape
        )
        assert torch.allclose(parallel[source].coordinates, serial[source].coordinates)


@_requires_table_sources
def test_build_pdb_cache_concurrently_deduplicates_sources():
    sources = _TABLE_SOURCES
    cache = build_pdb_cache_concurrently(
        pdb_sources=sources + sources[:3],  # duplicate a few sources
        pdb_cache_dir=default_pdb_cache_dir(),
        max_workers=4,
    )
    assert set(cache) == set(sources)


def test_recommend_render_workers_floors_at_one():
    assert recommend_render_workers(0) == 1
    assert recommend_render_workers(-5) == 1


def test_resolve_render_workers_passes_through_non_auto():
    assert resolve_render_workers(3, n_species=100) == 3
    assert resolve_render_workers(1, n_species=100) == 1


def test_resolve_render_workers_resolves_auto():
    assert (
        resolve_render_workers("auto", n_species=1000) == RECOMMENDED_MAX_RENDER_WORKERS
    )
    assert resolve_render_workers("auto", n_species=3) == 3


def test_resolve_render_devices_resolves_auto():
    devices = resolve_render_devices("cpu", "auto")
    if torch.cuda.is_available():
        assert devices == [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]
    else:
        assert devices == [torch.device("cpu")]


# readd_hydrogens has to survive the process boundary: the PDB cache is built
# in spawned workers, which get a fresh interpreter and only the arguments
# packed into the task tuple. $CLIBD_MON reaches them through the inherited
# environment, the flag through the tuple -- both are checked here, on the
# serial and pooled branches, since they build their PDBs by different code.
def _monomer_library() -> str | None:
    import os
    from pathlib import Path

    env = os.environ.get("CLIBD_MON")
    if env and Path(env).is_dir():
        return env
    bundled = Path.home() / "sffit" / "monomers"
    return str(bundled) if bundled.is_dir() else None


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
@pytest.mark.parametrize("max_workers", [1, 2], ids=["serial", "process-pool"])
def test_build_pdb_cache_forwards_readd_hydrogens(monkeypatch, max_workers):
    from pathlib import Path

    from specter.specimen._parallel_render import build_pdb_cache_concurrently

    monkeypatch.setenv("CLIBD_MON", _monomer_library())
    cif = str(Path(__file__).parent / "test_data" / "1mbo.cif")

    def n_hydrogens(readd):
        cache = build_pdb_cache_concurrently(
            pdb_sources=[cif],
            pdb_cache_dir=str(Path(cif).parent),
            max_workers=max_workers,
            compute_atom_species=True,
            readd_hydrogens=readd,
        )
        return int((cache[cif].atomic_numbers == 1).sum())

    # 1mbo carries no hydrogens, so True adds them and False must not.
    assert n_hydrogens(True) > 0
    assert n_hydrogens(False) == 0


def test_estimated_parse_seconds_scales_with_file_size(tmp_path):
    """
    The parse-cost estimate reads the structure file's size, not just its name.

    This is what replaced a fixed threshold on the NUMBER of sources: parse cost
    spans ~40x across the structures one tomogram config loads, so a count says
    nothing about whether a process pool can repay its spawn cost.
    """
    from specter.specimen._parallel_render import _estimated_parse_seconds

    small = tmp_path / "aaaa-assembly1.cif"
    large = tmp_path / "bbbb-assembly1.cif"
    small.write_bytes(b"x" * 200_000)
    large.write_bytes(b"x" * 20_000_000)

    t_small = _estimated_parse_seconds("aaaa", str(tmp_path))
    t_large = _estimated_parse_seconds("bbbb", str(tmp_path))

    assert t_large > 10 * t_small
    # An un-cached source has no size to read and must not be estimated at zero,
    # or a pool of them would always look free.
    assert _estimated_parse_seconds("zzzz", str(tmp_path)) > 1.0


def test_estimated_parse_seconds_accepts_a_direct_path(tmp_path):
    """A `pdb_source` given as a path is sized directly, not looked up in the cache."""
    from specter.specimen._parallel_render import _estimated_parse_seconds

    path = tmp_path / "structure.cif"
    path.write_bytes(b"x" * 10_000_000)

    assert _estimated_parse_seconds(str(path), "") > 6.0


def test_process_pool_is_worth_it_matches_measured_crossover():
    """
    The serial-vs-pool decision reproduces the measured crossover.

    Reference points from dev/perf-bench/bench_pool_crossover.py, real
    structures, 8 workers: a net LOSS at an estimated ~24 s of serial work
    (measured 0.96x), a clear win by ~42 s (measured 1.49x) and ~90 s (2.81x).
    Pinning both sides matters -- the previous count-based rule got the 90 s
    case wrong in one direction and would have got a pile of tiny structures
    wrong in the other.
    """
    from specter.specimen._parallel_render import _process_pool_is_worth_it

    # one 26 s structure plus three small ones: not worth a pool
    assert not _process_pool_is_worth_it([26.0, 0.7, 4.0, 1.4], max_workers=8)
    # add two more large ones and it is
    assert _process_pool_is_worth_it([26.0, 0.7, 4.0, 1.4, 18.0, 1.1], max_workers=8)
    # the default tomogram config's 24 sources
    assert _process_pool_is_worth_it([26.0, 18.0] + [2.5] * 22, max_workers=8)

    # A single worker can never overlap anything, and an empty list has no work.
    assert not _process_pool_is_worth_it([26.0] * 50, max_workers=1)
    assert not _process_pool_is_worth_it([], max_workers=8)

    # Many tiny structures stay serial: 20 x 0.5 s cannot repay the spawn cost.
    assert not _process_pool_is_worth_it([0.5] * 20, max_workers=8)


@_requires_table_sources
def test_build_pdb_cache_concurrently_submits_expensive_sources_first(monkeypatch):
    """
    The pool receives its most expensive sources first.

    Callers pass sources sorted alphabetically, which is uncorrelated with cost
    and on the default tomogram config puts the two priciest structures near the
    end. Starting the longest task last adds its whole duration to the makespan
    rather than overlapping it, worth a measured 35.6s -> 31.5s.
    """
    from specter.specimen._parallel_render import _estimated_parse_seconds

    submitted: list[str] = []

    class _RecordingPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, args):
            submitted.append(args[0])

            class _Fut:
                def result(self_inner):
                    return None

            return _Fut()

    monkeypatch.setattr(parallel_render_module, "ProcessPoolExecutor", _RecordingPool)
    monkeypatch.setattr(parallel_render_module, "as_completed", lambda futures: [])

    sources = sorted(_TABLE_SOURCES)
    build_pdb_cache_concurrently(
        pdb_sources=sources,
        pdb_cache_dir=default_pdb_cache_dir(),
        max_workers=8,
    )

    assert sorted(submitted) == sorted(sources)
    costs = [_estimated_parse_seconds(s, default_pdb_cache_dir()) for s in submitted]
    assert costs == sorted(costs, reverse=True)
    assert submitted != sources, "submission order should not still be alphabetical"
