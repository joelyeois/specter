"""Tests for specter.specimen._parallel_render -- the concurrency helpers
`specter build tomogram` (membrane mode) uses to fetch/parse multiple PDB
species (processes, build_pdb_cache_concurrently) and render their
PotentialBuilder templates (threads, build_templates_concurrently). The
thread-pool tests use dummy build functions (pure/stateless helper); the
process-pool tests use real, already-cached PDB codes since correctness
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
import torch

import specter.specimen._parallel_render as parallel_render_module
from specter.specimen import CRYOETSIM_PARTICLE_TABLE
from specter.specimen._parallel_render import (
    RECOMMENDED_MAX_RENDER_WORKERS,
    build_pdb_cache_concurrently,
    build_templates_concurrently,
    recommend_render_devices,
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


def test_build_pdb_cache_concurrently_serial():
    cache = build_pdb_cache_concurrently(
        pdb_sources=["1mbo"], pdb_cache_dir="specter-data/pdb", max_workers=1
    )
    assert set(cache) == {"1mbo"}
    assert cache["1mbo"].coordinates.shape[0] > 0


def test_build_pdb_cache_concurrently_below_threshold_skips_process_pool(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "ProcessPoolExecutor must not be used below "
            "_MIN_SOURCES_FOR_PROCESS_POOL -- see that constant's own comment"
        )

    monkeypatch.setattr(parallel_render_module, "ProcessPoolExecutor", _boom)
    cache = build_pdb_cache_concurrently(
        pdb_sources=["1mbo", "1fa2", "1a6m"],
        pdb_cache_dir="specter-data/pdb",
        max_workers=8,
    )
    assert set(cache) == {"1mbo", "1fa2", "1a6m"}


def test_build_pdb_cache_concurrently_parallel_matches_serial():
    # Above _MIN_SOURCES_FOR_PROCESS_POOL, so this genuinely exercises the
    # process-pool path (see the below-threshold test above for the
    # fallback-to-serial case).
    n = parallel_render_module._MIN_SOURCES_FOR_PROCESS_POOL + 2
    sources = [d["code"] for d in CRYOETSIM_PARTICLE_TABLE[:n]]
    serial = build_pdb_cache_concurrently(
        pdb_sources=sources, pdb_cache_dir="specter-data/pdb", max_workers=1
    )
    parallel = build_pdb_cache_concurrently(
        pdb_sources=sources, pdb_cache_dir="specter-data/pdb", max_workers=4
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


def test_build_pdb_cache_concurrently_deduplicates_sources():
    n = parallel_render_module._MIN_SOURCES_FOR_PROCESS_POOL + 2
    sources = [d["code"] for d in CRYOETSIM_PARTICLE_TABLE[:n]]
    cache = build_pdb_cache_concurrently(
        pdb_sources=sources + sources[:3],  # duplicate a few sources
        pdb_cache_dir="specter-data/pdb",
        max_workers=4,
    )
    assert set(cache) == set(sources)


def test_recommend_render_workers_caps_at_measured_sweet_spot():
    assert recommend_render_workers(1000) == RECOMMENDED_MAX_RENDER_WORKERS


def test_recommend_render_workers_never_exceeds_species_count():
    assert recommend_render_workers(3) == 3
    assert recommend_render_workers(1) == 1


def test_recommend_render_workers_floors_at_one():
    assert recommend_render_workers(0) == 1
    assert recommend_render_workers(-5) == 1


def test_recommend_render_devices_matches_visible_gpus():
    devices = recommend_render_devices()
    if torch.cuda.is_available():
        assert devices == [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        assert devices is None


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
    cif = str(Path(__file__).parent.parent / "specter-data" / "pdb" / "1mbo.cif")

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
