"""Tests for specter.specimen._parallel_render -- the thread-pool helper
`specter build tomogram` (membrane mode) uses to render multiple PDB
species' PotentialBuilder templates concurrently. Pure/stateless helper, so
these use dummy build functions rather than real PDB/PotentialBuilder work
(covered end-to-end, network-dependent, by
tests/test_tomogram_pipeline_membrane.py and tests/test_membrane_generator.py)."""

from __future__ import annotations

import threading
import time

import torch

from specter.specimen._parallel_render import (
    build_templates_concurrently,
    resolve_render_devices,
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
