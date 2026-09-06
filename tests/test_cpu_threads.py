"""Tests for `specter.cpu_threads`: the scoped cap only lowers the pool and
changes nothing about what runs under it."""

from __future__ import annotations

import torch

import specter
from specter.coords import poisson_disk_neighbors, poisson_disk_neighbors_3d
from specter.cpu_threads import limited_cpu_threads


def test_limited_cpu_threads_only_lowers_and_restores():
    before = torch.get_num_threads()
    with limited_cpu_threads(max(1, before - 1)):
        assert torch.get_num_threads() == max(1, before - 1)
    assert torch.get_num_threads() == before
    with limited_cpu_threads(before + 5):
        assert torch.get_num_threads() == before
    assert torch.get_num_threads() == before


def test_poisson_disk_samplers_are_identical_under_the_thread_cap():
    """The cap is a performance setting: the draws and the accept/reject
    arithmetic are the same at any thread count."""
    before = torch.get_num_threads()
    specter.seed(0)
    a3 = poisson_disk_neighbors_3d(30.0, box=(64.0, 128.0, 128.0))
    specter.seed(0)
    a2 = poisson_disk_neighbors(30.0, box=(128.0, 128.0))
    torch.set_num_threads(1)
    try:
        specter.seed(0)
        b3 = poisson_disk_neighbors_3d(30.0, box=(64.0, 128.0, 128.0))
        specter.seed(0)
        b2 = poisson_disk_neighbors(30.0, box=(128.0, 128.0))
    finally:
        torch.set_num_threads(before)
    assert torch.equal(a3, b3) and len(a3) > 5
    assert torch.equal(a2, b2) and len(a2) > 5
    assert torch.get_num_threads() == before
