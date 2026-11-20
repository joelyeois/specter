"""
Several devices give the one-device result, to float rounding.

Run on CPU ranks over gloo, so the suite covers the data-parallel paths
without GPUs (`tests/test_ghostbuster_multi_gpu.py` runs NCCL, mixed
precision and the auto micro-batch on two GPUs):

* an Adam fit on three ranks steps on the same particles as one device,
  every global batch split across the ranks, with a short last batch that
  leaves one rank nothing, and ends on the same map;
* a converged solve on three ranks, each summing its share of the
  particles, ends on the same map as one device.

The tolerance is float32 summation order: the shares' gradients are summed
in another order than one device sums the whole batch's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from _data_parallel_helpers import RecordingReconstructor, make_model, make_problem
from specter.ghostbuster import fit
from specter.ghostbuster._converge import solve_on_devices
from specter.ghostbuster._run_helpers import torch_devices
from specter.pipelines._reconstruct import _half_devices

MODELS = ["rytov", "firstborn", "multislice"]
CPU3 = [torch.device("cpu")] * 3


def _fit(
    problem: dict[str, Any], scattering_model: str, device: Any, **kwargs: Any
) -> RecordingReconstructor:
    """Two shuffled epochs of 10 particles in global batches of 4, 4 and 2."""
    model = make_model(problem, scattering_model, lr=0.1, **kwargs)
    loader = DataLoader(
        TensorDataset(problem["images"], torch.arange(len(problem["images"]))),
        batch_size=4,
        shuffle=True,
    )
    torch.manual_seed(123)  # the order the one-device sampler draws
    return fit(model, loader, max_epochs=2, device=device, progress=False)


def _relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / a.abs().max())


@pytest.mark.parametrize("scattering_model", MODELS)
def test_three_ranks_take_the_one_device_steps(scattering_model: str) -> None:
    problem = make_problem(10)
    one = _fit(problem, scattering_model, "cpu")
    three = _fit(problem, scattering_model, ["cpu"] * 3)

    # The same global batches, and the last one, of two particles, leaves
    # the third rank an empty share.
    assert three.global_batches() == one.global_batches()
    assert [len(b) for b in one.global_batches()] == [4, 4, 2] * 2
    assert three.seen_by_rank[2][2] == []
    assert three.global_step == one.global_step == 6

    assert _relative_difference(one.V.detach(), three.V.detach()) < 1e-4
    torch.testing.assert_close(
        torch.stack(three.log_total_loss),
        torch.stack(one.log_total_loss),
        rtol=1e-5,
        atol=0,
    )


def test_three_ranks_with_micro_batches_take_the_one_device_steps() -> None:
    """Each rank accumulates its share in chunks, weighted by the global
    batch, and the ranks sum once, after the last chunk."""
    problem = make_problem(10)
    one = _fit(problem, "rytov", "cpu", micro_batchsize=None)
    three = _fit(problem, "rytov", ["cpu"] * 3, micro_batchsize=1)
    assert three.global_batches() == one.global_batches()
    assert _relative_difference(one.V.detach(), three.V.detach()) < 1e-4


@pytest.mark.parametrize("scattering_model", MODELS)
def test_a_solve_on_three_ranks_matches_one_device(scattering_model: str) -> None:
    """25 particles in shares of 9, 8 and 8; the line search's subsets too."""
    problem = make_problem(25)
    ssnr = np.linspace(4.0, 0.2, 9)
    results = [
        solve_on_devices(
            make_model(problem, scattering_model, lr=0.1),
            problem["images"],
            ssnr,
            3,
            devices,
            batchsize=4,
            seed=0,
        )
        for devices in ([torch.device("cpu")], CPU3)
    ]
    one, three = results
    assert _relative_difference(one.volume, three.volume) < 1e-3
    np.testing.assert_allclose(
        [e["objective"] for e in three.log],
        [e["objective"] for e in one.log],
        rtol=1e-6,
    )
    assert three.info["n_devices"] == 3
    assert three.info["batchsize_per_device"] == [4, 4, 4]


@pytest.mark.parametrize(
    ("pool", "a", "b"),
    [
        (["cuda:0"], ["cuda:0"], ["cuda:0"]),
        (["cuda:0", "cuda:1"], ["cuda:0"], ["cuda:1"]),
        (["cuda:0", "cuda:1", "cuda:2"], ["cuda:0", "cuda:2"], ["cuda:1"]),
        (
            [f"cuda:{i}" for i in range(8)],
            ["cuda:0", "cuda:2", "cuda:4", "cuda:6"],
            ["cuda:1", "cuda:3", "cuda:5", "cuda:7"],
        ),
    ],
)
def test_gold_halves_split_the_device_pool(
    pool: list[str], a: list[str], b: list[str]
) -> None:
    assert _half_devices(pool) == (a, b)


def test_torch_devices() -> None:
    assert torch_devices("cpu") == [torch.device("cpu")]
    assert torch_devices(["cpu", "cpu"]) == [torch.device("cpu")] * 2
    with pytest.raises(ValueError, match="mixes"):
        torch_devices([0, "cpu"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="only string"):
        torch_devices("cuda")
