"""
Does Reconstructor's multi-GPU training (`fit` with several devices, as
`Ghostbuster.run(device=[...])` calls it) take the steps one GPU takes?

The loader's batch size is the global batch: rank 0 draws the epoch's order
as one GPU would, each rank takes its share of every global batch, weights
its loss by that share, and the gradients are summed. So two GPUs and one
see the same particles at every step and end on the same map, to float
rounding. The CPU-rank versions of these tests (`test_data_parallel.py`)
cover the scattering models and the solver; these run the NCCL path,
mixed precision and the auto micro-batch, which only exist on GPUs.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from _data_parallel_helpers import RecordingReconstructor, make_model, make_problem
from specter.ghostbuster import fit

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 GPUs",
)

N_PARTICLES = 10
BATCH = 4  # 4, 4, 2: the last global batch is short


def _train(problem: dict, devices: list[int], **kwargs: Any) -> RecordingReconstructor:
    """A fresh model fitted for two shuffled epochs on `devices`."""
    precision = kwargs.pop("precision", "32")
    model = make_model(problem, "rytov", lr=0.1, lr_T=0.01, **kwargs)
    loader = DataLoader(
        TensorDataset(problem["images"], torch.arange(N_PARTICLES)),
        batch_size=BATCH,
        shuffle=True,
    )
    torch.manual_seed(5)
    return fit(
        model,
        loader,
        max_epochs=2,
        device=devices,
        precision=precision,
        progress=False,
    )


def _assert_same_run(
    single: RecordingReconstructor, multi: RecordingReconstructor, tol: float
) -> None:
    assert multi.global_batches() == single.global_batches()
    # Three batches, two epochs, two optimisers (volume and shifts).
    assert multi.global_step == single.global_step == 12
    for name in ("V", "translations"):
        a, b = getattr(single, name).detach(), getattr(multi, name).detach()
        assert a.device == torch.device("cpu")
        relerr = float((a - b).abs().max() / a.abs().max())
        assert relerr < tol, f"{name} differs by {relerr:.2e} between one GPU and two"
    torch.testing.assert_close(
        torch.stack(multi.log_total_loss).cpu(),
        torch.stack(single.log_total_loss).cpu(),
        rtol=tol,
        atol=0,
    )


@pytest.mark.parametrize("micro_batchsize", [None, 1, "auto"])
def test_two_gpus_take_the_one_gpu_steps(micro_batchsize: Any) -> None:
    problem = make_problem(N_PARTICLES)
    single = _train(problem, [0], micro_batchsize=micro_batchsize)
    multi = _train(problem, [0, 1], micro_batchsize=micro_batchsize)
    assert not torch.isnan(multi.V).any()
    assert (multi.V - problem["V0"]).norm() > 1e-6
    _assert_same_run(single, multi, 1e-4)


def test_two_gpus_take_the_one_gpu_steps_under_16_mixed() -> None:
    """One loss-scale decision for every rank: the overflow check runs on
    the summed gradient, which is the same on both."""
    problem = make_problem(N_PARTICLES)
    single = _train(problem, [0], precision="16-mixed")
    multi = _train(problem, [0, 1], precision="16-mixed")
    # Half-precision intermediates, summed in another order.
    _assert_same_run(single, multi, 2e-2)
