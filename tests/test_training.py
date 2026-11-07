"""
The reconstructors' training loop (`specter.ghostbuster.fit`) keeps the
behaviour Lightning's `Trainer` had where a reconstruction can see it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from specter.ghostbuster import fit


class _Recorder(nn.Module):
    """Two parameters, two optimisers, and a log of every hook it sees."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))
        self.log = log
        self.steps: list[int] = []

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        return [
            torch.optim.SGD([self.a], lr=0.1),
            torch.optim.SGD([self.b], lr=0.1),
        ], []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        (x,) = batch
        loss = ((self.a - x) ** 2 + (self.b + x) ** 2).mean()
        for opt in self._training.optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in self._training.optimizers:
            opt.step()
            self._training.global_step += 1
        self.steps.append(self._training.global_step)
        return loss

    def on_train_batch_end(self, batch: Any, batch_idx: int) -> None:
        self.log.append(f"model:batch_end:{batch_idx}")

    def on_train_epoch_end(self) -> None:
        self.log.append(f"model:epoch_end:{self._training.epoch}")

    def on_fit_end(self) -> None:
        self.log.append(f"model:fit_end:{self.a.device.type}")


class _Callback:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def on_train_batch_end(self, model: nn.Module, batch: Any, batch_idx: int) -> None:
        self.log.append(f"callback:batch_end:{batch_idx}")

    def on_fit_end(self, model: nn.Module) -> None:
        self.log.append("callback:fit_end")


def test_hooks_run_callbacks_first_and_end_on_the_cpu() -> None:
    log: list[str] = []
    model = _Recorder(log)
    loader = DataLoader(TensorDataset(torch.ones(4)), batch_size=2)
    fit(
        model,
        loader,
        max_epochs=2,
        device="cpu",
        callbacks=[_Callback(log)],
        progress=False,
    )
    assert log == [
        "callback:batch_end:0",
        "model:batch_end:0",
        "callback:batch_end:1",
        "model:batch_end:1",
        "model:epoch_end:0",
        "callback:batch_end:0",
        "model:batch_end:0",
        "callback:batch_end:1",
        "model:batch_end:1",
        "model:epoch_end:1",
        "callback:fit_end",
        "model:fit_end:cpu",
    ]
    # One global step per optimiser per batch, as Lightning counted them.
    assert model.steps == [2, 4, 6, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_model_is_back_on_the_cpu_before_on_fit_end() -> None:
    log: list[str] = []
    model = _Recorder(log)
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    fit(model, loader, max_epochs=1, device=0, precision="16-mixed", progress=False)
    assert log[-1] == "model:fit_end:cpu"


def test_an_unknown_precision_is_refused() -> None:
    model = _Recorder([])
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    with pytest.raises(ValueError, match="precision"):
        fit(model, loader, max_epochs=1, device="cpu", precision="8-mixed")
