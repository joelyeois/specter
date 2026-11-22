"""
Shared by the data-parallel tests: a small reconstruction problem, and a
`Reconstructor` that records which particles every rank stepped on.

A module of its own, not a test file, because the spawned ranks unpickle
the recording model by its module path, and a test module imported under
xdist is not reliably importable by that name in a fresh process.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from specter.ghostbuster import Reconstructor
from specter.settings import Propagation

NXY = 16
PIXEL_SIZE = 1.5
VOLTAGE = 300.0
DOSE = 22.5


class RecordingReconstructor(Reconstructor):
    """A `Reconstructor` that keeps every batch's particle indices, per rank."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[list[int]] = []
        self.seen_by_rank: list[list[list[int]]] = []

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        self.seen.append(batch[1].tolist())
        return super().training_step(batch, batch_idx)

    def on_train_end(self) -> None:
        if dist.is_initialized():
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.seen)
            self.seen_by_rank = gathered
        else:
            self.seen_by_rank = [self.seen]

    def global_batches(self) -> list[list[int]]:
        """Each step's particles over every rank, in rank order."""
        return [
            [i for rank in self.seen_by_rank for i in rank[step]]
            for step in range(len(self.seen_by_rank[0]))
        ]


def make_problem(n_particles: int, seed: int = 0) -> dict[str, Any]:
    """Poses, CTF and images for `n_particles` particles of a 16^3 map."""
    import roma

    # roma's samplers draw from the global stream whatever generator they
    # are given, so seed that.
    torch.manual_seed(seed)
    return dict(
        V0=torch.rand(NXY, NXY, NXY) * 0.02,
        quats=roma.random_unitquat(size=(n_particles,)),
        trans=(torch.rand(n_particles, 2) - 0.5) * 5,
        ctf={
            "dfu": torch.empty(n_particles).uniform_(9000, 11000),
            "dfv": torch.empty(n_particles).uniform_(9000, 11000),
            "dfang": torch.empty(n_particles).uniform_(0, 180),
            "cs": torch.full((n_particles,), 2.7e7),
            "phaseshift": torch.zeros(n_particles),
        },
        scale=0.8 + 0.4 * torch.rand(n_particles),
        images=DOSE * PIXEL_SIZE**2 * (1 + 0.1 * torch.randn(n_particles, NXY, NXY)),
    )


def make_model(
    problem: dict[str, Any], scattering_model: str, **kwargs: Any
) -> RecordingReconstructor:
    """A fresh model on `problem`, real-space rotation, k-masked."""
    from specter.arrays import ball3d

    ctf = {k: v.clone() for k, v in problem["ctf"].items()}
    # A column of a table, as a parameter read from a .cs file is: a strided
    # view, which the ranks' startup broadcast must handle.
    ctf["cs"] = torch.stack([ctf["cs"], ctf["cs"]], dim=1)[:, 0]
    assert not ctf["cs"].is_contiguous()
    return RecordingReconstructor(
        problem["V0"].clone(),
        PIXEL_SIZE,
        problem["quats"].clone(),
        problem["trans"].clone(),
        ctf,
        VOLTAGE,
        DOSE,
        scale=problem["scale"].clone(),
        kmask=ball3d(NXY, NXY),
        propagation=Propagation(
            alpha=0.07, scattering_model=scattering_model, rotate_mode="real"
        ),
        **kwargs,
    )


class DiesWhenUnpickled:
    """Kills the process that unpickles it: a rank that dies before joining."""

    def __reduce__(self) -> tuple[Any, tuple[int]]:
        import os

        return (os._exit, (5,))


def dies_on_rank_1(
    rank: int, devices: list[torch.device], when: str, *rest: Any
) -> str:
    """Rank 1 dies after one collective (``when="mid"``); rank 0 waits in the next."""
    import os

    t = torch.ones(4, device=devices[rank])
    dist.all_reduce(t)
    if rank == 1 and when == "mid":
        os._exit(3)  # a hard death, as an out-of-memory kill would be
    dist.all_reduce(t)
    return f"{float(t.sum())}"
