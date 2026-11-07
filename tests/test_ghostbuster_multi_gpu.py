"""
Does Reconstructor's multi-GPU training (`fit` with several devices, as
`Ghostbuster.run(device=[...])` calls it) actually work?

Correctness check: two GPUs with per-GPU batch size B are mathematically
equivalent to one GPU with one combined batch of 2B (same particles), since
each rank's loss is already a mean over its own local batch and the loop
averages gradients across ranks -- the two normalizations compose to match a
single combined-batch run exactly. This also exercises pose refinement
(lr_R), which shares full-dataset-length parameters across ranks differently
than the volume V does.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from specter.ghostbuster import Reconstructor, fit
from specter.settings import Propagation

NXY = NZ = 16
N_PARTICLES = 4
PIXEL_SIZE = 1.5
VOLTAGE = 300.0
DOSE = 22.5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 GPUs",
)


def _make_problem() -> dict:
    import roma

    # roma's samplers draw from the global stream whatever generator they
    # are given, so seed that.
    torch.manual_seed(0)
    V0 = torch.rand(NZ, NXY, NXY) * 0.02
    quats = roma.random_unitquat(size=(N_PARTICLES,))
    trans = (torch.rand(N_PARTICLES, 2) - 0.5) * 5
    ctf = {
        "dfu": torch.empty(N_PARTICLES).uniform_(9000, 11000),
        "dfv": torch.empty(N_PARTICLES).uniform_(9000, 11000),
        "dfang": torch.empty(N_PARTICLES).uniform_(0, 180),
        "cs": torch.full((N_PARTICLES,), 2.7e7),
        "phaseshift": torch.zeros(N_PARTICLES),
        "tiltx": torch.zeros(N_PARTICLES),
        "tilty": torch.zeros(N_PARTICLES),
        "trefoil1": torch.zeros(N_PARTICLES),
        "trefoil2": torch.zeros(N_PARTICLES),
    }
    images = torch.rand(N_PARTICLES, NXY, NXY)
    return dict(V0=V0, quats=quats, trans=trans, ctf=ctf, images=images)


def _train(problem: dict, devices: list[int], batch_size: int) -> Reconstructor:
    """Train a fresh model for one epoch on `devices`, in file order."""
    model = Reconstructor(
        problem["V0"].clone(),
        PIXEL_SIZE,
        problem["quats"].clone(),
        problem["trans"].clone(),
        {k: v.clone() for k, v in problem["ctf"].items()},
        VOLTAGE,
        DOSE,
        lr=0.1,
        lr_R=0.01,
        symmetry=None,
        propagation=Propagation(
            alpha=0.0,
            scattering_model="rytov",
            rotate_mode="real",
            ews_curvature_sign="negative",
        ),
    )
    loader = DataLoader(
        TensorDataset(problem["images"], torch.arange(N_PARTICLES)),
        batch_size=batch_size,
        shuffle=False,
    )
    return fit(model, loader, max_epochs=1, device=devices, progress=False)


def test_reconstructor_multi_gpu_matches_single_gpu() -> None:
    problem = _make_problem()
    single = _train(problem, [0], N_PARTICLES)
    multi = _train(problem, [0, 1], N_PARTICLES // 2)

    for model in (single, multi):
        V = model.V.detach()
        assert V.device == torch.device("cpu")
        assert not torch.isnan(V).any()
        # Moved from initialization: gradients flowed and the step took effect.
        assert (V - problem["V0"]).norm() > 1e-6

    V_relerr = (single.V - multi.V).norm() / single.V.norm()
    rot_relerr = (single.rotations - multi.rotations).norm() / single.rotations.norm()
    assert V_relerr < 1e-3, f"V mismatch between single- and multi-GPU: {V_relerr}"
    assert rot_relerr < 1e-3, (
        f"rotations mismatch between single- and multi-GPU: {rot_relerr}"
    )
    # The logged loss is the mean over ranks, which is the combined batch's.
    torch.testing.assert_close(
        torch.stack(multi.log_total_loss).cpu(),
        torch.stack(single.log_total_loss).cpu(),
        rtol=1e-4,
        atol=0,
    )
