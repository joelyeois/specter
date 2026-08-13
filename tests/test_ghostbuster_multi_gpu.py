"""
Does Reconstructor's multi-GPU wiring (build_trainer's device-list ->
strategy="ddp" path, used by Ghostbuster.run(device=[...])) actually work?

Lightning's "ddp" strategy launches worker processes by re-invoking the
current command line, which doesn't play well with being launched from
inside a pytest process. So the real DDP run is driven as a subprocess that
re-invokes this file directly (`python test_ghostbuster_multi_gpu.py
single|multi`) -- exactly how a user would run it as a script -- and the
pytest test just checks the subprocess succeeded and that the results are
correct.

Correctness check: 2-GPU DDP with per-GPU batch size B is mathematically
equivalent to single-GPU with one combined batch of 2B (same particles),
since each rank's loss is already a mean over its own local batch and DDP
averages gradients across ranks -- the two normalizations should compose to
match a single combined-batch run exactly. This also exercises pose
refinement (lr_R), which shares full-dataset-length parameters across ranks
differently than the volume V does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

NXY = NZ = 16
N_PARTICLES = 4
PIXEL_SIZE = 1.5
VOLTAGE = 300.0
DOSE = 22.5


def _make_problem(seed: int = 0):
    import roma

    g = torch.Generator().manual_seed(seed)
    V0 = torch.rand(NZ, NXY, NXY, generator=g) * 0.02
    quats = roma.random_unitquat(size=(N_PARTICLES,), generator=g)
    trans = (torch.rand(N_PARTICLES, 2, generator=g) - 0.5) * 5
    ctf = {
        "dfu": torch.empty(N_PARTICLES).uniform_(9000, 11000, generator=g),
        "dfv": torch.empty(N_PARTICLES).uniform_(9000, 11000, generator=g),
        "dfang": torch.empty(N_PARTICLES).uniform_(0, 180, generator=g),
        "cs": torch.full((N_PARTICLES,), 2.7e7),
        "phaseshift": torch.zeros(N_PARTICLES),
        "tiltx": torch.zeros(N_PARTICLES),
        "tilty": torch.zeros(N_PARTICLES),
        "trefoil1": torch.zeros(N_PARTICLES),
        "trefoil2": torch.zeros(N_PARTICLES),
    }
    scale = torch.ones(N_PARTICLES)
    images = torch.rand(N_PARTICLES, NXY, NXY, generator=g)
    return dict(V0=V0, quats=quats, trans=trans, ctf=ctf, scale=scale, images=images)


def _build_model(problem, lr_R):
    from specter.ghostbuster import Reconstructor

    return Reconstructor(
        problem["V0"].clone(),
        PIXEL_SIZE,
        problem["quats"].clone(),
        problem["trans"].clone(),
        {k: v.clone() for k, v in problem["ctf"].items()},
        VOLTAGE,
        DOSE,
        alpha=0.0,
        scale=problem["scale"].clone(),
        scattering_model="rytov",
        aberration_model="holography",
        lr=0.1,
        lr_R=lr_R,
        symmetry=None,
        sparsity=0,
        rotate_mode="real",
        ews_curvature_sign="negative",
    )


def _run(devices, strategy, batch_size, out_path, problem_path):
    """Train one epoch and save the resulting V/rotations. Rank-0-only save."""
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    problem = torch.load(problem_path, weights_only=False)
    model = _build_model(problem, lr_R=0.01)
    idx = torch.arange(N_PARTICLES)
    loader = DataLoader(
        TensorDataset(problem["images"], idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    if trainer.is_global_zero:
        torch.save(
            dict(
                V_final=model.V.detach().cpu(),
                rot_final=model.rotations.detach().cpu(),
                log_norm_loss=torch.stack(model.log_norm_loss),
                log_total_loss=torch.stack(model.log_total_loss),
            ),
            out_path,
        )


if __name__ == "__main__":
    # sys.argv: [script, mode, out_path, problem_path]
    mode, out_path, problem_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "single":
        _run(
            devices=[0],
            strategy="auto",
            batch_size=N_PARTICLES,
            out_path=out_path,
            problem_path=problem_path,
        )
    elif mode == "multi":
        _run(
            devices=[0, 1],
            strategy="ddp",
            batch_size=N_PARTICLES // 2,
            out_path=out_path,
            problem_path=problem_path,
        )
    else:
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 GPUs",
)


def test_reconstructor_multi_gpu_ddp_matches_single_gpu(tmp_path: Path):
    problem_path = tmp_path / "problem.pt"
    torch.save(_make_problem(), problem_path)

    single_out = tmp_path / "single.pt"
    multi_out = tmp_path / "multi.pt"

    for mode, out in [("single", single_out), ("multi", multi_out)]:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                mode,
                str(out),
                str(problem_path),
            ],
            capture_output=True,
            encoding="utf-8",
            timeout=300,
        )
        assert result.returncode == 0, (
            f"{mode}-GPU run failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        assert out.exists(), f"{mode}-GPU run did not produce {out}"

    single = torch.load(single_out, weights_only=False)
    multi = torch.load(multi_out, weights_only=False)

    assert not torch.isnan(single["V_final"]).any()
    assert not torch.isnan(multi["V_final"]).any()

    # Both must have actually moved from initialization -- confirms gradients
    # flowed and the optimizer step took effect, not just "didn't crash".
    problem = torch.load(problem_path, weights_only=False)
    assert (single["V_final"] - problem["V0"]).norm() > 1e-6
    assert (multi["V_final"] - problem["V0"]).norm() > 1e-6

    V_relerr = (single["V_final"] - multi["V_final"]).norm() / single["V_final"].norm()
    rot_relerr = (single["rot_final"] - multi["rot_final"]).norm() / single[
        "rot_final"
    ].norm()
    assert V_relerr < 1e-3, f"V mismatch between single- and multi-GPU: {V_relerr}"
    assert rot_relerr < 1e-3, (
        f"rotations mismatch between single- and multi-GPU: {rot_relerr}"
    )

    # Logged metrics: multi-GPU's rank-0-saved log_norm_loss/log_total_loss
    # should be the *gathered* (mean-across-ranks) value, not just rank 0's
    # own local-batch loss -- and since both ranks have equal-sized shards
    # here, that gathered mean must equal the single-GPU combined-batch loss
    # exactly (mean-of-equal-size-group-means == overall mean).
    assert single["log_norm_loss"].shape == multi["log_norm_loss"].shape == (1,)
    norm_relerr = (single["log_norm_loss"] - multi["log_norm_loss"]).abs() / single[
        "log_norm_loss"
    ].abs()
    total_relerr = (single["log_total_loss"] - multi["log_total_loss"]).abs() / single[
        "log_total_loss"
    ].abs()
    assert norm_relerr < 1e-4, (
        f"log_norm_loss not gathered correctly under multi-GPU: "
        f"single={single['log_norm_loss'].item()} multi={multi['log_norm_loss'].item()}"
    )
    assert total_relerr < 1e-4, (
        f"log_total_loss not gathered correctly under multi-GPU: "
        f"single={single['log_total_loss'].item()} multi={multi['log_total_loss'].item()}"
    )
