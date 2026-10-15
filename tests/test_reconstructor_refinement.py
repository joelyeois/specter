"""
Pose, shift and defocus refinement in `Reconstructor`, against ground truth.

Images are simulated noiselessly at known poses, shifts and defoci from a
small synthetic volume; the reconstructor starts from the true volume,
frozen, with one parameter group perturbed, and must refine it back. Each
group recovers its truth to well under a tenth of its perturbation; with
Poisson noise they settle where a start from the truth also settles, the
noise-limited optimum (measured on 6BDF at 4 A, 16 particles at 50 e/A^2:
0.14 degrees, 0.07 A and 3 A).
"""

from __future__ import annotations

import math

import pytest
import roma
import torch
import torch.nn.functional as F

from specter.ghostbuster import Reconstructor
from specter.ghostbuster._losses import mse_loss
from specter.imaging import CTF, Exposure, Poses, Specimen, simulate_particles

N, BOX, VOXEL, DOSE, VOLTAGE = 8, 32, 4.0, 50.0, 300.0


def _volume() -> torch.Tensor:
    """
    Twelve small Gaussian blobs, seeded and centred on the box.

    Enough small features that every rotation is visible. A few broad blobs
    are not: a small rotation then moves little but their centroid, which a
    shift can stand in for, and joint rotation and shift refinement wanders.
    """
    g = torch.Generator().manual_seed(7)
    axis = torch.arange(BOX, dtype=torch.float32) - BOX / 2
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    centres = (torch.rand(12, 3, generator=g) - 0.5) * 16
    centres -= centres.mean(dim=0)
    sigmas = 1.0 + torch.rand(12, generator=g)
    volts = 5.0 + 5.0 * torch.rand(12, generator=g)
    V = torch.zeros(BOX, BOX, BOX)
    for (cz, cy, cx), sigma, height in zip(centres, sigmas, volts):
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        V += height * torch.exp(-r2 / (2 * sigma**2))
    return V


@pytest.fixture(scope="module")
def truth() -> tuple[torch.Tensor, Poses, CTF, torch.Tensor]:
    V = _volume()
    poses = Poses.random(N, shift=4.0, seed=0)
    ctf = CTF.sample(N, defocus=[10000.0, 20000.0], seed=0)
    with torch.no_grad():
        images = simulate_particles(
            Specimen(V, VOXEL),
            poses,
            ctf,
            Exposure(DOSE),
            voltage=VOLTAGE,
            batch_size=N,
            coherence=dict(cc=None),
            camera=dict(noise=None),
        ).array
    return V, poses, ctf, images


def _perturbed(
    poses: Poses, ctf: CTF, rotation_deg: float, shift: float, defocus: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(1)
    axis = F.normalize(torch.randn(N, 3, generator=g), dim=-1)
    turn = roma.rotvec_to_unitquat(axis * math.radians(rotation_deg))
    quaternions = roma.quat_product(turn, poses.quaternions)
    shifts = poses.shifts + shift * torch.randn(N, 2, generator=g)
    params = {k: v.clone() for k, v in ctf.params.items()}
    params["dfu"] = params["dfu"] + defocus
    params["dfv"] = params["dfv"] + defocus
    return quaternions, shifts, params


def _errors(model: Reconstructor, poses: Poses, defocus: float) -> dict[str, float]:
    """Mean rotation (degrees) and shift (A) error, and the defocus error (A)."""
    q = F.normalize(model.rotations.detach(), dim=-1)
    angle = roma.rotmat_geodesic_distance(
        roma.unitquat_to_rotmat(q), roma.unitquat_to_rotmat(poses.quaternions)
    )
    return {
        "R": float(torch.rad2deg(angle).mean()),
        "T": float((model.translations.detach() - poses.shifts).norm(dim=-1).mean()),
        "D": abs(float(model.defocus_offset.detach()) + defocus),
    }


@pytest.mark.parametrize(
    "groups, steps",
    [("R", 150), ("T", 150), ("D", 150), ("RTD", 200)],
    ids=["rotations", "shifts", "defocus", "joint"],
)
def test_refinement_recovers_the_true_parameters(truth, groups: str, steps: int):
    """
    The defocus case is the one AdamW's default weight decay broke: it pulled
    the offset towards zero by ``lr * 0.01 * offset`` a step against an Adam
    step of ~``lr``, so a 300 A error stalled near 200 A at any learning rate.
    """
    V, poses, ctf, images = truth
    rotation, shift, defocus = 3.0, 2.0, 300.0
    quaternions, shifts, params = _perturbed(
        poses,
        ctf,
        rotation if "R" in groups else 0.0,
        shift if "T" in groups else 0.0,
        defocus if "D" in groups else 0.0,
    )
    applied = defocus if "D" in groups else 0.0
    model = Reconstructor(
        V.clone(),
        voxel_size=VOXEL,
        quaternions=quaternions,
        translations=shifts,
        ctf_params=params,
        voltage=VOLTAGE,
        dose_per_angstrom=DOSE,
        lr=None,
        lr_R=1e-2 if "R" in groups else None,
        lr_T=0.1 if "T" in groups else None,
        lr_D=20.0 if "D" in groups else None,
    )
    before = _errors(model, poses, applied)
    optimizers, _ = model.configure_optimizers()
    index = torch.arange(N)
    for _ in range(steps):
        loss = mse_loss(model(index), images, model.scale)
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()
    after = _errors(model, poses, applied)

    tolerance = {"R": 0.05, "T": 0.02, "D": 2.0}  # degrees, A, A
    for group in groups:
        assert before[group] > 20 * tolerance[group]
        assert after[group] < tolerance[group], (group, before[group], after[group])


def test_a_trainer_run_refines_shifts_and_defocus(truth, tmp_path):
    """The production path: Lightning's manual-optimisation step over several
    optimisers moves the refined parameters to their truth."""
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    V, poses, ctf, images = truth
    _, shifts, params = _perturbed(poses, ctf, 0.0, 2.0, 300.0)
    model = Reconstructor(
        V.clone(),
        voxel_size=VOXEL,
        quaternions=poses.quaternions,
        translations=shifts,
        ctf_params=params,
        voltage=VOLTAGE,
        dose_per_angstrom=DOSE,
        lr=None,
        lr_T=0.1,
        lr_D=20.0,
    )
    loader = DataLoader(TensorDataset(images, torch.arange(N)), batch_size=N)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=150,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, loader)
    errors = _errors(model, poses, 300.0)
    assert errors["T"] < 0.02
    assert errors["D"] < 2.0
