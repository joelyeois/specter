"""
Tests for TomogramReconstructor: forward simulation, FOV masking, k-mask
application, optimizer/scheduler wiring, and run_dir file output.

Companion to test_ghostbuster.py, which covers Reconstructor. There were no
tests at all for TomogramReconstructor prior to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import roma
import torch
import torch.utils.data

from specter.arrays import ball3d
from specter.ghostbuster import TomogramReconstructor
from specter.settings import Propagation, TiltGeometry
from conftest import fit_one_epoch

SCHEDULERS = [
    "LambdaLR",
    "OneCycleLR",
    "CosineAnnealingWarmRestarts",
    "MultiplicativeLR",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_volume() -> torch.Tensor:
    """3D volume (8, 8, 8) with a box phantom."""
    volume = torch.zeros(8, 8, 8)
    volume[2:6, 2:6, 2:6] = 50.0
    return volume


@pytest.fixture
def tilt_quaternions() -> torch.Tensor:
    """Quaternions for 3 tilts about the x-axis: -20, 0, 20 degrees."""
    angles_deg = torch.tensor([-20.0, 0.0, 20.0])
    theta = torch.deg2rad(angles_deg)
    rotvecs = torch.stack(
        [theta, torch.zeros_like(theta), torch.zeros_like(theta)], dim=-1
    )
    return roma.rotvec_to_unitquat(rotvecs)


@pytest.fixture
def tilt_ctf_params() -> dict[str, torch.Tensor]:
    """Minimal per-tilt CTF parameters for 3 tilts."""
    n = 3
    return {
        "dfu": torch.full((n,), 5000.0),
        "cs": torch.full((n,), 2.7),
    }


@pytest.fixture
def tr_kwargs(
    small_volume: torch.Tensor,
    tilt_quaternions: torch.Tensor,
    tilt_ctf_params: dict[str, torch.Tensor],
) -> dict:
    """Shared TomogramReconstructor constructor kwargs (no scattering_model)."""
    return dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=tilt_quaternions,
        translations=torch.zeros(3, 2),
        ctf_params=tilt_ctf_params,
        voltage=300.0,
        dose_per_angstrom=1.0,
    )


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def test_forward_multislice_runs_and_is_finite(tr_kwargs: dict) -> None:
    """The multislice path (exercising _compute_nz_tilt + defocus z-offset
    correction) runs end-to-end and produces a finite image at high tilt."""
    model = TomogramReconstructor(
        **tr_kwargs,
        propagation=Propagation(scattering_model="multislice"),
    )
    img = model.forward(0)  # -20 degree tilt
    assert img.shape == (8, 8)
    assert torch.isfinite(img).all()


def test_forward_predicts_counts_in_proportion_to_each_tilts_dose(
    tr_kwargs: dict,
) -> None:
    """The forward model scales with the tilt's own dose per pixel."""
    kwargs = dict(tr_kwargs, propagation=Propagation(scattering_model="projection"))
    unit = TomogramReconstructor(**kwargs)
    kwargs["dose_per_angstrom"] = torch.tensor([1.0, 3.0, 1.0])
    dosed = TomogramReconstructor(**kwargs)
    pixel_area = tr_kwargs["voxel_size"] ** 2
    assert torch.allclose(unit.forward(1) * 3.0, dosed.forward(1))
    assert torch.allclose(unit.forward(0), dosed.forward(0))
    assert float(unit.forward(1).mean()) == pytest.approx(pixel_area, rel=0.05)


def test_dose_must_have_one_entry_per_tilt(tr_kwargs: dict) -> None:
    kwargs = dict(tr_kwargs, dose_per_angstrom=torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="one per tilt"):
        TomogramReconstructor(**kwargs)


# ---------------------------------------------------------------------------
# Real-FOV mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt_axis", ["x", "y"], ids=["tilt_axis_x", "tilt_axis_y"])
def test_fov_mask_cache_matches_the_per_step_computation(
    tr_kwargs: dict, tilt_axis: str
) -> None:
    """The cached per-tilt border widths give the mask the per-step
    evaluation from the tilt pose gave, for every tilt. At high tilt the
    border perpendicular to the tilt axis (Y for tilt_axis='x', X for 'y') is
    zeroed and the center remains real FOV."""
    model = TomogramReconstructor(
        **tr_kwargs,
        propagation=Propagation(scattering_model="projection"),
        tilt=TiltGeometry(tilt_axis=tilt_axis),
    )
    mask = model._fov_mask(0)  # -20 degree tilt
    assert mask is not None
    assert mask.shape == (8, 8)
    across = mask if tilt_axis == "x" else mask.T
    assert torch.all(across[0, :] == 0.0)
    assert torch.all(across[-1, :] == 0.0)
    assert mask[4, 4] == 1.0

    for idx, Q in enumerate(model.quaternions):
        theta = roma.unitquat_to_rotvec(Q.unsqueeze(0))[0].norm()
        real_fov = int(
            (model.nxy * torch.cos(theta) - model.nz * torch.sin(theta))
            .clamp(min=1)
            .item()
        )
        mask = model._fov_mask(idx)
        if real_fov >= model.nxy:
            assert mask is None
            continue
        pad = (model.nxy - real_fov) // 2
        expected = torch.ones(model.nxy, model.nxy)
        if tilt_axis == "x":
            expected[:pad, :] = 0.0
            expected[model.nxy - pad :, :] = 0.0
        else:
            expected[:, :pad] = 0.0
            expected[:, model.nxy - pad :] = 0.0
        assert mask is not None
        assert torch.equal(mask, expected)
    assert model._fov_pads() is model._fov_pads()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_configure_optimizers_all_schedulers(tr_kwargs: dict, scheduler: str) -> None:
    """Each supported scheduler string trains for one epoch without error."""
    torch.manual_seed(0)
    images = torch.randn(3, 8, 8)
    model = TomogramReconstructor(
        **tr_kwargs,
        lr=0.1,
        scheduler=scheduler,
        propagation=Propagation(scattering_model="projection"),
    )
    V_init = model.V.data.clone()
    fit_one_epoch(model, images, batch_size=3)
    assert not torch.equal(model.V.data, V_init)
    assert len(model.log_lrs) == 1


def test_configure_optimizers_returns_empty_when_lr_none(tr_kwargs: dict) -> None:
    """lr=None disables volume optimisation: no optimizers, no schedulers."""
    model = TomogramReconstructor(
        **tr_kwargs,
        lr=None,
        propagation=Propagation(scattering_model="projection"),
    )
    opts, schedulers = model.configure_optimizers()
    assert opts == []
    assert schedulers == []


# ---------------------------------------------------------------------------
# run_dir file output
# ---------------------------------------------------------------------------


def test_run_dir_writes_expected_artifacts(tmp_path: Path, tr_kwargs: dict) -> None:
    """A configured run_dir receives params, per-epoch volumes, and metrics."""
    torch.manual_seed(0)
    images = torch.randn(3, 8, 8)
    n = tr_kwargs["V"].shape[-1]
    model = TomogramReconstructor(
        **tr_kwargs,
        lr=0.1,
        kmask=ball3d(n, n),
        run_dir=tmp_path,
        propagation=Propagation(scattering_model="projection"),
    )
    fit_one_epoch(model, images, batch_size=3, max_epochs=2)

    assert (tmp_path / "params.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "volume.mrc").exists()
    assert (tmp_path / "kmask.pt").exists()
    assert (tmp_path / "epochs" / "001.mrc").exists()
    assert (tmp_path / "epochs" / "002.mrc").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["total_batches"] == 2  # 1 batch/epoch x 2 epochs
    assert {"epoch_01", "epoch_02"} <= set(metrics["epochs"])
