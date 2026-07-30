"""
Tests for TomogramReconstructor: forward simulation, FOV masking, k-mask
application, optimizer/scheduler wiring, and run_dir file output.

Companion to test_ghostbuster.py, which covers Reconstructor. There were no
tests at all for TomogramReconstructor prior to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import pytest
import roma
import torch
import torch.utils.data

from specter.arrays import ball3d
from specter.fft import fft3
from specter.ghostbuster import TomogramReconstructor

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
    vol = torch.zeros(8, 8, 8)
    vol[2:6, 2:6, 2:6] = 50.0
    return vol


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
    )


def _fit_one_epoch(
    model: TomogramReconstructor,
    images: torch.Tensor,
    batch_size: int = 1,
    max_epochs: int = 1,
) -> None:
    idx = torch.arange(len(images))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, idx), batch_size=batch_size
    )
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=max_epochs,
        precision="32",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def test_forward_output_shape(tr_kwargs: dict) -> None:
    """forward(tilt_idx) returns a (H, W) intensity image matching the volume box."""
    model = TomogramReconstructor(**tr_kwargs, scattering_model="projection")
    img = model.forward(1)
    assert img.shape == (8, 8)


def test_forward_is_deterministic(tr_kwargs: dict) -> None:
    """Two identical forward calls without noise produce identical output."""
    model = TomogramReconstructor(**tr_kwargs, scattering_model="projection")
    img1 = model.forward(0)
    img2 = model.forward(0)
    assert torch.equal(img1, img2)


def test_forward_multislice_runs_and_is_finite(tr_kwargs: dict) -> None:
    """The multislice path (exercising _compute_nz_tilt + defocus z-offset
    correction) runs end-to-end and produces a finite image at high tilt."""
    model = TomogramReconstructor(**tr_kwargs, scattering_model="multislice")
    img = model.forward(0)  # -20 degree tilt
    assert img.shape == (8, 8)
    assert torch.isfinite(img).all()


# ---------------------------------------------------------------------------
# Real-FOV mask
# ---------------------------------------------------------------------------


def test_fov_mask_none_at_zero_tilt(tr_kwargs: dict) -> None:
    """At (near-)zero tilt the full image is real FOV, so the mask is None."""
    model = TomogramReconstructor(**tr_kwargs, scattering_model="projection")
    assert model._fov_mask(1) is None  # tilt_idx 1 == 0 degrees


def test_fov_mask_zeros_border_at_high_tilt(tr_kwargs: dict) -> None:
    """At high tilt (tilt_axis='x') the Y-border of the mask is zeroed and the
    center remains real FOV."""
    model = TomogramReconstructor(
        **tr_kwargs, scattering_model="projection", tilt_axis="x"
    )
    mask = model._fov_mask(0)  # -20 degree tilt
    assert mask is not None
    assert mask.shape == (8, 8)
    assert torch.all(mask[0, :] == 0.0)
    assert torch.all(mask[-1, :] == 0.0)
    assert mask[4, 4] == 1.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def test_training_updates_volume(tr_kwargs: dict) -> None:
    """One epoch of manual optimisation over all 3 tilts updates V."""
    torch.manual_seed(0)
    images = torch.randn(3, 8, 8)
    model = TomogramReconstructor(**tr_kwargs, scattering_model="projection", lr=0.1)
    V_init = model.V.data.clone()
    _fit_one_epoch(model, images, batch_size=3)
    assert not torch.equal(model.V.data, V_init)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_configure_optimizers_all_schedulers(tr_kwargs: dict, scheduler: str) -> None:
    """Each supported scheduler string trains for one epoch without error."""
    torch.manual_seed(0)
    images = torch.randn(3, 8, 8)
    model = TomogramReconstructor(
        **tr_kwargs, scattering_model="projection", lr=0.1, scheduler=scheduler
    )
    V_init = model.V.data.clone()
    _fit_one_epoch(model, images, batch_size=3)
    assert not torch.equal(model.V.data, V_init)
    assert len(model.log_lrs) == 1


def test_configure_optimizers_returns_empty_when_lr_none(tr_kwargs: dict) -> None:
    """lr=None disables volume optimisation: no optimizers, no schedulers."""
    model = TomogramReconstructor(**tr_kwargs, scattering_model="projection", lr=None)
    opts, schedulers = model.configure_optimizers()
    assert opts == []
    assert schedulers == []


def test_configure_optimizers_unknown_scheduler_raises(tr_kwargs: dict) -> None:
    """An unrecognised scheduler string raises ValueError from configure_optimizers."""
    model = TomogramReconstructor(
        **tr_kwargs, scattering_model="projection", lr=0.1, scheduler="BogusScheduler"
    )
    with pytest.raises(ValueError, match="Unknown scheduler"):
        model.configure_optimizers()


# ---------------------------------------------------------------------------
# Fourier k-mask
# ---------------------------------------------------------------------------


def test_kmask_zeros_high_frequencies(tr_kwargs: dict) -> None:
    """on_train_batch_end applies the Fourier k-mask in-place after each step."""
    torch.manual_seed(0)
    n = tr_kwargs["V"].shape[-1]
    kwargs = dict(tr_kwargs)
    kwargs["V"] = torch.randn(n, n, n)
    model = TomogramReconstructor(
        **kwargs, scattering_model="projection", lr=0.1, kmask=ball3d(n, n // 2)
    )
    model.on_train_batch_end(None, None, 0)
    spectrum = fft3(model.V.data, shift=True)
    outside_mask = spectrum[model.kmask == 0]
    assert torch.allclose(outside_mask, torch.zeros_like(outside_mask), atol=1e-5)


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
        scattering_model="projection",
        lr=0.1,
        kmask=ball3d(n, n),
        run_dir=tmp_path,
    )
    _fit_one_epoch(model, images, batch_size=3, max_epochs=2)

    assert (tmp_path / "params.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "vol.mrc").exists()
    assert (tmp_path / "kmask.pt").exists()
    assert (tmp_path / "epochs" / "001.mrc").exists()
    assert (tmp_path / "epochs" / "002.mrc").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["total_batches"] == 2  # 1 batch/epoch x 2 epochs
    assert {"epoch_01", "epoch_02"} <= set(metrics["epochs"])
