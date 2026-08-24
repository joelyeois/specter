"""
Golden-output regression tests for Reconstructor.

Covers all scattering models and the full aberration parameter set:
defocus, spherical aberration, phase shift, beam tilt (tiltx/tilty),
trefoil (trefoil1/trefoil2), B-factor envelope, and anisotropic
magnification.

On the first run each test saves its output as a fixture under
tests/test_data/. Subsequent runs load the fixture and assert numerical
identity. Delete the corresponding .pt file and re-run to regenerate.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import mrcfile
import pytest
import torch
import torch.utils.data

from specter.arrays import ball3d
from specter.fft import fft3
from specter.ghostbuster import Reconstructor
from specter.symmetries import apply_symmetry, get_rotation_matrices

FIXTURE_DIR = Path(__file__).parent / "test_data"

SCATTERING_MODELS = ["multislice", "firstborn", "projection", "ctf", "rytov"]
SCHEDULERS = [
    "LambdaLR",
    "OneCycleLR",
    "CosineAnnealingWarmRestarts",
    "MultiplicativeLR",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_or_compare(name: str, tensor: torch.Tensor) -> None:
    """Save tensor as a fixture on first run; compare on subsequent runs."""
    path = FIXTURE_DIR / f"{name}.pt"
    if path.exists():
        expected = torch.load(path, weights_only=True)
        assert torch.allclose(tensor.float(), expected.float(), atol=1e-4), (
            f"Regression failure for '{name}'. "
            "Delete the fixture file and re-run to regenerate."
        )
    else:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.cpu(), path)
        pytest.skip(f"Fixture '{name}.pt' generated — re-run to verify.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_volume() -> torch.Tensor:
    """3D volume (16, 16, 16) with a box phantom."""
    volume = torch.zeros(16, 16, 16)
    volume[4:12, 4:12, 4:12] = 50.0
    return volume


@pytest.fixture
def full_ctf_params() -> dict[str, torch.Tensor]:
    """
    Complete set of supported per-particle CTF/aberration parameters.

    Non-zero values for tilt and trefoil so each contributes meaningfully
    to the transfer function.  Units follow the specter convention:
    defocus in Å, Cs in mm, tilt in rad, trefoil in Å³, B-factor in Å².
    """
    return {
        "dfu": torch.tensor([5000.0]),
        "dfv": torch.tensor([4800.0]),
        "dfang": torch.tensor([15.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.0]),
        "tiltx": torch.tensor([0.005]),
        "tilty": torch.tensor([-0.003]),
        "trefoil1": torch.tensor([200.0]),
        "trefoil2": torch.tensor([-150.0]),
        "bfactor": torch.tensor([60.0]),
    }


@pytest.fixture
def anisomag() -> torch.Tensor:
    """
    Anisotropic magnification matrix for one particle, shape (1, 2, 2).

    A 2 % stretch in x and 2 % compression in y relative to the identity.
    """
    mat = torch.eye(2).unsqueeze(0)  # (1, 2, 2)
    mat[0, 0, 0] = 1.02
    mat[0, 1, 1] = 0.98
    return mat


@pytest.fixture
def gb_kwargs(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> dict:
    """Shared Reconstructor constructor kwargs (no scattering_model — supplied per test)."""
    return dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.1,
    )


# ---------------------------------------------------------------------------
# Regression tests — one fixture per scattering model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scattering_model", SCATTERING_MODELS)
def test_ghostbuster_forward_regression(gb_kwargs: dict, scattering_model: str) -> None:
    """Reconstructor.forward with full CTF params regresses for each scattering model."""
    torch.manual_seed(0)
    gb = Reconstructor(**gb_kwargs, scattering_model=scattering_model)
    torch.manual_seed(0)
    images = gb.forward(torch.tensor([0]))
    _save_or_compare(f"ghostbuster_forward_{scattering_model}", images.cpu())


# ---------------------------------------------------------------------------
# Anisotropic magnification
# ---------------------------------------------------------------------------


def test_ghostbuster_stores_anisomag(gb_kwargs: dict, anisomag: torch.Tensor) -> None:
    """Reconstructorregisters anisomag as a buffer and passes it to the ImageGenerator."""
    gb = Reconstructor(**gb_kwargs, anisomag=anisomag, scattering_model="projection")
    assert gb.anisomag is not None
    assert gb.anisomag.shape == (1, 2, 2)
    assert torch.allclose(gb.anisomag, anisomag)
    assert gb.imagegenerator.anisomag is not None


def test_anisomag_forward_regression(
    gb_kwargs: dict,
    anisomag: torch.Tensor,
) -> None:
    """Reconstructorforward with anisotropic magnification and full CTF params."""
    torch.manual_seed(0)
    gb = Reconstructor(**gb_kwargs, anisomag=anisomag, scattering_model="projection")
    torch.manual_seed(0)
    images = gb.forward(torch.tensor([0]))
    _save_or_compare("ghostbuster_anisomag_projection", images.cpu())


def test_anisomag_changes_output(gb_kwargs: dict, anisomag: torch.Tensor) -> None:
    """Applying anisomag via Reconstructor produces a different image than no anisomag."""
    gb_iso = Reconstructor(**gb_kwargs, anisomag=None, scattering_model="projection")
    gb_aniso = Reconstructor(
        **gb_kwargs, anisomag=anisomag, scattering_model="projection"
    )
    img_iso = gb_iso.forward(torch.tensor([0]))
    img_aniso = gb_aniso.forward(torch.tensor([0]))
    assert not torch.equal(img_iso, img_aniso), (
        "Anisotropic magnification should produce a different image than isotropic."
    )


# ---------------------------------------------------------------------------
# Output shape and determinism
# ---------------------------------------------------------------------------


def test_ghostbuster_output_shape(gb_kwargs: dict) -> None:
    """forward returns (N, H, W) matching the volume spatial dims."""
    gb = Reconstructor(**gb_kwargs, scattering_model="projection")
    images = gb.forward(torch.tensor([0]))
    assert images.shape == (1, 16, 16)


def test_ghostbuster_forward_is_deterministic(gb_kwargs: dict) -> None:
    """Two identical forward calls without noise produce identical output."""
    gb = Reconstructor(**gb_kwargs, scattering_model="projection")
    img1 = gb.forward(torch.tensor([0]))
    img2 = gb.forward(torch.tensor([0]))
    assert torch.equal(img1, img2)


# ---------------------------------------------------------------------------
# CTF parameter sensitivity
# ---------------------------------------------------------------------------


def test_tilt_changes_output(small_volume: torch.Tensor) -> None:
    """Non-zero beam tilt changes the transfer function output."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.0,
        scattering_model="projection",
    )
    no_tilt = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    with_tilt = {
        "dfu": torch.tensor([5000.0]),
        "cs": torch.tensor([2.7]),
        "tiltx": torch.tensor([0.01]),
        "tilty": torch.tensor([0.01]),
    }
    img_no = Reconstructor(**base, ctf_params=no_tilt).forward(torch.tensor([0]))
    img_yes = Reconstructor(**base, ctf_params=with_tilt).forward(torch.tensor([0]))
    assert not torch.equal(img_no, img_yes)


def test_trefoil_changes_output(small_volume: torch.Tensor) -> None:
    """Non-zero trefoil changes the transfer function output."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.0,
        scattering_model="projection",
    )
    no_trefoil = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    with_trefoil = {
        "dfu": torch.tensor([5000.0]),
        "cs": torch.tensor([2.7]),
        "trefoil1": torch.tensor([300.0]),
        "trefoil2": torch.tensor([-200.0]),
    }
    img_no = Reconstructor(**base, ctf_params=no_trefoil).forward(torch.tensor([0]))
    img_yes = Reconstructor(**base, ctf_params=with_trefoil).forward(torch.tensor([0]))
    assert not torch.equal(img_no, img_yes)


def test_tetrafoil_changes_output(small_volume: torch.Tensor) -> None:
    """Non-zero tetrafoil changes the transfer function output."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.0,
        scattering_model="projection",
    )
    no_tetrafoil = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    with_tetrafoil = {
        "dfu": torch.tensor([5000.0]),
        "cs": torch.tensor([2.7]),
        "tetrafoil1": torch.tensor([100.0]),
        "tetrafoil2": torch.tensor([-80.0]),
        "tetrafoil3": torch.tensor([60.0]),
        "tetrafoil4": torch.tensor([-40.0]),
    }
    img_no = Reconstructor(**base, ctf_params=no_tetrafoil).forward(torch.tensor([0]))
    img_yes = Reconstructor(**base, ctf_params=with_tetrafoil).forward(
        torch.tensor([0])
    )
    assert not torch.equal(img_no, img_yes)
    assert torch.isfinite(img_yes).all()


def test_bfactor_damps_ghostbuster_output(small_volume: torch.Tensor) -> None:
    """B-factor envelope reduces high-frequency power in simulated images."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.0,
        scattering_model="projection",
    )
    params_no_b = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    params_b = {
        "dfu": torch.tensor([5000.0]),
        "cs": torch.tensor([2.7]),
        "bfactor": torch.tensor([200.0]),
    }
    img_no_b = Reconstructor(**base, ctf_params=params_no_b).forward(torch.tensor([0]))
    img_b = Reconstructor(**base, ctf_params=params_b).forward(torch.tensor([0]))
    # High B-factor should damp high-frequency content → lower variance
    assert img_b.var() < img_no_b.var()


def test_bfactor_kwarg_matches_ctf_params_bfactor(small_volume: torch.Tensor) -> None:
    """The bfactor kwarg produces the same result as an equal ctf_params["bfactor"]."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.0,
        scattering_model="projection",
    )
    params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    img_via_ctf_params = Reconstructor(
        **base, ctf_params={**params, "bfactor": torch.tensor([200.0])}
    ).forward(torch.tensor([0]))
    img_via_kwarg = Reconstructor(**base, ctf_params=params, bfactor=200.0).forward(
        torch.tensor([0])
    )
    assert torch.allclose(img_via_kwarg, img_via_ctf_params)


def test_bfactor_kwarg_overrides_ctf_params_bfactor(small_volume: torch.Tensor) -> None:
    """An explicit bfactor kwarg replaces (not adds to) ctf_params["bfactor"]."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.0,
        scattering_model="projection",
    )
    params = {
        "dfu": torch.tensor([5000.0]),
        "cs": torch.tensor([2.7]),
        "bfactor": torch.tensor([200.0]),
    }
    img_overridden_to_zero = Reconstructor(
        **base, ctf_params=params, bfactor=0.0
    ).forward(torch.tensor([0]))
    img_no_bfactor = Reconstructor(
        **base, ctf_params={k: v for k, v in params.items() if k != "bfactor"}
    ).forward(torch.tensor([0]))
    assert torch.allclose(img_overridden_to_zero, img_no_bfactor)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def test_reconstructor_training_updates_volume(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """One gradient step with C2 symmetry updates V from its initial value."""
    torch.manual_seed(42)
    n_particles = 4
    n = small_volume.shape[-1]

    ctf = {k: v.repeat(n_particles) for k, v in full_ctf_params.items()}
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_particles, 1)
    translations = torch.zeros(n_particles, 2)
    images = torch.randn(n_particles, n, n)

    V_init = small_volume.clone()
    model = Reconstructor(
        V=small_volume.clone(),
        voxel_size=2.0,
        quaternions=quaternions,
        translations=translations,
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        symmetry="C2",
    )

    opt = torch.optim.AdamW([model.V], lr=0.1)
    batch = (images, torch.arange(n_particles))
    loss, _, _ = model._common_step(batch, 0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert not torch.equal(model.V.data, V_init), (
        "V was not updated after one gradient step"
    )


# ---------------------------------------------------------------------------
# Per-particle scale weighting
# ---------------------------------------------------------------------------


def test_scale_defaults_to_ones(gb_kwargs: dict) -> None:
    """Without an explicit scale, every particle is weighted equally."""
    gb = Reconstructor(**gb_kwargs, scattering_model="projection")
    assert torch.equal(gb.scale, torch.ones(1))


def test_scale_is_registered_as_buffer(gb_kwargs: dict) -> None:
    """An explicit per-particle scale is stored and used verbatim."""
    scale = torch.tensor([1.5])
    gb = Reconstructor(**gb_kwargs, scale=scale, scattering_model="projection")
    assert torch.allclose(gb.scale, scale)


def test_scale_scales_mse_loss_linearly(gb_kwargs: dict) -> None:
    """Doubling every particle's scale doubles the plain-MSE loss."""
    n_particles = 3
    n = gb_kwargs["V"].shape[-1]
    images = torch.randn(n_particles, n, n)
    out = torch.randn(n_particles, n, n)
    idx = torch.arange(n_particles)

    base = dict(gb_kwargs)
    base.update(
        quaternions=base["quaternions"].repeat(n_particles, 1),
        translations=base["translations"].repeat(n_particles, 1),
        ctf_params={k: v.repeat(n_particles) for k, v in base["ctf_params"].items()},
        scattering_model="projection",
    )

    gb_unit = Reconstructor(**base, scale=torch.ones(n_particles))
    gb_double = Reconstructor(**base, scale=torch.full((n_particles,), 2.0))

    loss_unit = gb_unit._compute_loss(out, images, idx)
    loss_double = gb_double._compute_loss(out, images, idx)
    assert torch.allclose(loss_double, 2.0 * loss_unit, rtol=1e-4)


def test_small_scale_reduces_gradient_contribution(gb_kwargs: dict) -> None:
    """A particle with a small scale contributes less to the volume gradient."""
    n_particles = 2
    n = gb_kwargs["V"].shape[-1]
    torch.manual_seed(0)
    images = torch.randn(n_particles, n, n)

    base = dict(gb_kwargs)
    base.update(
        quaternions=base["quaternions"].repeat(n_particles, 1),
        translations=base["translations"].repeat(n_particles, 1),
        ctf_params={k: v.repeat(n_particles) for k, v in base["ctf_params"].items()},
        scattering_model="projection",
        lr=0.1,
    )
    idx = torch.arange(n_particles)

    def volume_grad_norm(scale: torch.Tensor) -> torch.Tensor:
        model = Reconstructor(**{**base, "V": base["V"].clone()}, scale=scale)
        out = model.forward(idx)
        loss = model._compute_loss(out, images, idx)
        (grad,) = torch.autograd.grad(loss, model.V)
        return grad.norm()

    grad_uniform = volume_grad_norm(torch.tensor([1.0, 1.0]))
    grad_downweighted = volume_grad_norm(torch.tensor([1.0, 0.01]))
    assert grad_downweighted < grad_uniform


# ---------------------------------------------------------------------------
# Optimizer / scheduler wiring
# ---------------------------------------------------------------------------


def _fit_one_epoch(
    model: Reconstructor, images: torch.Tensor, max_epochs: int = 1
) -> None:
    idx = torch.arange(len(images))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, idx), batch_size=len(images)
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


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_configure_optimizers_all_schedulers(
    small_volume: torch.Tensor,
    full_ctf_params: dict[str, torch.Tensor],
    scheduler: str,
) -> None:
    """Each supported scheduler string trains for one epoch without error."""
    n_particles = 2
    ctf = {k: v.repeat(n_particles) for k, v in full_ctf_params.items()}
    images = torch.randn(n_particles, *small_volume.shape[-2:])

    model = Reconstructor(
        V=small_volume.clone(),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_particles, 1),
        translations=torch.zeros(n_particles, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        scheduler=scheduler,
    )
    V_init = model.V.data.clone()

    _fit_one_epoch(model, images)

    assert not torch.equal(model.V.data, V_init)
    assert len(model.log_lrs) == 1


def test_configure_optimizers_unknown_scheduler_raises(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """An unrecognised scheduler string raises ValueError from configure_optimizers."""
    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        scheduler="BogusScheduler",
    )
    with pytest.raises(ValueError, match="Unknown scheduler"):
        model.configure_optimizers()


# ---------------------------------------------------------------------------
# Fourier k-mask
# ---------------------------------------------------------------------------


def test_kmask_zeros_high_frequencies(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """on_train_batch_end zeros Fourier components outside the k-mask, in-place."""
    torch.manual_seed(0)
    n = small_volume.shape[-1]
    model = Reconstructor(
        V=torch.randn(n, n, n),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        kmask=ball3d(n, n // 2),
    )
    model.on_train_batch_end(None, None, 0)
    spectrum = fft3(model.V.data, shift=True)
    outside_mask = spectrum[model.kmask == 0]
    assert torch.allclose(outside_mask, torch.zeros_like(outside_mask), atol=1e-5)


# ---------------------------------------------------------------------------
# run_dir file output and symmetry enforcement
# ---------------------------------------------------------------------------


def test_run_dir_writes_expected_artifacts(
    tmp_path: Path,
    small_volume: torch.Tensor,
    full_ctf_params: dict[str, torch.Tensor],
) -> None:
    """A configured run_dir receives params/metrics/volumes/previews across epochs."""
    n_particles = 2
    n = small_volume.shape[-1]
    ctf = {k: v.repeat(n_particles) for k, v in full_ctf_params.items()}
    images = torch.randn(n_particles, n, n)
    fsc_ref = torch.rand(n, n, n)
    cryosparc_ref = torch.rand(n, n, n)

    model = Reconstructor(
        V=small_volume.clone(),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_particles, 1),
        translations=torch.zeros(n_particles, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        symmetry="C2",
        kmask=ball3d(n, n),
        fsc_ref=fsc_ref,
        cryosparc_ref=cryosparc_ref,
        run_dir=tmp_path,
        halfset_label="A",
    )
    _fit_one_epoch(model, images, max_epochs=2)

    # No params_A.json/metrics_A.json of their own: a caller (here, the
    # pipeline layer) folds results_summary() into one job.json instead, so
    # two halfset workers never need to share a file while training.
    assert not list(tmp_path.glob("*.json"))
    assert (tmp_path / "volume_A.mrc").exists()
    assert (tmp_path / "kmask.pt").exists()
    assert (tmp_path / "fsc_A.png").exists()
    assert (tmp_path / "epochs" / "001_A.mrc").exists()
    assert (tmp_path / "epochs" / "002_A.mrc").exists()
    assert (tmp_path / "epochs" / "volume_001_A.png").exists()
    assert (tmp_path / "epochs" / "fsc_001_A.png").exists()

    summary = model.results_summary()
    assert "saved_arrays" in summary  # from on_fit_start's hparams
    metrics = summary["metrics"]
    assert metrics["total_batches"] == 2  # 1 batch/epoch x 2 epochs
    assert {"epoch_01", "epoch_02"} <= set(metrics["epochs"])


def test_symmetrize_applies_configured_symmetry(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """symmetrize() calls apply_symmetry with the constructor's symmetry settings."""
    torch.manual_seed(0)
    n = small_volume.shape[-1]
    model = Reconstructor(
        V=torch.randn(n, n, n),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        symmetry="C2",
        symmetry_mode="real",
        symmetry_batchsize=4,
    )
    v_before = model.V.data.clone()
    model.symmetrize()
    expected = apply_symmetry(
        v_before, get_rotation_matrices("C2"), batchsize=4, method="real"
    )
    assert torch.allclose(model.V.data, expected)


def test_on_train_epoch_end_applies_symmetry(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """on_train_epoch_end applies apply_symmetry to V (run_dir=None, so that's
    the only effect it has) using the constructor's symmetry settings."""
    torch.manual_seed(0)
    n = small_volume.shape[-1]
    model = Reconstructor(
        V=torch.randn(n, n, n),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        symmetry="C2",
        symmetry_mode="real",
        symmetry_batchsize=4,
    )
    v_before = model.V.data.clone()
    model.on_train_epoch_end()
    expected = apply_symmetry(
        v_before, get_rotation_matrices("C2"), batchsize=4, method="real"
    )
    assert torch.allclose(model.V.data, expected)


# ---------------------------------------------------------------------------
# FSC / CryoSPARC reference loading
# ---------------------------------------------------------------------------


def test_fsc_ref_and_mask_accept_tensor(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """fsc_ref and fsc_mask tensors are stored verbatim."""
    n = small_volume.shape[-1]
    ref = torch.rand(n, n, n)
    mask = torch.ones(n, n, n)
    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        fsc_ref=ref,
        fsc_mask=mask,
    )
    assert torch.equal(model.fsc_ref, ref)
    assert torch.equal(model.fsc_mask, mask)


def test_fsc_ref_and_cryosparc_ref_load_from_file(
    tmp_path: Path,
    small_volume: torch.Tensor,
    full_ctf_params: dict[str, torch.Tensor],
) -> None:
    """fsc_ref and cryosparc_ref accept a path to an .mrc file and load it."""
    n = small_volume.shape[-1]
    ref = torch.rand(n, n, n).numpy().astype("float32")
    cs_ref = torch.rand(n, n, n).numpy().astype("float32")
    ref_path = tmp_path / "ref.mrc"
    cs_path = tmp_path / "cs_ref.mrc"
    with mrcfile.new(str(ref_path), overwrite=True) as mrc:
        mrc.set_data(ref)
    with mrcfile.new(str(cs_path), overwrite=True) as mrc:
        mrc.set_data(cs_ref)

    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        fsc_ref=ref_path,
        cryosparc_ref=cs_path,
    )
    assert torch.allclose(model.fsc_ref, torch.as_tensor(ref))
    assert torch.allclose(model.cryosparc_ref, torch.as_tensor(cs_ref))


# ---------------------------------------------------------------------------
# 2D-projected FSC mask weighting
# ---------------------------------------------------------------------------


def test_use_2d_mask_weights_loss(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """use_2d_mask=True projects fsc_mask and produces a finite weighted loss."""
    n = small_volume.shape[-1]
    mask = torch.zeros(n, n, n)
    mask[4:12, 4:12, 4:12] = 1.0
    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        fsc_mask=mask,
        use_2d_mask=True,
    )
    out = model.forward(torch.tensor([0]))
    images = torch.randn_like(out)
    loss = model._compute_loss(out, images, torch.tensor([0]))
    assert torch.isfinite(loss)


def test_use_2d_mask_requires_tensor_mask(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """use_2d_mask=True without a 3D tensor fsc_mask raises ValueError."""
    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        use_2d_mask=True,
    )
    out = model.forward(torch.tensor([0]))
    images = torch.randn_like(out)
    with pytest.raises(ValueError, match="requires fsc_mask"):
        model._compute_loss(out, images, torch.tensor([0]))


# ---------------------------------------------------------------------------
# Alternate loss models (NCC / learned noise / NPS-weighted)
# ---------------------------------------------------------------------------


def test_compute_loss_ncc_branch_differs_from_mse(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """use_ncc=True routes _compute_loss through the NCC branch."""
    torch.manual_seed(0)
    n = small_volume.shape[-1]
    out = torch.randn(2, n, n)
    images = torch.randn(2, n, n)
    idx = torch.arange(2)
    ctf = {k: v.repeat(2) for k, v in full_ctf_params.items()}
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        translations=torch.zeros(2, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
    )
    loss_mse = Reconstructor(**base)._compute_loss(out, images, idx)
    loss_ncc = Reconstructor(**base, use_ncc=True)._compute_loss(out, images, idx)
    assert torch.isfinite(loss_ncc)
    assert not torch.allclose(loss_mse, loss_ncc)


def test_compute_loss_learn_noise_model_updates_sigma2(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """learn_noise_model=True updates sigma2_k via an EMA over residual power."""
    torch.manual_seed(0)
    n = small_volume.shape[-1]
    out = torch.randn(2, n, n)
    images = torch.randn(2, n, n)
    idx = torch.arange(2)
    ctf = {k: v.repeat(2) for k, v in full_ctf_params.items()}
    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        translations=torch.zeros(2, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
        learn_noise_model=True,
    )
    sigma2_before = model.sigma2_k.clone()
    loss = model._compute_loss(out, images, idx)
    assert torch.isfinite(loss)
    assert not torch.allclose(model.sigma2_k, sigma2_before)


def test_compute_loss_nps_weight_branch_differs_from_mse(
    small_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """A non-trivial nps_weight routes _compute_loss through the NPS-weighted branch."""
    torch.manual_seed(0)
    n = small_volume.shape[-1]
    out = torch.randn(2, n, n)
    images = torch.randn(2, n, n)
    idx = torch.arange(2)
    ctf = {k: v.repeat(2) for k, v in full_ctf_params.items()}
    nps = torch.rand(n, n // 2 + 1) + 0.5
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        translations=torch.zeros(2, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
    )
    loss_mse = Reconstructor(**base)._compute_loss(out, images, idx)
    loss_nps = Reconstructor(**base, nps_weight=nps)._compute_loss(out, images, idx)
    assert torch.isfinite(loss_nps)
    assert not torch.allclose(loss_mse, loss_nps)
