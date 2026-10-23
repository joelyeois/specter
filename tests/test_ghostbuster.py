"""
Golden-output regression tests for Reconstructor.

Covers all scattering models and the full aberration parameter set:
defocus, spherical aberration, phase shift, beam tilt (tiltx/tilty),
trefoil (trefoil1/trefoil2), B-factor envelope, and anisotropic
magnification.

Outputs are compared against committed tensors in tests/test_data/.
Missing references fail and must be regenerated explicitly after reviewing
an intentional output change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mrcfile
import pytest
import torch
import torch.utils.data

from specter.arrays import ball3d
from specter.fft import fft3
from specter.ghostbuster import Reconstructor
from specter.symmetries import apply_symmetry, get_rotation_matrices
from specter.settings import Propagation
from conftest import fit_one_epoch

SCATTERING_MODELS = ["multislice", "firstborn", "projection", "ctf", "rytov"]
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
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> dict:
    """Shared Reconstructor constructor kwargs (no scattering_model — supplied per test)."""
    return dict(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
    )


# ---------------------------------------------------------------------------
# Regression tests — one fixture per scattering model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scattering_model", SCATTERING_MODELS)
def test_ghostbuster_forward_regression(
    gb_kwargs: dict, scattering_model: str, save_or_compare
) -> None:
    """Reconstructor.forward with full CTF params regresses for each scattering model."""
    torch.manual_seed(0)
    gb = Reconstructor(
        **gb_kwargs,
        propagation=Propagation(alpha=0.1, scattering_model=scattering_model),
    )
    torch.manual_seed(0)
    images = gb.forward(torch.tensor([0]))
    save_or_compare(f"ghostbuster_forward_{scattering_model}", images.cpu())


# ---------------------------------------------------------------------------
# Anisotropic magnification
# ---------------------------------------------------------------------------


def test_anisomag_forward_regression(
    gb_kwargs: dict,
    anisomag: torch.Tensor,
    save_or_compare,
) -> None:
    """Reconstructor.forward with anisotropic magnification and full CTF params."""
    torch.manual_seed(0)
    gb = Reconstructor(
        **gb_kwargs,
        anisomag=anisomag,
        propagation=Propagation(alpha=0.1, scattering_model="projection"),
    )
    torch.manual_seed(0)
    images = gb.forward(torch.tensor([0]))
    save_or_compare("ghostbuster_anisomag_projection", images.cpu())


def test_anisomag_changes_output(gb_kwargs: dict, anisomag: torch.Tensor) -> None:
    """Applying anisomag via Reconstructor produces a different image than no anisomag."""
    gb_iso = Reconstructor(
        **gb_kwargs,
        anisomag=None,
        propagation=Propagation(alpha=0.1, scattering_model="projection"),
    )
    gb_aniso = Reconstructor(
        **gb_kwargs,
        anisomag=anisomag,
        propagation=Propagation(alpha=0.1, scattering_model="projection"),
    )
    img_iso = gb_iso.forward(torch.tensor([0]))
    img_aniso = gb_aniso.forward(torch.tensor([0]))
    assert not torch.equal(img_iso, img_aniso), (
        "Anisotropic magnification should produce a different image than isotropic."
    )


# ---------------------------------------------------------------------------
# CTF parameter sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_ctf_params",
    [
        {"tiltx": [0.01], "tilty": [0.01]},
        {"trefoil1": [300.0], "trefoil2": [-200.0]},
        {
            "tetrafoil1": [100.0],
            "tetrafoil2": [-80.0],
            "tetrafoil3": [60.0],
            "tetrafoil4": [-40.0],
        },
    ],
    ids=["beam_tilt", "trefoil", "tetrafoil"],
)
def test_higher_order_aberration_changes_output(
    tiny_volume: torch.Tensor, extra_ctf_params: dict[str, list[float]]
) -> None:
    """Non-zero beam tilt, trefoil or tetrafoil changes the transfer function
    output."""
    base = dict(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(alpha=0.0, scattering_model="projection"),
    )
    without = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    with_term = {
        **without,
        **{k: torch.tensor(v) for k, v in extra_ctf_params.items()},
    }
    img_no = Reconstructor(**base, ctf_params=without).forward(torch.tensor([0]))
    img_yes = Reconstructor(**base, ctf_params=with_term).forward(torch.tensor([0]))
    assert not torch.equal(img_no, img_yes)
    assert torch.isfinite(img_yes).all()


def test_bfactor_damps_ghostbuster_output(tiny_volume: torch.Tensor) -> None:
    """B-factor envelope reduces high-frequency power in simulated images."""
    base = dict(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(alpha=0.0, scattering_model="projection"),
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


def test_bfactor_kwarg_matches_ctf_params_bfactor(tiny_volume: torch.Tensor) -> None:
    """The bfactor kwarg produces the same result as an equal ctf_params["bfactor"]."""
    base = dict(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(alpha=0.0, scattering_model="projection"),
    )
    params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    img_via_ctf_params = Reconstructor(
        **base, ctf_params={**params, "bfactor": torch.tensor([200.0])}
    ).forward(torch.tensor([0]))
    img_via_kwarg = Reconstructor(**base, ctf_params=params, bfactor=200.0).forward(
        torch.tensor([0])
    )
    assert torch.allclose(img_via_kwarg, img_via_ctf_params)


def test_bfactor_kwarg_overrides_ctf_params_bfactor(tiny_volume: torch.Tensor) -> None:
    """An explicit bfactor kwarg replaces (not adds to) ctf_params["bfactor"]."""
    base = dict(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(alpha=0.0, scattering_model="projection"),
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
# Per-particle scale: an attenuation of the predicted signal
# ---------------------------------------------------------------------------


def _scaled_model(
    gb_kwargs: dict, scale: list[float], **kwargs: object
) -> Reconstructor:
    n = len(scale)
    base = {**gb_kwargs, "V": gb_kwargs["V"].clone()}
    base["quaternions"] = base["quaternions"].repeat(n, 1)
    base["translations"] = base["translations"].repeat(n, 1)
    base["ctf_params"] = {k: v.repeat(n) for k, v in base["ctf_params"].items()}
    return Reconstructor(
        **base,
        lr=0.1,
        scale=torch.tensor(scale),
        propagation=Propagation(scattering_model="projection"),
        **kwargs,
    )


def test_scale_multiplies_the_signal_about_the_flat_level(gb_kwargs: dict) -> None:
    """The scale attenuates contrast, not the dose: a prediction becomes
    N0 + s (f - N0), N0 = dose * voxel_size^2, and every particle counts once."""
    model = _scaled_model(gb_kwargs, [0.5, 1.5])
    idx = torch.tensor([0, 1])
    out = model.forward(idx).detach()
    images = out + torch.randn_like(out)
    level = model.dose_per_angstrom * model.voxel_size**2
    expected = level + torch.tensor([0.5, 1.5])[:, None, None] * (out - level)
    loss = model._compute_loss(out, images, idx)
    assert torch.allclose(loss, ((expected - images) ** 2).mean(), rtol=1e-5)


def test_zero_scale_gives_no_gradient(gb_kwargs: dict) -> None:
    model = _scaled_model(gb_kwargs, [0.0, 1.0])
    idx = torch.tensor([0])
    out = model.forward(idx)
    images = out.detach() + torch.randn_like(out)
    loss = model._compute_loss(out, images, idx)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.V.grad is not None and torch.count_nonzero(model.V.grad) == 0


def test_small_scale_reduces_gradient_contribution(gb_kwargs: dict) -> None:
    """A weakly scaled particle pulls on the volume weakly."""
    torch.manual_seed(0)
    n = gb_kwargs["V"].shape[-1]
    images = torch.randn(2, n, n)
    idx = torch.arange(2)

    def volume_grad_norm(scale: list[float]) -> torch.Tensor:
        model = _scaled_model(gb_kwargs, scale)
        loss = model._compute_loss(model.forward(idx), images, idx)
        (grad,) = torch.autograd.grad(loss, model.V)
        return grad.norm()

    assert volume_grad_norm([1.0, 0.01]) < volume_grad_norm([1.0, 1.0])


# ---------------------------------------------------------------------------
# Optimizer / scheduler wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_configure_optimizers_all_schedulers(
    tiny_volume: torch.Tensor,
    full_ctf_params: dict[str, torch.Tensor],
    scheduler: str,
) -> None:
    """Each supported scheduler string trains for one epoch without error."""
    n_particles = 2
    ctf = {k: v.repeat(n_particles) for k, v in full_ctf_params.items()}
    images = torch.randn(n_particles, *tiny_volume.shape[-2:])

    model = Reconstructor(
        V=tiny_volume.clone(),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_particles, 1),
        translations=torch.zeros(n_particles, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        lr=0.1,
        scheduler=scheduler,
        propagation=Propagation(scattering_model="projection"),
    )
    V_init = model.V.data.clone()

    fit_one_epoch(model, images)

    assert not torch.equal(model.V.data, V_init)
    assert len(model.log_lrs) == 1


def test_configure_optimizers_unknown_scheduler_raises(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """An unrecognised scheduler string raises ValueError from configure_optimizers."""
    model = Reconstructor(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        lr=0.1,
        scheduler="BogusScheduler",
        propagation=Propagation(scattering_model="projection"),
    )
    with pytest.raises(ValueError, match="Unknown scheduler"):
        model.configure_optimizers()


# ---------------------------------------------------------------------------
# Fourier k-mask
# ---------------------------------------------------------------------------


def test_kmask_zeros_high_frequencies(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """on_train_batch_end zeros Fourier components outside the k-mask, in-place."""
    torch.manual_seed(0)
    n = tiny_volume.shape[-1]
    model = Reconstructor(
        V=torch.randn(n, n, n),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        lr=0.1,
        kmask=ball3d(n, n // 2),
        propagation=Propagation(scattering_model="projection"),
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
    tiny_volume: torch.Tensor,
    full_ctf_params: dict[str, torch.Tensor],
) -> None:
    """A configured run_dir receives params/metrics/volumes/previews across epochs."""
    n_particles = 2
    n = tiny_volume.shape[-1]
    ctf = {k: v.repeat(n_particles) for k, v in full_ctf_params.items()}
    images = torch.randn(n_particles, n, n)
    fsc_ref = torch.rand(n, n, n)
    cryosparc_ref = torch.rand(n, n, n)

    model = Reconstructor(
        V=tiny_volume.clone(),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_particles, 1),
        translations=torch.zeros(n_particles, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        lr=0.1,
        symmetry="C2",
        kmask=ball3d(n, n),
        fsc_ref=fsc_ref,
        cryosparc_ref=cryosparc_ref,
        run_dir=tmp_path,
        halfset_label="A",
        propagation=Propagation(scattering_model="projection"),
    )
    fit_one_epoch(model, images, max_epochs=2)

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


def test_on_train_epoch_end_applies_symmetry(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """on_train_epoch_end applies apply_symmetry to V (run_dir=None, so that's
    the only effect it has) using the constructor's symmetry settings."""
    torch.manual_seed(0)
    n = tiny_volume.shape[-1]
    model = Reconstructor(
        V=torch.randn(n, n, n),
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        symmetry="C2",
        symmetry_mode="real",
        symmetry_batchsize=4,
        propagation=Propagation(scattering_model="projection"),
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


def test_fsc_ref_and_cryosparc_ref_load_from_file(
    tmp_path: Path,
    tiny_volume: torch.Tensor,
    full_ctf_params: dict[str, torch.Tensor],
) -> None:
    """fsc_ref and cryosparc_ref accept a path to an .mrc file and load it."""
    n = tiny_volume.shape[-1]
    ref = torch.rand(n, n, n).numpy().astype("float32")
    cs_ref = torch.rand(n, n, n).numpy().astype("float32")
    ref_path = tmp_path / "ref.mrc"
    cs_path = tmp_path / "cs_ref.mrc"
    with mrcfile.new(str(ref_path), overwrite=True) as mrc:
        mrc.set_data(ref)
    with mrcfile.new(str(cs_path), overwrite=True) as mrc:
        mrc.set_data(cs_ref)

    model = Reconstructor(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.zeros(1, 2),
        ctf_params=full_ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        fsc_ref=ref_path,
        cryosparc_ref=cs_path,
        propagation=Propagation(scattering_model="projection"),
    )
    assert torch.allclose(model.fsc_ref, torch.as_tensor(ref))
    assert torch.allclose(model.cryosparc_ref, torch.as_tensor(cs_ref))


# ---------------------------------------------------------------------------
# Refined-parameter binding
#
# `Reconstructor` owns the refined parameters and renders from them. The
# handoff used to rely on the renderer and the reconstructor holding the same
# tensor object, which `torch.nn.Module._apply` is free to break: it swaps a
# Parameter's `.data` in place but replaces a buffer with a fresh `.to()`
# copy. On a GPU that left the renderer using values frozen at device-move
# time.
# ---------------------------------------------------------------------------


def _binding_kwargs(tiny_volume: torch.Tensor, ctf: dict) -> dict:
    return dict(
        V=tiny_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(alpha=0.1, scattering_model="multislice"),
        lr=1e-2,
        lr_R=1e-3,
        lr_T=1e-3,
        lr_D=1e-3,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_refined_parameters_are_live_in_the_forward_model(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor], device: str
) -> None:
    """An in-place step on each refined parameter, after a device move, changes
    the rendered image: the forward model reads the module's own parameters."""
    gb = Reconstructor(**_binding_kwargs(tiny_volume, full_ctf_params)).to(device)
    idx = torch.tensor([0], device=device)
    steps = {
        "translations": torch.tensor([[5.0, 5.0]], device=device),
        "rotations": torch.tensor([[0.0, 0.1, 0.0, 0.0]], device=device),
        "V": torch.tensor(1.0, device=device),
        "defocus_offset": torch.tensor(500.0, device=device),
    }
    for name, step in steps.items():
        with torch.no_grad():
            before = gb(idx)
            getattr(gb, name).add_(step)
            after = gb(idx)
        assert not torch.allclose(before, after), f"{name} did not reach the image"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_every_refined_parameter_receives_a_gradient(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor], device: str
) -> None:
    """lr/lr_R/lr_T/lr_D each put a live leaf in the autograd graph."""
    gb = Reconstructor(**_binding_kwargs(tiny_volume, full_ctf_params)).to(device)
    n = tiny_volume.shape[-1]
    idx = torch.tensor([0], device=device)
    observed = torch.rand(1, n, n, device=device)

    loss, _, _ = gb._common_step((observed, idx), 0)
    loss.backward()

    for name in ("V", "rotations", "translations", "defocus_offset"):
        grad = getattr(gb, name).grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{name} gradient is identically zero"


@pytest.mark.parametrize("lr_D", [None, 1e-3], ids=["fixed", "refined"])
def test_default_defocus_offset_is_independent_between_reconstructors(
    gb_kwargs: dict, lr_D: float | None
) -> None:
    kwargs = dict(
        gb_kwargs, lr_D=lr_D, propagation=Propagation(scattering_model="projection")
    )
    first = Reconstructor(**kwargs)
    sibling = Reconstructor(**kwargs)
    with torch.no_grad():
        first.defocus_offset.add_(5000.0)
    later = Reconstructor(**kwargs)
    for model in (sibling, later):
        torch.testing.assert_close(model.defocus_offset, torch.tensor(0.0))


def test_defocus_offset_changes_the_rendered_image(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """defocus_offset is re-applied per forward, not folded in once at construction."""
    gb = Reconstructor(**_binding_kwargs(tiny_volume, full_ctf_params))
    idx = torch.tensor([0])

    torch.manual_seed(0)
    before = gb.forward(idx)
    with torch.no_grad():
        gb.defocus_offset.add_(5000.0)
    torch.manual_seed(0)
    after = gb.forward(idx)

    assert not torch.allclose(before, after)


def test_a_zero_offset_reproduces_the_constructed_defocus(
    tiny_volume: torch.Tensor, full_ctf_params: dict[str, torch.Tensor]
) -> None:
    """The batch's CTF carries the defocus the model was built with."""
    gb = Reconstructor(**_binding_kwargs(tiny_volume, full_ctf_params))
    ctf = gb._ctf(torch.tensor([0]))
    for name in ("dfu", "dfv"):
        assert torch.equal(ctf.params[name], full_ctf_params[name][[0]])


# ---------------------------------------------------------------------------
# k-space mask
# ---------------------------------------------------------------------------


def test_kmask_half_spectrum_matches_the_shifted_round_trip():
    """The per-step band limit equals the old centred complex round trip."""
    from specter.arrays import ball3d
    from specter.fft import fft3, ifft3
    from specter.ghostbuster._helpers import _apply_kmask_inplace, _kmask_half_spectrum

    torch.manual_seed(0)
    n = 24
    kmask = ball3d(n, n // 3)
    V = torch.randn(n, n, n)
    old = torch.real(ifft3(fft3(V, shift=True) * kmask, shift=True))
    new = V.clone()
    _apply_kmask_inplace(new, _kmask_half_spectrum(kmask))
    assert torch.allclose(new, old, rtol=1e-5, atol=1e-6)
    # And it is a band limit: the masked-out shell is empty afterwards.
    spectrum = fft3(new, shift=True)
    assert torch.allclose(
        spectrum[kmask == 0],
        torch.zeros(int((kmask == 0).sum()), dtype=spectrum.dtype),
        atol=1e-4,
    )


def _halfmap_recorder(run_dir: Path) -> Any:
    """The attributes `Reconstructor._record_halfmap_resolutions` reads, with
    halfset B finishing an epoch second, after A's volume is already on disk."""
    from types import SimpleNamespace

    (run_dir / "epochs").mkdir(parents=True)
    torch.manual_seed(0)
    for label in ("A", "B"):
        with mrcfile.new(str(run_dir / "epochs" / f"001_{label}.mrc")) as mrc:
            mrc.set_data(torch.randn(16, 16, 16).numpy())
    return SimpleNamespace(
        _run_dir=run_dir,
        _halfset_label="B",
        voxel_size=1.0,
        fsc_mask=1,
        _epoch_resolutions=[],
        _mask_for=lambda volume: None,
    )


def test_halfmap_figure_is_written_and_recorded(tmp_path: Path) -> None:
    """The second worker to finish an epoch writes a non-empty figure and
    records the gold-standard resolution. The claim file is created empty, so
    checking only that the PNG exists cannot tell the two apart."""
    recorder = _halfmap_recorder(tmp_path)

    Reconstructor._record_halfmap_resolutions(recorder, 1)

    figure = tmp_path / "epochs" / "fsc_halfmap_001.png"
    assert figure.stat().st_size > 0
    assert recorder._epoch_resolutions[0]["resolution_gold_standard"]


def test_failed_halfmap_figure_drops_its_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`save_halfmap_fsc_figure` swallows its own errors and returns None. That
    must drop the empty claim file and record nothing, not leave a 0-byte PNG
    behind next to a resolution that no figure shows."""
    from specter.ghostbuster import _reconstructor

    monkeypatch.setattr(
        _reconstructor, "save_halfmap_fsc_figure", lambda *args, **kwargs: None
    )
    recorder = _halfmap_recorder(tmp_path)

    Reconstructor._record_halfmap_resolutions(recorder, 1)

    assert not (tmp_path / "epochs" / "fsc_halfmap_001.png").exists()
    assert recorder._epoch_resolutions == []
