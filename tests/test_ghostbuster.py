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

import pytest
import torch

from specter.ghostbuster import Reconstructor

FIXTURE_DIR = Path(__file__).parent / "test_data"

SCATTERING_MODELS = ["multislice", "firstborn", "projection", "ctf", "rytov"]


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
    vol = torch.zeros(16, 16, 16)
    vol[4:12, 4:12, 4:12] = 50.0
    return vol


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
        energy=300.0,
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
    assert not torch.equal(
        img_iso, img_aniso
    ), "Anisotropic magnification should produce a different image than isotropic."


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
        energy=300.0,
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
        energy=300.0,
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


def test_bfactor_damps_ghostbuster_output(small_volume: torch.Tensor) -> None:
    """B-factor envelope reduces high-frequency power in simulated images."""
    base = dict(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        energy=300.0,
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
        energy=300.0,
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
        energy=300.0,
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
# Documented limitations
# ---------------------------------------------------------------------------


def test_tetrafoil_not_implemented(small_volume: torch.Tensor) -> None:
    """Tetrafoil keys in ctf_params cause TypeError because _tetrafoil returns None."""
    gb = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params={
            "dfu": torch.tensor([5000.0]),
            "tetrafoil1": torch.tensor([100.0]),
            "tetrafoil2": torch.tensor([0.0]),
            "tetrafoil3": torch.tensor([0.0]),
            "tetrafoil4": torch.tensor([0.0]),
        },
        energy=300.0,
        dose_per_angstrom=2.0,
        scattering_model="projection",
    )
    with pytest.raises(TypeError):
        gb.forward(torch.tensor([0]))


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
        energy=300.0,
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

    assert not torch.equal(
        model.V.data, V_init
    ), "V was not updated after one gradient step"
