"""
Tests for TomogramReconstructor: forward simulation, FOV masking, k-mask
application, optimizer/scheduler wiring, and run_dir file output.

Companion to test_ghostbuster.py, which covers Reconstructor. There were no
tests at all for TomogramReconstructor prior to this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import roma
import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# The delocalisation band
# ---------------------------------------------------------------------------


def _band_model(
    nxy: int,
    voxel_size: float,
    defocus: float,
    *,
    cs: float = 0.0,
    angles: tuple[float, ...] = (0.0,),
    model: str = "multislice",
    nz: int = 16,
) -> TomogramReconstructor:
    theta = torch.deg2rad(torch.tensor(angles))
    zeros = torch.zeros_like(theta)
    quaternions = roma.rotvec_to_unitquat(torch.stack([zeros, theta, zeros], -1))
    n = len(angles)
    return TomogramReconstructor(
        V=torch.zeros(nz, nxy, nxy),
        voxel_size=voxel_size,
        quaternions=quaternions,
        translations=torch.zeros(n, 2),
        ctf_params={"dfu": torch.full((n,), defocus), "cs": torch.full((n,), cs)},
        voltage=300.0,
        dose_per_angstrom=1.0,
        tilt=TiltGeometry(tilt_axis="y"),
        propagation=Propagation(scattering_model=model, alpha=0.1),
    )


def test_the_edge_band_is_the_lens_delocalisation_distance() -> None:
    """
    With no spherical aberration the band is lambda * k_Nyquist times the
    defocus of the deepest plane, the midplane's plus half the column, in
    pixels; it widens with defocus, and a model with no depth at zero defocus
    has none.
    """
    from specter.constants import energy_to_wavelength

    voxel = 8.0
    widths = [_band_model(512, voxel, df)._edge_pads()[0] for df in (5e3, 15e3, 3e4)]
    assert widths == sorted(widths) and widths[0] < widths[-1]
    half_depth = 16 * voxel / 2
    reach = energy_to_wavelength(300.0) * (0.5 / voxel) * (3e4 + half_depth)
    assert widths[-1] == math.ceil(reach / voxel)
    assert _band_model(512, voxel, 0.0, model="projection")._edge_pads() == [0]
    # Spherical aberration pulls the high frequencies back towards focus.
    assert _band_model(512, voxel, 3e4, cs=2.7e7)._edge_pads()[0] <= widths[-1]


def test_an_edge_band_wider_than_a_quarter_field_is_dropped() -> None:
    """A field too small for its defocus has no unreached pixels to keep."""
    assert _band_model(32, 2.0, 12000.0)._edge_pads() == [0]
    assert _band_model(512, 2.0, 12000.0)._edge_pads()[0] > 0


def test_the_loss_ignores_residuals_inside_the_edge_band() -> None:
    model = _band_model(64, 4.0, 12000.0)
    band = model._edge_pads()[0]
    assert 0 < band <= 64 // 4
    sim = torch.ones(64, 64)
    edge = sim.clone()
    edge[:band] += 5.0
    edge[:, -band:] += 5.0
    inside = sim.clone()
    inside[32, 32] += 5.0
    assert float(model._compute_loss(sim, edge, 0)) == 0.0
    assert float(model._compute_loss(sim, inside, 0)) > 0.0
    model.use_fov_mask = False
    assert float(model._compute_loss(sim, edge, 0)) > 0.0


def test_propagation_curvature_sign_reaches_the_scattering(tr_kwargs: dict) -> None:
    """`Propagation.ews_curvature_sign` is the sign the model propagates with."""
    for sign in ("negative", "positive"):
        model = TomogramReconstructor(
            **tr_kwargs,
            propagation=Propagation(
                scattering_model="multislice", ews_curvature_sign=sign
            ),
        )
        assert model.iterative_scattering.ews_curvature_sign == sign


# ---------------------------------------------------------------------------
# Surface forward model, covering volume, priors and the full-batch solver
# ---------------------------------------------------------------------------


def _blobs(shape: tuple[int, int, int], amplitude: float = 1.0) -> torch.Tensor:
    """A few Gaussians well inside ``shape``: band limited and compact."""
    nz, ny, nx = shape
    z, y, x = torch.meshgrid(
        torch.arange(nz) - nz // 2,
        torch.arange(ny) - ny // 2,
        torch.arange(nx) - nx // 2,
        indexing="ij",
    )
    volume = torch.zeros(shape)
    for (cz, cy, cx), sigma in (
        ((1, 2, -3), 1.5),
        ((-1, -3, 2), 1.8),
        ((0, 1, 4), 1.4),
    ):
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        volume += amplitude * torch.exp(-r2 / (2 * sigma**2))
    return volume


def _series_kwargs(angles: torch.Tensor, axis: str = "y") -> dict:
    theta = torch.deg2rad(angles)
    zero = torch.zeros_like(theta)
    rotvecs = torch.stack(
        [zero, theta, zero] if axis == "y" else [theta, zero, zero], -1
    )
    n = len(angles)
    return dict(
        voxel_size=4.0,
        quaternions=roma.rotvec_to_unitquat(rotvecs),
        translations=torch.zeros(n, 2),
        ctf_params={"dfu": torch.full((n,), 8000.0), "cs": torch.full((n,), 2.7e7)},
        voltage=300.0,
        dose_per_angstrom=5.0,
        tilt=TiltGeometry(tilt_axis=axis),
    )


def test_surface_forward_agrees_with_the_slice_loop() -> None:
    """A single-scatter model samples the Ewald surface unless told to walk
    the tilted volume, and the two predict the same image: to rounding
    untilted, where neither interpolates, and to the slice loop's trilinear
    blur at a tilt."""
    volume = _blobs((12, 24, 24))
    kwargs = _series_kwargs(torch.tensor([0.0, 25.0]))
    kwargs["translations"] = torch.tensor([[0.0, 0.0], [6.0, -3.0]])
    surface = TomogramReconstructor(
        volume.clone(), **kwargs, propagation=Propagation(scattering_model="rytov")
    )
    loop = TomogramReconstructor(
        volume.clone(),
        **kwargs,
        propagation=Propagation(scattering_model="rytov", rotate_mode="real"),
    )
    assert surface._surface and not loop._surface
    untilted = (surface.forward(0) - loop.forward(0)).abs().max() / loop.forward(
        0
    ).std()
    assert untilted < 1e-3
    a, b = surface.forward(1).flatten(), loop.forward(1).flatten()
    assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.995


def test_covering_shape_holds_every_ray() -> None:
    """The covering volume grows across the tilt axis by the tilted field
    plus the slab's shear, keeps the images' parity, and leaves the loss
    nothing to mask."""
    angles = torch.tensor([-45.0, 0.0, 45.0])
    kwargs = _series_kwargs(angles)
    nz, n, margin = 8, 24, 4
    shape = TomogramReconstructor.covering_shape(n, nz, kwargs["quaternions"], margin)
    across = n / math.cos(math.radians(45)) + nz * math.tan(math.radians(45))
    assert shape[0] == nz
    assert shape[1] == n + 2 * margin  # along the tilt axis (y)
    assert across + 2 * margin <= shape[2] <= across + 2 * margin + 3
    assert (shape[1] - n) % 2 == 0 and (shape[2] - n) % 2 == 0

    model = TomogramReconstructor(
        torch.zeros(shape),
        **kwargs,
        image_size=n,
        propagation=Propagation(scattering_model="projection"),
    )
    assert model._fov_pads() == [None, None, None]
    assert model.forward(0).shape == (n, n)
    assert model.field_volume().shape == (nz, n, n)


def test_a_wider_volume_needs_the_surface() -> None:
    kwargs = _series_kwargs(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="walks the tilted volume"):
        TomogramReconstructor(
            torch.zeros(8, 16, 24),
            **kwargs,
            image_size=16,
            propagation=Propagation(scattering_model="multislice"),
        )
    with pytest.raises(ValueError, match="parity"):
        TomogramReconstructor(
            torch.zeros(8, 16, 21),
            **kwargs,
            image_size=16,
            propagation=Propagation(scattering_model="rytov"),
        )


def test_regularisation_is_quadratic_then_linear_in_the_gradient() -> None:
    kwargs = _series_kwargs(torch.tensor([0.0]))
    prop = Propagation(scattering_model="projection")
    ramp = torch.arange(8.0).reshape(1, 1, 8).expand(8, 8, 8)  # unit gradient in x
    # Every voxel with a neighbour ahead of it in x, the last faces in y and
    # z included.
    n_grad = 8 * 8 * 7

    smooth = TomogramReconstructor(
        torch.zeros(8, 8, 8), **kwargs, propagation=prop, smoothness=2.0
    )
    assert float(smooth.regularisation(torch.ones(8, 8, 8))) == 0.0
    assert float(smooth.regularisation(0.1 * ramp)) == pytest.approx(
        0.5 * 2.0 * 0.01 * n_grad, rel=1e-5
    )

    tv = TomogramReconstructor(
        torch.zeros(8, 8, 8), **kwargs, propagation=prop, tv_weight=3.0, tv_delta=0.01
    )
    # Far below tv_delta the prior is g^2 / (2 delta); far above, |g| - delta.
    small, large = 1e-4, 1.0
    assert float(tv.regularisation(small * ramp)) == pytest.approx(
        3.0 * small**2 / (2 * 0.01) * n_grad, rel=1e-2
    )
    assert float(tv.regularisation(large * ramp)) == pytest.approx(
        3.0 * (large - 0.01) * n_grad, rel=1e-3
    )


def test_solver_fits_a_series_and_the_covering_volume_recovers_more() -> None:
    """The full-batch solver lowers chi-squared towards zero on noiseless
    data and recovers the phantom, and a volume holding every ray recovers
    it better than one as wide as the images with the loss masked."""
    from specter.ghostbuster._tomogram_solver import solve_tilt_series

    angles = torch.linspace(-50.0, 50.0, 11)
    kwargs = _series_kwargs(angles)
    nz, n = 12, 24
    prop = Propagation(scattering_model="rytov", oversample=1)
    shape = TomogramReconstructor.covering_shape(n, nz, kwargs["quaternions"], margin=4)
    # Structure everywhere, as a specimen has: what lies beside the field of
    # view enters every tilted image, and only the covering volume holds it.
    noise = torch.randn(1, 1, *shape, generator=torch.Generator().manual_seed(0))
    truth = _blobs(shape, amplitude=2.0) + F.avg_pool3d(noise, 3, 1, 1)[0, 0]
    simulator = TomogramReconstructor(truth, **kwargs, image_size=n, propagation=prop)
    with torch.no_grad():
        images = simulator._forward_tilts(truth, torch.arange(len(angles)))

    def recover(volume_shape: tuple[int, int, int]) -> tuple[float, float, float]:
        model = TomogramReconstructor(
            torch.zeros(volume_shape),
            **kwargs,
            image_size=n,
            propagation=prop,
            lr=0.1,
            smoothness=1e-3,
        )
        log = solve_tilt_series(model, images, 12, tilt_batch=4)
        got = model.field_volume().flatten()
        want = simulator.field_volume().flatten()
        cc = float(torch.corrcoef(torch.stack([got, want]))[0, 1])
        return log[0]["chi2_per_pixel"], log[-1]["chi2_per_pixel"], cc

    first, last, cc_covering = recover(shape)
    assert last < first / 5
    assert cc_covering > 0.7
    _, _, cc_masked = recover((nz, n, n))
    assert cc_covering > cc_masked + 0.1


def test_the_last_faces_are_penalised_like_the_interior() -> None:
    """A voxel on the volume's last face is held by the prior through every
    difference it takes part in, as an interior voxel is; a spike costs the
    same in the last corner as it would if the corner had neighbours behind
    it only."""
    kwargs = _series_kwargs(torch.tensor([0.0]))
    model = TomogramReconstructor(
        torch.zeros(8, 8, 8),
        **kwargs,
        propagation=Propagation(scattering_model="projection"),
        smoothness=2.0,
    )
    spike = torch.zeros(8, 8, 8)
    spike[4, 4, 4] = 1.0
    assert float(model.regularisation(spike)) == pytest.approx(0.5 * 2.0 * 6)
    spike = torch.zeros(8, 8, 8)
    spike[-1, -1, -1] = 1.0
    assert float(model.regularisation(spike)) == pytest.approx(0.5 * 2.0 * 3)
    spike = torch.zeros(8, 8, 8)
    spike[3, -1, -1] = 1.0  # on two last faces: four neighbours
    assert float(model.regularisation(spike)) == pytest.approx(0.5 * 2.0 * 4)


def test_the_preconditioned_solve_is_reproducible_and_converges() -> None:
    """Two identical solves under the quadratic prior agree, on noisy data.

    The curvature that preconditions the solve is probed without the
    observed images, whose term in the gradient left only rounding error in
    the probe: two probes agree, and the solve reaches its minimum within a
    few iterations and stops there instead of following the rounding of a
    converged gradient. Before, two runs differed by a third of the
    volume's root-mean-square value or more."""
    from specter.ghostbuster._tomogram_solver import _Objective, solve_tilt_series

    kwargs, shape, truth = _slanted_series()
    prop = Propagation(scattering_model="rytov", oversample=1)
    simulator = TomogramReconstructor(truth, **kwargs, propagation=prop)
    with torch.no_grad():
        clean = simulator._forward_tilts(truth, torch.arange(5))
    images = torch.poisson(clean, generator=torch.Generator().manual_seed(0))

    def solve(iterations: int) -> tuple[torch.Tensor, list[dict]]:
        model = TomogramReconstructor(
            torch.zeros(shape), **kwargs, propagation=prop, smoothness=0.03
        )
        log = solve_tilt_series(model, images, iterations)
        return model.V - model.V.mean(), log

    model = TomogramReconstructor(
        torch.zeros(shape), **kwargs, propagation=prop, smoothness=0.03
    )
    objective = _Objective(model, images, torch.arange(5), 8)
    first = objective.curvature(shape, torch.device("cpu"))
    second = objective.curvature(shape, torch.device("cpu"))
    assert float((first - second).norm() / first.norm()) < 1e-4

    (a, log_a), (b, _) = solve(20), solve(20)
    assert float((a - b).norm() / a.norm()) < 0.02
    # Converged within a few iterations, and stopped once it had.
    assert (
        log_a[4]["objective"] - log_a[-1]["objective"] < 1e-4 * log_a[-1]["objective"]
    )
    assert len(log_a) < 20


def test_pipeline_solves_a_covering_volume_and_saves_the_field(tmp_path: Path) -> None:
    from specter.ghostbuster import TomogramGhostbuster

    angles = torch.linspace(-30.0, 30.0, 5)
    kwargs = _series_kwargs(angles)
    nz, n = 8, 16
    prop = Propagation(scattering_model="rytov", oversample=1)
    shape = TomogramReconstructor.covering_shape(n, nz, kwargs["quaternions"])
    simulator = TomogramReconstructor(
        _blobs(shape), **kwargs, image_size=n, propagation=prop
    )
    with torch.no_grad():
        images = simulator._forward_tilts(simulator.V, torch.arange(len(angles)))

    tgb = TomogramGhostbuster(
        images,
        voxel_size=kwargs["voxel_size"],
        voltage=300.0,
        ctf_params=kwargs["ctf_params"],
        angles=angles,
        tilt=kwargs["tilt"],
        nz=nz,
        lr=0.1,
        propagation=prop,
        defocus_hand="negative",
        iterations=3,
        smoothness=1e-3,
        run_dir=tmp_path,
    )
    # No dose given: each tilt's mean counts per unit area.
    expected = images.mean(dim=(-2, -1)) / kwargs["voxel_size"] ** 2
    assert torch.allclose(tgb._dose_per_angstrom, expected)
    model = tgb.run(device="cpu")
    assert tuple(model.V.shape) == shape
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics) == 3
    assert metrics[-1]["chi2_per_pixel"] < metrics[0]["chi2_per_pixel"]
    from specter.io import read_mrc

    saved, _ = read_mrc(tmp_path / "volume.mrc")
    assert tuple(saved.shape) == (nz, n, n)


# ---------------------------------------------------------------------------
# Pixel mask and the strip-wise solve
# ---------------------------------------------------------------------------


def test_pixel_mask_selects_the_pixels_the_loss_compares() -> None:
    """Residuals outside ``pixel_mask`` do not count, and only a masked
    volume may be narrower than its images."""
    kwargs = _series_kwargs(torch.tensor([0.0, 20.0]))
    prop = Propagation(scattering_model="rytov", oversample=1)
    mask = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask[:, :, :8] = True
    model = TomogramReconstructor(
        torch.zeros(8, 16, 16),
        **kwargs,
        propagation=prop,
        use_fov_mask=False,
        pixel_mask=mask,
    )
    idx = torch.arange(2)
    sim = torch.ones(2, 16, 16)
    outside = sim.clone()
    outside[:, :, 8:] += 3.0
    inside = sim.clone()
    inside[:, 4, 4] += 3.0
    assert float(model._half_chi2(sim, outside, idx)) == 0.0
    assert float(model._half_chi2(sim, inside, idx)) > 0.0
    assert float(model._compute_loss(sim[0], outside[0], 0)) == 0.0
    assert model._n_valid_pixels(0) == 16 * 8

    with pytest.raises(ValueError, match="pixel_mask"):
        TomogramReconstructor(
            torch.zeros(8, 16, 10), **kwargs, image_size=16, propagation=prop
        )
    narrow = TomogramReconstructor(
        torch.zeros(8, 16, 10),
        **kwargs,
        image_size=16,
        propagation=prop,
        use_fov_mask=False,
        pixel_mask=mask,
    )
    assert narrow.forward(0).shape == (16, 16)


def _slanted_series(
    n: int = 80, nz: int = 6, slant_deg: float = 4.0
) -> tuple[dict, tuple[int, int, int], torch.Tensor]:
    """A tilt series about an axis ``slant_deg`` off the image's x, with
    shifts, posed in the volume frame turned to that axis; its covering
    volume's shape and a phantom of smooth blobs filling it."""
    from specter.tilt import turn_to_tilt_axis

    angles = torch.linspace(-40.0, 40.0, 5)
    slant = math.radians(slant_deg)
    axis = torch.tensor([math.cos(slant), math.sin(slant), 0.0])
    quaternions, turned = turn_to_tilt_axis(
        roma.rotvec_to_unitquat(torch.deg2rad(angles)[:, None] * axis)
    )
    assert turned == pytest.approx(-slant_deg, abs=1e-3)
    kwargs = dict(
        voxel_size=4.0,
        quaternions=quaternions,
        translations=torch.tensor(
            [[5.0, -7.0], [0.0, 3.0], [-4.0, 0.0], [6.0, 6.0], [-3.0, -5.0]]
        ),
        ctf_params={"dfu": torch.full((5,), 2500.0), "cs": torch.full((5,), 2.7e7)},
        voltage=300.0,
        dose_per_angstrom=5.0,
        image_size=n,
    )
    shape = TomogramReconstructor.covering_shape(n, nz, quaternions)
    _, ny, nx = shape
    z, y, x = torch.meshgrid(
        torch.arange(nz) - nz // 2,
        torch.arange(ny) - ny // 2,
        torch.arange(nx) - nx // 2,
        indexing="ij",
    )
    generator = torch.Generator().manual_seed(1)
    truth = torch.zeros(shape)
    for _ in range(40):
        cy = float(
            torch.randint(-ny // 2 + 12, ny // 2 - 12, (1,), generator=generator)
        )
        cx = float(
            torch.randint(-nx // 2 + 12, nx // 2 - 12, (1,), generator=generator)
        )
        cz = float(torch.randint(-1, 2, (1,), generator=generator))
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        truth += 2.0 * torch.exp(-r2 / (2 * 2.0**2))
    return kwargs, shape, truth


def test_strip_plan_divides_the_volume_into_equal_overlapping_strips() -> None:
    """The strips have one shape of the images' parity, their shares sum to
    one at every voxel, each strip's loss stops short of its inner faces by
    the taper and the transfer function's reach, and a series that does not
    tilt about a volume axis is refused."""
    from specter.ghostbuster._tomogram_solver import StripPlan

    kwargs, shape, truth = _slanted_series()
    prop = Propagation(scattering_model="rytov", oversample=1)
    model = TomogramReconstructor(truth, **kwargs, propagation=prop)
    plan = StripPlan.for_model(model, strips=2)
    assert plan.dim == 2 and plan.size == shape[2]
    assert len(plan.starts) == 2 and plan.starts[0] == 0
    assert plan.starts[-1] + plan.width == plan.size
    assert (plan.width - model.nxy) % 2 == 0
    assert plan.reach == max(model._delocalisation()) and plan.fade >= 8

    u = torch.arange(plan.size, dtype=torch.float64)
    shares = torch.stack([plan.weight(i, u) for i in range(2)])
    assert torch.allclose(shares.sum(0), torch.ones_like(u))
    for i, start in enumerate(plan.starts):  # no share outside the strip itself
        outside = (u < start) | (u >= start + plan.width)
        assert float(shares[i][outside].abs().max()) == 0.0
    margin = plan.taper + plan.reach
    inner_face = plan.starts[0] + plan.width - 1
    assert bool(plan.compared(0, torch.tensor([0.0, inner_face - margin])).all())
    assert not bool(plan.compared(0, torch.tensor([inner_face - margin + 1.0])).any())
    # A cross-fade lies where both strips compare every pixel it images onto.
    seam = plan.seams[0]
    for edge in (seam - plan.fade, seam + plan.fade):
        for i in range(2):
            assert bool(plan.compared(i, torch.tensor([edge - plan.reach])))
            assert bool(plan.compared(i, torch.tensor([edge + plan.reach])))
    assert float(plan.window(0)[0]) == 1.0 and float(plan.window(0)[-1]) < 0.05
    assert float(plan.window(1)[0]) < 0.05 and float(plan.window(1)[-1]) == 1.0

    assert len(StripPlan.for_model(model, max_voxels=truth.numel()).starts) == 1
    by_size = StripPlan.for_model(model, max_voxels=int(0.8 * truth.numel()))
    assert len(by_size.starts) == 2
    with pytest.raises(ValueError, match="no room"):
        StripPlan.for_model(model, strips=6)

    oblique = dict(kwargs)
    axis = torch.tensor([math.cos(0.6), math.sin(0.6), 0.0])
    oblique["quaternions"] = roma.rotvec_to_unitquat(
        torch.deg2rad(torch.linspace(-40.0, 40.0, 5))[:, None] * axis
    )
    with pytest.raises(ValueError, match="turn_to_tilt_axis"):
        StripPlan.for_model(
            TomogramReconstructor(truth, **oblique, propagation=prop), strips=2
        )


def test_strips_predict_the_images_the_whole_volume_predicts() -> None:
    """Each strip, posed by the shift its centre implies, predicts the
    pixels whose rays it holds as the whole volume does, with the tilt axis
    off the image axis and the tilts shifted."""
    from specter.ghostbuster._tomogram_solver import predict_tilt_series

    kwargs, _, truth = _slanted_series()
    for model_name in ("rytov", "ctf"):
        model = TomogramReconstructor(
            truth,
            **kwargs,
            propagation=Propagation(scattering_model=model_name, oversample=1),
        )
        whole = predict_tilt_series(model)
        strips = predict_tilt_series(model, strips=2)
        contrast = (whole - whole.mean(dim=(-2, -1), keepdim=True)).std()
        assert float((strips - whole).abs().max() / contrast) < 0.02


def test_strips_divide_along_y_when_the_tilt_axis_is_y() -> None:
    """A series tilting about the volume's y is divided along y, and its
    strips predict the whole volume's images."""
    from specter.ghostbuster._tomogram_solver import StripPlan, predict_tilt_series
    from specter.tilt import turn_to_tilt_axis

    n, nz = 80, 6
    axis = torch.tensor([math.sin(0.07), math.cos(0.07), 0.0])
    tilts = torch.deg2rad(torch.linspace(-40.0, 40.0, 5))
    quaternions, _ = turn_to_tilt_axis(
        roma.rotvec_to_unitquat(tilts[:, None] * axis), "y"
    )
    shape = TomogramReconstructor.covering_shape(n, nz, quaternions)
    z, y, x = torch.meshgrid(
        torch.arange(nz) - nz // 2,
        torch.arange(shape[1]) - shape[1] // 2,
        torch.arange(shape[2]) - shape[2] // 2,
        indexing="ij",
    )
    volume = torch.zeros(shape)
    for cy, cx in ((-30, 20), (-8, -40), (5, 33), (12, -15), (35, 2)):
        volume += 2.0 * torch.exp(-(z**2 + (y - cy) ** 2 + (x - cx) ** 2) / 8.0)
    model = TomogramReconstructor(
        volume,
        4.0,
        quaternions,
        torch.tensor([[5.0, -7.0], [0.0, 3.0], [-4.0, 0.0], [6.0, 6.0], [-3.0, -5.0]]),
        {"dfu": torch.full((5,), 2500.0), "cs": torch.full((5,), 2.7e7)},
        300.0,
        5.0,
        image_size=n,
        propagation=Propagation(scattering_model="rytov", oversample=1),
    )
    plan = StripPlan.for_model(model, strips=2)
    assert plan.dim == 1 and plan.size == shape[1]
    whole = predict_tilt_series(model)
    strips = predict_tilt_series(model, strips=2)
    contrast = (whole - whole.mean(dim=(-2, -1), keepdim=True)).std()
    assert float((strips - whole).abs().max() / contrast) < 0.02


@pytest.mark.parametrize(
    ("prior", "iterations"),
    [({"smoothness": 0.03}, 10), ({"tv_weight": 0.01, "tv_delta": 0.05}, 30)],
    ids=["quadratic", "tv"],
)
def test_strip_solve_agrees_with_the_whole_solve(prior: dict, iterations: int) -> None:
    """Solved in two strips and cross-faded, the volume is the one a single
    solve gives, in the strips' interiors and across the seam, up to the
    mean potential, which the data leave free."""
    from specter.ghostbuster._tomogram_solver import (
        StripPlan,
        solve_tilt_series,
        solve_tilt_series_in_strips,
    )

    kwargs, shape, truth = _slanted_series()
    prop = Propagation(scattering_model="rytov", oversample=1)
    simulator = TomogramReconstructor(truth, **kwargs, propagation=prop)
    with torch.no_grad():
        images = simulator._forward_tilts(truth, torch.arange(5))

    whole = TomogramReconstructor(
        torch.zeros(shape), **kwargs, propagation=prop, **prior
    )
    solve_tilt_series(whole, images, iterations)
    strips = TomogramReconstructor(
        torch.zeros(shape), **kwargs, propagation=prop, **prior
    )
    logs = solve_tilt_series_in_strips(strips, images, iterations, strips=2)
    assert all(0 < len(log) <= iterations for log in logs)

    plan = StripPlan.for_model(strips, strips=2)
    distance = (torch.arange(shape[2]) - plan.seams[0]).abs()
    zones = {"seam": distance <= plan.fade, "interior": distance > plan.fade}
    for name, zone in zones.items():
        a = strips.V[..., zone].flatten()
        b = whole.V[..., zone].flatten()
        cc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        assert cc > 0.99, (name, cc)
        difference = (a - a.mean()) - (b - b.mean())
        assert float(difference.norm() / (b - b.mean()).norm()) < 0.15, name
    # Both recover the phantom, so the agreement is not that of two empty fits.
    got, want = whole.V.flatten(), truth.flatten()
    assert float(torch.corrcoef(torch.stack([got, want]))[0, 1]) > 0.4


def test_pipeline_solves_in_strips_and_saves_the_field(tmp_path: Path) -> None:
    from specter.ghostbuster import TomogramGhostbuster

    kwargs, shape, truth = _slanted_series()
    prop = Propagation(scattering_model="rytov", oversample=1)
    simulator = TomogramReconstructor(truth, **kwargs, propagation=prop)
    with torch.no_grad():
        images = simulator._forward_tilts(truth, torch.arange(5))
    tgb = TomogramGhostbuster(
        images,
        voxel_size=kwargs["voxel_size"],
        voltage=300.0,
        ctf_params=kwargs["ctf_params"],
        quaternions=kwargs["quaternions"],
        translations=kwargs["translations"],
        nz=shape[0],
        propagation=prop,
        defocus_hand="negative",
        iterations=3,
        smoothness=0.03,
        strips=2,
        run_dir=tmp_path,
    )
    model = tgb.run(device="cpu")
    assert tuple(model.V.shape) == shape
    assert float(model.V.abs().max()) > 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics) == 2 and all(len(strip) == 3 for strip in metrics)
    with pytest.raises(ValueError, match="one or the other"):
        TomogramGhostbuster(
            images,
            voxel_size=4.0,
            voltage=300.0,
            ctf_params=kwargs["ctf_params"],
            quaternions=kwargs["quaternions"],
            defocus_hand="negative",
            strips=2,
            max_voxels=1000,
        )


# ---------------------------------------------------------------------------
# Defocus hand
# ---------------------------------------------------------------------------


def test_the_curvature_sign_is_the_hand_of_defocus_with_depth() -> None:
    """A scatterer at height z in the beam frame is imaged as one at the
    midplane with defocus df + z under "negative" and df - z under
    "positive", which is what `defocus_hand` documents."""
    n, nz, voxel, height, defocus = 48, 33, 4.0, 12, 20000.0
    z, y, x = torch.meshgrid(
        torch.arange(nz) - nz // 2,
        torch.arange(n) - n // 2,
        torch.arange(n) - n // 2,
        indexing="ij",
    )

    def image(cz: int, df: float, hand: str) -> torch.Tensor:
        blob = 3.0 * torch.exp(-((z - cz) ** 2 + y**2 + x**2) / (2 * 1.5**2))
        model = TomogramReconstructor(
            blob,
            voxel,
            roma.rotvec_to_unitquat(torch.zeros(1, 3)),
            torch.zeros(1, 2),
            {"dfu": torch.tensor([df]), "cs": torch.tensor([0.0])},
            300.0,
            1.0,
            propagation=Propagation(
                scattering_model="rytov", ews_curvature_sign=hand, oversample=1
            ),
            use_fov_mask=False,
        )
        return model.forward(0)

    for hand, sign in (("negative", 1), ("positive", -1)):
        raised = image(height, defocus, hand)
        misfit = {
            s: float((raised - image(0, defocus + s * height * voxel, hand)).norm())
            for s in (-1, 0, 1)
        }
        assert misfit[sign] < 0.5 * misfit[0] < 0.5 * misfit[-sign]


def test_pipeline_takes_the_defocus_hand_and_warns_without_one() -> None:
    from specter.ghostbuster import TomogramGhostbuster

    kwargs = dict(
        tilt_series=torch.ones(3, 16, 16),
        voxel_size=4.0,
        voltage=300.0,
        ctf_params={"dfu": torch.full((3,), 8000.0), "cs": torch.full((3,), 2.7e7)},
        angles=[-20.0, 0.0, 20.0],
        nz=8,
    )
    with pytest.warns(UserWarning, match="defocus_hand"):
        assumed = TomogramGhostbuster(**kwargs)
    assert assumed.propagation.ews_curvature_sign == "negative"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_ = TomogramGhostbuster(**kwargs, defocus_hand="positive")
        # A flat model has no depth, so it needs no hand.
        TomogramGhostbuster(**kwargs, propagation=Propagation(scattering_model="ctf"))
    assert set_.propagation.ews_curvature_sign == "positive"
    assert set_.propagation.scattering_model == "rytov"
    with pytest.raises(ValueError, match="defocus_hand"):
        TomogramGhostbuster(**kwargs, defocus_hand="left")
