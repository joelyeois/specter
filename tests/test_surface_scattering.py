"""
rotate_mode="surface": exit waves sampled from the unposed volume's transform.

The reference is the rotate-then-sum path on poses where it does not
interpolate (the identity, quarter turns, whole-voxel shifts), on an object
compact enough that neither path loses content at the box edge. There the two
must agree to float precision for every single-scatter model, both Ewald
signs and either padding.
"""

from __future__ import annotations

import math

import pytest
import roma
import torch

from specter.ghostbuster import Reconstructor
from specter.imaging import Poses, Specimen
from specter.settings import Propagation
from conftest import fit_one_epoch

VOXEL = 1.5


def _compact_object(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    z, y, x = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float64) - n // 2] * 3, indexing="ij"
    )
    g = torch.Generator().manual_seed(3)
    centres = (torch.rand(12, 3, generator=g, dtype=torch.float64) - 0.5) * 0.25 * n
    v = torch.zeros(n, n, n, dtype=torch.float64)
    for cx, cy, cz in centres:
        v += 5 * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 3.0)
    return v.to(dtype)


def _grid_poses() -> Poses:
    """Poses the rotate-then-sum path applies without interpolating."""
    turns = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, math.pi / 2],
        [0.0, 0.0, math.pi],
        [math.pi / 2, 0.0, 0.0],
        [-math.pi / 2, 0.0, 0.0],
        [0.0, math.pi / 2, 0.0],
    ]
    quaternions = roma.rotvec_to_unitquat(torch.tensor(turns))
    shifts = torch.tensor([[0, 0], [2, 0], [0, -3], [1, 1], [-2, 3], [3, -1]]) * VOXEL
    return Poses(quaternions, shifts.float(), torch.arange(len(turns)))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().norm() / (b - b.mean()).abs().norm())


def _waves(volume: torch.Tensor, rotate_mode: str, **settings: object) -> torch.Tensor:
    return (
        Specimen(volume, VOXEL)
        .scatter(_grid_poses(), voltage=300.0, rotate_mode=rotate_mode, **settings)
        .array
    )


@pytest.mark.parametrize("model", ["projection", "firstborn", "rytov", "ctf"])
@pytest.mark.parametrize("sign", ["negative", "positive"])
@pytest.mark.parametrize("pad_fft", [False, True])
def test_surface_matches_rotate_then_sum_on_grid_poses(
    model: str, sign: str, pad_fft: bool
) -> None:
    volume = _compact_object(25)
    settings = dict(
        scattering_model=model,
        ews_curvature_sign=sign,
        pad_fft=pad_fft,
        alpha=0.07,
        oversample=2,
    )
    surface = _waves(volume, "surface", **settings)
    rotated = _waves(volume, "real", **settings)
    assert surface.shape == rotated.shape
    assert _rel(surface, rotated) < 1e-4


def test_surface_tells_the_ewald_signs_apart() -> None:
    """The parity above would be vacuous if the curvature sign made no difference."""
    volume = _compact_object(25)
    surface = _waves(volume, "surface", scattering_model="firstborn", oversample=1)
    other_sign = _waves(
        volume,
        "real",
        scattering_model="firstborn",
        ews_curvature_sign="positive",
        oversample=1,
    )
    assert _rel(surface, other_sign) > 1e-3


def test_surface_gradients_match_finite_differences() -> None:
    volume = _compact_object(13, torch.float64).requires_grad_()
    quaternions = roma.random_unitquat(3, dtype=torch.float64)
    shifts = torch.randn(3, 2, dtype=torch.float64).requires_grad_()
    weights = torch.linspace(0.5, 1.5, 3 * 13 * 13, dtype=torch.float64)

    def loss(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        wave = (
            Specimen(v, VOXEL)
            .scatter(
                Poses(quaternions, t, torch.arange(3)),
                voltage=300.0,
                scattering_model="rytov",
                rotate_mode="surface",
                oversample=1,
            )
            .array
        )
        return (wave.abs() ** 2 * weights.reshape(wave.shape)).sum()

    loss(volume, shifts).backward()
    assert volume.grad is not None and shifts.grad is not None
    # The transform is accurate to the NUFFT's eps (1e-6), not to machine
    # precision, which bounds how closely a central difference can agree; a
    # wrong adjoint or sign is off by order one.
    dv, dt, h = torch.randn_like(volume), torch.randn_like(shifts), 1e-4
    with torch.no_grad():
        fd_v = (loss(volume + h * dv, shifts) - loss(volume - h * dv, shifts)) / (2 * h)
        fd_t = (loss(volume, shifts + h * dt) - loss(volume, shifts - h * dt)) / (2 * h)
    assert float((volume.grad * dv).sum()) == pytest.approx(float(fd_v), rel=1e-3)
    assert float((shifts.grad * dt).sum()) == pytest.approx(float(fd_t), rel=1e-3)


def test_surface_refuses_models_it_cannot_express() -> None:
    with pytest.raises(ValueError, match="single-scatter"):
        Propagation(scattering_model="multislice", rotate_mode="surface")
    with pytest.raises(ValueError, match="single-scatter"):
        Propagation(scattering_model="kinematic", rotate_mode="surface")


def test_surface_refuses_a_dressed_specimen() -> None:
    specimen = Specimen(_compact_object(25), VOXEL).crowd(min_distance=20.0)
    with pytest.raises(ValueError, match="crowding"):
        specimen.scatter(
            _grid_poses(),
            voltage=300.0,
            scattering_model="rytov",
            rotate_mode="surface",
        )
    with pytest.raises(ValueError, match="never forms the posed volume"):
        Specimen(_compact_object(25), VOXEL).materialize(
            _grid_poses(),
            voltage=300.0,
            scattering_model="rytov",
            rotate_mode="surface",
        )


def _reconstructor(rotate_mode: str, **kwargs: object) -> Reconstructor:
    n = 25
    ctf = {
        "dfu": torch.full((1,), 8000.0),
        "dfv": torch.full((1,), 8000.0),
        "dfang": torch.zeros(1),
        "cs": torch.full((1,), 2.7e7),
    }
    return Reconstructor(
        V=_compact_object(n),
        voxel_size=VOXEL,
        quaternions=roma.rotvec_to_unitquat(torch.tensor([[math.pi / 2, 0.0, 0.0]])),
        translations=torch.tensor([[1.5, -3.0]]),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(scattering_model="rytov", rotate_mode=rotate_mode),
        **kwargs,
    )


def test_reconstructor_forward_is_the_same_image_under_surface() -> None:
    idx = torch.tensor([0])
    surface = _reconstructor("surface").forward(idx)
    rotated = _reconstructor("real").forward(idx)
    assert _rel(surface, rotated) < 1e-4


def test_reconstructor_trains_under_surface_and_refuses_lr_r() -> None:
    model = _reconstructor("surface", lr=0.1)
    images = model.forward(torch.tensor([0])).detach() * 1.1
    before = model.V.detach().clone()
    fit_one_epoch(model, images, max_epochs=1)
    assert not torch.equal(model.V.detach(), before)
    with pytest.raises(ValueError, match="does not refine rotations"):
        _reconstructor("surface", lr=0.1, lr_R=1e-3)


def test_symmetrization_reuses_the_surface_plan() -> None:
    """Exact symmetrization at an epoch's end takes the plan the surface forward
    already holds, rather than a second oversampled grid beside a training
    step's cached memory (which ran a 560^3 run out of device memory)."""
    from specter import nufft
    from specter.symmetries import get_rotation_matrices, symmetrize_exact

    nufft.clear_plan_cache()
    volume = _compact_object(17)
    _waves(volume, "surface", scattering_model="rytov", oversample=1)
    held = set(nufft._PLANS)
    symmetrize_exact(volume, get_rotation_matrices("C2"))
    assert set(nufft._PLANS) == held
    nufft.clear_plan_cache()
