"""Tests for the converged reconstruction solver (`ghostbuster._converge`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import roma
import torch

from specter import nufft
from specter.arrays import ball3d
from specter.ghostbuster import Reconstructor
from specter.ghostbuster._converge import (
    FSC_CLIP,
    ConvergedSolver,
    auto_mask,
    frequency_shells,
    mask_check,
    masked_fsc,
    noise_substituted_fsc,
    spherical_mask,
    ssnr_from_fsc,
    unfilter,
)
from specter.settings import Propagation

N = 8
VOXEL = 4.0


def _blob(n: int, sigma: float, centre: float = 0.0) -> torch.Tensor:
    r = torch.arange(n, dtype=torch.float32) - n // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    return torch.exp(-((x - centre) ** 2 + y**2 + z**2) / (2 * sigma**2))


def _problem(
    scattering_model: str,
    n_particles: int = 24,
    noise: float = 1.0,
    symmetry: str | None = None,
    absorption: torch.Tensor | None = None,
) -> tuple[Reconstructor, torch.Tensor, torch.Tensor]:
    """A Reconstructor, its noisy images of a known map, and that map."""
    torch.manual_seed(0)
    truth = 3.0 * _blob(N, 1.5, 0.5) + 1.5 * _blob(N, 1.0, -1.5)
    ctf = {
        "dfu": 8000.0 + 2000.0 * torch.rand(n_particles),
        "dfv": 8000.0 + 2000.0 * torch.rand(n_particles),
        "dfang": 180.0 * torch.rand(n_particles),
        "cs": torch.full((n_particles,), 2.7),
        "phaseshift": torch.zeros(n_particles),
    }
    model = Reconstructor(
        V=torch.zeros(N, N, N),
        voxel_size=VOXEL,
        quaternions=roma.random_unitquat(n_particles),
        translations=torch.zeros(n_particles, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=50.0,
        scale=0.8 + 0.4 * torch.rand(n_particles),
        propagation=(
            Propagation(scattering_model=scattering_model)
            if absorption is None
            else Propagation(
                scattering_model=scattering_model, rotate_mode="surface", alpha=0.0
            )
        ),
        kmask=ball3d(N, N),
        lr=0.1,
        symmetry=symmetry,
    )
    solver = ConvergedSolver(model, torch.zeros(n_particles, N, N), np.ones(N // 2 + 1))
    with torch.no_grad():
        potential = truth if absorption is None else torch.complex(truth, absorption)
        clean = solver._predict(torch.arange(n_particles), potential)
    images = clean + noise * clean.std() * torch.randn_like(clean)
    return model, images, truth


def _start(truth: torch.Tensor) -> torch.Tensor:
    """A rough starting map, as a few Adam epochs would give."""
    generator = torch.Generator().manual_seed(1)
    return 0.5 * truth + 0.1 * torch.randn(truth.shape, generator=generator)


def _start_flat(truth: torch.Tensor) -> torch.Tensor:
    return _start(truth).flatten().double()


def test_unfilter_undoes_the_wiener_filter() -> None:
    torch.manual_seed(0)
    volume = torch.randn(N, N, N)
    ssnr = np.linspace(5.0, 0.01, N // 2 + 1)
    _, shell, _ = frequency_shells(N)
    wiener = torch.as_tensor(ssnr / (1 + ssnr), dtype=torch.float32)[shell]
    filtered = torch.fft.irfftn(torch.fft.rfftn(volume) * wiener, s=volume.shape)
    assert torch.allclose(unfilter(filtered, ssnr), volume, atol=1e-4)


def test_fsc_and_ssnr() -> None:
    torch.manual_seed(0)
    a = torch.randn(N, N, N)
    assert np.allclose(masked_fsc(a, a), 1.0, atol=1e-6)
    assert np.allclose(masked_fsc(a, -a), -1.0, atol=1e-6)
    ssnr = ssnr_from_fsc(np.array([1.0, 0.5, 0.0, -0.3]), smooth=1)
    lo, hi = FSC_CLIP
    assert np.allclose(ssnr, [hi / (1 - hi), 1.0, lo / (1 - lo), lo / (1 - lo)])
    # A noisy tail becomes smooth and non-increasing: no jump of the prior
    # between neighbouring shells.
    rng = np.random.default_rng(0)
    tail = np.concatenate([np.linspace(1.0, 0.05, 40), 0.03 * rng.standard_normal(40)])
    smoothed = ssnr_from_fsc(tail)
    assert np.all(np.diff(smoothed) <= 0)
    assert np.max(smoothed[:-1] / smoothed[1:]) < np.max(
        ssnr_from_fsc(tail, smooth=1)[:-1] / ssnr_from_fsc(tail, smooth=1)[1:]
    )


def test_auto_mask_covers_the_particle_and_not_the_solvent() -> None:
    """Two noisy half-maps of a particle with atomic detail inside a sphere:
    the mask holds the particle, leaves the corners out, has a soft edge,
    and keeps CryoSPARC's near/far proportions."""
    n = 48
    torch.manual_seed(0)
    centres = torch.randn(300, 3)
    centres = (
        centres / centres.norm(dim=1, keepdim=True) * 9 * torch.rand(300, 1) ** (1 / 3)
    )
    r = torch.arange(n, dtype=torch.float32) - n // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    grid = torch.stack([z, y, x], -1)
    truth = sum(torch.exp(-((grid - c) ** 2).sum(-1) / (2 * 0.8**2)) for c in centres)
    a = truth + 0.05 * torch.randn(n, n, n)
    b = truth + 0.05 * torch.randn(n, n, n)
    mask, info = auto_mask(a, b, voxel_size=1.0)
    assert float(mask[n // 2, n // 2, n // 2]) == pytest.approx(1.0)
    assert float(mask[0, 0, 0]) == pytest.approx(0.0)
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    assert bool(((mask > 0.05) & (mask < 0.95)).any())
    assert info["far"] == pytest.approx(2.5 * info["near"], rel=1e-3)


def test_noise_substitution_removes_mask_induced_correlation() -> None:
    """Halves sharing only a smooth blob have no true correlation in the high
    shells, but a sharp mask smears the blob's correlation into them; the
    noise-substituted FSC takes it back out."""
    n = 32
    torch.manual_seed(0)
    signal = 10.0 * _blob(n, 2.5)
    a = signal + 0.3 * torch.randn(n, n, n)
    b = signal + 0.3 * torch.randn(n, n, n)
    mask = (_blob(n, 3.0) > 0.3).float()
    high = slice(8, None)
    assert abs(masked_fsc(a, b)[high].mean()) < 0.02
    assert masked_fsc(a, b, mask)[high].mean() > 0.5
    corrected = noise_substituted_fsc(a, b, mask, start_shell=6)
    assert abs(corrected[high].mean()) < 0.03


def test_adjoint_tolerance_applies_inside_the_block_only() -> None:
    torch.manual_seed(0)
    grid = torch.randn(N, N, N, dtype=torch.float32, requires_grad=True)
    freqs = (torch.rand(200, 3) - 0.5) * 0.9
    weights = torch.randn(200, dtype=torch.complex64)

    def grad() -> torch.Tensor:
        grid.grad = None
        (nufft.sample_transform(grid, freqs) * weights).real.sum().backward()
        assert grid.grad is not None
        return grid.grad.clone()

    exact = grad()
    with nufft.adjoint_tolerance(1e-2):
        loose = grad()
    assert nufft._ADJOINT_EPS is None
    after = grad()
    loose_error = float((loose - exact).norm() / exact.norm())
    # FINUFFT's multithreaded spreading is not bit-reproducible, so "the
    # default again" means far closer than the loosened transform, not equal.
    assert 0 < loose_error < 5e-2
    assert float((after - exact).norm() / exact.norm()) < 0.01 * loose_error


@pytest.mark.parametrize("symmetry", [None, "C4"])
def test_linear_solve_reaches_the_dense_minimum(symmetry: str | None) -> None:
    """The ctf model is affine in V, so the regularised problem is a linear
    least-squares one whose minimum a dense solve finds independently.

    Convergence is judged by the objective, the quantity minimised: with 24
    small images some directions are nearly flat, so two maps a hair apart
    in objective can differ visibly, and the dense solve's minimum-norm
    answer is no more right than the solver's.

    With C4 on an even box the exact point-group average is not idempotent
    (a quarter turn does not map the grid onto itself), as it is not for a
    box-filling capsid, so the problem is the one the solver poses: minimise
    over U with the map S(U).
    """
    model, images, truth = _problem("ctf", symmetry=symmetry)
    solver = ConvergedSolver(
        model, images, np.linspace(4.0, 0.05, N // 2 + 1), line_search_fraction=1.0
    )
    result = solver.solve(_start(truth), n_iter=40)

    idx = torch.arange(len(images))
    n3 = N**3

    def predict(v: torch.Tensor) -> torch.Tensor:
        return solver._predict(idx, v.reshape(N, N, N)).flatten()

    zero = torch.zeros(n3)
    c = predict(zero).detach().double()
    J = torch.autograd.functional.jacobian(predict, zero).double()
    basis = torch.eye(n3).reshape(n3, N, N, N)
    lam = solver.prior_weight.cpu()
    L = torch.fft.irfftn(
        torch.fft.rfftn(basis, dim=(-3, -2, -1)) * lam, s=(N, N, N), dim=(-3, -2, -1)
    )
    L = L.reshape(n3, n3).double()
    # Column j of S is S applied to basis vector j. The maps S reaches are
    # spanned by its leading singular vectors (S is a projector here, to
    # 1e-6); solving in that orthonormal basis keeps the reference inside
    # them, where a least-squares solve over U would amplify S's rounding
    # along its null space.
    S = torch.stack([solver._project(b.clone()).flatten() for b in basis]).double().T
    U_, sv, _ = torch.linalg.svd(S)
    B = U_[:, sv > 0.5]
    lhs = B.T @ (J.T @ J + L) @ B
    rhs = B.T @ (J.T @ (images.flatten().double() - c))
    reference = B @ torch.linalg.solve(lhs, rhs)

    images_flat = images.flatten().double()

    def objective(v: torch.Tensor) -> float:
        residual = J @ v + c - images_flat
        return 0.5 * float(residual @ residual) + 0.5 * float(v @ L @ v)

    start = objective(S @ _start_flat(truth))
    gap = objective(result.volume.flatten().double()) - objective(reference)
    assert gap / (start - objective(reference)) < 1e-5, gap
    # The objective never rises along the way.
    objectives = [e["objective"] for e in result.log]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(objectives, objectives[1:]))


@pytest.mark.parametrize("symmetry", [None, "I1"])
def test_rytov_solve_lowers_the_objective(symmetry: str | None) -> None:
    """Every iteration lowers the objective. Under I1 the start carries noise
    in the box corners, which no icosahedral rotation maps onto the grid, so
    the point-group average is not idempotent there: the case where
    projecting the iterate instead of solving for V = S(U) climbed."""
    model, images, truth = _problem("rytov", symmetry=symmetry)
    solver = ConvergedSolver(model, images, np.linspace(4.0, 0.05, N // 2 + 1))
    result = solver.solve(_start(truth), n_iter=4)
    objectives = [e["objective"] for e in result.log]
    assert objectives[-1] < objectives[0]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(objectives, objectives[1:]))
    assert result.unfiltered.shape == (N, N, N)


def test_absorption_map_is_recovered() -> None:
    """Images of a complex potential V + iW: the joint solve lowers the
    objective every iteration and finds a W that correlates with the
    absorption the images were made with."""
    absorption = 0.8 * _blob(N, 1.5)
    model, images, truth = _problem("rytov", absorption=absorption, noise=0.05)
    solver = ConvergedSolver(
        model,
        images,
        np.linspace(4.0, 0.05, N // 2 + 1),
        line_search_fraction=1.0,
        absorption_resolution=2 * VOXEL,
    )
    result = solver.solve(_start(truth), n_iter=12)
    objectives = [e["objective"] for e in result.log]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(objectives, objectives[1:]))
    assert result.absorption is not None
    w = result.absorption.flatten()
    corr = float(torch.corrcoef(torch.stack([w, absorption.flatten()]))[0, 1])
    assert corr > 0.8, corr


def test_absorption_needs_zero_amplitude_contrast_and_the_surface_path() -> None:
    """W replaces alpha * V, and only the surface path samples a complex grid."""
    model, images, _ = _problem("rytov", absorption=0.1 * _blob(N, 1.5))
    model.propagation = dataclasses.replace(model.propagation, alpha=0.1)
    with pytest.raises(ValueError, match="alpha"):
        ConvergedSolver(model, images, np.ones(N // 2 + 1), absorption_resolution=8.0)
    model, images, _ = _problem("rytov")
    with pytest.raises(ValueError, match="surface"):
        ConvergedSolver(model, images, np.ones(N // 2 + 1), absorption_resolution=8.0)


def test_support_confines_the_map_and_still_lowers_the_objective() -> None:
    """With a support, the model sees support * S(U): nothing grows outside
    it, and the objective still falls every iteration."""
    model, images, truth = _problem("rytov")
    support = (_blob(N, 2.5) > 0.05).float()
    solver = ConvergedSolver(
        model, images, np.linspace(4.0, 0.05, N // 2 + 1), support=support
    )
    result = solver.solve(_start(truth), n_iter=4)
    assert float(result.volume[support == 0].abs().max()) == 0.0
    objectives = [e["objective"] for e in result.log]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(objectives, objectives[1:]))


def test_noise_weighting_measures_coloured_noise() -> None:
    """Images with low-pass (coloured) noise: the measured weighting rises
    with frequency, as 1 / sigma(k) should, and the weighted solve still
    lowers its objective every iteration."""
    model, images, truth = _problem("ctf", noise=0.0)
    torch.manual_seed(1)
    white = torch.randn_like(images)
    k = torch.fft.rfftfreq(N)[None, :] ** 2 + torch.fft.fftfreq(N)[:, None] ** 2
    coloured = torch.fft.irfft2(torch.fft.rfft2(white) / (1 + 40 * k), s=(N, N))
    images = images + images.std() * coloured
    solver = ConvergedSolver(
        model, images, np.linspace(4.0, 0.05, N // 2 + 1), noise_weighting=True
    )
    result = solver.solve(_start(truth), n_iter=4)
    sigma2 = np.array(result.info["noise_power"])
    assert sigma2[1] > 3 * sigma2[-1]
    objectives = [e["objective"] for e in result.log]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(objectives, objectives[1:]))


def test_a_pass_that_runs_out_of_memory_is_redone_with_smaller_batches() -> None:
    """Another process can take GPU memory mid-solve: the pass is redone with
    half the batch, and its result does not depend on the batch size."""
    model, images, truth = _problem("ctf")
    solver = ConvergedSolver(model, images, np.ones(N // 2 + 1), batchsize=8)
    V = solver._project(_start(truth))[None]
    expected, _ = solver._gradient(V)
    real_pass = solver._gradient_pass
    calls = {"n": 0}

    def flaky(X: torch.Tensor) -> tuple[torch.Tensor, float]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real_pass(X)

    solver._gradient_pass = flaky  # type: ignore[method-assign]
    grad, _ = solver._gradient(V)
    assert solver.batchsize == 4
    assert torch.allclose(grad, expected, rtol=1e-4, atol=1e-6)


def test_mask_check_flags_a_mask_that_correlates_the_halves_by_itself() -> None:
    """Halves sharing only a smooth blob plus independent noise: a sharp mask
    hugging the blob makes the masked FSC run past the noise-substituted one
    and is flagged; a loose soft sphere is not."""
    n = 32
    torch.manual_seed(0)
    signal = 10.0 * _blob(n, 2.5)
    a = signal + 0.3 * torch.randn(n, n, n)
    b = signal + 0.3 * torch.randn(n, n, n)
    tight = mask_check(a, b, (_blob(n, 3.0) > 0.3).float(), voxel_size=1.0)
    assert tight["too_tight"] and tight["shell_gap"] > 2
    loose = mask_check(a, b, spherical_mask(n), voxel_size=1.0)
    assert not loose["too_tight"]
