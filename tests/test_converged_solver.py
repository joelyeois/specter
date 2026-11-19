"""Tests for the converged reconstruction solver (`ghostbuster._converge`)."""

from __future__ import annotations


import numpy as np
import pytest
import roma
import torch

from specter import nufft
from specter.arrays import ball3d
from specter.ghostbuster import Reconstructor
from specter.ghostbuster._converge import _CPU_BATCHSIZE, ConvergedSolver
from specter.ghostbuster import _memory
from specter.ghostbuster._memory import SAFETY, _PeakMemory, fit_batchsize
from specter.ghostbuster._resolution import (
    FSC_CLIP,
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
    n: int = N,
) -> tuple[Reconstructor, torch.Tensor, torch.Tensor]:
    """A Reconstructor, its noisy images of a known map, and that map."""
    torch.manual_seed(0)
    truth = 3.0 * _blob(n, 1.5, 0.5) + 1.5 * _blob(n, 1.0, -1.5)
    ctf = {
        "dfu": 8000.0 + 2000.0 * torch.rand(n_particles),
        "dfv": 8000.0 + 2000.0 * torch.rand(n_particles),
        "dfang": 180.0 * torch.rand(n_particles),
        "cs": torch.full((n_particles,), 2.7),
        "phaseshift": torch.zeros(n_particles),
    }
    model = Reconstructor(
        V=torch.zeros(n, n, n),
        voxel_size=VOXEL,
        quaternions=roma.random_unitquat(n_particles),
        translations=torch.zeros(n_particles, 2),
        ctf_params=ctf,
        voltage=300.0,
        dose_per_angstrom=50.0,
        scale=0.8 + 0.4 * torch.rand(n_particles),
        propagation=Propagation(scattering_model=scattering_model),
        kmask=ball3d(n, n),
        lr=0.1,
        symmetry=symmetry,
    )
    solver = ConvergedSolver(model, torch.zeros(n_particles, n, n), np.ones(n // 2 + 1))
    with torch.no_grad():
        clean = solver._predict(torch.arange(n_particles), truth)
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


def test_curvature_probes_stay_inside_the_support() -> None:
    """A forward model that windows the map before projecting it cannot see a
    probe's energy outside the window, so whole-box probes read its curvature
    low; probes drawn inside the window read what an unwindowed model does."""
    model, images, truth = _problem("ctf")
    r = torch.arange(N, dtype=torch.float32) - N // 2
    radius = torch.sqrt(
        r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None] ** 2
    )
    window = (radius <= N / 2 - 1).float()
    ssnr = np.ones(N // 2 + 1)

    def windowed(solver: ConvergedSolver) -> ConvergedSolver:
        predict = solver._predict
        solver._predict = lambda idx, V: predict(idx, V * window)  # type: ignore[method-assign]
        return solver

    V = _start(truth)
    kw = {"n_bands": 2, "n_probes": 2, "n_probe_particles": 24, "low_shells": 1}
    plain, _ = ConvergedSolver(
        model, images, ssnr, probe_support=window
    ).measure_curvature(V, **kw)
    inside, _ = windowed(
        ConvergedSolver(model, images, ssnr, probe_support=window)
    ).measure_curvature(V, **kw)
    box, _ = windowed(
        ConvergedSolver(model, images, ssnr, probe_support=torch.ones(N, N, N))
    ).measure_curvature(V, **kw)
    np.testing.assert_allclose(inside, plain, rtol=1e-3)
    assert box.sum() < 0.8 * plain.sum()
    default = ConvergedSolver(model, images, ssnr)
    assert torch.equal(default.probe_support.cpu(), spherical_mask(N))
    with pytest.raises(ValueError, match="probe_support"):
        ConvergedSolver(model, images, ssnr, probe_support=torch.ones(N, N, N - 1))


def test_a_pass_that_runs_out_of_memory_is_redone_with_smaller_batches() -> None:
    """Another process can take GPU memory mid-solve: the pass is redone with
    half the batch, and its result does not depend on the batch size."""
    model, images, truth = _problem("ctf")
    solver = ConvergedSolver(model, images, np.ones(N // 2 + 1), batchsize=8)
    V = solver._project(_start(truth))
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
    # Batches sum in a different order: float32 rounding, ~1e-7 of the largest
    # element, which an absolute tolerance has to scale with.
    assert torch.allclose(
        grad, expected, rtol=1e-4, atol=1e-6 * float(expected.abs().max())
    )


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


def test_nufft_plans_do_not_accumulate_across_iterations() -> None:
    """Each batch size a pass meets caches its own NUFFT plans (the anisomag
    warp's are keyed by the batch, or each matrix group in it, padded to a
    power of two), so line-search subsets that split differently every
    iteration kept adding plans until cuFINUFFT ran out of device memory. An
    iteration starts from an empty cache, so the plans held stay bounded."""
    n_particles = 13
    model, images, truth = _problem("ctf", n_particles=n_particles)
    del model.anisomag
    model.register_buffer(
        "anisomag",
        torch.tensor([[1.004, 0.002], [0.002, 0.997]])
        .expand(n_particles, 2, 2)
        .clone(),
    )
    solver = ConvergedSolver(model, images, np.ones(N // 2 + 1), batchsize=4)
    solver.measure_curvature = lambda V: (np.ones(4), [1.0] * 5)  # type: ignore[method-assign]
    # Subsets of 3, 5, 6 and 7 particles: last batches of 3, 1, 2 and 3, which
    # need plans the full batches of 4 do not.
    sizes = iter([3, 5, 6, 7])
    solver._subset = lambda *a: torch.arange(next(sizes))  # type: ignore[method-assign]
    held: list[int] = []
    real_gradient = solver._gradient

    def gradient(V: torch.Tensor) -> tuple[torch.Tensor, float]:
        held.append(len(nufft._PLANS))
        return real_gradient(V)

    solver._gradient = gradient  # type: ignore[method-assign]
    solver.solve(_start(truth), 4)
    assert held == [0, 0, 0, 0]


def test_fit_batchsize_extrapolates_the_line_through_two_probes() -> None:
    """1 GiB at one particle and 1.5 at two: 0.5 GiB fixed plus 0.5 per
    particle, so 0.85 of 10 GiB free holds (8.5 - 0.5) / 0.5 = 16."""
    gib = 2.0**30
    p = fit_batchsize({1: 1.0 * gib, 2: 1.5 * gib}, free=10 * gib, maximum=100)
    assert p.batchsize == 16
    assert p.fixed == pytest.approx(0.5 * gib)
    assert p.per_particle == pytest.approx(0.5 * gib)
    assert p.predicted_peak(16) <= SAFETY * 10 * gib < p.predicted_peak(17)


def test_fit_batchsize_draws_the_line_through_the_two_largest_probes() -> None:
    """The first particle's step is unrepresentative; with a third probe the
    line runs through 2 and it: (5 - 1.5) / 14 = 0.25 GiB per particle."""
    gib = 2.0**30
    peaks = {1: 1.0 * gib, 2: 1.5 * gib, 16: 5.0 * gib}
    p = fit_batchsize(peaks, free=10 * gib, maximum=100)
    assert p.per_particle == pytest.approx(0.25 * gib)
    assert p.fixed == pytest.approx(1.0 * gib)
    assert p.batchsize == 30


def test_fit_batchsize_clamps_to_one_and_to_the_maximum() -> None:
    gib = 2.0**30
    peaks = {1: 1.0 * gib, 2: 1.5 * gib}
    assert fit_batchsize(peaks, free=1000 * gib, maximum=40).batchsize == 40
    # Not even one fits: one is still tried, and fails as it would have.
    assert fit_batchsize(peaks, free=0.5 * gib, maximum=40).batchsize == 1
    # A single particle needs no slope.
    assert fit_batchsize({1: 1.0 * gib}, free=10 * gib, maximum=1).batchsize == 1


def test_fit_batchsize_does_not_trust_a_slope_at_or_below_zero() -> None:
    """The second probe reading lower than the first is measurement noise;
    a line through it would allow any batch."""
    gib = 2.0**30
    p = fit_batchsize({1: 1.0 * gib, 2: 0.9 * gib}, free=10 * gib, maximum=10_000)
    assert p.per_particle > 0
    assert p.batchsize < 10_000
    # Two probes that measured nothing: no negative fixed cost to spend.
    assert fit_batchsize({1: 0.0, 2: 0.0}, free=1.0, maximum=8).batchsize == 1


def test_an_integer_batchsize_is_used_as_given() -> None:
    model, images, truth = _problem("ctf")
    solver = ConvergedSolver(model, images, np.ones(N // 2 + 1), batchsize=5)
    result = solver.solve(_start(truth), 1)
    assert solver.batchsize == 5 and result.info["batchsize"] == 5
    assert solver.batch_prediction is None
    for bad in (0, -3, 2.5, "big", True):
        with pytest.raises(ValueError, match="batchsize"):
            ConvergedSolver(model, images, np.ones(N // 2 + 1), batchsize=bad)  # type: ignore[arg-type]


def test_auto_batchsize_on_the_cpu_is_capped_by_the_particle_count() -> None:
    model, images, truth = _problem("ctf")
    solver = ConvergedSolver(model, images, np.ones(N // 2 + 1))
    solver.solve(_start(truth), 1)
    assert solver.batchsize == min(_CPU_BATCHSIZE, len(images))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("scattering_model", ["rytov", "firstborn", "multislice"])
def test_auto_batchsize_fits_a_small_device_and_does_not_change_the_map(
    scattering_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a device reporting little free memory, the predicted batch is a
    fraction of the particles, a three-iteration solve at it stays inside
    that memory, and the map equals one solved two particles at a time:
    batching only splits a sum. The free memory is capped in what the
    predictor reads rather than by filling the device, which would starve
    the suite's other workers; ``set_per_process_memory_fraction`` would cap
    PyTorch alone, not cuFINUFFT."""
    device = torch.device("cuda")
    n, n_particles = 96, 48
    model, images, truth = _problem(scattering_model, n_particles=n_particles, n=n)
    model = model.to(device)
    start = _start(truth).to(device)
    reference = ConvergedSolver(model, images, np.ones(n // 2 + 1), batchsize=2)
    expected = reference.solve(start, 3).volume

    budget = 2**30
    monkeypatch.setattr(_memory, "_free_bytes", lambda device: float(budget))
    solver = ConvergedSolver(model, images, np.ones(n // 2 + 1))
    nufft.clear_plan_cache()
    torch.cuda.empty_cache()
    with _PeakMemory(device) as meter:
        result = solver.solve(start, 3)
    prediction = solver.batch_prediction
    assert prediction is not None
    assert 2 <= solver.batchsize < n_particles
    assert meter.peak <= budget
    scale = float(expected.abs().max())
    assert torch.allclose(result.volume, expected, rtol=1e-3, atol=1e-4 * scale)
