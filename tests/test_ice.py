"""
Tests for IceBank.
"""

import warnings

import pytest
import torch

from specter.ice import (
    APIcemaker,
    GradientSKIcemaker,
    IceBank,
    MCMCIcemaker,
    RandomIcemaker,
)
from specter.ice._helpers import assemble_big_ice
from specter.ice._kernels import (
    build_atomic_potential_kernel,
    compute_native_target,
    ice_kspace_radial_grid,
    interpolate_target_kernel,
    load_mdsim_f_radial_avg,
)


def test_allocate_placeholder_shape_and_zeros():
    """
    allocate_placeholder() should produce a zero-filled bank of the same shape
    that build() would produce, without running any generation, so DDP ranks
    that skip the real build can still hold a buffer ready to receive
    Lightning's automatic module-state sync.
    """
    bank = IceBank(dx=1.0, n=16, method="random", num_unique=3)
    bank.allocate_placeholder()

    assert bank._bank is not None
    assert bank._bank.shape == (3, 16, 16, 16)
    assert torch.all(bank._bank == 0)


def test_allocate_placeholder_matches_build_shape():
    """allocate_placeholder() and build() should agree on bank shape."""
    placeholder_bank = IceBank(dx=1.0, n=16, method="random", num_unique=2)
    placeholder_bank.allocate_placeholder()

    built_bank = IceBank(dx=1.0, n=16, method="random", num_unique=2)
    built_bank.build()

    assert placeholder_bank._bank.shape == built_bank._bank.shape


def test_apicemaker_custom_mdsim_target_path_overrides_default(tmp_path):
    """A custom mdsim_target_path should replace the bundled default kernel."""
    custom_target = torch.linspace(1.0, 0.0, 80)
    target_path = tmp_path / "custom_target.pt"
    torch.save(custom_target, target_path)

    default_im = APIcemaker(n=32, dx=1.0, progressbars=False)
    custom_im = APIcemaker(
        n=32, dx=1.0, progressbars=False, mdsim_target_path=str(target_path)
    )

    assert torch.allclose(custom_im.mdsim_f_radial_avg, custom_target)
    assert not torch.allclose(custom_im.interp_f_kernel, default_im.interp_f_kernel)


def test_gradientskicemaker_forwards_custom_mdsim_target_path(tmp_path):
    """GradientSKIcemaker should forward mdsim_target_path to its internal APIcemaker."""
    custom_target = torch.linspace(1.0, 0.0, 80)
    target_path = tmp_path / "custom_target.pt"
    torch.save(custom_target, target_path)

    default_gd = GradientSKIcemaker(n=32, dx=1.0, progressbars=False)
    custom_gd = GradientSKIcemaker(
        n=32, dx=1.0, progressbars=False, mdsim_target_path=str(target_path)
    )

    assert not torch.allclose(custom_gd.f_target, default_gd.f_target)


# ---------------------------------------------------------------------------
# assemble_big_ice — the single shared tiling entry point, used solely by
# IceBank (every algorithm class only produces blocks; nothing needs a
# resolution-mismatched "generate at algorithm_dx, interpolate to dx" path
# now that method='gd' works fine at coarse pixel sizes).
# ---------------------------------------------------------------------------


def test_assemble_big_ice_shape_and_finite():
    torch.manual_seed(0)
    cubes = torch.rand(4, 8, 8, 8)
    out = assemble_big_ice(cubes, (2, 20, 20, 20))
    assert out.shape == (2, 20, 20, 20)
    assert torch.isfinite(out).all()


def test_assemble_big_ice_replace_faces_false_skips_face_replacement():
    torch.manual_seed(1)
    cubes = torch.rand(4, 8, 8, 8)
    out = assemble_big_ice(cubes, (1, 8, 8, 8), replace_faces=False)
    assert out.shape == (1, 8, 8, 8)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Algorithm classes only produce blocks now — tiling lives solely in IceBank
# (via assemble_big_ice). No algorithm class assembles large volumes itself.
# ---------------------------------------------------------------------------


def test_algorithm_classes_have_no_generate_big_ice():
    from specter.ice import RandomIcemaker

    for cls in (GradientSKIcemaker, MCMCIcemaker, APIcemaker, RandomIcemaker):
        assert not hasattr(cls, "generate_big_ice")
        assert not hasattr(cls, "generate_big_ice_fast")
        assert not hasattr(cls, "generate_big_ice_interpolate")


def test_apicemaker_warns_above_resolution_limit():
    # dx=1.6 (just above the 1.5 Å limit) with the default min_distance=1.9
    # keeps min_distance_vox = int(1.9/1.6) = 1, avoiding a separate,
    # pre-existing ZeroDivisionError in generate_ice_deltas' correction_factor
    # calc when min_distance_vox rounds all the way down to 0 (e.g. dx=3.0).
    im = APIcemaker(n=32, dx=1.6, progressbars=False)
    with pytest.warns(UserWarning, match="1.5"):
        im.generate_ice_deltas(batchsize=1, niter=1)


def test_apicemaker_no_warning_below_resolution_limit():
    im = APIcemaker(n=32, dx=1.0, progressbars=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im.generate_ice_deltas(batchsize=1, niter=1)


def test_mcmcicemaker_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        MCMCIcemaker(n=16, dx=1.0, progressbars=False)


# ---------------------------------------------------------------------------
# ice/_kernels.py — shared physics-kernel construction, used by APIcemaker,
# GradientSKIcemaker, and RandomIcemaker instead of each duplicating it.
# ---------------------------------------------------------------------------


def test_kernels_atomic_potential_matches_apicemaker_wrapper():
    dx = 1.0
    direct = build_atomic_potential_kernel(dx, "kirkland")
    im = APIcemaker(n=32, dx=dx, progressbars=False)
    assert torch.allclose(direct, im.create_ice_kernel())


def test_kernels_kspace_grid_shape_and_dc_center():
    n, nz, dx = 16, 12, 1.0
    K = ice_kspace_radial_grid(n, nz, dx)
    assert K.shape == (nz, n, n)
    assert torch.isfinite(K).all()
    # DC (zero frequency) sits at the center voxel after fftshift.
    assert K[nz // 2, n // 2, n // 2].item() == pytest.approx(0.0)


def test_kernels_interpolate_target_half_matches_full_slice_and_flip():
    im = APIcemaker(n=16, dx=1.0, progressbars=False)
    full = interpolate_target_kernel(
        im.K, im.mdsim_radial_k, im.mdsim_f_radial_avg, im.n_ice_molecules
    )
    half = interpolate_target_kernel(
        im.K,
        im.mdsim_radial_k,
        im.mdsim_f_radial_avg,
        im.n_ice_molecules,
        half=True,
    )
    n = full.shape[-1]
    expected_half = torch.flip(full[:, :, : n // 2 + 1], dims=[2])
    assert torch.allclose(half, expected_half)


def test_gradientskicemaker_does_not_import_apicemaker():
    import specter.ice._gradient as gradient_module

    assert "APIcemaker" not in dir(gradient_module)


def test_gradientskicemaker_ice_kernel_matches_direct_build():
    dx = 1.0
    gd = GradientSKIcemaker(n=16, dx=dx, progressbars=False)
    assert torch.allclose(gd._ice_kernel, build_atomic_potential_kernel(dx, "kirkland"))


# ---------------------------------------------------------------------------
# compute_native_target -- computes the S(k) target natively at the
# requested (n, dx) from a bundled real MD reference frame, instead of
# interpolating the fixed 400x400x400 dx=0.25 default across a mismatched
# dx (which was validated to converge poorly at coarse dx -- see
# GradientSKIcemaker's mdsim_target_path docstring). This is now
# GradientSKIcemaker's default whenever mdsim_target_path is not given.
# ---------------------------------------------------------------------------


def test_compute_native_target_shape_and_dc():
    k, f = compute_native_target(n=32, dx=1.0)
    assert k.shape == f.shape
    assert torch.isfinite(f).all()
    assert k[0].item() == pytest.approx(0.0)
    assert f[0].item() > 0  # DC term, ~sqrt(n_atoms) in the reference frame


def test_compute_native_target_is_deterministic():
    k1, f1 = compute_native_target(n=32, dx=1.0)
    k2, f2 = compute_native_target(n=32, dx=1.0)
    assert torch.equal(k1, k2)
    assert torch.equal(f1, f2)


def test_compute_native_target_differs_by_dx():
    """The whole point: a target computed at dx=0.5 must differ from one at
    dx=1.0, since it's meant to correctly reflect that grid's own
    discretization -- unlike interpolating a single fixed-dx reference."""
    _, f_dx05 = compute_native_target(n=64, dx=0.5)
    _, f_dx10 = compute_native_target(n=32, dx=1.0)
    assert not torch.allclose(f_dx05[:20], f_dx10[:20])


def test_compute_native_target_box_beyond_reference_still_works():
    """Box sizes beyond the real MD simulation's ~127A extent (e.g. n=256,
    dx=1.0 -> 256A) must not error -- the native computation caps at the
    reference's safe extent and relies on interpolate_target_kernel's own
    interp1d call to resample onto the requested grid's finer k-values."""
    k, f = compute_native_target(n=256, dx=1.0)
    assert torch.isfinite(f).all()
    # capped internally to the same dx -- matches the direct capped-box call
    k_capped, f_capped = compute_native_target(n=100, dx=1.0)
    assert torch.equal(f, f_capped)
    assert torch.equal(k, k_capped)


def test_compute_native_target_noncubic_nz():
    k, f = compute_native_target(n=32, nz=16, dx=1.0)
    assert torch.isfinite(f).all()
    k_cubic, f_cubic = compute_native_target(n=32, dx=1.0)
    assert not torch.equal(f, f_cubic)


def test_gradientskicemaker_default_target_is_native_not_bundled_fixed_grid():
    """Default (no mdsim_target_path) must use compute_native_target at this
    instance's own (n, dx), not the fixed bundled 400x400x400 dx=0.25 file."""
    dx = 1.0
    gd = GradientSKIcemaker(n=32, dx=dx, progressbars=False)
    expected_k, expected_f = compute_native_target(n=32, dx=dx)
    bundled_k, bundled_f = load_mdsim_f_radial_avg()

    K = ice_kspace_radial_grid(gd.n, gd.nz, gd.dx)
    expected_kernel = interpolate_target_kernel(
        K, expected_k, expected_f, gd.n_molecules
    ).float()
    bundled_kernel = interpolate_target_kernel(
        K, bundled_k, bundled_f, gd.n_molecules
    ).float()

    assert torch.allclose(gd.f_target, expected_kernel)
    assert not torch.allclose(gd.f_target, bundled_kernel)


# ---------------------------------------------------------------------------
# optimize() early stopping -- tracks the differentiable S(k) loss (cheap,
# already computed every step), not the ML-BOP energy (pure-Python per-atom
# diagnostic, too slow for per-step monitoring; see test_ice_energy.py /
# the manuscript notebook for ML-BOP used as an offline quality check).
# ---------------------------------------------------------------------------


def test_gradientskicemaker_optimize_stops_early_when_converged():
    """Uses the old cheap geometric-only loss (rep_strength=1.0,
    mlbop_strength=0.0) rather than the current mlbop_strength=0.5 default,
    since this test is about the tol/patience control flow itself, not
    which loss recipe is default -- the heavier default loss doesn't
    plateau this tightly within so few steps at this tiny scale."""
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd.init_random()

    history = gd.optimize(
        n_steps=60,
        record_every=1,
        rep_strength=1.0,
        mlbop_strength=0.0,
        mlbop_target=None,
        tol=1e-2,
        patience=3,
    )

    assert history["stopped_early"] is True
    assert history["step"][-1] < 59


def test_gradientskicemaker_optimize_runs_full_steps_when_tol_none():
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd.init_random()

    history = gd.optimize(n_steps=15, record_every=1, tol=None)

    assert history["stopped_early"] is False
    assert history["step"][-1] == 14


def test_gradientskicemaker_optimize_mlbop_strength_reduces_sk_loss():
    """
    mlbop_strength replaces the artificial FFT repulsion term with a
    differentiable ML-BOP energy penalty (see specter.ice._energy.MLBOP.
    compute_energy_differentiable). It should train like any other penalty
    weight: loss finite every step, and the *raw* S(k) MSE (tracked
    separately from the combined loss precisely so it's comparable across
    different penalty settings) should improve over the run.
    """
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd.init_random()

    history = gd.optimize(
        n_steps=10,
        record_every=1,
        rep_strength=0.0,
        mlbop_strength=0.05,
        tol=None,
    )

    assert all(torch.isfinite(torch.tensor(v)) for v in history["loss"])
    assert all(torch.isfinite(torch.tensor(v)) for v in history["sk_loss"])
    assert history["sk_loss"][-1] < history["sk_loss"][0]


def test_gradientskicemaker_mlbop_target_matches_target_not_unbounded_minimize():
    """
    Amorphous ice phases are metastable, not energy-minimal (LDA ice in
    particular is a compressed phase whose ML-BOP energy sits measurably
    above the true minimum -- see the real LDA-80K MD reference,
    ~-0.41 eV/atom). mlbop_target turns the penalty into
    (E_per_atom - target)**2 instead of minimizing E_per_atom directly, so
    the optimizer should end up *closer* to a deliberately-high target than
    unconstrained minimization would, not blow past it toward a lower,
    more crystalline-like energy.
    """
    torch.manual_seed(0)
    gd_min = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd_min.init_random()
    init_positions = gd_min.positions.clone()

    gd_target = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd_target.positions = init_positions.clone()

    # Deliberately above where unconstrained minimization reaches in this
    # many steps (empirically ~0.7-2 eV/atom at mlbop_strength=0.05 here).
    target = 5.0

    gd_min.optimize(
        n_steps=15, record_every=15, rep_strength=0.0, mlbop_strength=0.05, tol=None
    )
    gd_target.optimize(
        n_steps=15,
        record_every=15,
        rep_strength=0.0,
        mlbop_strength=0.05,
        mlbop_target=target,
        tol=None,
    )

    e_min = gd_min.mlbop_energy(progressbar=False)["E_per_atom"]
    e_target = gd_target.mlbop_energy(progressbar=False)["E_per_atom"]

    assert abs(e_target - target) < abs(e_min - target)


def test_gradientskicemaker_optimize_records_final_step_on_early_stop():
    """Even with a sparse record_every, the step that triggers early
    stopping must still land in history -- otherwise the caller can't see
    the converged state that early stopping actually converged to.

    Uses the old cheap geometric-only loss (see the similar note on
    test_gradientskicemaker_optimize_stops_early_when_converged) since this
    is a control-flow test, not one about the mlbop_strength=0.5 default."""
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd.init_random()

    history = gd.optimize(
        n_steps=60,
        record_every=1000,
        rep_strength=1.0,
        mlbop_strength=0.0,
        mlbop_target=None,
        tol=1e-2,
        patience=3,
    )

    assert history["stopped_early"] is True
    # record_every=1000 only naturally records step 0; the stopping step
    # must be appended on top of that, not silently dropped.
    assert len(history["step"]) == 2
    assert history["step"][0] == 0
    assert history["step"][1] > 0
    assert (
        len(history["step"]) == len(history["loss"]) == len(history["radial_profile"])
    )


# ---------------------------------------------------------------------------
# mlbop_energy() diagnostic wired into AP/Random/GradientSK (see
# specter.ice._energy) -- MCMCIcemaker is deprecated and intentionally
# excluded.
# ---------------------------------------------------------------------------

_MLBOP_KEYS = {
    "E_total",
    "E_per_atom",
    "rij_mean",
    "rij_var",
    "theta_mean",
    "theta_var",
}


def test_apicemaker_mlbop_energy_returns_one_result_per_batch_element():
    im = APIcemaker(n=16, dx=1.0, progressbars=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im.generate_ice_deltas(batchsize=2, niter=1)

    results = im.mlbop_energy(progressbar=False)

    assert len(results) == 2
    for result in results:
        assert set(result) == _MLBOP_KEYS
        assert torch.isfinite(torch.tensor(result["E_total"]))


def test_apicemaker_mlbop_energy_requires_generated_coordinates():
    im = APIcemaker(n=16, dx=1.0, progressbars=False)
    with pytest.raises(RuntimeError, match="No ice coordinates"):
        im.mlbop_energy(progressbar=False)


def test_randomicemaker_mlbop_energy():
    rm = RandomIcemaker(n=16, dx=1.0, progressbars=False)
    rm.init_random()

    result = rm.mlbop_energy(progressbar=False)

    assert set(result) == _MLBOP_KEYS
    assert torch.isfinite(torch.tensor(result["E_total"]))


def test_gradientskicemaker_mlbop_energy():
    gd = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd.init_random()

    result = gd.mlbop_energy(progressbar=False)

    assert set(result) == _MLBOP_KEYS
    assert torch.isfinite(torch.tensor(result["E_total"]))
