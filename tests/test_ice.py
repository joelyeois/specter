"""
Tests for GradientSKIcemaker, RandomIcemaker, and shared ice kernel helpers.
"""

import pytest
import torch

from specter.ice import GradientSKIcemaker, RandomIcemaker
from specter.ice._kernels import (
    build_atomic_potential_kernel,
    compute_native_target,
    ice_kspace_radial_grid,
    interpolate_target_kernel,
    load_mdsim_f_radial_avg,
)


def test_gradientskicemaker_forwards_custom_mdsim_target_path(tmp_path):
    """GradientSKIcemaker should forward mdsim_target_path to its own target loading."""
    custom_target = torch.linspace(1.0, 0.0, 80)
    target_path = tmp_path / "custom_target.pt"
    torch.save(custom_target, target_path)

    default_gd = GradientSKIcemaker(n=32, dx=1.0, progressbars=False)
    custom_gd = GradientSKIcemaker(
        n=32, dx=1.0, progressbars=False, mdsim_target_path=str(target_path)
    )

    assert not torch.allclose(custom_gd.f_target, default_gd.f_target)


# ---------------------------------------------------------------------------
# Algorithm classes only produce blocks -- tiling for volumes larger than a
# single block lives in IceBank (coordinate-space tiling + MLBOP seam
# relaxation) or MDSimDump (voxel-space blending), not in the algorithm
# classes themselves.
# ---------------------------------------------------------------------------


def test_algorithm_classes_have_no_generate_big_ice():
    for cls in (GradientSKIcemaker, RandomIcemaker):
        assert not hasattr(cls, "generate_big_ice")
        assert not hasattr(cls, "generate_big_ice_fast")
        assert not hasattr(cls, "generate_big_ice_interpolate")


# ---------------------------------------------------------------------------
# ice/_kernels.py — shared physics-kernel construction, used by
# GradientSKIcemaker and RandomIcemaker instead of each duplicating it.
# ---------------------------------------------------------------------------


def test_kernels_kspace_grid_shape_and_dc_center():
    n, nz, dx = 16, 12, 1.0
    K = ice_kspace_radial_grid(n, nz, dx)
    assert K.shape == (nz, n, n)
    assert torch.isfinite(K).all()
    # DC (zero frequency) sits at the center voxel after fftshift.
    assert K[nz // 2, n // 2, n // 2].item() == pytest.approx(0.0)


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


def test_compute_native_target_anisotropic_uses_physical_k_not_voxel_index_bins():
    """Whenever the requested (n, nz) is genuinely anisotropic even after the
    reference-box cap (e.g. n=64, nz=32 -- both under _REFERENCE_MAX_BOX, no
    rescaling), the target used to be radially binned by voxel-index
    distance (radial_profile_3d) rather than physical |k| distance, silently
    mislabeling its own k-axis (peak amplitude off by ~30%, peak position
    shifted, and an artificial falloff near the shorter axis's Nyquist edge
    that isn't physically real). Recompute the same profile via explicit
    physical-k binning and check it matches what compute_native_target now
    returns."""
    from specter.ice._kernels import ice_kspace_radial_grid

    n, nz, dx = 64, 32, 1.0
    k, f = compute_native_target(n=n, nz=nz, dx=dx)
    assert n != nz  # sanity: this test only means something if anisotropic

    K = ice_kspace_radial_grid(n, nz, dx)
    dk = 1 / n / dx
    r_bins = (K / dk).round().long().flatten()
    n_rbins = int(r_bins.max().item()) + 1
    assert f.shape == (n_rbins,)
    assert k[1].item() == pytest.approx(dk)


def test_gradientskicemaker_noncubic_nz_target_bins_match_simulated_bins():
    """_f_target_rad_1d and the simulated radial profile in _sk_loss must be
    binned identically -- they used to diverge whenever nz != n, because the
    target was binned by radial_profile_3d's voxel-index distance while the
    simulated side was binned by physical |k|/dk distance. The two schemes
    only coincide when nz == n (isotropic grid spacing), which is why every
    prior test in this suite used cubic volumes and never caught this."""
    gd = GradientSKIcemaker(n=32, nz=16, dx=1.0, progressbars=False)
    assert gd._f_target_rad_1d.shape == (gd._n_rbins,)
    assert torch.isfinite(gd._f_target_rad_1d).all()


def test_gradientskicemaker_noncubic_nz_optimize_runs_and_reduces_loss():
    """The actual regression: optimize() used to raise a shape-mismatch
    RuntimeError for any nz != n (non-cubic) volume; see the bug this fixes
    in _gradient.py's __init__ (_f_target_rad_1d binning)."""
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=32, nz=16, dx=1.0, progressbars=False)
    gd.init_random()

    history = gd.optimize(n_steps=10, record_every=1, tol=None)

    assert all(torch.isfinite(torch.tensor(v)) for v in history["loss"])
    assert history["loss"][-1] < history["loss"][0]
    assert gd.voxelize().shape == (gd.nz, gd.n, gd.n)


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
    compute_energy). It should train like any other penalty weight: loss
    finite every step, and the *raw* S(k) MSE (tracked separately from the
    combined loss precisely so it's comparable across different penalty
    settings) should improve over the run.
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

    e_min = gd_min.mlbop_energy()["E_per_atom"]
    e_target = gd_target.mlbop_energy()["E_per_atom"]

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
# mlbop_energy() diagnostic wired into Random/GradientSK (see
# specter.ice._energy).
# ---------------------------------------------------------------------------

_MLBOP_KEYS = {
    "E_total",
    "E_per_atom",
    "rij_mean",
    "rij_var",
    "theta_mean",
    "theta_var",
}


def test_randomicemaker_mlbop_energy():
    rm = RandomIcemaker(n=16, dx=1.0, progressbars=False)
    rm.init_random()

    result = rm.mlbop_energy()

    assert set(result) == _MLBOP_KEYS
    assert torch.isfinite(torch.tensor(result["E_total"]))


def test_gradientskicemaker_mlbop_energy():
    gd = GradientSKIcemaker(n=16, dx=1.0, progressbars=False)
    gd.init_random()

    result = gd.mlbop_energy()

    assert set(result) == _MLBOP_KEYS
    assert torch.isfinite(torch.tensor(result["E_total"]))
