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
    ice_kspace_radial_grid,
    interpolate_target_kernel,
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
