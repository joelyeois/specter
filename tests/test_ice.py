"""
Tests for IceBank.
"""

import torch

from specter.ice import APIcemaker, GradientSKIcemaker, IceBank


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
