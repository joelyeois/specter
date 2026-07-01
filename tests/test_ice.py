"""
Tests for IceBank.
"""

import torch

from specter.ice import IceBank


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
