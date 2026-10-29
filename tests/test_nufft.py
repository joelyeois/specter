"""
A stack of grids sampled at the same frequencies is one transform, not one per grid.

`sample_transform` sends a stack through cuFINUFFT a power-of-two number of grids per execute, padding the last
execute with zero grids; the values and the adjoint must be those of transforming every grid alone.
"""

from __future__ import annotations

import torch

from specter import nufft

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_a_stack_matches_its_grids_transformed_alone() -> None:
    """Five grids pad to an execute of eight; values and gradients are each grid's own."""
    g = torch.Generator().manual_seed(0)
    grids = (
        torch.randn(5, 24, 20, generator=g, dtype=torch.float64)
        .to(DEVICE)
        .requires_grad_(True)
    )
    freqs = (torch.rand(300, 2, generator=g, dtype=torch.float64) - 0.5).to(DEVICE)
    weights = torch.randn(5, 300, generator=g, dtype=torch.float64).to(DEVICE)

    stacked = nufft.sample_transform(grids, freqs, eps=1e-12)
    (stacked.real * weights).sum().backward()
    assert grids.grad is not None
    grad_stacked = grids.grad.clone()

    grids.grad = None
    alone = torch.stack(
        [nufft.sample_transform(grids[i], freqs, eps=1e-12) for i in range(5)]
    )
    (alone.real * weights).sum().backward()

    assert torch.allclose(stacked, alone, atol=1e-10)
    assert torch.allclose(grad_stacked, grids.grad, atol=1e-10)


def test_the_stack_size_is_a_capped_power_of_two() -> None:
    assert nufft._stack_size(1, (300, 300), False) == 1
    assert nufft._stack_size(5, (300, 300), False) == 8
    assert nufft._stack_size(128, (300, 300), False) == 128
    # A 300^3 volume's oversampled grid is ~0.4 GiB: no more than two per execute.
    assert nufft._stack_size(64, (300, 300, 300), False) <= 2
