"""
A stack of grids sampled at the same frequencies is one transform, not one per grid.

`sample_transform` sends a stack through cuFINUFFT a power-of-two number of grids per execute, padding the last
execute with zero grids; the values and the adjoint must be those of transforming every grid alone.
"""

from __future__ import annotations

import pytest
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


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def fresh_plans():
    nufft.clear_plan_cache()
    yield
    nufft.clear_plan_cache()


@needs_cuda
def test_a_refused_spreading_method_falls_back_to_the_librarys(
    monkeypatch, fresh_plans, caplog
) -> None:
    """A pinned gpu_method the GPU cannot run is replaced, not reported as out of memory."""
    import cufinufft

    real = cufinufft.Plan

    def plan(*args, **kwargs):
        if kwargs.get("gpu_method") == 3:
            raise RuntimeError("Error creating plan.")
        return real(*args, **kwargs)

    monkeypatch.setattr(cufinufft, "Plan", plan)
    with caplog.at_level("WARNING", logger="specter"):
        made, gpu = nufft._cached_plan(
            1, (16, 16, 16), True, torch.device("cuda"), 1e-9, gpu_method=3
        )
    assert gpu and made is not None
    assert "gpu_method=3" in caplog.text


@needs_cuda
def test_a_plan_no_method_can_make_is_out_of_memory(monkeypatch, fresh_plans) -> None:
    import cufinufft

    def plan(*args, **kwargs):
        raise RuntimeError("Error creating plan.")

    monkeypatch.setattr(cufinufft, "Plan", plan)
    with pytest.raises(torch.OutOfMemoryError, match="type-1 plan"):
        nufft._cached_plan(
            1, (16, 16, 16), True, torch.device("cuda"), 1e-9, gpu_method=3
        )


@needs_cuda
def test_double_precision_exact_rotation_at_the_smaller_upsampling(
    monkeypatch, fresh_plans
) -> None:
    """eps 1e-9 in float64 at upsampfac 1.25: block-gather spreading needs 134 kB of
    shared memory there, more than an L40 has; the rotation still runs, and its
    backward is its adjoint."""
    import roma

    from specter.rotations import _exact, build_affine_matrix, rotate_volume_exact

    monkeypatch.setattr(_exact, "_UPSAMPFAC_2_FRACTION", 0.0)
    _exact._grid.cache_clear()
    g = torch.Generator().manual_seed(3)
    V = torch.randn(32, 32, 32, dtype=torch.float64, generator=g).cuda()
    W = torch.randn(2, 32, 32, 32, dtype=torch.float64, generator=g).cuda()
    R = roma.random_rotmat(2, generator=g).double()
    theta = build_affine_matrix(R, torch.zeros(2, 3, dtype=torch.float64)).cuda()
    V.requires_grad_(True)
    with pytest.warns(UserWarning, match="wraps"):
        out = rotate_volume_exact(V, theta, eps=1e-9)
    assert _exact._grid((32, 32, 32), V.device, True).upsampfac == 1.25
    (grad,) = torch.autograd.grad((out * W).sum(), V)
    lhs = float((out.detach() * W).sum())
    rhs = float((V.detach() * grad).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-7
    _exact._grid.cache_clear()
