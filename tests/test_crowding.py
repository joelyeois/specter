from __future__ import annotations

import pytest
import torch

import specter
from specter.crowding import CrowdWithDuplicates

# Small enough to stay quick, large enough that the Poisson-disk sampling
# places several duplicates -- with fewer than two there is nothing for
# chunking to reorder and the invariance below would hold vacuously.
_GEOM = dict(dx=2.0, min_distance=30.0, nxy_out=192, nz_out=64)


def _crowd(chunk_size: int, device: str) -> torch.Tensor:
    specter.seed(0)
    V = torch.rand(32, 32, 32, device=device)
    crowd = CrowdWithDuplicates(V, chunk_size=chunk_size, progressbars=False, **_GEOM)
    specter.seed(0)
    return crowd()


def test_chunk_size_does_not_change_the_result_on_cpu() -> None:
    """
    `chunk_size` is a memory knob, and on the CPU it is exactly that.

    The chunk loop inserts duplicates in the same global order whatever the
    chunk size, so the only thing that can differ is the reduction order
    *inside* one batched insert. On the CPU that reduction is deterministic,
    so the result is bit-for-bit identical -- which is what lets the CLI QA
    sweep assert `expect="unchanged"` for this flag.
    """
    assert torch.equal(_crowd(1, "cpu"), _crowd(4, "cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunk_size_perturbs_only_at_rounding_level_on_gpu() -> None:
    """
    On the GPU the same knob is not bit-exact, but is bounded well below noise.

    A batched insert accumulates through atomics there, whose ordering is not
    fixed, so `chunk_size` perturbs the sum -- the constructor docstring quotes
    ~4e-6 relative. That is float-rounding, not a physics change, and this
    pins it as such: a regression that made chunking actually *move* density
    would blow through this bound rather than hide behind "GPU is nondeterministic".
    """
    a, b = _crowd(1, "cuda:0"), _crowd(4, "cuda:0")
    rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-30)
    assert rel < 1e-4, f"chunking perturbed the result by {rel:.2e} relative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_seeded_run_is_reproducible_on_gpu() -> None:
    """
    The atomics above must not leak into run-to-run reproducibility.

    Every simulate/build config defaults to `device="cuda"`, so `specter.seed`
    has to mean the same thing there as on the CPU. Holding `chunk_size` fixed,
    two seeded runs must agree exactly.
    """
    assert torch.equal(_crowd(1, "cuda:0"), _crowd(1, "cuda:0"))
