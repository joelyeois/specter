"""
Gradient accumulation in `Reconstructor` (``micro_batchsize``): a batch split
into chunks, each backpropagated with its loss weighted by its share of the
batch, takes the same optimiser step as the batch in one pass.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from specter.ghostbuster import Reconstructor, fit
from specter.settings import Propagation

NXY = 16
N_PARTICLES = 8
PIXEL_SIZE = 1.5
VOLTAGE = 300.0
DOSE = 22.5


def _problem(dtype: torch.dtype) -> dict[str, Any]:
    import roma

    # roma's samplers draw from the global stream, so seed that.
    torch.manual_seed(0)
    n = N_PARTICLES
    ctf = {
        "dfu": torch.empty(n).uniform_(9000, 11000),
        "dfv": torch.empty(n).uniform_(9000, 11000),
        "dfang": torch.empty(n).uniform_(0, 180),
        "cs": torch.full((n,), 2.7e7),
        "phaseshift": torch.zeros(n),
        "tiltx": torch.zeros(n),
        "tilty": torch.zeros(n),
    }
    return dict(
        V0=(torch.rand(NXY, NXY, NXY) * 0.02).to(dtype),
        quats=roma.random_unitquat(size=(n,)).to(dtype),
        trans=((torch.rand(n, 2) - 0.5) * 5).to(dtype),
        ctf={k: v.to(dtype) for k, v in ctf.items()},
        scale=torch.empty(n).uniform_(0.5, 1.5).to(dtype),
        images=(DOSE * PIXEL_SIZE**2 + torch.randn(n, NXY, NXY)).to(dtype),
    )


def _train(
    problem: dict[str, Any],
    *,
    batch_size: int = N_PARTICLES,
    device: Any = "cpu",
    precision: str = "32",
    refine: bool = False,
    **kwargs: Any,
) -> Reconstructor:
    """One epoch, in file order, on a fresh model."""
    lrs = dict(lr_R=1e-3, lr_T=1e-2, lr_D=10.0) if refine else {}
    model = Reconstructor(
        problem["V0"].clone(),
        PIXEL_SIZE,
        problem["quats"].clone(),
        problem["trans"].clone(),
        {k: v.clone() for k, v in problem["ctf"].items()},
        VOLTAGE,
        DOSE,
        scale=problem["scale"].clone(),
        lr=0.1,
        propagation=Propagation(alpha=0.07, scattering_model="rytov"),
        **lrs,
        **kwargs,
    )
    loader = DataLoader(
        TensorDataset(problem["images"], torch.arange(N_PARTICLES)),
        batch_size=batch_size,
        shuffle=False,
    )
    return fit(
        model, loader, max_epochs=1, device=device, precision=precision, progress=False
    )


def _refined(model: Reconstructor) -> dict[str, torch.Tensor]:
    """Every trained parameter and its last gradient, by name."""
    out = {}
    for name, p in model.named_parameters():
        out[name] = p.detach().clone()
        assert p.grad is not None, name
        out[f"{name}.grad"] = p.grad.detach().clone()
    return out


@pytest.fixture(scope="module")
def problem64() -> dict[str, Any]:
    return _problem(torch.float64)


@pytest.fixture(scope="module")
def whole_batch(problem64: dict[str, Any]) -> Reconstructor:
    return _train(problem64, refine=True)


@pytest.mark.parametrize("micro", [N_PARTICLES, N_PARTICLES // 2, 3, 1])
def test_split_step_matches_the_whole_batch(
    problem64: dict[str, Any], whole_batch: Reconstructor, micro: int
) -> None:
    """Volume, rotations, shifts and defocus: same gradients, same update,
    for an even split, an uneven one (3 + 3 + 2) and single particles."""
    split = _train(problem64, refine=True, micro_batchsize=micro)
    reference = _refined(whole_batch)
    got = _refined(split)
    assert got.keys() == reference.keys()
    assert len(reference) == 8  # V, rotations, translations, defocus_offset
    for name, value in reference.items():
        torch.testing.assert_close(got[name], value, rtol=1e-10, atol=1e-14, msg=name)
    # One step per optimiser, however many chunks.
    assert split.global_step == whole_batch.global_step == 4
    # One logged loss per batch, the batch's.
    assert len(split.log_total_loss) == 1
    torch.testing.assert_close(
        split.log_total_loss[0], whole_batch.log_total_loss[0], rtol=1e-12, atol=0
    )


def test_whole_batch_chunk_is_bitwise_the_unsplit_step(
    problem64: dict[str, Any], whole_batch: Reconstructor
) -> None:
    """A chunk as large as the batch is the unsplit step, bit for bit."""
    split = _train(problem64, refine=True, micro_batchsize=64)
    for name, value in _refined(whole_batch).items():
        assert torch.equal(_refined(split)[name], value), name


def test_several_batches_log_one_loss_each(problem64: dict[str, Any]) -> None:
    whole = _train(problem64, batch_size=4)
    split = _train(problem64, batch_size=4, micro_batchsize=3)
    assert len(split.log_total_loss) == len(whole.log_total_loss) == 2
    assert split.global_step == whole.global_step == 2
    torch.testing.assert_close(split.V, whole.V, rtol=1e-10, atol=1e-14)


def test_auto_halves_on_out_of_memory(
    problem64: dict[str, Any],
    whole_batch: Reconstructor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out of memory above 2 particles: 8 -> 4 -> 2, then the same step."""
    forward = Reconstructor.forward
    sizes: list[int] = []

    def tight(self: Reconstructor, idx: torch.Tensor) -> torch.Tensor:
        sizes.append(len(idx))
        if len(idx) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return forward(self, idx)

    monkeypatch.setattr(Reconstructor, "forward", tight)
    split = _train(problem64, refine=True, micro_batchsize="auto")
    assert sizes == [8, 4, 2, 2, 2, 2]
    assert split._auto_micro_batchsize == 2
    for name, value in _refined(whole_batch).items():
        torch.testing.assert_close(
            _refined(split)[name], value, rtol=1e-10, atol=1e-14, msg=name
        )


def test_fixed_size_does_not_swallow_out_of_memory(
    problem64: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def oom(self: Reconstructor, idx: torch.Tensor) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(Reconstructor, "forward", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _train(problem64, micro_batchsize=4)


@pytest.mark.parametrize("bad", [0, -2, 2.5, True, "half"])
def test_rejects_a_bad_micro_batchsize(problem64: dict[str, Any], bad: Any) -> None:
    with pytest.raises(ValueError, match="micro_batchsize"):
        Reconstructor(
            problem64["V0"],
            PIXEL_SIZE,
            problem64["quats"],
            problem64["trans"],
            problem64["ctf"],
            VOLTAGE,
            DOSE,
            micro_batchsize=bad,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_step_matches_under_16_mixed() -> None:
    """Every chunk scaled by the same GradScaler, which updates once per
    optimiser step: the same update to half-precision tolerance, the same
    scale afterwards."""
    problem = _problem(torch.float32)
    whole = _train(problem, device=0, precision="16-mixed", refine=True)
    split = _train(
        problem, device=0, precision="16-mixed", refine=True, micro_batchsize=3
    )
    assert whole._training is not None and split._training is not None
    assert whole._training.scaler is not None and split._training.scaler is not None
    assert split._training.scaler.get_scale() == whole._training.scaler.get_scale()
    assert split.global_step == whole.global_step == 4
    V0 = problem["V0"]
    rel = (split.V - whole.V).norm() / (whole.V - V0).norm()
    assert rel < 1e-3, f"16-mixed split moved V by {rel:.2e} of the step"
    for name in ("rotations", "translations", "defocus_offset"):
        a, b = getattr(split, name).detach(), getattr(whole, name).detach()
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("free", "chunk"), [(1.0, 1), (1e15, N_PARTICLES)])
def test_auto_predicts_the_chunk_from_the_free_memory(
    free: float, chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a device, "auto" probes a chunk of one and of two particles and
    takes the largest predicted to fit: one when almost nothing is free, the
    whole batch when plenty is. Either way the step is the unsplit one."""
    from specter.ghostbuster import _memory

    problem = _problem(torch.float32)
    whole = _train(problem, device=0)
    monkeypatch.setattr(_memory, "_free_bytes", lambda device: free)
    split = _train(problem, device=0, micro_batchsize="auto")
    assert split._auto_micro_batchsize == chunk
    torch.testing.assert_close(split.V, whole.V, rtol=1e-4, atol=1e-6)
