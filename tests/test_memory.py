from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter import memory
from specter.config import ParticleStackConfig, load_config
from specter.memory import (
    MAX_AUTO_BATCHSIZE,
    available_memory_bytes,
    estimate_peak_bytes,
    recommend_batchsize,
    resolve_batchsize,
)

# A default-shaped run: 256 box, pad_fft=True (pad_nxy = 2*nxy), nz = nxy.
_NXY = 256
_GEOM = (_NXY, _NXY, 2 * _NXY)  # nxy, nz, pad_nxy


def test_estimate_peak_grows_linearly_with_batchsize() -> None:
    """Per-particle cost is constant, so successive differences match."""
    peaks = [estimate_peak_bytes(b, *_GEOM) for b in (1, 2, 3, 4)]
    steps = [b - a for a, b in zip(peaks, peaks[1:])]
    assert all(s == pytest.approx(steps[0], rel=1e-6) for s in steps)
    assert peaks[0] < peaks[-1]


def test_estimate_peak_tracks_the_padded_box_not_num_pixels() -> None:
    """Turning off pad_fft quarters the per-particle term (pad_nxy^2), which
    is the whole reason a user can't guess this number from num_pixels."""
    padded = estimate_peak_bytes(4, _NXY, _NXY, 2 * _NXY)
    unpadded = estimate_peak_bytes(4, _NXY, _NXY, _NXY)
    overhead = estimate_peak_bytes(0, _NXY, _NXY, _NXY)
    assert (padded - overhead) == pytest.approx(4 * (unpadded - overhead), rel=1e-6)


def test_estimate_peak_matches_measured_l40_sweep() -> None:
    """Regression guard on the fitted constants: the estimate must stay above
    every measured peak in the sweep the module docstring documents, and not
    wander more than 25% above it (which would start wasting real memory).

    Measured with torch.cuda.max_memory_allocated on an NVIDIA L40, default
    multislice + IceBank + pad_fft path -- see src/specter/memory.py.
    """
    measured_bytes = {
        # (nxy, batchsize): measured peak
        (128, 1): 525_785_088,
        (128, 8): 1_292_624_896,
        (256, 1): 1_841_214_976,
        (256, 4): 5_246_796_800,
        (384, 1): 6_200_550_912,
        (384, 2): 9_906_527_232,
        (512, 3): 32_573_034_496,
    }
    for (nxy, b), measured in measured_bytes.items():
        predicted = estimate_peak_bytes(b, nxy, nxy, 2 * nxy)
        assert predicted >= measured, f"under-estimate at nxy={nxy}, B={b}"
        assert predicted <= 1.25 * measured, f"over-estimate at nxy={nxy}, B={b}"


def test_recommend_batchsize_scales_with_free_memory(monkeypatch) -> None:
    """Twice the memory, ~twice the batch (the fixed overhead makes it
    slightly more than twice, never less)."""
    per_particle = estimate_peak_bytes(1, *_GEOM) - estimate_peak_bytes(0, *_GEOM)
    overhead = estimate_peak_bytes(0, *_GEOM)

    def budget_for(n_particles_worth: int) -> int:
        # Undo the CPU safety fraction so the target lands where we expect.
        return int((overhead + n_particles_worth * per_particle) / 0.5)

    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: budget_for(4))
    assert recommend_batchsize(*_GEOM, "cpu") == 4
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: budget_for(8))
    assert recommend_batchsize(*_GEOM, "cpu") == 8


def test_recommend_batchsize_never_below_one(monkeypatch) -> None:
    """A box too big for the device still returns 1 -- the run then fails
    honestly on a real allocation rather than on an estimate."""
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 1024)
    assert recommend_batchsize(1024, 1024, 2048, "cpu") == 1


def test_recommend_batchsize_clamps_to_n_particles_and_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 10**13)
    assert recommend_batchsize(*_GEOM, "cpu", n_particles=3) == 3
    assert recommend_batchsize(*_GEOM, "cpu") == MAX_AUTO_BATCHSIZE


def test_available_memory_bytes_cpu_is_positive() -> None:
    assert available_memory_bytes("cpu") > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_available_memory_bytes_cuda_is_below_total() -> None:
    free = available_memory_bytes("cuda:0")
    _, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    assert 0 < free <= total


def test_resolve_batchsize_passes_ints_through() -> None:
    assert resolve_batchsize(7, *_GEOM, "cpu") == 7
    assert resolve_batchsize("auto", *_GEOM, "cpu", n_particles=2) == 2


def test_particle_stack_config_defaults_to_auto() -> None:
    assert ParticleStackConfig(pdb_code="6bdf").batchsize == "auto"


def test_config_batchsize_accepts_auto_and_int(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[potential]\npdb_code = "6bdf"\n\n[compute]\nbatchsize = "auto"\n')
    assert load_config(str(path)).batchsize == "auto"
    path.write_text('[potential]\npdb_code = "6bdf"\n\n[compute]\nbatchsize = 3\n')
    assert load_config(str(path)).batchsize == 3


def test_bundled_particle_toml_uses_auto() -> None:
    from specter.config import REPO_ROOT

    config = load_config(str(REPO_ROOT / "configs" / "particle.toml"))
    assert config.batchsize == "auto"
