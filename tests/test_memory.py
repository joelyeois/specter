from __future__ import annotations

import os
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

# A box small enough that `_SATURATION_PADDED_VOXELS` is not the binding
# constraint, so tests about the MEMORY bound and the `n_particles`/ceiling
# clamps still exercise those. At `_GEOM` one particle already saturates a GPU,
# which is the whole point of that cap -- see
# `test_recommend_batchsize_caps_at_gpu_saturation`.
# 40 rather than a round power of two so that
# `_PADDED_COPIES_PER_PARTICLE * nz * pad_nxy**2 * 4` lands on a whole number of
# bytes: `recommend_batchsize` keeps that product as a float while
# `estimate_peak_bytes` rounds it, and a sub-byte disagreement is enough to move
# a floor() by one.
_SMALL_NXY = 40
_SMALL_GEOM = (_SMALL_NXY, _SMALL_NXY, 2 * _SMALL_NXY)


def test_estimate_peak_grows_linearly_with_batchsize() -> None:
    """Per-particle cost is constant, so successive differences match."""
    peaks = [estimate_peak_bytes(b, *_GEOM) for b in (1, 2, 3, 4)]
    steps = [b - a for a, b in zip(peaks, peaks[1:])]
    assert all(s == pytest.approx(steps[0], rel=1e-6) for s in steps)
    assert peaks[0] < peaks[-1]


def test_estimate_peak_tracks_the_padded_box_not_num_pixels() -> None:
    """Turning off pad_fft quarters the per-particle term (pad_nxy^2), which
    is the whole reason a user can't guess this number from n_pixels."""
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
    per_particle = estimate_peak_bytes(1, *_SMALL_GEOM) - estimate_peak_bytes(
        0, *_SMALL_GEOM
    )
    overhead = estimate_peak_bytes(0, *_SMALL_GEOM)

    def budget_for(n_particles_worth: int) -> int:
        # Undo the CPU safety fraction so the target lands where we expect.
        return int((overhead + n_particles_worth * per_particle) / 0.5)

    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: budget_for(4))
    assert recommend_batchsize(*_SMALL_GEOM, "cpu") == 4
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: budget_for(8))
    assert recommend_batchsize(*_SMALL_GEOM, "cpu") == 8


def test_recommend_batchsize_never_below_one(monkeypatch) -> None:
    """A box too big for the device still returns 1 -- the run then fails
    honestly on a real allocation rather than on an estimate."""
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 1024)
    assert recommend_batchsize(1024, 1024, 2048, "cpu") == 1


def test_recommend_batchsize_clamps_to_n_particles_and_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 10**13)
    assert recommend_batchsize(*_SMALL_GEOM, "cpu", n_particles=3) == 3
    assert recommend_batchsize(*_SMALL_GEOM, "cpu") == MAX_AUTO_BATCHSIZE


def test_available_memory_bytes_cpu_is_positive() -> None:
    assert available_memory_bytes("cpu") > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_available_memory_bytes_cuda_is_below_total() -> None:
    free = available_memory_bytes("cuda:0")
    _, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    assert 0 < free <= total


def test_resolve_batchsize_passes_ints_through() -> None:
    assert resolve_batchsize(7, *_GEOM, "cpu") == 7
    assert resolve_batchsize("auto", *_SMALL_GEOM, "cpu", n_particles=2) == 2


def test_particle_stack_config_defaults_to_auto() -> None:
    assert ParticleStackConfig(pdb_source="6bdf").batchsize == "auto"


def test_config_batchsize_accepts_auto_and_int(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[potential]\npdb_source = "6bdf"\n\n[compute]\nbatchsize = "auto"\n'
    )
    assert load_config(str(path)).batchsize == "auto"
    path.write_text('[potential]\npdb_source = "6bdf"\n\n[compute]\nbatchsize = 3\n')
    assert load_config(str(path)).batchsize == 3


def test_bundled_particle_toml_uses_auto() -> None:
    from specter.config import REPO_ROOT

    config = load_config(str(REPO_ROOT / "configs" / "particle.toml"))
    assert config.batchsize == "auto"


@pytest.mark.parametrize(
    "conf,expected",
    [
        ("expandable_segments:True", True),
        ("expandable_segments:true", True),
        ("max_split_size_mb:128,expandable_segments:True", True),
        ("expandable_segments: True", True),
        ("expandable_segments:False", False),
        ("max_split_size_mb:128", False),
        ("", False),
    ],
)
def test_expandable_segments_detection(monkeypatch, conf, expected) -> None:
    """
    `PYTORCH_CUDA_ALLOC_CONF` is parsed leniently, and only ever loosens a bound.

    torch exposes no query for this, so it is read from the environment. A false
    negative costs a smaller batch and a false positive costs an OOM, hence the
    spellings that actually appear (case, whitespace, other keys alongside).
    """
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", conf)
    assert memory._expandable_segments_enabled() is expected


def test_recommend_batchsize_leaves_fragmentation_headroom_on_cuda(
    monkeypatch,
) -> None:
    """
    Without expandable segments, an auto batch is sized down for fragmentation.

    `estimate_peak_bytes` is fit against *allocated* bytes, but what has to fit
    on the card is what the allocator *reserves*. With the default segment
    allocator those diverge -- measured 1.29x of this model's own prediction at
    batch 30 -- which is how `batchsize="auto"` picked a batch that OOM'd on its
    second forward pass. The `specter` CLI sets the variable, so this guards the
    library caller who has not.

    Patched at `torch.device` level rather than run on a real GPU so it
    exercises on CPU-only machines.
    """
    # Sized so the MEMORY bound is what binds -- large enough to clear the fixed
    # overhead, small enough not to hit MAX_AUTO_BATCHSIZE or the saturation cap.
    per_particle = (
        memory._PADDED_COPIES_PER_PARTICLE
        * _SMALL_GEOM[1]
        * _SMALL_GEOM[2] ** 2
        * memory._BYTES_PER_ELEMENT
    )
    want = 4
    free = (estimate_peak_bytes(0, *_SMALL_GEOM) + want * per_particle) / 0.8
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: int(free))

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with_expandable = recommend_batchsize(*_SMALL_GEOM, "cuda")

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    without = recommend_batchsize(*_SMALL_GEOM, "cuda")

    assert with_expandable == want, "memory, not a ceiling, must be the binding bound"
    assert without < with_expandable, (
        "the default allocator must get a smaller batch than expandable segments"
    )


def test_fragmentation_headroom_does_not_apply_to_cpu(monkeypatch) -> None:
    """CPU has no CUDA caching allocator, so the headroom must not shrink it."""
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 40 * 10**9)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    on = recommend_batchsize(*_SMALL_GEOM, "cpu")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    off = recommend_batchsize(*_SMALL_GEOM, "cpu")
    assert on == off


def test_cli_sets_expandable_segments_without_clobbering_a_user_setting() -> None:
    """
    Importing the CLI turns expandable segments on, but never overrides a choice.

    The variable has to be set before the CUDA allocator initialises, so the CLI
    module does it at import; this pins both halves of that contract.
    """
    import subprocess
    import sys

    default = subprocess.run(
        [
            sys.executable,
            "-c",
            "import specter.cli._cli, os; print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert default.stdout.strip() == "expandable_segments:True"

    env_override = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}
    honored = subprocess.run(
        [
            sys.executable,
            "-c",
            "import specter.cli._cli, os; print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env_override,
    )
    assert honored.stdout.strip() == "max_split_size_mb:64"


def test_recommend_batchsize_caps_at_gpu_saturation(monkeypatch) -> None:
    """
    A big box gets a small batch even on an empty device.

    Batching amortizes per-call overhead only until one forward pass already
    saturates the device. Measured on an L40: at the default config's box 256
    (67.1M padded voxels per particle) throughput is flat to batch 16 and 24%
    *worse* at 32, while memory grows linearly -- 1.5 GB at batch 1 against
    30.8 GB at 32. Sizing purely to free memory therefore spends twenty times
    the memory for nothing, which is also what made `auto` pick a batch that
    OOM'd.

    Small boxes still get large batches, which is where batching earns its keep
    (2.2-2.5x at box 64).
    """
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # A device far larger than anything real, so only the saturation cap binds.
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 10**13)

    big_box = recommend_batchsize(256, 256, 512, "cuda")
    small_box = recommend_batchsize(64, 64, 128, "cuda")

    # Measured optima on an L40: 2 at box 256, 8 at box 64.
    assert big_box == 2, "a 67M-voxel-per-particle box must not batch up"
    assert small_box == MAX_AUTO_BATCHSIZE, "a small box should batch to the ceiling"


def test_saturation_cap_never_returns_zero(monkeypatch) -> None:
    """A box larger than the whole budget still yields 1, not 0."""
    monkeypatch.setattr(memory, "available_memory_bytes", lambda _d: 10**13)
    assert recommend_batchsize(2048, 2048, 4096, "cuda") == 1
