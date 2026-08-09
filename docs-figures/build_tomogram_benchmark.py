"""
Benchmark `specter build tomogram` at fixed physical field of view, across
three voxel sizes (10, 5, 2 A) -- the numbers behind the "Benchmarks"
section of docs/user-guide/build-tomogram.md.

Reuses quickstart.md's own example project (one spherical_harmonics
membrane, target species 1bxn x20, filler species 1mbo) so the benchmarked
workload is the same one already shown elsewhere in the docs, just resolved
at different voxel sizes. target_shape is scaled inversely with v_size to
hold the physical field of view fixed at quickstart's own (128, 256, 256)
voxels @ 5.0 A/voxel = (640, 1280, 1280) A.

Each resolution runs in its own subprocess (`--worker`), wrapped in
`/usr/bin/time -v`, so:
  - peak RAM (Maximum resident set size) is isolated per run, not
    contaminated by earlier runs' peaks still resident in the same process
  - peak GPU memory (torch.cuda.max_memory_allocated) starts from a clean
    CUDA context per run
  - wall time covers the actual `run_build_tomogram` call only; a forced
    CUDA context init (`torch.zeros(1, device=...)`) happens just before
    the timer starts (also required for reset_peak_memory_stats to work at
    all) so the one-time ~1-2s CUDA warm-up, identical across resolutions,
    doesn't dilute the resolution-to-resolution comparison

A single untimed warmup run (smallest shape) precedes the three timed runs,
so PDB fetch (network) is cached on disk before any measured run -- without
it, whichever resolution happened to run first would unfairly absorb the
one-time download cost.

Run with: uv run python docs-figures/build_tomogram_benchmark.py
Saves docs/assets/images/tomogram-benchmark-projections.png and prints a
markdown results table (hand-copied into build-tomogram.md, not included
live, matching this repo's other docs-figures/ scripts).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path("docs/assets/images")
SCRATCH_DIR = Path("docs-figures/data/tomogram_benchmark_scratch")
DEVICE = "cuda:3"  # idle GPU at benchmark time, see build-tomogram.md's hardware note

BASE_TARGET_SHAPE_ZYX = (128, 256, 256)  # quickstart.md's own example, voxels
BASE_V_SIZE = 5.0  # A/voxel
PHYSICAL_FOV_A = tuple(
    s * BASE_V_SIZE for s in BASE_TARGET_SHAPE_ZYX
)  # (640, 1280, 1280) A

V_SIZES = [10.0, 5.0, 2.0]


def target_shape_for(v_size: float) -> tuple[int, int, int]:
    return tuple(round(fov / v_size) for fov in PHYSICAL_FOV_A)  # type: ignore[return-value]


def _run_worker(v_size: float, out_mrc: Path, result_json: Path) -> None:
    """Runs inside the timed subprocess: build the config, run it, record
    wall time + GPU peak memory (RAM peak is measured by the parent via
    `/usr/bin/time -v` around this whole process instead)."""
    import torch

    from specter.config import TomogramConfig
    from specter.pipelines import run_build_tomogram

    shape_zyx = target_shape_for(v_size)
    cfg = TomogramConfig(
        targets=[{"pdb_source": "1bxn", "n_copies": 20}],
        filler=[{"pdb_source": "1mbo"}],
        membrane=[{"shape_backend": "spherical_harmonics"}],
        target_shape=list(shape_zyx),
        v_size=v_size,
        filler_occupancy_fraction=0.5,
        seed=42,
        device=DEVICE,
        accumulator_device="auto",
        render_workers="auto",
        write_picks=True,
        write_segmentation=True,
        output_dir=str(out_mrc.parent),
        filename=out_mrc.stem,
    )

    torch.zeros(1, device=DEVICE)  # force CUDA context init on this device
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    run_build_tomogram(cfg)
    elapsed_s = time.perf_counter() - start
    gpu_peak_bytes = torch.cuda.max_memory_allocated(DEVICE)

    result_json.write_text(
        json.dumps(
            {
                "v_size": v_size,
                "shape_zyx": list(shape_zyx),
                "n_voxels": shape_zyx[0] * shape_zyx[1] * shape_zyx[2],
                "elapsed_s": elapsed_s,
                "gpu_peak_bytes": gpu_peak_bytes,
            }
        )
    )


def _run_timed(v_size: float, tag: str) -> dict:
    """Spawns `_run_worker` as its own `/usr/bin/time -v`-wrapped
    subprocess and returns its result dict, plus parsed peak RSS."""
    out_mrc = SCRATCH_DIR / f"tomogram_{tag}.mrc"
    result_json = SCRATCH_DIR / f"result_{tag}.json"
    cmd = [
        "/usr/bin/time",
        "-v",
        sys.executable,
        __file__,
        "--worker",
        "--v_size",
        str(v_size),
        "--out_mrc",
        str(out_mrc),
        "--result_json",
        str(result_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (v_size={v_size}):\n{proc.stderr[-4000:]}")

    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    assert match is not None, f"couldn't parse /usr/bin/time output:\n{proc.stderr}"
    peak_rss_kb = int(match.group(1))

    result = json.loads(result_json.read_text())
    result["peak_rss_kb"] = peak_rss_kb
    result["mrc_path"] = str(out_mrc)
    return result


def main() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"warming up (PDB cache, CUDA context) at v_size={V_SIZES[0]} ...")
    _run_timed(V_SIZES[0], tag="warmup")

    results = []
    for v_size in V_SIZES:
        tag = f"v{v_size:g}"
        print(f"benchmarking v_size={v_size} A/voxel ...")
        result = _run_timed(v_size, tag=tag)
        print(
            f"  shape={result['shape_zyx']} time={result['elapsed_s']:.1f}s "
            f"gpu_peak={result['gpu_peak_bytes'] / 1e9:.2f} GB "
            f"ram_peak={result['peak_rss_kb'] / 1e6:.2f} GB"
        )
        results.append(result)

    _figure_projections(results)

    print(
        "\n| v_size (A/voxel) | shape (Z,Y,X voxels) | wall time | GPU peak | RAM peak |"
    )
    print("|---|---|---|---|---|")
    for r in results:
        shape_str = "x".join(str(s) for s in r["shape_zyx"])
        print(
            f"| {r['v_size']:g} | {shape_str} | {r['elapsed_s']:.0f} s | "
            f"{r['gpu_peak_bytes'] / 1e9:.2f} GB | {r['peak_rss_kb'] / 1e6:.2f} GB |"
        )


def _figure_projections(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mrcfile
    import numpy as np

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.6))
    for ax, r in zip(axes, results):
        with mrcfile.open(r["mrc_path"], permissive=True) as mrc:
            vol = np.asarray(mrc.data)
        projection = vol.max(axis=0)  # max-intensity Z projection
        ax.imshow(projection, cmap="gray", origin="lower")
        ax.set_title(
            f"v_size = {r['v_size']:g} Å\n{'x'.join(map(str, r['shape_zyx']))} voxels",
            fontsize=10,
        )
        ax.axis("off")
    fig.suptitle(
        "Same field of view (640 x 1280 x 1280 Å), three voxel sizes",
        fontsize=11,
        y=1.04,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = OUT_DIR / "tomogram-benchmark-projections.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--v_size", type=float)
    parser.add_argument("--out_mrc", type=Path)
    parser.add_argument("--result_json", type=Path)
    args = parser.parse_args()

    if args.worker:
        _run_worker(args.v_size, args.out_mrc, args.result_json)
    else:
        main()
