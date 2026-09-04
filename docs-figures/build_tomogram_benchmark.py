"""
Benchmark `specter build tomogram` at fixed physical field of view, across
three voxel sizes (10, 5, 2 A) -- the numbers behind the "Benchmarks"
section of docs/user-guide/build-tomogram.md.

Reuses quickstart.md's own example project (one spherical_harmonics
membrane, target species 1bxn x20) plus filler_from_pei2016 (20 species,
matching configs/tomogram.toml's own filler approach) so the benchmarked
workload is the same one already shown elsewhere in the docs, just resolved
at different voxel sizes. target_shape is scaled inversely with voxel_size to
hold the physical field of view fixed at PHYSICAL_FOV_A below --
(1500, 6000, 6000) A, matching configs/tomogram.toml's own production-scale
demo box (Z stretched from 1000 to 1500 A). At voxel_size=2 this is ~6.75
billion voxels -- the regime render_chunk_size/accumulator_device="auto" exist
for, per the "Compute & scaling flags" section this benchmark backs up.

A single hand-picked filler species (e.g. 1mbo) at filler_occupancy_fraction
default's own scale was tried first and rejected: at this box size it packed
~109,000 instances of that ONE small species to hit occupancy -- both
atypical (real configs spread the same occupancy budget across dozens of
species, per configs/tomogram.toml) and, at voxel_size=2, likely to make
per-instance rendering the dominant, very slow cost. filler_from_pei2016
avoids both problems the same way the canonical config does.

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

Each worker's stdout is kept, not discarded: `TomogramSpecimenGenerator`
already prints a `phase_done` line per stage (membrane, structure fetch,
packing, rendering, region labelling) and the pipeline prints placed-instance
counts and achieved occupancy. Those lines are what the "where the 2 A run's
time goes" paragraph in build-tomogram.md quotes, so they are parsed back out
here rather than re-measured by hand under a profiler.

Everything here runs the DEFAULT `packing_backend="shape"`, and there is
deliberately no mode for running the comparison against `"sphere"`: that
backend is on its way out, and a benchmark is a poor place to keep a
deprecated code path alive. The packing figures this replaces quoted a
shape-vs-sphere pair on two different docs pages, measured at different
times, which is why neither agreed with the other.

Run with: uv run python docs-figures/build_tomogram_benchmark.py
Saves docs/assets/images/tomogram-benchmark-projections.png and prints a
markdown results table (hand-copied into build-tomogram.md, not included
live, matching this repo's other docs-figures/ scripts).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path("docs/assets/images")
# NOT under the repo: at voxel_size=2 the density volume alone is ~27 GB, which
# blew through this user's home-directory (NFS) disk quota during testing,
# even after cleaning up between runs. /scratch has no such quota.
SCRATCH_DIR = Path("/scratch/loh/joel/tomogram_benchmark_scratch")
# Which GPU to benchmark on. All four cards on the reference host are
# identical L40s, so any idle one reproduces the published numbers -- pick
# whichever is free rather than hardcoding a card someone else may be using.
DEVICE = os.environ.get("SPECTER_BENCHMARK_DEVICE", "cuda:3")

PHYSICAL_FOV_A = (1500.0, 6000.0, 6000.0)  # (Z, Y, X) A -- production scale

V_SIZES = [10.0, 5.0, 2.0]


def target_shape_for(voxel_size: float) -> tuple[int, int, int]:
    return tuple(round(fov / voxel_size) for fov in PHYSICAL_FOV_A)  # type: ignore[return-value]


def _run_worker(voxel_size: float, out_mrc: Path, result_json: Path) -> None:
    """Runs inside the timed subprocess: build the config, run it, record
    wall time + GPU peak memory (RAM peak is measured by the parent via
    `/usr/bin/time -v` around this whole process instead)."""
    import torch

    from specter.config import TomogramConfig
    from specter.pipelines import run_build_tomogram

    shape_zyx = target_shape_for(voxel_size)
    cfg = TomogramConfig(
        targets=[{"pdb_source": "1bxn", "n_copies": 20}],
        filler_from_pei2016=True,
        membrane=[{"shape_backend": "spherical_harmonics"}],
        target_shape=list(shape_zyx),
        voxel_size=voxel_size,
        filler_occupancy_fraction=0.5,
        seed=42,
        device=DEVICE,
        accumulator_device="auto",
        render_workers="auto",
        render_chunk_size=64,
        # Off for the benchmark: picks/segmentation aren't needed for the
        # sum-projection figure, and at voxel_size=2 (~6.75B voxels) the label
        # volumes alone are ~40 GB on top of the ~27 GB density volume --
        # writing all of it blew through this filesystem's per-user disk
        # quota during testing (see the module docstring above).
        write_picks=False,
        write_segmentation=False,
        output_dir=str(out_mrc.parent),
        filename=out_mrc.stem,
    )

    torch.zeros(1, device=DEVICE)  # force CUDA context init on this device
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    run_build_tomogram(cfg)
    elapsed_s = time.perf_counter() - start
    gpu_peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    # The membrane distance transform (CuPy-backed) runs
    # in cupy's OWN allocator pool, which torch's counter cannot see -- add
    # what that pool reserved so the reported GPU peak isn't an undercount.
    try:
        import cupy

        cupy_pool_bytes = int(cupy.get_default_memory_pool().total_bytes())
    except ImportError:
        cupy_pool_bytes = 0

    result_json.write_text(
        json.dumps(
            {
                "voxel_size": voxel_size,
                "shape_zyx": list(shape_zyx),
                "n_voxels": shape_zyx[0] * shape_zyx[1] * shape_zyx[2],
                "elapsed_s": elapsed_s,
                "gpu_peak_bytes": gpu_peak_bytes,
                "cupy_pool_bytes": cupy_pool_bytes,
            }
        )
    )


# A `phase_done` line, e.g. "  Filler packing (cytosol): 1m 11s". Written by
# specter.progress._format_elapsed, which drops empty leading units and has
# whole-second resolution, so hours and minutes are both optional.
_PHASE_RE = re.compile(r"^(?P<label>.+?): (?:(\d+)h )?(?:(\d+)m )?(\d+)s\s*$")


def _parse_log(stdout: str) -> dict:
    """Pull the phase timings, placed-instance counts and achieved occupancy
    back out of a worker's own progress output.

    The generator and pipeline already print all three (`phase_done`, the
    per-region "N target(s), M filler placed" summary, and "Occupancy: X% of
    volume"), so a run reports its own breakdown. Re-deriving it under a
    profiler instead would both cost another run and measure a different,
    instrumented one."""
    phases: list[tuple[str, int]] = []
    placements: dict[str, tuple[int, int]] = {}
    occupancy: float | None = None

    for raw in stdout.splitlines():
        line = raw.rstrip()
        placed = re.match(
            r"^\s*(Cytosol|Lumen): (\d+) target\(s\), (\d+) filler placed", line
        )
        if placed:
            placements[placed.group(1).lower()] = (
                int(placed.group(2)),
                int(placed.group(3)),
            )
            continue
        occ = re.match(r"^\s*Occupancy: ([\d.]+)% of volume", line)
        if occ:
            occupancy = float(occ.group(1)) / 100.0
            continue
        phase = _PHASE_RE.match(line)
        if phase:
            h, m, s = (int(g or 0) for g in phase.groups()[1:])
            phases.append((phase.group("label").rstrip(), h * 3600 + m * 60 + s))

    n_placed = sum(t + f for t, f in placements.values())
    return {
        "phases": phases,
        "placements": placements,
        "n_placed": n_placed,
        "occupancy": occupancy,
    }


def _run_timed(voxel_size: float, tag: str, read_projection: bool = True) -> dict:
    """Spawns `_run_worker` as its own `/usr/bin/time -v`-wrapped
    subprocess and returns its result dict, plus parsed peak RSS and the
    phase breakdown parsed from the worker's own progress output."""
    out_mrc = SCRATCH_DIR / f"tomogram_{tag}.mrc"
    result_json = SCRATCH_DIR / f"result_{tag}.json"
    cmd = [
        "/usr/bin/time",
        "-v",
        sys.executable,
        __file__,
        "--worker",
        "--voxel_size",
        str(voxel_size),
        "--out_mrc",
        str(out_mrc),
        "--result_json",
        str(result_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed (voxel_size={voxel_size}):\n{proc.stderr[-4000:]}"
        )

    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    assert match is not None, f"couldn't parse /usr/bin/time output:\n{proc.stderr}"
    peak_rss_kb = int(match.group(1))

    result = json.loads(result_json.read_text())
    result["peak_rss_kb"] = peak_rss_kb
    # Kept on disk: the phase lines are the evidence behind the prose in
    # build-tomogram.md, and a run at voxel_size=2 is too expensive to repeat
    # because a parser here missed a line.
    log_path = SCRATCH_DIR / f"log_{tag}.txt"
    log_path.write_text(proc.stdout)
    result["log_path"] = str(log_path)
    result.update(_parse_log(proc.stdout))

    if read_projection:
        import mrcfile
        import numpy as np

        with mrcfile.open(out_mrc, permissive=True) as mrc:
            result["projection"] = np.asarray(mrc.data).sum(axis=0)  # sum Z projection
    out_mrc.unlink()  # each volume is tens of GB at voxel_size=2 -- don't keep more
    result_json.unlink()  # than one resident on disk at a time (disk quota)
    return result


def main() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"warming up (PDB cache, CUDA context) at voxel_size={V_SIZES[0]} ...")
    _run_timed(V_SIZES[0], tag="warmup", read_projection=False)

    results = []
    for voxel_size in V_SIZES:
        tag = f"v{voxel_size:g}"
        print(f"benchmarking voxel_size={voxel_size} A/voxel ...")
        result = _run_timed(voxel_size, tag=tag)
        print(
            f"  shape={result['shape_zyx']} time={result['elapsed_s']:.1f}s "
            f"gpu_peak={result['gpu_peak_bytes'] / 1e9:.2f} GB "
            f"(+{result.get('cupy_pool_bytes', 0) / 1e9:.2f} GB cupy) "
            f"ram_peak={result['peak_rss_kb'] / 1e6:.2f} GB"
        )
        results.append(result)

    _figure_projections(results)

    print(
        "\n| voxel_size (A/voxel) | shape (Z,Y,X voxels) | wall time | GPU peak | RAM peak |"
    )
    print("|---|---|---|---|---|")
    for r in results:
        shape_str = "x".join(str(s) for s in r["shape_zyx"])
        print(
            f"| {r['voxel_size']:g} | {shape_str} | {r['elapsed_s']:.0f} s | "
            f"{(r['gpu_peak_bytes'] + r.get('cupy_pool_bytes', 0)) / 1e9:.2f} GB | "
            f"{r['peak_rss_kb'] / 1e6:.2f} GB |"
        )

    for r in results:
        _print_breakdown(r)


def _print_breakdown(result: dict) -> None:
    """Where one run's time went, from its own progress output."""
    print(
        f"\nvoxel_size={result['voxel_size']:g} phase breakdown "
        f"({result['elapsed_s']:.0f} s total, log: {result['log_path']}):"
    )
    for label, seconds in result["phases"]:
        print(f"  {label:<48} {seconds:>6} s")
    if result["occupancy"] is not None:
        placed = ", ".join(
            f"{loc} {t} target / {f} filler"
            for loc, (t, f) in result["placements"].items()
        )
        print(
            f"  {result['n_placed']:,} instances placed ({placed}), "
            f"occupancy {result['occupancy']:.1%}"
        )


def _figure_projections(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.6))
    for ax, r in zip(axes, results):
        ax.imshow(r["projection"], cmap="gray_r", origin="lower")
        ax.set_title(
            f"voxel_size = {r['voxel_size']:g} Å\n{'x'.join(map(str, r['shape_zyx']))} voxels",
            fontsize=10,
        )
        ax.axis("off")
    fov_str = " x ".join(f"{a:g}" for a in PHYSICAL_FOV_A)
    fig.suptitle(
        f"Same field of view ({fov_str} Å), three voxel sizes",
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
    parser.add_argument("--voxel_size", type=float, default=5.0)
    parser.add_argument("--out_mrc", type=Path)
    parser.add_argument("--result_json", type=Path)
    args = parser.parse_args()

    if args.worker:
        _run_worker(args.voxel_size, args.out_mrc, args.result_json)
    else:
        main()
