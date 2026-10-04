"""Plot isolated radial-filter measurements alongside the separate CLI report."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "512 px native": "fft-512-final.json",
        "256 px native": "fft-256-final.json",
        "Tilt padded geometry": "fft-tilt-geometry.json",
    }
    summary = {}
    for label, file in files.items():
        data = json.loads((args.root / file).read_text())
        variants = {}
        for variant in ["before", "after"]:
            rows = [r for r in data["results"] if r["variant"] == variant]
            assert all(r["bitwise_equal"] for r in rows)
            variants[variant] = {
                "median_seconds": statistics.median(r["seconds"] for r in rows),
                "peak_allocated_GiB": max(r["peak_allocated_bytes"] for r in rows)
                / 2**30,
                "stages_seconds": [
                    statistics.median(r["stages"][i]["seconds"] for r in rows)
                    for i in range(3)
                ],
            }
        summary[label] = {
            "shape": data["results"][0]["shape"],
            "voxels": int(np.prod(data["results"][0]["shape"])),
            "bitwise_equal": True,
            "variants": variants,
            "speedup": variants["before"]["median_seconds"]
            / variants["after"]["median_seconds"],
            "allocated_saved_GiB": variants["before"]["peak_allocated_GiB"]
            - variants["after"]["peak_allocated_GiB"],
        }
    (args.output / "filter-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    x = np.arange(len(summary))
    for offset, variant, label in [
        (-0.18, "before", "Before"),
        (0.18, "after", "After"),
    ]:
        bars = axes[0].bar(
            x + offset,
            [r["variants"][variant]["median_seconds"] for r in summary.values()],
            width=0.36,
            label=label,
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=3)
        bars = axes[1].bar(
            x + offset,
            [r["variants"][variant]["peak_allocated_GiB"] for r in summary.values()],
            width=0.36,
            label=label,
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set_title("Isolated filter time (seconds)")
    axes[1].set_title("Allocated GPU peak, including live fields (GiB)")
    for axis in axes[:2]:
        axis.set_xticks(x, list(summary), rotation=15)
        axis.set_ylim(top=axis.get_ylim()[1] * 1.15)
    native = summary["512 px native"]["variants"]
    for offset, variant, label in [
        (-0.18, "before", "Before"),
        (0.18, "after", "After"),
    ]:
        bars = axes[2].bar(
            x + offset, native[variant]["stages_seconds"], width=0.36, label=label
        )
        axes[2].bar_label(bars, fmt="%.3f", padding=3)
    axes[2].set_xticks(x, ["Forward FFT", "Weighting", "Inverse FFT"], rotation=15)
    axes[2].set_title("Native 512 px filter stages (seconds)")
    axes[2].set_ylim(top=axes[2].get_ylim()[1] * 1.15)
    axes[0].legend()
    fig.suptitle(
        "Dense-volume diagnostic: identical outputs; whole CLI results are measured separately"
    )
    fig.savefig(args.output / "filter-timing-memory.png", dpi=160)
    fig.savefig(args.output / "filter-timing-memory.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
