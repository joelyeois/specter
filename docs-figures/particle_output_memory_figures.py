"""Plot production particle CLI measurements and the same-batch Fourier check.

Run with the benchmark root as the first argument and an artifact directory
as the second. Large MRC stacks stay in the benchmark root, outside Git.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def profile(stack, dx):
    n = stack.shape[-1]
    frequencies = np.fft.fftfreq(n, d=dx)
    radius = np.hypot(frequencies[:, None], frequencies[None, :])
    spacing = 1 / (n * dx)
    bins = np.floor(radius / spacing).astype(int)
    keep = bins <= n // 2
    count = np.bincount(bins[keep])
    total = np.zeros(len(count))
    for image in stack:
        centered = image.astype(np.float64) - float(image.mean(dtype=np.float64))
        power = np.abs(np.fft.fft2(centered)) ** 2 / n**2
        total += np.bincount(bins[keep], weights=power[keep], minlength=len(count))
    return np.arange(len(count)) * spacing, total / (count * len(stack))


def difference(before, after):
    squared_error = squared_reference = maximum = 0.0
    unequal = 0
    for a, b in zip(before, after, strict=True):
        d = b.astype(np.float64) - a.astype(np.float64)
        squared_error += np.sum(d**2)
        squared_reference += np.sum(a.astype(np.float64) ** 2)
        maximum = max(maximum, np.max(np.abs(d)))
        unequal += np.count_nonzero(a != b)
    return {
        "bitwise_equal": bool(unequal == 0),
        "unequal_pixels": int(unequal),
        "pixel_count": int(before.size),
        "relative_l2": float(np.sqrt(squared_error / squared_reference)),
        "max_abs": float(maximum),
    }


def main():
    root, output = map(Path, sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=True)
    cli = {
        name: json.loads((root / directory / "measurement.json").read_text())
        for name, directory in (
            ("before", "before-3000"),
            ("after", "after-3000"),
        )
    }
    paired_dir = root / "same-batches-3000"
    paired = json.loads((paired_dir / "paired-storage.json").read_text())
    with (
        mrcfile.mmap(paired_dir / "paired-before.mrcs") as old,
        mrcfile.mmap(paired_dir / "paired-after.mrcs") as new,
    ):
        before, after = old.data, new.data
        accuracy = difference(before, after)
        k, p0 = profile(before, 1.0)
        _, p1 = profile(after, 1.0)
        i = 0
        a, b = before[i], after[i]
        lo, hi = np.percentile(a, [1, 99])
        f0, f1 = [
            np.log10(1 + np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2)
            for x in (a, b)
        ]
        flo, fhi = np.percentile(f0, [1, 99.9])
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for ax, image, title in zip(
            axes[0],
            (a, b, b - a),
            ("Before: particle 0", "After: particle 0", "After − before"),
            strict=True,
        ):
            if title == "After − before":
                limit = max(float(np.abs(image).max()), 1e-6)
                plot = ax.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
                fig.colorbar(plot, ax=ax, shrink=0.7)
            else:
                ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(title)
            ax.set_axis_off()
        for ax, image, title in zip(
            axes[1, :2],
            (f0, f1),
            ("Fourier power: before", "Fourier power: after"),
            strict=True,
        ):
            ax.imshow(
                image, cmap="magma", vmin=flo, vmax=fhi, extent=(-0.5, 0.5, -0.5, 0.5)
            )
            ax.set_title(title)
            ax.set_xlabel("Spatial frequency (Å⁻¹)")
            ax.set_ylabel("Spatial frequency (Å⁻¹)")
        axes[1, 2].semilogy(k[1:], p0[1:], label="Before", linewidth=2)
        axes[1, 2].semilogy(k[1:], p1[1:], "--", label="After", linewidth=1.3)
        axes[1, 2].set_title("Mean radial power: all 3,000 particles")
        axes[1, 2].set_xlabel("Spatial frequency (Å⁻¹)")
        axes[1, 2].set_ylabel("Power per Fourier pixel")
        axes[1, 2].legend()
        fig.suptitle(
            "Particle output collection and normalization: identical generated batches\n"
            f"3,000 × 256 × 256 pixels; bitwise equal = {accuracy['bitwise_equal']}",
            fontsize=14,
        )
        fig.savefig(output / "images_and_fourier.png", dpi=160)
        fig.savefig(output / "images_and_fourier.pdf")
        plt.close(fig)
    with (output / "spectra.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["spatial_frequency_inv_angstrom", "before_power", "after_power"]
        )
        writer.writerows(zip(k, p0, p1, strict=True))

    ddp = {
        name: json.loads((root / directory / "measurement-rank0.json").read_text())
        for name, directory in (
            ("before", "ddp-before-3000-idle"),
            ("after", "ddp-after-3000-idle"),
        )
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    charts = (
        (
            "Single-GPU CLI",
            "Seconds",
            [cli[v]["cli_seconds"] for v in ("before", "after")],
        ),
        (
            "Two-GPU CLI",
            "Seconds",
            [ddp[v]["cli_seconds"] for v in ("before", "after")],
        ),
        (
            "Single-GPU host RSS",
            "GiB",
            [cli[v]["host_rss_max_bytes"] / 2**30 for v in ("before", "after")],
        ),
        (
            "Two-GPU process-tree PSS",
            "GiB",
            [
                ddp[v]["process_tree"]["peak_aggregate_pss_bytes"] / 2**30
                for v in ("before", "after")
            ],
        ),
    )
    for ax, (title, unit, values) in zip(axes, charts, strict=True):
        bars = ax.bar(["Before", "After"], values, color=["#637caa", "#df9251"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title)
        ax.set_ylabel(unit)
    fig.suptitle(
        "L40 GPUs · 3,000 particles · shipped particle.toml · fixed batch size 2"
    )
    fig.savefig(output / "timing_and_memory.png", dpi=160)
    fig.savefig(output / "timing_and_memory.pdf")
    plt.close(fig)

    with (
        mrcfile.mmap(root / "before-3000" / "particles.mrcs") as old,
        mrcfile.mmap(root / "after-3000" / "particles.mrcs") as new,
    ):
        independent_accuracy = difference(old.data, new.data)
    summary = {
        "cli": cli,
        "ddp_cli": ddp,
        "paired_storage": paired,
        "same_batch_accuracy": accuracy,
        "independent_cli_accuracy": independent_accuracy,
        "radial_power_relative_l2": float(np.linalg.norm(p1 - p0) / np.linalg.norm(p0)),
    }
    (output / "measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "same_batch_accuracy": accuracy,
                "independent_cli_accuracy": independent_accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
