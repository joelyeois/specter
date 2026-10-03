"""Compare all matching stacks and render before/after scientific figures.

Run after match_absorption_benchmark.py with before/after output directories
named before-material-0, after-material-0, before-uniform-256 and
after-uniform-256. Raw experimental/simulation data are not copied to docs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (np.isnan(left) and np.isnan(right))
    return left == right


def normalise_paths(value, directory):
    if isinstance(value, dict):
        return {k: normalise_paths(v, directory) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise_paths(v, directory) for v in value]
    if isinstance(value, str):
        return value.replace(str(directory), "<output>")
    return value


def spectrum(path):
    with mrcfile.mmap(path, permissive=True) as stack:
        power = np.zeros(stack.data.shape[-2:], dtype=np.float64)
        for start in range(0, len(stack.data), 8):
            block = np.asarray(stack.data[start : start + 8], dtype=np.float64)
            power += (np.abs(np.fft.fft2(block, norm="ortho")) ** 2).sum(axis=0)
        power /= len(stack.data)
        image = stack.data[0].copy()
    return image, np.fft.fftshift(power)


def radial(power, dx):
    n = power.shape[-1]
    frequency = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    y, x = np.meshgrid(frequency, frequency, indexing="ij")
    index = np.rint(np.hypot(x, y) * n * dx).astype(int)
    sums = np.bincount(index.ravel(), weights=power.ravel())
    counts = np.bincount(index.ravel())
    stop = n // 2 + 1
    return np.arange(stop) / (n * dx), sums[:stop] / counts[:stop]


def compare_case(root, output, case, dx):
    before, after = root / f"before-{case}", root / f"after-{case}"
    measurements = [
        json.loads((p / "measurement.json").read_text()) for p in (before, after)
    ]
    files = sorted(p.name for p in (before / "probes").glob("*.mrcs"))
    assert len(files) == 10
    assert files == sorted(p.name for p in (after / "probes").glob("*.mrcs"))
    rows = []
    for name in files:
        with (
            mrcfile.mmap(before / "probes" / name, permissive=True) as a,
            mrcfile.mmap(after / "probes" / name, permissive=True) as b,
        ):
            assert a.data.shape == b.data.shape
            hashes = [hashlib.sha256(), hashlib.sha256()]
            exact = True
            maximum = 0.0
            for start in range(0, len(a.data), 8):
                aa, bb = a.data[start : start + 8], b.data[start : start + 8]
                hashes[0].update(aa.tobytes())
                hashes[1].update(bb.tobytes())
                exact &= np.array_equal(aa, bb)
                maximum = max(maximum, float(np.max(np.abs(aa - bb))))
            rows.append(
                dict(
                    file=name,
                    shape=list(a.data.shape),
                    pixels=a.data.size,
                    bitwise_equal=bool(exact),
                    max_abs_difference=maximum,
                    before_sha256=hashes[0].hexdigest(),
                    after_sha256=hashes[1].hexdigest(),
                )
            )
    report_equal = equal(measurements[0]["report"], measurements[1]["report"])
    configs = [
        normalise_paths(tomllib.loads((p / "matched.toml").read_text()), p)
        for p in (before, after)
    ]
    config_equal = equal(*configs)
    result = dict(
        case=case,
        stacks=rows,
        pixels=sum(row["pixels"] for row in rows),
        report_bitwise_equal=report_equal,
        matched_settings_equal=config_equal,
    )
    assert all(row["bitwise_equal"] for row in rows), result
    assert report_equal and config_equal, result
    (output / f"precision-{case}.json").write_text(json.dumps(result, indent=2) + "\n")
    name = "battery_seed0.mrcs"
    images, powers = zip(
        *(spectrum(p / "probes" / name) for p in (before, after)), strict=True
    )
    assert np.array_equal(powers[0], powers[1])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    low, high = np.percentile(images[0], [1, 99])
    for axis, image, title in zip(
        axes[0, :2], images, ["Before", "After"], strict=True
    ):
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"{title}: particle 1, seed 0")
    diff = images[1] - images[0]
    axes[0, 2].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 2].set_title(f"Difference: max |Δ| = {np.abs(diff).max():g}")
    logs = np.log10(np.maximum(powers[0], np.finfo(float).tiny))
    low, high = np.percentile(logs, [1, 99])
    for axis, power, title in zip(
        axes[1, :2], powers, ["Before", "After"], strict=True
    ):
        axis.imshow(
            np.log10(np.maximum(power, np.finfo(float).tiny)),
            cmap="magma",
            vmin=low,
            vmax=high,
        )
        axis.set_title(f"{title}: mean Fourier power, all 200 particles")
    for power, title, style in zip(
        powers, ["Before", "After"], ["-", "--"], strict=True
    ):
        frequency, values = radial(power, dx)
        axes[1, 2].semilogy(frequency[1:], values[1:], style, label=title)
    axes[1, 2].set(
        xlabel="Spatial frequency (Å⁻¹)",
        ylabel="Mean power",
        title="Radial Fourier spectra (identical)",
    )
    axes[1, 2].legend()
    for axis in list(axes[0]) + list(axes[1, :2]):
        axis.set_axis_off()
    fig.suptitle(
        f"Matching — {case}: all {result['pixels']:,} output pixels and report metrics identical"
    )
    fig.savefig(output / f"fourier-{case}.png", dpi=160)
    fig.savefig(output / f"fourier-{case}.pdf")
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != "stacks"}))
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", choices=["material-0", "uniform-256", "all"], default="all"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {"material-0": 0.731, "uniform-256": 1.462}
    data = {}
    for case, dx in cases.items():
        if args.case in [case, "all"]:
            data[case] = compare_case(args.root, args.output, case, dx)
    summary = {}
    for case, (before, after) in data.items():
        row = {
            "before_seconds": before["cli_seconds"],
            "after_seconds": after["cli_seconds"],
            "speedup": before["cli_seconds"] / after["cli_seconds"],
            "runtime_change_percent": 100
            * (after["cli_seconds"] / before["cli_seconds"] - 1),
        }
        for key in ["peak_allocated_bytes", "peak_reserved_bytes", "driver_peak_bytes"]:
            row[key] = {
                "before_GiB": before[key] / 2**30,
                "after_GiB": after[key] / 2**30,
                "saved_GiB": (before[key] - after[key]) / 2**30,
                "saved_percent": 100 * (1 - after[key] / before[key]),
            }
        summary[case] = row
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, key, scale, title in zip(
        axes,
        ["cli_seconds", "peak_allocated_bytes", "peak_reserved_bytes"],
        [60, 2**30, 2**30],
        [
            "Whole CLI time (minutes)",
            "Peak allocated GPU memory (GiB)",
            "Peak reserved GPU memory (GiB)",
        ],
        strict=True,
    ):
        x = np.arange(len(data))
        for offset, index, label in [(-0.18, 0, "Before"), (0.18, 1, "After")]:
            bars = axis.bar(
                x + offset,
                [rows[index][key] / scale for rows in data.values()],
                width=0.36,
                label=label,
            )
            axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_xticks(
            x,
            [
                "512 px material" if k == "material-0" else "256 px uniform"
                for k in data
            ],
        )
        axis.set_title(title)
        axis.set_ylim(top=axis.get_ylim()[1] * 1.15)
    axes[0].legend()
    fig.savefig(args.output / "timing-memory.png", dpi=160)
    fig.savefig(args.output / "timing-memory.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
