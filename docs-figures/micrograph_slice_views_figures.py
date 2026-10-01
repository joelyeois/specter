"""Plot full-scale same-specimen images, Fourier spectra, timing and memory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
data, out = Path(a.data), Path(a.output)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
pair = json.loads((data / "pairs/pairs.json").read_text())
cli = [
    json.loads((data / folder / "measurement.json").read_text())
    for folder in ["baseline", "candidate"]
]
images = [
    torch.load(data / folder / "clean.pt", weights_only=True)[0].numpy()
    for folder in ["replay_before", "replay_after"]
]
noisy = []
for folder in ["replay_before", "replay_after"]:
    with mrcfile.open(data / folder / "micrographs.mrcs") as m:
        noisy.append(m.data.copy())
precision = {
    "clean_bitwise_equal": bool(np.array_equal(*images)),
    "noisy_bitwise_equal": bool(np.array_equal(*noisy)),
    "clean_max_abs_error": float(np.abs(images[1] - images[0]).max()),
    "noisy_max_abs_error": float(np.abs(noisy[1] - noisy[0]).max()),
}
power = [np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean()))) ** 2 for im in images]
precision["power_bitwise_equal"] = bool(np.array_equal(*power))
(out / "precision.json").write_text(json.dumps(precision, indent=2) + "\n")
fig, ax = plt.subplots(2, 3, figsize=(12, 7.7), constrained_layout=True)
crop = (slice(1536, 2560), slice(1536, 2560))
vmin, vmax = np.percentile(images[0][crop], [1, 99])
for i, label in enumerate(["Before", "After"]):
    ax[0, i].imshow(images[i][crop], cmap="gray", vmin=vmin, vmax=vmax)
    ax[0, i].set_title(f"{label}: clean micrograph, central 1024² crop")
    log = np.log10(power[i] + 1e-20)
    fmin, fmax = np.percentile(log, [1, 99])
    ax[1, i].imshow(
        log, cmap="magma", vmin=fmin, vmax=fmax, extent=[-0.5, 0.5, -0.5, 0.5]
    )
    ax[1, i].set_title(f"{label}: full 4096² Fourier power (log₁₀)")
    ax[1, i].set_xlabel("Spatial frequency (Å⁻¹)")
    ax[1, i].set_ylabel("Spatial frequency (Å⁻¹)")
    ax[0, i].set_axis_off()
ax[0, 2].imshow(np.abs(images[1] - images[0])[crop], cmap="gray", vmin=0, vmax=1)
ax[0, 2].set_title("Absolute difference: exactly zero")
ax[0, 2].set_axis_off()
n = images[0].shape[0]
coord = np.arange(n) - n // 2
radius = np.hypot(coord[:, None], coord[None, :]).astype(np.int32)
counts = np.bincount(radius.ravel())
for i, label in enumerate(["Before", "After"]):
    radial = np.bincount(radius.ravel(), weights=power[i].ravel()) / counts
    k = np.arange(len(radial)) / n
    mask = (k > 0) & (k <= 0.5)
    ax[1, 2].semilogy(
        k[mask],
        radial[mask],
        label=label,
        linestyle="-" if i == 0 else "--",
        linewidth=1.7,
    )
ax[1, 2].set_title("Radial spectra overlap exactly")
ax[1, 2].set_xlabel("Spatial frequency (Å⁻¹)")
ax[1, 2].set_ylabel("Mean Fourier power")
ax[1, 2].legend()
fig.suptitle(
    "simulate micrograph — same shipped-scale specimen, 500 × 4096² at 1 Å\nFloat32 potential / complex64 propagation; clean and fixed-seed noisy images bitwise identical"
)
for suffix in ["png", "pdf"]:
    fig.savefig(out / f"images_and_fourier.{suffix}", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
colors = ["#777777", "#147d92"]
summary = {}
for j, placement in enumerate(["resident", "streamed"]):
    trials = [
        [t for t in pair["trials"] if t["placement"] == placement and t["variant"] == v]
        for v in ["before", "after"]
    ]
    times = [np.median([t["seconds"] for t in ts]) for ts in trials]
    memory = [ts[0]["peak_allocated_bytes"] / 2**30 for ts in trials]
    summary[placement] = {
        "median_seconds": times,
        "peak_allocated_gib": memory,
        "speedup": times[0] / times[1],
    }
    ax[j].bar(["Before", "After"], times, color=colors)
    for i, ts in enumerate(trials):
        ax[j].scatter(
            [i] * len(ts), [t["seconds"] for t in ts], color="black", s=14, zorder=3
        )
        ax[j].text(i, times[i] + 0.35, f"{times[i]:.2f} s", ha="center")
    ax[j].set_title(f"{placement.capitalize()} propagation")
    ax[j].set_ylabel("Seconds (median of 3 full-volume runs)")
    ax[j].set_ylim(0, max(times) * 1.22)
ax[2].bar(["Before", "After"], summary["resident"]["peak_allocated_gib"], color=colors)
ax[2].set_ylim(0, 37)
ax[2].set_title("Resident propagation: 64 MiB less VRAM")
ax[2].set_ylabel("Peak PyTorch allocation (GiB)")
for i, value in enumerate(summary["resident"]["peak_allocated_gib"]):
    ax[2].text(i, value + 0.7, f"{value:.4f} GiB", ha="center")
fig.suptitle(
    "Full 500 × 4096² specimen, NVIDIA L40 — alternating before/after pairs\nShared GPU: observed timings, not an isolated end-to-end speedup claim"
)
for suffix in ["png", "pdf"]:
    fig.savefig(out / f"timing_and_memory.{suffix}", dpi=180)
summary["cli_seconds"] = [c["cli_seconds_excluding_volume_capture"] for c in cli]
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"precision": precision, "summary": summary}, indent=2))
