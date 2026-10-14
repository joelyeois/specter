"""
Figures and printed output for docs/walkthrough/reconstruction.md.

Run with: uv run python docs-figures/walkthrough/reconstruction.py
Takes about two minutes on one L40.
"""

from __future__ import annotations

import logging
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, save, show_image

# Lightning's banner and dataloader advice are not part of the page.
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="lightning")

transcript = Transcript("reconstruction")

transcript.start("setup")
# --8<-- [start:setup]
import specter
from specter.imaging import CTF, Exposure, Poses, Structure, simulate_particles

specter.seed(0)

structure = Structure.from_pdb("6bdf")
truth = structure.render(voxel_size=2.0, box=128, device="cuda")
print(f"ground truth: {tuple(truth.potential.shape)} voxels of {truth.voxel_size} Å")
# --8<-- [end:setup]

transcript.start("simulate")
# --8<-- [start:simulate]
n = 1000
poses = Poses.random(n, shift=3.0)
ctf = CTF.sample(n, defocus=[10000.0, 20000.0])
exposure = Exposure(dose=20.0)

stack = simulate_particles(
    truth, poses, ctf, exposure, voltage=300.0, seed=0, batch_size=200
)
images = stack.array.cpu()
print(f"{tuple(images.shape)} images, mean {images.mean():.1f} counts/pixel")
# --8<-- [end:simulate]

fig, axes = plt.subplots(1, 6, figsize=(10.5, 2.0))
for i, ax in enumerate(axes):
    defocus = float(ctf.params["dfu"][i]) / 1e4
    show_image(
        ax,
        images[i],
        title=f"{defocus:.2f} µm underfocus",
        pixel_size=2.0,
        scalebar=50 if i == 0 else None,
    )
fig.tight_layout()
save(fig, "reconstruction-data")

transcript.start("reconstruct")
# --8<-- [start:reconstruct]
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from specter.arrays import ball3d
from specter.ghostbuster import Reconstructor


class Snapshots(L.Callback):
    """Keep a copy of the map at the end of every epoch."""

    def __init__(self) -> None:
        self.volumes: list[torch.Tensor] = []

    def on_train_epoch_end(self, trainer, model) -> None:
        self.volumes.append(model.V.detach().cpu().clone())


def reconstruct(idx: torch.Tensor, epochs: int) -> tuple[Reconstructor, Snapshots]:
    """Fit a map to the images `idx`, starting from an empty box."""
    model = Reconstructor(
        torch.zeros(128, 128, 128),
        voxel_size=2.0,
        quaternions=poses.quaternions[idx],
        translations=poses.shifts[idx],
        ctf_params=ctf[idx].params,
        voltage=300.0,
        dose_per_angstrom=20.0,
        lr=0.1,
        kmask=ball3d(128, 128),  # keep the map inside the Nyquist sphere
    )
    loader = DataLoader(
        TensorDataset(images[idx], torch.arange(len(idx))),
        batch_size=16,
        shuffle=True,
    )
    snapshots = Snapshots()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        callbacks=[snapshots],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    return model, snapshots


model, snapshots = reconstruct(torch.arange(n), epochs=10)
V = model.V.detach().cpu()
print(f"reconstructed map: {tuple(V.shape)}, {len(model.log_total_loss)} steps")
# --8<-- [end:reconstruct]

fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0))
truth_cpu = truth.potential.cpu()
for row, (name, vol) in enumerate([("Ground truth", truth_cpu), ("Reconstruction", V)]):
    vol = as_numpy(vol)
    views = [
        (vol[vol.shape[0] // 2], "central slice"),
        (vol.sum(0), "projected along z"),
        (vol.sum(1), "projected along y"),
    ]
    for col, (view, label) in enumerate(views):
        show_image(
            axes[row, col],
            view,
            cmap="gray_r",
            title=f"{name}, {label}",
            pixel_size=2.0 if col == 0 else None,
            scalebar=50 if col == 0 else None,
        )
fig.tight_layout()
save(fig, "reconstruction-maps")

# The card thumbnail on docs/basic-examples.md: the reconstruction seen side on.
side = as_numpy(V.sum(1))[20:108, 20:108]
fig, ax = plt.subplots(figsize=(3, 3))
show_image(
    ax,
    side,
    cmap="gray_r",
    vmin=float(np.percentile(side, 1)),
    vmax=float(np.percentile(side, 99.7)),
)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "basic-reconstruction")

transcript.start("fsc")
# --8<-- [start:fsc]
from specter.fft import fourier_shell_correlation
from specter.plots import resolution_between

reference = truth.potential.cpu()
for epoch, volume in enumerate(snapshots.volumes, start=1):
    res = resolution_between(volume, reference, voxel_size=2.0, threshold=0.5)
    print(f"epoch {epoch}: FSC = 0.5 against the model at {res}")
# --8<-- [end:fsc]

nyquist = 1 / (2 * 2.0)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
shown = [1, 2, 4, 7, 10]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(shown)))
for color, epoch in zip(colors, shown):
    volume = snapshots.volumes[epoch - 1]
    k, fsc = fourier_shell_correlation(volume, reference, pixel_size=2.0)
    keep = k <= nyquist
    ax.plot(
        as_numpy(k[keep]),
        as_numpy(fsc[keep]),
        color=color,
        lw=1.2,
        label=f"epoch {epoch}",
    )
ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="FSC against the model",
    xlim=(0, nyquist),
    ylim=(-0.05, 1.02),
)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "reconstruction-fsc-epochs")

losses = np.array([float(x) for x in model.log_total_loss])
steps_per_epoch = len(losses) // len(snapshots.volumes)
floor = float(images.mean())
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(
    np.arange(len(losses)) / steps_per_epoch,
    losses,
    color=PALETTE[0],
    lw=0.6,
    alpha=0.6,
    label="per step",
)
smooth = np.convolve(losses, np.ones(50) / 50, mode="valid")
ax.plot(
    (np.arange(len(smooth)) + 25) / steps_per_epoch,
    smooth,
    color=PALETTE[0],
    lw=1.5,
    label="50-step mean",
)
ax.axhline(
    floor, color=PALETTE[3], lw=1.0, ls="--", label=f"shot-noise variance ({floor:.0f})"
)
ax.set(
    xlabel="Epoch",
    ylabel="Mean squared error (counts²)",
    ylim=(float(smooth.min()) - 1, float(np.percentile(losses, 99))),
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "reconstruction-loss")

transcript.start("loss")
for epoch in range(len(snapshots.volumes)):
    chunk = losses[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]
    print(f"epoch {epoch + 1}: mean loss {chunk.mean():.2f} counts²")
print(f"mean count per pixel: {floor:.2f}")

# Not on the page: the evidence for its statement about the lowest shells.
k_low, fsc_low = fourier_shell_correlation(V, reference, pixel_size=2.0)
print(
    "lowest shells, full map vs model:",
    [round(float(x), 3) for x in fsc_low[:4]],
    "at k =",
    [round(float(x), 4) for x in k_low[:4]],
    file=sys.__stdout__,
)

transcript.start("halves")
# --8<-- [start:halves]
# Twice the epochs, so each half takes as many optimiser steps as the full map.
half_a, _ = reconstruct(torch.arange(0, n, 2), epochs=20)
half_b, _ = reconstruct(torch.arange(1, n, 2), epochs=20)
A, B = half_a.V.detach().cpu(), half_b.V.detach().cpu()

print(f"half-map FSC = 0.143:            {resolution_between(A, B, 2.0, 0.143)}")
print(f"full map against the model, 0.5: {resolution_between(V, reference, 2.0, 0.5)}")
# --8<-- [end:halves]

k, fsc_half = fourier_shell_correlation(A, B, pixel_size=2.0)
_, fsc_full = fourier_shell_correlation(V, reference, pixel_size=2.0)
keep = k <= nyquist
k, fsc_half, fsc_full = (
    as_numpy(k[keep]),
    as_numpy(fsc_half[keep]),
    as_numpy(fsc_full[keep]),
)
# Rosenthal & Henderson (2003): the full map's FSC against the truth, predicted
# from the half-map FSC alone.
predicted = np.sqrt(np.clip(2 * fsc_half / (1 + fsc_half), 0, None))
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(k, fsc_half, color=PALETTE[0], lw=1.3, label="half A against half B")
ax.plot(k, fsc_full, color=PALETTE[1], lw=1.3, label="full map against the model")
ax.plot(
    k,
    predicted,
    color=PALETTE[1],
    lw=1.0,
    ls=":",
    label=r"$\sqrt{2\,\mathrm{FSC}_{1/2}/(1+\mathrm{FSC}_{1/2})}$",
)
ax.axhline(0.143, color=PALETTE[0], lw=0.8, ls="--")
ax.axhline(0.5, color=PALETTE[1], lw=0.8, ls="--")
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="FSC",
    xlim=(0, nyquist),
    ylim=(-0.05, 1.02),
)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "reconstruction-halves")

transcript.close()
