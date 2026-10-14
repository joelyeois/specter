"""
Figures and printed output for docs/walkthrough/specimens.md.

Run with: uv run python docs-figures/walkthrough/specimens.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, save, show_image

transcript = Transcript("specimens")


def radial_power_3d(volume: torch.Tensor, voxel_size: float):
    """Rotationally averaged 3D power spectrum of a cube, mean removed."""
    v = volume - volume.mean()
    power = torch.fft.fftshift(torch.fft.fftn(v)).abs().square().cpu().numpy()
    n = power.shape[-1]
    z, y, x = np.indices(power.shape)
    r = np.sqrt((z - n // 2) ** 2 + (y - n // 2) ** 2 + (x - n // 2) ** 2)
    r = r.astype(int)
    profile = np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel())
    k = np.arange(len(profile)) / (n * voxel_size)
    keep = (k > 0) & (k <= 0.5 / voxel_size)
    return k[keep], profile[keep]


transcript.start("setup")
# --8<-- [start:setup]
import torch

import specter
from specter.imaging import Exposure, Poses, Specimen, Structure

specter.seed(0)  # fixes the random poses drawn below
structure = Structure.from_pdb("6bdf")
specimen = structure.render(voxel_size=1.0, box=256, device="cuda")
diameter = round(structure.pdb.max_diameter)  # 192 Å
# --8<-- [end:setup]

transcript.start("recipe")
# --8<-- [start:recipe]
solvated = specimen.crowd(min_distance=diameter).solvate(model="gd", thickness=500.0)

for step in solvated.plan(voltage=300.0):
    print(step)
# --8<-- [end:recipe]

transcript.start("crowd")
# --8<-- [start:crowd]
poses = Poses.random(4)
crowded = specimen.crowd(min_distance=diameter)

alone = specimen.materialize(poses, voltage=300.0, seed=0)
together = crowded.materialize(poses, voltage=300.0, seed=0)
print(f"volumes {tuple(together.shape)}")
for i in range(len(poses)):
    extra = together[i].sum() / alone[i].sum() - 1
    print(f"image {i}: neighbours add {extra:.0%} to the template's potential")
# --8<-- [end:crowd]

fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.4))
for i in range(4):
    for row, (vol, label) in enumerate([(alone, "Alone"), (together, "Crowded")]):
        ax = axes[row, i]
        show_image(
            ax,
            vol[i].sum(0),
            cmap="gray_r",
            vmin=0,
            vmax=1500,
            pixel_size=1.0,
            scalebar=50 if (row, i) == (1, 0) else None,
        )
        if i == 0:
            ax.set_ylabel(label)
    axes[0, i].set_title(f"Image {i}")
fig.tight_layout()
save(fig, "specimens-crowding")
del alone, together

transcript.start("solvate")
# --8<-- [start:solvate]
volume = solvated.materialize(poses[:1], voltage=300.0, seed=0)[0]
dry = solvated.dry().materialize(poses[:1], voltage=300.0, seed=0)[0]
ice = volume - dry  # the ice alone, with the specimen's hole cut into it

print(f"volume {tuple(volume.shape)} (Z, Y, X)")
print(f"ice: mean {ice.mean():.2f} V over the box")
# --8<-- [end:solvate]

# A 16 Å slab through the centre, averaged along y: a single 1 Å slice
# resolves individual water molecules and hides the field they make.
y = volume.shape[1] // 2
slab = slice(y - 8, y + 8)
fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.7))
for i, (ax, (vol, title)) in enumerate(
    zip(axes, [(dry, "Specimen"), (ice, "Ice"), (volume, "Specimen + ice")])
):
    im = show_image(
        ax,
        vol[:, slab, :].mean(dim=1),
        cmap="gray_r",
        vmin=0,
        vmax=12,
        title=title,
        pixel_size=1.0,
        scalebar=100 if i == 0 else None,
    )
    ax.set_xlabel("x")
axes[0].set_ylabel("z (beam direction)")
fig.tight_layout()
fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02, label="V")
save(fig, "specimens-side-view")
del volume, dry, ice

transcript.start("thickness")
# --8<-- [start:thickness]
projections = {}
for thickness in [0.0, 100.0, 500.0, 1000.0, 2000.0]:
    s = crowded.solvate(model="gd", thickness=thickness)
    V = s.materialize(poses[:1], voltage=300.0, seed=0)[0]
    ice_only = V - s.dry().materialize(poses[:1], voltage=300.0, seed=0)[0]
    ice_projection = ice_only.sum(dim=0) * s.voxel_size
    projections[thickness] = V.sum(dim=0) * s.voxel_size
    print(
        f"{thickness:6.0f} Å requested: {s.nz:4d} slices, "
        f"ice projection mean {ice_projection.mean():6.0f} V·Å, "
        f"std {ice_projection.std():3.0f} V·Å"
    )
# --8<-- [end:thickness]

shown = [0.0, 500.0, 1000.0, 2000.0]
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6))
for ax, t in zip(axes, shown):
    p = as_numpy(projections[t])
    show_image(
        ax,
        p,
        cmap="gray_r",
        vmin=float(np.percentile(p, 0.5)),
        vmax=float(np.percentile(p, 99.5)),
        title=f"thickness={t:g} Å" if t else "thickness=0 (box depth)",
        pixel_size=1.0,
        scalebar=50 if t == 0.0 else None,
    )
fig.tight_layout()
save(fig, "specimens-thickness")
del projections

transcript.start("models")
# --8<-- [start:models]
empty = Specimen(torch.zeros(256, 256, 256, device="cuda"), voxel_size=1.0)

water = {}
for model in ["gd", "random"]:
    filled = empty.solvate(model=model)
    w = filled.materialize(Poses.random(1), voltage=300.0, seed=0)[0]
    water[model] = w
    print(f"{model:6s}: mean {w.mean():.2f} V, std {w.std():.2f} V")
# --8<-- [end:models]

k, p_gd = radial_power_3d(water["gd"], 1.0)
_, p_random = radial_power_3d(water["random"], 1.0)
structure_factor = p_gd / p_random
peak = k[np.argmax(structure_factor)]
# Quoted in the page's prose, not as output.
first = (k > 0.2) & (k < 0.35)
k1 = k[first][np.argmax(structure_factor[first])]
second = k > 0.4
k2 = k[second][np.argmax(structure_factor[second])]
low = (k > 0.03) & (k < 0.15)
for line in [
    f"first peak {k1:.3f} 1/Å ({1 / k1:.2f} Å), S = {structure_factor[first].max():.2f}",
    f"second peak {k2:.3f} 1/Å, S = {structure_factor[second].max():.2f}",
    f"0.03-0.15 1/Å: S from {structure_factor[low].min():.3f} "
    f"to {structure_factor[low].max():.3f}",
]:
    print(line, file=sys.__stdout__)

fig = plt.figure(figsize=(8.8, 2.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
for i, model in enumerate(["gd", "random"]):
    ax = fig.add_subplot(gs[0, i])
    show_image(
        ax,
        water[model][128, :96, :96],
        cmap="gray_r",
        vmin=0,
        vmax=40,
        title=f'model="{model}"',
        pixel_size=1.0,
        scalebar=20 if i == 0 else None,
    )
ax = fig.add_subplot(gs[0, 2])
ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
ax.plot(k, structure_factor, color=PALETTE[0], lw=1.2)
ax.axvline(1 / 3.7, color=PALETTE[1], lw=0.8, ls=":")
ax.text(1 / 3.7 + 0.008, 0.15, "3.7 Å", color=PALETTE[1], fontsize=8)
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel='Power, "gd" / "random"',
    title="Structure factor of the ice",
    xlim=(0, 0.5),
)
fig.tight_layout()
save(fig, "specimens-ice-models")
del water, empty

transcript.start("seeds")
# --8<-- [start:seeds]
a = solvated.materialize(poses, voltage=300.0, seed=0)
b = solvated.materialize(poses, voltage=300.0, seed=0)
c = solvated.materialize(poses[2:3], voltage=300.0, seed=0)
d = solvated.materialize(poses, voltage=300.0, seed=1)
print(f"same seed, same images:   {torch.equal(a, b)}")
print(f"image 2 drawn on its own: {torch.equal(a[2], c[0])}")
print(f"image 1 vs image 2:       {torch.equal(a[1], a[2])}")
print(f"seed 0 vs seed 1:         {torch.equal(a, d)}")
# --8<-- [end:seeds]
del a, b, c, d

transcript.start("damage")
# --8<-- [start:damage]
exposure = Exposure(dose=40.0)
damaged = specimen.crowd(min_distance=diameter).damage().solvate(model="gd")

for step in damaged.plan(voltage=300.0):
    print(step)
volumes = damaged.materialize(poses[:1], voltage=300.0, exposure=exposure, seed=0)
# --8<-- [end:damage]

transcript.close()
