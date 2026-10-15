"""
Figures and printed output for docs/user-guide/build-tomogram.md.

Runs the guide's own `specter build tomogram` command, with the shipped
config, in a scratch directory, and plots the volume and its ground truth.

Run with: uv run python docs-figures/cli_guides/build_tomogram.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from matplotlib.colors import ListedColormap

from _common import (
    PALETTE,
    REPO_ROOT,
    clipped,
    lowpass,
    run,
    save,
    show_image,
    tidy,
    write_text,
)

COMMAND = """specter build tomogram \\
    --config configs/tomogram.toml \\
    --voxel_size 5.0 \\
    --seed 0 \\
    --device cuda \\
    --output_dir tomograms"""
VOXEL = 5.0
TARGETS = ["7VD8", "1FA2", "6DRV", "6QZP", "7B75", "6N4V"]

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-tomogram-"))
(workdir / "configs").mkdir()
shutil.copy(REPO_ROOT / "configs" / "tomogram.toml", workdir / "configs")
write_text("build_tomogram-command", COMMAND + "\n")
log = tidy(run(COMMAND, workdir))
write_text(
    "build_tomogram-run",
    "\n".join(
        line
        for line in log.splitlines()
        if "UserWarning" not in line and not line.endswith(".ndjson")
    )
    + "\n",
)

out = workdir / "tomograms"
picks = {
    path.name.split("-")[0]: sum(1 for _ in path.open())
    for path in sorted(out.glob("*_orientedpoint.ndjson"))
}
lines = [
    f"{path.name}  {path.stat().st_size / 1e9:.2f} GB"
    for path in sorted(out.glob("*.mrc"))
]
lines.append(f"{len(picks)} species with picks, {sum(picks.values())} instances")
lines.append(
    "targets: " + ", ".join(f"{name} {picks.get(name, 0)}" for name in TARGETS)
)
write_text("build_tomogram-files", "\n".join(lines) + "\n")

# The slice shown is the one cutting through the most bilayer, and the side
# view runs along the row of that slice with the most bilayer, so both pass
# through membranes rather than only through the protein fill.
bilayer = np.asarray(mrcfile.read(out / "tomogram_membrane_labels.mrc")) > 0
z_show = int(bilayer.sum((1, 2)).argmax())
y_show = int(bilayer[z_show].sum(1).argmax())
del bilayer
regions_volume = np.asarray(mrcfile.read(out / "tomogram_regions.mrc"))
fractions = np.bincount(regions_volume.ravel(), minlength=4) / regions_volume.size
regions = regions_volume[z_show].astype(np.int64)
del regions_volume
with mrcfile.mmap(out / "tomogram.mrc", mode="r") as mrc:
    nz = mrc.data.shape[0]
    z0 = min(max(z_show - 4, 0), nz - 8)
    slab = np.asarray(mrc.data[z0 : z0 + 8]).sum(0) * VOXEL  # 40 Å, V·Å
    y0 = min(max(y_show - 40, 0), mrc.data.shape[1] - 80)
    side = np.asarray(mrc.data[:, y0 : y0 + 80, :]).sum(1) * VOXEL  # 400 Å, V·Å
with mrcfile.mmap(out / "tomogram_protein_labels.mrc", mode="r") as mrc:
    proteins = np.asarray(mrc.data[z_show]).astype(np.int64)
with mrcfile.mmap(out / "tomogram_membrane_labels.mrc", mode="r") as mrc:
    membranes = np.asarray(mrc.data[z_show]).astype(np.int64)

# The volume: a 40 Å slab through that slice, and a 400 Å slab seen side on.
fig = plt.figure(figsize=(10.0, 7.8))
grid = fig.add_gridspec(2, 2, height_ratios=[4.8, 2.6])
ax_top = fig.add_subplot(grid[0, 0])
smooth = lowpass(slab, VOXEL, 20.0)
show_image(
    ax_top,
    smooth,
    cmap="gray_r",
    title=f"40 Å slab at z = {z_show}, {slab.shape[-1] * VOXEL:.0f} Å across",
    **clipped(smooth, 1, 99.5),
)
ax_top.axhline(y_show, color=PALETTE[1], lw=0.8, ls="--")
cy = min(max(y_show, 200), slab.shape[0] - 200)
cx = slab.shape[1] // 2
ax_top.add_patch(
    plt.Rectangle((cx - 200, cy - 200), 400, 400, fill=False, color=PALETTE[0], lw=1.0)
)
ax_zoom = fig.add_subplot(grid[0, 1])
zoom = smooth[cy - 200 : cy + 200, cx - 200 : cx + 200]
show_image(
    ax_zoom,
    zoom,
    cmap="gray_r",
    title="2000 Å of it (boxed)",
    pixel_size=VOXEL,
    scalebar=500,
    **clipped(zoom, 1, 99.5),
)
ax_side = fig.add_subplot(grid[1, :])
side_smooth = lowpass(side, VOXEL, 20.0)
show_image(
    ax_side,
    side_smooth,
    cmap="gray_r",
    title="400 Å slab along the dashed line, seen from the side (x against z)",
    **clipped(side_smooth, 1, 99.5),
)
fig.tight_layout()
save(fig, "build_tomogram-volume")

# Ground truth on the same slice: protein instances, membranes, regions.
rng = np.random.default_rng(0)
colours = rng.uniform(0.2, 0.95, size=(int(proteins.max()) + 1, 3))
colours[0] = 1.0
tab10 = plt.get_cmap("tab10")(np.arange(10))[:, :3]
membrane_rgb = np.where(
    (membranes > 0)[..., None], tab10[(membranes - 1) % 10], np.ones(3)
)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
axes[0].imshow(colours[proteins], interpolation="nearest")
axes[0].set_title("protein_labels: one id per instance")
axes[1].imshow(membrane_rgb, interpolation="nearest")
axes[1].set_title("membrane_labels: one id per membrane")
region_colours = ListedColormap(["#f2f2f2", PALETTE[3], PALETTE[0], "#7f7f7f"])
axes[2].imshow(regions, cmap=region_colours, vmin=0, vmax=3, interpolation="nearest")
axes[2].set_title("regions: cytosol, shell, lumen, film")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
save(fig, "build_tomogram-labels")

write_text(
    "build_tomogram-labels",
    f"slice z = {z_show}: {len(np.unique(proteins)) - 1} protein instances, "
    f"{len(np.unique(membranes)) - 1} membranes\n"
    "whole volume: "
    + ", ".join(
        f"{name} {fractions[value]:.2%}"
        for value, name in [
            (0, "cytosol"),
            (1, "shell"),
            (2, "lumen"),
            (3, "carbon film"),
        ]
    )
    + "\n",
)
shutil.rmtree(workdir)
