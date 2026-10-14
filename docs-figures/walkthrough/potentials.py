"""
Figures and printed output for docs/walkthrough/potentials.md.

Run with: uv run python docs-figures/walkthrough/potentials.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    PALETTE,
    Transcript,
    as_numpy,
    radial_power,
    save,
    show_image,
)

transcript = Transcript("potentials")

transcript.start("load")
# --8<-- [start:load]
from specter.imaging import Structure

structure = Structure.from_pdb("6bdf")

pdb = structure.pdb
print(f"atoms:            {len(pdb.coordinates):,}")
print(f"molecular mass:   {structure.molecular_mass / 1e3:.0f} kDa")
print(f"largest diameter: {pdb.max_diameter:.0f} Å")
print(f"first species:    {pdb.atom_species[:4]}")
# --8<-- [end:load]

transcript.start("render")
# --8<-- [start:render]
specimen = structure.render(voxel_size=1.0, box=256, device="cuda")

V = specimen.potential
print(f"shape {tuple(V.shape)}, {V.dtype}, on {V.device}")
print(f"peak {V.max():.1f} V, integral {V.sum() * specimen.voxel_size**3:.3e} V·Å³")
# --8<-- [end:render]

# --8<-- [start:project]
central_slice = V[V.shape[0] // 2]  # (Y, X) at the box's midplane
projected = V.sum(dim=0) * specimen.voxel_size  # V·Å, integrated along the beam
side_view = V.sum(dim=1) * specimen.voxel_size  # (Z, X), looking along y
# --8<-- [end:project]

fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
show_image(
    axes[0],
    central_slice,
    title="Central slice (V)",
    colorbar="V",
    cmap="gray_r",
    vmin=0,
    vmax=30,
    pixel_size=1.0,
    scalebar=50,
)
show_image(
    axes[1], projected, title="Projected along z (V·Å)", colorbar="V·Å", cmap="gray_r"
)
show_image(
    axes[2], side_view, title="Projected along y (V·Å)", colorbar="V·Å", cmap="gray_r"
)
fig.tight_layout()
save(fig, "potentials-render")

transcript.start("voxel-size")
# --8<-- [start:voxel-size]
renders = {}
for voxel_size, box in [(0.5, 512), (1.0, 256), (2.0, 128), (4.0, 64)]:
    s = structure.render(voxel_size=voxel_size, box=box, device="cuda")
    renders[voxel_size] = s
    integral = s.potential.sum() * voxel_size**3
    print(
        f"{voxel_size:3.1f} Å  box {box:3d}  peak {s.potential.max():6.1f} V  "
        f"integral {integral:.4e} V·Å³"
    )
# --8<-- [end:voxel-size]

fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5))
for ax, (voxel_size, s) in zip(axes, renders.items()):
    show_image(
        ax,
        s.potential.sum(0) * voxel_size,
        cmap="gray_r",
        title=f"{voxel_size:g} Å voxels",
        pixel_size=voxel_size,
        scalebar=50,
    )
fig.tight_layout()
save(fig, "potentials-voxel-size")

transcript.start("parameterizations")
# --8<-- [start:parameterizations]
projections = {}
for factors in ["shtyrov", "kirkland", "lobato"]:
    s = Structure.from_pdb("6bdf", scattering_factors=factors).render(
        voxel_size=1.0, box=256, device="cuda"
    )
    projections[factors] = s.potential.sum(dim=0) * s.voxel_size
    print(f"{factors:9s} integral {s.potential.sum():.4e} V·Å³")
# --8<-- [end:parameterizations]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9))
row = 128
_, reference = radial_power(projections["shtyrov"], 1.0)
for color, (name, proj) in zip(PALETTE, projections.items()):
    p = as_numpy(proj)
    ax1.plot(
        np.arange(p.shape[-1]) - p.shape[-1] // 2,
        p[row],
        color=color,
        lw=1.0,
        label=name,
    )
    k, power = radial_power(p, 1.0)
    ax2.plot(k[1:], power[1:] / reference[1:], color=color, lw=1.0, label=name)
ax1.set(
    xlabel="x (Å)",
    ylabel="Projected potential (V·Å)",
    title="Line profile through the centre",
    xlim=(-65, -5),
)
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power relative to shtyrov",
    title="Rotationally averaged power spectrum",
)
ax1.legend(frameon=False)
fig.tight_layout()
save(fig, "potentials-parameterizations")

transcript.start("bfactors")
# --8<-- [start:bfactors]
blurred = structure.render(
    voxel_size=1.0, box=256, use_deposited_bfactors=True, device="cuda"
)
print(f"mean deposited B-factor {pdb.b_factors.float().mean():.1f} Å²")
print(f"peak {V.max():.1f} V -> {blurred.potential.max():.1f} V")
# --8<-- [end:bfactors]

sharp_p = as_numpy(V.sum(0))
blur_p = as_numpy(blurred.potential.sum(0))
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
crop = (slice(76, 116), slice(108, 148))  # a 40 Å patch of the ring wall
vmax = float(np.percentile(sharp_p[crop], 99.5))
show_image(
    axes[0],
    sharp_p[crop],
    cmap="gray_r",
    title="Without B-factors",
    vmin=0,
    vmax=vmax,
    pixel_size=1.0,
    scalebar=10,
)
show_image(
    axes[1], blur_p[crop], cmap="gray_r", title="Deposited B-factors", vmin=0, vmax=vmax
)
k, p_sharp = radial_power(sharp_p, 1.0)
_, p_blur = radial_power(blur_p, 1.0)
axes[2].plot(k[1:], p_blur[1:] / p_sharp[1:], color=PALETTE[0], lw=1.2)
axes[2].set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power ratio",
    title="Blurred / sharp",
    ylim=(0, 1.05),
)
fig.tight_layout()
save(fig, "potentials-bfactors")

transcript.close()
