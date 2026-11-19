"""
Figure and printed output for docs/tutorials/membrane-primitives.md.

Run with: uv run python docs-figures/tutorials/membrane_primitives.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("membrane_primitives")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt

import specter.specimen.membrane as membrane

# 1. The coarse working grid: a box in Å, ordered (x, y, z).
grid = membrane.Grid((1300.0, 1000.0, 1000.0), device="cuda")

# 2. The solid, as a signed distance: two balls joined by a neck.
a = grid.centre + [-200.0, 0.0, 0.0]
b = grid.centre + [330.0, 40.0, 0.0]
raw = membrane.union(
    membrane.ball(grid, a, 300.0),
    membrane.ball(grid, b, 170.0),
    membrane.tube(grid, [a, (a + b) / 2, b], 62.0),
)

# 3. Roll a ball over both sides, and measure the reach of the result.
smooth = membrane.regularise(raw, grid)
for name, phi in [("raw union", raw), ("regularised", smooth)]:
    points, normals, outer, inner = membrane.reach(phi, grid)
    print(f"{name}: reach {outer.min():.0f} Å outside, {inner.min():.0f} Å inside")

# 4. Render the bilayer profile around the surface, on any output grid.
voxel_size = 5.0
profile = membrane.build_measured_bilayer_profile(
    thickness_angstrom=37.0, device="cuda"
)
shape_zyx = tuple(round(extent / voxel_size) for extent in grid.box[::-1])
volume = membrane.render_bilayer(
    [smooth], grid, profile, shape_zyx, voxel_size, origin_xyz=tuple(grid.lo)
)
print(f"rendered {tuple(volume.shape)} voxels, peak {volume.max():.1f} V")

# The mid-surface (phi = 0) on the central z-plane, and the rendered slice.
x, y, z = (axis.flatten().cpu().numpy() for axis in grid.axes())
k = len(z) // 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for phi, color, name in [(raw, "C1", "raw union"), (smooth, "C0", "regularised")]:
    ax1.contour(x, y, phi[:, :, k].cpu().numpy().T, levels=[0.0], colors=color)
    ax1.plot([], [], color=color, label=name)
ax1.set(xlim=(0.0, grid.box[0]), ylim=(0.0, grid.box[1]), aspect="equal")
ax1.set(xlabel="x (Å)", ylabel="y (Å)", title="Mid-surface")
ax1.legend(frameon=False)
ax2.imshow(
    volume[volume.shape[0] // 2].cpu().numpy(),
    cmap="gray_r",
    origin="lower",
    extent=(0.0, grid.box[0], 0.0, grid.box[1]),
)
ax2.set(xlabel="x (Å)", ylabel="y (Å)", title="Rendered potential, central slice")
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "membrane_primitives")

thumbnail(as_numpy(volume[volume.shape[0] // 2])[:, 30:230], "membrane_primitives")

transcript.close()
