"""
Figure and printed output for docs/recipes/membrane-organelles.md.

Run with: uv run python docs-figures/recipes/membrane_organelles.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("membrane_organelles")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt

from specter.specimen.membrane import MembraneGenerator

voxel_size = 12.0  # Å per voxel of the rendered volume
kinds = ["vesicle", "tubules", "cisterna", "mitochondrion"]

fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for column, kind in enumerate(kinds):
    generator = MembraneGenerator(
        kind=kind, voxel_size=voxel_size, device="cuda", seed=3
    )
    volume = generator.generate()  # potential in volts, indexed (Z, Y, X)

    drawn = ", ".join(f"{k} {v:.4g}" for k, v in generator.params.items())
    print(f"{kind}: {tuple(volume.shape)} voxels, {drawn}")

    central = volume[volume.shape[0] // 2]
    projection = volume.sum(0) * voxel_size
    for ax, image in zip(axes[:, column], [central, projection]):
        ax.imshow(image.cpu().numpy(), cmap="gray_r", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, column].set_title(kind)

axes[0, 0].set_ylabel("central slice")
axes[1, 0].set_ylabel("projection")
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "membrane_organelles")

thumbnail(as_numpy(projection), "membrane_organelles")

transcript.close()
