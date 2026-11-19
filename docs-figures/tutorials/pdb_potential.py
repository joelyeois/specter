"""
Figure and printed output for docs/tutorials/pdb-potential.md.

Run with: uv run python docs-figures/tutorials/pdb_potential.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("pdb_potential")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt

from specter.imaging import Structure

voxel_size = 1.0  # Å per voxel
structure = Structure.from_pdb("6bdf")  # an accession code, or a local file
specimen = structure.render(voxel_size=voxel_size, box=256, device="cuda")
potential = specimen.potential  # volts, indexed (Z, Y, X)

print(
    f"{len(structure.pdb.coordinates)} atoms, {structure.molecular_mass / 1e3:.0f} kDa"
)
print(f"largest extent {structure.pdb.max_diameter:.0f} Å")
print(f"potential {tuple(potential.shape)} voxels, peak {potential.max():.0f} V")

nz, ny, nx = potential.shape
extent = (0.0, nx * voxel_size, 0.0, ny * voxel_size)
panels = {
    "central slice (V)": potential[nz // 2],
    "projection along z (V·Å)": potential.sum(0) * voxel_size,
    "projection along y (V·Å)": potential.sum(1) * voxel_size,
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, (title, image) in zip(axes, panels.items()):
    shown = ax.imshow(image.cpu().numpy(), cmap="gray_r", origin="lower", extent=extent)
    ax.set(title=title, xlabel="Å")
    fig.colorbar(shown, ax=ax, shrink=0.8)
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "pdb_potential")

thumbnail(as_numpy(potential.sum(0)), "pdb_potential")

transcript.close()
