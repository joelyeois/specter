"""
Figure and printed output for docs/tutorials/save-micrographs.md.

Run with: uv run python docs-figures/tutorials/save_micrographs.py
"""

from __future__ import annotations

import os
import tempfile

from _common import Transcript, save, thumbnail

transcript = Transcript("save_micrographs")

os.chdir(tempfile.mkdtemp())
transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt
import mrcfile
import starfile
import torch

from specter.imaging import CTF, Exposure, Field, Structure
from specter.io import create_micrograph_starfile, read_micrograph_starfile

# Three micrographs of one field, at three defoci.
structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=2.0, box=128, device="cuda")
field = Field(
    template.potential.cpu(),
    voxel_size=2.0,
    nxy=512,
    molecular_mass=structure.molecular_mass,
)
field = field.pack(min_distance=structure.pdb.max_diameter)
field = field.solvate(model="gd", thickness=400.0)
sample = field.draw(voltage=300.0, seed=0, index=0, device="cuda")

ctf = CTF.from_values(defocus=torch.tensor([8000.0, 15000.0, 25000.0]))
dose = 40.0  # e⁻/Å²
waves = sample.scatter(voltage=300.0, ids=torch.arange(3), device="cuda")
micrographs = waves.apply_ctf(ctf).intensity(cc=2.7)
micrographs = micrographs.detect(Exposure(dose=dose, n_frames=40), seed=0)

# The images, and one STAR row per image with its transfer function.
micrographs.save("micrographs.mrcs")
star_path = create_micrograph_starfile(
    len(micrographs.ids),
    voltage=300.0,
    pixel_size=micrographs.pixel_size,
    alpha=waves.alpha,
    ctf_params=ctf.params,
    filename="micrographs",
    dose_per_angstrom=dose,
)

# Read both back, as a downstream program would.
table = starfile.read(star_path)
print(table.T.to_string(header=False))
with mrcfile.open("micrographs.mrcs") as mrc:
    stack = mrc.data
    print(f"stack {stack.shape}, {mrc.voxel_size.x:.1f} Å per pixel")
defoci = read_micrograph_starfile(star_path)["dfu"]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, image, defocus in zip(axes, stack, defoci.tolist()):
    ax.imshow(image, cmap="gray", origin="lower")
    ax.set(title=f"defocus {defocus / 1e4:.1f} µm", xticks=[], yticks=[])
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "save_micrographs")

thumbnail(stack[1], "save_micrographs", cmap="gray")

transcript.close()
