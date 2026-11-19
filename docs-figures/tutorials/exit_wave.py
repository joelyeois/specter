"""
Figure and printed output for docs/tutorials/exit-wave.md.

Run with: uv run python docs-figures/tutorials/exit_wave.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("exit_wave")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt
import torch

from specter.imaging import CTF, Poses, Structure

specimen = Structure.from_pdb("6bdf").render(voxel_size=2.0, box=128, device="cuda")
poses = Poses.random(1, seed=3)

wave = specimen.scatter(poses, voltage=300.0)  # multislice by default
print(f"array {tuple(wave.array.shape)}, {wave.array.dtype}")
print(f"image {wave.nxy} px, propagated on {wave.pad_nxy} px, {wave.oversample}x finer")
print(f"exit face to the defocus plane: {wave.defocus_shift:.0f} Å")

psi = wave.pixels()[0]  # complex, on the pixel grid
phase = torch.angle(psi)
print(f"|psi| from {psi.abs().min():.3f} to {psi.abs().max():.3f}")
print(f"largest phase {phase.max():.2f} rad")

# The same wave after the lens, and the intensity it deposits.
image = wave.apply_ctf(CTF.from_values(defocus=15000.0)).intensity()

panels = {
    "exit wave: phase (rad)": phase,
    "exit wave: amplitude": psi.abs(),
    "intensity at 1.5 µm underfocus": image.pixels()[0],
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (title, panel) in zip(axes, panels.items()):
    shown = ax.imshow(panel.cpu().numpy(), cmap="gray", origin="lower")
    ax.set(title=title, xticks=[], yticks=[])
    fig.colorbar(shown, ax=ax, shrink=0.8)
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "exit_wave")

thumbnail(as_numpy(phase), "exit_wave", cmap="gray")

transcript.close()
