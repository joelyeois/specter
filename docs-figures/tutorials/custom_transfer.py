"""
Figure and printed output for docs/tutorials/custom-transfer-function.md.

Run with: uv run python docs-figures/tutorials/custom_transfer.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("custom_transfer")

transcript.start("recipe")
# --8<-- [start:recipe]
from dataclasses import replace

import matplotlib.pyplot as plt
import torch

from specter.imaging import CTF, Exposure, Poses, Structure


def knife_edge(wave: torch.Tensor, dx: float, softness: float = 0.002) -> torch.Tensor:
    """
    Block the scattered beams on one side of the back focal plane.

    `wave` is complex ``(n, Y, X)`` sampled every `dx` Å. The edge is at
    kx = 0 and rises over `softness` (1/Å); the unscattered beam, on the
    edge, passes at full amplitude.
    """
    kx = torch.fft.fftfreq(wave.shape[-1], d=dx, device=wave.device)
    edge = torch.clamp(1.0 + kx / softness, 0.0, 1.0)  # 0 for kx < -softness
    return torch.fft.ifft2(torch.fft.fft2(wave) * edge)


specimen = Structure.from_pdb("6bdf").render(voxel_size=2.0, box=128, device="cuda")
wave = specimen.scatter(Poses.random(1, seed=3), voltage=300.0)
exposure = Exposure(dose=40.0)

# The built-in lens, in focus, so that the edge is the only source of contrast.
in_focus = wave.apply_ctf(CTF.from_values(defocus=0.0))

# The custom stage acts on the wave's array, on the wave's own grid.
dx = in_focus.pixel_size / in_focus.oversample
edged = replace(in_focus, array=knife_edge(in_focus.array, dx))

images = {
    "in focus": in_focus,
    "in focus, knife edge": edged,
    "1.5 µm underfocus": wave.apply_ctf(CTF.from_values(defocus=15000.0)),
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (title, lensed) in zip(axes, images.items()):
    counts = lensed.intensity().detect(exposure, noise=None).array[0]
    print(f"{title}: contrast {counts.std() / counts.mean():.3f}")
    shown = ax.imshow(counts.cpu().numpy(), cmap="gray", origin="lower")
    ax.set(title=title, xticks=[], yticks=[])
    fig.colorbar(shown, ax=ax, shrink=0.8, label="expected counts")
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "custom_transfer")

relief = edged.intensity().detect(exposure, noise=None).array[0]
thumbnail(as_numpy(relief), "custom_transfer", cmap="gray")

transcript.close()
