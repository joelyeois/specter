"""
The overview figure for docs/walkthrough/index.md: one particle followed
through every stage of image formation, one panel per object.

Run with: uv run python docs-figures/walkthrough/chain.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch

from _common import as_numpy, save, show_image
from specter.imaging import CTF, Exposure, Poses, Structure

structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=1.0, box=256, device="cuda")
specimen = template.crowd(min_distance=structure.pdb.max_diameter).solvate(
    model="gd", thickness=0.0
)

# Looking down the ring's axis, so the particle reads at every stage.
poses = Poses(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(1, 2), torch.arange(1))
exposure = Exposure(dose=40.0)
volume = specimen.materialize(poses, voltage=300.0, exposure=exposure, seed=3)
waves = specimen.scatter(poses, voltage=300.0, exposure=exposure, seed=3)
lensed = waves.apply_ctf(CTF.from_values(defocus=15000.0))
intensity = lensed.intensity(cc=2.7)
image = intensity.detect(exposure, detector="k3_300kv", seed=3)

n = template.nxy
pad = (waves.pad_nxy - n) // 2


def crop(x: torch.Tensor) -> torch.Tensor:
    """The particle box out of the FFT-padded pixel grid."""
    return x[..., pad : pad + n, pad : pad + n]


def lowpass(x: torch.Tensor, resolution: float = 5.0) -> torch.Tensor:
    """A Gaussian low-pass to `resolution` Å, for display only."""
    x = x.float()
    k = torch.fft.fftfreq(x.shape[-1], d=template.voxel_size, device=x.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(-k2 * resolution**2)).real


panels = [
    ("Specimen", "rendered potential", template.potential.sum(0), "gray_r"),
    ("Specimen", "crowded, in ice", crop(volume[0].sum(0)), "gray_r"),
    (
        "ExitWave",
        "phase after scattering",
        crop(torch.angle(waves.pixels()[0])),
        "gray",
    ),
    (
        "ExitWave",
        "after the lens",
        lowpass(crop(lensed.pixels()[0].abs() ** 2)),
        "gray",
    ),
    ("Image", "detected counts", lowpass(image.array[0]), "gray"),
]
verbs = [
    "Structure.render()",
    ".crowd() .solvate()",
    ".scatter()",
    ".apply_ctf()",
    ".intensity() .detect()",
]

fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 2.9))
for ax, (name, what, data, cmap) in zip(axes, panels):
    show_image(ax, as_numpy(data), cmap=cmap)
    ax.set_title(f"{name}\n{what}", fontsize=9)
fig.tight_layout(w_pad=2.6)
fig.subplots_adjust(bottom=0.2)
for i in range(len(panels) - 1):
    left = axes[i].get_position()
    right = axes[i + 1].get_position()
    y = left.y0 + left.height / 2
    arrow = FancyArrowPatch(
        (left.x1 + 0.004, y),
        (right.x0 - 0.004, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        color="0.35",
        lw=1.0,
    )
    fig.patches.append(arrow)
for i, verb in enumerate(verbs):
    box = axes[i].get_position()
    fig.text(
        box.x0 + box.width / 2,
        box.y0 - 0.06,
        verb,
        ha="center",
        va="top",
        fontsize=8,
        family="monospace",
        color="0.25",
    )
save(fig, "walkthrough-chain")
