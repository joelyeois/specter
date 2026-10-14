"""
Square thumbnails for docs/gallery.md, one per walkthrough page that has no
thumbnail of its own (the particle-stack, micrograph, tilt-series and
reconstruction pages reuse docs/basic-examples.md's).

Run with: uv run python docs-figures/walkthrough/gallery.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import as_numpy, power_spectrum, save, show_image
from specter.aberrations._envelopes import cc_envelope
from specter.constants import energy_to_wavelength
from specter.imaging import CTF, Exposure, Poses, Structure


def square(image: np.ndarray, name: str, cmap: str = "gray") -> None:
    """One borderless square panel, contrast-stretched to 1-99%."""
    lo, hi = np.percentile(image, [1, 99])
    fig, ax = plt.subplots(figsize=(3, 3))
    show_image(ax, image, cmap=cmap, vmin=lo, vmax=hi)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, name)


def lowpass(x: torch.Tensor, pixel_size: float, resolution: float) -> torch.Tensor:
    """A Gaussian low-pass to `resolution` Å, for display only."""
    k = torch.fft.fftfreq(x.shape[-1], d=pixel_size, device=x.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    return torch.fft.ifft2(
        torch.fft.fft2(x.float()) * torch.exp(-k2 * resolution**2)
    ).real


structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=1.0, box=256, device="cuda")
top = Poses(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(1, 2), torch.arange(1))

# Structures and potentials: the projected potential.
square(
    as_numpy(template.potential.sum(0))[48:208, 48:208], "gallery-potentials", "gray_r"
)

# Specimens: a side view of the particle in 500 Å of ice, with its neighbours.
specimen = template.crowd(min_distance=192.0).solvate(model="gd", thickness=500.0)
side = Poses.random(1, seed=2)
volume = specimen.materialize(side, voltage=300.0, seed=1)[0]
c = volume.shape[-1] // 2
slab = as_numpy(volume[:, c - 8 : c + 8].sum(1))  # (Z, X), 16 Å thick
z0 = (slab.shape[0] - slab.shape[1]) // 2
square(slab[max(z0, 0) : max(z0, 0) + slab.shape[1]], "gallery-specimens", "gray_r")

# Scattering: the exit wave's phase in ice, looking down the ring.
waves = specimen.scatter(top, voltage=300.0, seed=1)
pad = (waves.pad_nxy - waves.nxy) // 2
phase = torch.angle(waves.pixels()[0])[pad : pad + 256, pad : pad + 256]
square(as_numpy(lowpass(phase, 1.0, 3.0)), "gallery-scattering")

# Contrast transfer function: Thon rings of the particle in ice at 1 µm.
image = waves.apply_ctf(CTF.from_values(defocus=10000.0)).intensity(cc=None)
pixels = image.pixels()[0][pad : pad + 256, pad : pad + 256]
square(power_spectrum(pixels), "gallery-ctf")

# Partial coherence: sin chi at 120 kV, damped by the focal spread's envelope.
n, dx, voltage = 768, 1.5, 120.0  # well sampled out to 0.33 1/Å
wavelength = energy_to_wavelength(voltage)
k = torch.fft.fftshift(torch.fft.fftfreq(n, d=dx))
k2 = k[:, None] ** 2 + k[None, :] ** 2
chi = np.pi * wavelength * k2 * (0.5 * 2.7e7 * wavelength**2 * k2 - 10000.0)
envelope = cc_envelope(k2, wavelength, 2.7e7, voltage * 1e3, 0.7, 0.06e-6, 0.01e-6)
square(as_numpy(torch.sin(chi) * envelope), "gallery-coherence", "RdBu_r")

# Detection: one particle at four doses, low-passed to 8 Å.
lensed = waves.apply_ctf(CTF.from_values(defocus=15000.0)).intensity(cc=2.7)
panels = []
for dose in (5.0, 20.0, 60.0, 200.0):
    counts = lensed.detect(Exposure(dose=dose), seed=0).array[0]
    panels.append(as_numpy(lowpass(counts, 1.0, 8.0))[48:208, 48:208])
fig, axes = plt.subplots(2, 2, figsize=(3, 3))
for ax, panel in zip(axes.ravel(), panels):
    lo, hi = np.percentile(panel, [1, 99])
    show_image(ax, panel, vmin=lo, vmax=hi)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.subplots_adjust(0, 0, 1, 1, wspace=0.02, hspace=0.02)
save(fig, "gallery-detection")
