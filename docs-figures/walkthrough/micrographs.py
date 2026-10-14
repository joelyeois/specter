"""
Figures and printed output for docs/walkthrough/micrographs.md.

Run with: uv run python docs-figures/walkthrough/micrographs.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from _common import (
    PALETTE,
    Transcript,
    as_numpy,
    power_spectrum,
    radial_power,
    save,
    show_image,
)


def lowpass(image: torch.Tensor | np.ndarray, resolution: float, pixel_size: float):
    """`image` with a Gaussian low-pass whose 1/e point is at `resolution` Å."""
    data = as_numpy(image).astype(np.float64)
    ky = np.fft.fftfreq(data.shape[-2], pixel_size)[:, None]
    kx = np.fft.fftfreq(data.shape[-1], pixel_size)[None, :]
    envelope = np.exp(-((kx**2 + ky**2) * resolution**2))
    return np.fft.ifft2(np.fft.fft2(data) * envelope).real


transcript = Transcript("micrographs")

transcript.start("setup")
# --8<-- [start:setup]
import torch

from specter.imaging import CTF, Exposure, Field, Structure

structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=1.0, box=256, device="cuda")
# --8<-- [end:setup]

transcript.start("field")
# --8<-- [start:field]
field = Field(
    template.potential.cpu(),
    voxel_size=1.0,
    nxy=1536,
    molecular_mass=structure.molecular_mass,
)
field = field.pack(min_distance=structure.pdb.max_diameter)
field = field.solvate(model="gd", thickness=500.0)
# --8<-- [end:field]

transcript.start("draw")
# --8<-- [start:draw]
sample = field.draw(voltage=300.0, seed=0, index=0, keep_dry=True, device="cuda")

V = sample.volume
print(f"volume {tuple(V.shape)} on {V.device}, {V.numel() * 4 / 1e9:.1f} GB")
print(
    f"{sample.nz} slices of {sample.voxel_size} Å = {sample.nz * sample.voxel_size:.0f} Å"
)
# --8<-- [end:draw]

dry = as_numpy(sample.dry[0])
row = V.shape[-2] // 2
# A 32 Å thick section across the field, low-passed to 10 Å so the particles
# stand out from the molecular texture of the ice.
side = as_numpy(V[0, :, row - 16 : row + 16, :].mean(dim=1))
side = lowpass(side, 10.0, 1.0)
dry_side = dry[:, row - 16 : row + 16, :].mean(axis=1)
fig, axes = plt.subplots(
    3, 1, figsize=(9.0, 8.4), gridspec_kw={"height_ratios": [3.07, 1, 1]}
)
show_image(
    axes[0],
    dry.sum(0),
    cmap="gray_r",
    title="Particles, projected along z (before ice)",
    pixel_size=1.0,
    scalebar=500,
)
axes[0].axhline(row, color=PALETTE[3], lw=0.8, ls="--")
axes[0].set_aspect(1.0)
show_image(
    axes[1],
    dry_side,
    cmap="gray_r",
    title=f"Section at y = {row} Å, before ice",
    vmin=0,
    vmax=float(np.percentile(dry_side, 99.8)),
)
show_image(
    axes[2],
    side,
    cmap="gray_r",
    title=f"Section at y = {row} Å, with ice (low-passed to 10 Å)",
    vmin=float(np.percentile(side, 0.5)),
    vmax=float(np.percentile(side, 99.5)),
    pixel_size=1.0,
    scalebar=500,
)
fig.tight_layout()
save(fig, "micrographs-sample")

transcript.start("image")
# --8<-- [start:image]
waves = sample.scatter(voltage=300.0, device="cuda")
waves = waves.apply_ctf(CTF.from_values(defocus=15000.0))
intensity = waves.intensity(cc=2.7)

exposure = Exposure(dose=40.0, n_frames=40)
micrograph = intensity.detect(exposure, seed=0)
print(f"wave {tuple(waves.array.shape)}, image {tuple(micrograph.array.shape)}")
print(f"mean {micrograph.array.mean():.1f} counts per pixel")
# --8<-- [end:image]

noiseless = as_numpy(intensity.pixels()[0])
pad = (noiseless.shape[-1] - 1536) // 2
noiseless = noiseless[pad : pad + 1536, pad : pad + 1536] if pad else noiseless
detected = as_numpy(micrograph.array[0])
smooth = lowpass(detected, 15.0, 1.0)


def clip(image: np.ndarray) -> dict[str, float]:
    return {
        "vmin": float(np.percentile(image, 0.5)),
        "vmax": float(np.percentile(image, 99.5)),
    }


fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
show_image(
    axes[0],
    detected,
    title="Detected, 40 e⁻/Å²",
    pixel_size=1.0,
    scalebar=500,
    **clip(detected),
)
show_image(axes[1], smooth, title="Low-passed to 15 Å", **clip(smooth))
y0, x0, w = 384, 384, 512
box = (slice(y0, y0 + w), slice(x0, x0 + w))
show_image(
    axes[2],
    smooth[box],
    title="Low-passed, zoomed",
    pixel_size=1.0,
    scalebar=100,
    **clip(smooth[box]),
)
axes[1].add_patch(Rectangle((x0, y0), w, w, fill=False, ec=PALETTE[3], lw=1.2))
fig.tight_layout()
save(fig, "micrographs-image")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
ps = power_spectrum(detected)
show_image(
    ax1,
    ps,
    title="Power spectrum of the detected image",
    vmin=float(np.percentile(ps, 1)),
    vmax=float(np.percentile(ps, 99.7)),
)
k, p_noiseless = radial_power(noiseless, 1.0)
_, p_detected = radial_power(detected / detected.mean(), 1.0)
ax2.semilogy(
    k[2:], p_detected[2:], color=PALETTE[0], lw=1.0, label="Detected (normalised)"
)
ax2.semilogy(
    k[2:], p_noiseless[2:], color=PALETTE[1], lw=1.0, label="Noiseless intensity"
)
ax2.set(xlabel="Spatial frequency (1/Å)", ylabel="Power", title="Rotational average")
high = k > 0.35
print(
    f"[check] floor detected - noiseless above 0.35/A: "
    f"{(p_detected[high] - p_noiseless[high]).mean():.4f} (1/40 = 0.025)",
    file=sys.__stdout__,
)
print(
    f"[check] water-ring peak at k = {k[2:][np.argmax(p_noiseless[2:])]:.3f}/A",
    file=sys.__stdout__,
)
ax2.legend(frameon=False)
fig.tight_layout()
save(fig, "micrographs-power-spectrum")

transcript.start("defocus-series")
# --8<-- [start:defocus-series]
defoci = torch.tensor([5000.0, 15000.0, 30000.0])
waves = sample.scatter(voltage=300.0, ids=torch.arange(3), device="cuda")
series = waves.apply_ctf(CTF.from_values(defocus=defoci)).intensity(cc=2.7)
print(f"{len(series.ids)} images of one sample: {tuple(series.array.shape)}")
# --8<-- [end:defocus-series]

crop = (slice(384, 896), slice(384, 896))
images = [as_numpy(series.pixels()[i]) for i in range(3)]
if pad:
    images = [im[pad : pad + 1536, pad : pad + 1536] for im in images]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
for ax, image, df in zip(axes, images, defoci.tolist()):
    patch = lowpass(image, 10.0, 1.0)[crop]
    show_image(
        ax,
        patch,
        title=f"Defocus {df / 1e4:g} µm",
        vmin=float(np.percentile(patch, 0.5)),
        vmax=float(np.percentile(patch, 99.5)),
        pixel_size=1.0,
        scalebar=100,
    )
fig.tight_layout()
save(fig, "micrographs-defocus-series")

transcript.start("seeds")
# --8<-- [start:seeds]
again = field.draw(voltage=300.0, seed=0, index=0, device="cuda")
other = field.draw(voltage=300.0, seed=0, index=1, device="cuda")
print("seed 0, index 0 twice:", torch.equal(again.volume, sample.volume))
print("seed 0, index 0 vs 1: ", torch.equal(other.volume, sample.volume))
# --8<-- [end:seeds]
del again, other

transcript.start("profile")
# --8<-- [start:profile]
from specter.ice import IceProfile

wedge = IceProfile(mode="wedge", thickness_range=(200.0, 800.0))
tilted = Field(
    template.potential.cpu(),
    voxel_size=1.0,
    nxy=1024,
    molecular_mass=structure.molecular_mass,
)
tilted = tilted.pack(min_distance=structure.pdb.max_diameter)
tilted = tilted.solvate(model="gd", profile=wedge)
wedge_sample = tilted.draw(voltage=300.0, seed=0, device="cuda")
print(f"volume {tuple(wedge_sample.volume.shape)}: the thickest column sets the depth")

image = (
    wedge_sample.scatter(voltage=300.0, device="cuda")
    .apply_ctf(CTF.from_values(defocus=15000.0))
    .intensity(cc=2.7)
)
# --8<-- [end:profile]

thickness = as_numpy(wedge.thickness(1024, 1.0))
side = as_numpy(wedge_sample.volume[0, :, 512, :])
img = as_numpy(image.pixels()[0])
p = (img.shape[-1] - 1024) // 2
img = img[p : p + 1024, p : p + 1024] if p else img
img = lowpass(img, 15.0, 1.0)
side = lowpass(side, 10.0, 1.0)
fig, axes = plt.subplots(
    1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]}
)
axes[0].plot(np.arange(1024), thickness[512], color=PALETTE[0], lw=1.4)
axes[0].set(
    xlabel="x (Å)", ylabel="Ice thickness (Å)", title="Wedge profile", ylim=(0, 900)
)
show_image(
    axes[1],
    side,
    cmap="gray_r",
    title="Section at y = 512 Å",
    vmin=float(np.percentile(side, 0.5)),
    vmax=float(np.percentile(side, 99.5)),
)
axes[1].set_aspect(1.0)
show_image(
    axes[2],
    img,
    title="Noiseless intensity, low-passed",
    vmin=float(np.percentile(img, 0.5)),
    vmax=float(np.percentile(img, 99.5)),
    pixel_size=1.0,
    scalebar=200,
)
fig.tight_layout()
save(fig, "micrographs-ice-profile")

transcript.close()
