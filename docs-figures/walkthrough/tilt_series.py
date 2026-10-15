"""
Figures and printed output for docs/walkthrough/tilt-series.md.

Run with: uv run python docs-figures/walkthrough/tilt_series.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from _common import PALETTE, Transcript, as_numpy, save, show_image
from specter.potential import ice_inelastic_mfp


def lowpass(image: torch.Tensor | np.ndarray, resolution: float, pixel_size: float):
    """`image` with a Gaussian low-pass whose 1/e point is at `resolution` Å."""
    data = as_numpy(image).astype(np.float64)
    ky = np.fft.fftfreq(data.shape[-2], pixel_size)[:, None]
    kx = np.fft.fftfreq(data.shape[-1], pixel_size)[None, :]
    envelope = np.exp(-((kx**2 + ky**2) * resolution**2))
    return np.fft.ifft2(np.fft.fft2(data) * envelope).real


def clip(image: np.ndarray) -> dict[str, float]:
    return {
        "vmin": float(np.percentile(image, 0.5)),
        "vmax": float(np.percentile(image, 99.5)),
    }


transcript = Transcript("tilt-series")

transcript.start("setup")
# --8<-- [start:setup]
import torch

from specter.imaging import CTF, Exposure, Field, Poses, Sample, Structure
from specter.settings import Ice

structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=4.0, box=64, device="cuda")
# --8<-- [end:setup]

transcript.start("volume")
# --8<-- [start:volume]
field = Field(
    template.potential.cpu(),
    voxel_size=4.0,
    nxy=1056,
    nz=150,
    molecular_mass=structure.molecular_mass,
)
field = field.pack(min_distance=structure.pdb.max_diameter)
volume = field.draw(voltage=300.0, seed=0).volume  # no ice: (1, Z, Y, X)

sample = Sample.from_volume(volume, voxel_size=4.0)
print(
    f"specimen {tuple(volume.shape)}: {sample.nz * 4.0:.0f} Å thick, "
    f"{sample.nxy * 4.0:.0f} Å across"
)
# --8<-- [end:volume]

V = as_numpy(volume[0])
row = V.shape[1] // 2
fig, axes = plt.subplots(
    2, 1, figsize=(7.0, 8.3), gridspec_kw={"height_ratios": [7.04, 1]}
)
show_image(
    axes[0],
    V.sum(0),
    cmap="gray_r",
    title="Specimen, projected along z",
    pixel_size=4.0,
    scalebar=1000,
)
axes[0].axhline(row, color=PALETTE[3], lw=0.8, ls="--")
c0 = (V.shape[-1] - 384) / 2
axes[0].add_patch(Rectangle((c0, c0), 384, 384, fill=False, ec=PALETTE[0], lw=1.4))
side = V[:, row - 4 : row + 4, :].mean(axis=1)
show_image(
    axes[1],
    side,
    cmap="gray_r",
    title=f"Section at y = {row * 4} Å",
    vmin=0,
    vmax=float(np.percentile(side, 99.8)),
    pixel_size=4.0,
    scalebar=1000,
)
fig.tight_layout()
save(fig, "tilt-series-specimen")

transcript.start("acquisition")
# --8<-- [start:acquisition]
angles = [0.0]
for step in range(3, 61, 3):  # dose-symmetric: 0, +3, -3, +6, -6, ...
    angles += [float(step), float(-step)]

poses = Poses.tilt_series(angles)
exposure = Exposure.tilt_series(3.0, len(angles), n_frames=10)
print(f"{len(poses)} tilts, {exposure.doses(len(poses)).sum():.0f} e⁻/Å² in total")
print(
    f"pre-exposure of the last tilt ({angles[-1]:+.0f}°): "
    f"{exposure.pre_exposure[-1]:.0f} e⁻/Å²"
)
# --8<-- [end:acquisition]

pre = as_numpy(exposure.pre_exposure)
ang = np.array(angles)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0))
ax1.plot(np.arange(len(ang)), ang, "o-", color=PALETTE[0], ms=3, lw=0.8)
ax1.set(
    xlabel="Acquisition order", ylabel="Tilt angle (°)", title="Dose-symmetric order"
)
order = np.argsort(ang)
ax2.bar(ang[order], pre[order], width=2.4, color=PALETTE[0])
ax2.set(
    xlabel="Tilt angle (°)",
    ylabel="Pre-exposure (e⁻/Å²)",
    title="Dose received before each tilt",
)
fig.tight_layout()
save(fig, "tilt-series-acquisition")

transcript.start("series")
# --8<-- [start:series]
series = sample.tilt_series(
    poses,
    exposure,
    voltage=300.0,
    micrograph_size=384,
    ice=Ice(model="gd"),
    absorption_model="inelastic_mfp",
    seed=0,
)
coverage = series.coverage
print(
    f"±{coverage.max_tilt_angle_deg:.0f}° needs {coverage.recommended_nxy_for_max_tilt} px "
    f"across (+{coverage.edge_margin} px margin each side); the volume has {sample.nxy}"
)
# --8<-- [end:series]

transcript.start("scatter")
# --8<-- [start:scatter]
ctf = CTF.from_values(defocus=30000.0, n=len(series))

images, tilts = [], []
for i in range(len(series)):
    wave = series.scatter(i, device="cuda")
    image = wave.apply_ctf(ctf[i : i + 1]).intensity(cc=2.7)
    images.append(image)
    tilts.append(image.detect(exposure[i : i + 1], seed=0))

print(
    f"tilt {angles[-1]:+.0f}°: wave {tuple(wave.array.shape)}, "
    f"image {tuple(tilts[-1].array.shape)}"
)
# --8<-- [end:scatter]


def field(image) -> torch.Tensor:
    """The first image on the field's pixels, the FFT margin cropped."""
    pixels = image.pixels()[0]
    m = (image.pad_nxy - image.nxy) // 2
    return pixels[m : m + image.nxy, m : m + image.nxy]


show = [-60.0, -30.0, 0.0, 30.0, 60.0]
fig, axes = plt.subplots(2, len(show), figsize=(2.0 * len(show), 4.3))
for col, a in enumerate(show):
    i = angles.index(a)
    clean = lowpass(field(images[i]), 12.0, 4.0)
    noisy = lowpass(tilts[i].array[0], 12.0, 4.0)
    show_image(
        axes[0, col],
        clean,
        title=f"{a:+.0f}°  (#{i})",
        **clip(clean),
        pixel_size=4.0 if col == 0 else None,
        scalebar=500 if col == 0 else None,
    )
    show_image(axes[1, col], noisy, **clip(noisy))
axes[0, 0].set_ylabel("Noiseless")
axes[1, 0].set_ylabel("Detected, 3 e⁻/Å²")
fig.tight_layout()
save(fig, "tilt-series-gallery")

# Zero-loss intensity against tilt: the mean of each noiseless image (inside
# the field, so the reflect-padded flanks never enter) against Beer-Lambert
# attenuation over the slanted path t / cos(theta).
mfp = ice_inelastic_mfp(300.0)
thickness = sample.nz * 4.0
mean = np.array([float(field(im).mean()) for im in images])
theta = np.linspace(-62, 62, 200)
model = np.exp(-thickness / (mfp * np.cos(np.deg2rad(theta))))
predicted = np.exp(-thickness / (mfp * np.cos(np.deg2rad(ang))))
print(
    f"[check] mfp {mfp:.0f} A; mean at 0 {mean[0]:.4f} vs {predicted[0]:.4f}; "
    f"at +60 {mean[angles.index(60.0)]:.4f} vs "
    f"{predicted[angles.index(60.0)]:.4f}; "
    f"max |rel err| {np.max(np.abs(mean / predicted - 1)):.2e}",
    file=sys.__stdout__,
)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(
    theta,
    model,
    color=PALETTE[1],
    lw=1.2,
    label=r"$\exp[-t/(\lambda_\mathrm{in}\cos\theta)]$",
)
ax.plot(ang, mean, "o", color=PALETTE[0], ms=3.5, label="Simulated tilts")
ax.set(
    xlabel="Tilt angle (°)",
    ylabel="Mean intensity (zero-loss)",
    title=f"{thickness:.0f} Å slab, " + r"$\lambda_\mathrm{in}$" + f" = {mfp:.0f} Å",
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "tilt-series-path-length")

transcript.close()
