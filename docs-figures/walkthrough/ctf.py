"""
Figures and printed output for docs/walkthrough/ctf.md.

Run with: uv run python docs-figures/walkthrough/ctf.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import (
    PALETTE,
    Transcript,
    as_numpy,
    power_spectrum,
    radial_power,
    save,
    show_image,
)

from specter.constants import energy_to_wavelength

transcript = Transcript("ctf")


def chi(k: np.ndarray, defocus: float, cs_mm: float, voltage: float) -> np.ndarray:
    """Wavefront aberration phase for isotropic defocus and Cs (Kirkland's sign)."""
    wavelength = energy_to_wavelength(voltage)
    cs = cs_mm * 1e7
    return np.pi * wavelength * k**2 * (0.5 * cs * wavelength**2 * k**2 - defocus)


def show_spectrum(ax: plt.Axes, image: torch.Tensor, title: str | None = None) -> None:
    """Log power spectrum, clipped for contrast."""
    ps = power_spectrum(image)
    show_image(
        ax,
        ps,
        title=title,
        vmin=float(np.percentile(ps, 1)),
        vmax=float(np.percentile(ps, 99.7)),
    )


def lowpass(image: torch.Tensor, resolution: float = 5.0) -> np.ndarray:
    """Gaussian low-pass for display, halving the amplitude at `resolution` Å."""
    data = as_numpy(image)
    ky = np.fft.fftfreq(data.shape[-2])[:, None]
    kx = np.fft.fftfreq(data.shape[-1])[None, :]
    k2 = (ky**2 + kx**2) * resolution**2  # pixel size 1 Å
    return np.fft.ifft2(np.fft.fft2(data) * np.exp(-np.log(2) * k2)).real


transcript.start("setup")
# --8<-- [start:setup]
import specter
from specter.imaging import CTF, Poses, Structure

specimen = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256, device="cuda")
specimen = specimen.solvate(model="gd")

specter.seed(0)
poses = Poses.random(1)
waves = specimen.scatter(poses, voltage=300.0, seed=0)
print(f"exit wave {tuple(waves.array.shape)} {waves.array.dtype}")
# --8<-- [end:setup]

transcript.start("defocus")
# --8<-- [start:defocus]
images = {}
for defocus in [5000.0, 10000.0, 20000.0, 30000.0]:  # Å, positive = underfocus
    ctf = CTF.from_values(defocus, cs=2.7)
    image = waves.apply_ctf(ctf).intensity(cc=None)
    images[defocus] = image.pixels()[0]  # (Y, X) on the pixel grid
# --8<-- [end:defocus]

fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.6))
for col, (defocus, img) in enumerate(images.items()):
    show_image(
        axes[0, col],
        lowpass(img),
        title=f"{defocus / 1e4:.1f} µm underfocus",
        pixel_size=1.0,
        scalebar=50 if col == 0 else None,
    )
    show_spectrum(axes[1, col], img)
axes[0, 0].set_ylabel("Image")
axes[1, 0].set_ylabel("Power spectrum")
fig.tight_layout()
save(fig, "ctf-defocus-series")

transcript.start("thon-rings")
# --8<-- [start:thon-rings]
wavelength = energy_to_wavelength(300.0)  # Å
defocus = 10000.0  # Å; the defocus of the box's midplane, see below
zeros = [np.sqrt(n / (wavelength * defocus)) for n in range(1, 6)]
print("first zeros of sin χ (1/Å):", ", ".join(f"{k:.4f}" for k in zeros))
# --8<-- [end:thon-rings]

k, power = radial_power(images[10000.0], 1.0)
k_fine = np.linspace(0, 0.5, 4000)
sin2 = np.sin(chi(k_fine, defocus, 2.7, 300.0)) ** 2
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(7.0, 3.6), sharex=True, height_ratios=[2, 1]
)
ax.semilogy(k[1:], power[1:], color=PALETTE[0], lw=1.0)
ax.set(ylabel="Image power", title="1 µm underfocus, Cs 2.7 mm, 300 kV")
ax2.plot(k_fine, sin2, color=PALETTE[1], lw=0.9)
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel=r"$\sin^2\chi(k)$",
    xlim=(0, 0.3),
    ylim=(0, 1.05),
)
for z in zeros:
    for a in (ax, ax2):
        a.axvline(z, color="0.5", lw=0.6, ls=":")
fig.tight_layout()
save(fig, "ctf-thon-rings")

transcript.start("astigmatism")
# --8<-- [start:astigmatism]
specter.seed(2)
batch = specimen.scatter(Poses.random(8), voltage=300.0, seed=0)

astigmatic = {
    "none": CTF.from_values(10000.0, n=8),
    "4000 Å at 0°": CTF.from_values(10000.0, n=8, astigmatism=4000.0),
    "4000 Å at 45°": CTF.from_values(
        10000.0, n=8, astigmatism=4000.0, astigmatism_angle=45.0
    ),
}
spectra = {
    name: batch.apply_ctf(ctf).intensity(cc=None).pixels()
    for name, ctf in astigmatic.items()
}
print({k: v[0].item() for k, v in astigmatic["4000 Å at 45°"].params.items()})
# --8<-- [end:astigmatism]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (name, imgs) in zip(axes, spectra.items()):
    ps = np.mean([power_spectrum(img) for img in imgs], axis=0)[90:167, 90:167]
    show_image(
        ax,
        ps,
        title=f"Astigmatism: {name}",
        vmin=float(np.percentile(ps, 1)),
        vmax=float(np.percentile(ps, 99.5)),
    )
fig.tight_layout()
save(fig, "ctf-astigmatism")

transcript.start("phase-plate")
# --8<-- [start:phase-plate]
in_focus = waves.apply_ctf(CTF.from_values(0.0)).intensity(cc=None)
phase_plate = waves.apply_ctf(CTF.from_values(0.0, phase_shift=90.0)).intensity(cc=None)
# --8<-- [end:phase-plate]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
panels = [
    ("In focus", lowpass(in_focus.pixels()[0])),
    ("In focus, 90° phase plate", lowpass(phase_plate.pixels()[0])),
    ("1 µm underfocus", lowpass(images[10000.0])),
]
lo = min(float(np.percentile(as_numpy(p), 0.5)) for _, p in panels)
hi = max(float(np.percentile(as_numpy(p), 99.5)) for _, p in panels)
for ax, (name, img) in zip(axes, panels):
    show_image(ax, img, title=name, vmin=lo, vmax=hi)
fig.tight_layout()
save(fig, "ctf-phase-plate")

transcript.start("bfactor")
# --8<-- [start:bfactor]
ctf = CTF.from_values(10000.0)
blurred = {
    bfactor: waves.apply_ctf(ctf, bfactor=bfactor).intensity(cc=None).pixels()[0]
    for bfactor in [0.0, 50.0, 150.0]  # Å²
}
# --8<-- [end:bfactor]

fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), width_ratios=[1, 1, 1.5])
show_image(axes[0], blurred[0.0], title="B = 0 Å²")
show_image(axes[1], blurred[150.0], title="B = 150 Å²")
k, reference = radial_power(blurred[0.0], 1.0)
ratios = {}
for color, bfactor in zip(PALETTE[1:], [50.0, 150.0]):
    _, power = radial_power(blurred[bfactor], 1.0)
    ratios[bfactor] = power / reference
    axes[2].plot(
        k[1:], ratios[bfactor][1:], color=color, lw=1.0, label=f"B = {bfactor:g} Å²"
    )
    axes[2].plot(k, np.exp(-bfactor * k**2 / 2), color=color, lw=1.0, ls="--")
axes[2].set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power relative to B = 0",
    title=r"Measured (solid), $e^{-Bk^2/2}$ (dashed)",
    ylim=(0, 1.1),
)
axes[2].legend(frameon=False)
fig.tight_layout()
save(fig, "ctf-bfactor")
for bfactor, ratio in ratios.items():
    band = (k > 0.1) & (k < 0.3)
    predicted = np.exp(-bfactor * k[band] ** 2 / 2)
    err = np.abs(ratio[band] / predicted - 1)
    print(
        f"B={bfactor:g}: measured/predicted power, 0.1-0.3 1/Å: "
        f"median |dev| {np.median(err):.3f}",
        file=sys.__stdout__,
    )

transcript.start("reference-plane")
# --8<-- [start:reference-plane]
print(f"box-depth ice: {specimen.nz} slices, defocus_shift {waves.defocus_shift:.0f} Å")

thick = specimen.solvate(model="gd", thickness=2000.0)
thick_waves = thick.scatter(poses, voltage=300.0, seed=0)
print(
    f"2000 Å of ice: {thick.nz} slices, defocus_shift {thick_waves.defocus_shift:.0f} Å"
)
# --8<-- [end:reference-plane]

transcript.start("depth")
# --8<-- [start:depth]
from specter.imaging import Specimen

nz, n = 2000, 128
z, y, x = torch.meshgrid(
    *(torch.arange(s, device="cuda") for s in (nz, n, n)), indexing="ij"
)
one_pose = Poses.tilt_series([0.0])  # one untilted view: the identity pose
nominal = CTF.from_values(10000.0, cs=0.0)
trial_defoci = np.arange(8800.0, 11201.0, 25.0)

matched = {}
for depth in [100, 550, 1000, 1450, 1900]:  # slice index; the beam enters at slice 0
    blob = 5.0 * torch.exp(
        -((z - depth) ** 2 + (y - n // 2) ** 2 + (x - n // 2) ** 2) / 18
    )
    point = Specimen(blob, voxel_size=1.0)
    thick_image = point.scatter(one_pose, voltage=300.0).apply_ctf(nominal)
    thick_image = thick_image.intensity(cc=None).pixels()
    flat = point.scatter(one_pose, voltage=300.0, scattering_model="projection")
    errors = [
        (
            flat.apply_ctf(CTF.from_values(d, cs=0.0)).intensity(cc=None).pixels()
            - thick_image
        )
        .square()
        .mean()
        .item()
        for d in trial_defoci
    ]
    below_entry = depth
    matched[below_entry] = trial_defoci[int(np.argmin(errors))]
    print(
        f"{below_entry:4d} Å below the entry face: images at {matched[below_entry]:.0f} Å"
    )
# --8<-- [end:depth]

ring_image = thick_waves.apply_ctf(CTF.from_values(10000.0)).intensity(cc=None)
k, power = radial_power(ring_image.pixels()[0], 1.0)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 2.9), width_ratios=[1.6, 1])
ax.semilogy(k[1:], power[1:], color="0.2", lw=1.0)
for color, style, (label, d) in zip(
    PALETTE,
    ["-", "--", ":"],
    [
        ("midplane, 1.0 µm", 10000.0),
        ("entry face, 0.9 µm", 9000.0),
        ("exit face, 1.1 µm", 11000.0),
    ],
):
    for i in range(1, 6):
        ax.axvline(
            np.sqrt(i / (wavelength * d)),
            color=color,
            lw=1.0,
            ls=style,
            label=f"zeros at the {label}" if i == 1 else None,
        )
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power",
    xlim=(0, 0.2),
    title="2000 Å of ice at a nominal 1 µm",
)
ax.legend(frameon=False, loc="lower right", fontsize=7)
depths = np.array(sorted(matched))
ax2.plot(depths, [matched[d] / 1e4 for d in depths], "o", color=PALETTE[0])
ax2.plot([0, nz], [0.9, 1.1], color="0.5", lw=0.8, ls="--")
ax2.axvline(nz / 2, color="0.7", lw=0.6, ls=":")
ax2.axhline(1.0, color="0.7", lw=0.6, ls=":")
ax2.set(
    xlabel="Depth below the entry face (Å)",
    ylabel="Defocus it images at (µm)",
    title="A point scatterer at five depths",
)
fig.tight_layout()
save(fig, "ctf-reference-plane")
transcript.start("sample")
# --8<-- [start:sample]
specter.seed(1)
poses = Poses.random(6)
ctf = CTF.sample(6, defocus=[5000.0, 25000.0])
stack = specimen.scatter(poses, voltage=300.0, seed=0).apply_ctf(ctf).intensity(cc=None)
print("dfu (Å):", ", ".join(f"{d:.0f}" for d in ctf.params["dfu"]))
# --8<-- [end:sample]

pixels = stack.pixels()
fig, axes = plt.subplots(1, 6, figsize=(10.2, 1.9))
for i, ax in enumerate(axes):
    show_image(ax, lowpass(pixels[i]), title=f"{ctf.params['dfu'][i] / 1e4:.2f} µm")
fig.tight_layout()
save(fig, "ctf-sample")

transcript.close()
