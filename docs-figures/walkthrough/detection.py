"""
Figures and printed output for docs/walkthrough/detection.md.

Run with: uv run python docs-figures/walkthrough/detection.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, radial_power, save, show_image

transcript = Transcript("detection")

transcript.start("setup")
# --8<-- [start:setup]
import torch

import specter
from specter.imaging import CTF, Exposure, Poses, Structure

specter.seed(0)
specimen = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256, device="cuda")
waves = specimen.scatter(Poses.random(1), voltage=300.0)
image = waves.apply_ctf(CTF.from_values(defocus=15000.0, n=1)).intensity(cc=2.7)

print(f"relative intensity: mean {image.array.mean():.4f}, {tuple(image.array.shape)}")
# --8<-- [end:setup]

transcript.start("detect")
# --8<-- [start:detect]
exposure = Exposure(dose=20.0)  # e⁻/Å² per image
counts = image.detect(exposure, seed=0)

print(f"counts: {tuple(counts.array.shape)}, mean {counts.array.mean():.2f}")
print(f"variance {counts.array.var():.2f}")
# --8<-- [end:detect]

transcript.start("dose")
# --8<-- [start:dose]
doses = [5.0, 20.0, 60.0, 200.0]
noisy = {}
for dose in doses:
    exposure = Exposure(dose=dose)
    expected = image.detect(exposure, noise=None).array  # mean counts, no noise
    observed = image.detect(exposure, seed=0).array
    noisy[dose] = observed
    snr = expected.var() / (observed - expected).var()
    print(f"{dose:5.0f} e⁻/Å²   mean {observed.mean():6.1f} counts/px   SNR {snr:.3f}")
# --8<-- [end:dose]

clean = as_numpy(image.detect(Exposure(dose=1.0), noise=None).array[0])
fig, axes = plt.subplots(1, len(doses) + 1, figsize=(2.1 * (len(doses) + 1), 2.3))
show_image(axes[0], clean, title="Expected (no noise)", pixel_size=1.0, scalebar=50)
for ax, dose in zip(axes[1:], doses):
    show_image(ax, noisy[dose][0], title=f"{dose:g} e⁻/Å²")
fig.tight_layout()
save(fig, "detection-dose")

transcript.start("detectors")
# --8<-- [start:detectors]
from specter.detectors import detector_mtf, dqe0_for_detector

cameras = ["perfect", "k3_300kv", "k2_300kv", "falcon4i_300kv", "falcon3ec_300kv"]
for camera in cameras:
    mtf = detector_mtf(camera, n=256, pixel_size=1.0)  # (256, 256), FFT order
    print(
        f"{camera:16s} DQE(0) {dqe0_for_detector(camera):.2f}   "
        f"MTF at Nyquist/2 {mtf[0, 64]:.2f}, at Nyquist {mtf[0, 128]:.2f}"
    )
# --8<-- [end:detectors]

# Spectral SNR per camera at one dose, relative to an ideal counter
# (detector=None: every electron counted, no blur). For a counting detector the
# noise stays white, so the ratio is DQE(0) * MTF(k)^2, the detector's DQE(k).
exposure = Exposure(dose=40.0)
n_seeds = 16


def spectral_snr(camera: str | None) -> tuple[np.ndarray, np.ndarray]:
    expected = image.detect(exposure, detector=camera, noise=None).array[0]
    noise_power = 0.0
    for s in range(n_seeds):
        observed = image.detect(exposure, detector=camera, seed=s).array[0]
        noise_power = noise_power + radial_power(observed - expected, 1.0)[1] / n_seeds
    k, signal_power = radial_power(expected, 1.0)
    return k, signal_power / noise_power


k, ideal = spectral_snr(None)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0))
k_axis = np.arange(129) / 256.0
for color, camera in zip(PALETTE, cameras):
    mtf = as_numpy(detector_mtf(camera, n=256, pixel_size=1.0))[0, :129]
    ax1.plot(k_axis, mtf, color=color, lw=1.2, label=camera)
    _, snr = spectral_snr(camera)
    ax2.plot(k[1:], snr[1:] / ideal[1:], color=color, lw=0.8, alpha=0.8)
    ax2.plot(k_axis, dqe0_for_detector(camera) * mtf**2, color=color, lw=1.4, ls="--")
ax1.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="MTF",
    title="Detector MTF",
    ylim=(0, 1.05),
    xlim=(0, 0.5),
)
ax1.legend(frameon=False, loc="lower left")
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="SNR / SNR of an ideal counter",
    title="Measured (solid) and DQE(0)·MTF² (dashed)",
    xlim=(0, 0.5),
    ylim=(0, 1.15),
)
fig.tight_layout()
save(fig, "detection-detectors")

transcript.start("coincidence")
# --8<-- [start:coincidence]
from specter.detectors import coincidence_occupancy, coincidence_radius_for_simulation
from specter.imaging import Image

# Uniform illumination: 16 empty images at the vacuum intensity of 1.
flat = Image(
    torch.ones(16, 256, 256, device="cuda"),
    pixel_size=1.0,
    nxy=256,
    pad_nxy=256,
    ids=torch.arange(16),
)
exposure = Exposure(dose=40.0, n_frames=40)

recorded = {}
for rate in [0.0, 4.0, 16.0]:  # e⁻ per physical pixel per second
    occupancy = coincidence_occupancy(2.0, rate, frame_rate_hz=320.0)  # Falcon 4i
    radius = coincidence_radius_for_simulation(occupancy, 40.0, 1.0, 40)
    recorded[rate] = flat.detect(exposure, coincidence_radius=radius, seed=0).array
    kept = recorded[rate].mean() / 40.0
    print(f"{rate:4.0f} e⁻/px/s   radius {radius:.3f} px   counted {kept:.1%}")
# --8<-- [end:coincidence]

rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
counted = []
for rate in rates:
    radius = coincidence_radius_for_simulation(
        coincidence_occupancy(2.0, rate, frame_rate_hz=320.0), 40.0, 1.0, 40
    )
    counted.append(
        float(flat.detect(exposure, coincidence_radius=radius, seed=0).array.mean())
        / 40.0
    )
print(
    "counted vs rate:",
    dict(zip(rates, [round(c, 3) for c in counted])),
    file=sys.__stdout__,
)

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1, 1.4]}
)
ax1.semilogx(rates, counted, "o-", color=PALETTE[0], lw=1.2, ms=4)
ax1.set(
    xlabel="Dose rate (e⁻/physical px/s)",
    ylabel="Fraction counted",
    title="Falcon 4i, 2.0 px exclusion, 320 Hz",
    ylim=(0, 1.05),
)
for color, (rate, stack) in zip(PALETTE, recorded.items()):
    nps = 0.0
    for frame in stack:
        k, power = radial_power(frame, 1.0)
        nps = nps + power / len(stack)
    label = "Poisson (no coincidence)" if rate == 0 else f"{rate:g} e⁻/px/s"
    ax2.plot(k[1:], nps[1:] / float(stack.mean()), color=color, lw=1.0, label=label)
ax2.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Noise power / mean count",
    title="Noise power spectrum of flat illumination",
    xlim=(0, 0.5),
)
ax2.legend(frameon=False)
fig.tight_layout()
save(fig, "detection-coincidence")

os.chdir(tempfile.mkdtemp())
transcript.start("save")
# --8<-- [start:save]
import mrcfile

counts = image.detect(Exposure(dose=40.0), detector="falcon4i_300kv", seed=0)
counts.save("particle.mrc")

with mrcfile.open("particle.mrc") as mrc:
    print(mrc.data.shape, mrc.data.dtype, f"voxel size {mrc.voxel_size.x:.2f} Å")
# --8<-- [end:save]

transcript.close()
