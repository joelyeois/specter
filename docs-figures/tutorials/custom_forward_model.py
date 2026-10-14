"""
Figures and printed output for docs/tutorials/custom-forward-model.md.

Run with: uv run python docs-figures/tutorials/custom_forward_model.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, radial_power, save, show_image

transcript = Transcript("custom_forward_model")


def lowpass(x: torch.Tensor, pixel_size: float, resolution: float) -> torch.Tensor:
    """A Gaussian low-pass to `resolution` Å, for display only."""
    x = x.float()
    k = torch.fft.fftfreq(x.shape[-1], d=pixel_size, device=x.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(-k2 * resolution**2)).real


transcript.start("objects")
# --8<-- [start:objects]
from specter.imaging import CTF, Exposure, Poses, Structure

template = Structure.from_pdb("6bdf").render(voxel_size=2.0, box=256, device="cuda")
poses = Poses.random(1, seed=3)
ctf = CTF.from_values(defocus=15000.0)
exposure = Exposure(dose=40.0)

waves = template.scatter(poses, voltage=300.0, alpha=0.1)
reference = (
    waves.apply_ctf(ctf)
    .intensity(cc=2.7)
    .detect(exposure, detector="k3_300kv", noise=None)
)
print(f"objects: {tuple(reference.array.shape)} expected counts")
# --8<-- [end:objects]

transcript.start("rotate")
# --8<-- [start:rotate]
import roma

from specter import rotations
from specter.rotations import VolumeRotator

V0 = template.potential  # (Z, Y, X), volts
nz, ny, nx = V0.shape
px = template.voxel_size

rotator = VolumeRotator(nz, ny, nx, origin="relion").to("cuda")
R = roma.unitquat_to_rotmat(poses.quaternions.to("cuda"))
T = rotations.translations_angstrom_to_torch(poses.shifts.to("cuda"), nx, px)
V = rotator(V0, rotations.build_affine_matrix(R, T))  # (1, Z, Y, X)
print(f"rotated volume {tuple(V.shape)}")
# --8<-- [end:rotate]

transcript.start("scatter")
# --8<-- [start:scatter]
from specter.scattering import Scattering

oversample = 2
scattering = Scattering(
    nx,
    px,
    300.0,
    scattering_model="multislice",
    oversample=oversample,
    crop=False,  # keep the propagation grid; the detector samples it down
    nz=nz,
    alpha=0.1,
    progressbars=False,
).to("cuda")
with torch.no_grad():
    psi = scattering(V)
print(f"exit wave {tuple(psi.shape)}, {psi.dtype}: {oversample}x the pixel grid")
# --8<-- [end:scatter]

transcript.start("lens")
# --8<-- [start:lens]
from specter.aberrations import Aberration, defocus_midplane_shift

# The wave left the exit face; the nominal defocus is the midplane's.
shift = defocus_midplane_shift(nz, px)
params = {k: v.to("cuda") for k, v in ctf.params.items()}
params["dfu"] = params["dfu"] - shift
params["dfv"] = params["dfv"] - shift

lens = Aberration(
    nx * oversample,
    px / oversample,
    300.0,
    specimen_absorption=True,
    progressbars=False,
).to("cuda")
psi_image = lens(psi, params)
print(f"defocus shift {shift:.0f} Å; wave at the detector {tuple(psi_image.shape)}")
# --8<-- [end:lens]

transcript.start("detect")
# --8<-- [start:detect]
from specter.aberrations import FocalSpread
from specter.constants import energy_to_wavelength
from specter.detectors import detector_mtf, dqe0_for_detector
from specter.microscope import Detector

focal = FocalSpread.from_instrument(
    2.7e7,  # Cc, Angstrom
    300e3,  # beam energy, eV
    0.7,  # energy spread, FWHM eV
    0.06e-6,  # high-tension instability
    0.01e-6,  # lens-current instability
    energy_to_wavelength(300.0),
)
intensity = focal.intensity(psi_image, px / oversample, oversample)

camera = Detector(
    px,
    mtf=detector_mtf("k3_300kv", nx, px),
    dqe0=dqe0_for_detector("k3_300kv"),
    oversample=oversample,
    progressbars=False,
).to("cuda")
dose = torch.tensor([40.0], device="cuda")
expected = dose[:, None, None] * px**2 * camera.dqe0 * intensity
by_hand = camera.detect_expected_counts(expected, dose, torch.zeros(1, device="cuda"))

difference = (by_hand - reference.array).abs().max() / reference.array.abs().max()
print(
    f"by hand: {tuple(by_hand.shape)}; largest difference {difference:.1e} of the peak"
)
# --8<-- [end:detect]

# ---------------------------------------------------------------------------
# Figure: every stage of the by-hand chain.
stages = [
    ("Rotated potential\nprojected (V·Å)", V[0].sum(0) * px, "gray_r"),
    ("Exit-wave phase\n(rad)", torch.angle(psi[0]), "gray"),
    ("|ψ|² after the lens", psi_image[0].abs() ** 2, "gray"),
    ("Focal spread summed", intensity[0], "gray"),
    ("Detected (expected\ncounts, K3)", by_hand[0], "gray"),
]
fig, axes = plt.subplots(1, len(stages), figsize=(11.0, 2.6))
for ax, (title, img, cmap) in zip(axes, stages):
    data = as_numpy(img)
    lo, hi = np.percentile(data, [0.5, 99.5])
    show_image(ax, data, title=title, cmap=cmap, vmin=lo, vmax=hi)
fig.tight_layout()
save(fig, "custom_forward_model-stages")

# Figure: objects against the hand-built chain.
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
ref_np, hand_np = as_numpy(reference.array[0]), as_numpy(by_hand[0])
lo, hi = np.percentile(ref_np, [0.5, 99.5])
show_image(axes[0], ref_np, title="Objects", vmin=lo, vmax=hi)
show_image(axes[1], hand_np, title="Engines by hand", vmin=lo, vmax=hi)
show_image(
    axes[2],
    (hand_np - ref_np) / np.abs(ref_np).max(),
    title="Difference / peak",
    cmap="RdBu_r",
    vmin=-1e-6,
    vmax=1e-6,
    colorbar="relative difference",
)
fig.tight_layout()
save(fig, "custom_forward_model-equal")

transcript.start("plate")
# --8<-- [start:plate]
import math


def phase_plate(wave, phase, cut_on, dx):
    """
    A phase plate with a finite hole: scattered beams beyond the cut-on
    frequency are shifted by `phase` (radians) relative to the unscattered
    beam; those inside it, nearer the beam, are not.
    """
    k = torch.fft.fftfreq(wave.shape[-1], d=dx, device=wave.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    shift = phase * (1.0 - torch.exp(-k2 / cut_on**2))
    return torch.fft.ifft2(torch.fft.fft2(wave) * torch.exp(1j * shift))


in_focus = {k: v.to("cuda") for k, v in CTF.from_values(defocus=0.0).params.items()}
for name in ("dfu", "dfv"):
    in_focus[name] = in_focus[name] - shift
psi_focus = lens(psi, in_focus)

images = {"no plate": psi_focus.abs() ** 2}
for cut_on in (1 / 30, 1 / 100):
    psi_plate = phase_plate(psi_focus, math.pi / 2, cut_on, px / oversample)
    images[f"hole 1/{1 / cut_on:.0f} Å⁻¹"] = psi_plate.abs() ** 2

# The built-in term shifts every scattered beam: a plate with no hole.
built_in = {
    k: v.to("cuda")
    for k, v in CTF.from_values(defocus=0.0, phase_shift=90.0).params.items()
}
for name in ("dfu", "dfv"):
    built_in[name] = built_in[name] - shift
images["CTF(phase_shift=90)"] = lens(psi, built_in).abs() ** 2

no_hole = phase_plate(psi_focus, math.pi / 2, 1e-5, px / oversample).abs() ** 2
match = (no_hole - images["CTF(phase_shift=90)"]).abs().max()
print(f"a vanishing hole against the built-in phase shift: max difference {match:.1e}")
# --8<-- [end:plate]

# Figure: in-focus images with each plate.
names = list(images)
fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.5))
crop = slice(oversample * 48, oversample * 208)
pixel = {n: lowpass(im[0], px / oversample, 4.0) for n, im in images.items()}
shown = np.concatenate([as_numpy(pixel[n][crop, crop]).ravel() for n in names])
lo, hi = (float(v) for v in np.percentile(shown, [0.05, 99.95]))
for ax, n in zip(axes, names):
    show_image(ax, as_numpy(pixel[n][crop, crop]), title=n, vmin=lo, vmax=hi)
fig.tight_layout()
save(fig, "custom_forward_model-plates")

# Figure: the plate's phase against frequency, and the in-focus image power
# each hole leaves, as a fraction of the ideal plate's.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9))
kk = np.linspace(0, 0.08, 400)
ideal_k, ideal_p = radial_power(
    as_numpy(images["CTF(phase_shift=90)"][0]), px / oversample
)
for color, cut_on in zip(PALETTE, (1 / 30, 1 / 100)):
    label = f"hole 1/{1 / cut_on:.0f} Å⁻¹"
    ax1.plot(
        kk, 90 * (1 - np.exp(-(kk**2) / cut_on**2)), color=color, lw=1.2, label=label
    )
    k, p = radial_power(as_numpy(images[label][0]), px / oversample)
    ax2.plot(k[1:], p[1:] / ideal_p[1:], color=color, lw=1.2, label=label)
    for ax in (ax1, ax2):
        ax.axvline(cut_on, color=color, lw=0.6, ls=":")
k, p = radial_power(as_numpy(images["no plate"][0]), px / oversample)
ax2.plot(k[1:], p[1:] / ideal_p[1:], color="0.5", lw=1.0, label="no plate")
ax1.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Phase shift (degrees)",
    title="The plate's phase",
)
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power / ideal plate",
    title="In-focus image power",
    xlim=(0, 0.08),
    ylim=(0, 1.8),
)
ax1.legend(frameon=False)
ax2.legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "custom_forward_model-cut-on")

# Thumbnail: the ideal plate's in-focus image.
fig, ax = plt.subplots(figsize=(3, 3))
thumb = as_numpy(pixel["CTF(phase_shift=90)"][crop, crop])
show_image(ax, thumb, vmin=np.percentile(thumb, 1), vmax=np.percentile(thumb, 99.5))
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "custom_forward_model-thumbnail")

transcript.close()
