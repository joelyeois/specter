"""
Figures and printed output for docs/tutorials/dose-damage.md.

Run with: uv run python docs-figures/tutorials/dose_damage.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import (
    PALETTE,
    Transcript,
    as_numpy,
    radial_power,
    save,
    show_image,
)

transcript = Transcript("dose_damage")

RING = (0.255, 0.285)  # 1/Å, the band around the 3.7 Å water ring


def mean_radial_power(stack: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Rotationally averaged power spectrum, averaged over a stack of images."""
    powers = [radial_power(image, 1.0) for image in stack]
    return powers[0][0], np.mean([p for _, p in powers], axis=0)


def log_power(stack: torch.Tensor) -> np.ndarray:
    """log10 of the centred power spectrum, mean removed, averaged over a stack."""
    data = as_numpy(stack)
    data = data - data.mean(axis=(-2, -1), keepdims=True)
    ps = np.abs(np.fft.fftshift(np.fft.fft2(data), axes=(-2, -1))) ** 2
    return np.log10(ps.mean(axis=0) + 1e-12)


def band(k: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    """Mean of `power` over the shells with lo <= k < hi."""
    return float(power[(k >= lo) & (k < hi)].mean())


transcript.start("setup")
# --8<-- [start:setup]
import torch

from specter.imaging import CTF, Exposure, Poses, Structure

structure = Structure.from_pdb("6bdf")
particle = structure.render(voxel_size=1.0, box=256, device="cuda")
specimen = particle.solvate(model="gd")  # ice filling the 256 Å box

poses = Poses.random(4, seed=0)
ctf = CTF.from_values(15000.0, n=4)  # 1.5 µm underfocus
print(f"{len(poses)} particles of 6BDF in {specimen.nz} Å of ice")
# --8<-- [end:setup]

transcript.start("envelope")
# --8<-- [start:envelope]
from specter.aberrations import critical_exposure, dose_envelope

k = torch.tensor([1 / 10, 1 / 6.7, 1 / 3.7])  # 1/Å
print("critical exposure N_e:", [f"{n:.1f}" for n in critical_exposure(k).tolist()])
for dose in [10.0, 20.0, 40.0, 80.0]:
    weighted = dose_envelope(k, torch.tensor(dose))
    plain = dose_envelope(k, torch.tensor(dose), weighted=False)
    print(
        f"{dose:4.0f} e/Å²  dose-weighted sum {weighted.numpy().round(2)}"
        f"  plain sum {plain.numpy().round(2)}"
    )
# --8<-- [end:envelope]

kk = torch.linspace(1e-3, 0.5, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.0))
for color, dose in zip(PALETTE, [10.0, 20.0, 40.0, 80.0]):
    ax1.plot(
        kk,
        dose_envelope(kk, torch.tensor(dose)),
        color=color,
        lw=1.2,
        label=f"{dose:g} e⁻/Å²",
    )
    ax1.plot(
        kk,
        dose_envelope(kk, torch.tensor(dose), weighted=False),
        color=color,
        lw=0.9,
        ls="--",
    )
ax1.axvline(1 / 3.7, color="0.5", lw=0.8, ls=":")
ax1.text(1 / 3.7 + 0.005, 0.95, "3.7 Å", fontsize=7, color="0.4")
ax1.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Amplitude envelope",
    title="Dose-weighted sum (solid), plain sum (dashed)",
    ylim=(0, 1.02),
)
ax1.legend(frameon=False, loc="upper right")
ax2.loglog(kk[5:], critical_exposure(kk[5:]), color=PALETTE[0], lw=1.2)
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Critical exposure (e⁻/Å²)",
    title="Grant & Grigorieff critical exposure",
)
fig.tight_layout()
save(fig, "dose_damage-envelope")

transcript.start("routes")
# --8<-- [start:routes]
waves = specimen.scatter(poses, voltage=300.0, seed=0)
undamaged = waves.apply_ctf(ctf).intensity(cc=None).pixels()

doses = [10.0, 20.0, 40.0, 80.0]
on_lens, on_specimen = {}, {}
for dose in doses:
    exposure = Exposure(dose)
    # Route 1: fade the whole image through the transfer function.
    lensed = waves.apply_ctf(ctf, dose_envelope=exposure)
    on_lens[dose] = lensed.intensity(cc=None).pixels()
    # Route 2: damage the particle's potential before the ice is added.
    damaged = specimen.damage().scatter(poses, voltage=300.0, exposure=exposure, seed=0)
    on_specimen[dose] = damaged.apply_ctf(ctf).intensity(cc=None).pixels()
# --8<-- [end:routes]

k_r, p_ref = mean_radial_power(undamaged)
ring_lens, ring_spec = [], []
for dose in doses:
    _, p_lens = mean_radial_power(on_lens[dose])
    _, p_spec = mean_radial_power(on_specimen[dose])
    ring_lens.append(band(k_r, p_lens, *RING) / band(k_r, p_ref, *RING))
    ring_spec.append(band(k_r, p_spec, *RING) / band(k_r, p_ref, *RING))
ring_k = torch.tensor([float(np.mean(RING))])
predicted = [float(dose_envelope(ring_k, torch.tensor(d))[0] ** 2) for d in doses]
print("water-ring power relative to the undamaged image (0.255-0.285 1/Å):")
for dose, a, b, e in zip(doses, ring_lens, ring_spec, predicted):
    print(f"{dose:4.0f} e/Å²  transfer function {a:.3f}  specimen {b:.3f}  E² {e:.3f}")

transcript.start("dry")
# --8<-- [start:dry]
dry = particle.scatter(poses, voltage=300.0, seed=0)
dry_damaged = particle.damage().scatter(
    poses, voltage=300.0, exposure=Exposure(40.0), seed=0
)
dry_before = dry.apply_ctf(ctf).intensity(cc=None).pixels()
dry_after = dry_damaged.apply_ctf(ctf).intensity(cc=None).pixels()
# --8<-- [end:dry]

_, p_dry0 = mean_radial_power(dry_before)
_, p_dry1 = mean_radial_power(dry_after)
_, p_lens40 = mean_radial_power(on_lens[40.0])
_, p_spec40 = mean_radial_power(on_specimen[40.0])
e40 = dose_envelope(torch.from_numpy(k_r).float(), torch.tensor(40.0)).numpy() ** 2
keep = k_r > 0.02
dry_ratio = p_dry1 / p_dry0
for lo, hi in [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3)]:
    sel = keep & (k_r >= lo) & (k_r < hi)
    print(
        f"{lo:.2f}-{hi:.2f} 1/Å at 40 e/Å²: particle alone {dry_ratio[sel].mean():.3f}"
        f" (E² {e40[sel].mean():.3f}); in ice, specimen route"
        f" {(p_spec40 / p_ref)[sel].mean():.3f}, transfer-function route"
        f" {(p_lens40 / p_ref)[sel].mean():.3f}"
    )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.1))
ax1.plot(
    k_r[keep],
    (p_lens40 / p_ref)[keep],
    color=PALETTE[0],
    lw=1.0,
    label="in ice, on the transfer function",
)
ax1.plot(
    k_r[keep],
    (p_spec40 / p_ref)[keep],
    color=PALETTE[1],
    lw=1.0,
    label="in ice, on the specimen",
)
ax1.plot(
    k_r[keep],
    dry_ratio[keep],
    color=PALETTE[2],
    lw=1.0,
    label="particle alone, on the specimen",
)
ax1.plot(k_r[keep], e40[keep], color="0.2", lw=1.0, ls="--", label="E² at 40 e⁻/Å²")
ax1.axvspan(*RING, color="0.9", zorder=0)
ax1.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power / undamaged",
    title="40 e⁻/Å²",
    ylim=(0, 1.45),
)
ax1.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
ax2.plot(
    doses, ring_lens, "o-", color=PALETTE[0], lw=1.0, label="on the transfer function"
)
ax2.plot(doses, ring_spec, "o-", color=PALETTE[1], lw=1.0, label="on the specimen")
ax2.plot(doses, predicted, color="0.2", lw=1.0, ls="--", label="E² at 3.7 Å")
ax2.set(
    xlabel="Dose (e⁻/Å²)",
    ylabel="Water-ring power / undamaged",
    title="The 3.7 Å ring under exposure",
    ylim=(0, 1.1),
)
ax2.legend(frameon=False, fontsize=7, loc="center right")
fig.tight_layout()
save(fig, "dose_damage-routes")

spectra = {
    "Undamaged": log_power(undamaged),
    "80 e⁻/Å² on the transfer function": log_power(on_lens[80.0]),
    "80 e⁻/Å² on the specimen": log_power(on_specimen[80.0]),
}
lo, hi = np.percentile(spectra["Undamaged"], [5, 99.8])
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
for ax, (title, ps) in zip(axes, spectra.items()):
    show_image(ax, ps, title=title, vmin=lo, vmax=hi)
fig.tight_layout()
save(fig, "dose_damage-spectra")

# Thumbnail: the two routes' spectra, split down the middle, on one scale.
ps_lens = spectra["80 e⁻/Å² on the transfer function"]
ps_spec = spectra["80 e⁻/Å² on the specimen"]
half = ps_lens.shape[-1] // 2
split = np.concatenate([ps_lens[:, :half], ps_spec[:, half:]], axis=1)
fig, ax = plt.subplots(figsize=(3, 3))
show_image(ax, split, vmin=lo, vmax=hi)
ax.axvline(half - 0.5, color="white", lw=1.5)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "dose_damage-thumbnail")

transcript.start("motion")
from specter.ice import solvent_exposure_power

# --8<-- [start:motion]
from specter.imaging import Specimen
from specter.settings import Ice

empty = Specimen(torch.zeros(256, 256, 256, device="cuda"), voxel_size=1.0)
frozen_ice = empty.solvate(model="gd")
moving_ice = empty.solvate(Ice(model="gd", motion_variance=0.38))

exposure = Exposure(40.0, n_frames=40)
frozen = frozen_ice.materialize(poses, voltage=300.0, exposure=exposure, seed=0)
moving = moving_ice.materialize(poses, voltage=300.0, exposure=exposure, seed=0)
print(f"mean ice potential: frozen {frozen.mean():.4f} V, moving {moving.mean():.4f} V")
# --8<-- [end:motion]

proj_frozen = frozen.sum(dim=1)  # (n, Y, X), V·Å
proj_moving = moving.sum(dim=1)
k_m, pf = mean_radial_power(proj_frozen)
_, pm = mean_radial_power(proj_moving)
measured = pm / pf
kt = torch.from_numpy(k_m).float()
relaxed = solvent_exposure_power(kt, 40.0, 0.38, 40).numpy()
gaussian = solvent_exposure_power(kt, 40.0, 0.38, 40, model="gaussian").numpy()
relaxed10 = solvent_exposure_power(kt, 10.0, 0.38, 10).numpy()
for lo, hi in [(0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.255, 0.285), (0.4, 0.5)]:
    sel = (k_m >= lo) & (k_m < hi)
    print(
        f"{lo:.3f}-{hi:.3f} 1/Å: surviving ice power {measured[sel].mean():.3f}"
        f" (relaxed model {relaxed[sel].mean():.3f}, Gaussian {gaussian[sel].mean():.3f})"
    )

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(9.4, 3.0), gridspec_kw={"width_ratios": [1.5, 1, 1]}
)
sel = k_m > 0.005
ax1.semilogy(
    k_m[sel],
    relaxed10[sel],
    color=PALETTE[1],
    lw=1.0,
    ls=":",
    label="relaxed, 10 e⁻/Å²",
)
ax1.semilogy(
    k_m[sel], relaxed[sel], color=PALETTE[1], lw=1.2, label="relaxed, 40 e⁻/Å²"
)
ax1.semilogy(
    k_m[sel],
    gaussian[sel],
    color=PALETTE[0],
    lw=1.0,
    ls="--",
    label="Gaussian, 40 e⁻/Å²",
)
ax1.semilogy(
    k_m[sel], measured[sel], ".", ms=2.5, color="0.2", label="measured, 40 e⁻/Å²"
)
ax1.axvspan(*RING, color="0.9", zorder=0)
ax1.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Surviving ice power",
    title="σ₀² = 0.38 Å² per e⁻/Å²",
    ylim=(1e-3, 1.5),
)
ax1.legend(frameon=False, fontsize=7, loc="lower right")


def lowpass(image: torch.Tensor, resolution: float = 10.0) -> np.ndarray:
    data = as_numpy(image)
    data = data - data.mean()
    ky = np.fft.fftfreq(data.shape[-2])[:, None]
    kx = np.fft.fftfreq(data.shape[-1])[None, :]
    k2 = (ky**2 + kx**2) * resolution**2
    return np.fft.ifft2(np.fft.fft2(data) * np.exp(-np.log(2) * k2)).real


a, b = lowpass(proj_frozen[0]), lowpass(proj_moving[0])
lim = float(np.percentile(np.abs(a), 99.5))
show_image(
    ax2,
    a,
    title="Frozen ice, ≥10 Å",
    vmin=-lim,
    vmax=lim,
    cmap="gray_r",
    pixel_size=1.0,
    scalebar=50,
)
show_image(ax3, b, title="Moving ice, ≥10 Å", vmin=-lim, vmax=lim, cmap="gray_r")
fig.tight_layout()
save(fig, "dose_damage-motion")

transcript.start("refusals")
# --8<-- [start:refusals]
damaged_waves = specimen.damage().scatter(
    poses[:1], voltage=300.0, exposure=Exposure(40.0), seed=0
)
moving_waves = moving_ice.scatter(poses[:1], voltage=300.0, exposure=exposure, seed=0)
for exit_wave in (damaged_waves, moving_waves):
    try:
        exit_wave.apply_ctf(ctf[:1], dose_envelope=Exposure(40.0))
    except ValueError as error:
        print(f"ValueError: {error}")
try:
    specimen.damage().scatter(poses[:1], voltage=300.0, seed=0)
except ValueError as error:
    print(f"ValueError: {error}")
# --8<-- [end:refusals]

transcript.close()
