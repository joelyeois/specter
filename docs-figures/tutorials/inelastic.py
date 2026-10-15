"""
Figures and printed output for docs/tutorials/inelastic.md.

Run with: uv run python docs-figures/tutorials/inelastic.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import PALETTE, Transcript, as_numpy, radial_power, save, show_image

transcript = Transcript("inelastic")


def lowpass(image: torch.Tensor, pixel_size: float, resolution: float) -> np.ndarray:
    """Gaussian low-pass for display, halving the amplitude at `resolution` Å."""
    data = as_numpy(image)
    ky = np.fft.fftfreq(data.shape[-2], d=pixel_size)[:, None]
    kx = np.fft.fftfreq(data.shape[-1], d=pixel_size)[None, :]
    k2 = (ky**2 + kx**2) * resolution**2
    return np.fft.ifft2(np.fft.fft2(data) * np.exp(-np.log(2) * k2)).real


transcript.start("setup")
# --8<-- [start:setup]
import math

import torch

from specter.imaging import CTF, Poses, Specimen, Structure
from specter.potential import INELASTIC_MFP_PROTEIN_A, ice_inelastic_mfp

for voltage in [300.0, 120.0]:
    print(f"ice at {voltage:3.0f} kV:     {ice_inelastic_mfp(voltage):.0f} Å")
print(f"protein at 300 kV: {INELASTIC_MFP_PROTEIN_A:.0f} Å")

pose = Poses.random(1, seed=0)
# --8<-- [end:setup]

transcript.start("beer-lambert")
# --8<-- [start:beer-lambert]
empty = Specimen(torch.zeros(128, 128, 128, device="cuda"), voxel_size=2.0)

transmission = {}
for voltage in [300.0, 120.0]:
    for thickness in [256.0, 500.0, 1000.0, 1500.0, 2000.0]:
        ice = empty.solvate(model="gd", thickness=thickness)
        waves = ice.scatter(
            pose, voltage=voltage, seed=0, absorption_model="inelastic_mfp"
        )
        mean = float(waves.pixels().abs().square().mean())
        expected = math.exp(-thickness / ice_inelastic_mfp(voltage))
        transmission[voltage, thickness] = mean
        print(
            f"{voltage:3.0f} kV  {thickness:6.0f} Å  zero-loss {mean:.4f}"
            f"  exp(-t/λ) {expected:.4f}"
        )
# --8<-- [end:beer-lambert]

transcript.start("alpha")
# --8<-- [start:alpha]
alpha_transmission = {}
for thickness in [256.0, 500.0, 1000.0, 1500.0, 2000.0]:
    ice = empty.solvate(model="gd", thickness=thickness)
    waves = ice.scatter(pose, voltage=300.0, seed=0, alpha=0.1)
    mean = alpha_transmission[thickness] = float(waves.pixels().abs().square().mean())
    print(
        f"alpha = 0.1  {thickness:6.0f} Å  mean intensity {mean:.4f}"
        f"  as a mean free path {-thickness / math.log(mean):.0f} Å"
    )
# --8<-- [end:alpha]

ts = np.linspace(0, 2100, 200)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for color, voltage in zip(PALETTE, [300.0, 120.0]):
    lam = ice_inelastic_mfp(voltage)
    ax.plot(
        ts,
        np.exp(-ts / lam),
        color=color,
        lw=1.0,
        label=f"exp(−t/{lam:.0f} Å), {voltage:g} kV",
    )
    xs = sorted(t for v, t in transmission if v == voltage)
    ax.plot(
        xs,
        [transmission[voltage, t] for t in xs],
        "o",
        color=color,
        ms=4,
        label=f"inelastic_mfp, {voltage:g} kV",
    )
xs = sorted(alpha_transmission)
ax.plot(
    xs,
    [alpha_transmission[t] for t in xs],
    "s--",
    color=PALETTE[3],
    ms=4,
    lw=0.8,
    label="alpha = 0.1, 300 kV",
)
ax.set(
    xlabel="Ice thickness (Å)",
    ylabel="Mean zero-loss intensity",
    ylim=(0, 1.05),
    xlim=(0, 2100),
)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "inelastic-beer-lambert")

transcript.start("contrast")
# --8<-- [start:contrast]
particle = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256, device="cuda")
specimen = particle.solvate(model="gd", thickness=500.0)

models = {
    "no absorption": dict(alpha=0.0),
    "alpha = 0.1": dict(alpha=0.1),
    "inelastic_mfp": dict(
        absorption_model="inelastic_mfp",
        inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A,
    ),
}
in_focus, defocused = {}, {}
for name, settings in models.items():
    waves = specimen.scatter(pose, voltage=300.0, seed=0, **settings)
    in_focus[name] = waves.apply_ctf(CTF.from_values(0.0)).intensity(cc=None)
    defocused[name] = waves.apply_ctf(CTF.from_values(15000.0)).intensity(cc=None)
    image = in_focus[name].pixels()[0]
    print(f"{name:14s} mean {image.mean():.4f}")
# --8<-- [end:contrast]

# Absorption contrast: the particle's shadow in focus, relative to the ice
# around it, from the projected footprint of the particle's potential.
footprint = as_numpy(particle.potential.sum(0)) > 300.0  # V·Å, inside the ring
ring = np.zeros_like(footprint)
yy, xx = np.indices(footprint.shape)
r = np.hypot(yy - 128, xx - 128)
background = (r > 110) & (r < 125)
for name in models:
    img = lowpass(in_focus[name].pixels()[0], 1.0, 10.0)
    shadow = img[footprint].mean() / img[background].mean() - 1
    print(f"{name:14s} in focus, particle vs ice at 10 Å: {100 * shadow:+.2f}%")

ref = defocused["no absorption"].pixels()[0]
_, p_ref = radial_power(ref / ref.mean(), 1.0)
for name in ["alpha = 0.1", "inelastic_mfp"]:
    img = defocused[name].pixels()[0]
    k, p = radial_power(img / img.mean(), 1.0)
    ratio = p / p_ref
    parts = []
    for lo, hi in [(0.005, 0.02), (0.02, 0.04), (0.04, 0.2)]:
        sel = (k >= lo) & (k < hi)
        parts.append(
            f"{lo:.3f}-{hi:.2f}: {ratio[sel].mean():.2f} (range "
            f"{ratio[sel].min():.2f}-{ratio[sel].max():.2f})"
        )
    print(f"{name:14s} 1.5 µm contrast power / no absorption, " + "; ".join(parts))

fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
for col, name in enumerate(models):
    for row, (label, store) in enumerate(
        [("in focus", in_focus), ("1.5 µm underfocus", defocused)]
    ):
        img = lowpass(store[name].pixels()[0], 1.0, 8.0)
        img = img / img.mean()
        lim = 0.05 if row == 0 else 0.25
        show_image(
            axes[row, col],
            img,
            vmin=1 - lim,
            vmax=1 + lim,
            title=f"{name}, {label}",
            pixel_size=1.0 if col == 0 else None,
            scalebar=50 if col == 0 else None,
        )
fig.tight_layout()
save(fig, "inelastic-contrast")

ref_k, ref_p = radial_power(defocused["no absorption"].pixels()[0], 1.0)
fig, ax = plt.subplots(figsize=(5.4, 3.1))
for color, name in zip(PALETTE[1:], ["alpha = 0.1", "inelastic_mfp"]):
    img = defocused[name].pixels()[0]
    k, p = radial_power(img / img.mean(), 1.0)
    ref = defocused["no absorption"].pixels()[0]
    _, p0 = radial_power(ref / ref.mean(), 1.0)
    keep = k > 0.004
    ax.plot(k[keep], (p / p0)[keep], color=color, lw=1.0, label=name)
ax.axhline(1.0, color="0.5", lw=0.8)
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Contrast power / no absorption",
    xlim=(0, 0.2),
)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "inelastic-contrast-spectrum")

transcript.start("aperture")
# --8<-- [start:aperture]
from specter.constants import energy_to_wavelength
from specter.potential import aperture_mfp_ice

slab = empty.solvate(model="gd", thickness=500.0)
measured = {}
for aperture in [None, 30.0, 12.0, 5.0]:  # semi-angle in mrad
    waves = slab.scatter(
        pose,
        voltage=300.0,
        seed=0,
        absorption_model="inelastic_mfp",
        objective_aperture=aperture,
    )
    mean = measured[aperture] = float(waves.pixels().abs().square().mean())
    rate = 1 / ice_inelastic_mfp(300.0)
    if aperture is not None:
        rate += 1 / aperture_mfp_ice(aperture, 300.0)
    print(
        f"aperture {str(aperture):>5} mrad  zero-loss {mean:.4f}"
        f"  exp(-t/λ_in - t/λ_ap) {math.exp(-500.0 * rate):.4f}"
    )

k_cut = 5e-3 / energy_to_wavelength(300.0)
print(f"a 5 mrad aperture passes spatial frequencies up to {k_cut:.3f} 1/Å")
# --8<-- [end:aperture]

aperture_images = {}
for aperture in [None, 5.0]:
    waves = specimen.scatter(
        pose,
        voltage=300.0,
        seed=0,
        absorption_model="inelastic_mfp",
        inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A,
        objective_aperture=aperture,
    )
    aperture_images[aperture] = (
        waves.apply_ctf(CTF.from_values(15000.0)).intensity(cc=None).pixels()[0]
    )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.1))
apertures = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
lam_in = ice_inelastic_mfp(300.0)
ax1.plot(
    apertures,
    [math.exp(-500 / lam_in - 500 / aperture_mfp_ice(a, 300.0)) for a in apertures],
    "o-",
    color=PALETTE[0],
    ms=3.5,
    lw=1.0,
    label="inelastic + beyond the aperture",
)
ax1.axhline(
    math.exp(-500 / lam_in), color="0.4", lw=0.9, ls="--", label="inelastic only"
)
ax1.plot(
    [a for a in measured if a is not None],
    [m for a, m in measured.items() if a is not None],
    "s",
    color=PALETTE[1],
    ms=6,
    mfc="none",
    label="simulated",
)
ax1.set(
    xscale="log",
    xlabel="Aperture semi-angle (mrad)",
    ylabel="Zero-loss transmission, 500 Å ice",
    ylim=(0.8, 0.9),
)
ax1.set_xticks(apertures)
ax1.set_xticklabels([f"{a:g}" for a in apertures])
ax1.minorticks_off()
ax1.legend(frameon=False, fontsize=7)
for color, (aperture, img) in zip(PALETTE, aperture_images.items()):
    k, p = radial_power(img, 1.0)
    keep = k > 0.01
    ax2.semilogy(
        k[keep],
        p[keep],
        color=color,
        lw=0.9,
        label="no aperture" if aperture is None else f"{aperture:g} mrad",
    )
ax2.axvline(k_cut, color="0.4", lw=0.8, ls=":")
ax2.set(xlabel="Spatial frequency (1/Å)", ylabel="Image power")
ax2.legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "inelastic-aperture")

transcript.start("frozen")
# --8<-- [start:frozen]
from specter.imaging import simulate_frozen
from specter.inelastic import PlasmonFilter, PotentialState
from specter.potential import inelastic_absorption_potential

coarse = Structure.from_pdb("6bdf").render(voxel_size=2.0, box=128, device="cuda")
coarse = coarse.solvate(model="gd", thickness=500.0)
elastic = coarse.materialize(pose, voltage=300.0, seed=0)  # (1, Z, Y, X)
dry = coarse.dry().materialize(pose, voltage=300.0, seed=0)
source = inelastic_absorption_potential(
    dry,
    2.0,
    300.0,
    mfp_solvent_A=ice_inelastic_mfp(300.0),
    mfp_specimen_A=INELASTIC_MFP_PROTEIN_A,
)
state = PotentialState(elastic, source)

zero_loss = {}
for name, plasmons, absorbing in [
    ("elastic only", None, PotentialState(elastic, torch.zeros_like(source))),
    ("local", None, state),
    ("delocalised", PlasmonFilter.approximate_drude(2.0, 300.0, padding=0), state),
]:
    result = simulate_frozen(
        lambda step: absorbing,
        plasmons,
        [40.0],  # one frame of 40 e⁻/Å²
        voxel_size=2.0,
        voltage=300.0,
        nxy=128,
        noise=None,
        device="cuda",
    )
    # The zero-loss intensity before detection, relative to the vacuum beam,
    # on the image's pixel grid: (frames, images, Y, X).
    zero_loss[name] = result.intensities[0, 0]
    print(f"{name:12s} mean zero-loss intensity {zero_loss[name].mean():.4f}")

# What the absorption alone does: each image over the elastic-only one.
local = zero_loss["local"] / zero_loss["elastic only"]
delocalised = zero_loss["delocalised"] / zero_loss["elastic only"]
# --8<-- [end:frozen]

pf = as_numpy(coarse.dry().materialize(pose, voltage=300.0, seed=0)[0].sum(0)) > 150.0
yy, xx = np.indices(pf.shape)
bg = np.hypot(yy - 64, xx - 64) > 55
for name, ratio in [("local", local), ("delocalised", delocalised)]:
    r = as_numpy(ratio)
    print(
        f"{name:12s} absorption: particle {r[pf].mean():.4f}, ice {r[bg].mean():.4f}"
        f" ({100 * (r[pf].mean() / r[bg].mean() - 1):+.2f}%), pixel std {r.std():.4f}"
    )
a, b = as_numpy(local), as_numpy(delocalised)
lo, hi = float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5))
fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
show_image(
    axes[0],
    a,
    vmin=lo,
    vmax=hi,
    title="Local absorption",
    pixel_size=2.0,
    scalebar=50,
    colorbar=None,
)
show_image(axes[1], b, vmin=lo, vmax=hi, title="Delocalised (Drude filter)")
x = np.arange(128) * 2.0 - 128
axes[2].plot(x, a[64], color=PALETTE[0], lw=1.0, label="local")
axes[2].plot(x, b[64], color=PALETTE[1], lw=1.0, label="delocalised")
axes[2].set(
    xlabel="x (Å)", ylabel="Zero-loss / elastic only", title="Line through the centre"
)
axes[2].legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "inelastic-frozen")

# Thumbnail: the zero-loss shadow the particle's absorption casts.
fig, ax = plt.subplots(figsize=(3, 3))
show_image(ax, a[14:114, 14:114], vmin=lo, vmax=hi)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "inelastic-thumbnail")

transcript.close()
