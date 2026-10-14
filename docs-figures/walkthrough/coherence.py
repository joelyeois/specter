"""
Figures and printed output for docs/walkthrough/coherence.md.

Run with: uv run python docs-figures/walkthrough/coherence.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import PALETTE, Transcript, radial_power, save

from specter.aberrations._envelopes import cc_envelope, cs_envelope

transcript = Transcript("coherence")

transcript.start("setup")
# --8<-- [start:setup]
import specter
from specter.imaging import CTF, Poses, Structure

structure = Structure.from_pdb("6bdf")
specimen = structure.render(voxel_size=1.0, box=256, device="cuda").solvate(model="gd")

specter.seed(0)
poses = Poses.random(1)
ctf = CTF.from_values(10000.0, cs=2.7)  # 1 µm underfocus
waves = specimen.scatter(poses, voltage=300.0, seed=0).apply_ctf(ctf)
# --8<-- [end:setup]

transcript.start("focal-spread")
# --8<-- [start:focal-spread]
coherent = waves.intensity(cc=None)
partial = waves.intensity(cc=2.7, energy_spread=0.7)  # the default

for name, image in [("coherent", coherent), ("Cc 2.7 mm, 0.7 eV", partial)]:
    pixels = image.pixels()
    print(f"{name:18s} mean {pixels.mean():.4f}  std {pixels.std():.4f}")
# --8<-- [end:focal-spread]


def power_ratio(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Radial power of image `a` over that of image `b`, on the pixel grid."""
    k, pa = radial_power(a.pixels()[0], 1.0)
    _, pb = radial_power(b.pixels()[0], 1.0)
    return k, pa / pb


ratios = {300.0: power_ratio(partial, coherent)}
waves_120 = specimen.scatter(poses, voltage=120.0, seed=0).apply_ctf(ctf)
ratios[120.0] = power_ratio(
    waves_120.intensity(cc=2.7, energy_spread=0.7), waves_120.intensity(cc=None)
)

transcript.start("quadrature")
# --8<-- [start:quadrature]
from specter.aberrations import FocalSpread
from specter.constants import energy_to_wavelength

nodes = {}
for voltage in [300.0, 120.0]:
    spread = FocalSpread.from_instrument(
        cc=2.7e7,  # Å
        voltage=voltage * 1e3,  # V
        energy_spread=0.7,  # eV, FWHM
        deltaV_V=0.06e-6,
        deltaI_I=0.01e-6,
        wavelength=energy_to_wavelength(voltage),
    )
    # the wave's grid: 1 Å pixels, oversampled twice
    nodes[voltage] = spread.nodes(dx=0.5, image_oversample=2)  # offsets, weights
    n = len(nodes[voltage][0])
    print(f"{voltage:.0f} kV: sigma_f {spread.sigma_f:.1f} Å, {n} defocus nodes")
# --8<-- [end:quadrature]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 2.9), width_ratios=[1, 1.4])
for color, (voltage, (off, w)) in zip(PALETTE, nodes.items()):
    ax1.plot(
        off,
        w,
        "o-",
        color=color,
        ms=3,
        lw=0.8,
        label=f"{voltage:.0f} kV, {len(off)} nodes",
    )
ax1.set(xlabel="Defocus offset (Å)", ylabel="Weight", title="Gauss-Hermite nodes")
ax1.legend(frameon=False)
for color, (voltage, (k, ratio)) in zip(PALETTE, ratios.items()):
    envelope = cc_envelope(
        torch.as_tensor(k**2),
        energy_to_wavelength(voltage),
        2.7e7,
        voltage * 1e3,
        0.7,
        0.06e-6,
        0.01e-6,
    ).numpy()
    ax2.plot(k[1:], ratio[1:], color=color, lw=1.0, label=f"{voltage:.0f} kV, measured")
    ax2.plot(
        k,
        envelope**2,
        color=color,
        lw=1.0,
        ls="--",
        label=rf"{voltage:.0f} kV, $E_{{C_c}}(k)^2$",
    )
for voltage, (k, ratio) in ratios.items():
    shell = k > 0.45
    print(
        f"{voltage:.0f} kV: coherent / partial power, shell 0.45-0.5 1/Å "
        f"{1 / ratio[shell].mean():.3f}; at 0.3 1/Å "
        f"{1 / ratio[(k > 0.29) & (k < 0.31)].mean():.3f}",
        file=sys.__stdout__,
    )
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power, partial / coherent",
    title="Power the focal spread removes",
    ylim=(0, 1.15),
)
ax2.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
save(fig, "coherence-focal-spread")

transcript.start("envelope")
# --8<-- [start:envelope]
thick = structure.render(voxel_size=1.0, box=256, device="cuda").solvate(
    model="gd", thickness=2000.0
)
compared = {}
for voltage in [300.0, 120.0]:
    w = thick.scatter(poses, voltage=voltage, seed=0).apply_ctf(ctf)
    integrated = w.intensity(cc=2.7)  # sum over the focal spread
    envelope = w.intensity(cc=2.7, temporal_coherence="envelope")
    compared[voltage] = (integrated, envelope)
    deficit = envelope.pixels().mean() / integrated.pixels().mean() - 1
    print(f"{voltage:.0f} kV: envelope mean intensity {deficit:+.2%} against the sum")
# --8<-- [end:envelope]

fig, ax = plt.subplots(figsize=(7.0, 2.8))
for color, (voltage, (integrated, envelope)) in zip(PALETTE, compared.items()):
    k, ratio = power_ratio(envelope, integrated)
    ax.plot(k[1:], ratio[1:], color=color, lw=1.0, label=f"{voltage:.0f} kV")
    low = (k > 0) & (k < 0.1)
    _, pe = radial_power(envelope.pixels()[0], 1.0)
    _, pi = radial_power(integrated.pixels()[0], 1.0)
    print(
        f"{voltage:.0f} kV: envelope power below 10 Å {pe[low].sum() / pi[low].sum() - 1:+.2%}, "
        f"per-shell range {ratio[1:].min():.3f} to {ratio[1:].max():.3f}",
        file=sys.__stdout__,
    )
ax.axhline(1.0, color="0.6", lw=0.6)
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power, envelope / integrated",
    title="2000 Å of ice, Cc 2.7 mm, 0.7 eV",
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "coherence-envelope")

transcript.start("spatial")
# --8<-- [start:spatial]
spatial = {}
for defocus in [10000.0, 30000.0]:
    plain = specimen.scatter(poses, voltage=300.0, seed=0)
    defocused = CTF.from_values(defocus, cs=2.7)
    spatial[defocus] = {
        angle: plain.apply_ctf(defocused, convergence_angle=angle).intensity(cc=None)
        for angle in [None, 0.01, 0.05]  # mrad, the illumination's 1/e half-angle
    }
# --8<-- [end:spatial]

fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8), sharey=True)
for ax, (defocus, images) in zip(axes, spatial.items()):
    for color, angle in zip(PALETTE, [0.01, 0.05]):
        k, ratio = power_ratio(images[angle], images[None])
        kt = torch.as_tensor(k)
        predicted = cs_envelope(
            kt,
            energy_to_wavelength(300.0),
            torch.tensor(2.7e7),
            torch.tensor(defocus - waves.defocus_shift),
            angle,
        ).numpy()
        shell = k > 0.45
        print(
            f"{defocus / 1e4:.0f} um, {angle} mrad: measured power ratio, 0.45-0.5 "
            f"1/A {ratio[shell].mean():.3f}, predicted {(predicted[shell] ** 2).mean():.3f}; "
            f"at 0.2 1/A {ratio[(k > 0.19) & (k < 0.21)].mean():.3f}",
            file=sys.__stdout__,
        )
        ax.plot(k[1:], ratio[1:], color=color, lw=1.0, label=f"{angle} mrad, measured")
        ax.plot(
            k,
            predicted**2,
            color=color,
            lw=1.0,
            ls="--",
            label=rf"{angle} mrad, $E_s(k)^2$",
        )
    ax.set(
        xlabel="Spatial frequency (1/Å)",
        title=f"{defocus / 1e4:.0f} µm underfocus",
        ylim=(0, 1.15),
    )
axes[0].set_ylabel("Power relative to a point source")
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
save(fig, "coherence-spatial")

transcript.close()
