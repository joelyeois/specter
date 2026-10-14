"""
Figures and printed output for docs/tutorials/ice-library.md.

Run with: uv run python docs-figures/tutorials/ice_library.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, save, show_image
from specter.arrays import radial_profile_3d

transcript = Transcript("ice_library")

transcript.start("setup")
# --8<-- [start:setup]
import os
import tempfile

import torch

from specter.ice import (
    MLBOP,
    GradientSKIcemaker,
    IceBank,
    build_one_ice_config,
    ice_config_filename,
)

library = tempfile.mkdtemp(prefix="ice-library-")
# --8<-- [end:setup]

transcript.start("build")
# --8<-- [start:build]
path = os.path.join(library, ice_config_filename(1000))
metadata = build_one_ice_config(
    path, n=64, dx=1.0, seed=1000, device="cuda", progressbars=False
)

print(f"wrote {os.path.basename(path)}: a {metadata['box_L']:.0f} Å periodic cell")
print(f"{metadata['n_steps_actual']} L-BFGS steps in {metadata['wall_time']:.0f} s")
print(f"S(k) loss {metadata['sk_loss']:.2e}")
print(f"ML-BOP energy {metadata['energy']['E_per_atom']:.3f} eV per molecule")
# --8<-- [end:build]

transcript.start("trajectory")
# --8<-- [start:trajectory]
torch.manual_seed(1000)  # the seed build_one_ice_config set
maker = GradientSKIcemaker(n=64, dx=1.0, device="cuda", progressbars=False)
maker.init_random()
start_energy = maker.mlbop_energy()["E_per_atom"]
history = maker.optimize(n_steps=250, record_every=1)
final_energy = maker.mlbop_energy()["E_per_atom"]

print(f"{maker.n_molecules} water molecules in the cell")
print(f"ML-BOP energy {start_energy:.2f} -> {final_energy:.3f} eV per molecule")
print(f"S(k) loss {history['sk_loss'][0]:.2e} -> {history['sk_loss'][-1]:.2e}")
# --8<-- [end:trajectory]

# The random start's |F(k)| profile, the optimised one, and the target, on
# the same physical k bins.
with torch.no_grad():
    torch.manual_seed(1000)
    fresh = GradientSKIcemaker(n=64, dx=1.0, device="cuda", progressbars=False)
    fresh.init_random()
    _, f_random = fresh._sk_loss(fresh.positions)
    random_profile = radial_profile_3d(f_random.detach().cpu())
k = np.arange(len(history["radial_profile"][-1])) * maker.dk
target = as_numpy(maker._f_target_rad_1d)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.1))
ax1.semilogy(history["step"], history["sk_loss"], color=PALETTE[0], lw=1.2)
ax1.set(xlabel="L-BFGS step", ylabel="S(k) loss", title="Optimisation")
n = int(min(len(k), len(target), np.searchsorted(k, 0.5, side="right")))
for profile, label, color in [
    (as_numpy(random_profile), "random placement", PALETTE[1]),
    (as_numpy(history["radial_profile"][-1]), "optimised", PALETTE[0]),
]:
    ax2.plot(k[1:n], profile[1:n], color=color, lw=1.2, label=label)
ax2.plot(k[1:n], target[1:n], color="0.2", lw=1.0, ls="--", label="target")
ax2.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="|F(k)|, radial average",
    title="Fourier amplitude of the positions",
)
ax2.legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "ice_library-optimisation")

transcript.start("bank")
# --8<-- [start:bank]
ours = IceBank(cache_dir=library, device="cuda", progressbars=False)
bundled = IceBank(device="cuda", progressbars=False)  # the library SPECTER ships
print(f"ours: {len(ours)} configuration, bundled: {len(bundled)} configurations")

generator = torch.Generator().manual_seed(0)
crops = {
    name: bank.generate_ice(n=64, dx=1.0, generator=generator)[0]
    for name, bank in [("ours", ours), ("bundled", bundled)]
}
for name, crop in crops.items():
    print(f"{name:8s} crop {tuple(crop.shape)}, mean {crop.mean():.2f} V")
# --8<-- [end:bank]

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, (name, crop) in zip(axes, crops.items()):
    show_image(
        ax,
        as_numpy(crop[32]),
        cmap="gray_r",
        title=f"{name}: central slice",
        pixel_size=1.0,
        scalebar=10,
    )
for color, (name, crop) in zip(PALETTE, crops.items()):
    c = crop - crop.mean()
    power = torch.fft.fftshift(torch.fft.fftn(c).abs() ** 2).cpu() / c.numel()
    profile = as_numpy(radial_profile_3d(power))
    kk = np.arange(len(profile)) / 64.0
    keep = kk <= 0.5
    axes[2].semilogy(kk[keep][1:], profile[keep][1:], color=color, lw=1.2, label=name)
axes[2].axvline(1 / 3.7, color="0.6", ls=":", lw=1)
axes[2].set(xlabel="Spatial frequency (1/Å)", ylabel="Power", title="3D power spectrum")
axes[2].legend(frameon=False)
fig.tight_layout()
save(fig, "ice_library-crops")

# The gallery thumbnail: random placement against the optimised positions,
# which follow the target's peaks.
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(k[1:n], as_numpy(random_profile)[1:n], color=PALETTE[1], lw=2.5)
ax.plot(k[1:n], target[1:n], color="0.75", lw=6)
ax.plot(k[1:n], as_numpy(history["radial_profile"][-1])[1:n], color=PALETTE[0], lw=2)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
save(fig, "ice_library-thumbnail")

transcript.start("specimen")
# --8<-- [start:specimen]
from specter.imaging import Poses, Structure

specimen = Structure.from_pdb("1ubq").render(voxel_size=1.0, box=64, device="cuda")
specimen = specimen.solvate(model="gd", cache_dir=library)
volume = specimen.materialize(Poses.random(1, seed=0), voltage=300.0, seed=0)
print(f"solvated from our library: {tuple(volume.shape)}")
# --8<-- [end:specimen]

transcript.start("tiling")
# --8<-- [start:tiling]
model = MLBOP(device="cuda")
canvases = {}
for relax_steps in [0, 50, 200]:
    positions = ours.big_ice_positions(
        n=128,  # 128 x 128 x 64 Å: two cells along x and y
        dx=1.0,
        nz=64,
        relax_steps=relax_steps,
        generator=torch.Generator().manual_seed(0),
        to_host=True,
    )
    canvases[relax_steps] = positions
    energy = model.compute_energy(
        positions.cuda(), box_size=(128.0, 128.0, 64.0), pbc=True
    )["E_per_atom"]
    print(f"relax_steps={relax_steps:3d}: {energy:.3f} eV per molecule")
# --8<-- [end:tiling]

# ML-BOP has no per-atom output, so the profile is the energy per molecule of
# 3 A thick slabs across x, each cut out of the canvas the same way: their
# absolute level sits above the bulk value, and only differences along x mean
# anything. The tiles meet at x = 0.
centres = np.arange(-60.0, 60.5, 1.0)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for color, (relax_steps, positions) in zip(PALETTE, canvases.items()):
    profile = []
    for xc in centres:
        inside = (positions[:, 0] - xc).abs() < 1.5
        result = model.compute_energy(
            positions[inside].cuda(), box_size=(128.0, 128.0, 64.0), pbc=False
        )
        profile.append(result["E_per_atom"].item())
    ax.plot(centres, profile, color=color, lw=1.2, label=f"relax_steps = {relax_steps}")
ax.axvline(0.0, color="0.6", ls=":", lw=1)
ax.set(
    xlabel="x (Å)",
    ylabel="ML-BOP energy per molecule (eV)",
    title="Across the seam between two tiles",
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "ice_library-seams")

transcript.close()
