"""
Figures and printed output for docs/user-guide/ice-cache.md.

Runs `specter build ice` for one small configuration in a scratch directory
and compares its structure factor with the bundled library's.

Run with: uv run python docs-figures/cli_guides/ice_cache.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, run, save, tidy, write_text
from specter.arrays import radial_profile_3d, soft_voxelize_coordinates
from specter.fft import fft3
from specter.ice import IceBank

COMMAND = (
    "specter build ice --n_configs 1 --n 64 --dx 1.0 --device cuda --output_dir ice"
)

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-ice-"))
write_text("ice_cache-command", COMMAND + "\n")
print(tidy(run(COMMAND, workdir)))

# The diagnostics "Checking what was generated" discusses, read from the
# manifest. Wall time is left out: it measures the GPU the run shared, not
# the configuration (the Cost tables are the measured costs).
manifest = json.loads((workdir / "ice" / "manifest.json").read_text())
write_text(
    "ice_cache-summary",
    "\n".join(
        f"{c['file']}: {c['box_L']:.0f} Å cell, {c['n_atoms']} water beads\n"
        f"steps {c['n_steps_actual']} of {c['n_steps']},"
        f" stopped_early {str(c['stopped_early']).lower()}\n"
        f"S(k) loss {c['sk_loss']:.2e}, ML-BOP energy {c['E_per_atom']:.3f} eV/atom"
        for c in manifest["configs"]
    )
    + "\n",
)


def amplitude_profile(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    The rotationally averaged Fourier amplitude of a configuration's molecule
    positions, |F(k)| / sqrt(N), against k in 1/Angstrom: the square root of
    its structure factor, as the library's own diagnostics compute it.
    """
    positions, n, dx = config["positions"], config["n"], config["dx"]
    voxels = soft_voxelize_coordinates(
        positions.float(), grid_shape=(n, n, n), voxel_size=dx, periodic=True
    )
    amplitude = torch.abs(fft3(voxels, shift=True)) / positions.shape[0] ** 0.5
    radius, profile = radial_profile_3d(amplitude, return_r=True)
    k = np.asarray(radius, dtype=float) / (n * dx)
    return k, np.asarray(profile)


new = IceBank(str(workdir / "ice"), progressbars=False)
bundled = IceBank(progressbars=False)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, config in enumerate(bundled._configs[:3]):
    k, profile = amplitude_profile(config)
    keep = (k > 0) & (k <= 0.5)
    ax.plot(
        k[keep],
        profile[keep],
        color="0.6",
        lw=0.8,
        label="bundled library (256 Å cells)" if i == 0 else None,
    )
k, profile = amplitude_profile(new._configs[0])
keep = (k > 0) & (k <= 0.5)
ax.plot(k[keep], profile[keep], color=PALETTE[0], lw=1.4, label="this run (64 Å cell)")
ax.axvline(1 / 3.7, color="0.3", lw=0.6, ls=":")
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel=r"$|F(k)|/\sqrt{N}$",
    title="Structure of the water positions",
    xlim=(0, 0.5),
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "ice_cache-structure")

shutil.rmtree(workdir)
