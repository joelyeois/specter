"""
Figures and printed output for docs/user-guide/micrograph.md.

Runs the guide's own `specter simulate micrograph` command in a scratch
directory, with the shipped config at a smaller field, and plots what it
wrote.

Run with: uv run python docs-figures/cli_guides/micrograph.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import starfile
from matplotlib.patches import Rectangle

from _common import (
    PALETTE,
    REPO_ROOT,
    clipped,
    lowpass,
    power_spectrum,
    radial_power,
    run,
    save,
    show_image,
    tidy,
    write_text,
)

COMMAND = """specter simulate micrograph \\
    --config configs/micrograph.toml \\
    --pdb_source 6bdf \\
    --micrograph_size 1536 \\
    --seed 0 \\
    --device cuda \\
    --output_dir micrographs"""

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-micrograph-"))
(workdir / "configs").mkdir()
shutil.copy(REPO_ROOT / "configs" / "micrograph.toml", workdir / "configs")
write_text("micrograph-command", COMMAND + "\n")
write_text("micrograph-run", tidy(run(COMMAND, workdir)))

stack = mrcfile.read(workdir / "micrographs" / "micrographs.mrcs")
table = starfile.read(workdir / "micrographs" / "micrographs.star")
table = table["micrographs"] if isinstance(table, dict) else table
pixel = float(table["rlnImagePixelSize"].iloc[0])
micrograph = stack[0].astype(np.float64)
defocus = float(table["rlnDefocusU"].iloc[0])
write_text(
    "micrograph-output",
    f"stack {stack.shape}, {pixel:.1f} Å/px, mean {micrograph.mean():.1f} counts\n"
    f"defocus {defocus:.0f} Å, dose {float(table['specterDosePerAngstrom'].iloc[0]):.0f}"
    " e-/Å^2\n",
)

# The micrograph at 15 Å, a 400 Å zoom at 8 Å, and its power spectrum.
smooth = lowpass(micrograph, pixel, 15.0)
zoom = slice(568, 968)
detail = lowpass(micrograph, pixel, 8.0)[zoom, zoom]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
show_image(
    axes[0],
    smooth,
    title="Micrograph, 15 Å low-pass",
    pixel_size=pixel,
    scalebar=500,
    **clipped(smooth),
)
axes[0].add_patch(
    Rectangle((568, 568), 400, 400, fill=False, edgecolor=PALETTE[1], lw=1.2)
)
show_image(
    axes[1],
    detail,
    title="400 Å zoom, 8 Å low-pass",
    pixel_size=pixel,
    scalebar=100,
    **clipped(detail),
)
spectrum = power_spectrum(micrograph)
show_image(axes[2], spectrum, title="Power spectrum", **clipped(spectrum, 1, 99.5))
fig.tight_layout()
save(fig, "micrograph-image")

# The rotational average of the spectrum, with the CTF's zeros for the
# micrograph's defocus.
k, power = radial_power(micrograph, pixel)
wavelength = 0.019687  # Å at 300 kV
cs = 2.0e7  # Å, the config's 2 mm
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.semilogy(k[2:], power[2:], color=PALETTE[0], lw=1.0)
for n in range(1, 12):
    # Zeros of sin(chi) for chi = pi*lambda*k^2*(Cs*lambda^2*k^2/2 - df): the
    # positive root of the quadratic in k^2 for chi = -n*pi.
    a = 0.5 * cs * wavelength**3
    b = -defocus * wavelength
    k2 = (-b - np.sqrt(b * b - 4 * a * n)) / (2 * a)
    if k2 > 0 and np.sqrt(k2) < k[-1]:
        ax.axvline(np.sqrt(k2), color="0.6", lw=0.6, ls=":")
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power",
    title=f"Rotationally averaged power, {defocus / 1e4:.2f} µm underfocus",
)
fig.tight_layout()
save(fig, "micrograph-spectrum")

shutil.rmtree(workdir)
