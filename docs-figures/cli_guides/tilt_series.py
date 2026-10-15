"""
Figures and printed output for docs/user-guide/tilt-series.md.

Runs the guide's one-command chain, `specter simulate tiltseries` with
`--tomogram_config`, on the shipped configs in a scratch directory, and
plots the tilts it wrote.

Run with: uv run python docs-figures/cli_guides/tilt_series.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import starfile

from _common import (
    REPO_ROOT,
    clipped,
    lowpass,
    run,
    save,
    show_image,
    tidy,
    write_text,
)

COMMAND = """specter simulate tiltseries \\
    --config configs/tiltseries.toml \\
    --tomogram_config configs/tomogram.toml \\
    --n_tilts 41 \\
    --min_tilt_angle -60 --max_tilt_angle 60 \\
    --seed 0 \\
    --device cuda"""

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-tiltseries-"))
(workdir / "configs").mkdir()
for name in ("tiltseries.toml", "tomogram.toml"):
    shutil.copy(REPO_ROOT / "configs" / name, workdir / "configs")
write_text("tilt_series-command", COMMAND + "\n")
log = tidy(run(COMMAND, workdir))
write_text(
    "tilt_series-run",
    "\n".join(
        line
        for line in log.splitlines()
        if "UserWarning" not in line and not line.startswith("  ✓ tomograms/")
    )
    + "\n",
)

tilts = mrcfile.read(workdir / "tiltseries" / "tilt_series.mrcs")
table = starfile.read(workdir / "tiltseries" / "tilt_series.star")
table = (
    table["particles"] if isinstance(table, dict) and "particles" in table else table
)
if isinstance(table, dict):
    table = next(iter(table.values()))
angle_column = next(c for c in table.columns if "Tilt" in c or "tilt" in c)
pixel = float(table["rlnImagePixelSize"].iloc[0])
angles = table[angle_column].to_numpy(dtype=float)
columns = [angle_column, "rlnDefocusU", "specterDosePerAngstrom"]
columns = [c for c in columns if c in table.columns]
write_text(
    "tilt_series-output",
    f"stack {tilts.shape}, {pixel:.1f} Å/px\n"
    + table[columns].head(5).to_string(float_format=lambda x: f"{x:.1f}")
    + "\n",
)

show = [-60.0, -30.0, 0.0, 30.0, 60.0]
fig, axes = plt.subplots(2, len(show), figsize=(12, 5.0))
for col, angle in enumerate(show):
    i = int(np.argmin(np.abs(angles - angle)))
    image = tilts[i].astype(np.float64)
    whole = lowpass(image, pixel, 40.0)
    show_image(
        axes[0, col],
        whole,
        title=f"{angles[i]:+.0f}° (tilt {i + 1} of {len(angles)})",
        pixel_size=pixel if col == 0 else None,
        scalebar=1000 if col == 0 else None,
        **clipped(whole),
    )
    centre = image.shape[-1] // 2
    zoom = lowpass(image, pixel, 20.0)[
        centre - 200 : centre + 200, centre - 200 : centre + 200
    ]
    show_image(
        axes[1, col],
        zoom,
        pixel_size=pixel if col == 0 else None,
        scalebar=500 if col == 0 else None,
        **clipped(zoom),
    )
axes[0, 0].set_ylabel("whole field, 40 Å low-pass")
axes[1, 0].set_ylabel("central 2000 Å, 20 Å low-pass")
fig.tight_layout()
save(fig, "tilt_series-gallery")

shutil.rmtree(workdir)
