"""
Figures and printed output for docs/user-guide/particle-stack.md.

Runs the guide's own `specter simulate particles` command in a scratch
directory, with the shipped config, and plots what it wrote.

Run with: uv run python docs-figures/cli_guides/particle_stack.py
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

COMMAND = """specter simulate particles \\
    --config configs/particle.toml \\
    --pdb_source 6bdf \\
    --n_particles 16 \\
    --batchsize 4 \\
    --seed 0 \\
    --device cuda \\
    --output_dir particles"""

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-particles-"))
(workdir / "configs").mkdir()
shutil.copy(REPO_ROOT / "configs" / "particle.toml", workdir / "configs")
write_text("particle_stack-command", COMMAND + "\n")
write_text("particle_stack-run", tidy(run(COMMAND, workdir)))

images = mrcfile.read(workdir / "particles" / "particles.mrcs")
table = starfile.read(workdir / "particles" / "particles.star")
table = table["particles"] if isinstance(table, dict) else table

columns = [
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnOriginXAngst",
    "rlnDefocusU",
    "specterDosePerAngstrom",
]
with np.printoptions(precision=1):
    head = table[columns].head(4).to_string(float_format=lambda x: f"{x:.1f}")
write_text(
    "particle_stack-star",
    f"{len(table)} rows, {len(table.columns)} columns\n{head}\n",
)

fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for i, ax in enumerate(axes.ravel()):
    smooth = lowpass(images[i], 1.0, 15.0)
    show_image(
        ax,
        smooth,
        title=f"{table['rlnDefocusU'][i] / 1e4:.2f} µm",
        pixel_size=1.0,
        scalebar=50 if i == 0 else None,
        **clipped(smooth),
    )
fig.tight_layout()
save(fig, "particle_stack-gallery")

shutil.rmtree(workdir)
