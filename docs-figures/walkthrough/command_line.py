"""
Figures and printed output for docs/walkthrough/command-line.md.

Runs the `specter` commands the page shows, in a scratch directory, and
keeps what they print and write. The commands themselves are written to
docs-includes/walkthrough/ too, so the page shows exactly what was run.

Run with: uv run python docs-figures/walkthrough/command_line.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import TEXT_DIR, Transcript, save, show_image

COMMAND = "specter simulate particles --pdb_source 6bdf --n_particles 8 --seed 0"
DOSES = [5, 20, 80]
SWEEP = (
    "for dose in 5 20 80; do\n"
    "    specter simulate particles --pdb_source 6bdf --n_particles 8 --seed 0 \\\n"
    '        --dose "$dose" --output_dir "dose-$dose"\n'
    "done"
)


def run(command: str, cwd: Path) -> str:
    """Run a shell command in `cwd` and return its standard output."""
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "COLUMNS": "80", "TERM": "dumb"},
    )
    return result.stdout


def tidy(stdout: str) -> str:
    """Drop the progress bar's redraws and the local home directory."""
    lines = [line for line in stdout.splitlines() if "Generating images:" not in line]
    text = "\n".join(lines)
    return re.sub(re.escape(str(Path.home())), "~", text).strip() + "\n"


def lowpass(image: np.ndarray, pixel_size: float, resolution: float) -> np.ndarray:
    """A Gaussian low-pass to `resolution` Angstrom, for display only."""
    ky = np.fft.fftfreq(image.shape[0], pixel_size)[:, None]
    kx = np.fft.fftfreq(image.shape[1], pixel_size)[None, :]
    sigma = 1 / resolution
    envelope = np.exp(-(kx**2 + ky**2) / (2 * sigma**2))
    return np.fft.ifft2(np.fft.fft2(image) * envelope).real


def clipped(image: np.ndarray) -> dict[str, float]:
    """Grey-scale limits at the 1st and 99th percentiles."""
    return {
        "vmin": float(np.percentile(image, 1)),
        "vmax": float(np.percentile(image, 99)),
    }


TEXT_DIR.mkdir(parents=True, exist_ok=True)
(TEXT_DIR / "command_line-command.txt").write_text(COMMAND + "\n")
(TEXT_DIR / "command_line-sweep.txt").write_text(SWEEP + "\n")

workdir = Path(tempfile.mkdtemp(prefix="specter-cli-"))
(TEXT_DIR / "command_line-run.txt").write_text(tidy(run(COMMAND, workdir)))
run(SWEEP, workdir)

os.chdir(workdir)
transcript = Transcript("command_line")

transcript.start("read")
# --8<-- [start:read]
import mrcfile
import starfile

images = mrcfile.read("particles/particles.mrcs")
table = starfile.read("particles/particles.star")

print(f"images: {images.shape}, {images.dtype}")
print(table[["rlnDefocusU", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].head(3))
# --8<-- [end:read]
transcript.close()

fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for i in range(8):
    smooth = lowpass(images[i], 1.0, 10.0)
    show_image(
        axes[0, i],
        images[i],
        title=f"{table['rlnDefocusU'][i] / 1e4:.2f} µm",
        pixel_size=1.0,
        scalebar=50 if i == 0 else None,
        **clipped(images[i]),
    )
    show_image(axes[1, i], smooth, **clipped(smooth))
axes[0, 0].set_ylabel("as written")
axes[1, 0].set_ylabel("10 Å low-pass")
fig.tight_layout()
save(fig, "command_line-stack")

fig, axes = plt.subplots(2, len(DOSES), figsize=(6.6, 4.6))
for col, dose in enumerate(DOSES):
    image = mrcfile.read(f"dose-{dose}/particles.mrcs")[0]
    smooth = lowpass(image, 1.0, 10.0)
    show_image(
        axes[0, col],
        image,
        title=f"--dose {dose}",
        pixel_size=1.0,
        scalebar=50 if col == 0 else None,
        **clipped(image),
    )
    show_image(axes[1, col], smooth, **clipped(smooth))
axes[0, 0].set_ylabel("as written")
axes[1, 0].set_ylabel("10 Å low-pass")
fig.tight_layout()
save(fig, "command_line-dose")

os.chdir(Path(__file__).resolve().parent)
shutil.rmtree(workdir)
