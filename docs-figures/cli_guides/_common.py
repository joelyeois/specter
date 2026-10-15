"""
Shared plumbing for the command-line guide scripts in docs-figures/cli_guides/.

The walkthrough's (``docs-figures/walkthrough/_common.py``), reused as is but
writing figures to docs/assets/images/cli-guides/ and printed output to
docs-includes/cli-guides/. See that module for how a page quotes a script.

Run a guide script with ``uv run python docs-figures/cli_guides/<page>.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "walkthrough" / "_common.py"
_spec = importlib.util.spec_from_file_location("_walkthrough_common", _path)
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

REPO_ROOT = _base.REPO_ROOT
_base.OUT_DIR = REPO_ROOT / "docs" / "assets" / "images" / "cli-guides"
_base.TEXT_DIR = REPO_ROOT / "docs-includes" / "cli-guides"
OUT_DIR = _base.OUT_DIR
TEXT_DIR = _base.TEXT_DIR

PALETTE = _base.PALETTE
Transcript = _base.Transcript
as_numpy = _base.as_numpy
image_row = _base.image_row
power_spectrum = _base.power_spectrum
quantize_png = _base.quantize_png
radial_average = _base.radial_average
radial_power = _base.radial_power
save = _base.save
show_image = _base.show_image


# --- What every guide script does: run a command, then look at what it wrote.

import os
import re
import subprocess

import numpy as np


def run(command: str, cwd: Path) -> str:
    """
    Run a shell command in `cwd` and return what it printed.

    The specter commands are run exactly as a reader types them; the device
    is the command's own ``--device``.
    """
    env = {**os.environ, "COLUMNS": "80", "TERM": "dumb"}
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`{command}` failed ({result.returncode}):\n{result.stdout[-3000:]}\n"
            f"{result.stderr[-3000:]}"
        )
    return result.stdout


def tidy(stdout: str) -> str:
    """Drop progress-bar redraws and the local home directory from output."""
    keep = [
        line
        for line in stdout.splitlines()
        if not re.search(r"\d+%\|", line) and "it/s]" not in line
    ]
    text = "\n".join(keep)
    return re.sub(re.escape(str(Path.home())), "~", text).strip() + "\n"


def write_text(name: str, text: str) -> None:
    """Keep `text` as docs-includes/cli-guides/<name>.txt for a page to include."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / f"{name}.txt").write_text(text, encoding="utf-8")


def lowpass(image: np.ndarray, pixel_size: float, resolution: float) -> np.ndarray:
    """A Gaussian low-pass to `resolution` Angstrom, for display only."""
    ky = np.fft.fftfreq(image.shape[-2], pixel_size)[:, None]
    kx = np.fft.fftfreq(image.shape[-1], pixel_size)[None, :]
    sigma = 1 / resolution
    envelope = np.exp(-(kx**2 + ky**2) / (2 * sigma**2))
    return np.fft.ifft2(np.fft.fft2(image) * envelope).real


def clipped(image: np.ndarray, low: float = 1, high: float = 99) -> dict[str, float]:
    """Grey-scale limits at the given percentiles."""
    return {
        "vmin": float(np.percentile(image, low)),
        "vmax": float(np.percentile(image, high)),
    }
