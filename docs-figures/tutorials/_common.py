"""
Shared plumbing for the tutorial scripts in docs-figures/tutorials/.

The walkthrough's (``docs-figures/walkthrough/_common.py``), reused as is but
writing figures to docs/assets/images/tutorials/ and printed output to
docs-includes/tutorials/. See that module for how a page quotes a script.

A short recipe (docs/tutorials/index.md lists them apart) differs in one way:
its page quotes the whole script body as a single block a reader can paste
and run, so its plotting stays inside the quoted section, in plain
matplotlib ending in ``plt.show()``.

Run a tutorial with ``uv run python docs-figures/tutorials/<page>.py``.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_path = Path(__file__).resolve().parents[1] / "walkthrough" / "_common.py"
_spec = importlib.util.spec_from_file_location("_walkthrough_common", _path)
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

REPO_ROOT = _base.REPO_ROOT
_base.OUT_DIR = REPO_ROOT / "docs" / "assets" / "images" / "tutorials"
_base.TEXT_DIR = REPO_ROOT / "docs-includes" / "tutorials"
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

# A quoted recipe ends in plt.show(), which the Agg backend answers with a
# warning; the figure is saved after the quoted section.
warnings.filterwarnings("ignore", message=".*non-interactive.*")


def thumbnail(image: np.ndarray, name: str, cmap: str = "gray_r") -> None:
    """A recipe's square index card: one borderless panel, 1-99% stretch."""
    lo, hi = np.percentile(image, [1, 99])
    fig, ax = plt.subplots(figsize=(3, 3))
    show_image(ax, image, cmap=cmap, vmin=lo, vmax=hi)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, f"{name}-thumbnail")
