"""
Shared plumbing for the recipe scripts in docs-figures/recipes/.

A recipe is one short task with one figure, and its page in docs/recipes/
quotes the whole script body as a single block a reader can paste and run.
The plotting therefore stays inside the quoted section, in plain matplotlib,
unlike the walkthrough and tutorial scripts, which keep it out of sight. What
is left for this module is where the outputs go: figures in
docs/assets/images/recipes/, printed output in docs-includes/recipes/.

Run a recipe with ``uv run python docs-figures/recipes/<page>.py``.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_walkthrough_common",
    Path(__file__).resolve().parent.parent / "walkthrough" / "_common.py",
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_base.OUT_DIR = _base.REPO_ROOT / "docs" / "assets" / "images" / "recipes"
_base.TEXT_DIR = _base.REPO_ROOT / "docs-includes" / "recipes"

# A quoted recipe ends in plt.show(), which the Agg backend answers with a
# warning; the figure is saved after the quoted section.
warnings.filterwarnings("ignore", message=".*non-interactive.*")

Transcript = _base.Transcript
as_numpy = _base.as_numpy
save = _base.save
show_image = _base.show_image


def thumbnail(image: np.ndarray, name: str, cmap: str = "gray_r") -> None:
    """The recipe's square gallery card: one borderless panel, 1-99% stretch."""
    lo, hi = np.percentile(image, [1, 99])
    fig, ax = plt.subplots(figsize=(3, 3))
    show_image(ax, image, cmap=cmap, vmin=lo, vmax=hi)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, f"{name}-thumbnail")
