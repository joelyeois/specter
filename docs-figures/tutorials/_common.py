"""
Shared plumbing for the tutorial scripts in docs-figures/tutorials/.

The walkthrough's (``docs-figures/walkthrough/_common.py``), reused as is but
writing figures to docs/assets/images/tutorials/ and printed output to
docs-includes/tutorials/. See that module for how a page quotes a script.

Run a tutorial with ``uv run python docs-figures/tutorials/<page>.py``.
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
