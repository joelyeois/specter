"""Shared CLI helpers for the mj_dsa4288 scripts."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
REPO_ROOT = PKG_ROOT.parent
RESULTS = PKG_ROOT / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_toml(path: str | Path) -> dict[str, Any]:
    """Read a TOML file into a dict."""
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def default_config(name: str) -> Path:
    """Path of a bundled config file, e.g. ``default_config("phase_a")``."""
    return PKG_ROOT / "configs" / f"{name}.toml"
