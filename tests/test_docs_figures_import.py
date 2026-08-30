"""Every `docs-figures/` script must still import.

These scripts regenerate the figures and tables in `docs/`, and CLAUDE.md
requires them to stay reachable for anyone reproducing one after an
algorithm change. Nothing enforced that: `cryoet_specimen_bilayer.py`
imported two functions deleted in 019f6ed and stayed broken through a
full green suite, because the suite never executes this directory.

Importing is the whole check. It is cheap -- every script guards its work
behind ``if __name__ == "__main__"`` -- and it catches the failure mode
that actually happens, which is a script left pointing at an API that has
been renamed or removed. It deliberately does NOT run them: a full figure
regeneration is minutes of real simulation, which is why these live in
`docs-figures/` and not in the suite in the first place.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs-figures"
SCRIPTS = sorted(p.stem for p in FIGURE_DIR.glob("*.py"))


def test_figure_scripts_are_discoverable() -> None:
    """Guard the guard: an empty glob would make every check below vacuous."""
    assert FIGURE_DIR.is_dir(), f"{FIGURE_DIR} is missing"
    assert len(SCRIPTS) > 10, f"only found {SCRIPTS!r} -- glob likely broken"


@pytest.mark.parametrize("module_name", SCRIPTS)
def test_figure_script_imports(module_name: str) -> None:
    """One case per script, so a failure names the file to fix."""
    # On sys.path because the scripts import their shared `_render` helper
    # as a top-level module, the way running them from the repo root does.
    inserted = str(FIGURE_DIR) not in sys.path
    if inserted:
        sys.path.insert(0, str(FIGURE_DIR))
    try:
        importlib.import_module(module_name)
    finally:
        if inserted:
            sys.path.remove(str(FIGURE_DIR))
