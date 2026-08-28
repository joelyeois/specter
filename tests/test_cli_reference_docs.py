"""The generated CLI reference must still match the CLI it documents."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "docs-figures" / "cli_reference.py"
REFERENCE = REPO_ROOT / "docs-includes" / "cli-reference.md"


def _load_generator() -> ModuleType:
    """
    Import ``docs-figures/cli_reference.py`` by path.

    Returns
    -------
    ModuleType
        The generator module. ``docs-figures`` is not a package and its name is
        not an importable identifier, so it is loaded from its path rather than
        by ``import``.
    """
    spec = importlib.util.spec_from_file_location("cli_reference", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_cli_reference_is_up_to_date() -> None:
    """
    The committed reference is what the generator produces from today's CLI.

    This is the guard that makes the page trustworthy: without it, adding a
    flag or editing a help string leaves `docs/api/cli.md` describing a CLI
    that no longer exists, and nothing says so. Regenerating is the fix, not
    editing the Markdown.
    """
    expected = _load_generator().render()
    actual = REFERENCE.read_text()

    assert actual == expected, (
        f"{REFERENCE.relative_to(REPO_ROOT)} is out of date with the `specter` "
        "CLI. Regenerate it with:\n\n"
        "    python docs-figures/cli_reference.py\n"
    )
