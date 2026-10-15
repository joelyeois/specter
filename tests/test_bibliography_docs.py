"""The bibliography page must list what the docs pages cite, written one way."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "docs-figures" / "bibliography.py"


def _load_generator() -> ModuleType:
    """Import ``docs-figures/bibliography.py`` by path (not a package)."""
    spec = importlib.util.spec_from_file_location("bibliography", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bibliography_is_up_to_date() -> None:
    """
    The committed bibliography is what the pages' References sections give.

    A reference added to a page, or removed from one, changes the
    bibliography; regenerating is the fix, not editing the Markdown.
    """
    generator = _load_generator()
    assert generator.OUTPUT.read_text(encoding="utf-8") == generator.render(), (
        "docs-includes/bibliography.md is out of date with the docs pages. "
        "Regenerate it with:\n\n    python docs-figures/bibliography.py\n"
    )


def test_each_doi_is_written_one_way() -> None:
    """Two pages citing one work give the same entry, up to its DOI link."""
    _, conflicts = _load_generator().collect()
    assert not conflicts, "\n".join(conflicts)


def test_entries_continue_over_indented_lines() -> None:
    entries = _load_generator().page_entries(
        "# Page\n\nText.\n\n## References\n\n"
        "- Author, A. (2000). A title.\n  *Journal*, 1, 2–3.\n"
        "- Other, B. (2001). Another.\n\n## Next\n\n- not a reference\n"
    )
    assert entries == [
        "Author, A. (2000). A title. *Journal*, 1, 2–3.",
        "Other, B. (2001). Another.",
    ]
