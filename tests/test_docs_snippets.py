"""
Every snippet a docs page quotes, and every image it shows, exists.

The walkthrough, tutorial and command-line-guide pages pull their code out of
scripts by section name (``--8<-- "script.py:section"``) and their printed
output from docs-includes/. A renamed section or a deleted output breaks only
the docs build, which the test suite does not run; this catches it in
seconds. Whether the printed numbers are current is a separate question,
answered by docs-figures/check_outputs.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted((ROOT / "docs").rglob("*.md"))
SNIPPET = re.compile(r'--8<-- "([^"]+)"')
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+\.(?:png|svg|jpg|gif))")


def _includes(page: Path) -> list[str]:
    return SNIPPET.findall(page.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "page", [p for p in PAGES if _includes(p)], ids=lambda p: p.stem
)
def test_every_quoted_snippet_exists(page: Path) -> None:
    for include in _includes(page):
        path, _, section = include.partition(":")
        target = ROOT / path
        assert target.is_file(), f"{page.name} quotes missing {path}"
        if section:
            text = target.read_text(encoding="utf-8")
            for edge in ("start", "end"):
                marker = f"# --8<-- [{edge}:{section}]"
                assert marker in text, f"{page.name}: {path} has no {marker}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_every_shown_image_exists(page: Path) -> None:
    for image in IMAGE.findall(page.read_text(encoding="utf-8")):
        if image.startswith("http"):
            continue
        assert (page.parent / image).resolve().is_file(), f"{page.name}: {image}"
