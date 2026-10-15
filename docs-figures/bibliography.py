"""
Collect every docs page's references into one bibliography.

Writes `docs-includes/bibliography.md`, which `docs/references.md` pulls in
with a pymdownx.snippets include. Each page cites the work it draws on in its
own `## References` section; this lists every distinct work once, sorted by
author, with links back to the pages that cite it.

Usage
-----
    python docs-figures/bibliography.py

`tests/test_bibliography_docs.py` fails if the committed file no longer
matches what this script produces, and if two pages write the same DOI as two
different entries, so a reference added to a page reaches the bibliography,
and is written the same way everywhere, or a test says so.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
OUTPUT = REPO_ROOT / "docs-includes" / "bibliography.md"
# The bibliography page itself, and internal measurement notes kept beside
# their figures, are not pages that cite.
SKIP = {DOCS / "references.md"}
SKIP_DIRS = {DOCS / "assets"}

# A DOI may hold balanced parentheses (10.1016/0304-3991(88)90003-4), which a
# Markdown link destination keeps as they are.
_DOI_LINK = re.compile(
    r"\[doi:[^\]]+\]\(https://doi\.org/((?:[^()\s]|\([^()\s]*\))+)\)"
)


@dataclass
class Work:
    """One cited work: its entry text and the pages citing it."""

    text: str
    pages: dict[Path, str] = field(default_factory=dict)


def _pages() -> list[Path]:
    return sorted(
        p
        for p in DOCS.rglob("*.md")
        if p not in SKIP and not any(d in p.parents for d in SKIP_DIRS)
    )


def page_entries(text: str) -> list[str]:
    """
    The entries of a page's ``## References`` section, each on one line.

    An entry is a top-level ``- `` bullet and the indented lines that
    continue it; the section ends at the next ``## `` heading.
    """
    if "\n## References" not in "\n" + text:
        return []
    section = ("\n" + text).split("\n## References", 1)[1]
    section = re.split(r"\n## ", section, maxsplit=1)[0]
    entries: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            entries.append(line[2:].strip())
        elif line.startswith(" ") and line.strip() and entries:
            entries[-1] += " " + line.strip()
    return entries


def citation(entry: str) -> tuple[str, str]:
    """
    An entry's key and the text the bibliography shows for it.

    With a DOI, the key is the DOI and the text ends at its link: whatever a
    page appends after it (a chapter or equation) belongs to that page.
    Without one, the whole entry is both.
    """
    match = _DOI_LINK.search(entry)
    if match is None:
        return entry.casefold(), entry
    return match.group(1).casefold(), entry[: match.end()]


def _title(page: Path) -> str:
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return page.stem


def _sort_key(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text)
    return "".join(c for c in plain if not unicodedata.combining(c)).casefold()


def collect() -> tuple[dict[str, Work], list[str]]:
    """
    Every cited work by key, and every DOI written two different ways.

    Returns
    -------
    works : dict of str to Work
    conflicts : list of str
        One line per DOI whose entry text differs between pages.
    """
    works: dict[str, Work] = {}
    conflicts: list[str] = []
    for page in _pages():
        for entry in page_entries(page.read_text(encoding="utf-8")):
            key, text = citation(entry)
            work = works.setdefault(key, Work(text))
            if work.text != text:
                conflicts.append(
                    f"{key}: {page.relative_to(DOCS)} writes it differently from "
                    f"{', '.join(str(p.relative_to(DOCS)) for p in work.pages)}"
                )
            work.pages[page] = _title(page)
    return works, conflicts


def render() -> str:
    """The bibliography's Markdown."""
    works, _ = collect()
    lines = []
    for work in sorted(works.values(), key=lambda w: _sort_key(w.text)):
        cited = ", ".join(
            f"[{title}]({page.relative_to(DOCS).as_posix()})"
            for page, title in sorted(work.pages.items())
        )
        lines.append(f"- {work.text} Cited in {cited}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    _, conflicts = collect()
    for conflict in conflicts:
        print(f"inconsistent: {conflict}")
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
