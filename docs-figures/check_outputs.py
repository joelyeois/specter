"""
Rerun the docs' figure scripts and report what their printed output changed.

The walkthrough, tutorial and command-line-guide pages quote their code from
a script and show what it printed (docs-includes/<section>/<page>-<name>.txt),
but their prose also types numbers from those runs. A code change can move an
output and leave the prose behind, which no test sees. This reruns the
scripts, diffs every output file against the version before the run, and
lists, for each page that shows a changed output, every number the page
quotes that appeared in the old output and is missing from the new one --
the places to re-read.

    uv run python docs-figures/check_outputs.py                 # every script
    uv run python docs-figures/check_outputs.py ctf tilt_series # by stem

Scripts run one at a time on ``CUDA_VISIBLE_DEVICES`` as set. Figures and
outputs are rewritten in place; review with ``git diff`` and commit what is
intended. Run it before a release, alongside tools/cli-qa/.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = {
    "walkthrough": ROOT / "docs" / "walkthrough",
    "tutorials": ROOT / "docs" / "tutorials",
    "cli_guides": ROOT / "docs" / "user-guide",
}
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])")


def scripts(stems: list[str]) -> list[Path]:
    """Every page script, or those whose stem was asked for."""
    found = []
    for section in SECTIONS:
        folder = ROOT / "docs-figures" / section
        if folder.is_dir():
            found += sorted(
                p for p in folder.glob("*.py") if not p.name.startswith("_")
            )
    return [p for p in found if not stems or p.stem in stems]


def outputs() -> dict[Path, str]:
    """The text every script printed last time, by path."""
    texts = {}
    for folder in (ROOT / "docs-includes").iterdir():
        if folder.is_dir():
            for path in folder.glob("*.txt"):
                texts[path] = path.read_text(encoding="utf-8")
    return texts


def pages_including(path: Path) -> list[Path]:
    """The Markdown pages that show the output file `path`."""
    rel = path.relative_to(ROOT).as_posix()
    return [
        page
        for page in (ROOT / "docs").rglob("*.md")
        if rel in page.read_text(encoding="utf-8")
    ]


def main(stems: list[str]) -> int:
    before = outputs()
    env = dict(os.environ, PYTHONUTF8="1")
    failed = []
    for script in scripts(stems):
        print(f"running {script.relative_to(ROOT)}", flush=True)
        run = subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,
            env=env,
            capture_output=True,
            text=True,
        )
        if run.returncode:
            failed.append(script)
            print(run.stderr[-2000:])
    after = outputs()

    stale = 0
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path, ""), after.get(path, "")
        if old == new:
            continue
        print(f"\n== {path.relative_to(ROOT)} changed")
        gone = set(NUMBER.findall(old)) - set(NUMBER.findall(new))
        for page in pages_including(path):
            prose = page.read_text(encoding="utf-8")
            for number in sorted(gone & set(NUMBER.findall(prose))):
                stale += 1
                print(f"   {page.relative_to(ROOT)} quotes {number}, gone from it")
    if failed:
        print("\nscripts that failed: " + ", ".join(str(p.name) for p in failed))
    print(f"\n{stale} quoted number(s) to re-check; {len(failed)} script(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
