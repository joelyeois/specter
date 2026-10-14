"""
Every name the demo notebooks and the docs' Python examples import from
`specter` still exists.

Running the notebooks is minutes of real simulation each, so this checks
the part that goes stale first when the API moves: a renamed or deleted
module, class or function. Each code cell or fenced ``python`` block is
parsed, not executed, and every ``from specter... import name`` and
``import specter...`` in it is resolved.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    p
    for p in (ROOT / "demo-notebooks").rglob("*.ipynb")
    if ".ipynb_checkpoints" not in p.parts
)
DOCS = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
_FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _notebook_sources(path: Path) -> list[str]:
    cells = json.loads(path.read_text())["cells"]
    sources = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython magics and shell escapes are not Python.
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        sources.append("\n".join(lines))
    return sources


def _doc_sources(path: Path) -> list[str]:
    return [m.group(1) for m in _FENCE.finditer(path.read_text())]


def _specter_imports(source: str) -> list[tuple[str, str | None]]:
    """``(module, name)`` for each specter import; name None for a bare import."""
    found: list[tuple[str, str | None]] = []
    # An invalid escape in a plot label is not what this test is about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "specter":
                found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "specter":
                    found.append((alias.name, None))
    return found


def _check(sources: list[str], where: Path) -> None:
    missing = []
    for source in sources:
        for module, name in _specter_imports(source):
            try:
                imported = importlib.import_module(module)
            except ImportError as exc:
                missing.append(f"{module}: {exc}")
                continue
            if name is None or hasattr(imported, name):
                continue
            # `from specter import pipelines` imports a submodule.
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(f"{module}.{name}")
    assert not missing, (
        f"{where.relative_to(ROOT)} imports what no longer exists: {missing}"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_resolve(path: Path) -> None:
    _check(_notebook_sources(path), path)


@pytest.mark.parametrize(
    "path",
    [p for p in DOCS if _doc_sources(p)],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_doc_example_imports_resolve(path: Path) -> None:
    _check(_doc_sources(path), path)
