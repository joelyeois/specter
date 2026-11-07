"""
Pins what the light modules, and the `specter` command, pull in at import
time.

The metadata and maths modules must not import matplotlib, which costs
seconds and is only needed to draw. Lightning, whose `torchmetrics` import
used to drag matplotlib into every simulator and reconstructor module, has
not been a dependency since 2026-11-07, when the reconstructors moved to
their own training loop (`specter.ghostbuster.fit`).

Each check runs in a subprocess: `sys.modules` is process-global, so a test
running after anything that already imported the plotting code would
trivially pass in-process. A subprocess import costs 3-5 s, so the light
modules are checked in ONE interpreter rather than one each, and only a
failure pays to bisect which of them was responsible.
"""

from __future__ import annotations

import functools
import subprocess
import sys
import tempfile
import textwrap

import pytest

# Modules a user may reasonably import for a small helper, none of which
# draw. `specter.io` reads/writes .cs and .star metadata; the rest are
# array, coordinate and rotation maths.
LIGHT_MODULES = [
    "specter",
    "specter.arrays",
    "specter.constants",
    "specter.coords",
    "specter.fft",
    "specter.filters",
    "specter.io",
    "specter.rotations",
    "specter.symmetries",
]


@functools.lru_cache(maxsize=None)
def _loaded_modules(import_stmt: str) -> frozenset[str]:
    """
    Import something in a fresh interpreter and report what got loaded.

    Parameters
    ----------
    import_stmt : str
        Python import statement to execute, e.g. ``"import specter.io"``.

    Returns
    -------
    frozenset of str
        The names in ``sys.modules`` after the statement runs. Cached, so
        repeating a statement across tests costs one interpreter launch.

    Raises
    ------
    AssertionError
        If the subprocess exited non-zero, carrying its stderr.
    """
    script = textwrap.dedent(f"""
        {import_stmt}
        import sys
        print("\\n".join(sys.modules))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        # Force a neutral cwd rather than inheriting pytest's. `python -c` puts
        # the working directory on sys.path, and src/specter/ contains an `io/`
        # package that shadows the stdlib `io` module from there, which fails
        # every import in this file with a confusing relative-import error.
        cwd=tempfile.gettempdir(),
    )
    assert proc.returncode == 0, f"{import_stmt!r} failed:\n{proc.stderr}"
    return frozenset(proc.stdout.split())


def _blame(heavy: str) -> str:
    """
    Name the light module(s) responsible for pulling in a heavy dependency.

    Parameters
    ----------
    heavy : str
        Top-level package that should not have been imported, e.g.
        ``"matplotlib"``.

    Returns
    -------
    str
        Comma-separated culprits, for the assertion message. Only called on
        failure, since it costs one interpreter launch per light module.
    """
    culprits = [
        module
        for module in LIGHT_MODULES
        if heavy in _loaded_modules(f"import {module}")
    ]
    return ", ".join(culprits) or "(none individually; check import order)"


def test_light_modules_do_not_import_matplotlib() -> None:
    """
    None of the metadata/maths modules may drag in matplotlib.

    All of them are imported into a single interpreter, so one subprocess
    covers the whole list; the per-module bisect only runs if that fails.
    """
    loaded = _loaded_modules("; ".join(f"import {m}" for m in LIGHT_MODULES))
    assert "matplotlib" not in loaded, (
        f"{_blame('matplotlib')} now import(s) matplotlib at module level. "
        f"Guard the offending import with `if TYPE_CHECKING:` if it is "
        f"annotation-only, or import it where it is used."
    )


@pytest.mark.parametrize("heavy", ["torch", "numpy"])
def test_cli_entry_point_does_not_import_torch(heavy: str) -> None:
    """
    Loading the `specter` command must not pay for torch or numpy.

    `specter --help` and argument errors import only the CLI; torch alone is
    ~3 s of it. `specter.seed` used to import both at module level, which
    every `import specter` inherited.
    """
    assert heavy not in _loaded_modules("import specter.cli._cli")


def test_specter_seed_still_seeds_every_generator() -> None:
    """The lazily importing `specter.seed` seeds random, numpy and torch."""
    import random

    import numpy as np
    import torch

    import specter

    draws = []
    for _ in range(2):
        specter.seed(1234)
        draws.append((random.random(), float(np.random.rand()), float(torch.rand(1))))
    assert draws[0] == draws[1]
