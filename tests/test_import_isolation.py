"""
Pins which modules are allowed to pull `lightning` in at import time.

`lightning` costs roughly 6 s on top of `torch` on a warm cache, because it
eagerly imports `torchmetrics` (which in turn imports `matplotlib`). Only the
reconstructors' training loop uses it; the forward model's modules have been
plain `nn.Module`s (`specter._device_module.DeviceModule`) since 2026-11-07.
The metadata and maths modules must not pull in matplotlib either.

Each check runs in a subprocess: `sys.modules` is process-global, so a test
running after anything that already imported the reconstructor would trivially
pass in-process. A subprocess import costs 3-5 s, so each list of modules is
checked in ONE interpreter rather than one each, and only a failure pays to
bisect which of them was responsible.
"""

from __future__ import annotations

import functools
import subprocess
import sys
import tempfile
import textwrap

import pytest

# Modules a user may reasonably import for a small helper, none of which need
# Lightning. `specter.io` reads/writes .cs and .star metadata; the rest are
# array, coordinate and rotation maths.
LIGHTNING_FREE_MODULES = [
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

# The forward model: free of Lightning, though some of it plots.
SIMULATOR_MODULES = [
    "specter.aberrations",
    "specter.crowding",
    "specter.detectors",
    "specter.ice",
    "specter.imaging",
    "specter.pipelines",
    "specter.potential",
    "specter.scattering",
    "specter.specimen",
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


def _blame(heavy: str, modules: list[str]) -> str:
    """
    Name the light module(s) responsible for pulling in a heavy dependency.

    Parameters
    ----------
    heavy : str
        Top-level package that should not have been imported, e.g.
        ``"lightning"``.
    modules : list of str
        The modules that were imported together.

    Returns
    -------
    str
        Comma-separated culprits, for the assertion message. Only called on
        failure, since it costs one interpreter launch per light module.
    """
    culprits = [
        module for module in modules if heavy in _loaded_modules(f"import {module}")
    ]
    return ", ".join(culprits) or "(none individually; check import order)"


@pytest.mark.parametrize(
    "modules, heavy",
    [(LIGHTNING_FREE_MODULES, h) for h in ("lightning", "torchmetrics", "matplotlib")]
    + [(SIMULATOR_MODULES, h) for h in ("lightning", "torchmetrics")],
)
def test_modules_do_not_import_heavy_deps(modules: list[str], heavy: str) -> None:
    """
    Neither the light modules nor the forward model may drag in Lightning.

    Each list is imported into a single interpreter, so one subprocess covers
    it; the per-module bisect only runs if that fails.
    """
    loaded = _loaded_modules("; ".join(f"import {m}" for m in modules))
    assert heavy not in loaded, (
        f"{_blame(heavy, modules)} now import(s) {heavy} at module level. "
        f"Lightning costs ~6 s on top of torch and pulls torchmetrics and "
        f"matplotlib with it. Guard the offending import with `if "
        f"TYPE_CHECKING:` if it is annotation-only, or import it where it is "
        f"used."
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
