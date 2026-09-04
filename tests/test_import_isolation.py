"""
Pins which modules are allowed to pull `lightning` in at import time.

`lightning` costs roughly 6 s on top of `torch` on a warm cache, because it
eagerly imports `torchmetrics` (which in turn imports `matplotlib`). That is
unavoidable for the simulator and the reconstructor, whose classes genuinely
subclass `L.LightningModule`. It is pure overhead for the modules that only
read metadata or do array maths, and two of them used to pay it anyway:

* `specter.io` imported `..imagegenerator` in `_relion.py` for a single type
  annotation, taking it from ~4 s to ~11 s.
* `specter.rotations` eagerly re-exported `VolumeRotator`, so importing it for
  `rotate_volume` cost ~9.4 s instead of ~4.8 s. `specter.symmetries` inherited
  that through its own `..rotations` import.

Both were fixed 2026-09-04 (a `TYPE_CHECKING` guard and a PEP 562 lazy
`__getattr__` respectively). These tests fail if either regresses, or if a new
eager `lightning` import reaches one of the light modules.

Each check runs in a subprocess: `sys.modules` is process-global, so a test
running after anything that already imported the simulator would trivially
pass in-process. A subprocess import costs 3-5 s, so the light modules are
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
        ``"lightning"``.

    Returns
    -------
    str
        Comma-separated culprits, for the assertion message. Only called on
        failure, since it costs one interpreter launch per light module.
    """
    culprits = [
        module
        for module in LIGHTNING_FREE_MODULES
        if heavy in _loaded_modules(f"import {module}")
    ]
    return ", ".join(culprits) or "(none individually; check import order)"


@pytest.mark.parametrize("heavy", ["lightning", "torchmetrics", "matplotlib"])
def test_light_modules_do_not_import_heavy_deps(heavy: str) -> None:
    """
    None of the metadata/maths modules may drag in the Lightning stack.

    All of them are imported into a single interpreter, so one subprocess
    covers the whole list; the per-module bisect only runs if that fails.
    """
    loaded = _loaded_modules("; ".join(f"import {m}" for m in LIGHTNING_FREE_MODULES))
    assert heavy not in loaded, (
        f"{_blame(heavy)} now import(s) {heavy} at module level. Lightning "
        f"costs ~6 s on top of torch and pulls torchmetrics and matplotlib "
        f"with it. Guard the offending import with `if TYPE_CHECKING:` if it "
        f"is annotation-only, or re-export it lazily via a module "
        f"`__getattr__` (see specter/rotations/__init__.py)."
    )


def test_volume_rotator_is_still_reachable_from_the_package() -> None:
    """
    The lazy re-export must be transparent.

    `from specter.rotations import VolumeRotator` has to keep returning the
    same class object as reaching into the private submodule, or the PEP 562
    `__getattr__` has changed behaviour rather than just deferring it.
    """
    from specter.rotations import VolumeRotator
    from specter.rotations._volume_rotator import VolumeRotator as Direct

    assert VolumeRotator is Direct


def test_volume_rotator_access_is_what_loads_lightning() -> None:
    """Touching the rotator does load Lightning; that part is expected."""
    loaded = _loaded_modules("from specter.rotations import VolumeRotator")
    assert "lightning" in loaded


def test_rotations_dir_advertises_the_lazy_names() -> None:
    """`dir()` must list lazy exports so tab completion still finds them."""
    import specter.rotations as rotations

    listed = dir(rotations)
    for name in ("VolumeRotator", "_resolve_roi", "_normalize_slice_indices"):
        assert name in listed


def test_rotations_still_raises_attribute_error_for_unknown_names() -> None:
    """The `__getattr__` must not swallow genuine typos."""
    import specter.rotations as rotations

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_here'"):
        rotations.definitely_not_here
