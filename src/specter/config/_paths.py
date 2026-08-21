"""Filesystem path helpers shared by every config dataclass."""

from __future__ import annotations

import os
from pathlib import Path

import specter


# specter/__init__.py -> specter/ -> src/ -> repo root. Anchoring here (rather
# than cwd or a caller's __file__) makes path resolution work identically from
# the script (cwd = repo root, per README.md) and the notebook (cwd =
# demo-notebooks/particle_stack/, and Jupyter cells have no __file__ at all).
REPO_ROOT = Path(specter.__file__).resolve().parents[2]

#: Environment variable overriding where downloaded PDB/mmCIF files are cached,
#: mirroring how HF_HOME/TORCH_HOME work for those libraries.
PDB_CACHE_ENV_VAR = "SPECTER_PDB_CACHE"


#: Everything specter writes into a working directory lives under this one
#: folder, so a run leaves a single recognisable directory behind rather than
#: scattering caches and results at top level.
SPECTER_DATA_DIR = "specter-data"


def find_specter_project_root(start: str | Path | None = None) -> Path:
    """
    Find the directory a specter project is rooted at, the way ``git``
    resolves the nearest ancestor containing ``.git``.

    Walks up from ``start`` looking for an existing ``specter-data/``. This
    is what makes running a tracked command (e.g. ``specter reconstruct``)
    from a *subdirectory* of an already-initialised project still land in
    the same project, instead of quietly starting a second, disconnected
    ``specter-data/`` tree right where you happened to be standing -- and,
    with it, job numbering (``J001``, ``J002``, ...) that starts over from
    scratch instead of continuing the real sequence.

    Parameters
    ----------
    start : str or Path, optional
        Directory to start searching from. Defaults to the current working
        directory.

    Returns
    -------
    Path
        The nearest ancestor directory containing ``specter-data/``, if one
        exists. Otherwise ``start`` itself, resolved to an absolute path --
        a directory with no ``specter-data/`` anywhere above it becomes the
        root of a new one, the same way ``git init`` creates a fresh repo
        at cwd rather than erroring when no ``.git`` is found.
    """
    current = Path(start if start is not None else os.getcwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / SPECTER_DATA_DIR).is_dir():
            return candidate
    return current


def default_output_dir(artifact: str) -> str:
    """
    Default output location for one kind of simulated data.

    Each config class writes to a folder named for what it produces
    (`particles`, `micrographs`, `tiltseries`, `tomograms`) rather than a
    shared `output/`, so two different commands run in the same working
    directory don't pile their results into one folder -- and so the folder
    name alone says what is inside, without having to remember which command
    made it.

    Parameters
    ----------
    artifact : str
        Plural name of the artifact produced, e.g. ``"tomograms"``.

    Returns
    -------
    str
        ``specter-data/<artifact>``, relative to the current working
        directory like every other path in a specter config.
    """
    return os.path.join(SPECTER_DATA_DIR, artifact)


def default_pdb_cache_dir() -> str:
    """
    Default location for the downloaded-structure cache.

    Deliberately NOT anchored to `REPO_ROOT`: that only resolves to the repo
    for an editable install from a checkout. Installed as a wheel,
    `specter/__init__.py` lives in `site-packages/specter/`, so `parents[2]`
    would be the virtualenv's `lib/` directory and the cache would be written
    inside the venv.

    Relative, and therefore resolved against the current working directory --
    the same rule every other path in a specter config follows, so there is
    exactly one thing to remember: specter writes into `./specter-data/`. The
    tradeoff is that running from two different directories gives two caches;
    set `$SPECTER_PDB_CACHE` to an absolute path to share one between them.

    Returns
    -------
    str
        `$SPECTER_PDB_CACHE` when set, else `specter-data/pdb`.
    """
    override = os.environ.get(PDB_CACHE_ENV_VAR)
    if override:
        return override
    return os.path.join(SPECTER_DATA_DIR, "pdb")
