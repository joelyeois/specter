"""Filesystem path helpers shared by every config dataclass."""

from __future__ import annotations

from specter.progress import console

import os
import sys
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

#: Marker file naming the root of a specter project, the way ``.git`` names
#: the root of a repository (RELION does the same with ``.gui_projectdir``).
#: A marker rather than an umbrella output folder: results live at the top of
#: the project (``tomograms/``, ``particles/``, ...) instead of one level down
#: inside a wrapper directory, so a project reads like a RELION or CryoSPARC
#: one. The file's *presence* is the whole signal -- nothing reads its
#: contents, so it may be empty.
PROJECT_MARKER = ".specter"


def find_specter_project_root(start: str | Path | None = None) -> Path:
    """
    Find the directory a specter project is rooted at, the way ``git``
    resolves the nearest ancestor containing ``.git``.

    Walks up from ``start`` looking for a `PROJECT_MARKER` file. This is
    what makes running a tracked command (e.g. ``specter reconstruct``)
    from a *subdirectory* of an already-initialised project still land in
    the same project, instead of quietly starting a second, disconnected
    job tree right where you happened to be standing -- and, with it, job
    numbering (``J001``, ``J002``, ...) that starts over from scratch
    instead of continuing the real sequence.

    Pure: never creates the marker, never prompts. `ensure_project_root`
    is the find-or-create counterpart, and is deliberately the only thing
    with side effects -- a non-main DDP rank calls *this* to agree on a
    path without racing its siblings to create anything.

    Parameters
    ----------
    start : str or Path, optional
        Directory to start searching from. Defaults to the current working
        directory.

    Returns
    -------
    Path
        The nearest ancestor directory containing `PROJECT_MARKER`, if one
        exists. Otherwise ``start`` itself, resolved to an absolute path --
        a directory with no marker anywhere above it is where a new project
        would be rooted, the same way ``git init`` creates a fresh repo at
        cwd rather than erroring when no ``.git`` is found.
    """
    current = Path(start if start is not None else os.getcwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_MARKER).is_file():
            return candidate
    return current


def ensure_project_root(
    start: str | Path | None = None, *, interactive: bool | None = None
) -> Path:
    """
    Find the project root, creating the marker if there isn't one yet.

    The find-or-create counterpart to `find_specter_project_root`, and the
    only function here with side effects. Call it exactly once per run,
    from the process that owns the run -- see that function's note on DDP
    ranks.

    When no ancestor carries a marker, ``start`` becomes a new project
    root. Whether that is confirmed first depends on ``interactive``:
    RELION prompts ("Do you want to start a new project here?") from its
    GUI but takes a ``--do_projdir`` flag to skip the dialog, and the same
    split applies here for the same reason. A blocking prompt in a script,
    a batch job or CI would hang forever, so the prompt is offered only
    when there is a terminal to answer it.

    Parameters
    ----------
    start : str or Path, optional
        Directory to start searching from, and to root a new project at if
        no marker is found. Defaults to the current working directory.
    interactive : bool, optional
        Whether to ask before creating a marker. Default (``None``) decides
        from ``sys.stdin.isatty()``: ask at a terminal, create silently
        (with a printed notice) otherwise.

    Returns
    -------
    Path
        The project root, now guaranteed to carry a marker.

    Raises
    ------
    SystemExit
        If the user declines the prompt. Declining means "not here", and
        continuing would write the job tree into a directory the user just
        said no to.
    """
    root = find_specter_project_root(start)
    marker = root / PROJECT_MARKER
    if marker.is_file():
        return root

    if interactive is None:
        interactive = sys.stdin.isatty()

    if interactive:
        answer = input(
            f"{root} is not a specter project.\n"
            f"Start a new project here (creates {PROJECT_MARKER})? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            raise SystemExit(
                "Declined -- no project created. Run from an existing project "
                "directory, or pass an explicit --output_dir."
            )

    marker.touch()
    console.print(f"Initialised specter project at {root}")
    return root


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
        ``<artifact>``, relative to the current working directory like every
        other path in a specter config.
    """
    return artifact


def default_pdb_cache_dir() -> str:
    """
    Default location for the downloaded-structure cache.

    A user-level cache rather than a per-project directory, following the
    XDG Base Directory convention (``$XDG_CACHE_HOME``, else ``~/.cache``)
    that `torch.hub` and HuggingFace already use for the same job. A
    structure fetched from RCSB is regenerable and identical for every
    project, so caching it once per user downloads it once rather than once
    per working directory.

    Deliberately NOT anchored to `REPO_ROOT`: that only resolves to the repo
    for an editable install from a checkout. Installed as a wheel,
    `specter/__init__.py` lives in `site-packages/specter/`, so `parents[2]`
    would be the virtualenv's `lib/` directory and the cache would be written
    inside the venv.

    This holds *only* downloads. A structure the user supplies by path is
    read where it lies (see `specter.pdb.PDB`) and never copied here, which
    is what keeps ``specter cache clean`` safe: everything in this directory
    can be re-fetched, so nothing irreplaceable is ever in it.

    Returns
    -------
    str
        `$SPECTER_PDB_CACHE` when set, else ``$XDG_CACHE_HOME/specter/pdb``,
        else ``~/.cache/specter/pdb``.
    """
    override = os.environ.get(PDB_CACHE_ENV_VAR)
    if override:
        return override
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return str(base / "specter" / "pdb")
