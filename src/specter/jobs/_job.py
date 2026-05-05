from __future__ import annotations

import json
import os
import subprocess
import types
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import matplotlib.figure
    import torch


_SESSION_BASE_DIR: Path | None = None


def base_directory(path: str | Path) -> None:
    """
    Set the default job output directory for the current Python session.

    Call this once at the top of your notebook or script before creating any
    :class:`Job`. It overrides the ``SPECTER_JOBS_DIR`` environment variable.
    Alternatively, pass ``base_dir`` explicitly to each :class:`Job` call, or
    set ``SPECTER_JOBS_DIR`` in your shell environment.

    Parameters
    ----------
    path : str or Path
        Root directory under which project folders and job folders are created.

    Examples
    --------
    >>> import specter.jobs as jobs
    >>> jobs.base_directory("/scratch/user/cryo-runs")
    >>> with jobs.Job("ghostbuster", "apoferritin") as job:
    ...     print(job.dir)   # /scratch/user/cryo-runs/apoferritin/J001
    """
    global _SESSION_BASE_DIR
    _SESSION_BASE_DIR = Path(path)


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    if _SESSION_BASE_DIR is not None:
        return _SESSION_BASE_DIR
    env = os.environ.get("SPECTER_JOBS_DIR")
    if env:
        return Path(env)
    raise RuntimeError(
        "No job output directory configured. Do one of:\n"
        "  1. jobs.base_directory('/your/data/path')  # recommended in notebooks\n"
        "  2. export SPECTER_JOBS_DIR=/your/data/path  # in your shell / PBS script\n"
        "  3. Job('type', 'project', base_dir='/your/data/path')  # per-job override"
    )


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _get_specter_version() -> str:
    try:
        return version("specter")
    except PackageNotFoundError:
        return "unknown"


def _next_job_id(project_dir: Path) -> str:
    existing = sorted(project_dir.glob("J[0-9][0-9][0-9]"))
    if not existing:
        return "J001"
    last = int(existing[-1].name[1:])
    if last >= 999:
        raise RuntimeError("Job ID overflow: project has 999 jobs. Use a new project.")
    return f"J{last + 1:03d}"


def _serialize_value(v: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(i) for i in v]
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return {
                "__type__": "Tensor",
                "shape": list(v.shape),
                "dtype": str(v.dtype).replace("torch.", ""),
            }
    except ImportError:
        pass
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    return {"__type__": type(v).__name__, "repr": str(v)[:200]}


class Job:
    """
    Context manager that creates a unique output folder for a SPECTER job and
    records a parameter snapshot alongside the outputs.

    Parameters
    ----------
    job_type : str
        Free-form label, e.g. ``"ghostbuster"``, ``"tilt-series"``.
    project : str
        Groups related jobs under a shared folder.
    base_dir : str or Path, optional
        Root directory for all job folders. Defaults to ``~/specter-data/``
        or the ``SPECTER_JOBS_DIR`` environment variable.
    """

    def __init__(
        self,
        job_type: str,
        project: str,
        base_dir: str | Path | None = None,
        job_id: str | None = None,
    ) -> None:
        self._job_type = job_type
        self._project = project
        self._base_dir = _resolve_base_dir(base_dir)
        self._resume_job_id = job_id
        self._is_resume = False
        self._dir: Path | None = None
        self._job_id: str | None = None
        self._created_at: str | None = None
        self._params: dict[str, Any] = {}

    @property
    def dir(self) -> Path:
        """Path to the job output folder. Only valid inside the context manager."""
        if self._dir is None:
            raise RuntimeError(
                "Job has not been entered yet. Use 'with Job(...) as job:'"
            )
        return self._dir

    def __enter__(self) -> Job:
        project_dir = self._base_dir / self._project
        project_dir.mkdir(parents=True, exist_ok=True)

        if self._resume_job_id is not None:
            self._job_id = self._resume_job_id
            self._dir = project_dir / self._job_id
            job_json = self._dir / "job.json"
            if job_json.exists():
                existing = json.loads(job_json.read_text())
                self._params = existing.get("params") or {}
                self._created_at = existing.get("created_at")
                self._is_resume = True
            else:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._created_at = datetime.now(timezone.utc).isoformat()
        else:
            self._job_id = _next_job_id(project_dir)
            self._dir = project_dir / self._job_id
            self._dir.mkdir()
            self._created_at = datetime.now(timezone.utc).isoformat()

        self._write_json("running")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> Literal[False]:
        completed_at = datetime.now(timezone.utc).isoformat()
        if exc_type is None:
            self._write_json("complete", completed_at=completed_at)
        else:
            self._write_json("failed", completed_at=completed_at, error=str(exc_val))
        return False

    def _write_json(self, status: str, **extra: Any) -> None:
        if self._dir is None:
            raise RuntimeError(
                "Job has not been entered yet. Use 'with Job(...) as job:'"
            )
        if self._job_id is None:
            raise RuntimeError(
                "Job has not been entered yet. Use 'with Job(...) as job:'"
            )
        data: dict[str, Any] = {
            "id": self._job_id,
            "type": self._job_type,
            "project": self._project,
            "status": status,
            "created_at": self._created_at,
            "specter_version": _get_specter_version(),
            "specter_commit": _get_git_commit(),
            "params": self._params,
        }
        data.update(extra)
        (self._dir / "job.json").write_text(json.dumps(data, indent=2))

    def log(self, params: dict[str, Any]) -> None:
        """
        Merge additional key-value pairs into the recorded parameters.

        Use for pre-processing values computed outside the class constructor,
        e.g. dataset paths, number of particles, dose scaling factors.

        Parameters
        ----------
        params : dict
            JSON-serializable key-value pairs to record.
        """
        self._params.update(params)
        self._write_json("running")

    def create(self, cls: type, *args: Any, **kwargs: Any) -> Any:
        """
        Capture all constructor parameters via ``inspect`` and instantiate the class.

        If ``cls.__init__`` accepts a ``run_dir`` parameter, it is automatically
        set to ``self.dir`` — the user does not need to pass it explicitly.

        If ``cls`` defines ``_job_log_exclude`` (a tuple of parameter names), those
        keys are omitted from the recorded params in ``job.json``.

        In resume mode (``job_id`` was passed at construction), the incoming params
        (after exclusion) are validated against the stored params. A ``ValueError``
        is raised if they differ, preventing accidental halfset mismatches.

        Parameters
        ----------
        cls : type
            The SPECTER class to instantiate (e.g. ``Ghostbuster``).
        *args, **kwargs
            Forwarded to ``cls.__init__``.

        Returns
        -------
        instance
            The newly created object.
        """
        import inspect

        sig = inspect.signature(cls.__init__)  # type: ignore[misc]
        bound = sig.bind(None, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self", None)

        if "run_dir" in sig.parameters:
            arguments["run_dir"] = self._dir

        exclude: frozenset[str] = frozenset(getattr(cls, "_job_log_exclude", ()))
        serialized = {
            k: _serialize_value(v)
            for k, v in arguments.items()
            if k not in exclude and k != "run_dir"
        }

        if self._is_resume:
            mismatches = {
                k: (self._params.get(k), serialized.get(k))
                for k in set(self._params) | set(serialized)
                if self._params.get(k) != serialized.get(k)
            }
            if mismatches:
                diff_lines = "\n".join(
                    f"  {k}: stored={a!r}  incoming={b!r}"
                    for k, (a, b) in sorted(mismatches.items())
                )
                raise ValueError(
                    f"Params mismatch for job {self._job_id!r}. "
                    f"Both halfsets must use identical settings:\n{diff_lines}"
                )
        else:
            self._params.update(serialized)
            self._write_json("running")

        return cls(**arguments)

    def save(self, tensor: torch.Tensor, filename: str) -> Path:
        """
        Save a tensor to the job folder.

        Parameters
        ----------
        tensor : torch.Tensor
            Data to save.
        filename : str
            Target filename. ``.mrc``/``.mrcs`` use mrcfile; ``.pt`` uses torch.save.
        """
        import mrcfile  # type: ignore[import-untyped]
        import torch

        path = self.dir / filename
        if filename.endswith((".mrc", ".mrcs")):
            data = tensor.cpu().numpy()
            with mrcfile.new(str(path), overwrite=True) as mrc:
                mrc.set_data(data)
        elif filename.endswith(".pt"):
            torch.save(tensor, path)
        else:
            raise ValueError(
                f"Unsupported format for '{filename}'. Use .mrc, .mrcs, or .pt"
            )
        return path

    def save_figure(self, fig: matplotlib.figure.Figure, filename: str) -> Path:
        """
        Save a matplotlib figure to the job folder.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filename : str
            Target filename, e.g. ``"preview.png"``.
        """
        path = self.dir / filename
        fig.savefig(path)
        return path
