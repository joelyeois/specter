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


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    env = os.environ.get("SPECTER_JOBS_DIR")
    if env:
        return Path(env)
    return Path.home() / "specter-data"


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
            if not self._dir.exists():
                raise FileNotFoundError(
                    f"Job {self._resume_job_id!r} not found in project "
                    f"{self._project!r} (looked in {self._dir})"
                )
            existing = json.loads((self._dir / "job.json").read_text())
            self._params = existing.get("params", {})
            self._created_at = existing.get("created_at")
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

        serialized = {
            k: _serialize_value(v) for k, v in arguments.items() if k != "run_dir"
        }
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
