# Job Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `specter.jobs` module with a `Job` context manager that auto-creates unique output folders, captures all constructor parameters via `inspect`, and provides a CLI for browsing and diffing past runs.

**Architecture:** A `Job` context manager wraps any SPECTER class instantiation via `job.create()`, uses `inspect` to bind and serialize all arguments (tensors become shape summaries), and writes a `job.json` alongside outputs in an auto-numbered folder. `JobDatabase` reads those folders for querying. The CLI wraps `JobDatabase` with `rich` tables.

**Tech Stack:** Python `inspect`, `json`, `subprocess` (git commit), `rich` (already a dep), `mrcfile`, `torch`, `argparse`

---

## File Map

| Path | Role |
|---|---|
| `src/specter/jobs/__init__.py` | Exports `Job`, `JobDatabase` |
| `src/specter/jobs/_job.py` | `Job` context manager, `_serialize_value`, helpers |
| `src/specter/jobs/_database.py` | `JobDatabase` — reads all `job.json` files |
| `src/specter/jobs/_cli.py` | `main()` entry point for `specter-jobs` CLI |
| `src/specter/__init__.py` | Add `from .jobs import Job, JobDatabase` |
| `pyproject.toml` | Add `[project.scripts]` entry for `specter-jobs` |
| `tests/test_jobs.py` | All job manager tests |

---

## Task 1: Folder creation and job ID assignment

**Files:**
- Create: `src/specter/jobs/_job.py`
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_jobs.py`:

```python
from __future__ import annotations

import json
import os
import subprocess as proc
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch

from specter.jobs._job import Job, _resolve_base_dir


def test_job_creates_folder(tmp_path: Path) -> None:
    with Job("ghostbuster", project="test-project", base_dir=tmp_path) as job:
        assert job.dir.exists()
        assert job.dir.is_dir()


def test_job_dir_name_is_j001(tmp_path: Path) -> None:
    with Job("ghostbuster", project="test-project", base_dir=tmp_path) as job:
        assert job.dir.name == "J001"


def test_job_id_sequence(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job1:
        first = job1.dir.name
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job2:
        second = job2.dir.name
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job3:
        third = job3.dir.name
    assert first == "J001"
    assert second == "J002"
    assert third == "J003"


def test_job_ids_scoped_per_project(tmp_path: Path) -> None:
    with Job("ghostbuster", project="alpha", base_dir=tmp_path) as job_a:
        pass
    with Job("ghostbuster", project="beta", base_dir=tmp_path) as job_b:
        pass
    assert job_a.dir.name == "J001"
    assert job_b.dir.name == "J001"


def test_resolve_base_dir_from_arg(tmp_path: Path) -> None:
    assert _resolve_base_dir(tmp_path) == tmp_path


def test_resolve_base_dir_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTER_JOBS_DIR", str(tmp_path))
    assert _resolve_base_dir(None) == tmp_path


def test_resolve_base_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECTER_JOBS_DIR", raising=False)
    assert _resolve_base_dir(None) == Path.home() / "specter-data"


def test_job_dir_raises_outside_context(tmp_path: Path) -> None:
    job = Job("ghostbuster", project="p", base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not been entered"):
        _ = job.dir
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /mnt/cbis/home/e0788253/czii/specter && python -m pytest tests/test_jobs.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` — `specter.jobs._job` doesn't exist yet.

- [ ] **Step 3: Create `src/specter/jobs/_job.py` with folder creation**

```python
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any


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
    return f"J{last + 1:03d}"


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
    ) -> None:
        self._job_type = job_type
        self._project = project
        self._base_dir = _resolve_base_dir(base_dir)
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

    def __enter__(self) -> "Job":
        project_dir = self._base_dir / self._project
        project_dir.mkdir(parents=True, exist_ok=True)
        self._job_id = _next_job_id(project_dir)
        self._dir = project_dir / self._job_id
        self._dir.mkdir()
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._write_json("running")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        completed_at = datetime.now(timezone.utc).isoformat()
        if exc_type is None:
            self._write_json("complete", completed_at=completed_at)
        else:
            self._write_json("failed", completed_at=completed_at, error=str(exc_val))
        return False

    def _write_json(self, status: str, **extra: Any) -> None:
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
        assert self._dir is not None
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

        sig = inspect.signature(cls.__init__)
        bound = sig.bind(None, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self", None)

        if "run_dir" in sig.parameters:
            arguments["run_dir"] = self._dir

        serialized = {
            k: _serialize_value(v)
            for k, v in arguments.items()
            if k != "run_dir"
        }
        self._params.update(serialized)
        self._write_json("running")

        return cls(**arguments)

    def save(self, tensor: "torch.Tensor", filename: str) -> Path:
        """
        Save a tensor to the job folder.

        Parameters
        ----------
        tensor : torch.Tensor
            Data to save.
        filename : str
            Target filename. ``.mrc``/``.mrcs`` use mrcfile; ``.pt`` uses torch.save.
        """
        import mrcfile
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

    def save_figure(self, fig: "matplotlib.figure.Figure", filename: str) -> Path:
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
```

- [ ] **Step 4: Run failing tests**

```bash
python -m pytest tests/test_jobs.py::test_job_creates_folder tests/test_jobs.py::test_job_dir_name_is_j001 tests/test_jobs.py::test_job_id_sequence tests/test_jobs.py::test_job_ids_scoped_per_project tests/test_jobs.py::test_resolve_base_dir_from_arg tests/test_jobs.py::test_resolve_base_dir_from_env tests/test_jobs.py::test_resolve_base_dir_default tests/test_jobs.py::test_job_dir_raises_outside_context -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/specter/jobs/_job.py tests/test_jobs.py
git commit -m "feat: add Job context manager with folder creation and ID sequencing"
```

---

## Task 2: job.json writing (status tracking)

**Files:**
- Modify: `tests/test_jobs.py` — add status tests

- [ ] **Step 1: Add tests for job.json contents**

Append to `tests/test_jobs.py`:

```python
import json


def test_job_writes_json_on_enter(tmp_path: Path) -> None:
    with Job("tilt-series", project="p", base_dir=tmp_path) as job:
        data = json.loads((job.dir / "job.json").read_text())
        assert data["id"] == "J001"
        assert data["type"] == "tilt-series"
        assert data["project"] == "p"
        assert data["status"] == "running"
        assert "created_at" in data
        assert "specter_version" in data
        assert "specter_commit" in data
        assert data["params"] == {}


def test_job_status_complete(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        pass
    data = json.loads((job.dir / "job.json").read_text())
    assert data["status"] == "complete"
    assert "completed_at" in data


def test_job_status_failed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="boom"):
        with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
            raise ValueError("boom")
    data = json.loads((job.dir / "job.json").read_text())
    assert data["status"] == "failed"
    assert data["error"] == "boom"
    assert "completed_at" in data
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_jobs.py::test_job_writes_json_on_enter tests/test_jobs.py::test_job_status_complete tests/test_jobs.py::test_job_status_failed -v
```

Expected: All 3 PASS (implementation already handles this from Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/test_jobs.py
git commit -m "test: add job.json status tracking tests"
```

---

## Task 3: `job.create()` — parameter capture via inspect

**Files:**
- Modify: `tests/test_jobs.py` — add `create()` tests

- [ ] **Step 1: Add tests for `job.create()`**

Append to `tests/test_jobs.py`:

```python
import torch


class _DummyClass:
    """Minimal class for testing job.create() parameter capture."""

    def __init__(
        self,
        name: str,
        value: float = 1.0,
        flag: bool = False,
        run_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.flag = flag
        self.run_dir = run_dir


class _NoRunDir:
    def __init__(self, x: int = 42) -> None:
        self.x = x


def test_job_create_returns_instance(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        obj = job.create(_DummyClass, "hello")
    assert isinstance(obj, _DummyClass)


def test_job_create_captures_explicit_args(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_DummyClass, "hello", value=3.14)
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["name"] == "hello"
    assert data["params"]["value"] == 3.14


def test_job_create_captures_defaults(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_DummyClass, "hello")
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["flag"] is False
    assert data["params"]["value"] == 1.0


def test_job_create_injects_run_dir(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        obj = job.create(_DummyClass, "hello")
    assert obj.run_dir == job.dir


def test_job_create_run_dir_not_in_params(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_DummyClass, "hello")
    data = json.loads((job.dir / "job.json").read_text())
    assert "run_dir" not in data["params"]


def test_job_create_no_run_dir_class(tmp_path: Path) -> None:
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        obj = job.create(_NoRunDir, x=7)
    assert obj.x == 7
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["x"] == 7


def test_job_create_tensor_summary(tmp_path: Path) -> None:
    class _WithTensor:
        def __init__(self, volume: torch.Tensor, lr: float = 0.1) -> None:
            self.volume = volume
            self.lr = lr

    vol = torch.zeros(32, 32, 32)
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_WithTensor, vol)
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["volume"] == {
        "__type__": "Tensor",
        "shape": [32, 32, 32],
        "dtype": "float32",
    }


def test_job_create_dict_of_tensors(tmp_path: Path) -> None:
    class _WithCtf:
        def __init__(self, ctf_params: dict) -> None:
            self.ctf_params = ctf_params

    ctf = {"dfu": torch.ones(10), "dfv": torch.ones(10)}
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_WithCtf, ctf)
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["ctf_params"]["dfu"]["__type__"] == "Tensor"
    assert data["params"]["ctf_params"]["dfu"]["shape"] == [10]
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_jobs.py -k "create" -v
```

Expected: All `create` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jobs.py
git commit -m "test: add job.create() parameter capture and run_dir injection tests"
```

---

## Task 4: `job.log()` and `job.save()`

**Files:**
- Modify: `tests/test_jobs.py` — add log and save tests

- [ ] **Step 1: Add tests**

Append to `tests/test_jobs.py`:

```python
def test_job_log_stores_params(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        job.log({"n_particles": 100, "dataset": "empiar-12391"})
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["n_particles"] == 100
    assert data["params"]["dataset"] == "empiar-12391"


def test_job_log_merges(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        job.log({"a": 1})
        job.log({"b": 2})
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["a"] == 1
    assert data["params"]["b"] == 2


def test_job_save_mrc(tmp_path: Path) -> None:
    tensor = torch.ones(8, 8, 8)
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        out = job.save(tensor, "vol.mrc")
    assert out.exists()
    with mrcfile.open(str(out)) as mrc:
        assert mrc.data.shape == (8, 8, 8)
        assert np.allclose(mrc.data, 1.0)


def test_job_save_pt(tmp_path: Path) -> None:
    tensor = torch.arange(12).float()
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        out = job.save(tensor, "data.pt")
    assert out.exists()
    loaded = torch.load(out, weights_only=True)
    assert torch.allclose(loaded, tensor)


def test_job_save_unsupported_format_raises(tmp_path: Path) -> None:
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        with pytest.raises(ValueError, match="Unsupported format"):
            job.save(torch.ones(4), "output.npy")


def test_job_save_figure(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    with Job("ghostbuster", project="p", base_dir=tmp_path) as job:
        out = job.save_figure(fig, "plot.png")
    assert out.exists()
    plt.close(fig)
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_jobs.py -k "log or save" -v
```

Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jobs.py
git commit -m "test: add job.log() and job.save() tests"
```

---

## Task 5: `JobDatabase`

**Files:**
- Create: `src/specter/jobs/_database.py`
- Modify: `tests/test_jobs.py` — add database tests

- [ ] **Step 1: Add tests**

Add this import at the top of `tests/test_jobs.py` (after existing imports):

```python
from specter.jobs._database import JobDatabase
```

Then append these tests:

```python

def _make_job(tmp_path: Path, project: str, job_type: str, params: dict) -> None:
    with Job(job_type, project=project, base_dir=tmp_path) as job:
        job.log(params)


def test_database_list_all(tmp_path: Path) -> None:
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.1})
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.05})
    _make_job(tmp_path, "proj-b", "tilt-series", {"defocus": 1.5})
    db = JobDatabase(base_dir=tmp_path)
    all_jobs = db.list()
    assert len(all_jobs) == 3


def test_database_list_by_project(tmp_path: Path) -> None:
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.1})
    _make_job(tmp_path, "proj-b", "tilt-series", {"defocus": 1.5})
    db = JobDatabase(base_dir=tmp_path)
    assert len(db.list(project="proj-a")) == 1
    assert len(db.list(project="proj-b")) == 1


def test_database_get(tmp_path: Path) -> None:
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.1})
    db = JobDatabase(base_dir=tmp_path)
    entry = db.get("proj-a", "J001")
    assert entry["id"] == "J001"
    assert entry["params"]["lr"] == 0.1


def test_database_get_missing_raises(tmp_path: Path) -> None:
    db = JobDatabase(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        db.get("proj-a", "J999")


def test_database_diff_changed_keys(tmp_path: Path) -> None:
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.1, "symmetry": "I1"})
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.05, "symmetry": "I1"})
    db = JobDatabase(base_dir=tmp_path)
    diff = db.diff("proj-a", "J001", "J002")
    assert "lr" in diff
    assert diff["lr"] == (0.1, 0.05)
    assert "symmetry" not in diff


def test_database_diff_missing_key(tmp_path: Path) -> None:
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.1})
    _make_job(tmp_path, "proj-a", "ghostbuster", {"lr": 0.1, "new_param": "hello"})
    db = JobDatabase(base_dir=tmp_path)
    diff = db.diff("proj-a", "J001", "J002")
    assert diff["new_param"] == (None, "hello")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_jobs.py -k "database" -v 2>&1 | head -20
```

Expected: `ImportError` — `_database` doesn't exist yet.

- [ ] **Step 3: Create `src/specter/jobs/_database.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._job import _resolve_base_dir


class JobDatabase:
    """
    Read-only view over all job folders under a base directory.

    Parameters
    ----------
    base_dir : str or Path, optional
        Root directory to scan. Defaults to ``~/specter-data/`` or
        the ``SPECTER_JOBS_DIR`` environment variable.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = _resolve_base_dir(base_dir)

    def list(self, project: str | None = None) -> list[dict[str, Any]]:
        """
        Return all job records, optionally filtered by project.

        Parameters
        ----------
        project : str, optional
            If given, only return jobs from this project folder.

        Returns
        -------
        list[dict]
            Each element is the parsed contents of a ``job.json`` file,
            sorted by ``created_at`` ascending.
        """
        results = []
        search_root = self._base_dir / project if project else self._base_dir
        for job_json in sorted(search_root.glob("**/job.json")):
            try:
                results.append(json.loads(job_json.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def get(self, project: str, job_id: str) -> dict[str, Any]:
        """
        Return a single job record by project and ID.

        Parameters
        ----------
        project : str
            Project folder name.
        job_id : str
            Job ID, e.g. ``"J001"``.

        Returns
        -------
        dict
            Parsed ``job.json`` contents.

        Raises
        ------
        FileNotFoundError
            If the job folder or ``job.json`` does not exist.
        """
        path = self._base_dir / project / job_id / "job.json"
        if not path.exists():
            raise FileNotFoundError(f"No job found at {path}")
        return json.loads(path.read_text())

    def diff(
        self, project: str, job_id_a: str, job_id_b: str
    ) -> dict[str, tuple[Any, Any]]:
        """
        Compare the ``params`` dicts of two jobs.

        Parameters
        ----------
        project : str
            Project folder name.
        job_id_a, job_id_b : str
            Job IDs to compare.

        Returns
        -------
        dict
            Keys where values differ, mapped to ``(value_in_a, value_in_b)``.
            Keys present in one job but not the other have ``None`` on the
            missing side.
        """
        params_a = self.get(project, job_id_a).get("params", {})
        params_b = self.get(project, job_id_b).get("params", {})
        all_keys = set(params_a) | set(params_b)
        return {
            k: (params_a.get(k), params_b.get(k))
            for k in all_keys
            if params_a.get(k) != params_b.get(k)
        }
```

- [ ] **Step 4: Run database tests**

```bash
python -m pytest tests/test_jobs.py -k "database" -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/specter/jobs/_database.py tests/test_jobs.py
git commit -m "feat: add JobDatabase with list, get, and diff"
```

---

## Task 6: CLI (`specter-jobs`)

**Files:**
- Create: `src/specter/jobs/_cli.py`
- Modify: `tests/test_jobs.py` — add CLI smoke test

- [ ] **Step 1: Add CLI smoke test**

Append to `tests/test_jobs.py`:

```python
def test_cli_list_smoke(tmp_path: Path) -> None:
    _make_job(tmp_path, "my-project", "ghostbuster", {"lr": 0.1, "symmetry": "I1"})
    result = proc.run(
        [sys.executable, "-m", "specter.jobs._cli", "list", "--base-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "my-project" in result.stdout
    assert "J001" in result.stdout


def test_cli_show_smoke(tmp_path: Path) -> None:
    _make_job(tmp_path, "my-project", "ghostbuster", {"lr": 0.1})
    result = proc.run(
        [
            sys.executable, "-m", "specter.jobs._cli",
            "show", "my-project", "J001",
            "--base-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "J001" in result.stdout


def test_cli_diff_smoke(tmp_path: Path) -> None:
    _make_job(tmp_path, "my-project", "ghostbuster", {"lr": 0.1})
    _make_job(tmp_path, "my-project", "ghostbuster", {"lr": 0.05})
    result = proc.run(
        [
            sys.executable, "-m", "specter.jobs._cli",
            "diff", "my-project", "J001", "J002",
            "--base-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "lr" in result.stdout
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_jobs.py -k "cli" -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` for `specter.jobs._cli`.

- [ ] **Step 3: Create `src/specter/jobs/_cli.py`**

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ._database import JobDatabase

console = Console()


def _short_params(params: dict[str, Any], max_items: int = 4) -> str:
    """Return a compact one-line summary of scalar params only."""
    scalars = {
        k: v
        for k, v in params.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    items = list(scalars.items())[:max_items]
    return "  ".join(f"{k}={v}" for k, v in items)


def cmd_list(args: argparse.Namespace) -> None:
    db = JobDatabase(base_dir=args.base_dir)
    jobs = db.list(project=args.project or None)
    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Project")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Params")
    for entry in jobs:
        created = entry.get("created_at", "")[:16].replace("T", " ")
        status = entry.get("status", "")
        color = {"complete": "green", "running": "yellow", "failed": "red"}.get(
            status, "white"
        )
        table.add_row(
            entry.get("project", ""),
            entry.get("id", ""),
            entry.get("type", ""),
            f"[{color}]{status}[/{color}]",
            created,
            _short_params(entry.get("params", {})),
        )
    console.print(table)


def cmd_show(args: argparse.Namespace) -> None:
    db = JobDatabase(base_dir=args.base_dir)
    try:
        entry = db.get(args.project, args.job_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print_json(json.dumps(entry, indent=2))


def cmd_diff(args: argparse.Namespace) -> None:
    db = JobDatabase(base_dir=args.base_dir)
    try:
        diff = db.diff(args.project, args.job_id_a, args.job_id_b)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if not diff:
        console.print("[green]Jobs are identical.[/green]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column(args.job_id_a)
    table.add_column(args.job_id_b)
    for key, (val_a, val_b) in sorted(diff.items()):
        table.add_row(key, str(val_a), str(val_b))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="specter-jobs", description="SPECTER job manager CLI")
    parser.add_argument("--base-dir", default=None, help="Override SPECTER_JOBS_DIR")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List all jobs")
    p_list.add_argument("--project", default=None, help="Filter by project name")

    p_show = sub.add_parser("show", help="Show full details for a job")
    p_show.add_argument("project")
    p_show.add_argument("job_id")

    p_diff = sub.add_parser("diff", help="Diff params of two jobs")
    p_diff.add_argument("project")
    p_diff.add_argument("job_id_a")
    p_diff.add_argument("job_id_b")

    args = parser.parse_args()
    {"list": cmd_list, "show": cmd_show, "diff": cmd_diff}[args.command](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

```bash
python -m pytest tests/test_jobs.py -k "cli" -v
```

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/specter/jobs/_cli.py tests/test_jobs.py
git commit -m "feat: add specter-jobs CLI with list, show, diff subcommands"
```

---

## Task 7: Wire up module exports and `pyproject.toml`

**Files:**
- Create: `src/specter/jobs/__init__.py`
- Modify: `src/specter/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create `src/specter/jobs/__init__.py`**

```python
from ._job import Job
from ._database import JobDatabase

__all__ = ["Job", "JobDatabase"]
```

- [ ] **Step 2: Add exports to `src/specter/__init__.py`**

Append to the end of the existing file:

```python
from .jobs import Job, JobDatabase
```

- [ ] **Step 3: Add console script to `pyproject.toml`**

Add after the `[tool.setuptools.package-data]` block:

```toml
[project.scripts]
specter-jobs = "specter.jobs._cli:main"
```

- [ ] **Step 4: Reinstall the package so the entry point is registered**

```bash
uv sync
```

- [ ] **Step 5: Verify the CLI entry point works**

```bash
specter-jobs --help
```

Expected output:
```
usage: specter-jobs [-h] [--base-dir BASE_DIR] {list,show,diff} ...
```

- [ ] **Step 6: Verify top-level imports work**

```bash
python -c "from specter import Job, JobDatabase; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/test_jobs.py -v
```

Expected: All tests PASS.

- [ ] **Step 8: Run linter and formatter**

```bash
ruff check src/specter/jobs/ tests/test_jobs.py
ruff format src/specter/jobs/ tests/test_jobs.py
```

Fix any issues reported, then re-run to confirm clean.

- [ ] **Step 9: Commit**

```bash
git add src/specter/jobs/__init__.py src/specter/__init__.py pyproject.toml
git commit -m "feat: wire up specter.jobs exports and specter-jobs CLI entry point"
```

---

## Task 8: Full test suite and final verification

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All existing tests still PASS. No regressions.

- [ ] **Step 2: Run mypy**

```bash
mypy src/specter/jobs/
```

Fix any type errors. Common ones to expect:
- `_write_json` called before `_job_id` / `_created_at` assigned — guarded by `assert self._dir is not None`, but mypy may flag `_job_id` being `str | None`. Add `assert self._job_id is not None` before the dict in `_write_json`.

- [ ] **Step 3: Final commit**

```bash
git add -u
git commit -m "chore: fix mypy issues in specter.jobs"
```
