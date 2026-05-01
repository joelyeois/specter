from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch

from specter.jobs._job import Job, _resolve_base_dir
from specter.jobs._database import JobDatabase


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


def test_resolve_base_dir_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPECTER_JOBS_DIR", str(tmp_path))
    assert _resolve_base_dir(None) == tmp_path


def test_resolve_base_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECTER_JOBS_DIR", raising=False)
    assert _resolve_base_dir(None) == Path.home() / "specter-data"


def test_job_dir_raises_outside_context(tmp_path: Path) -> None:
    job = Job("ghostbuster", project="p", base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not been entered"):
        _ = job.dir


# ---------------------------------------------------------------------------
# Task 2: job.json status tracking
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 3: job.create() parameter capture and run_dir injection
# ---------------------------------------------------------------------------


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
        def __init__(self, ctf_params: dict) -> None:  # type: ignore[type-arg]
            self.ctf_params = ctf_params

    ctf = {"dfu": torch.ones(10), "dfv": torch.ones(10)}
    with Job("dummy", project="p", base_dir=tmp_path) as job:
        job.create(_WithCtf, ctf)
    data = json.loads((job.dir / "job.json").read_text())
    assert data["params"]["ctf_params"]["dfu"]["__type__"] == "Tensor"
    assert data["params"]["ctf_params"]["dfu"]["shape"] == [10]


# ---------------------------------------------------------------------------
# Task 4: job.log() and job.save()
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 5: JobDatabase — list, get, diff
# ---------------------------------------------------------------------------


def _make_job(tmp_path: Path, project: str, job_type: str, params: dict) -> None:  # type: ignore[type-arg]
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
