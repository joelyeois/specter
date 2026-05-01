from __future__ import annotations

from pathlib import Path

import pytest

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
