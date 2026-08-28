"""Tests for `specter.pipelines._common`'s job-tracking helpers --
_tracked_output_dir (shared by run_particle_stack/run_micrograph/
run_tilt_series/run_build_tomogram) and _reserve_next_job_id (used to
cascade tracking through run_tilt_series's tomogram_config chaining). A
minimal stand-in config (not ParticleStackConfig) keeps these fast and
focused on the wrapper's own branching, independent of any real pipeline's
setup cost.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from specter.jobs import Job
from specter.pipelines._common import (
    _reserve_next_job_id,
    _tracked_output_dir,
    resolve_output_dir,
)


@dataclasses.dataclass
class _Config:
    output_dir: str | None = None
    project: str | None = None
    job_id: str | None = None
    dummy_field: int = 1


def test_resolve_output_dir_set_is_used_verbatim_either_way() -> None:
    """One field, two readings: the leaf untracked, the job-tree root tracked.

    `resolve_output_dir` returns the path itself in both cases -- it is
    `_tracked_output_dir`/`Job` that appends `[project/]<job_type>/J00N`
    on top when tracking is on.
    """
    assert resolve_output_dir(_Config(output_dir="chosen"), "particles") == "chosen"
    tracked = _Config(output_dir="chosen", project="p")
    assert resolve_output_dir(tracked, "particles") == "chosen"


def test_resolve_output_dir_unset_defaults_differ_by_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset, the two cases must NOT share a default.

    The tracked layout supplies its own `<job_type>` segment, so defaulting
    both to `particles/` would yield `particles/particles/J001`.
    """
    (tmp_path / ".specter").touch()
    monkeypatch.chdir(tmp_path)

    assert resolve_output_dir(_Config(), "particles") == "particles"
    assert resolve_output_dir(_Config(project="p"), "particles") == str(tmp_path)


def test_untracked_yields_output_dir_unchanged(tmp_path: Path) -> None:
    config = _Config(output_dir=str(tmp_path / "flat"))
    with _tracked_output_dir(config, "particles") as output_dir:
        assert output_dir == config.output_dir
    # No Job side effects: nothing created besides what the caller itself makes.
    assert not (tmp_path / "particles").exists()


def test_tracked_by_project_opens_a_job(tmp_path: Path) -> None:
    config = _Config(project="apoferritin", output_dir=str(tmp_path))
    with _tracked_output_dir(config, "particles") as output_dir:
        assert output_dir == str(tmp_path / "apoferritin" / "particles" / "J001")
        assert Path(output_dir).is_dir()
    job = json.loads((Path(output_dir) / "job.json").read_text())
    assert job["status"] == "complete"
    assert job["params"]["dummy_field"] == 1


def test_tracked_by_job_id_alone_opens_a_job(tmp_path: Path) -> None:
    """project isn't required for tracking to trigger -- job_id alone does too."""
    config = _Config(job_id="J005", output_dir=str(tmp_path))
    with _tracked_output_dir(config, "particles") as output_dir:
        assert output_dir == str(tmp_path / "particles" / "J005")
    assert (Path(output_dir) / "job.json").exists()


def test_tracked_output_dir_defaults_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".specter").touch()
    monkeypatch.chdir(tmp_path)
    config = _Config(project="p")  # output_dir left unset
    with _tracked_output_dir(config, "particles") as output_dir:
        assert output_dir == str(tmp_path / "p" / "particles" / "J001")


def test_not_is_main_computes_same_path_without_touching_filesystem(
    tmp_path: Path,
) -> None:
    """A non-main DDP rank must agree on the exact path is_main resolves,
    without creating anything itself -- job_id is required here in real
    pipelines (see validate_config) precisely so this is a pure string
    join, safe to compute redundantly and independently."""
    config = _Config(project="apoferritin", job_id="J003", output_dir=str(tmp_path))
    with _tracked_output_dir(config, "particles", is_main=True) as main_dir:
        pass
    with _tracked_output_dir(config, "particles", is_main=False) as worker_dir:
        assert worker_dir == main_dir

    # The worker's entry didn't create anything new on its own -- only
    # is_main's Job() call above did.
    assert set(os.listdir(tmp_path / "apoferritin" / "particles")) == {"J003"}


def test_not_is_main_without_project_computes_job_type_dir_directly(
    tmp_path: Path,
) -> None:
    config = _Config(job_id="J007", output_dir=str(tmp_path))
    with _tracked_output_dir(config, "particles", is_main=False) as output_dir:
        assert output_dir == str(tmp_path / "particles" / "J007")
    # is_main=False never touches the filesystem at all.
    assert not (tmp_path / "particles").exists()


def test_tracked_records_failure_status(tmp_path: Path) -> None:
    config = _Config(project="p", output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="boom"):
        with _tracked_output_dir(config, "particles") as output_dir:
            raise ValueError("boom")
    job = json.loads((Path(output_dir) / "job.json").read_text())
    assert job["status"] == "failed"
    assert job["error"] == "boom"


def test_reserve_next_job_id_matches_what_job_itself_would_assign(
    tmp_path: Path,
) -> None:
    assert _reserve_next_job_id("apoferritin", str(tmp_path)) == "J001"

    with Job("tomograms", project="apoferritin", base_dir=tmp_path):
        pass
    assert _reserve_next_job_id("apoferritin", str(tmp_path)) == "J002"


def test_reserve_next_job_id_no_project(tmp_path: Path) -> None:
    with Job("particles", project=None, base_dir=tmp_path):
        pass
    assert _reserve_next_job_id(None, str(tmp_path)) == "J002"


@pytest.mark.parametrize("cuda_present", [True, False])
def test_bare_cuda_falls_back_to_cpu_only_when_there_is_none(
    monkeypatch, cuda_present
) -> None:
    """
    `device="cuda"` runs on the CPU, with a warning, on a machine without one.

    Every simulate/build config defaults to `"cuda"`, which is what a user with
    a GPU wants. Taken literally without one, torch raises `RuntimeError: No
    CUDA GPUs are available` from inside the forward model -- a traceback that
    names neither the setting nor the fix. A default that cannot run everywhere
    is not a default.
    """
    import torch

    from specter.pipelines._common import resolve_available_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_present)

    if cuda_present:
        assert resolve_available_device("cuda") == "cuda"
    else:
        with pytest.warns(UserWarning, match="no CUDA device is available"):
            assert resolve_available_device("cuda") == "cpu"


@pytest.mark.parametrize("device", ["cuda:0", "cuda:2", "0,1", "0"])
def test_an_explicit_device_index_is_never_silently_moved(monkeypatch, device) -> None:
    """
    Naming particular hardware fails rather than running somewhere else.

    `"cuda"` means "the GPU, if there is one" and may degrade. `"cuda:2"` or
    `"0,1"` names specific devices, so answering with the CPU would be
    answering a different question -- those still raise, and should.
    """
    import torch
    import warnings

    from specter.pipelines._common import resolve_available_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here is a failure
        assert resolve_available_device(device) == device


def test_cpu_is_left_alone_with_no_warning(monkeypatch) -> None:
    """An explicit CPU request must not warn, GPU present or not."""
    import torch
    import warnings

    from specter.pipelines._common import resolve_available_device

    for present in (True, False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: present)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_available_device("cpu") == "cpu"
