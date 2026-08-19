"""Tests for `specter reconstruct particle` and the pipeline behind it.

The end-to-end tests fake the CryoSPARC `.cs` reader (the same `_FakeDataset`
trick as `test_cryosparc.py`) so a real reconstruction runs on CPU against a
tiny synthetic stack. That is what makes them cheap enough for CI while still
exercising the whole path: config -> `Ghostbuster` -> `Reconstructor` -> files
on disk.
"""

from __future__ import annotations

import inspect
import json
import subprocess as proc
import sys
from dataclasses import fields
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from specter.config import ReconstructionConfig
from specter.ghostbuster import Ghostbuster
from specter.io import _cryosparc
from specter.pipelines._reconstruct import (
    _ghostbuster_kwargs,
    _reconstruct_device,
    run_reconstruction,
)

_N_PARTICLES = 4
_BOX = 16


class _FakeDataset(dict):
    """Minimal stand-in for `cryosparc_tools`' Dataset, keyed as a real .cs file."""

    @classmethod
    def load(cls, csfile_path: str) -> "_FakeDataset":
        n = _N_PARTICLES
        dtype = np.float32
        rng = np.random.default_rng(0)
        return cls(
            {
                "alignments3D/shift": rng.normal(size=(n, 2)).astype(dtype),
                "alignments3D/psize_A": np.full(n, 1.5, dtype=dtype),
                "ctf/cs_mm": np.full(n, 2.7, dtype=dtype),
                "ctf/df_angle_rad": rng.normal(size=n).astype(dtype),
                "ctf/df1_A": (rng.normal(size=n) + 10000).astype(dtype),
                "ctf/df2_A": (rng.normal(size=n) + 10000).astype(dtype),
                "ctf/amp_contrast": np.full(n, 0.1, dtype=dtype),
                "ctf/accel_kv": np.full(n, 300.0, dtype=dtype),
                "alignments3D/pose": rng.normal(size=(n, 3)).astype(dtype),
                "alignments3D/split": np.array([0, 1, 0, 1]),
                "ctf/tilt_A": np.zeros((n, 2), dtype=dtype),
                "ctf/phase_shift_rad": np.zeros(n, dtype=dtype),
                "ctf/shift_A": np.zeros((n, 2), dtype=dtype),
                "ctf/trefoil_A": np.zeros((n, 2), dtype=dtype),
                "ctf/tetra_A": np.zeros((n, 4), dtype=dtype),
                "alignments3D/alpha": np.ones(n, dtype=dtype),
                "ctf/anisomag": np.zeros((n, 4), dtype=dtype),
            }
        )


@pytest.fixture
def particle_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake .cs file (contents never read) and a real, tiny particle stack."""
    monkeypatch.setattr(_cryosparc, "Dataset", _FakeDataset)

    cs_file = tmp_path / "particles.cs"
    cs_file.write_bytes(b"")

    mrc_file = tmp_path / "particles.mrc"
    rng = np.random.default_rng(0)
    images = rng.normal(size=(_N_PARTICLES, _BOX, _BOX)).astype(np.float32)
    with mrcfile.new(str(mrc_file), overwrite=True) as mrc:
        mrc.set_data(images)

    return cs_file, mrc_file


def _config(cs_file: Path, mrc_file: Path, output_dir: Path) -> ReconstructionConfig:
    """A config small and cheap enough to fit on CPU inside a test."""
    return ReconstructionConfig(
        cs_file=str(cs_file),
        mrc_file=str(mrc_file),
        dose_per_angstrom=40.0,
        test_run=True,
        bin_factor=4,
        batchsize=2,
        device="cpu",
        num_workers=0,
        output_dir=str(output_dir),
    )


# ---------------------------------------------------------------------------
# Config -> Ghostbuster wiring
# ---------------------------------------------------------------------------


def test_ghostbuster_kwargs_are_all_real_constructor_arguments() -> None:
    """Every field the pipeline forwards names an actual `Ghostbuster` argument.

    `_ghostbuster_kwargs` forwards by subtraction (every field except a
    denylist), so a new config field that is *not* a Ghostbuster argument
    would otherwise surface as a TypeError only once someone runs the
    command with real data.
    """
    config = ReconstructionConfig(cs_file="a.cs", mrc_file="b.mrc", dose_per_angstrom=1)
    accepted = set(inspect.signature(Ghostbuster.__init__).parameters)
    forwarded = set(_ghostbuster_kwargs(config))

    assert forwarded <= accepted, sorted(forwarded - accepted)
    assert "run_dir" not in forwarded


def test_every_config_field_is_forwarded_or_deliberately_held_back() -> None:
    """No config field silently does nothing.

    A field is either a `Ghostbuster` argument or one of the run-layout
    fields the pipeline consumes itself. There is no third category, and a
    field that fell into one would be accepted by the CLI and then ignored.
    """
    from specter.pipelines._reconstruct import _NON_GHOSTBUSTER_FIELDS

    config = ReconstructionConfig(cs_file="a.cs", mrc_file="b.mrc", dose_per_angstrom=1)
    all_fields = {f.name for f in fields(config)}
    assert all_fields == set(_ghostbuster_kwargs(config)) | _NON_GHOSTBUSTER_FIELDS


@pytest.mark.parametrize(
    ("device_str", "expected"),
    [
        ("cpu", "cpu"),
        ("cuda", 0),
        ("cuda:1", 1),
        ("2", 2),
        ("0,1", [0, 1]),
    ],
)
def test_reconstruct_device(device_str: str, expected: int | list[int] | str) -> None:
    assert _reconstruct_device(device_str) == expected


def test_job_id_without_project_is_rejected(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    cs_file, mrc_file = particle_data
    config = _config(cs_file, mrc_file, tmp_path / "out")
    config.job_id = "J001"
    with pytest.raises(ValueError, match="job_id.*needs project"):
        run_reconstruction(config)


def test_missing_input_file_fails_before_any_work(tmp_path: Path) -> None:
    config = ReconstructionConfig(
        cs_file=str(tmp_path / "nope.cs"),
        mrc_file=str(tmp_path / "nope.mrc"),
        dose_per_angstrom=40.0,
    )
    with pytest.raises(ValueError, match="cs_file"):
        run_reconstruction(config)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_reconstruction_writes_a_run_directory(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    cs_file, mrc_file = particle_data
    output_dir = tmp_path / "out"
    config = _config(cs_file, mrc_file, output_dir)
    config.run_name = "my_run"

    run_reconstruction(config)

    run_dir = output_dir / "my_run"
    assert (run_dir / "volume.mrc").exists()
    assert (run_dir / "params.json").exists()

    recorded = json.loads((run_dir / "reconstruct_config.json").read_text())
    assert recorded["dose_per_angstrom"] == 40.0
    assert recorded["device"] == "cpu"
    assert recorded["test_run"] is True


def test_run_reconstruction_halfsets_share_one_job(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """The A/B workflow: two runs into one pinned job, side by side.

    This is what the jobs branch exists for -- `return_class` is excluded
    from the job parameter log, so the second run resumes into the same
    directory instead of failing the identical-settings check.
    """
    cs_file, mrc_file = particle_data
    output_dir = tmp_path / "out"

    for return_class in ("0", "1"):
        config = _config(cs_file, mrc_file, output_dir)
        config.project = "test-project"
        config.job_id = "J001"
        config.return_class = return_class  # type: ignore[assignment]
        run_reconstruction(config)

    job_dir = output_dir / "test-project" / "J001"
    assert (job_dir / "volume_A.mrc").exists()
    assert (job_dir / "volume_B.mrc").exists()
    assert (job_dir / "reconstruct_config_A.json").exists()
    assert (job_dir / "reconstruct_config_B.json").exists()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"
    assert job["params"]["dose_per_angstrom"] == 40.0


# ---------------------------------------------------------------------------
# CLI layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["reconstruct", "ghostbuster"])
def test_cli_reconstruct_particle_help_smoke(group: str) -> None:
    """Both names reach the same command, and each reports itself correctly.

    `specter ghostbuster` is an alias for `specter reconstruct`, built as its
    own group object -- registering one instance under two keys would have
    the usage line name whichever was registered first, whatever the user
    typed.
    """
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", group, "particle", "--help"],
        capture_output=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "--cs_file" in result.stdout
    assert "--test_run" in result.stdout
    assert "--project" in result.stdout
    assert f"specter {group} particle" in result.stdout


def test_cli_reconstruct_particle_reports_a_missing_stack(tmp_path: Path) -> None:
    """A bad path fails fast, naming the field, rather than deep inside a load."""
    config_path = tmp_path / "reconstruct.toml"
    config_path.write_text(
        f"""
[data]
cs_file = "{tmp_path / "nope.cs"}"
mrc_file = "{tmp_path / "nope.mrc"}"
dose_per_angstrom = 40.0
"""
    )
    result = proc.run(
        [
            sys.executable,
            "-m",
            "specter.cli._cli",
            "reconstruct",
            "particle",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert result.returncode != 0
    assert "cs_file" in result.stderr
    assert "no such file" in result.stderr
