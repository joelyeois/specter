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

from specter.config import (
    ReconstructionConfig,
    cryosparc_ref_for_halfset,
    validate_config,
)
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


@pytest.fixture
def real_particle_data(tmp_path: Path) -> tuple[Path, Path]:
    """A genuine .cs file on disk (not monkeypatched) and a tiny particle stack.

    `multiprocessing`'s ``spawn`` context starts a fresh interpreter that
    doesn't inherit `particle_data`'s monkeypatch on `_cryosparc.Dataset` --
    gold-standard mode spawns exactly that kind of subprocess per halfset, so
    it needs a real, loadable file instead.
    """
    from cryosparc.dataset import Dataset

    n = _N_PARTICLES
    dtype = np.float32
    rng = np.random.default_rng(0)
    dataset = Dataset(
        allocate={
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
    cs_file = tmp_path / "particles.cs"
    dataset.save(str(cs_file))

    mrc_file = tmp_path / "particles.mrc"
    images = rng.normal(size=(n, _BOX, _BOX)).astype(np.float32)
    with mrcfile.new(str(mrc_file), overwrite=True) as mrc:
        mrc.set_data(images)

    return cs_file, mrc_file


def _config(cs_file: Path, mrc_file: Path, job_base_dir: Path) -> ReconstructionConfig:
    """A config small and cheap enough to fit on CPU inside a test.

    ``job_base_dir`` keeps every run's job.json under tmp_path -- every run
    is tracked now, there's no untracked output_dir to scope a test to
    instead.
    """
    return ReconstructionConfig(
        cs_file=str(cs_file),
        mrc_file=str(mrc_file),
        dose_per_angstrom=40.0,
        test_run=True,
        bin_factor=4,
        batchsize=2,
        device="cpu",
        num_workers=0,
        job_base_dir=str(job_base_dir),
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


# ---------------------------------------------------------------------------
# Per-halfset CryoSPARC reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cryosparc_ref", "halfset", "expected"),
    [
        (None, "A", None),
        (None, "gold", None),
        # A single path is shared by whichever halfset runs.
        ("a.mrc", "A", "a.mrc"),
        ("a.mrc", "B", "a.mrc"),
        ("a.mrc", "all", "a.mrc"),
        # A pair gives each halfset its own.
        ("a.mrc,b.mrc", "A", "a.mrc"),
        ("a.mrc,b.mrc", "B", "b.mrc"),
        (" a.mrc , b.mrc ", "B", "b.mrc"),
    ],
)
def test_cryosparc_ref_for_halfset(
    cryosparc_ref: str | None, halfset: str, expected: str | None
) -> None:
    assert cryosparc_ref_for_halfset(cryosparc_ref, halfset) == expected


def test_gold_halfset_b_gets_its_own_cryosparc_reference() -> None:
    """Halfset B is compared against CryoSPARC's half-map B, not half-map A.

    `halfset="gold"` reconstructs both halves from one config, so without the
    pair form both would be plotted against whichever single reference was
    given -- silently mislabelling half B's FSC overlay as "CryoSPARC" while
    showing CryoSPARC's *other* half. The gold workers reach `Ghostbuster`
    through `_ghostbuster_kwargs` with `halfset` already narrowed to A or B,
    which is where the pair is resolved.
    """
    config = ReconstructionConfig(
        cs_file="a.cs",
        mrc_file="b.mrc",
        dose_per_angstrom=1,
        cryosparc_ref="half_A.mrc,half_B.mrc",
    )

    config.halfset = "A"
    assert _ghostbuster_kwargs(config)["cryosparc_ref"] == "half_A.mrc"

    config.halfset = "B"
    assert _ghostbuster_kwargs(config)["cryosparc_ref"] == "half_B.mrc"


def test_cryosparc_ref_pair_validates_both_halves(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Both halves of a pair are checked for existence, not just the first."""
    cs_file, mrc_file = particle_data
    config = _config(cs_file, mrc_file, tmp_path / "out")
    real = tmp_path / "ref_A.mrc"
    real.write_bytes(b"")

    config.cryosparc_ref = f"{real},{tmp_path / 'nope_B.mrc'}"
    with pytest.raises(ValueError, match="nope_B.mrc.*no such file"):
        validate_config(config)

    config.cryosparc_ref = f"{real},{real}"
    validate_config(config)


def test_cryosparc_ref_rejects_malformed_and_meaningless_pairs(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Three paths is never meaningful, and a pair is meaningless for "all"."""
    cs_file, mrc_file = particle_data
    config = _config(cs_file, mrc_file, tmp_path / "out")
    real = tmp_path / "ref_A.mrc"
    real.write_bytes(b"")

    config.cryosparc_ref = f"{real},{real},{real}"
    with pytest.raises(ValueError, match="pair"):
        validate_config(config)

    config.cryosparc_ref = f"{real},"
    with pytest.raises(ValueError, match="pair"):
        validate_config(config)

    # "all" is one volume from every particle, so there is no second half to
    # reference -- a pair there means the user expected a split that isn't
    # happening.
    config.cryosparc_ref = f"{real},{real}"
    config.halfset = "all"
    with pytest.raises(ValueError, match='halfset="all"'):
        validate_config(config)


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


def test_job_id_without_project_pins_under_implicit_default_project(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """job_id without project is valid: it pins a number directly under
    job_base_dir's implicit default project, not a named one -- omitting
    `project` never meant "untracked"."""
    cs_file, mrc_file = particle_data
    job_base_dir = tmp_path / "out"
    config = _config(cs_file, mrc_file, job_base_dir)
    config.job_id = "J005"
    config.halfset = "all"

    run_reconstruction(config)

    job_dir = job_base_dir / "reconstructions" / "J005"
    assert (job_dir / "job.json").exists()


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


def test_run_reconstruction_writes_a_numbered_job_directory(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Leaving project unset doesn't mean untracked: it's still numbered,
    just under job_base_dir's implicit default project (no project-name
    segment) instead of a named one."""
    cs_file, mrc_file = particle_data
    job_base_dir = tmp_path / "out"
    config = _config(cs_file, mrc_file, job_base_dir)
    config.halfset = "all"  # single-run path; gold-standard is covered separately

    run_reconstruction(config)

    job_dir = job_base_dir / "reconstructions" / "J001"
    assert (job_dir / "volume.mrc").exists()
    assert (job_dir / "params.json").exists()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["project"] is None
    assert job["params"]["dose_per_angstrom"] == 40.0
    assert job["params"]["device"] == "cpu"
    assert job["params"]["test_run"] is True


def test_run_reconstruction_halfsets_share_one_job(
    particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """The A/B workflow: two runs into one pinned job, side by side.

    This is what the jobs branch exists for -- `halfset` is excluded
    from the job parameter log, so the second run resumes into the same
    directory instead of failing the identical-settings check.
    """
    cs_file, mrc_file = particle_data
    job_base_dir = tmp_path / "out"

    for halfset in ("A", "B"):
        config = _config(cs_file, mrc_file, job_base_dir)
        config.project = "test-project"
        config.job_id = "J001"
        config.halfset = halfset  # type: ignore[assignment]
        run_reconstruction(config)

    job_dir = job_base_dir / "test-project" / "reconstructions" / "J001"
    assert (job_dir / "volume_A.mrc").exists()
    assert (job_dir / "volume_B.mrc").exists()
    # job.json already has the full config; a tracked run no longer writes
    # a redundant reconstruct_config_{A,B}.json copy of the same thing.
    assert not (job_dir / "reconstruct_config_A.json").exists()
    assert not (job_dir / "reconstruct_config_B.json").exists()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"
    assert job["params"]["dose_per_angstrom"] == 40.0
    # Fields job.create's Ghostbuster-argument introspection can't see on its
    # own (test_run isn't a Ghostbuster constructor arg) still land in
    # job.json, logged separately by run_reconstruction.
    assert job["params"]["test_run"] is True
    assert job["params"]["device"] == "cpu"


def test_run_reconstruction_gold_standard_default(
    real_particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Leaving halfset unset defaults to gold-standard: both halves, then FSC.

    Needs `real_particle_data`, not the monkeypatched `particle_data` --
    gold-standard mode spawns a real subprocess per halfset, which doesn't
    inherit a pytest monkeypatch. Leaving `project` unset too doesn't mean
    untracked -- there's no such mode -- just no project-name segment.
    """
    cs_file, mrc_file = real_particle_data
    job_base_dir = tmp_path / "out"
    config = _config(cs_file, mrc_file, job_base_dir)
    assert config.halfset == "gold"  # exactly what's under test: no override

    run_reconstruction(config)

    job_dir = job_base_dir / "reconstructions" / "J001"
    assert (job_dir / "volume_A.mrc").exists()
    assert (job_dir / "volume_B.mrc").exists()
    assert (job_dir / "fsc_gold_standard.png").exists()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["project"] is None
    assert "resolution_gold_standard" in job["params"]


def test_run_reconstruction_gold_standard_tracked(
    real_particle_data: tuple[Path, Path], tmp_path: Path
) -> None:
    """Gold-standard with --project logs the resolution into job.json."""
    cs_file, mrc_file = real_particle_data
    job_base_dir = tmp_path / "out"
    config = _config(cs_file, mrc_file, job_base_dir)
    config.project = "test-project"

    run_reconstruction(config)

    job_dir = job_base_dir / "test-project" / "reconstructions" / "J001"
    assert (job_dir / "volume_A.mrc").exists()
    assert (job_dir / "volume_B.mrc").exists()
    assert (job_dir / "fsc_gold_standard.png").exists()
    # job.json already has the full config; no redundant sidecar copy.
    assert not (job_dir / "reconstruct_config_A.json").exists()
    assert not (job_dir / "reconstruct_config_B.json").exists()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"
    assert "resolution_gold_standard" in job["params"]


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
