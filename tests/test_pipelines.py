from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
import torch
from cryosparc.dataset import Dataset

from specter.config import (
    MicrographConfig,
    ParticleStackConfig,
    TiltSeriesConfig,
    TomogramConfig,
)
from specter.pipelines import (
    run_build_tomogram,
    run_micrograph,
    run_particle_stack,
    run_tilt_series,
)
from specter.pipelines._tomogram import tomogram_output_path

# Every field _load_csfile_parameters (specter.io._cryosparc) reads off a
# CryoSPARC passthrough .cs -- fabricated here so the test needs no real
# external dataset. Values are deliberately distinct from
# ParticleStackConfig's defaults (voltage=300.0, alpha=0.1, cs=2.0) so a
# passing assertion actually proves they came from the .cs file.
_VOLTAGE_KV = 200.0
_PIXEL_SIZE_A = 1.35
_CS_MM = 2.7
_ALPHA = 0.12
_DFU_A = 9000.0
_DFV_A = 8700.0


def _write_minimal_csfile(path: Path, n: int) -> None:
    Dataset(
        {
            "uid": np.arange(n, dtype=np.uint64),
            "alignments3D/shift": np.zeros((n, 2), dtype=np.float32),
            "alignments3D/psize_A": np.full(n, _PIXEL_SIZE_A, dtype=np.float32),
            "alignments3D/pose": np.zeros((n, 3), dtype=np.float32),
            "alignments3D/split": np.zeros(n, dtype=np.uint32),
            "alignments3D/alpha": np.ones(n, dtype=np.float32),
            "ctf/cs_mm": np.full(n, _CS_MM, dtype=np.float32),
            "ctf/df_angle_rad": np.zeros(n, dtype=np.float32),
            "ctf/df1_A": np.full(n, _DFU_A, dtype=np.float32),
            "ctf/df2_A": np.full(n, _DFV_A, dtype=np.float32),
            "ctf/amp_contrast": np.full(n, _ALPHA, dtype=np.float32),
            "ctf/accel_kv": np.full(n, _VOLTAGE_KV, dtype=np.float32),
            "ctf/tilt_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/phase_shift_rad": np.zeros(n, dtype=np.float32),
            "ctf/shift_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/trefoil_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/tetra_A": np.zeros((n, 4), dtype=np.float32),
            "ctf/anisomag": np.zeros((n, 4), dtype=np.float32),
        }
    ).save(str(path))


def test_run_particle_stack_from_csfile(tmp_path: Path) -> None:
    """cs_path drives pixel_size/voltage/alpha/poses/CTF from a CryoSPARC
    passthrough .cs file instead of synthetic sampling -- see
    src/specter/pipelines/_particles.py. For a real-data demonstration of
    this same code path against EMPIAR-11377, see
    docs/user-guide/particle-stack.md and
    docs-figures/particle_stack_empiar_11377.py.
    """
    cs_path = tmp_path / "minimal.cs"
    _write_minimal_csfile(cs_path, n=5)

    config = ParticleStackConfig(
        pdb_source="6bdf",
        n_pixels=64,
        cs_path=str(cs_path),
        n_particles=5,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        batchsize=5,
        output_dir=str(tmp_path),
        filename="cs_particles",
    )
    run_particle_stack(config)

    with mrcfile.open(tmp_path / "cs_particles.mrcs") as mrc:
        assert mrc.data.shape == (5, 64, 64)

    df = starfile.read(tmp_path / "cs_particles.star")
    assert len(df) == 5
    # These come from the .cs file, not ParticleStackConfig's (unused) defaults.
    assert df["rlnVoltage"].iloc[0] == _VOLTAGE_KV
    assert df["rlnSphericalAberration"].iloc[0] == _CS_MM
    assert abs(df["rlnImagePixelSize"].iloc[0] - _PIXEL_SIZE_A) < 1e-4
    assert df["rlnAmplitudeContrast"].iloc[0] == _ALPHA
    assert df["rlnDefocusU"].iloc[0] == _DFU_A
    assert df["rlnDefocusV"].iloc[0] == _DFV_A


def _write_minimal_starfile(path: Path, n: int) -> None:
    """A single-block RELION .star with every column _load_starfile_parameters reads."""
    starfile.write(
        pd.DataFrame(
            {
                "rlnVoltage": np.full(n, _VOLTAGE_KV),
                "rlnImagePixelSize": np.full(n, _PIXEL_SIZE_A),
                "rlnAmplitudeContrast": np.full(n, _ALPHA),
                "rlnSphericalAberration": np.full(n, _CS_MM),
                "rlnDefocusU": np.full(n, _DFU_A),
                "rlnDefocusV": np.full(n, _DFV_A),
                "rlnDefocusAngle": np.zeros(n),
                "rlnPhaseShift": np.zeros(n),
                "rlnAngleRot": np.zeros(n),
                "rlnAngleTilt": np.zeros(n),
                "rlnAnglePsi": np.zeros(n),
                "rlnOriginXAngst": np.zeros(n),
                "rlnOriginYAngst": np.zeros(n),
            }
        ),
        path,
        overwrite=True,
    )


def test_run_particle_stack_from_starfile(tmp_path: Path) -> None:
    """star_path is the RELION counterpart of cs_path -- same code path in
    src/specter/pipelines/_particles.py, fed by
    `extract_parameters_from_starfile` instead, which returns the same
    10-tuple.
    """
    star_path = tmp_path / "minimal.star"
    _write_minimal_starfile(star_path, n=5)

    config = ParticleStackConfig(
        pdb_source="6bdf",
        n_pixels=64,
        star_path=str(star_path),
        n_particles=5,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        batchsize=5,
        output_dir=str(tmp_path),
        filename="star_particles",
    )
    run_particle_stack(config)

    with mrcfile.open(tmp_path / "star_particles.mrcs") as mrc:
        assert mrc.data.shape == (5, 64, 64)

    df = starfile.read(tmp_path / "star_particles.star")
    assert len(df) == 5
    # These come from the input .star file, not ParticleStackConfig's defaults.
    assert df["rlnVoltage"].iloc[0] == _VOLTAGE_KV
    assert df["rlnSphericalAberration"].iloc[0] == _CS_MM
    assert abs(df["rlnImagePixelSize"].iloc[0] - _PIXEL_SIZE_A) < 1e-4
    assert df["rlnAmplitudeContrast"].iloc[0] == _ALPHA
    assert df["rlnDefocusU"].iloc[0] == _DFU_A
    assert df["rlnDefocusV"].iloc[0] == _DFV_A


def test_run_particle_stack_rejects_both_cs_and_star_path(tmp_path: Path) -> None:
    """The two dataset sources are mutually exclusive -- silently preferring
    one would make the ignored flag look like it had been applied.
    """
    cs_path = tmp_path / "minimal.cs"
    _write_minimal_csfile(cs_path, n=5)
    star_path = tmp_path / "minimal.star"
    _write_minimal_starfile(star_path, n=5)

    config = ParticleStackConfig(
        pdb_source="6bdf",
        cs_path=str(cs_path),
        star_path=str(star_path),
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="can't both be set"):
        run_particle_stack(config)


def test_run_particle_stack_applies_tetrafoil(tmp_path: Path) -> None:
    """config.tetrafoil1-4 reach the CTF (see _particles.py's ctf_params).

    tetrafoil3 is a k^4 cos(4*theta) phase term in Angstrom^4 -- at k = 0.25
    1/Angstrom, 256 Angstrom^4 is ~1 radian, so a value this size has to
    visibly change the images relative to the same run with tetrafoil off.
    """
    common = dict(
        pdb_source="6bdf",
        n_pixels=64,
        pixel_size=2.0,
        n_particles=2,
        seed=1234,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        noise_model="none",
        normalize_particles=False,
        device="cpu",
        batchsize=2,
        output_dir=str(tmp_path),
    )
    run_particle_stack(ParticleStackConfig(**common, filename="no_tetrafoil"))
    run_particle_stack(
        ParticleStackConfig(**common, filename="with_tetrafoil", tetrafoil3=2000.0)
    )

    with mrcfile.open(tmp_path / "no_tetrafoil.mrcs") as mrc:
        off = mrc.data.copy()
    with mrcfile.open(tmp_path / "with_tetrafoil.mrcs") as mrc:
        on = mrc.data.copy()
    assert not np.allclose(off, on)


def test_run_particle_stack_auto_batchsize(tmp_path: Path) -> None:
    """batchsize="auto" (the default) sizes itself and runs to completion.

    On CPU with a 32-pixel box the recommendation is memory-unconstrained, so
    this pins the clamp instead: never more than n_particles.
    """
    config = ParticleStackConfig(
        pdb_source="6bdf",
        n_pixels=32,
        pixel_size=4.0,
        n_particles=3,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        output_dir=str(tmp_path),
        filename="auto_batch",
    )
    assert config.batchsize == "auto"
    run_particle_stack(config)

    with mrcfile.open(tmp_path / "auto_batch.mrcs") as mrc:
        assert mrc.data.shape == (3, 32, 32)


def test_run_particle_stack_rejects_seed_with_auto_batchsize(tmp_path: Path) -> None:
    """
    A seeded run must not silently accept batchsize="auto".

    Batching decides which random draw (ice crop, Poisson noise) reaches which
    particle, and "auto" is sized to the memory free on the device at run time
    -- so the same seed would give a different stack on a different machine.
    """
    config = ParticleStackConfig(
        pdb_source="6bdf",
        n_pixels=32,
        n_particles=2,
        seed=11,
        device="cpu",
        output_dir=str(tmp_path),
    )
    assert config.batchsize == "auto"
    with pytest.raises(ValueError, match="batchsize"):
        run_particle_stack(config)


def test_run_particle_stack_tracked_by_project(tmp_path: Path) -> None:
    """--project routes output through specter.jobs instead of output_dir/
    filename -- opt-in, unlike reconstruction's always-on tracking."""
    config = ParticleStackConfig(
        pdb_source="6bdf",
        n_pixels=32,
        n_particles=2,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        batchsize=2,
        project="apoferritin",
        output_dir=str(tmp_path),
    )
    run_particle_stack(config)

    job_dir = tmp_path / "apoferritin" / "particles" / "J001"
    assert (job_dir / "particles.mrcs").exists()
    assert (job_dir / "particles.star").exists()

    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"
    assert job["params"]["n_particles"] == 2


def test_run_micrograph_tracked_by_project(tmp_path: Path) -> None:
    config = MicrographConfig(
        pdb_source="6bdf",
        n_pixels=32,
        micrograph_size=64,
        n_micrographs=1,
        scattering_model="ctf",
        ice_model="none",
        detector_model="none",
        noise_model="none",
        device="cpu",
        project="apoferritin",
        output_dir=str(tmp_path),
    )
    run_micrograph(config)

    job_dir = tmp_path / "apoferritin" / "micrographs" / "J001"
    with mrcfile.open(job_dir / "micrographs.mrcs") as mrc:
        assert mrc.data.shape == (1, 64, 64)
    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"


def test_run_tilt_series_tracked_by_project(tmp_path: Path) -> None:
    import torch

    volume_path = tmp_path / "volume.pt"
    torch.save(torch.rand(32, 48, 48) * 0.01, volume_path)

    config = TiltSeriesConfig(
        volume_path=str(volume_path),
        voxel_size=4.0,
        micrograph_size=48,
        min_tilt_angle=-5,
        max_tilt_angle=5,
        n_tilts=3,
        n_frames=1,
        ice_model="none",
        detector_model="none",
        scattering_model="ctf",
        device="cpu",
        project="apoferritin",
        output_dir=str(tmp_path),
    )
    run_tilt_series(config)

    job_dir = tmp_path / "apoferritin" / "tiltseries" / "J001"
    with mrcfile.open(job_dir / "tilt_series.mrcs") as mrc:
        assert mrc.data.shape == (3, 48, 48)
    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"


def _minimal_tomogram_config(**overrides) -> TomogramConfig:
    defaults = dict(
        target_shape=[24, 48, 48],
        voxel_size=12.0,
        targets=[{"pdb_source": "6bdf", "n_copies": 1}],
        device="cpu",
        write_picks=False,
        write_segmentation=False,
    )
    defaults.update(overrides)
    return TomogramConfig(**defaults)


def test_run_build_tomogram_tracked_by_project(tmp_path: Path) -> None:
    config = _minimal_tomogram_config(project="apoferritin", output_dir=str(tmp_path))
    run_build_tomogram(config)

    job_dir = tmp_path / "apoferritin" / "tomograms" / "J001"
    assert (job_dir / "tomogram.mrc").exists()
    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"


def test_tomogram_output_path_tracked_requires_pinned_job_id() -> None:
    config = _minimal_tomogram_config(project="apoferritin")
    with pytest.raises(ValueError, match="job_id"):
        tomogram_output_path(config)


def test_tomogram_output_path_tracked_with_pinned_job_id(tmp_path: Path) -> None:
    config = _minimal_tomogram_config(
        project="apoferritin", job_id="J005", output_dir=str(tmp_path)
    )
    assert tomogram_output_path(config) == str(
        tmp_path / "apoferritin" / "tomograms" / "J005" / "tomogram.mrc"
    )


def test_run_tilt_series_chained_tomogram_config_creates_two_separate_jobs(
    tmp_path: Path,
) -> None:
    """A tracked run with tomogram_config produces two separate, same-project
    jobs (one "tomograms", one "tiltseries"), linked implicitly by
    volume_path -- not one merged job, and not one tracked + one flat."""
    tomogram_config = _minimal_tomogram_config()
    assert tomogram_config.project is None  # not independently configured

    config = TiltSeriesConfig(
        voxel_size=12.0,
        micrograph_size=48,
        min_tilt_angle=-5,
        max_tilt_angle=5,
        n_tilts=3,
        n_frames=1,
        ice_model="none",
        noise_model="none",
        detector_model="none",
        scattering_model="ctf",
        device="cpu",
        project="apoferritin",
        output_dir=str(tmp_path),
    )
    run_tilt_series(config, tomogram_config=tomogram_config)

    tomo_dir = tmp_path / "apoferritin" / "tomograms" / "J001"
    tilt_dir = tmp_path / "apoferritin" / "tiltseries" / "J002"
    assert (tomo_dir / "tomogram.mrc").exists()
    assert (tilt_dir / "tilt_series.mrcs").exists()

    tomo_job = json.loads((tomo_dir / "job.json").read_text())
    tilt_job = json.loads((tilt_dir / "job.json").read_text())
    assert tomo_job["status"] == "complete"
    assert tilt_job["status"] == "complete"
    # The link: the tiltseries job's own recorded volume_path points
    # straight into the tomogram job's directory.
    assert tilt_job["params"]["volume_path"] == str(tomo_dir / "tomogram.mrc")


def test_run_tilt_series_chained_tomogram_config_respects_explicit_tracking(
    tmp_path: Path,
) -> None:
    """If tomogram_config already has its own project, cascading doesn't
    override it -- e.g. reusing one tracked tomogram build across several
    tiltseries runs in a different project."""
    tomogram_config = _minimal_tomogram_config(
        project="shared-tomograms", output_dir=str(tmp_path)
    )
    config = TiltSeriesConfig(
        voxel_size=12.0,
        micrograph_size=48,
        min_tilt_angle=-5,
        max_tilt_angle=5,
        n_tilts=3,
        n_frames=1,
        ice_model="none",
        noise_model="none",
        detector_model="none",
        scattering_model="ctf",
        device="cpu",
        project="this-run",
        output_dir=str(tmp_path),
    )
    run_tilt_series(config, tomogram_config=tomogram_config)

    assert (
        tmp_path / "shared-tomograms" / "tomograms" / "J001" / "tomogram.mrc"
    ).exists()
    assert (tmp_path / "this-run" / "tiltseries" / "J001" / "tilt_series.mrcs").exists()


class _FakeGenerator:
    """Minimal stand-in for _report_devices, which reads only these two."""

    def __init__(self, device, accumulator_device):
        self.device = device
        self.accumulator_device = torch.device(accumulator_device)


@pytest.mark.parametrize(
    ("device", "accumulator", "raw", "expected"),
    [
        # The common case: one device, nothing to say beyond naming it. In
        # particular "cuda" (a str, as a caller passes it) against
        # torch.device("cuda") must read as the SAME device -- comparing the
        # two directly is always unequal, which made every run claim a split.
        ("cuda", "cuda", None, "Device: cuda"),
        ("cuda", "cuda", "auto", "Device: cuda"),
        ("cpu", "cpu", None, "Device: cpu"),
        # A split is the exception, so it is what gets reported -- and "auto"
        # is named as the thing that chose, since the user did not.
        ("cuda", "cpu", "auto", "Device: cuda, accumulator on cpu (auto)"),
        ("cuda", "cpu", "cpu", "Device: cuda, accumulator on cpu"),
        ("cuda", "cuda:2", "cuda:2", "Device: cuda, accumulator on cuda:2"),
    ],
)
def test_report_devices_names_a_split_accumulator(
    device, accumulator, raw, expected, capsys
):
    """
    A run reports where its canvas landed, but only when that differs.

    `accumulator_device="auto"` decides from *currently free* VRAM, so the
    same config silently puts the canvas on the GPU or the CPU depending on
    what else was running. Unreported, the only symptom is a run that is
    inexplicably slower than the last one.
    """
    from specter.pipelines._tomogram import _report_devices

    _report_devices(_FakeGenerator(device, accumulator), raw)
    assert capsys.readouterr().out.strip() == expected
