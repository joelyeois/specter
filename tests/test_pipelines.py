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

# These tests check pipeline wiring and output bookkeeping. A small bundled
# structure exercises the same paths without parsing a large assembly or
# depending on the developer's download cache.
_PDB_FIXTURE = str(Path(__file__).parent / "test_data" / "1mbo.cif")

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


@pytest.mark.parametrize(
    ("write_input", "config_field", "filename"),
    [
        (_write_minimal_csfile, "cs_path", "minimal.cs"),
        (_write_minimal_starfile, "star_path", "minimal.star"),
    ],
    ids=["csfile", "starfile"],
)
def test_run_particle_stack_from_dataset_file(
    tmp_path: Path, write_input, config_field: str, filename: str
) -> None:
    """cs_path drives pixel_size/voltage/alpha/poses/CTF from a CryoSPARC
    passthrough .cs file instead of synthetic sampling -- see
    src/specter/pipelines/_particles.py. star_path is its RELION
    counterpart -- same code path, fed by `extract_parameters_from_starfile`
    instead, which returns the same 10-tuple. For a real-data demonstration
    of this same code path against EMPIAR-11377, see
    docs/user-guide/particle-stack.md and
    docs-figures/particle_stack_empiar_11377.py.
    """
    input_path = tmp_path / filename
    write_input(input_path, n=5)

    config = ParticleStackConfig(
        pdb_source="6bdf",
        n_pixels=64,
        n_particles=5,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        batchsize=5,
        output_dir=str(tmp_path),
        filename="particles_out",
        **{config_field: str(input_path)},
    )
    run_particle_stack(config)

    with mrcfile.open(tmp_path / "particles_out.mrcs") as mrc:
        assert mrc.data.shape == (5, 64, 64)

    df = starfile.read(tmp_path / "particles_out.star")
    assert len(df) == 5
    # These come from the input file, not ParticleStackConfig's (unused) defaults.
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


def test_run_particle_stack_seed_reproduces_across_batch_sizes(tmp_path: Path) -> None:
    """
    The same seed gives the same stack at batchsize 1 and 3, and split over
    two worker processes, with every stochastic stage on (random ice,
    crowding, Poisson noise): per-particle draws are derived from (seed,
    particle index), so batching -- including the "auto" batchsize a seeded
    run used to be refused with -- and sharding cannot change which draw
    reaches which particle.
    """
    stacks = []
    for batchsize, name, device in (
        (1, "one", "cpu"),
        (3, "three", "cpu"),
        ("auto", "auto", "cpu"),
        # Two workers, two particles and one: each share is drawn from the
        # same per-particle seeds, so the stitched stack is the same stack.
        (1, "shared", "cpu,cpu"),
    ):
        config = ParticleStackConfig(
            pdb_source=_PDB_FIXTURE,
            n_pixels=32,
            pixel_size=4.0,
            n_particles=3,
            seed=11,
            batchsize=batchsize,
            scattering_model="projection",
            ice_model="random",
            noise_model="poisson",
            detector_model="none",
            device=device,
            output_dir=str(tmp_path),
            filename=name,
        )
        run_particle_stack(config)
        with mrcfile.open(tmp_path / f"{name}.mrcs") as mrc:
            stacks.append(mrc.data.copy())
    assert np.array_equal(stacks[0], stacks[1])
    assert np.array_equal(stacks[0], stacks[2])
    assert np.array_equal(stacks[0], stacks[3])


def _minimal_tomogram_config(**overrides) -> TomogramConfig:
    defaults = dict(
        target_shape=[24, 48, 48],
        # Coarse voxels grow the supersampled atomic kernels cubically.
        # At 4 A this box still comfortably holds the small test structure.
        voxel_size=4.0,
        targets=[{"pdb_source": _PDB_FIXTURE, "n_copies": 1}],
        device="cpu",
        write_picks=False,
        write_segmentation=False,
    )
    defaults.update(overrides)
    return TomogramConfig(**defaults)


def _tracked_particle_config(tmp_path: Path) -> ParticleStackConfig:
    return ParticleStackConfig(
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


def _tracked_micrograph_config(tmp_path: Path) -> MicrographConfig:
    return MicrographConfig(
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


def _tracked_tilt_series_config(tmp_path: Path) -> TiltSeriesConfig:
    volume_path = tmp_path / "volume.pt"
    torch.save(torch.rand(32, 48, 48) * 0.01, volume_path)
    return TiltSeriesConfig(
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


def _tracked_tomogram_config(tmp_path: Path) -> TomogramConfig:
    return _minimal_tomogram_config(project="apoferritin", output_dir=str(tmp_path))


@pytest.mark.parametrize(
    ("make_config", "run", "job_type", "artifacts", "params"),
    [
        (
            _tracked_particle_config,
            run_particle_stack,
            "particles",
            {"particles.mrcs": None, "particles.star": None},
            {"n_particles": 2},
        ),
        (
            _tracked_micrograph_config,
            run_micrograph,
            "micrographs",
            {"micrographs.mrcs": (1, 64, 64)},
            {},
        ),
        (
            _tracked_tilt_series_config,
            run_tilt_series,
            "tiltseries",
            {"tilt_series.mrcs": (3, 48, 48)},
            {},
        ),
        (
            _tracked_tomogram_config,
            run_build_tomogram,
            "tomograms",
            {"tomogram.mrc": None},
            {},
        ),
    ],
    ids=["particles", "micrograph", "tilt_series", "tomogram"],
)
def test_run_pipeline_tracked_by_project(
    tmp_path: Path, make_config, run, job_type: str, artifacts: dict, params: dict
) -> None:
    """--project routes output through specter.jobs instead of output_dir/
    filename -- opt-in, unlike reconstruction's always-on tracking. Every
    image file carries its pixel size in the header, which the micrograph,
    tilt-series and particle stacks left at 0 until 2026-10-15."""
    config = make_config(tmp_path)
    run(config)
    pixel_size = getattr(config, "pixel_size", None) or config.voxel_size

    job_dir = tmp_path / "apoferritin" / job_type / "J001"
    for name, shape in artifacts.items():
        assert (job_dir / name).exists()
        if name.endswith((".mrc", ".mrcs")):
            with mrcfile.open(job_dir / name) as mrc:
                if shape is not None:
                    assert mrc.data.shape == shape
                assert float(mrc.voxel_size.x) == pytest.approx(pixel_size, rel=1e-6)

    job = json.loads((job_dir / "job.json").read_text())
    assert job["status"] == "complete"
    for key, value in params.items():
        assert job["params"][key] == value


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
        voxel_size=tomogram_config.voxel_size,
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
    # One seed reproduces both: the series' seed, drawn because none was
    # given, is recorded, and the tomogram's is derived from it.
    from specter.imaging._stages import SEED_STAGE_SPECIMEN
    from specter.random_seed import derive_seed

    seed = tilt_job["params"]["seed"]
    assert isinstance(seed, int)
    assert tomo_job["params"]["seed"] == derive_seed(seed, SEED_STAGE_SPECIMEN)


def test_run_tilt_series_chained_tomogram_config_respects_explicit_tracking(
    tmp_path: Path,
) -> None:
    """If tomogram_config already has its own project, or seed, cascading
    doesn't override it -- e.g. reusing one tracked tomogram build across
    several tiltseries runs in a different project."""
    tomogram_config = _minimal_tomogram_config(
        project="shared-tomograms", output_dir=str(tmp_path), seed=11
    )
    config = TiltSeriesConfig(
        seed=5,
        voxel_size=tomogram_config.voxel_size,
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

    tomo_dir = tmp_path / "shared-tomograms" / "tomograms" / "J001"
    assert (tomo_dir / "tomogram.mrc").exists()
    assert (tmp_path / "this-run" / "tiltseries" / "J001" / "tilt_series.mrcs").exists()
    # Its own seed is kept too.
    assert json.loads((tomo_dir / "job.json").read_text())["params"]["seed"] == 11


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


class _Captured(Exception):
    """Raised by a stub generator once it has recorded its arguments."""


_EXPOSURE_FIELDS = dict(
    absorption_model="inelastic_mfp",
    inelastic_mfp_solvent=3000.0,
    inelastic_mfp_specimen=2460.0,
    objective_aperture=12.0,
    dose_envelope=True,
    dose_envelope_target="specimen",
    ice_motion_variance=0.38,
)


def test_run_micrograph_forwards_absorption_and_exposure_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every absorption and exposure setting reaches the specimen draw."""
    from specter.imaging import Field

    seen: dict = {}

    def draw(self: Field, **kwargs: object) -> None:
        seen.update(kwargs, field=self)
        raise _Captured

    monkeypatch.setattr(Field, "draw", draw)
    config = MicrographConfig(
        pdb_source="6bdf",
        n_pixels=32,
        micrograph_size=64,
        ice_model="random",
        bulk_scattering_factors="lobato",
        device="cpu",
        output_dir=str(tmp_path),
        **_EXPOSURE_FIELDS,
    )
    with pytest.raises(_Captured):
        run_micrograph(config)
    assert seen["absorption_model"] == "inelastic_mfp"
    assert seen["alpha"] == 0.0  # the config's 0.1 is dropped, not doubled
    assert seen["inelastic_mfp_solvent"] == 3000.0
    assert seen["inelastic_mfp_specimen"] == 2460.0
    assert seen["objective_aperture"] == 12.0
    field = seen["field"]
    assert field.damaged  # dose_envelope_target="specimen"
    assert field.ice.motion_variance == 0.38
    assert field.icemaker.parameterization == "lobato"
    from specter.ice._kernels import build_water_kernel

    torch.testing.assert_close(
        field.icemaker.ice_kernel, build_water_kernel(config.pixel_size, "lobato")
    )


def test_run_tilt_series_forwards_absorption_and_exposure_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every absorption and exposure setting reaches the series' preparation."""
    import torch

    from specter.imaging import Sample

    seen: dict = {}

    def tilt_series(
        self: Sample, poses: object, exposure: object, **kwargs: object
    ) -> None:
        seen.update(kwargs, exposure=exposure)
        raise _Captured

    monkeypatch.setattr(Sample, "tilt_series", tilt_series)
    volume_path = tmp_path / "volume.pt"
    torch.save(torch.rand(16, 32, 32) * 0.01, volume_path)
    config = TiltSeriesConfig(
        volume_path=str(volume_path),
        voxel_size=4.0,
        n_tilts=3,
        ice_model="random",
        device="cpu",
        output_dir=str(tmp_path),
        **_EXPOSURE_FIELDS,
    )
    with pytest.raises(_Captured):
        run_tilt_series(config)
    propagation = seen["propagation"]
    assert propagation.absorption_model == "inelastic_mfp"
    assert propagation.alpha == 0.0  # the config's 0.1 is dropped, not doubled
    assert propagation.inelastic_mfp_solvent == 3000.0
    assert propagation.inelastic_mfp_specimen == 2460.0
    assert seen["objective_aperture"] == 12.0
    assert seen["damaged"]  # dose_envelope_target="specimen"
    assert seen["ice"].motion_variance == 0.38
    assert not seen["exposure"].weighted  # a tilt is a plain sum


@pytest.mark.parametrize(
    ("ice_model", "suffix"), [("random", "exitwave"), ("none", "clean_exitwave")]
)
def test_tilt_series_exitwaves_are_named_by_whether_ice_was_added(
    tmp_path: Path, ice_model: str, suffix: str
) -> None:
    """
    Exit waves with ice are ``exitwave``, without it ``clean_exitwave``.

    The pipeline once read the ice setting off a model that reset it, so a
    run with ice also labelled its exit waves "clean".
    """
    volume_path = tmp_path / "volume.pt"
    torch.save(torch.rand(8, 24, 24) * 0.01, volume_path)
    config = TiltSeriesConfig(
        volume_path=str(volume_path),
        voxel_size=4.0,
        n_tilts=2,
        ice_model=ice_model,
        save_exitwaves=True,
        noise_model="none",
        device="cpu",
        output_dir=str(tmp_path / "out"),
        filename="ts",
    )
    run_tilt_series(config)
    out = tmp_path / "out"
    assert (out / f"ts_{suffix}_magnitude.mrcs").exists()
    assert (out / f"ts_{suffix}_phase.mrcs").exists()


def test_run_micrograph_writes_clean_exit_waves_without_ice(tmp_path: Path) -> None:
    """With no ice the specimen is its own clean form; asking for its clean
    exit waves used to raise AttributeError, since none had been kept."""
    config = MicrographConfig(
        pdb_source=_PDB_FIXTURE,
        n_pixels=24,
        micrograph_size=48,
        pixel_size=3.0,
        n_micrographs=1,
        ice_model="none",
        noise_model="none",
        save_exitwaves=True,
        save_clean_exitwaves=True,
        seed=1,
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_micrograph(config)
    for part in ("magnitude", "phase"):
        with mrcfile.open(tmp_path / f"micrographs_exitwave_{part}.mrcs") as m:
            wave = m.data.copy()
        with mrcfile.open(tmp_path / f"micrographs_clean_exitwave_{part}.mrcs") as m:
            clean = m.data.copy()
        assert np.array_equal(wave, clean)
