from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from cryosparc.dataset import Dataset

from specter.config import ParticleStackConfig
from specter.pipelines import run_particle_stack

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
        pdb_code="6bdf",
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
        pdb_code="6bdf",
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
        pdb_code="6bdf",
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
        pdb_code="6bdf",
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
        pdb_code="6bdf",
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
        pdb_code="6bdf",
        n_pixels=32,
        n_particles=2,
        seed=11,
        device="cpu",
        output_dir=str(tmp_path),
    )
    assert config.batchsize == "auto"
    with pytest.raises(ValueError, match="batchsize"):
        run_particle_stack(config)
