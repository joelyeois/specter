"""Phase-shift units and the laser phase plate from a config.

The plate's phase is radians everywhere a user gives it, named
``phase_shift_rad`` as CryoSPARC's ``ctf/phase_shift_rad``; ``phaseshift``
(config) and ``phase_shift`` in degrees (Python) are deprecated spellings.
RELION files carry it in degrees as ``rlnPhaseShift``.
"""

from __future__ import annotations

import math
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
import torch
from cryosparc.dataset import Dataset

from specter.config import (
    MicrographConfig,
    ParticleStackConfig,
    TiltSeriesConfig,
    apply_overrides,
    load_config,
)
from specter.imaging import CTF
from specter.io import (
    convert_csfile_to_starfile,
    create_micrograph_starfile,
    extract_parameters_from_csfile,
    extract_parameters_from_starfile,
    read_micrograph_starfile,
)
from specter.pipelines import run_tilt_series
from specter.pipelines._common import _Microscope
from specter.settings import LaserPhasePlate

LPP_FIELDS = {
    "lpp_NA": 0.1,
    "lpp_laser_wavelength_angstrom": 10640.0,
    "lpp_focal_length_angstrom": 2e7,
    "lpp_laser_xy_angle_deg": 0.0,
    "lpp_laser_xz_angle_deg": 0.0,
    "lpp_laser_long_offset_angstrom": 0.0,
    "lpp_laser_trans_offset_angstrom": 0.0,
    "lpp_laser_polarization_angle_deg": 0.0,
    "lpp_peak_phase_deg": 90.0,
}


# --- config alias -----------------------------------------------------------


@pytest.mark.parametrize(
    ("config_cls", "value", "toml_value"),
    [
        (ParticleStackConfig, [0.5, 1.5], "[0.5, 1.5]"),
        (MicrographConfig, 1.5, "1.5"),
        (TiltSeriesConfig, 1.5, "1.5"),
    ],
)
def test_phaseshift_alias_loads_with_a_deprecation_warning(
    tmp_path: Path, config_cls: type, value: object, toml_value: str
) -> None:
    """A config written with ``phaseshift`` loads into ``phase_shift_rad``."""
    path = tmp_path / "c.toml"
    head = "" if config_cls is TiltSeriesConfig else 'pdb_source = "6bdf"\n'
    path.write_text(f"{head}[advanced]\nphaseshift = {toml_value}\n")
    with pytest.warns(DeprecationWarning, match="phase_shift_rad"):
        config = load_config(str(path), config_cls)
    assert config.phase_shift_rad == value


def test_phaseshift_alias_and_new_name_together_raise(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("phaseshift = 1.0\nphase_shift_rad = 1.0\n")
    with pytest.raises(ValueError, match="deprecated alias"):
        load_config(str(path), MicrographConfig)


def test_phaseshift_alias_in_overrides() -> None:
    config = TiltSeriesConfig()
    with pytest.warns(DeprecationWarning):
        apply_overrides(config, {"phaseshift": 0.25})
    assert config.phase_shift_rad == 0.25


# --- Python API: radians, degrees deprecated --------------------------------


def test_ctf_phase_shift_rad_is_stored_as_given() -> None:
    ctf = CTF.from_values(
        torch.tensor([1e4, 1e4]), phase_shift_rad=torch.tensor([0.1, math.pi / 2])
    )
    torch.testing.assert_close(
        ctf.params["phaseshift"], torch.tensor([0.1, math.pi / 2])
    )


def test_ctf_degrees_are_deprecated_and_equivalent() -> None:
    with pytest.warns(DeprecationWarning, match="phase_shift_rad"):
        degrees = CTF.from_values(1e4, n=2, phase_shift=90.0)
    radians = CTF.from_values(1e4, n=2, phase_shift_rad=math.pi / 2)
    torch.testing.assert_close(
        degrees.params["phaseshift"], radians.params["phaseshift"]
    )


def test_ctf_sample_degree_range_draws_the_radian_range() -> None:
    with pytest.warns(DeprecationWarning):
        degrees = CTF.sample(5, defocus=1e4, phase_shift=[30.0, 90.0], seed=3)
    radians = CTF.sample(
        5, defocus=1e4, phase_shift_rad=[math.pi / 6, math.pi / 2], seed=3
    )
    torch.testing.assert_close(
        degrees.params["phaseshift"], radians.params["phaseshift"], rtol=0, atol=1e-6
    )


def test_ctf_both_units_raise() -> None:
    with pytest.raises(ValueError, match="not both"):
        CTF.from_values(1e4, phase_shift=90.0, phase_shift_rad=1.0)
    with pytest.raises(ValueError, match="not both"):
        CTF.sample(2, defocus=1e4, phase_shift=90.0, phase_shift_rad=1.0)


# --- STAR and .cs round trips -----------------------------------------------


def test_micrograph_star_writes_degrees_and_reads_radians(tmp_path: Path) -> None:
    """``rlnPhaseShift`` is degrees on disk and radians once read back."""
    shifts = torch.tensor([0.0, 0.3, math.pi / 2])
    params = {
        "cs": torch.full((3,), 2.7e7),
        "dfu": torch.full((3,), 1e4),
        "phaseshift": shifts,
    }
    path = create_micrograph_starfile(3, 300.0, 1.0, 0.1, params, str(tmp_path))
    on_disk = starfile.read(path)["rlnPhaseShift"].to_numpy()
    np.testing.assert_allclose(on_disk, np.degrees(shifts.numpy()), atol=1e-5)
    back = read_micrograph_starfile(path)
    torch.testing.assert_close(back["phaseshift"], shifts, rtol=0, atol=1e-6)
    torch.testing.assert_close(back["dfu"], params["dfu"])


def test_micrograph_star_omits_an_absent_phase_shift(tmp_path: Path) -> None:
    params = {"cs": torch.full((2,), 2.7e7), "dfu": torch.full((2,), 1e4)}
    path = create_micrograph_starfile(2, 300.0, 1.0, 0.1, params, str(tmp_path))
    assert "rlnPhaseShift" not in starfile.read(path)
    assert "phaseshift" not in read_micrograph_starfile(path)


def test_csfile_phase_shift_round_trips_through_star(tmp_path: Path) -> None:
    """``ctf/phase_shift_rad`` reads as radians, converts to degrees, reads back."""
    n = 2
    shifts = np.array([0.3, math.pi / 2], dtype=np.float32)
    Dataset(
        {
            "uid": np.arange(n, dtype=np.uint64),
            "blob/path": np.array(["particles.mrcs"] * n),
            "blob/idx": np.arange(n, dtype=np.int64),
            "blob/shape": np.array([[32, 32]] * n, dtype=np.int64),
            "blob/psize_A": np.full(n, 1.0, dtype=np.float32),
            "alignments3D/shift": np.zeros((n, 2), dtype=np.float32),
            "alignments3D/psize_A": np.full(n, 1.0, dtype=np.float32),
            "alignments3D/pose": np.zeros((n, 3), dtype=np.float32),
            "alignments3D/split": np.zeros(n, dtype=np.uint32),
            "alignments3D/alpha": np.ones(n, dtype=np.float32),
            "ctf/cs_mm": np.full(n, 2.7, dtype=np.float32),
            "ctf/df_angle_rad": np.zeros(n, dtype=np.float32),
            "ctf/df1_A": np.full(n, 1e4, dtype=np.float32),
            "ctf/df2_A": np.full(n, 1e4, dtype=np.float32),
            "ctf/amp_contrast": np.full(n, 0.1, dtype=np.float32),
            "ctf/accel_kv": np.full(n, 300.0, dtype=np.float32),
            "ctf/tilt_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/phase_shift_rad": shifts,
            "ctf/shift_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/trefoil_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/tetra_A": np.zeros((n, 4), dtype=np.float32),
            "ctf/anisomag": np.zeros((n, 4), dtype=np.float32),
        }
    ).save(str(tmp_path / "p.cs"))
    from_cs = extract_parameters_from_csfile(str(tmp_path / "p.cs"))
    np.testing.assert_allclose(from_cs.ctf_params["phaseshift"].numpy(), shifts)

    star = tmp_path / "p.star"
    convert_csfile_to_starfile(str(tmp_path / "p.cs"), str(star))
    particles = starfile.read(str(star))["particles"]
    np.testing.assert_allclose(
        particles["rlnPhaseShift"].to_numpy(), np.degrees(shifts), atol=1e-4
    )
    from_star = extract_parameters_from_starfile(str(star))
    np.testing.assert_allclose(
        from_star.ctf_params["phaseshift"].numpy(), shifts, atol=1e-6
    )


# --- laser phase plate from a config ----------------------------------------


def test_laser_phase_plate_settings_build_lpp_params() -> None:
    plate = LaserPhasePlate(**{k[4:]: v for k, v in LPP_FIELDS.items()})
    assert plate.lpp_params() == {k[4:]: v for k, v in LPP_FIELDS.items()}
    assert LaserPhasePlate().lpp_params() is None
    with pytest.raises(ValueError, match="lpp_focal_length_angstrom"):
        LaserPhasePlate(NA=0.1).lpp_params()


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"phase_plate": "laser", "aberration_backend": "torch_ctf"}, "needs lpp"),
        ({"aberration_backend": "torch_ctf", **LPP_FIELDS}, "laser phase plate"),
        (
            {"phase_plate": "volta", "aberration_backend": "torch_ctf", **LPP_FIELDS},
            "laser phase plate",
        ),
        ({"phase_plate": "laser", **LPP_FIELDS}, "torch_ctf"),
        (
            {"phase_plate": "laser", "aberration_backend": "torch_ctf", "lpp_NA": 0.1},
            "missing",
        ),
    ],
)
def test_microscope_refuses_bad_laser_configs(fields: dict, match: str) -> None:
    config = TiltSeriesConfig(**fields)
    with pytest.raises(ValueError, match=match):
        _Microscope.from_config(config)


def _tilt_series(tmp_path: Path, name: str, **fields: object) -> np.ndarray:
    volume_path = tmp_path / "volume.pt"
    if not volume_path.exists():
        torch.manual_seed(0)
        torch.save(torch.rand(8, 24, 24) * 0.01, volume_path)
    out = tmp_path / name
    config = TiltSeriesConfig(
        volume_path=str(volume_path),
        voxel_size=4.0,
        n_tilts=2,
        ice_model="none",
        noise_model="none",
        detector_model="none",
        device="cpu",
        output_dir=str(out),
        filename="ts",
        **fields,
    )
    run_tilt_series(config)
    with mrcfile.open(out / "ts.mrcs") as f:
        return np.asarray(f.data, dtype=np.float64)


def test_laser_config_renders(tmp_path: Path) -> None:
    """A tilt-series config with phase_plate="laser" renders, unlike in focus."""
    laser = _tilt_series(
        tmp_path,
        "laser",
        defocus=0.0,
        phase_plate="laser",
        aberration_backend="torch_ctf",
        **LPP_FIELDS,
    )
    plain = _tilt_series(tmp_path, "plain", defocus=0.0, aberration_backend="torch_ctf")
    assert np.isfinite(laser).all()
    assert not np.allclose(laser, plain)


def test_tilt_series_star_records_the_phase_shift(tmp_path: Path) -> None:
    _tilt_series(tmp_path, "pp", phase_shift_rad=math.pi / 2)
    back = read_micrograph_starfile(str(tmp_path / "pp" / "ts.star"))
    torch.testing.assert_close(
        back["phaseshift"], torch.full((2,), math.pi / 2), rtol=0, atol=1e-6
    )
