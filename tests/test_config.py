from __future__ import annotations

from pathlib import Path

from specter.config import (
    REPO_ROOT,
    ParticleStackConfig,
    TiltSeriesConfig,
    apply_overrides,
    load_config,
)


def _write_toml(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return str(path)


def test_load_config_flattens_tables(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [potential]
        pdb_code = "6bdf"
        num_pixels = 128

        [microscope]
        voltage = 200.0
        """,
    )
    config = load_config(path)
    assert config.pdb_code == "6bdf"
    assert config.num_pixels == 128
    assert config.voltage == 200.0


def test_load_config_fills_defaults_for_missing_fields(tmp_path: Path) -> None:
    path = _write_toml(tmp_path, '[potential]\npdb_code = "6bdf"\n')
    config = load_config(path)
    assert config.num_pixels == 256
    assert config.pixel_size == 1.0
    assert config.scattering_model == "multislice"
    assert config.dose == "20"


def test_load_config_resolves_relative_pdb_savefolder_to_repo_root(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path, '[potential]\npdb_code = "6bdf"\npdb_savefolder = "my-cache"\n'
    )
    config = load_config(path)
    assert config.pdb_savefolder == str(REPO_ROOT / "my-cache")


def test_load_config_preserves_absolute_pdb_savefolder(tmp_path: Path) -> None:
    absolute = str(tmp_path / "cache")
    path = _write_toml(
        tmp_path, f'[potential]\npdb_code = "6bdf"\npdb_savefolder = "{absolute}"\n'
    )
    config = load_config(path)
    assert config.pdb_savefolder == absolute


def test_particle_stack_config_requires_pdb_code() -> None:
    import pytest

    with pytest.raises(TypeError):
        ParticleStackConfig()  # type: ignore[call-arg]


def test_apply_overrides_sets_fields() -> None:
    config = ParticleStackConfig(pdb_code="6bdf")
    result = apply_overrides(config, {"n_particles": 500, "device": "cuda:0"})
    assert result is config
    assert config.n_particles == 500
    assert config.device == "cuda:0"


def test_apply_overrides_with_empty_dict_is_noop() -> None:
    config = ParticleStackConfig(pdb_code="6bdf")
    apply_overrides(config, {})
    assert config.n_particles == 20  # dataclass default, untouched


def test_particle_toml_loads_and_matches_expected_values() -> None:
    path = str(REPO_ROOT / "configs" / "particle.toml")
    config = load_config(path)
    assert config.pdb_code == "6bdf"
    assert config.num_pixels == 256
    assert config.pixel_size == 1.0
    assert config.scattering_model == "multislice"
    assert config.device == "cpu"


def test_particle_stack_config_advanced_field_defaults() -> None:
    """Defaults for the newly-exposed 'Advanced' fields should reproduce
    today's previously-hardcoded behavior (or, for ice_parameterization, the
    deliberately-changed shtyrov default -- see config.py's docstring)."""
    config = ParticleStackConfig(pdb_code="6bdf")
    assert config.potential_parameterization == "shtyrov"
    assert config.potential_method == "analytic"
    assert config.rcut is None
    assert config.conv_backend == "fftconvolve"
    assert config.periodic is False
    assert config.atom_species is None
    assert config.ews_curvature_sign == "positive"
    assert config.klim is None
    assert config.rotate_mode == "real"
    assert config.ice_parameterization == "shtyrov"
    assert config.ice_relax_steps == 0
    assert config.crowd_chunk_size == 1
    assert config.crowd_max_distance_xy is None
    assert config.crowd_method == "3d"
    assert config.crowd_n_points is None
    assert config.crowd_seed == "origin"
    assert config.crowd_move_to_cpu is False
    assert config.water_air_interface is False
    assert config.seed is None
    assert config.astigmatism == "0"
    assert config.astigmatism_angle == "0,180"
    assert config.phaseshift == "0"
    assert config.tiltx == "0"
    assert config.tilty == "0"
    assert config.trefoil1 == "0"
    assert config.trefoil2 == "0"
    assert (
        config.anisomag_m00,
        config.anisomag_m01,
        config.anisomag_m10,
        config.anisomag_m11,
    ) == (1.0, 0.0, 0.0, 1.0)


def test_particle_stack_config_falcon4i_is_valid_detector_model() -> None:
    config = ParticleStackConfig(pdb_code="6bdf", detector_model="falcon4i_300kv")
    assert config.detector_model == "falcon4i_300kv"


def test_particle_toml_loads_advanced_fields() -> None:
    """The bundled particle.toml's [advanced] block should parse and match
    the dataclass defaults it was written to mirror."""
    path = str(REPO_ROOT / "configs" / "particle.toml")
    config = load_config(path)
    assert config.potential_parameterization == "shtyrov"
    assert config.ice_parameterization == "shtyrov"
    assert config.astigmatism == "0"
    assert config.anisomag_m11 == 1.0


def test_load_config_tilt_series_parses_scalar_fields(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [specimen]
        target_v_size = 5.0

        [tilt_geometry]
        n_tilts = 21
        tilt_axis = "x"
        """,
    )
    config = load_config(path, TiltSeriesConfig)
    assert config.target_v_size == 5.0
    assert config.n_tilts == 21
    assert config.tilt_axis == "x"


def test_load_config_tilt_series_fills_defaults_for_missing_fields(
    tmp_path: Path,
) -> None:
    path = _write_toml(tmp_path, "[specimen]\ntarget_v_size = 5.0\n")
    config = load_config(path, TiltSeriesConfig)
    assert config.volume_path == ""
    assert config.n_tilts == 61
    assert config.scattering_model == "multislice"
    assert config.ice_model == "gd"


def test_tilt_series_config_constructs_with_no_args() -> None:
    """The volume_path path shouldn't require any other TOML table."""
    config = TiltSeriesConfig()
    assert config.volume_path == ""


def test_tilt_series_toml_loads_and_matches_expected_values() -> None:
    path = str(REPO_ROOT / "configs" / "tilt_series.toml")
    config = load_config(path, TiltSeriesConfig)
    assert config.target_v_size == 2.0
    assert config.n_tilts == 61
    assert config.scattering_model == "multislice"
    assert config.device == "cpu"
    assert config.volume_path == ""


def test_tilt_series_toml_volume_path_round_trip(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [specimen]
        volume_path = "path/to/specimen.mrc"

        [tilt_geometry]
        n_tilts = 5
        """,
    )
    config = load_config(path, TiltSeriesConfig)
    assert config.volume_path == "path/to/specimen.mrc"
    assert config.n_tilts == 5
