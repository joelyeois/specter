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
        energy = 200.0
        """,
    )
    config = load_config(path)
    assert config.pdb_code == "6bdf"
    assert config.num_pixels == 128
    assert config.energy == 200.0


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


def test_load_config_tilt_series_parses_array_of_tables(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [[protein_specs]]
        PDB_CODE = "7VD8"
        PMER_OCC = 0.078

        [[protein_specs]]
        PDB_CODE = "6QZP"

        [[membrane_specs]]
        MB_TYPE = "sphere"
        MB_THICK_RG = [25.0, 35.0]
        MB_MIN_RAD = 180.0
        MB_MAX_RAD = 450.0

        [specimen]
        target_shape = [92, 256, 256]
        target_v_size = 5.0
        """,
    )
    config = load_config(path, TiltSeriesConfig)
    assert config.protein_specs == [
        {"PDB_CODE": "7VD8", "PMER_OCC": 0.078},
        {"PDB_CODE": "6QZP"},
    ]
    assert config.membrane_specs == [
        {
            "MB_TYPE": "sphere",
            "MB_THICK_RG": [25.0, 35.0],
            "MB_MIN_RAD": 180.0,
            "MB_MAX_RAD": 450.0,
        }
    ]
    assert config.target_shape == [92, 256, 256]
    assert config.target_v_size == 5.0


def test_load_config_tilt_series_fills_defaults_for_missing_fields(
    tmp_path: Path,
) -> None:
    path = _write_toml(tmp_path, '[[protein_specs]]\nPDB_CODE = "6bdf"\n')
    config = load_config(path, TiltSeriesConfig)
    assert config.membrane_specs == []
    assert config.filler_occupancy is None
    assert config.n_tilts == 61
    assert config.scattering_model == "multislice"
    assert config.ice_model == "gd"


def test_tilt_series_config_requires_protein_specs() -> None:
    import pytest

    with pytest.raises(TypeError):
        TiltSeriesConfig()  # type: ignore[call-arg]


def test_tilt_series_toml_loads_and_matches_expected_values() -> None:
    path = str(REPO_ROOT / "configs" / "tilt_series.toml")
    config = load_config(path, TiltSeriesConfig)
    assert len(config.protein_specs) == 6
    assert len(config.membrane_specs) == 1
    assert config.target_shape == [184, 630, 630]
    assert config.target_v_size == 5.0
    assert config.scattering_model == "multislice"
    assert config.device == "cpu"
