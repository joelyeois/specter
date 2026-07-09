from __future__ import annotations

from pathlib import Path

from specter.config import REPO_ROOT, ParticleStackConfig, apply_overrides, load_config


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
    assert config.dose_max is None


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


def test_default_toml_loads_and_matches_expected_values() -> None:
    path = str(REPO_ROOT / "configs" / "particle_stack" / "default.toml")
    config = load_config(path)
    assert config.pdb_code == "6bdf"
    assert config.num_pixels == 256
    assert config.pixel_size == 1.0
    assert config.scattering_model == "multislice"
    assert config.device == "cpu"
