from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from specter.config import (
    PDB_CACHE_ENV_VAR,
    REPO_ROOT,
    MicrographConfig,
    ParticleStackConfig,
    TiltSeriesConfig,
    TomogramConfig,
    apply_overrides,
    PROJECT_MARKER,
    ensure_project_root,
    find_specter_project_root,
    load_config,
    validate_config,
    parse_scalar_or_range,
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
        pdb_source = "6bdf"
        n_pixels = 128

        [microscope]
        voltage = 200.0
        """,
    )
    config = load_config(path)
    assert config.pdb_source == "6bdf"
    assert config.n_pixels == 128
    assert config.voltage == 200.0


def test_load_config_fills_defaults_for_missing_fields(tmp_path: Path) -> None:
    path = _write_toml(tmp_path, '[potential]\npdb_source = "6bdf"\n')
    config = load_config(path)
    assert config.n_pixels == 256
    assert config.pixel_size == 1.0
    assert config.scattering_model == "multislice"
    assert config.dose == 20.0


def test_parse_scalar_or_range_accepts_numbers_lists_and_strings() -> None:
    """Numbers/lists (the TOML spelling) and strings (the CLI spelling) agree."""
    assert parse_scalar_or_range(20) == (20.0, 20.0)
    assert parse_scalar_or_range(20.0) == (20.0, 20.0)
    assert parse_scalar_or_range("20") == (20.0, 20.0)
    assert parse_scalar_or_range([5000, 15000]) == (5000.0, 15000.0)
    assert parse_scalar_or_range((5000.0, 15000.0)) == (5000.0, 15000.0)
    assert parse_scalar_or_range("5000,15000") == (5000.0, 15000.0)


def test_parse_scalar_or_range_rejects_more_than_two_values() -> None:
    with pytest.raises(ValueError):
        parse_scalar_or_range([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        parse_scalar_or_range("1,2,3")


def test_load_config_scalar_or_range_fields_accept_numeric_toml(
    tmp_path: Path,
) -> None:
    """Sampling fields read as plain TOML numbers/arrays, no quoting needed."""
    path = _write_toml(
        tmp_path,
        """
        [potential]
        pdb_source = "6bdf"

        [microscope]
        dose = 40

        [sampling]
        defocus = [8000.0, 12000.0]
        """,
    )
    config = load_config(path)
    assert parse_scalar_or_range(config.dose) == (40.0, 40.0)
    assert parse_scalar_or_range(config.defocus) == (8000.0, 12000.0)


def test_load_config_scalar_or_range_fields_still_accept_legacy_strings(
    tmp_path: Path,
) -> None:
    """Configs written before the numeric spelling keep working unchanged."""
    path = _write_toml(
        tmp_path,
        """
        [potential]
        pdb_source = "6bdf"

        [microscope]
        dose = "40"

        [sampling]
        defocus = "8000,12000"
        """,
    )
    config = load_config(path)
    assert parse_scalar_or_range(config.dose) == (40.0, 40.0)
    assert parse_scalar_or_range(config.defocus) == (8000.0, 12000.0)


def test_load_config_keeps_relative_pdb_cache_dir_verbatim(
    tmp_path: Path,
) -> None:
    """A path the user wrote is theirs -- resolved against cwd, not rewritten."""
    path = _write_toml(
        tmp_path, '[potential]\npdb_source = "6bdf"\npdb_cache_dir = "my-cache"\n'
    )
    config = load_config(path)
    assert config.pdb_cache_dir == "my-cache"


def test_results_are_cwd_relative_and_the_cache_is_not() -> None:
    """Results are project-local; the download cache deliberately is not.

    The output default stays relative so it never depends on REPO_ROOT,
    which only resolves to the repo for an editable install and would
    point inside the virtualenv for a wheel install. The structure cache
    is the opposite case: an absolute user-level path, so one download is
    shared by every project instead of re-fetched per working directory.
    """
    from specter.pipelines._common import resolve_output_dir

    config = ParticleStackConfig(pdb_source="6bdf")
    output_dir = resolve_output_dir(config, "particles")
    assert output_dir == "particles"
    assert not os.path.isabs(output_dir)
    assert os.path.isabs(config.pdb_cache_dir)
    assert config.pdb_cache_dir.endswith(os.path.join("specter", "pdb"))


def test_find_specter_project_root_uses_cwd_when_nothing_found(tmp_path: Path) -> None:
    """A fresh directory with no .specter anywhere above it becomes the
    root of a new project -- like `git init` creating a repo at cwd."""
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert find_specter_project_root(fresh) == fresh.resolve()


def test_find_specter_project_root_finds_marker_at_start(tmp_path: Path) -> None:
    (tmp_path / ".specter").touch()
    assert find_specter_project_root(tmp_path) == tmp_path.resolve()


def test_find_specter_project_root_walks_up_from_subdirectory(tmp_path: Path) -> None:
    """Running from inside an already-initialised project's subdirectory
    still resolves to the project root, not a stray new tree -- the whole
    point of the git-style search."""
    (tmp_path / ".specter").touch()
    subdir = tmp_path / "notebooks" / "nested"
    subdir.mkdir(parents=True)
    assert find_specter_project_root(subdir) == tmp_path.resolve()


def test_find_specter_project_root_prefers_nearest_ancestor(tmp_path: Path) -> None:
    """Two marked projects, one nested inside the other's subtree -- the
    nearer one wins, since that's the one actually enclosing cwd."""
    (tmp_path / ".specter").touch()
    inner = tmp_path / "other-project"
    inner.mkdir(parents=True)
    (inner / ".specter").touch()
    assert find_specter_project_root(inner) == inner.resolve()


def test_find_specter_project_root_defaults_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".specter").touch()
    monkeypatch.chdir(tmp_path)
    assert find_specter_project_root() == tmp_path.resolve()


def test_each_config_outputs_to_its_own_artifact_folder() -> None:
    """Folder name says what is inside, so two commands run in the same
    working directory don't pile results into a shared `output/`.

    Asserted through `resolve_output_dir` rather than off the field: the
    default is deliberately `None` on the dataclass, because which default
    applies depends on whether the run is tracked.
    """
    from specter.pipelines._common import resolve_output_dir

    for config, artifact in (
        (ParticleStackConfig(pdb_source="6bdf"), "particles"),
        (MicrographConfig(pdb_source="6bdf"), "micrographs"),
        (TiltSeriesConfig(), "tiltseries"),
        (TomogramConfig(), "tomograms"),
    ):
        assert config.output_dir is None
        assert resolve_output_dir(config, artifact) == artifact


def test_ensure_project_root_creates_the_marker_non_interactively(
    tmp_path: Path,
) -> None:
    """A batch job or CI run must never block on a confirmation prompt."""
    root = ensure_project_root(tmp_path, interactive=False)
    assert root == tmp_path.resolve()
    assert (tmp_path / PROJECT_MARKER).is_file()


def test_ensure_project_root_reuses_an_ancestor_marker(tmp_path: Path) -> None:
    """Running from a subdirectory joins the existing project rather than
    marking a second root inside it -- what keeps job numbering continuous."""
    (tmp_path / PROJECT_MARKER).touch()
    subdir = tmp_path / "analysis" / "run3"
    subdir.mkdir(parents=True)

    assert ensure_project_root(subdir, interactive=False) == tmp_path.resolve()
    assert not (subdir / PROJECT_MARKER).exists()


def test_ensure_project_root_declined_raises_rather_than_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering "no" must not then write the job tree into that directory."""
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with pytest.raises(SystemExit):
        ensure_project_root(tmp_path, interactive=True)
    assert not (tmp_path / PROJECT_MARKER).exists()


def test_find_specter_project_root_never_creates_a_marker(tmp_path: Path) -> None:
    """The pure lookup is what non-main DDP ranks call, so it must not
    race its siblings by creating anything."""
    assert find_specter_project_root(tmp_path) == tmp_path.resolve()
    assert not (tmp_path / PROJECT_MARKER).exists()


def test_pdb_cache_follows_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Honouring XDG is what lets an HPC user move the cache onto scratch
    for every XDG-aware tool at once, instead of per-tool."""
    from specter.config import default_pdb_cache_dir

    monkeypatch.delenv(PDB_CACHE_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "scratch-cache"))
    assert default_pdb_cache_dir() == str(
        tmp_path / "scratch-cache" / "specter" / "pdb"
    )


def test_specter_pdb_cache_beats_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(PDB_CACHE_ENV_VAR, str(tmp_path / "explicit"))
    from specter.config import default_pdb_cache_dir

    assert default_pdb_cache_dir() == str(tmp_path / "explicit")


def test_pdb_cache_env_var_overrides_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(PDB_CACHE_ENV_VAR, str(tmp_path / "elsewhere"))
    config = ParticleStackConfig(pdb_source="6bdf")
    assert config.pdb_cache_dir == str(tmp_path / "elsewhere")


def test_load_config_preserves_absolute_pdb_cache_dir(tmp_path: Path) -> None:
    absolute = str(tmp_path / "cache")
    path = _write_toml(
        tmp_path, f'[potential]\npdb_source = "6bdf"\npdb_cache_dir = "{absolute}"\n'
    )
    config = load_config(path)
    assert config.pdb_cache_dir == absolute


def test_particle_stack_config_requires_pdb_source() -> None:
    import pytest

    with pytest.raises(TypeError):
        ParticleStackConfig()  # type: ignore[call-arg]


def test_apply_overrides_sets_fields() -> None:
    config = ParticleStackConfig(pdb_source="6bdf")
    result = apply_overrides(config, {"n_particles": 500, "device": "cuda:0"})
    assert result is config
    assert config.n_particles == 500
    assert config.device == "cuda:0"


def test_apply_overrides_with_empty_dict_is_noop() -> None:
    config = ParticleStackConfig(pdb_source="6bdf")
    apply_overrides(config, {})
    assert config.n_particles == 20  # dataclass default, untouched


def test_particle_toml_loads_and_matches_expected_values() -> None:
    path = str(REPO_ROOT / "configs" / "particle.toml")
    config = load_config(path)
    assert config.pdb_source == "6bdf"
    assert config.n_pixels == 256
    assert config.pixel_size == 1.0
    assert config.scattering_model == "multislice"
    assert config.device == "cpu"


def test_particle_stack_config_advanced_field_defaults() -> None:
    """Defaults for the newly-exposed 'Advanced' fields should reproduce
    today's previously-hardcoded behavior (ice_parameterization is None, which
    resolves to potential_parameterization -- see
    test_ice_parameterization_defaults_to_following_the_structure)."""
    config = ParticleStackConfig(pdb_source="6bdf")
    assert config.potential_parameterization == "shtyrov"
    assert config.potential_method == "analytic"
    assert config.rcut is None
    assert config.conv_backend == "fftconvolve"
    assert config.periodic is False
    assert config.atom_species is None
    assert config.ews_curvature_sign == "positive"
    assert config.klim is None
    assert config.rotate_mode == "real"
    assert config.ice_parameterization is None  # follows potential_parameterization
    assert config.ice_relax_steps == 0
    assert config.crowd_chunk_size == 1
    assert config.crowd_max_distance_xy is None
    assert config.crowd_method == "3d"
    assert config.crowd_n_points is None
    assert config.crowd_seed == "origin"
    assert config.crowd_move_to_cpu is False
    assert config.water_air_interface is False
    assert config.seed is None
    assert config.astigmatism == 0.0
    assert config.astigmatism_angle == [0.0, 180.0]
    assert config.phaseshift == 0.0
    assert config.tiltx == 0.0
    assert config.tilty == 0.0
    assert config.trefoil1 == 0.0
    assert config.trefoil2 == 0.0
    assert (
        config.anisomag_m00,
        config.anisomag_m01,
        config.anisomag_m10,
        config.anisomag_m11,
    ) == (1.0, 0.0, 0.0, 1.0)


def test_particle_stack_config_falcon4i_is_valid_detector_model() -> None:
    config = ParticleStackConfig(pdb_source="6bdf", detector_model="falcon4i_300kv")
    assert config.detector_model == "falcon4i_300kv"


def test_particle_toml_loads_advanced_fields() -> None:
    """The bundled particle.toml's [advanced] block should parse and match
    the dataclass defaults it was written to mirror."""
    path = str(REPO_ROOT / "configs" / "particle.toml")
    config = load_config(path)
    assert config.potential_parameterization == "shtyrov"
    assert config.ice_parameterization is None  # follows potential_parameterization
    assert config.astigmatism == 0.0
    assert config.anisomag_m11 == 1.0


def test_load_config_tilt_series_parses_scalar_fields(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [specimen]
        voxel_size = 5.0

        [tilt_geometry]
        n_tilts = 21
        tilt_axis = "x"
        """,
    )
    config = load_config(path, TiltSeriesConfig)
    assert config.voxel_size == 5.0
    assert config.n_tilts == 21
    assert config.tilt_axis == "x"


def test_load_config_tilt_series_fills_defaults_for_missing_fields(
    tmp_path: Path,
) -> None:
    path = _write_toml(tmp_path, "[specimen]\nvoxel_size = 5.0\n")
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
    assert config.voxel_size == 2.0
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


def test_load_config_names_unknown_fields_and_renames(tmp_path):
    """A stale or mistyped table used to surface as a bare
    `TypeError: __init__() got an unexpected keyword argument`, which names
    the key but not what to write instead. `[[grid]]` in particular shipped
    commented-out in the canonical config before it became
    `[[carbon_film]]`, so a copied-and-uncommented block hits this."""
    from specter.config import TomogramConfig, load_config

    stale = tmp_path / "stale.toml"
    stale.write_text("[[grid]]\nhole_radius = 6000.0\n")
    with pytest.raises(ValueError, match=r"renamed to 'carbon_film'"):
        load_config(str(stale), TomogramConfig)

    typo = tmp_path / "typo.toml"
    typo.write_text("[specimen]\nv_siz = 5.0\n")
    with pytest.raises(ValueError, match="unknown TomogramConfig field"):
        load_config(str(typo), TomogramConfig)


def test_apply_overrides_rejects_unknown_field() -> None:
    """A misnamed override must fail loudly, not attach a field nothing reads."""
    config = ParticleStackConfig(pdb_source="1abc")
    with pytest.raises(ValueError, match="no such field"):
        apply_overrides(config, {"deltav_v": 1e-5})


def test_build_config_options_preserves_mixed_case_field_names() -> None:
    """
    Click must not lowercase a mixed-case field into a name no field matches.

    `--deltaV_V` used to arrive as "deltav_v", so apply_overrides set an
    attribute nothing reads and the flag silently did nothing.
    """
    from specter.cli._click_options import build_config_options

    names = {opt.name for opt in build_config_options(ParticleStackConfig)}
    assert "deltaV_V" in names
    assert "deltaI_I" in names
    assert "deltav_v" not in names


@pytest.mark.parametrize("config_cls", [MicrographConfig, TiltSeriesConfig])
def test_seed_is_configurable(config_cls: type) -> None:
    """Every generation command needs a seed to be reproducible at all."""
    assert "seed" in {f.name for f in dataclasses.fields(config_cls)}
    assert config_cls.seed is None


def test_ice_parameterization_defaults_to_following_the_structure() -> None:
    """
    Unset, the ice is modelled the same way as the structure it surrounds.

    The two potentials are summed into one volume, so a shtyrov protein sitting
    in kirkland ice is a choice worth making deliberately rather than
    inheriting from two independent defaults.
    """
    config = ParticleStackConfig(pdb_source="1abc")
    assert config.ice_parameterization is None

    for parameterization in ("shtyrov", "kirkland", "lobato"):
        config.potential_parameterization = parameterization  # type: ignore[assignment]
        resolved = config.ice_parameterization or config.potential_parameterization
        assert resolved == parameterization

    # An explicit value still wins, so deliberately differing stays possible.
    config.potential_parameterization = "shtyrov"  # type: ignore[assignment]
    config.ice_parameterization = "kirkland"  # type: ignore[assignment]
    assert (
        config.ice_parameterization or config.potential_parameterization
    ) == "kirkland"


def test_shipped_particle_config_leaves_ice_parameterization_unset() -> None:
    """configs/particle.toml must not pin it, or the following above is dead."""
    config = load_config(str(REPO_ROOT / "configs" / "particle.toml"))
    assert config.ice_parameterization is None


# --- validation -----------------------------------------------------------
# Every one of these used to reach the simulation. The numeric ones surfaced
# ~10 s later as ZeroDivisionError / "tensor with negative dimension" /
# "cannot convert float NaN to integer", naming nothing the user typed; the
# rest ran to completion and produced a plausible-looking, meaningless result.


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_pixels", 0),
        ("n_pixels", -32),
        ("pixel_size", 0.0),
        ("pixel_size", -1.0),
        ("n_particles", 0),
        ("n_particles", -5),
        ("voltage", 0.0),
        ("voltage", -300.0),
        ("dose", 0.0),
        ("dose", -20.0),
        ("alpha", -0.1),
        ("alpha", 1.5),
        ("cs", -2.0),
        ("ice_thickness", -100.0),
        ("batchsize", 0),
        ("batchsize", -4),
    ],
)
def test_validate_rejects_impossible_values(field: str, value: object) -> None:
    config = ParticleStackConfig(pdb_source="6bdf")
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        validate_config(config)


@pytest.mark.parametrize("field,value", [("dose", "60,20"), ("defocus", "15000,5000")])
def test_validate_rejects_reversed_ranges(field: str, value: str) -> None:
    """`--dose 60,20` sampled between 60 and 20, silently giving nothing like
    the intended range."""
    config = ParticleStackConfig(pdb_source="6bdf")
    setattr(config, field, value)
    with pytest.raises(ValueError, match="reversed"):
        validate_config(config)


def test_validate_rejects_invalid_literal_from_toml(tmp_path: Path) -> None:
    """
    Click validates a --flag against its Choice; a TOML file bypasses that.

    Nothing enforces a `Literal` at runtime, so `scattering_model = "banana"`
    in a config file used to sail through to the simulator.
    """
    path = _write_toml(
        tmp_path,
        """
        [models]
        pdb_source = "6bdf"
        scattering_model = "banana"
        """,
    )
    config = load_config(path)
    with pytest.raises(ValueError, match="scattering_model"):
        validate_config(config)


def test_validate_rejects_missing_input_files(tmp_path: Path) -> None:
    config = ParticleStackConfig(pdb_source="6bdf")
    config.cs_path = str(tmp_path / "nope.cs")
    with pytest.raises(ValueError, match="cs_path"):
        validate_config(config)


def test_validate_accepts_list_valued_fields() -> None:
    """
    Scalar bounds must skip fields holding a pair or a list of specs.

    bead_roughness is "one number, or a [low, high] pair", and `[0.0, 0.2] < 0`
    is a TypeError rather than a check -- which is exactly how the first
    version of this validation failed.
    """
    config = TomogramConfig()
    config.bead_roughness = [0.0, 0.2]
    validate_config(config)

    config.bead_roughness = [-0.5, 0.2]
    with pytest.raises(ValueError, match="bead_roughness"):
        validate_config(config)


def test_validate_accepts_every_shipped_config() -> None:
    """The configs specter ships must themselves pass validation."""
    for name, cls in (
        ("particle.toml", ParticleStackConfig),
        ("micrograph.toml", MicrographConfig),
        ("tilt_series.toml", TiltSeriesConfig),
        ("tomogram.toml", TomogramConfig),
    ):
        config = load_config(str(REPO_ROOT / "configs" / name), cls)
        validate_config(config)


# readd_hydrogens is `bool | Literal["auto"]` so a TOML can use the natural
# `true`/`false` as well as "auto". The union flattens to bool for the CLI flag,
# the same way batchsize's `int | Literal["auto"]` flattens to int.
@pytest.mark.parametrize(
    "written,expected", [('"auto"', "auto"), ("true", True), ("false", False)]
)
def test_readd_hydrogens_accepts_auto_and_booleans(tmp_path, written, expected):
    from specter.config import ParticleStackConfig, load_config

    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        f'[potential]\npdb_source = "1a6m"\nreadd_hydrogens = {written}\n'
    )
    cfg = load_config(str(cfg_file), ParticleStackConfig)
    assert cfg.readd_hydrogens == expected


@pytest.mark.parametrize("pipeline", ["particles", "micrograph"])
def test_hydrogen_settings_reach_pdb(monkeypatch, pipeline):
    """Both pipelines forward readd_hydrogens into PDB.

    The library location itself comes from $CLIBD_MON -- the mechanism the
    Monomer Library documents -- rather than a second config field saying the
    same thing.
    """
    import specter.pipelines._micrograph as micrograph_module
    import specter.pipelines._particles as particles_module
    from specter.config import MicrographConfig, ParticleStackConfig

    mod, cls, run = (
        (particles_module, ParticleStackConfig, "run_particle_stack")
        if pipeline == "particles"
        else (micrograph_module, MicrographConfig, "run_micrograph")
    )
    seen: dict = {}

    class _Stop(Exception):
        pass

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(mod, "PDB", _spy)
    cfg = cls(
        pdb_source=str(Path(__file__).parent / "test_data" / "1mbo.cif"),
        readd_hydrogens=False,
    )
    with pytest.raises(_Stop):
        getattr(mod, run)(cfg)

    assert seen["readd_hydrogens"] is False
    assert "monomer_library_path" not in seen, (
        "library location should come from $CLIBD_MON, not a config field"
    )
