"""
`--help` puts the advanced flags last, and says nothing machine-specific.

Panel order is set by the order options are emitted, not by the order fields
happen to be declared on the config dataclass. Before `build_config_options`
ordered its output by the group list, a single advanced field declared early
(`MicrographConfig.pdb_cache_dir`, field #3) pulled the whole "Advanced"
panel to the top of `--help`, above every flag a first run actually sets.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import rich_click as click

from specter.cli._click_options import build_config_options
from specter.cli.build import _ICE_GROUPS, _TOMOGRAM_GROUPS
from specter.cli.reconstruct import _RECONSTRUCT_PARTICLE_GROUPS
from specter.cli.simulate import (
    _MICROGRAPH_GROUPS,
    _PARTICLE_STACK_GROUPS,
    _TILT_SERIES_GROUPS,
)
from specter.config import (
    IceCacheConfig,
    MicrographConfig,
    ParticleStackConfig,
    ReconstructionConfig,
    TiltSeriesConfig,
    TomogramConfig,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

#: (config dataclass, its CLI group list). `build ice` has no "Advanced"
#: panel: every one of its handful of flags is a setting a library build
#: genuinely has to choose.
GROUPED_CONFIGS = [
    (ParticleStackConfig, _PARTICLE_STACK_GROUPS),
    (MicrographConfig, _MICROGRAPH_GROUPS),
    (TiltSeriesConfig, _TILT_SERIES_GROUPS),
    (TomogramConfig, _TOMOGRAM_GROUPS),
    (IceCacheConfig, _ICE_GROUPS),
    (ReconstructionConfig, _RECONSTRUCT_PARTICLE_GROUPS),
]

#: Canonical config -> (its CLI group list, the tables it is expected to
#: declare, in order). Every canonical config is organised the same way as
#: the `--help` of the command it drives, so a user reading one already knows
#: where to look in the other.
CANONICAL_CONFIGS: dict[str, tuple[list, list[str]]] = {
    "particle.toml": (
        _PARTICLE_STACK_GROUPS,
        ["potential", "microscope", "sampling", "models"]
        + ["postprocessing", "compute", "output", "job", "advanced"],
    ),
    "micrograph.toml": (
        _MICROGRAPH_GROUPS,
        ["specimen", "microscope", "models", "dataset"]
        + ["compute", "output", "job", "advanced"],
    ),
    "tilt_series.toml": (
        _TILT_SERIES_GROUPS,
        ["specimen", "microscope", "defocus", "tilt_geometry", "models"]
        + ["postprocessing", "compute", "output", "job", "advanced"],
    ),
    # The specimen's contents are arrays of tables with no command-line
    # spelling and so no panel of their own; they sit where their generation
    # stage runs, between the Specimen and Picks panels' tables.
    "tomogram.toml": (
        _TOMOGRAM_GROUPS,
        ["specimen", "targets", "filler", "membrane"]
        + ["membrane_transmembrane_specs", "membrane_settings"]
        + ["filaments", "microtubules", "carbon_film", "beads"]
        + ["picks", "compute", "output", "job", "advanced"],
    ),
    "ice.toml": (_ICE_GROUPS, ["library", "optimisation", "compute", "output"]),
    "reconstruct.toml": (
        _RECONSTRUCT_PARTICLE_GROUPS,
        ["data", "optimisation", "symmetry", "sanity_check", "compute"]
        + ["job", "reference", "refinement", "advanced"],
    ),
}


@pytest.mark.parametrize(
    "config_cls,groups", GROUPED_CONFIGS, ids=lambda x: getattr(x, "__name__", "")
)
def test_every_flag_has_a_panel(config_cls: type, groups: list) -> None:
    """A field left out of the groups falls into rich-click's catch-all
    "Options" panel next to --help, where it reads as a CLI mechanism rather
    than a setting. build_config_options rejects that rather than rendering it."""
    options = build_config_options(config_cls, field_groups=groups)
    assert [o.name for o in options] == [n for _, names in groups for n in names]


@pytest.mark.parametrize(
    "config_cls,groups", GROUPED_CONFIGS, ids=lambda x: getattr(x, "__name__", "")
)
def test_advanced_panel_is_last(config_cls: type, groups: list) -> None:
    """Panels render in the order rich-click first sees them, so emission
    order is panel order."""
    titles = [title for title, _ in groups]
    if "Advanced" not in titles:
        pytest.skip(f"{config_cls.__name__} has no Advanced panel")
    assert titles[-1] == "Advanced"


@pytest.mark.parametrize("filename", CANONICAL_CONFIGS)
def test_toml_tables_mirror_help_panels(filename: str) -> None:
    """A canonical config declares the tables it is expected to, in order."""
    groups, expected = CANONICAL_CONFIGS[filename]
    with open(CONFIGS_DIR / filename, "rb") as f:
        assert list(tomllib.load(f)) == expected


@pytest.mark.parametrize("filename", CANONICAL_CONFIGS)
def test_toml_table_order_follows_panel_order(filename: str) -> None:
    """The rule behind the expected table lists above, checked against the
    group lists rather than against another hand-written constant: a table
    holds fields from at most one panel, and the tables run in the order
    --help prints those panels. A table setting no flag-bearing field at all
    (an array of tables like [[membrane]], or one whose every line is
    commented out) names no panel and is skipped."""
    groups, _ = CANONICAL_CONFIGS[filename]
    panel_of = {name: title for title, names in groups for name in names}
    with open(CONFIGS_DIR / filename, "rb") as f:
        raw = tomllib.load(f)

    sequence = []
    for table, contents in raw.items():
        if not isinstance(contents, dict):
            continue
        panels = {panel_of[key] for key in contents if key in panel_of}
        assert len(panels) <= 1, (
            f"{filename}: [{table}] mixes fields from panels {sorted(panels)}, "
            "so it mirrors no single one"
        )
        sequence += panels

    remaining = iter(title for title, _ in groups)
    assert all(panel in remaining for panel in sequence), (
        f"{filename}: tables run {sequence}, which is not in --help's panel "
        f"order {[title for title, _ in groups]}"
    )


@pytest.mark.parametrize("filename", CANONICAL_CONFIGS)
def test_advanced_table_holds_exactly_the_advanced_panel(filename: str) -> None:
    """What [advanced] promises is that everything above it is a decision a
    first run makes, so a field belongs in that table if and only if --help
    puts it in the Advanced panel. Only fields the config actually sets are
    checked: a commented-out line takes its default either way."""
    groups, _ = CANONICAL_CONFIGS[filename]
    with open(CONFIGS_DIR / filename, "rb") as f:
        raw = tomllib.load(f)
    advanced = {
        name for title, names in groups if title == "Advanced" for name in names
    }
    for table, contents in raw.items():
        if not isinstance(contents, dict):
            continue  # an array of tables, e.g. [[membrane]] -- TOML-only
        for key in contents:
            assert (key in advanced) == (table == "advanced"), (
                f"{filename}: {key!r} is set in [{table}] but "
                f"{'belongs in' if key in advanced else 'is not in'} "
                "the Advanced panel"
            )


def test_shown_defaults_are_not_machine_specific() -> None:
    """`pdb_cache_dir`'s real default is computed from $HOME, so rendering it
    verbatim printed one developer's absolute path into every --help and into
    any documentation copied from it. What --help shows is the rule instead."""
    options = build_config_options(MicrographConfig, field_groups=_MICROGRAPH_GROUPS)
    shown = {o.name: o.show_default for o in options if isinstance(o.show_default, str)}
    assert shown["pdb_cache_dir"] == "$SPECTER_PDB_CACHE, else ~/.cache/specter/pdb"
    assert not [v for v in shown.values() if v.startswith("/")]


def test_grouping_rejects_a_stale_field_name() -> None:
    """A group naming a field that no longer exists silently does nothing,
    which is how a rename leaves the renamed flag unpanelled."""
    groups = [
        (title, [*names, "pdb_code"] if title == "Specimen" else names)
        for title, names in _MICROGRAPH_GROUPS
    ]
    with pytest.raises(ValueError, match="not a settable field"):
        build_config_options(MicrographConfig, field_groups=groups)


def test_grouping_rejects_an_ungrouped_field() -> None:
    with pytest.raises(ValueError, match="not in any group"):
        build_config_options(
            MicrographConfig, field_groups=[("Specimen", ["pdb_source"])]
        )


def test_ungrouped_build_still_works() -> None:
    """`field_groups=None` keeps the old field-order, no-panel behaviour."""
    options = build_config_options(MicrographConfig)
    assert isinstance(options[0], click.RichOption)
    assert all(o.panel is None for o in options)
