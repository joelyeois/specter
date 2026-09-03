"""
`--help` puts the advanced flags last, and says nothing machine-specific.

Panel order is set by the order options are emitted, not by the order fields
happen to be declared on the config dataclass. Before `build_config_options`
ordered its output by the group list, a single advanced field declared early
(`MicrographConfig.pdb_cache_dir`, field #3) pulled the whole "Advanced"
panel to the top of `--help`, above every flag a first run actually sets.
"""

from __future__ import annotations

import pytest
import rich_click as click

from specter.cli._click_options import build_config_options
from specter.cli.build import _ICE_GROUPS, _TOMOGRAM_GROUPS
from specter.cli.match import _MATCH_PARTICLE_GROUPS
from specter.cli.reconstruct import _RECONSTRUCT_PARTICLE_GROUPS
from specter.cli.simulate import (
    _MICROGRAPH_GROUPS,
    _PARTICLE_STACK_GROUPS,
    _TILT_SERIES_GROUPS,
)
from specter.config import (
    IceCacheConfig,
    MatchConfig,
    MicrographConfig,
    ParticleStackConfig,
    ReconstructionConfig,
    TiltSeriesConfig,
    TomogramConfig,
)

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
    (MatchConfig, _MATCH_PARTICLE_GROUPS),
]


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
