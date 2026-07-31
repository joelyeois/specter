"""`specter build` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import (
    TOMOGRAM_HELP,
    TomogramConfig,
    apply_overrides,
    load_config,
)

from ._click_options import build_config_options, collect_overrides

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# (panel title, field names) -- basic-first-advanced-last, same convention as
# cli/simulate.py's _PARTICLE_STACK_GROUPS/_TILT_SERIES_GROUPS. protein_specs
# itself is list[dict]-typed and skipped entirely by build_config_options
# (TOML-only) -- not listed here, same treatment as that module's own
# protein_specs/membrane_specs.
_TOMOGRAM_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Specimen",
        ["target_shape", "v_size", "occupancy_fraction", "gap_angstrom"],
    ),
    ("Models", ["parameterization"]),
    ("Picks", ["write_picks", "annotation_version"]),
    ("Compute", ["device"]),
    ("Output", ["output_dir", "filename"]),
    ("Advanced", ["pdb_savefolder", "seed"]),
]


def _default_tomogram_config_path() -> str:
    from specter.config import REPO_ROOT

    return str(REPO_ROOT / "configs" / "tomogram.toml")


def _field_panels(groups: list[tuple[str, list[str]]]) -> dict[str, str]:
    """Field name -> rich-click panel title, derived from a (title, field names) list."""
    return {name: title for title, names in groups for name in names}


def _tomogram_callback(config: str, **_overrides_raw: object) -> None:
    """Handle `specter build tomogram`."""
    from specter.pipelines import run_build_tomogram

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config"})

    cfg = load_config(config, TomogramConfig)
    apply_overrides(cfg, overrides)
    run_build_tomogram(cfg)


def _build_tomogram_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=_default_tomogram_config_path(),
            show_default=True,
            help="TOML config file. Always loaded first, before any flags below "
            "are applied.",
            panel="Config",
        ),
        *build_config_options(
            TomogramConfig,
            field_help=TOMOGRAM_HELP,
            field_panels=_field_panels(_TOMOGRAM_GROUPS),
        ),
    ]
    return click.RichCommand(
        name="tomogram",
        params=params,
        callback=_tomogram_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Pack a specimen volume from PDB species via hard-sphere RSA and "
        "save it as .mrc (usable as `specter simulate tiltseries`'s "
        "--volume_path) plus copick-style .ndjson picks. A TOML config "
        "(--config) is always loaded first -- every flag below is optional "
        "and, if given, overrides one field of it.",
    )


def build_build_group() -> click.RichGroup:
    """Build the `build` command group and its subcommands."""
    group = click.RichGroup(
        name="build",
        help="Build specimen volumes",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(_build_tomogram_command())
    return group
