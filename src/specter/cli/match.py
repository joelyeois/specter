"""`specter match` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import MATCH_HELP, MatchConfig

from ._click_options import (
    CONFIG_OPTION_HELP,
    build_config_options,
    collect_overrides,
    load_validated_config,
)
from .simulate import CONTEXT_SETTINGS

# (panel title, field names) for `specter match particles`, basic first.
# What a run is about (the refinement, its images, the structure), then the
# acquisition card no file carries, then how much to probe, then output.
_MATCH_PARTICLE_GROUPS: list[tuple[str, list[str]]] = [
    ("Inputs", ["metadata_path", "images_path", "pdb_source", "assembly"]),
    ("Acquisition", ["detector_model", "dose", "dose_rate", "energy_filter"]),
    ("Probing", ["n_probe", "n_battery", "write_stack"]),
    ("Compute", ["device", "seed"]),
    ("Output & job tracking", ["output_dir", "project", "job_id"]),
    ("Advanced", ["pdb_cache_dir", "monomer_library_path", "n_frames"]),
]


def _particles_callback(config: str | None, **_overrides_raw: object) -> None:
    """Handle `specter match particles`."""
    from specter.pipelines import run_match

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config"})
    cfg = load_validated_config(MatchConfig, config, overrides)
    run_match(cfg)


def _build_particles_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=None,
            show_default=False,
            help=CONFIG_OPTION_HELP,
            panel="Config",
        ),
        *build_config_options(
            MatchConfig, field_help=MATCH_HELP, field_groups=_MATCH_PARTICLE_GROUPS
        ),
    ]
    return click.RichCommand(
        name="particles",
        params=params,
        callback=_particles_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Derive a simulation config that matches a real particle set. Takes a "
        "refined particle set (.cs or .star), its images and the atomic model, "
        "checks that the poses reproduce the experimental views, sets the "
        "detector, coincidence and damage terms from the acquisition card, probes "
        "ice thickness and neighbour spacing against the images, and writes a "
        "matched.toml for `specter simulate particles` together with a report of "
        "how close the match is and what, if anything, no parameter can close.",
    )


def build_match_group() -> click.RichGroup:
    """Build the `match` command group and its subcommands."""
    group = click.RichGroup(
        name="match",
        help="Derive simulation settings that match an experimental dataset",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(_build_particles_command())
    return group
