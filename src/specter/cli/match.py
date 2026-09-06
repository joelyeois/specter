"""`specter match` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import MATCH_HELP, MatchConfig

from ._click_options import CONTEXT_SETTINGS, build_config_command, load_cli_config

# (panel title, field names) for `specter match particles`, basic first.
# What a run is about (the refinement, its images, the structure), then the
# acquisition card no file carries, then how much to probe, then output.
_MATCH_PARTICLE_GROUPS: list[tuple[str, list[str]]] = [
    ("Inputs", ["metadata_path", "images_path", "pdb_source", "assembly"]),
    ("Acquisition", ["detector_model", "dose", "dose_rate", "energy_filter"]),
    ("Probing", ["n_probe", "n_battery", "probe_bin", "write_stack"]),
    ("Compute", ["device", "probe_workers", "seed"]),
    ("Output & job tracking", ["output_dir", "project", "job_id"]),
    ("Advanced", ["pdb_cache_dir", "monomer_library_path", "n_frames"]),
]


def _particles_callback(config: str | None, **_overrides_raw: object) -> None:
    """Handle `specter match particles`."""
    from specter.pipelines import run_match

    report = run_match(load_cli_config(MatchConfig, config))
    if not report.pose.passed:
        # A shell chain (`specter match ... && specter simulate ...`) must not
        # go on to simulate from the INCOMPLETE matched.toml this run wrote.
        raise click.ClickException(
            "pose-alignment check failed; matched.toml is marked incomplete. "
            + report.warnings[0]
        )


def build_match_group() -> click.RichGroup:
    """Build the `match` command group and its subcommands."""
    group = click.RichGroup(
        name="match",
        help="Derive simulation settings that match an experimental dataset",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(
        build_config_command(
            "particles",
            MatchConfig,
            MATCH_HELP,
            _MATCH_PARTICLE_GROUPS,
            _particles_callback,
            help="Derive a simulation config that matches a real particle set. Takes a "
            "refined particle set (.cs or .star), its images and the atomic model, "
            "checks that the poses reproduce the experimental views, sets the "
            "detector, coincidence and damage terms from the acquisition card, probes "
            "ice thickness and neighbour spacing against the images, and writes a "
            "matched.toml for `specter simulate particles` together with a report of "
            "how close the match is and what, if anything, no parameter can close.",
        )
    )
    return group
