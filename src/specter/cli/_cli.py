"""Entry point for the `specter` CLI: `specter simulate particles ...`."""

from __future__ import annotations

import rich_click as click

from specter.jobs._cli import build_jobs_group

from .build import build_build_group
from .cache import build_cache_group
from .reconstruct import build_reconstruct_group
from .simulate import CONTEXT_SETTINGS, build_simulate_group


@click.group(cls=click.RichGroup, context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """SPECTER command-line interface."""


cli.add_command(build_simulate_group())
cli.add_command(build_build_group())
cli.add_command(build_reconstruct_group())
# The same group under the name the solver goes by everywhere else, so
# `specter ghostbuster particle` and `specter reconstruct particle` are one
# command. See build_reconstruct_group for why it is an alias inside `specter`
# rather than a `ghostbuster` console script of its own.
cli.add_command(build_reconstruct_group(name="ghostbuster"))
cli.add_command(build_jobs_group())
cli.add_command(build_cache_group())


def main() -> None:
    """Entry point for the `specter` CLI."""
    cli(prog_name="specter")


if __name__ == "__main__":
    main()
