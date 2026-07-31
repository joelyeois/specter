"""Entry point for the `specter` CLI: `specter simulate particles ...`."""

from __future__ import annotations

import rich_click as click

from .build import build_build_group
from .simulate import CONTEXT_SETTINGS, build_simulate_group


@click.group(cls=click.RichGroup, context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """SPECTER command-line interface."""


cli.add_command(build_simulate_group())
cli.add_command(build_build_group())


def main() -> None:
    """Entry point for the `specter` CLI."""
    cli(prog_name="specter")


if __name__ == "__main__":
    main()
