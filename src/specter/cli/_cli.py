"""Entry point for the `specter` CLI: `specter simulate particles ...`."""

from __future__ import annotations

import os

# Set before anything below imports torch, since the CUDA caching allocator
# reads this when it initialises.
#
# Without it the allocator's *reserved* footprint runs 1.3-1.6x its *allocated*
# footprint on the particle pipeline, and reserved is what actually has to fit
# on the card. Measured at box 256 / pad 512, batch 30: 28.7 GB allocated but
# 45.9 GB reserved, which is what made `simulate particles --n_particles 500`
# die on its second batch with ~19 GB "reserved but unallocated" free. The same
# batch reserves 30.8 GB with expandable segments -- 15 GB back, for free.
#
# `setdefault`, so anyone who has tuned `PYTORCH_CUDA_ALLOC_CONF` themselves
# keeps their setting. Done here rather than in `specter/__init__.py` because
# the CLI owns its process; a library import has no business reconfiguring the
# allocator of an application that merely imports specter.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import rich_click as click  # noqa: E402

from specter.jobs._cli import build_jobs_group  # noqa: E402

from .build import build_build_group  # noqa: E402
from .cache import build_cache_group  # noqa: E402
from .reconstruct import build_reconstruct_group  # noqa: E402
from .simulate import CONTEXT_SETTINGS, build_simulate_group  # noqa: E402


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
