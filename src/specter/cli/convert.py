"""`specter convert` -- translate particle metadata between packages.

Metadata only. No image stack is read, copied or rewritten: the written
``.star`` addresses the ``.mrcs`` files the input already references, so a
conversion costs no disk and cannot invalidate a stack.
"""

from __future__ import annotations

import os

import rich_click as click
from rich.console import Console

from ._click_options import CONTEXT_SETTINGS

console = Console()


def build_convert_group() -> click.RichGroup:
    """Build the `specter convert` command group."""

    @click.group(name="convert", context_settings=CONTEXT_SETTINGS)
    def convert() -> None:
        """Convert particle metadata between CryoSPARC and RELION.

        Only the metadata is converted. The output points at the same image
        stacks the input already referenced, so nothing is copied and the
        original files are left untouched.
        """

    @convert.command(name="cs2star", context_settings=CONTEXT_SETTINGS)
    @click.argument("csfile", type=click.Path(exists=True, dir_okay=False))
    @click.argument("starfile_path", type=click.Path(dir_okay=False))
    @click.option(
        "--passthrough",
        "-p",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="The job's *_passthrough_particles.cs, for jobs that split the "
        "image address and the pose/CTF across two files.",
    )
    @click.option(
        "--image-prefix",
        type=click.Path(file_okay=False),
        default=None,
        help="Prepended to each image path. CryoSPARC records them relative "
        "to the project directory, so pass that directory to get a .star "
        "readable from anywhere.",
    )
    @click.option(
        "--image-basename",
        is_flag=True,
        default=False,
        help="Write image paths as bare filenames. CryoSPARC's particle "
        "importer takes the stack directory separately and matches on "
        "filename, so use this when importing back into CryoSPARC.",
    )
    @click.option(
        "--overwrite",
        "-f",
        is_flag=True,
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    def cs2star(
        csfile: str,
        starfile_path: str,
        passthrough: str | None,
        image_prefix: str | None,
        image_basename: bool,
        overwrite: bool,
    ) -> None:
        """Convert a CryoSPARC particle .cs file to a RELION .star file.

        CSFILE must carry the `alignments3D/*` columns, i.e. come from a
        refinement rather than an extraction job. Jobs that split their
        output (restack among them) keep the pose and CTF in a separate
        `*_passthrough_particles.cs`. That file is found automatically when
        it sits in the same directory, and the two are joined on particle
        uid; pass --passthrough to name it yourself, which you must do if
        the job directory holds more than one.

        Image paths are written as CryoSPARC stored them, relative to the
        project directory. Pass --image-prefix <project dir> to make them
        absolute, so the .star can be read from anywhere.

        Writes one file holding the two data blocks RELION 3.1 expects,
        `optics` and `particles`. Pose, origins, defocus, astigmatism, phase
        shift, beam tilt, anisotropic magnification, half-set labels and the
        per-particle scale factor are all carried over. Trefoil and tetrafoil
        are dropped, with a warning naming them, because RELION expresses
        higher-order aberrations in a basis that depends on a chosen maximum
        resolution.
        """
        # Checked here rather than in the library call so that the guard is
        # the command line's, where a clobbered file is unrecoverable and
        # unasked for; the API keeps its more convenient default.
        if os.path.exists(starfile_path) and not overwrite:
            raise click.ClickException(
                f"{starfile_path} already exists. Pass --overwrite to replace it."
            )

        from specter.io import convert_csfile_to_starfile

        try:
            convert_csfile_to_starfile(
                csfile,
                starfile_path,
                passthrough_path=passthrough,
                image_prefix=image_prefix,
                image_basename=image_basename,
                overwrite=True,
            )
        except KeyError as exc:
            # A .cs missing the columns this needs is the user's input being
            # wrong, not a crash. args[0] avoids KeyError's repr quoting.
            raise click.ClickException(str(exc.args[0])) from exc
        console.print(f"  [green]✓[/green] {starfile_path}")

    return convert
