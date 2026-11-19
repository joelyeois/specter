"""`specter align` -- bring a map onto its symmetry axes, or onto another map.

Both commands read an MRC volume, find one rotation and one shift, and write
the volume resampled in the new frame. Neither takes a config: the two
searches have no setting a run needs to choose beyond the symmetry.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import rich_click as click

from specter.progress import console

from ._click_options import CONTEXT_SETTINGS

if TYPE_CHECKING:
    import torch

    from specter.alignment import Alignment


def _refuse_to_overwrite(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise click.ClickException(
            f"{path} already exists. Pass --overwrite to replace it."
        )


def _read_cubic(path: str, device: str) -> tuple[torch.Tensor, float]:
    """Read a volume onto the device, refusing anything but a cube."""
    from specter.io import read_mrc

    volume, voxel_size = read_mrc(path)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise click.ClickException(
            f"{path} has shape {tuple(volume.shape)}; a cubic volume is needed."
        )
    return volume.to(device), voxel_size


def _report(result: Alignment, voxel_size: float, output: str) -> None:
    """Print the pose, how well it is supported, and where the map went."""
    rows = "\n".join(
        "    " + "  ".join(f"{v: .6f}" for v in row) for row in result.rotation.tolist()
    )
    centre = result.centre.tolist()
    console.print("  rotation (aligned x, y, z -> input x, y, z):")
    console.print(rows, highlight=False)
    console.print(
        "  centre (x, y, z): "
        + "  ".join(f"{v:.3f}" for v in centre)
        + " voxels = "
        + "  ".join(f"{v * voxel_size:.3f}" for v in centre)
        + " Å",
        highlight=False,
    )
    console.print(
        f"  score {result.score:.4f}, significance {result.significance:.1f}, "
        f"to {result.radius:.0f} Fourier voxels",
        highlight=False,
    )
    mark = "[green]✓[/green]" if result.reliable else "[yellow]![/yellow]"
    console.print(f"  {mark} {output}")


_DEVICE_OPTION = click.option(
    "--device",
    default="cuda",
    show_default=True,
    help="Device the search runs on. Falls back to the CPU when CUDA is "
    "requested and absent.",
)
_OVERWRITE_OPTION = click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite the output file if it already exists.",
)


def build_align_group() -> click.RichGroup:
    """Build the `specter align` command group."""

    @click.group(name="align", context_settings=CONTEXT_SETTINGS)
    def align() -> None:
        """Align a map to its symmetry axes, or to another map.

        Each command finds one rotation and one shift, prints them, and
        writes the input resampled in the new frame without interpolation.
        """

    @align.command(name="symmetry", context_settings=CONTEXT_SETTINGS)
    @click.argument("volume", type=click.Path(exists=True, dir_okay=False))
    @click.argument("output", type=click.Path(dir_okay=False))
    @click.option(
        "--sym",
        required=True,
        help="Point group of the map: Cn, Dn, T, O or I.",
    )
    @_DEVICE_OPTION
    @_OVERWRITE_OPTION
    def symmetry(
        volume: str, output: str, sym: str, device: str, overwrite: bool
    ) -> None:
        """Align a map to the axes of its point-group symmetry.

        VOLUME is a cubic MRC map with the symmetry --sym about axes that
        need not pass through the box centre or lie along the box axes.
        OUTPUT is the same map with its symmetry axes in the standard
        positions and its symmetry centre on the central voxel, so that it
        can be symmetrised, or compared with another map aligned the same
        way.

        Of the equivalent standard frames, the one that turns the map least
        is taken. For a cyclic group the spin about the axis and the
        position along it are left as they were.

        The significance printed with the pose is the agreement between the
        symmetry copies at resolutions the search had not used, in standard
        deviations. Below 4 the output is marked with a warning: the map is
        then too noisy to show the symmetry, and the pose may be wrong. A
        clean map of a different symmetry is not caught this way; it shows
        as a score well below 1.
        """
        _refuse_to_overwrite(output, overwrite)

        from specter.alignment import align_to_symmetry
        from specter.devices import resolve_available_device
        from specter.io import write_mrc

        data, voxel_size = _read_cubic(volume, resolve_available_device(device))
        try:
            with warnings.catch_warnings():
                # The report below marks an unreliable result itself.
                warnings.simplefilter("ignore", UserWarning)
                result = align_to_symmetry(data, sym)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        assert result.volume is not None  # resample=True
        write_mrc(output, result.volume, voxel_size)
        _report(result, voxel_size, output)

    @align.command(name="volumes", context_settings=CONTEXT_SETTINGS)
    @click.argument("volume", type=click.Path(exists=True, dir_okay=False))
    @click.argument("reference", type=click.Path(exists=True, dir_okay=False))
    @click.argument("output", type=click.Path(dir_okay=False))
    @_DEVICE_OPTION
    @_OVERWRITE_OPTION
    def volumes(
        volume: str, reference: str, output: str, device: str, overwrite: bool
    ) -> None:
        """Align one map to another by a rotation and a shift.

        VOLUME is moved onto REFERENCE and written to OUTPUT on the
        reference's grid; REFERENCE is left as it is. The two must be cubic
        MRC maps of the same box and voxel size. They may differ in scale,
        offset and noise, as two independent reconstructions of one
        specimen do.

        The significance printed with the pose is the correlation of the
        two maps at resolutions the search had not used, in standard
        deviations. Below 4 the output is marked with a warning: the maps
        are then too noisy, or too different, for the pose to be trusted.
        Maps of opposite hand are not recognised.
        """
        _refuse_to_overwrite(output, overwrite)

        from specter.alignment import align_volumes
        from specter.devices import resolve_available_device
        from specter.io import write_mrc

        device = resolve_available_device(device)
        data, voxel_size = _read_cubic(volume, device)
        target, target_voxel_size = _read_cubic(reference, device)
        if data.shape != target.shape:
            raise click.ClickException(
                f"{volume} has box {data.shape[-1]} and {reference} has box "
                f"{target.shape[-1]}; resample one onto the other's grid first."
            )
        if abs(voxel_size - target_voxel_size) > 1e-4 * target_voxel_size:
            raise click.ClickException(
                f"{volume} has voxel size {voxel_size:g} Å and {reference} "
                f"{target_voxel_size:g} Å; resample one onto the other's grid first."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = align_volumes(data, target)
        assert result.volume is not None  # resample=True
        write_mrc(output, result.volume, target_voxel_size)
        _report(result, target_voxel_size, output)

    return align
