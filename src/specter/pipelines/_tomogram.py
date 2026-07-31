"""Tomogram specimen-building pipeline: `TomogramConfig` in, .mrc + copick
.ndjson picks out.

Drives `specter.specimen.SpherePackingSpecimenGenerator` (hard-sphere RSA
placement, see `specter.crowding.pack_hard_spheres_3d`), saves the assembled
volume as .mrc -- directly usable as `specter simulate tiltseries`'s
`--volume_path` -- plus one copick-style .ndjson pick file per species.
"""

from __future__ import annotations

import os
import time

from specter.config import TomogramConfig
from specter.specimen import SpherePackingSpecimenGenerator, SphereProteinSpec

from ._common import _console, _format_elapsed, _section


def run_build_tomogram(config: TomogramConfig) -> None:
    """
    Pack a specimen volume from PDB species via hard-sphere RSA and save it
    as ``.mrc`` (+ copick-style ``.ndjson`` picks).

    Parameters
    ----------
    config : TomogramConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.protein_specs`` must be non-empty.
    """
    if not config.protein_specs:
        raise ValueError(
            "run_build_tomogram: config.protein_specs must be non-empty -- "
            "at least one {'pdb_source': ...} species is required."
        )

    t_start = time.perf_counter()

    _section("Building specimen volume")
    protein_specs = [
        SphereProteinSpec(
            pdb_source=spec["pdb_source"], ratio=float(spec.get("ratio", 1.0))
        )
        for spec in config.protein_specs
    ]
    gen = SpherePackingSpecimenGenerator(
        protein_specs=protein_specs,
        target_shape=tuple(config.target_shape),  # type: ignore[arg-type]
        v_size=config.v_size,
        occupancy_fraction=config.occupancy_fraction,
        gap_angstrom=config.gap_angstrom,
        pdb_cache_dir=config.pdb_savefolder,
        parameterization=config.parameterization,
        seed=config.seed,
        device=config.device,
    )
    volume = gen.generate()
    _console.print(f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X)")
    _console.print(
        f"  Placed {len(gen.placements)}/{gen.n_candidates} candidate instances "
        f"across {len(protein_specs)} species"
    )

    _section("Saving")
    import mrcfile

    os.makedirs(config.output_dir, exist_ok=True)
    mrc_path = os.path.join(config.output_dir, config.filename + ".mrc")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(volume.numpy().astype("float32"))
        mrc.voxel_size = config.v_size
    _console.print(f"  [green]✓[/green] {mrc_path}")

    if config.write_picks:
        written = gen.export_picks(
            config.output_dir, annotation_version=config.annotation_version
        )
        for path in written.values():
            _console.print(f"  [green]✓[/green] {path}")

    elapsed = time.perf_counter() - t_start
    _console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(elapsed)}")
