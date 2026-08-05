"""Tomogram specimen-building pipeline: `TomogramConfig` in, .mrc + copick
.ndjson picks out.

Drives `specter.specimen.SpherePackingSpecimenGenerator` (two-stage hard-
sphere placement -- exact-count targets first, then equal-attempt-weight
filler avoiding them; see `specter.specimen.packing.pack_hard_spheres_3d`), saves
the assembled volume as .mrc -- directly usable as `specter simulate
tiltseries`'s `--volume_path` -- plus one copick-style .ndjson pick file per
species.
"""

from __future__ import annotations

import os
import time

from specter.config import TomogramConfig
from specter.specimen import SpherePackingSpecimenGenerator, SphereProteinSpec

from ._common import _console, _format_elapsed, _section


def run_build_tomogram(config: TomogramConfig) -> None:
    """
    Pack a specimen volume from PDB species via hard-sphere packing and save
    it as ``.mrc`` (+ copick-style ``.ndjson`` picks).

    Parameters
    ----------
    config : TomogramConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.targets`` and ``config.filler`` can't both be empty.
    """
    if not config.targets and not config.filler:
        raise ValueError(
            "run_build_tomogram: config.targets and config.filler can't "
            "both be empty -- at least one species is required."
        )

    t_start = time.perf_counter()

    _section("Building specimen volume")
    target_specs = [
        SphereProteinSpec(pdb_source=spec["pdb_source"], n_copies=int(spec["n_copies"]))
        for spec in config.targets
    ]
    filler_specs = [
        SphereProteinSpec(pdb_source=spec["pdb_source"]) for spec in config.filler
    ]
    gen = SpherePackingSpecimenGenerator(
        target_specs=target_specs,
        filler_specs=filler_specs,
        target_shape=tuple(config.target_shape),  # type: ignore[arg-type]
        v_size=config.v_size,
        filler_occupancy_fraction=config.filler_occupancy_fraction,
        gap_angstrom=config.gap_angstrom,
        clip_axes=tuple(config.clip_axes),  # type: ignore[arg-type]
        packing_method=config.packing_method,
        pad_fraction=config.pad_fraction,
        pdb_cache_dir=config.pdb_savefolder,
        seed=config.seed,
        device=config.device,
    )
    volume = gen.generate()
    size_angstrom = tuple(s * config.v_size for s in volume.shape)
    _console.print(
        f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X), "
        f"{size_angstrom[0]:.0f} x {size_angstrom[1]:.0f} x {size_angstrom[2]:.0f} A"
    )
    _console.print(f"  Targets: {gen.n_targets_placed}/{gen.n_target_requested} placed")
    _console.print(
        f"  Filler: {len(gen.placements) - gen.n_targets_placed} placed "
        f"from a {gen.n_filler_candidates}-candidate pool"
    )
    assert gen.instance_labels is not None
    occupancy_fraction = float((gen.instance_labels > 0).float().mean())
    _console.print(f"  Occupancy: {occupancy_fraction:.1%} of volume")

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
