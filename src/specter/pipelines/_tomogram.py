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

import dataclasses
import os
import time

from specter.config import TomogramConfig
from specter.specimen import (
    CRYOETSIM_PARTICLE_TABLE,
    PEI2016_CROWDING_TABLE,
    SpherePackingSpecimenGenerator,
    SphereProteinSpec,
    build_filler_pool_specs,
)

from ._common import _console, _format_elapsed, _section


def run_build_tomogram(config: TomogramConfig, n_tomograms: int = 1) -> None:
    """
    Pack one or more specimen volumes from PDB species via hard-sphere
    packing and save each as ``.mrc`` (+ copick-style ``.ndjson`` picks).

    Parameters
    ----------
    config : TomogramConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.targets``, ``config.filler``,
        ``config.filler_from_pei2016``, and ``config.filler_from_cryoetsim``
        can't all be empty/False.
    n_tomograms : int, optional
        Number of independent tomograms to generate, default 1. Each one
        beyond the first is written into its own numbered subdirectory of
        ``config.output_dir`` (``0001/``, ``0002/``, ...) so outputs don't
        collide -- pick files in particular are named after the species,
        not ``config.filename`` (see `SpherePackingSpecimenGenerator.
        export_picks`), so a shared flat output_dir would silently
        overwrite them across tomograms. If ``config.seed`` is set, each
        tomogram also gets its own seed (``config.seed``, ``config.seed +
        1``, ...) so runs are reproducible but distinct; if ``config.seed``
        is ``None``, each call simply draws from wherever the global RNG
        state already is, which likewise differs run to run.
    """
    if (
        not config.targets
        and not config.filler
        and not config.filler_from_pei2016
        and not config.filler_from_cryoetsim
    ):
        raise ValueError(
            "run_build_tomogram: config.targets, config.filler, "
            "config.filler_from_pei2016, and config.filler_from_cryoetsim "
            "can't all be empty/False -- at least one species source is "
            "required."
        )

    for i in range(n_tomograms):
        run_config = config
        if n_tomograms > 1:
            _section(f"Tomogram {i + 1}/{n_tomograms}")
            run_config = dataclasses.replace(
                config,
                output_dir=os.path.join(config.output_dir, f"{i + 1:04d}"),
                seed=None if config.seed is None else config.seed + i,
            )
        _run_single_tomogram(run_config)


def _run_single_tomogram(config: TomogramConfig) -> None:
    """Build and save exactly one tomogram from an already-resolved config."""
    t_start = time.perf_counter()

    _section("Building specimen volume")
    target_specs = [
        SphereProteinSpec(pdb_source=spec["pdb_source"], n_copies=int(spec["n_copies"]))
        for spec in config.targets
    ]
    filler_specs = [
        SphereProteinSpec(pdb_source=spec["pdb_source"]) for spec in config.filler
    ]
    if config.filler_from_pei2016:
        filler_specs += [
            SphereProteinSpec(pdb_source=d["pdb_source"])
            for d in build_filler_pool_specs(
                PEI2016_CROWDING_TABLE,
                max_mw_kda=config.filler_table_max_mw_kda,
                min_mw_kda=config.filler_table_min_mw_kda,
            )
        ]
    if config.filler_from_cryoetsim:
        filler_specs += [
            SphereProteinSpec(pdb_source=d["pdb_source"])
            for d in build_filler_pool_specs(
                CRYOETSIM_PARTICLE_TABLE,
                max_mw_kda=config.filler_table_max_mw_kda,
                min_mw_kda=config.filler_table_min_mw_kda,
                categories=config.filler_table_categories,
            )
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
