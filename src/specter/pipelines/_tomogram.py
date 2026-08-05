"""Tomogram specimen-building pipeline: `TomogramConfig` in, .mrc + copick
.ndjson picks out.

Two mutually exclusive modes, selected by whether `config.membrane` is set:

- Sphere-packing mode (default): drives `specter.specimen.
  SpherePackingSpecimenGenerator` (two-stage hard-sphere placement --
  exact-count targets first, then equal-attempt-weight filler avoiding
  them; see `specter.specimen.packing.pack_hard_spheres_3d`).
- Membrane mode (`config.membrane` set): drives `specter.specimen.
  MembraneGenerator` (organic membrane shape + transmembrane placement)
  composed with `specter.specimen.MembraneTomogramGenerator` (region-gated
  cytosol/lumen protein packing).

Either way, saves the assembled volume as .mrc -- directly usable as
`specter simulate tiltseries`'s `--volume_path` -- plus one copick-style
.ndjson pick file per species.
"""

from __future__ import annotations

import dataclasses
import os
import time

from specter.config import TomogramConfig
from specter.specimen import (
    CRYOETSIM_PARTICLE_TABLE,
    PEI2016_CROWDING_TABLE,
    MembraneGenerator,
    MembraneTomogramGenerator,
    SpherePackingSpecimenGenerator,
    SphereProteinSpec,
    TomogramProteinSpec,
    TransmembraneSpec,
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
    has_sphere_sources = bool(
        config.targets
        or config.filler
        or config.filler_from_pei2016
        or config.filler_from_cryoetsim
    )
    has_membrane = bool(config.membrane)
    if not has_sphere_sources and not has_membrane:
        raise ValueError(
            "run_build_tomogram: config.targets, config.filler, "
            "config.filler_from_pei2016, config.filler_from_cryoetsim, and "
            "config.membrane can't all be empty/False -- at least one "
            "species source is required."
        )
    if has_sphere_sources and has_membrane:
        raise ValueError(
            "run_build_tomogram: config.membrane can't be combined with "
            "targets/filler/filler_from_pei2016/filler_from_cryoetsim in "
            "v1 -- MembraneTomogramGenerator's own region-gated RSA packing "
            "and SpherePackingSpecimenGenerator's two-stage target/filler "
            "packing are two independent backends, not reconciled into one "
            "call yet."
        )
    if len(config.membrane) > 1:
        raise ValueError(
            "run_build_tomogram: config.membrane must have at most one "
            f"[[membrane]] entry (v1 single-instance scope), got "
            f"{len(config.membrane)}."
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


def _build_sphere_packing_generator(
    config: TomogramConfig,
) -> SpherePackingSpecimenGenerator:
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
    return SpherePackingSpecimenGenerator(
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


def _build_membrane_tomogram_generator(
    config: TomogramConfig,
) -> MembraneTomogramGenerator:
    membrane_kwargs = dict(config.membrane[0])
    parameterization = membrane_kwargs.pop("parameterization", "shtyrov")
    transmembrane_specs = [
        TransmembraneSpec(
            pdb_source=d["pdb_source"],
            frequency=int(d.get("frequency", 1)),
            parameterization=d.get("parameterization", "shtyrov"),
        )
        for d in config.membrane_transmembrane_specs
    ]
    protein_specs = [
        TomogramProteinSpec(
            pdb_source=d["pdb_source"],
            location=d.get("location", "cytosol"),
            ratio=float(d.get("ratio", 1.0)),
        )
        for d in config.membrane_protein_specs
    ]
    mgen = MembraneGenerator(
        target_shape_zyx=tuple(config.target_shape),  # type: ignore[arg-type]
        v_size=config.v_size,
        transmembrane_specs=transmembrane_specs,
        pdb_cache_dir=config.pdb_savefolder,
        device=config.device,
        seed=config.seed,
        **membrane_kwargs,
    )
    return MembraneTomogramGenerator(
        membrane_generator=mgen,
        protein_specs=protein_specs,
        occupancy_fraction=config.membrane_occupancy_fraction,
        gap_angstrom=config.gap_angstrom,
        region_density_threshold=config.membrane_region_density_threshold,
        region_max_passes=config.membrane_region_max_passes,
        min_transmembrane_spacing_a=config.membrane_min_transmembrane_spacing_a,
        pdb_cache_dir=config.pdb_savefolder,
        parameterization=parameterization,
        seed=config.seed,
        device=config.device,
    )


def _run_single_tomogram(config: TomogramConfig) -> None:
    """Build and save exactly one tomogram from an already-resolved config."""
    t_start = time.perf_counter()

    _section("Building specimen volume")
    gen: SpherePackingSpecimenGenerator | MembraneTomogramGenerator
    if config.membrane:
        membrane_gen = _build_membrane_tomogram_generator(config)
        volume = membrane_gen.generate()
        size_angstrom = tuple(s * config.v_size for s in volume.shape)
        _console.print(
            f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X), "
            f"{size_angstrom[0]:.0f} x {size_angstrom[1]:.0f} x {size_angstrom[2]:.0f} A"
        )
        _console.print(
            f"  Transmembrane: {len(membrane_gen.transmembrane_placements)} placed"
        )
        for location in ("cytosol", "lumen"):
            n_here = sum(1 for p in membrane_gen.placements if p.location == location)
            _console.print(f"  {location.capitalize()}: {n_here} placed")
        assert membrane_gen.instance_labels is not None
        occupancy_fraction = float((membrane_gen.instance_labels > 0).float().mean())
        _console.print(f"  Occupancy: {occupancy_fraction:.1%} of volume")
        gen = membrane_gen
    else:
        sphere_gen = _build_sphere_packing_generator(config)
        volume = sphere_gen.generate()
        size_angstrom = tuple(s * config.v_size for s in volume.shape)
        _console.print(
            f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X), "
            f"{size_angstrom[0]:.0f} x {size_angstrom[1]:.0f} x {size_angstrom[2]:.0f} A"
        )
        _console.print(
            f"  Targets: {sphere_gen.n_targets_placed}/{sphere_gen.n_target_requested} placed"
        )
        _console.print(
            f"  Filler: {len(sphere_gen.placements) - sphere_gen.n_targets_placed} "
            f"placed from a {sphere_gen.n_filler_candidates}-candidate pool"
        )
        assert sphere_gen.instance_labels is not None
        occupancy_fraction = float((sphere_gen.instance_labels > 0).float().mean())
        _console.print(f"  Occupancy: {occupancy_fraction:.1%} of volume")
        gen = sphere_gen

    _section("Saving")
    import mrcfile

    os.makedirs(config.output_dir, exist_ok=True)
    mrc_path = os.path.join(config.output_dir, config.filename + ".mrc")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(volume.cpu().numpy().astype("float32"))
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
