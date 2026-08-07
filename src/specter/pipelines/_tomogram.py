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
  cytosol/lumen protein packing). This mode also accepts `config.filaments`/
  `config.actin` -- filament species (e.g. F-actin, microtubules) scattered
  through the volume via `specter.specimen.filament.place_filaments`,
  independent of the membrane/protein packing above.

Either way, saves the assembled volume as .mrc -- directly usable as
`specter simulate tiltseries`'s `--volume_path` -- plus one copick-style
.ndjson pick file per species.
"""

from __future__ import annotations

import dataclasses
import os
import time

import torch

from specter.config import TomogramConfig
from specter.specimen import (
    ACTIN_SPEC,
    CRYOETSIM_PARTICLE_TABLE,
    PEI2016_CROWDING_TABLE,
    FilamentSpec,
    MembraneGenerator,
    MembraneInstance,
    MembraneTomogramGenerator,
    SpherePackingSpecimenGenerator,
    SphereProteinSpec,
    TomogramProteinSpec,
    TransmembraneSpec,
    build_filler_pool_specs,
    render_transmembrane_template,
)
from specter.specimen._parallel_render import (
    build_templates_concurrently,
    resolve_render_devices,
    resolve_render_workers,
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
    filament_specs = [FilamentSpec(**d) for d in config.filaments]
    if config.actin:
        filament_specs.append(ACTIN_SPEC)

    # Pre-render every transmembrane species' template ONCE here, shared
    # across every [[membrane]] entry AND every one of its n_instances
    # copies below -- otherwise each instance's own MembraneGenerator would
    # redundantly rebuild the same handful of species from scratch (see
    # render_transmembrane_template's own docstring). Rendering itself can
    # run concurrently across config.render_workers/config.render_devices
    # (see TomogramConfig.render_workers's own docstring); attaching the
    # result via dataclasses.replace(..., template=...) makes every
    # downstream MembraneGenerator._build_template call a no-op cache hit
    # regardless of how many instances end up sharing it.
    if transmembrane_specs:
        devices = resolve_render_devices(config.device, config.render_devices)
        workers = resolve_render_workers(
            config.render_workers, len(transmembrane_specs)
        )
        built = build_templates_concurrently(
            keys=list(range(len(transmembrane_specs))),
            build_one=lambda i, device: render_transmembrane_template(
                transmembrane_specs[i], config.v_size, config.pdb_savefolder, device
            ),
            devices=devices,
            max_workers=workers,
        )
        transmembrane_specs = [
            dataclasses.replace(spec, template=built[i])
            for i, spec in enumerate(transmembrane_specs)
        ]

    membrane_instances = []
    for entry in config.membrane:
        # dict(entry) copies the OUTER dict before .pop() -- config.membrane[i]
        # itself is never mutated, so this stays safe even when the same
        # TomogramConfig (and its membrane list) is reused, un-deep-copied,
        # across multiple dataclasses.replace(...) calls in a --n_tomograms>1
        # run (see run_build_tomogram).
        instance_kwargs = dict(entry)
        n_instances = int(instance_kwargs.pop("n_instances", 1))

        position_xyz_raw = instance_kwargs.pop("position_xyz", None)
        if n_instances > 1 and position_xyz_raw is not None:
            raise ValueError(
                "run_build_tomogram: a [[membrane]] entry can't combine "
                f"n_instances={n_instances} with an explicit position_xyz -- "
                "every copy would want the same spot. Omit position_xyz "
                "(each instance is placed automatically, collision-checked "
                "against the others) or use n_instances=1 for manual "
                "placement."
            )
        # None (not (0,0,0)) by default -- MembraneTomogramGenerator resolves
        # an omitted position_xyz via collision-rejecting random placement
        # (see its own docstring); forcing every unspecified instance to the
        # literal origin, the old behaviour, defeats that entirely once more
        # than one instance is in play.
        position_xyz = tuple(position_xyz_raw) if position_xyz_raw is not None else None

        # None (not config.target_shape) by default -- MembraneGenerator
        # auto-sizes a small working grid from the organelle's own size when
        # omitted (see its own docstring), instead of every instance
        # rendering on a working grid the size of the WHOLE tomogram
        # canvas, the old behaviour. A [[membrane]] entry can still request
        # a specific target_shape_zyx explicitly (e.g. to match a scale
        # requirement of its own), popped here rather than left for
        # MembraneGenerator's **kwargs since it needs a tuple() cast like
        # config.target_shape gets below.
        target_shape_zyx_raw = instance_kwargs.pop("target_shape_zyx", None)
        target_shape_zyx = (
            tuple(target_shape_zyx_raw) if target_shape_zyx_raw is not None else None
        )

        for i in range(n_instances):
            # Restarts from config.seed at i=0 for EVERY entry (not a
            # running counter across entries) -- editing/adding another
            # [[membrane]] entry then never perturbs an earlier entry's own
            # instances' random realizations. The tradeoff (two entries at
            # the same index CAN start from an identical seed if they also
            # share shape_backend/kwargs) only matters when seed is set at
            # all -- the default (seed=None) draws every instance
            # independently regardless, so this is purely an advanced-user,
            # reproducibility-mode consideration.
            instance_seed = None if config.seed is None else config.seed + i
            mgen = MembraneGenerator(
                target_shape_zyx=target_shape_zyx,  # type: ignore[arg-type]
                v_size=config.v_size,
                transmembrane_specs=list(transmembrane_specs),
                pdb_cache_dir=config.pdb_savefolder,
                device=config.device,
                seed=instance_seed,
                **instance_kwargs,
            )
            membrane_instances.append(
                MembraneInstance(generator=mgen, position_xyz=position_xyz)  # type: ignore[arg-type]
            )

    return MembraneTomogramGenerator(
        membrane_instances=membrane_instances,
        target_shape_zyx=tuple(config.target_shape),  # type: ignore[arg-type]
        v_size=config.v_size,
        protein_specs=protein_specs,
        filament_specs=filament_specs,
        occupancy_fraction=config.membrane_occupancy_fraction,
        gap_angstrom=config.gap_angstrom,
        region_density_threshold=config.membrane_region_density_threshold,
        region_max_passes=config.membrane_region_max_passes,
        min_transmembrane_spacing_a=config.membrane_min_transmembrane_spacing_a,
        pdb_cache_dir=config.pdb_savefolder,
        parameterization=config.membrane_parameterization,
        seed=config.seed,
        device=config.device,
        render_workers=config.render_workers,
        render_devices=config.render_devices,  # type: ignore[arg-type]
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
        if membrane_gen.filament_specs:
            _console.print(
                f"  Filaments: {len(membrane_gen.filament_instances)} "
                f"monomer instance(s) placed ({len(membrane_gen.filament_specs)} species)"
            )
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

    if config.write_segmentation:
        # The segmentation mask, not a coordinate file, is the intended
        # ground truth for membrane geometry (a membrane surface has no
        # single natural "position" the way a protein does) -- see
        # TomogramConfig.write_segmentation's own docstring. MRC has no
        # int32 mode (verified directly against mrcfile), so labels are
        # cast to uint16 (65535 headroom) -- uint8 was considered for the
        # 3-category region mask, but mrcfile silently upconverts uint8 to
        # the same on-disk uint16 mode anyway (no true unsigned-8-bit MRC
        # mode exists), so there's no actual size benefit to requesting it.
        def _write_label_mrc(suffix: str, labels: torch.Tensor, dtype: str) -> None:
            path = os.path.join(config.output_dir, config.filename + suffix)
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(labels.cpu().numpy().astype(dtype))
                mrc.voxel_size = config.v_size
            _console.print(f"  [green]✓[/green] {path}")

        assert gen.instance_labels is not None
        _write_label_mrc("_protein_labels.mrc", gen.instance_labels, "uint16")

        if config.membrane:
            assert membrane_gen.membrane_labels is not None
            _write_label_mrc(
                "_membrane_labels.mrc", membrane_gen.membrane_labels, "uint16"
            )

            assert membrane_gen.regions is not None
            regions_volume = torch.zeros_like(membrane_gen.membrane_labels)
            regions_volume[membrane_gen.regions["shell"]] = 1
            regions_volume[membrane_gen.regions["lumen"]] = 2
            _write_label_mrc("_regions.mrc", regions_volume, "uint16")

    elapsed = time.perf_counter() - t_start
    _console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(elapsed)}")
