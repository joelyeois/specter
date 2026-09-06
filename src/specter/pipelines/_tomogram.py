"""Tomogram specimen-building pipeline: `TomogramConfig` in, .mrc + copick
.ndjson picks out.

Drives `specter.specimen.tomogram.TomogramSpecimenGenerator` -- the ONE
generator behind `specter build tomogram`. An optional composited organic
membrane (`config.membrane`, `specter.specimen.MembraneGenerator` per
instance), optional scattered filament species (`config.filaments`/
`config.actin`, `specter.specimen.filament.place_filaments`), optional gold
fiducial beads (`config.beads`) and an optional carbon support film
(`config.carbon_film`), and densely packed protein species (`config.targets`/
`config.filler`, region-gated to `location: "cytosol"|"lumen"` when a
membrane is present -- otherwise the whole volume is one cytosol region).
Generation order is carbon film, then membranes, then filaments, then
beads, then protein fill; each stage avoids the previous ones' placements
(see `TomogramSpecimenGenerator`'s own module docstring).

Any combination of membrane/filaments/beads/grid/targets/filler is valid
as long as at least one is non-empty -- there's no separate non-membrane
"sphere-packing" mode/generator to choose between anymore.

Saves the assembled volume as .mrc -- directly usable as `specter simulate
tiltseries`'s `--volume_path` -- plus one copick-style .ndjson pick file
per species.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from specter.arrays import count_nonzero_chunked
from specter.config import TomogramConfig, validate_config
from specter.specimen import (
    ACTIN_SPEC,
    CRYOETSIM_PARTICLE_TABLE,
    PEI2016_CROWDING_TABLE,
    FilamentSpec,
    MicrotubuleSpec,
    CarbonFilmSpec,
    DEFAULT_SH_AXES_RANGE_A,
    DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_A,
    DEFAULT_SWEPT_TUBE_RADIUS_RANGE_A,
    MembraneGenerator,
    MembraneInstance,
    TomogramSpecimenGenerator,
    TomogramBeadSpec,
    TomogramProteinSpec,
    TransmembraneSpec,
    build_filler_pool_specs,
    render_transmembrane_template,
)
from specter.specimen._parallel_render import (
    build_templates_concurrently,
    parse_device_pool,
    resolve_render_devices,
    resolve_render_workers,
)

from specter.progress import console, format_elapsed, section
from ._common import (
    _deterministic_tracked_path,
    _tracked_output_dir,
    resolve_available_device,
    resolve_output_dir,
)


#: Largest label an MRC integer mode can hold. MRC's widest unsigned
#: integer mode is uint16 (mode 6); there is no int32 mode, verified
#: directly against mrcfile, which rejects the dtype outright. Past this,
#: `_write_label_mrc` escalates to float32 (mode 2), exact for integers to
#: 2**24 and so good for 256x this ceiling.
_MRC_UINT16_MAX = 65535


def tomogram_output_path(config: TomogramConfig) -> str:
    """
    The ``.mrc`` path `run_build_tomogram` writes for a given config.

    Deterministic from ``config.output_dir``/``config.filename`` (or, if
    tracked, ``output_dir``/``project``/``job_id``) alone, so callers
    that chain straight into `specter simulate tiltseries` (e.g.
    ``run_tilt_series(..., tomogram_config=...)``) can compute it without
    re-deriving the naming convention themselves, or waiting for
    `run_build_tomogram` to hand anything back.

    Parameters
    ----------
    config : TomogramConfig
        Fully-resolved run configuration.

    Returns
    -------
    str
        Path to the ``.mrc`` file this config's `run_build_tomogram` call
        writes (or would write, for ``n_tomograms=1``).

    Raises
    ------
    ValueError
        If ``config`` is tracked (``project`` or ``job_id`` set) but
        ``job_id`` itself is unpinned -- an auto-assigned id isn't knowable
        ahead of the run, which this function deliberately never queries
        the filesystem to find out. Pin ``job_id`` explicitly to call this
        before the run happens, e.g. for tomogram_config chaining.
    """
    if config.project is None and config.job_id is None:
        output_dir = resolve_output_dir(config, "tomograms")
    elif config.job_id is None:
        raise ValueError(
            "tomogram_output_path: config.job_id must be pinned explicitly "
            "when project is set -- an auto-assigned id can't be known "
            "before the run actually happens."
        )
    else:
        output_dir = _deterministic_tracked_path(config, "tomograms")
    return os.path.join(output_dir, config.filename + ".mrc")


#: Z-slices per slab in :func:`_write_volume_mrc`. Sized so one slab of a
#: 3000x3000 float32 tomogram is ~0.6 GB, small enough that the temporaries
#: the statistics pass builds stay off the memory profile entirely.
_MRC_WRITE_SLAB_Z = 16


def _write_volume_mrc(
    path: str,
    array: "np.ndarray",
    voxel_size: float,
) -> None:
    """
    Write a volume to MRC without letting mrcfile reduce it whole.

    ``mrc.set_data(...)`` calls mrcfile's ``update_header_stats``, which
    fills the header's ``dmin``/``dmax``/``dmean``/``rms`` fields by
    running ``ndarray.std()`` over the entire volume. numpy's variance
    materializes a full-size temporary to do it, so at the 2 A production
    grid (750x3000x3000, 27 GB) that one header field was profiled at
    262 s -- half the run's wall clock -- and its temporary is a
    substantial share of the peak RSS. The write itself is ~31 s.

    Memory-mapping the file and filling it a slab at a time writes the same
    bytes and accumulates the same four statistics, with nothing
    volume-sized allocated on top of the array already in hand.

    Parameters
    ----------
    path : str
        Destination file. Overwritten if it exists.
    array : numpy.ndarray
        Volume to write, shape ``(Z, Y, X)``, already in its final dtype.
    voxel_size : float
        Isotropic voxel size in Angstrom, written to the header.
    """
    import mrcfile
    import numpy as np

    nz = array.shape[0]
    total = float(array.size)
    vmin = np.inf
    vmax = -np.inf
    accum = 0.0
    with mrcfile.new_mmap(
        path,
        shape=array.shape,
        mrc_mode=mrcfile.utils.mode_from_dtype(array.dtype),
        overwrite=True,
    ) as mrc:
        for z0 in range(0, nz, _MRC_WRITE_SLAB_Z):
            slab = array[z0 : z0 + _MRC_WRITE_SLAB_Z]
            mrc.data[z0 : z0 + _MRC_WRITE_SLAB_Z] = slab
            vmin = min(vmin, float(slab.min()))
            vmax = max(vmax, float(slab.max()))
            accum += float(slab.sum(dtype=np.float64))
        mean = accum / total

        # Second pass for the RMS, deliberately, rather than deriving it
        # from a running sum of squares in the same pass: `rms` is the
        # deviation from the mean, and E[x^2] - mean^2 cancels
        # catastrophically for a volume whose values sit far from zero,
        # which a scattering potential in volts does.
        sq = 0.0
        for z0 in range(0, nz, _MRC_WRITE_SLAB_Z):
            slab = array[z0 : z0 + _MRC_WRITE_SLAB_Z]
            d = slab.astype(np.float64, copy=False) - mean
            sq += float(np.dot(d.reshape(-1), d.reshape(-1)))

        mrc.header.dmin = np.float32(vmin)
        mrc.header.dmax = np.float32(vmax)
        mrc.header.dmean = np.float32(mean)
        mrc.header.rms = np.float32(np.sqrt(sq / total))
        mrc.voxel_size = voxel_size


def run_build_tomogram(config: TomogramConfig, n_tomograms: int = 1) -> None:
    """
    Build one or more specimen tomogram volumes and save each as ``.mrc``
    (+ copick-style ``.ndjson`` picks).

    Parameters
    ----------
    config : TomogramConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`, or constructed directly in
        Python. ``config.targets``, ``config.filler``,
        ``config.filler_from_pei2016``, ``config.filler_from_cryoetsim``,
        ``config.membrane``, ``config.filaments``, ``config.actin``,
        ``config.carbon_film``, and ``config.beads`` can't all be empty/False --
        at least one species source is required.
    n_tomograms : int, optional
        Number of independent tomograms to generate, default 1. Each one
        beyond the first is written into its own numbered subdirectory of
        ``config.output_dir`` (``0001/``, ``0002/``, ...) so outputs don't
        collide -- pick files in particular are named after the species,
        not ``config.filename`` (see `TomogramSpecimenGenerator.
        export_picks`), so a shared flat output_dir would silently
        overwrite them across tomograms. If ``config.seed`` is set, each
        tomogram also gets its own seed (``config.seed``, ``config.seed +
        1``, ...) so runs are reproducible but distinct; if ``config.seed``
        is ``None``, each call simply draws from wherever the global RNG
        state already is, which likewise differs run to run.
    """
    # n_tomograms is a call argument rather than a config field, so
    # validate_config never sees it.
    if n_tomograms <= 0:
        raise ValueError(
            f"n_tomograms={n_tomograms} is invalid: must be greater than 0."
        )

    validate_config(config)

    if not config.has_any_species:
        fields = ", ".join(f"config.{name}" for name in config.SPECIES_SOURCE_FIELDS)
        raise ValueError(
            f"run_build_tomogram: {fields} can't all be empty/False -- at "
            "least one species source is required."
        )
    if len(config.carbon_film) > 1:
        raise ValueError(
            f"run_build_tomogram: config.carbon_film has "
            f"{len(config.carbon_film)} entries "
            "-- there is only one carbon film per tomogram, at most one "
            "[[carbon_film]] table is allowed."
        )
    with _tracked_output_dir(config, "tomograms") as base_output_dir:
        for i in range(n_tomograms):
            if n_tomograms > 1:
                section(f"Tomogram {i + 1}/{n_tomograms}")
            # Tracking (if any) is resolved once, above, into base_output_dir --
            # run_config always gets a plain, already-resolved output_dir and
            # cleared tracking fields, so _run_single_tomogram/tomogram_output_path
            # never need to know whether this call was tracked.
            run_config = dataclasses.replace(
                config,
                output_dir=(
                    base_output_dir
                    if n_tomograms == 1
                    else os.path.join(base_output_dir, f"{i + 1:04d}")
                ),
                project=None,
                job_id=None,
                seed=None if config.seed is None else config.seed + i,
            )
            _run_single_tomogram(run_config)


def _protein_specs_from_dicts(
    dicts: list[dict], *, default_ratio: float = 1.0
) -> list[TomogramProteinSpec]:
    """Convert `config.targets`/`config.filler`-style dicts into
    `TomogramProteinSpec`s. A dict with `n_copies` becomes an exact-count
    ("target") spec; one without becomes a ratio-weighted ("filler") spec
    -- `location` is optional on either (default "cytosol")."""
    specs = []
    for d in dicts:
        n_copies = d.get("n_copies")
        specs.append(
            TomogramProteinSpec(
                pdb_source=d["pdb_source"],
                location=d.get("location", "cytosol"),
                ratio=float(d.get("ratio", default_ratio)),
                n_copies=int(n_copies) if n_copies is not None else None,
            )
        )
    return specs


# MembraneGenerator's own default size-draw ranges, read rather than retyped:
# two copies could drift with nothing failing, and only for a membrane whose
# size the user left entirely unspecified -- the quietest possible way to be
# wrong. Not the zero-cross-generator-coupling convention the earlier copy
# cited: that is about generators importing each other, and this module
# already imports MembraneGenerator to construct it. Used only as the UPPER
# bound to cap against below; the biology-motivated LOWER bounds are never
# shrunk.

# Fraction of the tomogram box's own LIMITING axis extent an auto-sized
# (target_shape=None) [[membrane]] instance's DIAMETER (2x bounding
# radius) is allowed to reach. "Limiting"
# = the smallest extent among NON-clippable axes (pack_hard_spheres_3d
# requires the FULL sphere to fit there -- see config.clip_axes), or, if
# every axis is clippable (nothing requires a full fit anywhere), the
# LARGEST extent instead, purely as a "reasonable size" reference rather
# than a geometric requirement. MembraneGenerator has no way to know the
# outer tomogram box it'll be composited into unless told -- left
# uncapped, its own default draw ranges are pure biology-motivated
# "realistic organelle size" guesses with no upper limit relative to any
# particular box, so a real box with a thin non-clippable axis (e.g. Z
# much smaller than X/Y, as any real tomogram-shaped canvas is, if Z isn't
# also made clippable) can end up drawing an organelle whose bounding
# sphere literally cannot fit that axis at all -- confirmed directly:
# swept_spline's default range alone can reach a diameter of ~3300 A, far
# exceeding a 2000 A Z-extent even before accounting for gap or
# multiple co-existing instances.
#
# 0.75 is close to the true ceiling for a NON-clippable limiting axis, not
# a conservative default -- diameter can never reach 1.0 (the full axis)
# regardless of this fraction there: at exactly 1.0 an instance's center
# would have exactly ONE valid position on that axis (dead center, zero
# slack), which pack_hard_spheres_3d's random sampling essentially never
# finds and gap alone would already violate. Multiple co-existing
# instances need even more slack than one. For the all-clippable fallback
# (limiting axis = largest, no hard ceiling exists at all), 0.75 is just
# reused as a reasonable default rather than re-deriving a second number.
# Raise this only with the non-clippable case's real ceiling in mind, not
# just "bigger is better".
_MEMBRANE_AUTO_SIZE_BOX_FRACTION = 0.75


def _cap_membrane_auto_size_ranges(
    instance_kwargs: dict,
    target_shape: tuple[int, int, int] | None,
    config: TomogramConfig,
) -> dict:
    """
    Return `instance_kwargs` with a size-range override added, IF this
    entry's own (already-resolved) `target_shape` is None AND every
    size-controlling key for its `shape_backend` is unset in
    `instance_kwargs` -- i.e. only when nothing already constrains how
    big the auto-drawn organelle can get. An entry that sets ANY of those
    explicitly is trusted as-is and returned unchanged (never second-
    guessed here).

    Caps the draw so DIAMETER (2x bounding radius) never exceeds
    `_MEMBRANE_AUTO_SIZE_BOX_FRACTION` of the tomogram box's own LIMITING
    axis extent (see that constant's own comment for exactly what that
    means and why), computed from `config.target_shape`/`config.voxel_size`/
    `config.clip_axes`.
    """
    if target_shape is not None:
        return instance_kwargs
    shape_backend = instance_kwargs.get("shape_backend", "spherical_harmonics")
    box_extent_a = tuple(
        s * config.voxel_size for s in config.target_shape
    )  # (Z, Y, X)
    non_clippable_extents_a = [
        extent
        for extent, clippable in zip(box_extent_a, config.clip_axes)
        if not clippable
    ]
    limiting_extent_a = (
        min(non_clippable_extents_a) if non_clippable_extents_a else max(box_extent_a)
    )
    max_reach_a = 0.5 * _MEMBRANE_AUTO_SIZE_BOX_FRACTION * limiting_extent_a

    if shape_backend == "spherical_harmonics":
        if "sh_axes" in instance_kwargs or "sh_axes_range" in instance_kwargs:
            return instance_kwargs
        lo, hi = DEFAULT_SH_AXES_RANGE_A
        capped_hi = min(hi, max_reach_a)
        instance_kwargs = dict(instance_kwargs)
        instance_kwargs["sh_axes_range"] = (min(lo, capped_hi), capped_hi)
    elif shape_backend == "swept_spline":
        if any(
            k in instance_kwargs
            for k in (
                "swept_total_length",
                "swept_total_length_range",
                "swept_tube_radius",
                "swept_tube_radius_range",
            )
        ):
            return instance_kwargs
        total_lo, total_hi = DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_A
        tube_lo, tube_hi = DEFAULT_SWEPT_TUBE_RADIUS_RANGE_A
        worst_case_reach_a = 0.5 * total_hi + tube_hi
        if worst_case_reach_a > max_reach_a:
            scale = max_reach_a / worst_case_reach_a
            instance_kwargs = dict(instance_kwargs)
            instance_kwargs["swept_total_length_range"] = (
                total_lo * scale,
                total_hi * scale,
            )
            instance_kwargs["swept_tube_radius_range"] = (
                tube_lo * scale,
                tube_hi * scale,
            )
    return instance_kwargs


def build_tomogram_generator(config: TomogramConfig) -> TomogramSpecimenGenerator:
    """
    Build a `TomogramSpecimenGenerator` from a `TomogramConfig`, without
    calling `.generate()` or writing anything to disk.

    This is the same config-to-generator translation `run_build_tomogram`
    uses internally, exposed directly for callers (e.g. notebooks) that
    want the assembled volume tensor and the generator's own inspectable
    attributes (`.placements`, `.regions`, `.membrane_labels`, etc.)
    in-process, without a disk round-trip through `.mrc` + `load_specimen_volume`.

    Parameters
    ----------
    config : TomogramConfig
        Fully-resolved run configuration, e.g. from
        :func:`specter.config.load_config`.

    Returns
    -------
    TomogramSpecimenGenerator
        Not yet `.generate()`-called.
    """
    device, render_devices = parse_device_pool(resolve_available_device(config.device))

    protein_specs = _protein_specs_from_dicts(
        config.targets
    ) + _protein_specs_from_dicts(config.filler)
    if config.filler_from_pei2016:
        protein_specs += [
            TomogramProteinSpec(pdb_source=d["pdb_source"], location="cytosol")
            for d in build_filler_pool_specs(
                PEI2016_CROWDING_TABLE,
                max_mw_kda=config.filler_table_max_mw_kda,
                min_mw_kda=config.filler_table_min_mw_kda,
            )
        ]
    if config.filler_from_cryoetsim:
        protein_specs += [
            TomogramProteinSpec(pdb_source=d["pdb_source"], location="cytosol")
            for d in build_filler_pool_specs(
                CRYOETSIM_PARTICLE_TABLE,
                max_mw_kda=config.filler_table_max_mw_kda,
                min_mw_kda=config.filler_table_min_mw_kda,
                categories=config.filler_table_categories,
            )
        ]

    transmembrane_specs = [
        TransmembraneSpec(
            pdb_source=d["pdb_source"],
            # TOML says n_copies (one count spelling across every entry
            # type); TransmembraneSpec's own field stays `frequency` since
            # it doubles as the weight in the per-site species draw.
            frequency=int(d.get("n_copies", 1)),
            parameterization=d.get("parameterization", config.scattering_factors),
        )
        for d in config.membrane_transmembrane_specs
    ]
    filament_specs = [FilamentSpec(**d) for d in config.filaments]
    if config.actin:
        filament_specs.append(ACTIN_SPEC)
    microtubule_specs = [MicrotubuleSpec(**d) for d in config.microtubules]

    # Pre-render every transmembrane species' template ONCE here, shared
    # across every [[membrane]] entry AND every one of its n_copies
    # copies below -- otherwise each instance's own MembraneGenerator would
    # redundantly rebuild the same handful of species from scratch (see
    # render_transmembrane_template's own docstring). Rendering itself can
    # run concurrently across config.render_workers/device's own GPU pool
    # (see TomogramConfig.render_workers's own docstring); attaching the
    # result via dataclasses.replace(..., template=...) makes every
    # downstream MembraneGenerator._build_template call a no-op cache hit
    # regardless of how many instances end up sharing it.
    if transmembrane_specs:
        devices = resolve_render_devices(device, render_devices)
        workers = resolve_render_workers(
            config.render_workers, len(transmembrane_specs)
        )
        built = build_templates_concurrently(
            keys=list(range(len(transmembrane_specs))),
            build_one=lambda i, device: render_transmembrane_template(
                transmembrane_specs[i],
                config.voxel_size,
                config.pdb_cache_dir,
                device,
                readd_hydrogens=config.readd_hydrogens,
                monomer_library_path=config.monomer_library_path,
                use_deposited_bfactors=config.use_deposited_bfactors,
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
        n_copies = int(instance_kwargs.pop("n_copies", 1))

        # None (not config.target_shape) by default -- MembraneGenerator
        # auto-sizes a small working grid from the organelle's own size when
        # omitted (see its own docstring), instead of every instance
        # rendering on a working grid the size of the WHOLE tomogram
        # canvas. A [[membrane]] entry can still request
        # a specific target_shape explicitly (e.g. to match a scale
        # requirement of its own), popped here rather than left for
        # MembraneGenerator's **kwargs since it needs a tuple() cast like
        # config.target_shape gets below.
        target_shape_zyx_raw = instance_kwargs.pop("target_shape", None)
        target_shape = (
            tuple(target_shape_zyx_raw) if target_shape_zyx_raw is not None else None
        )
        instance_kwargs = _cap_membrane_auto_size_ranges(
            instance_kwargs, target_shape, config
        )

        # One tomogram, one set of scattering factors: the bilayer is summed
        # into the same volume as the proteins packed around it, so modelling
        # the two with different factors is a choice to make deliberately, not
        # to inherit from MembraneGenerator's own default. setdefault, so a
        # [[membrane]] table that names its own parameterization still wins.
        instance_kwargs.setdefault("parameterization", config.scattering_factors)

        for i in range(n_copies):
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
                target_shape=target_shape,  # type: ignore[arg-type]
                voxel_size=config.voxel_size,
                transmembrane_specs=list(transmembrane_specs),
                pdb_cache_dir=config.pdb_cache_dir,
                readd_hydrogens=config.readd_hydrogens,
                monomer_library_path=config.monomer_library_path,
                use_deposited_bfactors=config.use_deposited_bfactors,
                device=device,
                seed=instance_seed,
                **instance_kwargs,
            )
            membrane_instances.append(MembraneInstance(generator=mgen))

    # len(config.carbon_film) > 1 is already rejected in run_build_tomogram.
    carbon_film_spec = (
        CarbonFilmSpec(**config.carbon_film[0]) if config.carbon_film else None
    )
    bead_specs = [
        TomogramBeadSpec(
            radius=d["radius"],
            count=int(d.get("n_copies", 1)),
        )
        for d in config.beads
    ]

    gen = TomogramSpecimenGenerator(
        membrane_instances=membrane_instances,
        target_shape=tuple(config.target_shape),  # type: ignore[arg-type]
        voxel_size=config.voxel_size,
        protein_specs=protein_specs,
        filament_specs=filament_specs,
        microtubule_specs=microtubule_specs,
        carbon_film_spec=carbon_film_spec,
        bead_specs=bead_specs,
        bead_roughness=config.bead_roughness,
        occupancy_fraction=config.filler_occupancy_fraction,
        packing_max_retries=config.packing_max_retries,
        packing_voxel_size=config.packing_voxel_size,
        clip_axes=tuple(config.clip_axes),  # type: ignore[arg-type]
        region_density_threshold=config.membrane_region_density_threshold,
        region_max_passes=config.membrane_region_max_passes,
        min_transmembrane_spacing=config.membrane_min_transmembrane_spacing,
        pdb_cache_dir=config.pdb_cache_dir,
        parameterization=config.scattering_factors,
        bulk_parameterization=config.bulk_scattering_factors,
        readd_hydrogens=config.readd_hydrogens,
        monomer_library_path=config.monomer_library_path,
        use_deposited_bfactors=config.use_deposited_bfactors,
        seed=config.seed,
        device=device,
        accumulator_device=config.accumulator_device,
        render_workers=config.render_workers,
        render_devices=render_devices,  # type: ignore[arg-type]
        chunk_size=config.render_chunk_size,
    )
    _report_devices(gen, config.accumulator_device)
    return gen


def _report_devices(
    gen: TomogramSpecimenGenerator,
    accumulator_device: str | None,
) -> None:
    """
    Print the devices the generator actually resolved to.

    `accumulator_device="auto"` decides from the chosen device's *currently
    free* VRAM (`recommend_accumulator_device`), so the same config lands the
    canvas on the GPU or on the CPU depending on what else happened to be
    running at the time. Reporting it costs one line and makes that decision
    visible; without it the only symptom is a run that is inexplicably slower
    than the last one, with nothing in the log to distinguish the two.

    Parameters
    ----------
    gen : TomogramSpecimenGenerator
        Already constructed, so its `device`/`accumulator_device` are resolved.
    accumulator_device : str or None
        The config's own raw value, to name "auto" as the thing that chose.
    """
    # gen.device is whatever string the caller passed ("cuda"); the resolved
    # accumulator is always a torch.device, so compare as torch.device or the
    # two never match and every run claims a split it does not have.
    compute = torch.device(gen.device)
    line = f"[bold]Device:[/bold] {compute}"
    if gen.accumulator_device != compute:
        why = " (auto)" if accumulator_device == "auto" else ""
        line += f", accumulator on {gen.accumulator_device}{why}"
    console.print(line)


def _run_single_tomogram(config: TomogramConfig) -> None:
    """Build and save exactly one tomogram from an already-resolved config."""
    t_start = time.perf_counter()

    section("Building specimen volume")
    # Printed right after the header, before any actual work -- target_shape/
    # voxel_size are already fully resolved from config at this point, so
    # there's no reason to make a caller wait for generation to finish
    # just to confirm they're building the size they meant to. Doubles as
    # a sanity check: if this looks wrong, no need to wait out the rest
    # of the run to find out.
    target_shape = tuple(config.target_shape)
    size_angstrom = tuple(s * config.voxel_size for s in target_shape)
    console.print(
        f"[bold]Volume:[/bold] {target_shape} voxels (Z, Y, X) @ "
        f"{config.voxel_size:.2f} A/voxel = {size_angstrom[0]:.0f} x "
        f"{size_angstrom[1]:.0f} x {size_angstrom[2]:.0f} A"
    )

    gen = build_tomogram_generator(config)
    volume = gen.generate()
    if gen.membrane_instances:
        console.print(
            f"  Membrane instances: {len(gen.placed_membrane_instances)}/"
            f"{len(gen.membrane_instances)} placed"
        )
        console.print(f"  Transmembrane: {len(gen.transmembrane_placements)} placed")
    if gen.filament_specs:
        console.print(
            f"  Filaments: {len(gen.filament_instances)} "
            f"monomer instance(s) placed ({len(gen.filament_specs)} species)"
        )
    if gen.microtubule_specs:
        console.print(
            f"  Microtubules: {len(gen.microtubule_instances)} tube(s), "
            f"{len(gen.microtubule_dimer_instances)} dimer instance(s) placed"
        )
    if gen.carbon_film_spec is not None:
        console.print("  Carbon film: generated")
    if gen.bead_specs:
        n_requested = sum(spec.count for spec in gen.bead_specs)
        console.print(
            f"  Gold fiducial beads: {len(gen.bead_instances)}/{n_requested} placed"
        )
    for location in ("cytosol", "lumen"):
        placed_here = [p for p in gen.placements if p.location == location]
        if not placed_here:
            continue
        n_target = sum(1 for p in placed_here if p.role == "target")
        n_filler = len(placed_here) - n_target
        console.print(
            f"  {location.capitalize()}: {n_target} target(s), {n_filler} filler placed"
        )
    assert gen.instance_labels is not None
    # Chunked, and never materialized as float: `(labels > 0).float().mean()`
    # built a 0.40 GiB bool and a 1.61 GiB float32 copy of the whole volume
    # just to average it.
    occupancy_fraction = count_nonzero_chunked(gen.instance_labels) / float(
        gen.instance_labels.numel()
    )
    console.print(f"  Occupancy: {occupancy_fraction:.1%} of volume")

    section("Saving")

    # run_build_tomogram always hands this function an already-resolved,
    # untracked config, so this is the plain-string leaf directory rather
    # than a job-tree root -- but the field is Optional on the dataclass,
    # so resolve it once here instead of asserting at each use below.
    output_dir = resolve_output_dir(config, "tomograms")
    os.makedirs(output_dir, exist_ok=True)
    mrc_path = tomogram_output_path(config)
    _write_volume_mrc(
        mrc_path,
        volume.cpu().numpy().astype("float32", copy=False),
        config.voxel_size,
    )
    console.print(f"  [green]✓[/green] {mrc_path}")

    if config.write_picks:
        written = gen.export_picks(
            output_dir, annotation_version=config.annotation_version
        )
        for path in written.values():
            console.print(f"  [green]✓[/green] {path}")

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
            # Escalated to float32 rather than truncated. numpy's astype
            # wraps silently on overflow -- instance 65,536 becomes 0 and
            # reads back as BACKGROUND, and anything past that collides
            # with a real instance's id, while the .ndjson picks stay
            # correct so the two disagree with nothing to say so. MRC has
            # no integer mode wider than uint16, but float32 (mode 2) is
            # exact for integers to 2**24, which is 256x the uint16
            # ceiling and past any instance count the packer can reach.
            #
            # In practice this never fires: a 1500x6000x6000 A volume, the
            # largest specter is built for, plateaus near 22,000 instances
            # because RSA packing jams on placement attempts rather than
            # on volume. It exists because the failure it replaces is
            # silent.
            peak = int(labels.max())
            if dtype.startswith("uint") and peak > _MRC_UINT16_MAX:
                console.print(
                    f"  [yellow]note[/yellow] {peak:,} labels exceeds uint16's "
                    f"{_MRC_UINT16_MAX:,}; writing{suffix} as float32 "
                    "(exact for integers, MRC has no wider integer mode)"
                )
                dtype = "float32"
            path = os.path.join(output_dir, config.filename + suffix)
            _write_volume_mrc(
                path, labels.cpu().numpy().astype(dtype), config.voxel_size
            )
            console.print(f"  [green]✓[/green] {path}")

        assert gen.instance_labels is not None
        _write_label_mrc("_protein_labels.mrc", gen.instance_labels, "uint16")

        if config.membrane:
            assert gen.membrane_labels is not None
            _write_label_mrc("_membrane_labels.mrc", gen.membrane_labels, "uint16")

            assert gen.regions is not None
            regions_volume = torch.zeros_like(gen.membrane_labels)
            regions_volume[gen.regions["shell"]] = 1
            regions_volume[gen.regions["lumen"]] = 2
            _write_label_mrc("_regions.mrc", regions_volume, "uint16")

    elapsed = time.perf_counter() - t_start
    console.print(f"\n[bold]Total time:[/bold] {format_elapsed(elapsed)}")
