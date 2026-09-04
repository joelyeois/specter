"""`specter build` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import (
    ICE_CACHE_HELP,
    TOMOGRAM_HELP,
    IceCacheConfig,
    TomogramConfig,
)

from ._click_options import (
    build_config_options,
    collect_overrides,
    CONFIG_OPTION_HELP,
    load_validated_config,
    prerequisite_usage_error,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# (panel title, field names) -- basic-first-advanced-last, same convention as
# cli/simulate.py's _PARTICLE_STACK_GROUPS/_TILT_SERIES_GROUPS. targets/filler/
# membrane/membrane_transmembrane_specs/filaments are all list[dict]-typed
# and skipped entirely by build_config_options (TOML-only) -- not listed
# here, same treatment as that module's own protein_specs/membrane_specs --
# nor is `target_shape` (list[int]), for the same reason.
# "actin" (bool) is the one filaments-related field that IS a real CLI flag.
_TOMOGRAM_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Specimen",
        ["voxel_size", "filler_occupancy_fraction"],
    ),
    (
        "Filler tables",
        [
            "filler_from_pei2016",
            "filler_from_cryoetsim",
            "filler_table_max_mw_kda",
            "filler_table_min_mw_kda",
            "scattering_factors",
            "bulk_scattering_factors",
        ],
    ),
    (
        "Membrane",
        [
            "membrane_region_density_threshold",
            "membrane_region_max_passes",
            "membrane_min_transmembrane_spacing",
        ],
    ),
    ("Filaments", ["actin"]),
    (
        "Picks & segmentation",
        ["write_picks", "annotation_version", "write_segmentation"],
    ),
    (
        "Compute",
        ["device", "accumulator_device", "render_workers", "render_chunk_size"],
    ),
    # One panel, not two: --output_dir and the tracking flags are alternative
    # answers to the same question, and split across separate panels nothing
    # showed that setting the latter makes the former a no-op.
    (
        "Output & job tracking",
        ["output_dir", "filename", "project", "job_id"],
    ),
    (
        "Advanced",
        [
            "pdb_cache_dir",
            "readd_hydrogens",
            "monomer_library_path",
            "use_deposited_bfactors",
            "packing_max_retries",
            "packing_voxel_size",
            "bead_roughness",
            "seed",
        ],
    ),
]

_ICE_GROUPS: list[tuple[str, list[str]]] = [
    ("Library", ["num_configs", "n", "dx", "seed_start"]),
    ("Optimisation", ["n_steps"]),
    ("Compute", ["device"]),
    ("Output", ["output_dir", "overwrite", "diagnostics"]),
]


def _tomogram_callback(
    config: str | None, n_tomograms: int, **_overrides_raw: object
) -> None:
    """Handle `specter build tomogram`."""
    from specter.pipelines import run_build_tomogram

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config", "n_tomograms"})

    cfg = load_validated_config(TomogramConfig, config, overrides)

    # Checked here as well as in run_build_tomogram so a CLI user gets a
    # usage error rather than that function's ValueError as a traceback.
    # Seven of the ten species sources are TOML-only, so this cannot be
    # phrased as "pass --x" the way a single missing field can.
    if not cfg.has_any_species:
        prerequisite_usage_error(
            "A tomogram needs at least one species source, and every one of "
            "them defaults to empty. Quickest runnable start: --config "
            "configs/tomogram.toml. Or with a single flag: "
            "--filler_from_pei2016 True for cytosolic crowding, --actin True "
            "for filaments. Targets, membranes, microtubules, carbon film "
            "and gold beads are TOML-only -- [[targets]], [[membrane]], "
            "[[microtubules]], [[carbon_film]], [[beads]]."
        )

    run_build_tomogram(cfg, n_tomograms=n_tomograms)


def _build_tomogram_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=None,
            show_default=False,
            help=CONFIG_OPTION_HELP,
            panel="Config",
        ),
        click.RichOption(
            ["--n_tomograms"],
            type=int,
            default=1,
            show_default=True,
            help="Number of independent tomograms to generate. Beyond the "
            "first, each one is written into its own numbered subdirectory "
            "of --output_dir (0001/, 0002/, ...) and, if --seed/"
            "config.seed is set, gets its own incrementing seed, so runs "
            "don't collide but stay reproducible.",
            panel="Config",
        ),
        *build_config_options(
            TomogramConfig,
            field_help=TOMOGRAM_HELP,
            field_groups=_TOMOGRAM_GROUPS,
        ),
    ]
    return click.RichCommand(
        name="tomogram",
        params=params,
        callback=_tomogram_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Build a specimen tomogram from any combination of: one or "
        "more composited organic membranes (--config's TOML [[membrane]] "
        "entries, each its own shape_backend), filament "
        "species scattered through it (--actin or [[filaments]]), and "
        "densely packed protein species (--config's [targets]/[filler] "
        "tables, region-gated to cytosol/lumen when a membrane is present). "
        "Generation order is membranes, then filaments, then protein fill -- "
        "each stage avoids the previous ones' placements. Saves the volume "
        "as .mrc (usable as `specter simulate tiltseries`'s --volume_path) "
        "plus copick-style .ndjson picks and, by default, segmentation "
        "label volumes (--write_segmentation). A TOML config (--config) is "
        "loaded first when given, otherwise every setting takes its "
        "built-in default -- every flag below is optional and, if "
        "given, overrides one field of it.",
    )


def _ice_callback(config: str | None, **_overrides_raw: object) -> None:
    """Handle `specter build ice`."""
    from specter.pipelines import run_build_ice_cache

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config"})

    cfg = load_validated_config(IceCacheConfig, config, overrides)
    run_build_ice_cache(cfg)


def _build_ice_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=None,
            show_default=False,
            help=CONFIG_OPTION_HELP,
            panel="Config",
        ),
        *build_config_options(
            IceCacheConfig,
            field_help=ICE_CACHE_HELP,
            field_groups=_ICE_GROUPS,
        ),
    ]
    return click.RichCommand(
        name="ice",
        params=params,
        callback=_ice_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Generate a library of amorphous-ice configurations for "
        "`IceBank` to draw from, as an alternative to the one bundled with "
        "specter. Useful when simulations need ice at a pixel size or in a "
        "volume larger than the bundled library covers (256 A cells at 1 "
        "A/voxel), or more independent configurations. Each one is a "
        "full GradientSKIcemaker optimisation against the S(k) and ML-BOP "
        "energy of real amorphous ice, costing tens of minutes at "
        "production scale -- pass several GPUs to --device to shard them, "
        "and re-run the same command to resume an interrupted run. Point a "
        "simulation config's ice_cache_dir at the output directory to use "
        "the result. A TOML config (--config) is loaded first when given, otherwise every setting takes its built-in default -- "
        "every flag below is optional and, if given, overrides one field of "
        "it.",
    )


def build_build_group() -> click.RichGroup:
    """Build the `build` command group and its subcommands."""
    group = click.RichGroup(
        name="build",
        help="Build specimen volumes and reusable assets",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(_build_tomogram_command())
    group.add_command(_build_ice_command())
    return group
