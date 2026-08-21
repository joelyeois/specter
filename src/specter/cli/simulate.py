"""`specter simulate` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import (
    MICROGRAPH_HELP,
    PARTICLE_STACK_HELP,
    TILT_SERIES_HELP,
    MicrographConfig,
    ParticleStackConfig,
    TiltSeriesConfig,
    TomogramConfig,
    apply_overrides,
    load_config,
)

from ._click_options import (
    build_config_options,
    collect_overrides,
    default_config_path,
    field_panels,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# (panel title, field names) -- basic-themed panels first, in the order most
# runs actually tune them, then a single catch-all "Advanced" panel for
# everything a user is unlikely to need to touch (mirrors the field ordering
# in ParticleStackConfig itself -- rich-click panels render in first-seen
# order, so that ordering is what controls panel order here too).
_PARTICLE_STACK_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Structure & Potential",
        ["pdb_code", "assembly", "n_pixels", "pixel_size"],
    ),
    (
        "Microscope",
        ["voltage", "dose", "cs", "alpha"],
    ),
    (
        "Sampling",
        ["defocus", "shift", "n_particles"],
    ),
    (
        "Models",
        ["scattering_model", "detector_model"],
    ),
    (
        "Post-processing",
        ["normalize_particles", "save_exitwaves", "save_clean_exitwaves"],
    ),
    ("Compute", ["device", "batchsize"]),
    ("Output", ["output_dir", "filename"]),
    ("Job tracking", ["project", "job_id", "job_base_dir"]),
    (
        "Advanced",
        [
            "pdb_savefolder",
            "cs_path",
            "star_path",
            "n_frames",
            "convergence_angle",
            "cc",
            "energy_spread",
            "deltaV_V",
            "deltaI_I",
            "dose_envelope",
            "bfactor",
            "aberration_model",
            "noise_model",
            "coincidence_radius",
            "ice_model",
            "ice_thickness",
            "ice_cache_dir",
            "crowd_min_distance",
            "crowd_max_distance_z",
            "potential_scale",
            "pad_fft",
            "potential_parameterization",
            "potential_method",
            "rcut",
            "conv_backend",
            "periodic",
            "shtyrov_params_path",
            "ews_curvature_sign",
            "klim",
            "rotate_mode",
            "ice_parameterization",
            "ice_relax_steps",
            "crowd_chunk_size",
            "crowd_max_distance_xy",
            "crowd_method",
            "crowd_n_points",
            "crowd_seed",
            "crowd_move_to_cpu",
            "water_air_interface",
            "seed",
            "astigmatism",
            "astigmatism_angle",
            "phaseshift",
            "tiltx",
            "tilty",
            "trefoil1",
            "trefoil2",
            "tetrafoil1",
            "tetrafoil2",
            "tetrafoil3",
            "tetrafoil4",
            "anisomag_m00",
            "anisomag_m01",
            "anisomag_m10",
            "anisomag_m11",
        ],
    ),
]


# (panel title, field names) for `specter simulate tiltseries` -- same
# basic-first-advanced-last convention as `_PARTICLE_STACK_GROUPS`. Drives
# the `volume_path` specimen source only (see `specter.pipelines.
# run_tilt_series`) -- specimen BUILDING is `specter build tomogram`'s job
# (a separate command/config entirely), not this one's.
_TILT_SERIES_GROUPS: list[tuple[str, list[str]]] = [
    ("Specimen", ["volume_path", "voxel_size"]),
    (
        "Microscope",
        ["voltage", "dose_per_tilt", "n_frames", "cs", "alpha"],
    ),
    ("Defocus", ["defocus"]),
    (
        "Tilt geometry",
        ["min_tilt_angle", "max_tilt_angle", "n_tilts", "tilt_axis"],
    ),
    (
        "Models",
        ["scattering_model", "aberration_model", "noise_model", "detector_model"],
    ),
    (
        "Post-processing",
        ["normalize_tilt_series", "save_exitwaves"],
    ),
    ("Compute", ["device"]),
    ("Output", ["output_dir", "filename"]),
    ("Job tracking", ["project", "job_id", "job_base_dir"]),
    (
        "Advanced",
        [
            "micrograph_size",
            "convergence_angle",
            "cc",
            "energy_spread",
            "deltaV_V",
            "deltaI_I",
            "dose_envelope",
            "coincidence_radius",
            "ice_model",
            "ice_cache_dir",
            "ice_relax_steps",
            "pad_fft",
        ],
    ),
]


# (panel title, field names) for `specter simulate micrograph` -- same
# basic-first-advanced-last convention as `_PARTICLE_STACK_GROUPS`, mirroring
# `MicrographConfig`'s own field ordering.
_MICROGRAPH_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Specimen",
        ["pdb_code", "assembly", "n_pixels", "pixel_size", "micrograph_size"],
    ),
    (
        "Microscope",
        ["voltage", "dose", "cs", "alpha"],
    ),
    ("Defocus", ["defocus"]),
    ("Dataset", ["n_micrographs"]),
    (
        "Models",
        ["scattering_model", "aberration_model", "noise_model", "detector_model"],
    ),
    (
        "Post-processing",
        ["normalize_micrographs", "save_exitwaves", "save_clean_exitwaves"],
    ),
    ("Compute", ["device"]),
    ("Output", ["output_dir", "filename"]),
    ("Job tracking", ["project", "job_id", "job_base_dir"]),
    (
        "Advanced",
        [
            "pdb_savefolder",
            "n_frames",
            "convergence_angle",
            "cc",
            "energy_spread",
            "deltaV_V",
            "deltaI_I",
            "dose_envelope",
            "coincidence_radius",
            "ice_model",
            "ice_thickness",
            "ice_profile",
            "ice_thickness_range",
            "ice_profile_angle",
            "ice_hole_radius",
            "ice_rim_thickness",
            "ice_hole_offset",
            "ice_tilt",
            "ice_cache_dir",
            "crowd_min_distance",
            "crowd_max_distance_z",
            "water_air_interface",
            "potential_scale",
            "pad_fft",
            "specimen_chunk_size",
        ],
    ),
]


def _particles_callback(config: str, **_overrides_raw: object) -> None:
    """Handle `specter simulate particles`."""
    from specter.pipelines import run_particle_stack

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config"})

    cfg = load_config(config, ParticleStackConfig)
    apply_overrides(cfg, overrides)
    run_particle_stack(cfg)


def _build_particles_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=default_config_path("particle"),
            show_default=True,
            help="TOML config file. Always loaded first, before any flags below "
            "are applied.",
            panel="Config",
        ),
        *build_config_options(
            ParticleStackConfig,
            field_help=PARTICLE_STACK_HELP,
            field_panels=field_panels(_PARTICLE_STACK_GROUPS),
        ),
    ]
    return click.RichCommand(
        name="particles",
        params=params,
        callback=_particles_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Simulate a cryo-EM particle stack and save it as .mrcs + .star. "
        "A TOML config (--config) is always loaded first -- every flag below "
        "is optional and, if given, overrides one field of it.",
    )


def _micrograph_callback(config: str, **_overrides_raw: object) -> None:
    """Handle `specter simulate micrograph`."""
    from specter.pipelines import run_micrograph

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config"})

    cfg = load_config(config, MicrographConfig)
    apply_overrides(cfg, overrides)
    run_micrograph(cfg)


def _build_micrograph_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=default_config_path("micrograph"),
            show_default=True,
            help="TOML config file. Always loaded first, before any flags below "
            "are applied.",
            panel="Config",
        ),
        *build_config_options(
            MicrographConfig,
            field_help=MICROGRAPH_HELP,
            field_panels=field_panels(_MICROGRAPH_GROUPS),
        ),
    ]
    return click.RichCommand(
        name="micrograph",
        params=params,
        callback=_micrograph_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Simulate one or more cryo-EM micrographs and save them as .mrcs + "
        ".star. A TOML config (--config) is always loaded first -- every flag "
        "below is optional and, if given, overrides one field of it. Each "
        "micrograph gets an independently regenerated ice/crowding specimen; "
        "single-device only.",
    )


def _tiltseries_callback(
    config: str, tomogram_config: str | None, **_overrides_raw: object
) -> None:
    """Handle `specter simulate tiltseries`."""
    from specter.pipelines import run_tilt_series

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config", "tomogram_config"})

    cfg = load_config(config, TiltSeriesConfig)
    apply_overrides(cfg, overrides)

    tomogram_cfg = (
        load_config(tomogram_config, TomogramConfig)
        if tomogram_config is not None
        else None
    )
    run_tilt_series(cfg, tomogram_config=tomogram_cfg)


def _build_tiltseries_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=default_config_path("tilt_series"),
            show_default=True,
            help="TOML config file. Always loaded first, before any flags below "
            "are applied.",
            panel="Config",
        ),
        click.RichOption(
            ["--tomogram_config"],
            type=str,
            default=None,
            help="Optional TOML config for `specter build tomogram` "
            "(TomogramConfig). If given, that specimen volume is built "
            "first and its output used as this run's specimen -- chains "
            "`specter build tomogram` + `specter simulate tiltseries` in "
            "one command. Mutually exclusive with --volume_path/--config's "
            "volume_path. Always builds exactly one tomogram -- "
            "`--n_tomograms` isn't a TomogramConfig field and isn't "
            "settable here; for several tomograms, run `specter build "
            "tomogram --n_tomograms N` yourself and call `simulate "
            "tiltseries --volume_path ...` once per output volume instead.",
            panel="Config",
        ),
        *build_config_options(
            TiltSeriesConfig,
            field_help=TILT_SERIES_HELP,
            field_panels=field_panels(_TILT_SERIES_GROUPS),
        ),
    ]
    return click.RichCommand(
        name="tiltseries",
        params=params,
        callback=_tiltseries_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Simulate a cryo-ET tilt series from a pre-built specimen volume "
        "(--volume_path) and save it as .mrcs + .star. A TOML config "
        "(--config) is always loaded first -- every flag below is optional "
        "and, if given, overrides one field of it. Pass --tomogram_config "
        "instead of --volume_path to build the specimen volume first "
        "(`specter build tomogram`) and image it in the same command.",
    )


def build_simulate_group() -> click.RichGroup:
    """Build the `simulate` command group and its subcommands."""
    group = click.RichGroup(
        name="simulate",
        help="Simulate cryo-EM/cryo-ET data",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(_build_particles_command())
    group.add_command(_build_micrograph_command())
    group.add_command(_build_tiltseries_command())
    return group
