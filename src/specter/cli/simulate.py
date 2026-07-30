"""`specter simulate` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import (
    PARTICLE_STACK_HELP,
    ParticleStackConfig,
    apply_overrides,
    load_config,
)

from ._click_options import build_config_options, collect_overrides

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# (panel title, field names) -- basic-themed panels first, in the order most
# runs actually tune them, then a single catch-all "Advanced" panel for
# everything a user is unlikely to need to touch (mirrors the field ordering
# in ParticleStackConfig itself -- rich-click panels render in first-seen
# order, so that ordering is what controls panel order here too).
_PARTICLE_STACK_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Structure & Potential",
        ["pdb_code", "assembly", "num_pixels", "pixel_size"],
    ),
    (
        "Microscope",
        ["energy", "dose", "cs", "alpha"],
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
    (
        "Advanced",
        [
            "pdb_savefolder",
            "cs_path",
            "num_frames",
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
            "mmcif_filepath",
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
            "anisomag_m00",
            "anisomag_m01",
            "anisomag_m10",
            "anisomag_m11",
        ],
    ),
]


def _default_particle_config_path() -> str:
    from specter.config import REPO_ROOT

    return str(REPO_ROOT / "configs" / "particle.toml")


def _field_panels() -> dict[str, str]:
    """Field name -> rich-click panel title, derived from `_PARTICLE_STACK_GROUPS`."""
    return {name: title for title, names in _PARTICLE_STACK_GROUPS for name in names}


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
            default=_default_particle_config_path(),
            show_default=True,
            help="TOML config file. Always loaded first, before any flags below "
            "are applied.",
            panel="Config",
        ),
        *build_config_options(
            ParticleStackConfig,
            field_help=PARTICLE_STACK_HELP,
            field_panels=_field_panels(),
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


def build_simulate_group() -> click.RichGroup:
    """Build the `simulate` command group and its subcommands."""
    group = click.RichGroup(
        name="simulate",
        help="Simulate cryo-EM/cryo-ET data",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(_build_particles_command())
    return group
