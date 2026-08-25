"""`specter reconstruct` command group."""

from __future__ import annotations

import rich_click as click

from specter.config import (
    RECONSTRUCTION_HELP,
    ReconstructionConfig,
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

# (panel title, field names) -- basic-first-advanced-last, the same convention
# as cli/simulate.py's _PARTICLE_STACK_GROUPS and cli/build.py's
# _TOMOGRAM_GROUPS.
_RECONSTRUCT_PARTICLE_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Data",
        ["cs_file", "mrc_file", "dose_per_angstrom", "halfset", "num_particles"],
    ),
    ("Optimisation", ["epochs", "batchsize", "lr", "scheduler", "lr_decay"]),
    ("Symmetry", ["symmetry", "symmetry_mode", "symmetry_batchsize"]),
    ("Sanity check", ["test_run", "bin_factor"]),
    ("Compute", ["device", "precision", "num_workers"]),
    ("Output & job tracking", ["output_dir", "project", "job_id"]),
    ("Reference maps", ["fsc_ref", "fsc_mask", "cryosparc_ref", "use_2d_mask"]),
    (
        "Refinement (unverified)",
        ["lr_R", "lr_T", "lr_D", "defocus_offset"],
    ),
    (
        "Advanced",
        [
            "scattering_model",
            "aberration_backend",
            "ews_curvature_sign",
            "bfactor",
            "klim",
            "sparsity",
            "rotate_mode",
            "learn_noise_model",
            "use_ncc",
        ],
    ),
]


def _particle_callback(config: str, **_overrides_raw: object) -> None:
    """Handle `specter reconstruct particle`."""
    from specter.pipelines import run_reconstruction

    ctx = click.get_current_context()
    assert ctx is not None
    overrides = collect_overrides(ctx, exclude={"config"})

    cfg = load_config(config, ReconstructionConfig)
    apply_overrides(cfg, overrides)
    run_reconstruction(cfg)


def _reconstruct_particle_command() -> click.RichCommand:
    params: list[click.Parameter] = [
        click.RichOption(
            ["--config"],
            type=str,
            default=default_config_path("reconstruct"),
            show_default=True,
            help="TOML config file. Always loaded first, before any flags below "
            "are applied.",
            panel="Config",
        ),
        *build_config_options(
            ReconstructionConfig,
            field_help=RECONSTRUCTION_HELP,
            field_panels=field_panels(_RECONSTRUCT_PARTICLE_GROUPS),
        ),
    ]
    return click.RichCommand(
        name="particle",
        params=params,
        callback=_particle_callback,
        context_settings=CONTEXT_SETTINGS,
        help="Reconstruct a 3D volume from an experimental single-particle "
        "stack, by fitting the same physics-based forward model `specter "
        "simulate particles` generates with. Takes a CryoSPARC .cs file for "
        "the poses and CTF parameters plus the .mrc/.mrcs stack it indexes, "
        "and writes the reconstructed volume, per-epoch volumes and FSC "
        "curves into a run directory. Start with --test_run, which fits one "
        "epoch on binned images and exercises every path and parameter for a "
        "fraction of the cost. Pose, translation and defocus refinement "
        "(--lr_R/--lr_T/--lr_D) are wired in but unverified against ground "
        "truth. A TOML config (--config) is always loaded first -- every "
        "flag below is optional and, if given, overrides one field of it.",
    )


def build_reconstruct_group(name: str = "reconstruct") -> click.RichGroup:
    """
    Build the `reconstruct` command group and its subcommands.

    Parameters
    ----------
    name : str
        Name to register the group under. `cli/_cli.py` builds it twice, as
        `reconstruct` and as `ghostbuster`, so the solver is reachable under
        the name the docs and the Python API call it by. A second group
        object rather than the same one registered twice: click derives a
        command's own `name` at construction, and sharing one instance across
        two keys would have `--help` print whichever name was registered
        first, whatever the user actually typed.

        A separate `ghostbuster` console script would have been the other way
        to do this, and is deliberately not: the name is already taken on
        PyPI by an unrelated AWS tool whose package installs a `ghostbuster`
        entry point, so both would write the same file into a shared
        environment and the later install would silently win.

    Returns
    -------
    click.RichGroup
        The group, with its subcommands attached.
    """
    group = click.RichGroup(
        name=name,
        help="Reconstruct 3D volumes from experimental images",
        context_settings=CONTEXT_SETTINGS,
    )
    group.add_command(_reconstruct_particle_command())
    return group
