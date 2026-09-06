"""ReconstructionConfig: parameters for single-particle reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from ._field import help_of, setting
from typing import Literal
from specter.options import EwaldSphereSign, RotateMode, Scheduler


@dataclass
class ReconstructionConfig:
    """Parameters for single-particle reconstruction, loaded from a TOML config file.

    Drives `specter.pipelines.run_reconstruction` (the `specter reconstruct
    particles` command), which builds a `specter.ghostbuster.Ghostbuster` from
    these fields and fits it. Every field maps one-to-one onto a `Ghostbuster`
    constructor argument, except the output/job fields, which decide where the
    run writes, and `test_run`/`bin_factor`, which pick `test_run()` over
    `run()`.

    Two of `Ghostbuster`'s arguments have no field here. `nps_weight` is a
    per-frequency tensor and `lpp_params` a nested table of laser-phase-plate
    settings; neither has a spelling a config field or a CLI flag can carry,
    so both remain Python-API-only.
    """

    # --- Data ---
    # The CryoSPARC .cs file supplying poses, CTF parameters and per-particle
    # scale, and the particle stack it indexes into. No .star equivalent yet:
    # Ghostbuster reads .cs only (io/_relion.py's reader is not wired in).
    cs_file: str = setting(
        help=(
            "CryoSPARC .cs file holding the particle poses, CTF "
            "parameters and per-particle scale."
        ),
        check="existing_file",
    )
    mrc_file: str = setting(
        help="Particle stack (.mrc/.mrcs) the .cs file indexes into.",
        check="existing_file",
    )
    # Total fluence per image, e-/A^2. Sets the Poisson statistics the loss is
    # weighted by, so it has to be the real value for the dataset.
    dose_per_angstrom: float = setting(
        help=(
            "Total fluence per image in e-/Angstrom^2. Sets the "
            "Poisson statistics the loss is weighted by, so it must be the dataset's "
            "real value."
        ),
        check="positive",
    )
    # Which gold-standard half-set to reconstruct. "gold" (the default)
    # reconstructs A and B and computes the halfmap FSC between them; "A"/"B"
    # select a single halfset (and name the outputs volume_A.mrc / volume_B.mrc),
    # useful for a quick test of one half; "all" uses every particle in one
    # single-volume run, ignoring the split entirely.
    halfset: Literal["A", "B", "all", "gold"] = setting(
        "gold",
        help=(
            "Which gold-standard half-set to reconstruct. gold (the "
            "default) reconstructs A and B and computes the halfmap FSC between "
            "them; A or B alone reconstructs just that half, e.g. for a quick test; "
            "all uses every particle in one single-volume run, ignoring the split."
        ),
    )
    # Reconstruct only the first N particles. Useful for a quick run before
    # committing to the full stack; None uses all of them.
    n_particles: int | None = setting(
        None,
        help="Reconstruct only the first N particles instead of the whole stack.",
        check="positive",
    )

    # --- Optimisation ---
    epochs: int = setting(
        5, help="Number of passes over the particle stack.", check="positive"
    )
    batchsize: int = setting(
        3, help="Particles per optimisation step.", check="positive"
    )
    # Volume learning rate. None freezes the volume, which is what refining
    # poses alone means.
    lr: float | None = setting(
        0.1,
        help=(
            "Learning rate for the volume. Unset (None) freezes the volume, "
            "which is what refining poses alone means."
        ),
        check="non_negative",
    )
    scheduler: Scheduler = setting(
        "LambdaLR", help="Learning-rate schedule for the volume optimiser."
    )
    lr_decay: float = setting(
        0.1,
        help=(
            "Decay rate for the LambdaLR schedule: multiplier = "
            "1 / (1 + lr_decay * sqrt(global_step)). Unused by other schedulers."
        ),
        check="non_negative",
    )

    # --- Symmetry ---
    symmetry: str | None = setting(
        None,
        help=(
            "Point-group symmetry enforced on the volume, e.g. C3, D7, I1. Omit for C1."
        ),
    )
    symmetry_mode: RotateMode = setting(
        "fourier", help="Domain in which symmetry is applied."
    )
    symmetry_batchsize: int | None = setting(
        None,
        help=(
            "Batch size used when applying symmetry operators "
            "to the volume. Lower it if symmetry expansion runs out of memory."
        ),
        check="positive",
    )

    # --- Sanity check ---
    # Run one epoch on bin_factor-binned images and stop, instead of the full
    # fit. Every file path, .cs field and physics setting is exercised, at a
    # fraction of the cost -- run this before a multi-hour job.
    test_run: bool = setting(
        False,
        help=(
            "Run one epoch on binned images and stop, instead of the "
            "full fit. Exercises every path, .cs field and physics setting at a "
            "fraction of the cost -- do this before a multi-hour job."
        ),
    )
    bin_factor: int = setting(
        8, help="Spatial binning factor used by --test_run.", check="positive"
    )

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:N" | a bare GPU index | a comma-separated list of
    # GPU indices ("0,1"), which trains across them via Lightning DDP.
    device: str = setting(
        "cuda",
        help=(
            "cpu | cuda | cuda:N | a bare GPU index | a comma-separated "
            "list of GPU indices (0,1), which trains across them via Lightning DDP "
            "with gradients all-reduced every step."
        ),
    )
    precision: str = setting(
        "16-mixed",
        help="Lightning trainer precision, e.g. 16-mixed or 32. Forced to 32 on CPU.",
    )
    num_workers: int = setting(
        8, help="Dataloader worker processes.", check="non_negative"
    )

    # --- Output & job tracking ---
    # Every run is numbered and routed through `specter.jobs`: the directory
    # becomes output_dir/[project/]reconstructions/<job_id>/, numbered
    # J001, J002, ... (shared across every job type in the project, not just
    # reconstructions), with a job.json recording the full parameter set,
    # git commit and status. That is what makes two halfset runs (halfset
    # "A" then "B") shareable into one job directory, and what lets a batch
    # script pin a job_id up front and resume into it.
    #
    # `project` is optional, not required: leaving it unset doesn't mean
    # "untracked" -- it drops just the project-name segment
    # (output_dir/reconstructions/<job_id>/), the implicit default project
    # for whatever output_dir resolves to. Pass `--project` to split one
    # output_dir into several named projects, e.g. one shared scratch
    # directory used across unrelated structures.
    project: str | None = setting(
        None,
        help=(
            "Name for a group of jobs, e.g. one structure's worth of "
            "runs. Optional: omitting it doesn't mean untracked -- every run is "
            "numbered and gets a job.json regardless -- it just drops the "
            "project-name segment, using output_dir's implicit default project "
            "instead of a named one. Pass this to split one output_dir into "
            "several, e.g. one shared scratch directory used across structures."
        ),
    )
    job_id: str | None = setting(
        None,
        help=(
            "Pin the job directory (e.g. J001) rather than auto-assigning "
            "the next one: resumes into it if it exists, creates it otherwise. This "
            "is how two halfset runs share one job."
        ),
    )
    # Defaults to the project root discovered by walking up from cwd looking
    # for an existing .specter marker (find_specter_project_root()), the
    # same way `git` resolves the nearest ancestor containing .git -- so
    # running from a subdirectory of an already-initialised project still
    # lands in the same project, rather than starting a second,
    # disconnected job tree (and job numbering restarting from J001) right
    # where you happened to be standing.
    output_dir: str | None = setting(
        None,
        help=(
            "The one directory this run writes under. Every reconstruction is tracked, so this is always the root of the numbered "
            "job tree ([<project>/]reconstructions/J00N/), never the leaf the "
            "files land in directly. Defaults to the project root found by "
            "walking up from cwd looking for an existing .specter marker, the same "
            "way git finds the nearest .git -- so running from a subdirectory of "
            "an already-initialised project lands in the same project."
        ),
    )

    # --- Reference maps (FSC logging only, never optimised against) ---
    fsc_ref: str | None = setting(
        None,
        help=(
            "Reference volume (.mrc) for map-to-model FSC logging. Never "
            "optimised against -- reporting only."
        ),
        check="existing_file",
    )
    fsc_mask: str | None = setting(
        None,
        help="Mask volume (.mrc) applied before the FSC is computed.",
        check="existing_file",
    )
    # One path, used for whichever halfset runs, or a comma-separated
    # "<A>,<B>" pair naming one reference per halfset. The pair is what a
    # halfset="gold" run wants: it reconstructs both halves, and each should
    # be compared against CryoSPARC's matching half-map rather than both
    # against the same one. See `cryosparc_ref_for_halfset`.
    cryosparc_ref: str | None = setting(
        None,
        help=(
            "CryoSPARC's own reconstruction (.mrc), plotted "
            "alongside --fsc_ref for comparison. Only used when --fsc_ref is given "
            "too. Accepts a comma-separated '<A>,<B>' pair to compare each halfset "
            "against CryoSPARC's matching half-map, which is what --halfset gold "
            "wants; a single path is used for whichever halfset runs."
        ),
    )
    # Rotate fsc_mask per particle, project it to 2D, and weight the
    # image-domain loss by it.
    use_2d_mask: bool = setting(
        False,
        help=(
            "Rotate --fsc_mask per particle, project it to 2D, and "
            "weight the image-domain loss by it."
        ),
    )

    # --- Refinement (unverified) ---
    # Pose and defocus refinement. Wired in but unverified: no test checks
    # recovered rotations, translations or defocus against ground truth, so
    # treat a run with these set as an experiment, not a result.
    lr_R: float | None = setting(
        None,
        help=(
            "Learning rate for per-particle rotations. UNVERIFIED: pose "
            "refinement is wired in but no test checks recovered rotations against "
            "ground truth."
        ),
        check="non_negative",
    )
    lr_T: float | None = setting(
        None,
        help="Learning rate for per-particle translations. UNVERIFIED, see --lr_R.",
        check="non_negative",
    )
    lr_D: float | None = setting(
        None,
        help="Learning rate for the defocus offset. UNVERIFIED, see --lr_R.",
        check="non_negative",
    )
    # Constant offset in Angstrom added to every particle's dfu/dfv before the
    # transfer function is built. The starting point for lr_D refinement.
    defocus_offset: float = setting(
        0.0,
        help=(
            "Constant offset in Angstrom added to every particle's "
            "dfu/dfv. The starting point when --lr_D refinement is enabled."
        ),
    )

    # --- Forward model ---
    # Defaults to rytov rather than the simulator's multislice: reconstruction
    # runs the forward model once per particle per step, and rytov is the
    # cheapest model that still carries curvature of the Ewald sphere.
    scattering_model: Literal["multislice", "rytov", "firstborn", "projection"] = (
        setting(
            "rytov",
            help=(
                "Wave propagation model used by the forward pass. "
                "rytov is the default here rather than multislice: the model runs once "
                "per particle per step, and rytov is the cheapest one that still carries "
                "Ewald-sphere curvature."
            ),
        )
    )
    ews_curvature_sign: EwaldSphereSign = setting(
        "negative",
        help=(
            "Sign convention for Ewald-sphere curvature. negative matches CryoSPARC."
        ),
    )
    bfactor: float | None = setting(
        None,
        help=(
            "Isotropic B-factor envelope in Angstrom^2 damping "
            "high-resolution signal in the forward model."
        ),
        check="non_negative",
    )
    # Hard frequency cutoff (1/A) applied to the simulated images.
    klim: float | None = setting(
        None,
        help="Hard frequency cutoff (1/Angstrom) applied to the simulated images.",
        check="non_negative",
    )
    sparsity: float | None = setting(
        None, help="L1 regularisation weight on the volume.", check="non_negative"
    )
    rotate_mode: RotateMode = setting(
        "real", help="Domain in which per-particle rotations are applied."
    )
    learn_noise_model: bool = setting(
        False,
        help=(
            "Estimate sigma^2(k) from the residuals during "
            "training (RELION-style) instead of assuming it."
        ),
    )
    use_ncc: bool = setting(
        False, help="Use a normalised cross-correlation loss instead of MSE."
    )


RECONSTRUCTION_HELP: dict[str, str] = help_of(ReconstructionConfig)


def parse_cryosparc_ref(cryosparc_ref: str | None) -> list[str]:
    """
    Split a ``cryosparc_ref`` into its component paths.

    Parameters
    ----------
    cryosparc_ref : str or None
        One path, or a comma-separated ``"<A>,<B>"`` pair of them.

    Returns
    -------
    list of str
        Empty when nothing was given, otherwise the comma-separated parts
        with surrounding whitespace stripped. The shape is not validated
        here -- `specter.config.validate_config` rejects anything that is
        not one or two non-empty paths.
    """
    if cryosparc_ref is None:
        return []
    return [part.strip() for part in str(cryosparc_ref).split(",")]


def cryosparc_ref_for_halfset(cryosparc_ref: str | None, halfset: str) -> str | None:
    """
    The CryoSPARC reference a single halfset reconstruction should use.

    A ``halfset="gold"`` run reconstructs both halves from one config, so the
    pair form is the only way to give each half its own reference -- without
    it both halves would be plotted against CryoSPARC's half-map A, which
    silently mislabels the overlay on half B's FSC figure.

    Parameters
    ----------
    cryosparc_ref : str or None
        One path, used for whichever halfset runs, or a comma-separated
        ``"<A>,<B>"`` pair naming one reference per halfset.
    halfset : str
        The halfset being reconstructed, as spelled by
        `ReconstructionConfig.halfset`.

    Returns
    -------
    str or None
        The reference for ``halfset``, or None if none was given.
    """
    refs = parse_cryosparc_ref(cryosparc_ref)
    if not refs:
        return None
    return refs[1] if halfset == "B" and len(refs) == 2 else refs[0]
