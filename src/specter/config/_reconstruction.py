"""ReconstructionConfig: parameters for single-particle reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    cs_file: str
    mrc_file: str
    # Total fluence per image, e-/A^2. Sets the Poisson statistics the loss is
    # weighted by, so it has to be the real value for the dataset.
    dose_per_angstrom: float
    # Which gold-standard half-set to reconstruct. "gold" (the default)
    # reconstructs A and B and computes the halfmap FSC between them; "A"/"B"
    # select a single halfset (and name the outputs vol_A.mrc / vol_B.mrc),
    # useful for a quick test of one half; "all" uses every particle in one
    # single-volume run, ignoring the split entirely.
    halfset: Literal["A", "B", "all", "gold"] = "gold"
    # Reconstruct only the first N particles. Useful for a quick run before
    # committing to the full stack; None uses all of them.
    num_particles: int | None = None

    # --- Optimisation ---
    epochs: int = 5
    batchsize: int = 3
    # Volume learning rate. None freezes the volume, which is what refining
    # poses alone means.
    lr: float | None = 0.1
    scheduler: Literal[
        "LambdaLR",
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
        "MultiplicativeLR",
    ] = "LambdaLR"
    lr_decay: float = 0.1

    # --- Symmetry ---
    symmetry: str | None = None
    symmetry_mode: Literal["real", "fourier"] = "fourier"
    symmetry_batchsize: int | None = None

    # --- Sanity check ---
    # Run one epoch on bin_factor-binned images and stop, instead of the full
    # fit. Every file path, .cs field and physics setting is exercised, at a
    # fraction of the cost -- run this before a multi-hour job.
    test_run: bool = False
    bin_factor: int = 8

    # --- Compute ---
    # "cpu" | "cuda" | "cuda:N" | a bare GPU index | a comma-separated list of
    # GPU indices ("0,1"), which trains across them via Lightning DDP.
    device: str = "cuda"
    precision: str = "16-mixed"
    num_workers: int = 8

    # --- Output & job tracking ---
    # Every run is numbered and routed through `specter.jobs`: the directory
    # becomes job_base_dir/[project/]reconstructions/<job_id>/, numbered
    # J001, J002, ... (shared across every job type in the project, not just
    # reconstructions), with a job.json recording the full parameter set,
    # git commit and status. That is what makes two halfset runs (halfset
    # "A" then "B") shareable into one job directory, and what lets a batch
    # script pin a job_id up front and resume into it.
    #
    # `project` is optional, not required: leaving it unset doesn't mean
    # "untracked" -- it drops just the project-name segment
    # (job_base_dir/reconstructions/<job_id>/), the implicit default project
    # for whatever job_base_dir resolves to. Pass `--project` to split one
    # job_base_dir into several named projects, e.g. one shared scratch
    # directory used across unrelated structures.
    project: str | None = None
    job_id: str | None = None
    # Defaults to the project root discovered by walking up from cwd looking
    # for an existing specter-data/ (find_specter_project_root() --
    # find_specter_project_root()/specter-data), the same way `git` resolves
    # the nearest ancestor containing .git -- so running from a subdirectory
    # of an already-initialised project still lands in the same project,
    # rather than starting a second, disconnected specter-data/ tree (and
    # job numbering restarting from J001) right where you happened to be
    # standing.
    job_base_dir: str | None = None

    # --- Reference maps (FSC logging only, never optimised against) ---
    fsc_ref: str | None = None
    fsc_mask: str | None = None
    cryosparc_ref: str | None = None
    # Rotate fsc_mask per particle, project it to 2D, and weight the
    # image-domain loss by it.
    use_2d_mask: bool = False

    # --- Refinement (unverified) ---
    # Pose and defocus refinement. Wired in but unverified: no test checks
    # recovered rotations, translations or defocus against ground truth, so
    # treat a run with these set as an experiment, not a result.
    lr_R: float | None = None
    lr_T: float | None = None
    lr_D: float | None = None
    # Constant offset in Angstrom added to every particle's dfu/dfv before the
    # transfer function is built. The starting point for lr_D refinement.
    defocus_offset: float = 0.0

    # --- Forward model ---
    # Defaults to rytov rather than the simulator's multislice: reconstruction
    # runs the forward model once per particle per step, and rytov is the
    # cheapest model that still carries curvature of the Ewald sphere.
    scattering_model: Literal["multislice", "rytov", "firstborn", "projection"] = (
        "rytov"
    )
    aberration_model: Literal["holography", "ctf"] = "holography"
    aberration_backend: Literal["legacy", "torch_ctf"] = "legacy"
    ews_curvature_sign: Literal["negative", "positive"] = "negative"
    bfactor: float | None = None
    # Hard frequency cutoff (1/A) applied to the simulated images.
    klim: float | None = None
    sparsity: float | None = None
    rotate_mode: Literal["real", "fourier"] = "real"
    learn_noise_model: bool = False
    use_ncc: bool = False


RECONSTRUCTION_HELP: dict[str, str] = {
    "cs_file": "CryoSPARC .cs file holding the particle poses, CTF "
    "parameters and per-particle scale.",
    "mrc_file": "Particle stack (.mrc/.mrcs) the .cs file indexes into.",
    "dose_per_angstrom": "Total fluence per image in e-/Angstrom^2. Sets the "
    "Poisson statistics the loss is weighted by, so it must be the dataset's "
    "real value.",
    "halfset": "Which gold-standard half-set to reconstruct. gold (the "
    "default) reconstructs A and B and computes the halfmap FSC between "
    "them; A or B alone reconstructs just that half, e.g. for a quick test; "
    "all uses every particle in one single-volume run, ignoring the split.",
    "num_particles": "Reconstruct only the first N particles instead of the "
    "whole stack.",
    "fsc_ref": "Reference volume (.mrc) for map-to-model FSC logging. Never "
    "optimised against -- reporting only.",
    "fsc_mask": "Mask volume (.mrc) applied before the FSC is computed.",
    "cryosparc_ref": "CryoSPARC's own reconstruction (.mrc), plotted "
    "alongside --fsc_ref for comparison. Only used when --fsc_ref is given "
    "too.",
    "use_2d_mask": "Rotate --fsc_mask per particle, project it to 2D, and "
    "weight the image-domain loss by it.",
    "scattering_model": "Wave propagation model used by the forward pass. "
    "rytov is the default here rather than multislice: the model runs once "
    "per particle per step, and rytov is the cheapest one that still carries "
    "Ewald-sphere curvature.",
    "aberration_model": "Aberration model applied to the exit wave.",
    "aberration_backend": "Which engine computes the transfer function. "
    "legacy uses aberrations.Aberration; torch_ctf uses the verified "
    "ctf.LegacyAberrationAdapter port.",
    "ews_curvature_sign": "Sign convention for Ewald-sphere curvature. "
    "negative matches CryoSPARC.",
    "bfactor": "Isotropic B-factor envelope in Angstrom^2 damping "
    "high-resolution signal in the forward model.",
    "defocus_offset": "Constant offset in Angstrom added to every particle's "
    "dfu/dfv. The starting point when --lr_D refinement is enabled.",
    "klim": "Hard frequency cutoff (1/Angstrom) applied to the simulated images.",
    "symmetry": "Point-group symmetry enforced on the volume, e.g. C3, D7, "
    "I1. Omit for C1.",
    "symmetry_batchsize": "Batch size used when applying symmetry operators "
    "to the volume. Lower it if symmetry expansion runs out of memory.",
    "symmetry_mode": "Domain in which symmetry is applied.",
    "epochs": "Number of passes over the particle stack.",
    "batchsize": "Particles per optimisation step.",
    "lr": "Learning rate for the volume. Unset (None) freezes the volume, "
    "which is what refining poses alone means.",
    "lr_R": "Learning rate for per-particle rotations. UNVERIFIED: pose "
    "refinement is wired in but no test checks recovered rotations against "
    "ground truth.",
    "lr_T": "Learning rate for per-particle translations. UNVERIFIED, see --lr_R.",
    "lr_D": "Learning rate for the defocus offset. UNVERIFIED, see --lr_R.",
    "scheduler": "Learning-rate schedule for the volume optimiser.",
    "lr_decay": "Decay rate for the LambdaLR schedule: multiplier = "
    "1 / (1 + lr_decay * sqrt(global_step)). Unused by other schedulers.",
    "sparsity": "L1 regularisation weight on the volume.",
    "rotate_mode": "Domain in which per-particle rotations are applied.",
    "learn_noise_model": "Estimate sigma^2(k) from the residuals during "
    "training (RELION-style) instead of assuming it.",
    "use_ncc": "Use a normalised cross-correlation loss instead of MSE.",
    "test_run": "Run one epoch on binned images and stop, instead of the "
    "full fit. Exercises every path, .cs field and physics setting at a "
    "fraction of the cost -- do this before a multi-hour job.",
    "bin_factor": "Spatial binning factor used by --test_run.",
    "device": "cpu | cuda | cuda:N | a bare GPU index | a comma-separated "
    "list of GPU indices (0,1), which trains across them via Lightning DDP "
    "with gradients all-reduced every step.",
    "precision": "Lightning trainer precision, e.g. 16-mixed or 32. Forced "
    "to 32 on CPU.",
    "num_workers": "Dataloader worker processes.",
    "project": "Name for a group of jobs, e.g. one structure's worth of "
    "runs. Optional: omitting it doesn't mean untracked -- every run is "
    "numbered and gets a job.json regardless -- it just drops the "
    "project-name segment, using job_base_dir's implicit default project "
    "instead of a named one. Pass this to split one job_base_dir into "
    "several, e.g. one shared scratch directory used across structures.",
    "job_id": "Pin the job directory (e.g. J001) rather than auto-assigning "
    "the next one: resumes into it if it exists, creates it otherwise. This "
    "is how two halfset runs share one job.",
    "job_base_dir": "Root directory for job folders. Defaults to the "
    "project root found by walking up from cwd looking for an existing "
    "specter-data/, the same way git finds the nearest .git -- so running "
    "from a subdirectory of an already-initialised project lands in the "
    "same project.",
}
