"""
Shared `setting()` factories for the output, job-tracking and seed fields.

Every command that writes output declares ``output_dir``, ``project``,
``job_id`` and ``seed``, and what those fields mean is the same everywhere
(see `specter.pipelines._common.resolve_output_dir`). Declaring them once
here keeps their ``--help`` text from drifting apart between commands, while
each factory still takes the per-command detail -- what the command writes,
its job-type folder, whether it runs under multi-GPU dispatch -- as an
argument.
"""

from __future__ import annotations

from typing import Any

from ._field import setting

#: Why a tracked run under Lightning DDP needs its ``job_id`` pinned; matches
#: the rule `validate_config` enforces for `_dispatches_via_ddp` configs.
DDP_JOB_ID_NOTE = (
    "Mandatory when combining tracking with multi-GPU device strings -- "
    "auto-numbering needs one process to decide, but multi-GPU dispatch "
    "re-runs this pipeline once per rank."
)


def output_dir_setting(writes: str, job_type: str) -> Any:
    """
    The ``output_dir`` field of a command whose tracking is opt-in.

    Parameters
    ----------
    writes : str
        What the command does in that directory, completing "Directory to
        ...", e.g. ``"save .mrcs and .star files"``.
    job_type : str
        The command's job-type folder, which is also its untracked default
        (`default_output_dir`), e.g. ``"particles"``.

    Returns
    -------
    dataclasses.Field
    """
    return setting(
        None,
        help=(
            f"Directory to {writes} when untracked. Setting --project or "
            "--job_id instead makes this the root of the numbered job tree, so "
            "tracking organises output within the folder you chose rather than "
            f"moving it elsewhere. Unset defaults to {job_type}/ untracked, and "
            "to the project root found by walking up from cwd for an existing "
            ".specter marker when tracked."
        ),
    )


def project_setting(job_type: str, note: str = "") -> Any:
    """
    The ``project`` field of a command whose tracking is opt-in.

    Parameters
    ----------
    job_type : str
        The command's job-type folder, e.g. ``"particles"``.
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Optional: number and track this run through specter.jobs. Not "
        "required for tracking -- job_id alone also triggers it. The run lands "
        f"in <output_dir>/[<project>/]{job_type}/J00N/ with a job.json recording "
        "every parameter, the git commit and the run's status."
    )
    return setting(None, help=f"{help_text} {note}" if note else help_text)


def job_id_setting(note: str = "") -> Any:
    """
    The ``job_id`` field shared by every tracked command.

    Parameters
    ----------
    note : str, optional
        A command-specific sentence appended to the help, e.g.
        `DDP_JOB_ID_NOTE` for a command that runs under multi-GPU dispatch.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Pin the job directory (e.g. J001) rather than auto-assigning the next "
        "one: resumes into it if it exists, creates it otherwise."
    )
    return setting(None, help=f"{help_text} {note}" if note else help_text)


def seed_setting(draws: str, unset: str = "Auto-generated and logged if unset.") -> Any:
    """
    The ``seed`` field of a simulating command.

    Parameters
    ----------
    draws : str
        What the seed controls, completing "RNG seed for ...".
    unset : str, optional
        What happens when no seed is given.

    Returns
    -------
    dataclasses.Field
    """
    return setting(None, help=f"RNG seed for {draws}. {unset}")


# --- Absorption and exposure physics ---------------------------------------
#
# Shared by the particle, micrograph and tilt-series configs, which model the
# same physics through the same settings groups (`Propagation`, `Optics`,
# `Envelopes`, `Ice`), so the fields read the same in every command's help.


def absorption_model_setting() -> Any:
    """The ``absorption_model`` field."""
    return setting(
        "alpha",
        help=(
            "Where the imaginary potential comes from. 'alpha' scales the real "
            "potential by the amplitude-contrast ratio, tying absorption to every "
            "atomic cusp. 'inelastic_mfp' derives it per material from a measured "
            "inelastic mean free path instead, and ignores alpha (including a "
            ".cs/.star file's, which CTF estimation takes as an input and never "
            "fits)."
        ),
    )


def inelastic_mfp_solvent_setting() -> Any:
    """The ``inelastic_mfp_solvent`` field, in Angstrom."""
    return setting(
        None,
        help=(
            "Inelastic mean free path of the ice, in Angstrom, for "
            "absorption_model='inelastic_mfp'. Unset takes the measured value for "
            "the run's voltage (3950 at 300 kV, 2030 at 120 kV), and elsewhere an "
            "estimate from those two with a warning (200 kV: 3040 +/- 7%; 100 kV: "
            "1730 +/- 20%)."
        ),
        check="positive",
    )


def inelastic_mfp_specimen_setting() -> Any:
    """The ``inelastic_mfp_specimen`` field, in Angstrom."""
    return setting(
        None,
        help=(
            "Inelastic mean free path of the specimen, in Angstrom, for "
            "absorption_model='inelastic_mfp'. Unset gives the specimen the ice's "
            "value, so it absorbs like the water it displaces and carries no "
            "absorption contrast. 2460 is the derived value for protein at 300 kV."
        ),
        check="positive",
    )


def objective_aperture_setting() -> Any:
    """The ``objective_aperture`` field, in milliradians."""
    return setting(
        None,
        help=(
            "Objective aperture semi-angle in milliradians (a 70 um aperture on a "
            "Krios is ~12 mrad), for absorption_model='inelastic_mfp'. Electrons "
            "scattered elastically beyond it leave the image; the loss is charged "
            "per material from the scattering cross section, since no practical "
            "grid carries it (2.7% of the beam through 400 Angstrom of ice at "
            "12 mrad, 300 kV). Unset: no aperture."
        ),
        check="positive",
    )


def dose_envelope_target_setting(owner: str, note: str = "") -> Any:
    """
    The ``dose_envelope_target`` field.

    Parameters
    ----------
    owner : str
        What is damaged, e.g. ``"particle"`` or ``"specimen"``.
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Where the dose envelope acts. 'transfer_function' filters the whole "
        f"image, solvent included. 'specimen' damages the {owner}'s own "
        "potential before the ice is added, with occupancy read from the "
        f"undamaged {owner}, so the water keeps its 3.7 A ring (raw movies "
        "show it does not fade with dose); required for ice_motion_variance."
    )
    return setting(
        "transfer_function", help=f"{help_text} {note}" if note else help_text
    )


def ice_motion_variance_setting(note: str = "") -> Any:
    """
    The ``ice_motion_variance`` field, in A^2 per e-/A^2.

    Parameters
    ----------
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Beam-induced displacement of the water, per axis, in A^2 per e-/A^2 "
        "(McMullan et al. 2015's sigma0^2, 0.38 for their 300 kV exposure, "
        "as measured from the 3.7 A ring). The ice fluctuation is filtered to "
        "what survives the summed exposure, frame weights included, using a "
        "decorrelation measured on relaxed ice trajectories; its mean is kept. "
        "With dose_envelope on, needs dose_envelope_target='specimen'. Unset: "
        "frozen ice."
    )
    return setting(
        None, help=f"{help_text} {note}" if note else help_text, check="non_negative"
    )
