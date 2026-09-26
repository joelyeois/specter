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
