"""Runtime validation for config dataclasses, rejecting physically impossible values."""

from __future__ import annotations

import os
import types
from dataclasses import fields
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeGuard,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ._reconstruction import parse_cryosparc_ref
from ..devices import parse_device
from ._scalar_range import parse_scalar_or_range

# ---------------------------------------------------------------------------
# Validation
#
# Physically impossible values used to travel all the way into the simulation
# before failing -- a zero pixel_size surfaced as "ZeroDivisionError: float
# division by zero", a negative n_pixels as "Trying to create tensor with
# negative dimension", each ~10 s and six frames deep, naming nothing the user
# typed. Worse, a negative dose or an amplitude contrast ratio of 1.5 ran to
# completion and produced a plausible-looking, meaningless stack.
#
# These checks run off the config alone, before a structure is fetched or a
# voxel is written.
# ---------------------------------------------------------------------------


def _fail(field: str, value: Any, requirement: str) -> None:
    raise ValueError(f"{field}={value!r} is invalid: {requirement}.")


def _is_scalar(value: Any) -> TypeGuard[int | float]:
    """
    A plain number this module can compare against a bound.

    Skips None (unset), strings (sentinels like ``batchsize="auto"``, and
    ranges like ``"20,60"`` which `_require_ordered` handles), bools, and
    anything list-like -- several fields hold a ``[low, high]`` pair or a whole
    list of species specs, and ``[0.0, 0.2] < 0`` is a TypeError, not a check.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_positive(config: Any, *fields: str) -> None:
    """Fields that are meaningless at zero (counts, sizes, voltages)."""
    for name in fields:
        value = getattr(config, name, None)
        if not _is_scalar(value):
            continue
        if value <= 0:
            _fail(name, value, "must be greater than 0")


def _require_non_negative(config: Any, *fields: str) -> None:
    """Fields where zero is meaningful ("off") but negative is not."""
    for name in fields:
        value = getattr(config, name, None)
        if not _is_scalar(value):
            continue
        if value < 0:
            _fail(name, value, "must be 0 or greater")


def _require_range(config: Any, name: str, low: float, high: float) -> None:
    value = getattr(config, name, None)
    if not _is_scalar(value):
        return
    if not low <= value <= high:
        _fail(name, value, f"must be between {low} and {high}")


def _require_ordered(config: Any, *fields: str) -> None:
    """
    Scalar-or-[low, high] fields: reject a reversed pair.

    `--dose 60,20` sampled uniformly between 60 and 20, which torch is happy
    to do and which silently yields nothing at all like the intended range.
    """
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        try:
            low, high = parse_scalar_or_range(value)
        except ValueError as exc:
            _fail(name, value, str(exc))
        if low > high:
            _fail(
                name, value, f"range is reversed -- low ({low}) exceeds high ({high})"
            )


def _require_positive_ordered(config: Any, *fields: str) -> None:
    """`_require_ordered`, plus both bounds strictly positive."""
    _require_ordered(config, *fields)
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        low, _ = parse_scalar_or_range(value)
        if low <= 0:
            _fail(name, value, "must be greater than 0")


def _require_non_negative_ordered(config: Any, *fields: str) -> None:
    """`_require_ordered`, plus both bounds at 0 or above."""
    _require_ordered(config, *fields)
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        low, _ = parse_scalar_or_range(value)
        if low < 0:
            _fail(name, value, "must be 0 or greater")


def _require_existing_file(config: Any, *fields: str) -> None:
    for name in fields:
        value = getattr(config, name, None)
        if value is None:
            continue
        if not Path(str(value)).is_file():
            _fail(name, value, "no such file")


def _require_valid_cryosparc_ref(config: Any) -> None:
    """
    ``cryosparc_ref`` names one existing file, or a ``"<A>,<B>"`` pair of them.

    Handled separately from `_require_existing_file` because of the pair
    form, which supplies a per-halfset reference. That is what a
    ``halfset="gold"`` run needs -- it reconstructs both halves from one
    config, so a single reference would put CryoSPARC's half-map A on both
    halves' FSC figures. A pair is meaningless for ``halfset="all"``, which
    reconstructs one volume from every particle rather than a halfset pair.
    """
    value = getattr(config, "cryosparc_ref", None)
    if value is None:
        return

    refs = parse_cryosparc_ref(value)
    if len(refs) > 2 or not all(refs):
        _fail(
            "cryosparc_ref",
            value,
            'must be a path, or a comma-separated "<A>,<B>" pair of them',
        )
    if len(refs) == 2 and getattr(config, "halfset", None) == "all":
        _fail(
            "cryosparc_ref",
            value,
            'must be a single path when halfset="all", which reconstructs one '
            "volume from every particle rather than one per halfset",
        )
    for ref in refs:
        if not Path(ref).is_file():
            _fail("cryosparc_ref", ref, "no such file")


def _require_valid_literals(config: Any) -> None:
    """
    Every ``Literal``-typed field actually holds one of its allowed values.

    Click validates the CLI flags, but a TOML file bypasses that entirely --
    nothing enforces a `Literal` at runtime, so `scattering_model = "banana"`
    in a config file used to sail through to the simulator.
    """
    hints = get_type_hints(type(config))
    for f in fields(config):
        hint = hints.get(f.name)
        if get_origin(hint) in (Union, types.UnionType):
            args = [a for a in get_args(hint) if a is not type(None)]
            hint = args[0] if args else None
        if get_origin(hint) is not Literal:
            continue
        value = getattr(config, f.name, None)
        if value is None:
            continue
        allowed = get_args(hint)
        if value not in allowed:
            _fail(f.name, value, f"must be one of {', '.join(map(repr, allowed))}")


def validate_config(config: Any) -> None:
    """
    Reject physically impossible values before any work is done.

    Called at the top of every pipeline, so it covers both the CLI (which
    reaches a pipeline after `load_config` + `apply_overrides`) and direct
    Python callers constructing a config themselves.

    Parameters
    ----------
    config : ParticleStackConfig | MicrographConfig | TiltSeriesConfig | TomogramConfig

    Raises
    ------
    ValueError
        Naming the offending field, its value, and what was required.
    """
    _require_valid_literals(config)

    # Shared across the imaging configs.
    _require_positive(
        config,
        "n_pixels",
        "pixel_size",
        "voltage",
        "n_particles",
        "n_micrographs",
        "micrograph_size",
        "n_frames",
        "batchsize",
        "n_tilts",
        "voxel_size",
        "crowd_chunk_size",
        "crowd_n_points",
        "render_chunk_size",
        "membrane_region_max_passes",
        # IceCacheConfig. "n"/"dx" are the ice cell's own geometry -- no other
        # config class has a field by either name.
        "num_configs",
        "n",
        "dx",
        "n_steps",
        # ReconstructionConfig.
        "dose_per_angstrom",
        "num_particles",
        "symmetry_batchsize",
        "epochs",
        "bin_factor",
    )
    _require_non_negative(
        config,
        "seed_start",
        "cs",
        "ice_thickness",
        "ice_hole_radius",
        "ice_rim_thickness",
        "shift",
        "bfactor",
        "ice_relax_steps",
        "energy_spread",
        "convergence_angle",
        "cc",
        "deltaV_V",
        "deltaI_I",
        "crowd_min_distance",
        "crowd_max_distance_z",
        "crowd_max_distance_xy",
        "rcut",
        "klim",
        "membrane_min_transmembrane_spacing",
        "filler_table_min_mw_kda",
        "filler_table_max_mw_kda",
        "dose_per_tilt",
        # ReconstructionConfig. A learning rate of exactly 0 is a legitimate
        # way to freeze one parameter group while refining another.
        "lr",
        "lr_R",
        "lr_T",
        "lr_D",
        "lr_decay",
        "sparsity",
        "num_workers",
    )
    _require_range(config, "alpha", 0.0, 1.0)
    _require_range(config, "filler_occupancy_fraction", 0.0, 1.0)
    _require_range(config, "membrane_region_density_threshold", 0.0, 1.0)

    _require_positive_ordered(config, "dose", "potential_scale", "ice_thickness_range")
    _require_non_negative_ordered(
        config, "defocus", "coincidence_radius", "astigmatism", "bead_roughness"
    )
    _require_ordered(config, "astigmatism_angle", "phaseshift")

    _require_existing_file(
        config,
        "cs_path",
        "star_path",
        "shtyrov_params_path",
        # ReconstructionConfig. Every one of these is read at Ghostbuster
        # construction time, so a typo'd path otherwise surfaces minutes in.
        "cs_file",
        "mrc_file",
        "fsc_ref",
        "fsc_mask",
    )
    _require_valid_cryosparc_ref(config)

    # Grammar first, and here rather than in the pipeline: a device string is
    # otherwise parsed several stages into a run, so a typo surfaced either as
    # a raw torch error about valid device types or -- on the reconstruction
    # path, which mapped anything unrecognised to index 0 -- as a silent
    # cuda:0. Both are cheap to catch at load.
    device = getattr(config, "device", None)
    if device is not None:
        try:
            parse_device(str(device))
        except ValueError as exc:
            _fail("device", device, str(exc).split(". ", 1)[-1].rstrip("."))

    # Multi-GPU dispatch (ParticleStackConfig only -- tiltseries/micrograph
    # are single-device) re-executes the whole pipeline once per rank (see
    # pipelines._common._tracked_output_dir's docstring), so auto-assigning
    # a job_id would mean every rank racing to scan the directory
    # independently. Require it pinned explicitly whenever tracking and
    # multi-GPU combine, so every rank's independent config-parse agrees on
    # the same path as a pure string join, without touching the filesystem.
    tracked = getattr(config, "project", None) is not None or (
        getattr(config, "job_id", None) is not None
    )
    if (
        device is not None
        and "," in str(device)
        and tracked
        and getattr(config, "job_id", None) is None
    ):
        _fail(
            "job_id",
            None,
            "must be pinned explicitly when combining project tracking "
            'with a multi-GPU device string (e.g. "0,1"): auto-assigning '
            "a job_id needs one process to decide, but multi-GPU dispatch "
            "re-runs this pipeline once per rank independently",
        )

    min_tilt = getattr(config, "min_tilt_angle", None)
    max_tilt = getattr(config, "max_tilt_angle", None)
    if min_tilt is not None and max_tilt is not None and min_tilt >= max_tilt:
        _fail(
            "min_tilt_angle",
            min_tilt,
            f"must be less than max_tilt_angle ({max_tilt})",
        )

    # A structure given as a path has to exist; a 4-character accession is
    # fetched, so it can only be checked by trying.
    pdb_source = getattr(config, "pdb_source", None)
    if pdb_source and (
        os.sep in str(pdb_source) or str(pdb_source).endswith((".cif", ".pdb"))
    ):
        if not Path(str(pdb_source)).is_file():
            _fail("pdb_source", pdb_source, "looks like a path, but no such file")
