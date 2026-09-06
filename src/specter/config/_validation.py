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
# Without these checks a physically impossible value travels all the way into
# the simulation before failing -- a zero pixel_size surfaces as
# "ZeroDivisionError: float division by zero", a negative n_pixels as "Trying
# to create tensor with negative dimension", each ~10 s and six frames deep,
# naming nothing the user typed. Worse, a negative dose or an amplitude
# contrast ratio of 1.5 runs to completion and produces a plausible-looking,
# meaningless stack.
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


def _require_valid_types(config: Any) -> None:
    """
    Every field actually holds a value of the type it declares.

    TOML is typed, but nothing checked that its types matched the
    dataclass's. The dangerous case was booleans: ``assembly = "false"``
    loads the STRING ``'false'``, which is truthy, so the field silently
    means the opposite of what was written. That reached every config --
    `assembly`, `save_exitwaves`, `write_picks`, `overwrite` -- and it is
    the worst failure class, silently wrong rather than loudly broken. An
    `overwrite = "false"` would have let a run overwrite an existing ice
    library.

    Checked against each field's DECLARED union rather than a guessed
    type, because some unions are deliberately wide: `ScalarOrRange` is
    ``str | float | int | list[float]`` and accepts ``"5000,15000"`` on
    purpose, since a CLI flag can only ever carry a string.

    Two adjustments to plain `isinstance`:

    - ``bool`` is a subclass of ``int`` in Python, so a bool would satisfy
      an ``int`` field. Rejected unless ``bool`` is actually declared.
    - An ``int`` satisfies a ``float`` field, so TOML's ``voltage = 300``
      is accepted for ``float``. Requiring ``300.0`` would be pedantry.

    ``Literal`` fields are skipped; `_require_valid_literals` checks the
    value itself, which is stricter than checking its type.
    """
    hints = get_type_hints(type(config))
    for f in fields(config):
        hint = hints.get(f.name)
        if hint is None:
            continue
        args = (
            [a for a in get_args(hint) if a is not type(None)]
            if get_origin(hint) in (Union, types.UnionType)
            else [hint]
        )
        if any(get_origin(a) is Literal for a in args):
            continue  # _require_valid_literals is stricter

        value = getattr(config, f.name, None)
        if value is None:
            continue

        allowed = tuple(get_origin(a) or a for a in args)
        bool_ok = bool in allowed
        if isinstance(value, bool) and not bool_ok:
            _fail(f.name, value, f"must be {_type_names(allowed)}, not a boolean")
        if isinstance(value, int) and not isinstance(value, bool) and float in allowed:
            continue  # an int is an acceptable float
        if not isinstance(value, allowed):
            _fail(f.name, value, f"must be {_type_names(allowed)}")


def _type_names(allowed: tuple[type, ...]) -> str:
    """Human-readable ``a, b or c`` for a tuple of types."""
    names = [
        {
            "str": "a string",
            "int": "an integer",
            "float": "a number",
            "bool": "true or false",
            "list": "a list",
        }.get(t.__name__, t.__name__)
        for t in allowed
    ]
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen[0] if len(seen) == 1 else ", ".join(seen[:-1]) + " or " + seen[-1]


def _require_valid_literals(config: Any) -> None:
    """
    Every ``Literal``-typed field actually holds one of its allowed values.

    Click validates the CLI flags, but a TOML file bypasses that entirely --
    nothing enforces a `Literal` at runtime, so `scattering_model = "banana"`
    in a config file would otherwise sail through to the simulator.
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


def _require_bfactor_backend(config: Any) -> None:
    """
    Reject `use_deposited_bfactors` on a backend that cannot apply it.

    `PotentialBuilder` raises for these combinations too, but only once a
    structure has been fetched, parsed and typed -- minutes into a tomogram
    run. The condition is knowable from the config alone, so check it here.
    """
    if not getattr(config, "use_deposited_bfactors", False):
        return

    if getattr(config, "scattering_factors", "shtyrov") != "shtyrov":
        _fail(
            "use_deposited_bfactors",
            True,
            "requires scattering_factors='shtyrov' -- Kirkland's and Lobato's "
            "Lorentzian terms convolve with a Gaussian to a Voigt profile, "
            "which has no closed form to voxel-average",
        )

    # Only ParticleStackConfig exposes the voxelization method; everything
    # else always renders 'analytic'.
    if getattr(config, "potential_method", "analytic") != "analytic":
        _fail(
            "use_deposited_bfactors",
            True,
            "requires potential_method='analytic' -- '2d'/'3d' convolve one "
            "precomputed kernel per element group, so a per-atom Gaussian "
            "width has nowhere to go",
        )


def _require_field_rules(config: Any) -> None:
    """Apply each field's own ``check``/``range`` rule, see `setting`."""
    for f in fields(config):
        check = f.metadata.get("check")
        if check == "positive":
            _require_positive(config, f.name)
        elif check == "non_negative":
            _require_non_negative(config, f.name)
        elif check == "ordered":
            _require_ordered(config, f.name)
        elif check == "positive_ordered":
            _require_positive_ordered(config, f.name)
        elif check == "non_negative_ordered":
            _require_non_negative_ordered(config, f.name)
        elif check == "existing_file":
            _require_existing_file(config, f.name)
        elif check is not None:
            raise ValueError(f"{f.name}: unknown check {check!r}")
        bounds = f.metadata.get("range")
        if bounds is not None:
            _require_range(config, f.name, *bounds)


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
    _require_valid_types(config)
    _require_valid_literals(config)

    _require_field_rules(config)
    _require_valid_cryosparc_ref(config)
    _require_bfactor_backend(config)

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
