"""The scalar-or-[low, high] field convention shared by every config dataclass."""

from __future__ import annotations

from typing import Any

#: A field that is either one number (constant for every particle) or a
#: two-element range sampled uniformly per particle. In TOML, write these as
#: plain numbers -- ``dose = 20`` or ``defocus = [5000, 15000]`` -- so config
#: files never mix quoted and unquoted numbers. The string forms (``"20"``,
#: ``"5000,15000"``) remain valid: CLI flags can only ever carry strings, and
#: older configs still parse. See :func:`parse_scalar_or_range`.
#:
#: ``str`` is deliberately first in the union: `cli/_click_options.py` types a
#: union-valued flag from its first member, and only ``str`` accepts both the
#: ``8000`` and ``5000,15000`` spellings on a command line.
ScalarOrRange = str | float | int | list[float]


def parse_scalar_or_range(value: ScalarOrRange) -> tuple[float, float]:
    """
    Parse a scalar or a two-element range into a (low, high) range.

    A bare scalar becomes a zero-width range (``low == high``), so callers
    can always uniformly sample between the two bounds without special-
    casing the constant case.

    Parameters
    ----------
    value : str or float or int or list of float
        A single number (e.g. ``20`` or ``"20"``), or a low/high pair as
        either a two-element sequence (e.g. ``[5000, 15000]``, the form a
        TOML config should use) or a comma-separated string
        (e.g. ``"5000,15000"``, the form a CLI flag has to use).

    Returns
    -------
    tuple[float, float]
        ``(low, high)``.

    Raises
    ------
    ValueError
        If ``value`` isn't one or two numbers.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), float(value)
    if isinstance(value, (list, tuple)):
        parts: list[Any] = list(value)
    else:
        parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        v = float(parts[0])
        return v, v
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError(f"Expected a scalar or a [low, high] pair, got {value!r}")
