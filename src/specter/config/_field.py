"""
`setting()`: a config field that carries its own help text and validation
rule, so a field, what `--help` says about it, and what `validate_config`
requires of it are one declaration.
"""

from __future__ import annotations

from dataclasses import MISSING, field, fields
from typing import Any, Callable, Literal

#: What `validate_config` requires of a field's value, when set:
#: ``"positive"`` (> 0), ``"non_negative"`` (>= 0), ``"ordered"`` (a
#: scalar-or-[low, high] whose pair is not reversed), ``"positive_ordered"``
#: and ``"non_negative_ordered"`` (ordered, with the bound on both ends), and
#: ``"existing_file"`` (a path that resolves to a file).
Check = Literal[
    "positive",
    "non_negative",
    "ordered",
    "positive_ordered",
    "non_negative_ordered",
    "existing_file",
]


def setting(
    default: Any = MISSING,
    *,
    help: str | None = None,
    factory: Callable[[], Any] | None = None,
    check: Check | None = None,
    range: tuple[float, float] | None = None,
) -> Any:
    """
    Declare a config field with its help text and validation rule.

    Parameters
    ----------
    default : Any, optional
        The field's default. Omit it (and ``factory``) for a required field.
    help : str, optional
        What the field means, shown by ``--help`` and the CLI reference.
    factory : callable, optional
        A default factory, for a mutable default such as a list.
    check : str, optional
        The rule `validate_config` applies, see `Check`.
    range : tuple of float, optional
        Inclusive ``(low, high)`` bounds a scalar value must fall within.

    Returns
    -------
    dataclasses.Field
    """
    metadata: dict[str, Any] = {}
    if help is not None:
        metadata["help"] = help
    if check is not None:
        metadata["check"] = check
    if range is not None:
        metadata["range"] = range
    if factory is not None:
        if default is not MISSING:
            raise ValueError("give either default or factory, not both")
        return field(default_factory=factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def help_of(config_cls: type) -> dict[str, str]:
    """The ``{field name: help text}`` of every field declared with a help."""
    return {
        f.name: f.metadata["help"] for f in fields(config_cls) if "help" in f.metadata
    }
