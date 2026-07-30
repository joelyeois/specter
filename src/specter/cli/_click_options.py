"""Generic dataclass -> Click option adapter for specter's config-driven CLI commands."""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import rich_click as click
from click.core import ParameterSource


def build_config_options(
    config_cls: type,
    field_help: dict[str, str] | None = None,
    field_panels: dict[str, str] | None = None,
) -> list[click.RichOption]:
    """
    Build one ``--field-name`` Click option per field of a config dataclass.

    Every option defaults to ``None`` and is left unset unless the caller
    passes it -- combined with :func:`specter.config.apply_overrides` and
    Click's ``ctx.get_parameter_source``, this means CLI flags only ever
    override the fields a caller actually set on the command line, leaving
    everything else at whatever the loaded TOML config (or the dataclass's
    own defaults) already has.

    Parameters
    ----------
    config_cls : type
        A dataclass whose fields become CLI options. ``Literal[...]`` fields
        become ``click.Choice``; ``bool`` fields accept Click's flexible
        boolean parsing (``true``/``false``/``1``/``0``/``yes``/``no``,
        case-insensitive); ``X | None`` fields are treated as ``X``.
    field_help : dict[str, str], optional
        Field name -> help text describing what the flag itself does. Fields
        not present here fall back to a generic "Overrides the 'x' field in
        --config." message.
    field_panels : dict[str, str], optional
        Field name -> rich-click panel title, used to group flags into
        titled, bordered panels in ``--help`` output (``RichOption``'s
        ``panel=`` kwarg). Fields not present here get no explicit panel and
        fall back to rich-click's default "Options" panel.

    Notes
    -----
    Each option's actual ``default`` stays ``None`` regardless of the
    dataclass's own default -- that's what makes "did the caller pass this
    flag" detectable via ``ctx.get_parameter_source``. The dataclass's real
    default is still shown in ``--help`` via ``show_default`` (a literal
    string override, not the value Click would actually use), so users can
    see what a field falls back to without that value ever being applied as
    a real CLI default.

    Returns
    -------
    list[click.RichOption]
        One option per dataclass field, suitable for ``click.RichCommand(params=...)``.

    Raises
    ------
    TypeError
        If a field's type isn't one of ``str``, ``int``, ``float``, ``bool``,
        or ``Literal[...]`` (optionally wrapped in ``| None``).
    """
    field_help = field_help or {}
    field_panels = field_panels or {}
    hints = get_type_hints(config_cls)
    options = []

    for f in dataclasses.fields(config_cls):
        ftype: Any = hints[f.name]

        # `X | None` (PEP 604) resolves to types.UnionType, while
        # `Optional[X]`/`Union[X, None]` resolves to typing.Union -- config.py
        # uses the former throughout, but handle both.
        if get_origin(ftype) in (Union, types.UnionType):
            non_none = [a for a in get_args(ftype) if a is not type(None)]
            ftype = non_none[0]

        help_text = field_help.get(
            f.name, f"Overrides the '{f.name}' field in --config."
        )
        kwargs: dict[str, Any] = {
            "default": None,
            "help": help_text,
            "panel": field_panels.get(f.name),
        }

        if f.default is not dataclasses.MISSING:
            kwargs["show_default"] = str(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            kwargs["show_default"] = str(f.default_factory())
        # else: no dataclass default at all (a required field, e.g. pdb_code)
        # -- leave show_default unset, nothing meaningful to display.

        if get_origin(ftype) is Literal:
            kwargs["type"] = click.Choice(list(get_args(ftype)))
        elif ftype is bool:
            kwargs["type"] = bool
            kwargs["metavar"] = "True|False"
        elif ftype in (str, int, float):
            kwargs["type"] = ftype
        else:
            raise TypeError(
                f"build_config_options: unsupported field type {ftype!r} for "
                f"'{config_cls.__name__}.{f.name}'."
            )

        options.append(click.RichOption([f"--{f.name}"], **kwargs))

    return options


def collect_overrides(ctx: click.Context, exclude: set[str]) -> dict[str, Any]:
    """
    Collect only the CLI parameters a caller explicitly passed on the command line.

    Parameters
    ----------
    ctx : click.Context
        The current command's Click context.
    exclude : set[str]
        Parameter names to always omit (e.g. ``"config"``).

    Returns
    -------
    dict[str, Any]
        Field name -> value, for parameters whose source is the command line.
    """
    return {
        name: value
        for name, value in ctx.params.items()
        if name not in exclude
        and ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
    }
