"""Generic dataclass -> Click option adapter for specter's config-driven CLI commands."""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import rich_click as click
from click.core import ParameterSource


#: Field name -> the string ``--help`` shows in place of the real default.
#: For a default computed from the environment, the real value is a machine
#: -specific absolute path (``/home/<user>/.cache/...``), which is both noisy
#: and misleading in documentation copied from one machine to another. The
#: rule it follows is what the reader actually needs.
DEFAULT_DISPLAY_OVERRIDES: dict[str, str] = {
    "pdb_cache_dir": "$SPECTER_PDB_CACHE, else ~/.cache/specter/pdb",
}


def build_config_options(
    config_cls: type,
    field_help: dict[str, str] | None = None,
    field_groups: list[tuple[str, list[str]]] | None = None,
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
        ``list[...]``-typed fields are skipped entirely (config/TOML-only,
        e.g. a per-atom list sized to the structure -- no single CLI value
        can represent them).
    field_help : dict[str, str], optional
        Field name -> help text describing what the flag itself does. Fields
        not present here fall back to a generic "Overrides the 'x' field in
        --config." message.
    field_groups : list of (str, list of str), optional
        ``(panel title, field names)`` in the order they should appear in
        ``--help``. Each field's panel comes from the group naming it
        (``RichOption``'s ``panel=`` kwarg), and the returned options are
        emitted in this order rather than in dataclass field order. Omit it
        to emit in field order with no panels at all.

    Notes
    -----
    Emission order is what controls panel order: rich-click renders panels in
    the order it first sees them in a command's parameter list. Ordering by
    the dataclass instead would hand panel order to whatever sequence the
    fields happen to be declared in, which is how a single advanced field
    declared early (``pdb_cache_dir``, field #3 of `MicrographConfig`) used
    to drag the whole "Advanced" panel to the top of ``--help``.

    Each option's actual ``default`` stays ``None`` regardless of the
    dataclass's own default -- that's what makes "did the caller pass this
    flag" detectable via ``ctx.get_parameter_source``. The dataclass's real
    default is still shown in ``--help`` via ``show_default`` (a literal
    string override, not the value Click would actually use), so users can
    see what a field falls back to without that value ever being applied as
    a real CLI default. `DEFAULT_DISPLAY_OVERRIDES` replaces that string for
    the few fields whose real default is machine-specific.

    Returns
    -------
    list[click.RichOption]
        One option per dataclass field, suitable for ``click.RichCommand(params=...)``.

    Raises
    ------
    TypeError
        If a field's type isn't one of ``str``, ``int``, ``float``, ``bool``,
        or ``Literal[...]`` (optionally wrapped in ``| None``).
    ValueError
        If ``field_groups`` doesn't name every field that yields an option,
        or names something that isn't one. Both are silent failures
        otherwise: an unlisted field falls into rich-click's catch-all
        "Options" panel next to ``--help``, and a stale name in a group does
        nothing at all.
    """
    field_help = field_help or {}
    panels = {name: title for title, names in (field_groups or []) for name in names}
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

        # list-typed fields (e.g. `atom_species: list[str] | None`) are sized
        # to the structure/dataset, not a single value a flag can hold --
        # config/TOML-only, skip rather than raising.
        if get_origin(ftype) is list:
            continue

        help_text = field_help.get(
            f.name, f"Overrides the '{f.name}' field in --config."
        )
        kwargs: dict[str, Any] = {
            "default": None,
            "help": help_text,
            "panel": panels.get(f.name),
        }

        if f.name in DEFAULT_DISPLAY_OVERRIDES:
            kwargs["show_default"] = DEFAULT_DISPLAY_OVERRIDES[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs["show_default"] = str(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            kwargs["show_default"] = str(f.default_factory())
        # else: no dataclass default at all (a required field, e.g. pdb_source)
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

        # The bare `f.name` decl pins the parameter name to the dataclass
        # field exactly. Without it Click derives the name from the flag by
        # lowercasing, so a mixed-case field (deltaV_V, deltaI_I) would come
        # back from `collect_overrides` as "deltav_v" and never match the
        # field it is meant to override.
        options.append(click.RichOption([f"--{f.name}", f.name], **kwargs))

    if field_groups is None:
        return options

    by_name = {option.name: option for option in options}
    ordered = [
        by_name.pop(name)
        for _, names in field_groups
        for name in names
        if name in by_name
    ]
    stale = sorted(
        name
        for _, names in field_groups
        for name in names
        if name not in {option.name for option in options}
    )
    if by_name or stale:
        problems = []
        if by_name:
            problems.append(f"not in any group: {', '.join(sorted(by_name))}")
        if stale:
            problems.append(f"grouped but not a settable field: {', '.join(stale)}")
        raise ValueError(
            f"build_config_options: {config_cls.__name__} field grouping is "
            f"incomplete -- {'; '.join(problems)}. Every field that yields a "
            "flag needs a group, or it lands in rich-click's catch-all "
            "'Options' panel."
        )
    return ordered


#: Where a user without a checkout finds the worked example configs. They are
#: deliberately not shipped in the wheel: they restate defaults the dataclasses
#: already hold, and a config fetched from a different version than the one
#: installed fails `load_config`'s unknown-key check.
EXAMPLE_CONFIGS_URL = "https://github.com/joelyeois/specter/tree/main/configs"

CONFIG_OPTION_HELP = (
    "TOML config file, loaded before any flag below is applied. Optional: "
    "without it every setting takes its built-in default, so only settings "
    f"that have none must be passed as flags. Worked examples: {EXAMPLE_CONFIGS_URL}"
)


def config_from_defaults(config_cls: type, overrides: dict[str, Any]) -> Any:
    """
    Build a config from the dataclass defaults, for a run that names no TOML.

    ``--config`` is optional because the dataclasses are the single source of
    truth for every default; a TOML file that only restated them would be a
    second copy free to drift. What a dataclass cannot supply is a field with
    no default -- an input the run is *about*, like which structure to
    simulate -- so those become required flags in this mode.

    Deliberately not anchored to a packaged default file. ``configs/`` is not
    in the wheel, so resolving a default path against the source tree only
    works for an editable install from a checkout (see `EXAMPLE_CONFIGS_URL`).

    Parameters
    ----------
    config_cls : type
        The config dataclass to build.
    overrides : dict
        Field name -> value for the flags the caller actually passed, from
        :func:`collect_overrides`.

    Returns
    -------
    object
        An instance of ``config_cls``.

    Raises
    ------
    click.UsageError
        If a field with no default was not supplied as a flag.
    """
    required = [
        f.name
        for f in dataclasses.fields(config_cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    missing = [name for name in required if name not in overrides]
    if missing:
        flags = ", ".join(f"--{name}" for name in missing)
        plural = "s" if len(missing) > 1 else ""
        raise click.UsageError(
            f"No --config was given, so every setting comes from its built-in "
            f"default -- but {flags} ha{'ve' if plural else 's'} none. Pass "
            f"{flags}, or point --config at a TOML file that sets "
            f"{'them' if plural else 'it'}. Worked example configs: "
            f"{EXAMPLE_CONFIGS_URL}"
        )
    return config_cls(**{name: overrides[name] for name in required})


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
