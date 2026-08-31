"""Generic dataclass -> Click option adapter for specter's config-driven CLI commands."""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Literal, NoReturn, Union, get_args, get_origin, get_type_hints

import rich_click as click

from specter.config import apply_overrides, load_config, validate_config
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


def prerequisite_usage_error(message: str) -> NoReturn:
    """
    Report an unmet config prerequisite as a CLI usage error.

    For preconditions :func:`config_from_defaults` structurally cannot
    catch. That function reports a field whose dataclass declares no
    default, which covers a single required setting like `pdb_source`.
    It cannot express a DISJUNCTION -- "at least one of these ten", or
    "this field or that argument" -- because every field involved does
    have a default, individually.

    Those preconditions are checked in `pipelines/`, which correctly
    raises `ValueError`: for a Python API caller a traceback is the right
    answer, and it points at the line that failed. Reaching a CLI user
    that way is not: `specter build tomogram` printed 32 lines, 26 of
    them click and rich_click internals, and named `config.targets` and
    the rest -- identifiers with no CLI spelling, since seven of those
    ten sources are TOML-only and have no flag at all. This raises
    `click.UsageError` instead, so the CLI path gets the same bordered
    panel and exit code 2 as every other command, and the pipeline keeps
    its own guard for direct callers.

    Parameters
    ----------
    message : str
        What is missing and how to supply it, in CLI vocabulary: flags
        and config file names, not config field names. The worked-configs
        URL is appended.

    Raises
    ------
    click.UsageError
        Always.
    """
    raise click.UsageError(f"{message} Worked example configs: {EXAMPLE_CONFIGS_URL}")


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


def load_validated_config(
    config_cls: type, config_path: str | None, overrides: dict[str, Any]
) -> Any:
    """
    Build a config for a CLI run, reporting anything wrong with it as a
    usage error.

    Everything a config can be wrong about reaches the user through a
    `click.UsageError` here, rather than as a traceback out of the
    pipeline. The messages were already good; they were just arriving
    badly. A bad value in a TOML file used to print 38 lines ending in
    ``ValueError: noise_model='bogus' is invalid: must be one of
    'poisson', 'none'``, while the identical mistake as a flag printed 8
    lines in a bordered panel, because click builds a `Choice` from each
    field's `Literal` and catches it at parse time. A typo in a config
    file is the likelier of the two.

    Three failure modes are covered: a value the field does not allow
    (`ValueError` from `validate_config`), a key the dataclass does not
    have or TOML that will not parse (`ValueError`, `tomllib` raising
    `TOMLDecodeError`, a subclass), and a `--config` path that does not
    exist (`FileNotFoundError`).

    `validate_config` runs here as well as at the top of every pipeline.
    It is a pure check that rejects impossible values before any work
    happens, so running it twice costs nothing, and the pipeline keeps
    its own call for direct Python callers -- where a traceback is the
    right answer, since it points at the line that failed.

    Parameters
    ----------
    config_cls : type
        The config dataclass for this command.
    config_path : str or None
        ``--config``, or None to build from dataclass defaults.
    overrides : dict
        Field name -> value for flags the caller actually passed, from
        :func:`collect_overrides`.

    Returns
    -------
    object
        A validated instance of `config_cls`.

    Raises
    ------
    click.UsageError
        For any of the above.
    """
    try:
        config = (
            load_config(config_path, config_cls)
            if config_path is not None
            else config_from_defaults(config_cls, overrides)
        )
        apply_overrides(config, overrides)
        validate_config(config)
    except click.UsageError:
        raise  # config_from_defaults already raises these, phrased for the CLI
    except (ValueError, FileNotFoundError) as exc:
        raise click.UsageError(str(exc)) from exc
    return config


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
