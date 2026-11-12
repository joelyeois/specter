"""Load a config dataclass from TOML and apply CLI overrides on top."""

from __future__ import annotations

import os
import tomllib
from dataclasses import fields
from typing import Any, TypeVar, overload

from ._ice import IceCacheConfig
from ._match import MatchConfig
from ._micrograph import MicrographConfig
from ._particle import ParticleStackConfig
from ._reconstruction import ReconstructionConfig
from ._tiltseries import TiltSeriesConfig
from ._tomogram import TomogramConfig

#: TOML keys that have been renamed, mapped to their current spelling. Used
#: only to turn `load_config`'s "unknown field" error into one that says what
#: to write instead.
RENAMED_CONFIG_KEYS = {
    "grid": "carbon_film",
    "specimen_chunk_size": "crowd_chunk_size",
}


ConfigT = TypeVar(
    "ConfigT",
    ParticleStackConfig,
    MicrographConfig,
    TiltSeriesConfig,
    TomogramConfig,
    IceCacheConfig,
    ReconstructionConfig,
    MatchConfig,
)


@overload
def load_config(path: str) -> ParticleStackConfig: ...
@overload
def load_config(path: str, config_cls: type[ConfigT]) -> ConfigT: ...
def load_config(
    path: str,
    config_cls: type[ConfigT] | type[ParticleStackConfig] = ParticleStackConfig,
) -> ConfigT | ParticleStackConfig:
    """
    Load a config dataclass from a TOML file.

    Parameters
    ----------
    path : str
        Path to a TOML config file. May use tables (e.g. `[potential]`) to
        group fields for readability; all tables are flattened into one
        namespace before validation. List-of-tables fields (e.g.
        `[[protein_specs]]`) are passed through as `list[dict]`.
    config_cls : type[ParticleStackConfig] | type[MicrographConfig] | type[TiltSeriesConfig] | type[TomogramConfig] | type[IceCacheConfig] | type[ReconstructionConfig]
        Dataclass to populate from the TOML fields. Defaults to
        `ParticleStackConfig` for backward compatibility.

    Returns
    -------
    ParticleStackConfig | MicrographConfig | TiltSeriesConfig | TomogramConfig | IceCacheConfig | ReconstructionConfig
        Config with unset fields filled from `config_cls` defaults.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat: dict = {}
    where: dict[str, list[str]] = {}
    for table, value in raw.items():
        entries = value if isinstance(value, dict) else {table: value}
        for key in entries:
            where.setdefault(key, []).append(
                f"[{table}]" if isinstance(value, dict) else "the top level"
            )
        flat.update(entries)
    repeated = {key: places for key, places in where.items() if len(places) > 1}
    if repeated:
        # Tables are flattened into one namespace, so a key set in two of
        # them would otherwise be resolved silently by table order.
        detail = "; ".join(
            f"{key!r} in {', '.join(places)}" for key, places in repeated.items()
        )
        raise ValueError(f"{path}: field(s) set more than once: {detail}.")
    unknown = sorted(set(flat) - {f.name for f in fields(config_cls)})
    if unknown:
        # Without this, a stale or mistyped table surfaces as a bare
        # "__init__() got an unexpected keyword argument", which names the
        # key but says nothing about what to write instead.
        detail = ", ".join(
            f"{key!r} (renamed to {RENAMED_CONFIG_KEYS[key]!r})"
            if key in RENAMED_CONFIG_KEYS
            else repr(key)
            for key in unknown
        )
        raise ValueError(
            f"{path}: unknown {config_cls.__name__} field(s): {detail}. "
            "Run the matching `specter` command with --help for the "
            "supported fields."
        )
    # Paths are deliberately NOT rewritten here: a relative path a user wrote
    # in a TOML (or passed on the CLI) is resolved against the current working
    # directory, like every other CLI tool's path argument. Only a leading "~"
    # is expanded, as a shell would have, and only an omitted pdb_cache_dir
    # gets a computed default -- see default_pdb_cache_dir.
    return config_cls(**{k: _expand_home(k, v) for k, v in flat.items()})


def apply_overrides(config: ConfigT, overrides: dict) -> ConfigT:
    """
    Set fields on a config dataclass in place from a dict of overrides.

    Parameters
    ----------
    config : ParticleStackConfig | MicrographConfig
        Config to mutate.
    overrides : dict
        Field name -> value pairs, e.g. from parsed CLI arguments.

    Returns
    -------
    ParticleStackConfig | MicrographConfig
        The same `config` instance, mutated.
    """
    valid = {f.name for f in fields(config)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        # A blind setattr would happily attach an override under a name no
        # field reads, so the flag would parse, be accepted, and do nothing.
        raise ValueError(
            f"apply_overrides: no such field on {type(config).__name__}: "
            f"{', '.join(repr(k) for k in unknown)}. Valid fields: "
            f"{', '.join(sorted(valid))}."
        )
    for key, value in overrides.items():
        setattr(config, key, _expand_home(key, value))
    return config


_PATH_SUFFIXES = ("_dir", "_path", "_file", "_source", "_ref")


def _expand_home(name: str, value: Any) -> Any:
    """
    `value` with a leading ``~`` expanded, when `name` is a path field.

    A shell expands ``~`` before a program sees it, but not inside a TOML
    string or a quoted argument; left verbatim it names a directory called
    ``~`` in the working directory. Applied to path-like fields only (and to
    the same keys inside a list of tables, such as a tomogram's
    ``[[targets]]``), so no other string is touched.
    """
    if isinstance(value, list):
        return [
            {k: _expand_home(k, v) for k, v in item.items()}
            if isinstance(item, dict)
            else _expand_home(name, item)
            for item in value
        ]
    if (
        isinstance(value, str)
        and value.startswith("~")
        and name.endswith(_PATH_SUFFIXES)
    ):
        return os.path.expanduser(value)
    return value
