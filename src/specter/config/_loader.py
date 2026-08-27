"""Load a config dataclass from TOML and apply CLI overrides on top."""

from __future__ import annotations

import tomllib
from dataclasses import fields
from typing import TypeVar, overload

from ._ice import IceCacheConfig
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
)


@overload
def load_config(
    path: str, config_cls: type[ParticleStackConfig] = ...
) -> ParticleStackConfig: ...
@overload
def load_config(path: str, config_cls: type[MicrographConfig]) -> MicrographConfig: ...
@overload
def load_config(path: str, config_cls: type[TiltSeriesConfig]) -> TiltSeriesConfig: ...
@overload
def load_config(path: str, config_cls: type[TomogramConfig]) -> TomogramConfig: ...
@overload
def load_config(path: str, config_cls: type[IceCacheConfig]) -> IceCacheConfig: ...
@overload
def load_config(
    path: str, config_cls: type[ReconstructionConfig]
) -> ReconstructionConfig: ...
def load_config(
    path: str,
    config_cls: type[ParticleStackConfig]
    | type[MicrographConfig]
    | type[TiltSeriesConfig]
    | type[TomogramConfig]
    | type[IceCacheConfig]
    | type[ReconstructionConfig] = ParticleStackConfig,
) -> (
    ParticleStackConfig
    | MicrographConfig
    | TiltSeriesConfig
    | TomogramConfig
    | IceCacheConfig
    | ReconstructionConfig
):
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
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
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
    # directory, like every other CLI tool's path argument. Only an omitted
    # pdb_cache_dir gets a computed default -- see default_pdb_cache_dir.
    return config_cls(**flat)


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
        setattr(config, key, value)
    return config
