"""
`--config` is optional, and must not default to a path.

`configs/` is not packaged in the wheel, so a `--config` default resolved
against the source tree only exists for an editable install from a checkout.
Installed from a wheel, every config-taking command used to die in
`tomllib.load` on a path under the virtualenv's ``lib/``. Defaults now come
from the config dataclasses, which are the single source of truth for them.
"""

from __future__ import annotations

import dataclasses
import subprocess as proc
import sys

import pytest
import rich_click as click

from specter.cli._click_options import EXAMPLE_CONFIGS_URL, config_from_defaults
from specter.cli._cli import cli
from specter.config import (
    IceCacheConfig,
    MicrographConfig,
    ParticleStackConfig,
    ReconstructionConfig,
    TiltSeriesConfig,
    TomogramConfig,
)

CONFIG_COMMANDS = [
    ("simulate", "particles"),
    ("simulate", "micrograph"),
    ("simulate", "tiltseries"),
    ("build", "tomogram"),
    ("build", "ice"),
    ("reconstruct", "particle"),
]


def _config_option(path: tuple[str, ...]) -> click.Option:
    cmd: click.Command = cli
    for name in path:
        cmd = cmd.commands[name]  # type: ignore[attr-defined]
    for param in cmd.params:
        if param.name == "config":
            return param  # type: ignore[return-value]
    raise AssertionError(f"{' '.join(path)} has no --config option")


@pytest.mark.parametrize("path", CONFIG_COMMANDS, ids=lambda p: " ".join(p))
def test_config_option_has_no_path_default(path: tuple[str, ...]) -> None:
    """A filesystem default here only resolves for an editable install."""
    assert _config_option(path).default is None


@pytest.mark.parametrize(
    "config_cls",
    [
        ParticleStackConfig,
        MicrographConfig,
        TiltSeriesConfig,
        TomogramConfig,
        IceCacheConfig,
        ReconstructionConfig,
    ],
    ids=lambda c: c.__name__,
)
def test_every_config_builds_from_defaults_plus_required(config_cls: type) -> None:
    """Only fields with no default need supplying, and supplying them suffices."""
    required = [
        f.name
        for f in dataclasses.fields(config_cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    cfg = config_from_defaults(config_cls, {name: "x" for name in required})
    assert isinstance(cfg, config_cls)
    for name in required:
        assert getattr(cfg, name) == "x"


def test_missing_required_field_names_the_flag_and_the_examples() -> None:
    """The error has to say which flag to pass and where the example configs are."""
    with pytest.raises(click.UsageError) as excinfo:
        config_from_defaults(ParticleStackConfig, {})
    message = str(excinfo.value)
    assert "--pdb_source" in message
    assert EXAMPLE_CONFIGS_URL in message


def test_particles_without_config_or_pdb_source_exits_cleanly(tmp_path) -> None:
    """No traceback, and the message points at the flag rather than a missing file."""
    result = proc.run(
        [sys.executable, "-m", "specter.cli._cli", "simulate", "particles"],
        capture_output=True,
        encoding="utf-8",
        cwd=str(tmp_path),
        env={"COLUMNS": "200", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 2
    combined = (result.stdout + result.stderr).replace("\n", " ")
    assert "--pdb_source" in combined
    assert "Traceback" not in combined
