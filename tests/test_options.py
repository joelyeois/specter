"""
The option vocabularies in `specter.options` are the wide end of every
enumerated switch: a config field may expose a subset, never a value the
implementing class does not accept.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Union, get_args, get_origin, get_type_hints

import pytest

from specter import options
from specter.config import (
    MatchConfig,
    MicrographConfig,
    ParticleStackConfig,
    ReconstructionConfig,
    TiltSeriesConfig,
    TomogramConfig,
)

# config field name -> the alias its values must be drawn from
_FIELD_ALIAS = {
    "scattering_model": options.ScatteringModel,
    "noise_model": options.NoiseModel,
    "detector_model": options.DetectorModel,
    "ice_model": options.IceModel,
    "ews_curvature_sign": options.EwaldSphereSign,
    "scattering_factors": options.ScatteringFactors,
    "bulk_scattering_factors": options.ScatteringFactors,
    "potential_method": options.PotentialMethod,
    "conv_backend": options.ConvBackend,
    "rotate_mode": options.RotateMode,
    "symmetry_mode": options.RotateMode,
    "tilt_axis": options.TiltAxis,
    "scheduler": options.Scheduler,
}

_CONFIGS = [
    ParticleStackConfig,
    MicrographConfig,
    TiltSeriesConfig,
    TomogramConfig,
    ReconstructionConfig,
    MatchConfig,
]

# MatchConfig.detector_model takes "unknown" for a dataset whose camera is
# not on the acquisition card; `run_match` resolves it before any generator
# sees a detector_model.
_EXEMPT = {(MatchConfig, "detector_model")}


def _literal_values(hint: object) -> set[str] | None:
    if get_origin(hint) is Union or type(hint).__name__ == "UnionType":
        hint = next(a for a in get_args(hint) if a is not type(None))
    if get_origin(hint) is Literal:
        return set(get_args(hint))
    return None


@pytest.mark.parametrize("config_cls", _CONFIGS, ids=lambda c: c.__name__)
def test_config_literal_fields_are_subsets_of_the_option_vocabulary(config_cls):
    hints = get_type_hints(config_cls)
    checked = 0
    for f in dataclasses.fields(config_cls):
        alias = _FIELD_ALIAS.get(f.name)
        if alias is None or (config_cls, f.name) in _EXEMPT:
            continue
        values = _literal_values(hints[f.name])
        assert values is not None, f"{config_cls.__name__}.{f.name} is not a Literal"
        assert values <= set(get_args(alias)), (
            f"{config_cls.__name__}.{f.name} allows {values - set(get_args(alias))}, "
            "which the implementing class does not"
        )
        checked += 1
    # MatchConfig's only enumerated field is the exempt one.
    assert checked > 0 or config_cls is MatchConfig


def test_every_alias_is_exported():
    for name in options.__all__:
        assert get_origin(getattr(options, name)) is Literal
