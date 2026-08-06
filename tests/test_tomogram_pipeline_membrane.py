"""Tests for run_build_tomogram's membrane-mode config -> MembraneInstance
wiring (specter.pipelines._tomogram._build_membrane_tomogram_generator) --
specifically the n_instances/position_xyz/target_shape_zyx defaulting logic,
not the (expensive, already covered by tests/test_tomogram_generator.py)
actual generation. No PDB files need to exist for these: protein_specs'
pdb_source strings are only resolved inside MembraneTomogramGenerator.
generate(), which none of these tests call."""

from __future__ import annotations

import pytest

from specter.config import TomogramConfig
from specter.pipelines._tomogram import _build_membrane_tomogram_generator

_BASE_KWARGS = dict(
    target_shape=[64, 64, 64],
    v_size=8.0,
    membrane_protein_specs=[{"pdb_source": "1mbo"}],
)


def test_membrane_entry_defaults_to_auto_size_and_auto_position():
    config = TomogramConfig(
        membrane=[{"shape_backend": "swept_spline"}],
        **_BASE_KWARGS,
    )
    gen = _build_membrane_tomogram_generator(config)
    assert len(gen.membrane_instances) == 1
    instance = gen.membrane_instances[0]
    assert instance.position_xyz is None
    # target_shape_zyx=None was passed through to MembraneGenerator, which
    # then auto-sized its OWN (concrete, non-None) working grid -- distinct
    # from (and, for a single small organelle, much smaller than) the shared
    # tomogram canvas in _BASE_KWARGS.
    assert instance.generator.target_shape_zyx is not None
    assert instance.generator.target_shape_zyx != tuple(_BASE_KWARGS["target_shape"])


def test_membrane_entry_explicit_position_and_target_shape_zyx_are_honored():
    config = TomogramConfig(
        membrane=[
            {
                "shape_backend": "spherical_harmonics",
                "sh_axes_a": [50.0, 50.0, 50.0],
                "position_xyz": [10.0, -5.0, 0.0],
                "target_shape_zyx": [20, 20, 20],
            }
        ],
        **_BASE_KWARGS,
    )
    gen = _build_membrane_tomogram_generator(config)
    instance = gen.membrane_instances[0]
    assert instance.position_xyz == (10.0, -5.0, 0.0)
    assert instance.generator.target_shape_zyx == (20, 20, 20)


def test_n_instances_expands_into_independent_seeded_instances():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics", "n_instances": 3}],
        seed=100,
        **_BASE_KWARGS,
    )
    gen = _build_membrane_tomogram_generator(config)
    assert len(gen.membrane_instances) == 3
    seeds = [mi.generator.seed for mi in gen.membrane_instances]
    assert seeds == [100, 101, 102]
    assert all(mi.position_xyz is None for mi in gen.membrane_instances)


def test_n_instances_restarts_per_entry_not_running_across_entries():
    config = TomogramConfig(
        membrane=[
            {"shape_backend": "spherical_harmonics", "n_instances": 2},
            {"shape_backend": "swept_spline", "n_instances": 2},
        ],
        seed=5,
        **_BASE_KWARGS,
    )
    gen = _build_membrane_tomogram_generator(config)
    seeds = [mi.generator.seed for mi in gen.membrane_instances]
    # Second entry's seeds restart at config.seed, not continue from the
    # first entry's last seed (which would give [5, 6, 7, 8]).
    assert seeds == [5, 6, 5, 6]


def test_n_instances_greater_than_one_with_explicit_position_xyz_raises():
    config = TomogramConfig(
        membrane=[
            {
                "shape_backend": "spherical_harmonics",
                "n_instances": 2,
                "position_xyz": [0.0, 0.0, 0.0],
            }
        ],
        **_BASE_KWARGS,
    )
    with pytest.raises(ValueError, match="n_instances"):
        _build_membrane_tomogram_generator(config)


def test_membrane_config_entry_dict_never_mutated():
    entry = {"shape_backend": "spherical_harmonics", "n_instances": 2}
    config = TomogramConfig(membrane=[entry], seed=0, **_BASE_KWARGS)
    _build_membrane_tomogram_generator(config)
    assert entry == {"shape_backend": "spherical_harmonics", "n_instances": 2}
