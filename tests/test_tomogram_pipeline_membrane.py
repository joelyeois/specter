"""Tests for run_build_tomogram's membrane-mode config -> MembraneInstance
wiring (specter.pipelines.build_tomogram_generator) --
specifically the n_instances/position_xyz/target_shape_zyx defaulting logic,
not the (expensive, already covered by tests/test_tomogram_generator.py)
actual generation. No PDB files need to exist for these: protein_specs'
pdb_source strings are only resolved inside MembraneTomogramGenerator.
generate(), which none of these tests call."""

from __future__ import annotations

import pytest
import torch

from specter.config import TomogramConfig
from specter.pipelines import build_tomogram_generator

_BASE_KWARGS = dict(
    target_shape=[64, 64, 64],
    v_size=8.0,
    filler=[{"pdb_source": "1mbo"}],
)


def test_membrane_entry_defaults_to_auto_size_and_auto_position():
    config = TomogramConfig(
        membrane=[{"shape_backend": "swept_spline"}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
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
    gen = build_tomogram_generator(config)
    instance = gen.membrane_instances[0]
    assert instance.position_xyz == (10.0, -5.0, 0.0)
    assert instance.generator.target_shape_zyx == (20, 20, 20)


def test_n_instances_expands_into_independent_seeded_instances():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics", "n_instances": 3}],
        seed=100,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
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
    gen = build_tomogram_generator(config)
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
        build_tomogram_generator(config)


def test_membrane_config_entry_dict_never_mutated():
    entry = {"shape_backend": "spherical_harmonics", "n_instances": 2}
    config = TomogramConfig(membrane=[entry], seed=0, **_BASE_KWARGS)
    build_tomogram_generator(config)
    assert entry == {"shape_backend": "spherical_harmonics", "n_instances": 2}


def test_render_workers_and_devices_reach_membrane_tomogram_generator():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics", "n_instances": 2}],
        render_workers=4,
        render_devices=["cpu"],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert gen.render_workers == 4
    assert gen.render_devices == [torch.device("cpu")]
    # Every MembraneGenerator instance still gets its own default (1) --
    # transmembrane_specs is empty in _BASE_KWARGS, so there's nothing for
    # a per-instance render pass to parallelize anyway (see
    # build_tomogram_generator's own pre-render comment).
    assert all(mi.generator.render_workers == 1 for mi in gen.membrane_instances)


def test_render_workers_default_is_serial():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert gen.render_workers == 1
    assert gen.render_devices == [torch.device("cpu")]


def test_filaments_config_builds_filament_specs():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        filaments=[{"code": "1TUB", "step": 85.0, "flex_deg": 3.0, "n_filaments": 4}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.filament_specs) == 1
    spec = gen.filament_specs[0]
    assert spec.code == "1TUB"
    assert spec.step == 85.0
    assert spec.flex_deg == 3.0
    assert spec.n_filaments == 4


def test_actin_flag_appends_actin_spec():
    from specter.specimen import ACTIN_SPEC

    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        actin=True,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert ACTIN_SPEC in gen.filament_specs


def test_actin_flag_is_additive_to_filaments():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        filaments=[{"code": "1TUB", "step": 85.0, "flex_deg": 3.0}],
        actin=True,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.filament_specs) == 2
    codes = {spec.code for spec in gen.filament_specs}
    assert codes == {"1TUB", "1J6Z"}


def test_no_filaments_or_actin_leaves_filament_specs_empty():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert gen.filament_specs == []
