"""Tests for run_build_tomogram's membrane-mode config -> MembraneInstance
wiring (specter.pipelines.build_tomogram_generator) --
specifically the n_copies/target_shape defaulting logic,
not the (expensive, already covered by tests/test_tomogram_generator.py)
actual generation. No PDB files need to exist for these: protein_specs'
pdb_source strings are only resolved inside TomogramSpecimenGenerator.
generate(), which none of these tests call."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.config import TomogramConfig
from specter.pipelines import build_tomogram_generator

_SMALL_FIXTURE = Path(__file__).parent / "test_data" / "1mbo.cif"

_BASE_KWARGS = dict(
    target_shape=[64, 64, 64],
    voxel_size=8.0,
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
    # target_shape=None was passed through to MembraneGenerator, which
    # then auto-sized its OWN (concrete, non-None) working grid -- distinct
    # from (and, for a single small organelle, much smaller than) the shared
    # tomogram canvas in _BASE_KWARGS.
    assert instance.generator.target_shape is not None
    assert instance.generator.target_shape != tuple(_BASE_KWARGS["target_shape"])


def test_membrane_entry_explicit_target_shape_zyx_is_honored():
    config = TomogramConfig(
        membrane=[
            {
                "shape_backend": "spherical_harmonics",
                "sh_axes": [50.0, 50.0, 50.0],
                "target_shape": [20, 20, 20],
            }
        ],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    instance = gen.membrane_instances[0]
    assert instance.generator.target_shape == (20, 20, 20)


def test_n_copies_expands_into_independent_seeded_instances():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics", "n_copies": 3}],
        seed=100,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.membrane_instances) == 3
    seeds = [mi.generator.seed for mi in gen.membrane_instances]
    assert seeds == [100, 101, 102]
    assert all(mi.position_xyz is None for mi in gen.membrane_instances)


def test_n_copies_restarts_per_entry_not_running_across_entries():
    config = TomogramConfig(
        membrane=[
            {"shape_backend": "spherical_harmonics", "n_copies": 2},
            {"shape_backend": "swept_spline", "n_copies": 2},
        ],
        seed=5,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    seeds = [mi.generator.seed for mi in gen.membrane_instances]
    # Second entry's seeds restart at config.seed, not continue from the
    # first entry's last seed (which would give [5, 6, 7, 8]).
    assert seeds == [5, 6, 5, 6]


def test_membrane_config_entry_dict_never_mutated():
    entry = {"shape_backend": "spherical_harmonics", "n_copies": 2}
    config = TomogramConfig(membrane=[entry], seed=0, **_BASE_KWARGS)
    build_tomogram_generator(config)
    assert entry == {"shape_backend": "spherical_harmonics", "n_copies": 2}


def test_render_workers_and_devices_reach_membrane_tomogram_generator():
    # A comma-separated `device` pools multiple GPUs for concurrent
    # per-species rendering (see specter.specimen._parallel_render.
    # parse_device_pool) -- torch.device("cuda:N") is a valid descriptor
    # regardless of whether GPU N is actually present, so this doesn't
    # need real multi-GPU hardware to test the wiring.
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics", "n_copies": 2}],
        render_workers=4,
        device="0,1",
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert gen.render_workers == 4
    assert gen.render_devices == [torch.device("cuda:0"), torch.device("cuda:1")]
    assert gen.device == "cuda:0"
    # Every MembraneGenerator instance still gets its own default (1) --
    # transmembrane_specs is empty in _BASE_KWARGS, so there's nothing for
    # a per-instance render pass to parallelize anyway (see
    # build_tomogram_generator's own pre-render comment).
    assert all(mi.generator.render_workers == 1 for mi in gen.membrane_instances)


def test_render_workers_default_is_serial():
    # device pinned so this is about render_workers, not about which device the
    # config defaults to (which is "cuda", resolved against the host at run time).
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        device="cpu",
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert gen.render_workers == 1
    assert gen.render_devices == [torch.device("cpu")]


def test_filaments_config_builds_filament_specs():
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        filaments=[{"code": "1TUB", "step": 85.0, "flex_deg": 3.0, "n_copies": 4}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.filament_specs) == 1
    spec = gen.filament_specs[0]
    assert spec.code == "1TUB"
    assert spec.step == 85.0
    assert spec.flex_deg == 3.0
    assert spec.n_copies == 4


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


# `n_copies` is the one count spelling across every config entry type. The
# two specs below still store it under their own pre-existing field names
# (TransmembraneSpec.frequency, which doubles as the species-draw weight;
# TomogramBeadSpec.count), so these check the translation at the config
# boundary specifically -- a typo'd .get() key would silently fall back to
# the default of 1 rather than raising.
@pytest.mark.parametrize(
    ("spec_dict", "expected"),
    [({"n_copies": 7}, 7), ({}, 1)],
    ids=["explicit", "omitted-defaults-to-1"],
)
def test_transmembrane_n_copies_maps_onto_spec_frequency(spec_dict, expected):
    # A local fixture path, not a PDB code: build_tomogram_generator renders
    # transmembrane templates eagerly, so a code would download here.
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        membrane_transmembrane_specs=[{"pdb_source": str(_SMALL_FIXTURE), **spec_dict}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    specs = gen.membrane_instances[0].generator.transmembrane_specs
    assert [spec.frequency for spec in specs] == [expected]


@pytest.mark.parametrize(
    ("spec_dict", "expected"),
    [({"n_copies": 20}, 20), ({}, 1)],
    ids=["explicit", "omitted-defaults-to-1"],
)
def test_bead_n_copies_maps_onto_spec_count(spec_dict, expected):
    config = TomogramConfig(
        membrane=[{"shape_backend": "spherical_harmonics"}],
        beads=[{"radius": 100.0, **spec_dict}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert [(spec.radius, spec.count) for spec in gen.bead_specs] == [(100.0, expected)]


def test_microtubules_config_reaches_the_generator():
    config = TomogramConfig(
        microtubules=[{"n_copies": 3, "n_protofilaments": 14}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.microtubule_specs) == 1
    spec = gen.microtubule_specs[0]
    assert spec.n_copies == 3
    assert spec.n_protofilaments == 14
    # A microtubule is not a filament species -- the two lists stay separate.
    assert gen.filament_specs == []


def test_microtubules_alone_satisfy_the_species_source_check():
    """`[[microtubules]]` is a species source in its own right: a tomogram
    of nothing but microtubules must build, not raise."""
    from specter.pipelines._tomogram import run_build_tomogram

    config = TomogramConfig(
        target_shape=[64, 64, 64],
        voxel_size=8.0,
        microtubules=[{"n_copies": 1}],
    )
    # Reaching the generator at all is the assertion; this would have
    # raised ValueError("...can't all be empty/False...") before.
    gen = build_tomogram_generator(config)
    assert len(gen.microtubule_specs) == 1
    assert callable(run_build_tomogram)


def test_scattering_factors_reach_every_renderer_in_the_tomogram():
    """One tomogram, one set of scattering factors.

    Everything placed is summed into the same volume, so the bilayer and
    the transmembrane proteins have to follow the same parameterization as
    the packed proteins rather than MembraneGenerator's own default.
    """
    config = TomogramConfig(
        scattering_factors="lobato",
        membrane=[{"shape_backend": "swept_spline"}],
        membrane_transmembrane_specs=[{"pdb_source": "1mbo"}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    membrane = gen.membrane_instances[0].generator

    assert gen.parameterization == "lobato"
    assert membrane.parameterization == "lobato"
    assert [s.parameterization for s in membrane.transmembrane_specs] == ["lobato"]


def test_membrane_entry_may_override_the_tomogram_scattering_factors():
    """The global setting is a default, not a lock: a [[membrane]] table
    naming its own parameterization still wins for that population."""
    config = TomogramConfig(
        scattering_factors="kirkland",
        membrane=[{"shape_backend": "swept_spline", "parameterization": "lobato"}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)

    assert gen.parameterization == "kirkland"
    assert gen.membrane_instances[0].generator.parameterization == "lobato"
