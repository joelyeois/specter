"""Tests for run_build_tomogram's membrane-mode config -> MembraneInstance
wiring (specter.pipelines.build_tomogram_generator) --
specifically the n_copies/target_shape defaulting logic,
not the (expensive, already covered by tests/test_tomogram_generator.py)
actual generation. No PDB files need to exist for these: protein_specs'
pdb_source strings are only resolved inside TomogramSpecimenGenerator.
generate(), which none of these tests call."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from specter.config import TomogramConfig
from specter.pipelines import build_tomogram_generator
from specter.specimen.filament import ACTIN_SPEC

_SMALL_FIXTURE = Path(__file__).parent / "test_data" / "1mbo.cif"

_BASE_KWARGS = dict(
    target_shape=[64, 64, 64],
    voxel_size=8.0,
    filler=[{"pdb_source": "1mbo"}],
)


def test_membrane_entry_defaults_to_auto_size_and_auto_position():
    config = TomogramConfig(
        membrane=[{"kind": "tubules"}],
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
                "kind": "vesicle",
                "radius": 80.0,
                "target_shape": [20, 20, 20],
            }
        ],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    instance = gen.membrane_instances[0]
    assert instance.generator.target_shape == (20, 20, 20)


def test_membrane_table_fields_reach_the_organelle_plan():
    """A [[membrane]] table's kind and fields are the generator's own."""
    config = TomogramConfig(
        membrane=[
            {
                "kind": "vesicle",
                "radius": 90.0,
                "elongation": 1.0,
                "undulation": 0.0,
            }
        ],
        **_BASE_KWARGS,
    )
    generator = build_tomogram_generator(config).membrane_instances[0].generator
    assert generator.kind == "vesicle"
    assert generator.params["radius"] == 90.0
    assert generator.params["elongation"] == 1.0
    assert generator.params["undulation"] == 0.0


def test_membrane_table_refuses_a_field_of_another_kind():
    config = TomogramConfig(
        membrane=[{"kind": "vesicle", "crista_spacing": 250.0}],
        **_BASE_KWARGS,
    )
    with pytest.raises(ValueError, match="crista_spacing"):
        build_tomogram_generator(config)


def test_membrane_table_refuses_an_unknown_kind():
    config = TomogramConfig(membrane=[{"kind": "metaball"}], **_BASE_KWARGS)
    with pytest.raises(ValueError, match="metaball"):
        build_tomogram_generator(config)


def test_n_copies_makes_that_many_instances():
    config = TomogramConfig(
        membrane=[{"kind": "vesicle", "n_copies": 3}, {"kind": "vesicle"}],
        seed=2,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.membrane_instances) == 4
    # Copies of one entry are different draws, not one organelle repeated.
    radii = [mi.generator.params["radius"] for mi in gen.membrane_instances[:3]]
    assert len(set(radii)) == 3


def test_auto_sized_membranes_are_shrunk_to_fit_the_tomogram():
    """
    Without a `target_shape`, the entry's default-sized fields are capped so
    the organelle fits the box: 0.75 of the limiting extent, as a diameter.
    """
    from specter.pipelines._tomogram import _membrane_max_radius

    config = TomogramConfig(membrane=[{"kind": "vesicle"}], seed=0, **_BASE_KWARGS)
    limit = _membrane_max_radius(config)
    assert limit == pytest.approx(0.5 * 0.75 * 64 * 8.0)

    generator = build_tomogram_generator(config).membrane_instances[0].generator
    assert generator.bounding_radius <= limit + 1e-6
    # The default vesicle radius range starts at 200 A, above this limit.
    assert generator.params["radius"] < 200.0


def test_max_radius_follows_the_axes_that_cannot_clip():
    from specter.pipelines._tomogram import _membrane_max_radius

    thin = TomogramConfig(
        target_shape=[20, 100, 100],
        voxel_size=10.0,
        clip_axes=[True, False, False],
        filler=[{"pdb_source": "1mbo"}],
    )
    assert _membrane_max_radius(thin) == pytest.approx(0.5 * 0.75 * 1000.0)
    all_fixed = dataclasses.replace(thin, clip_axes=[False, False, False])
    assert _membrane_max_radius(all_fixed) == pytest.approx(0.5 * 0.75 * 200.0)


def test_an_explicit_size_is_never_shrunk_to_fit() -> None:
    config = TomogramConfig(
        membrane=[{"kind": "vesicle", "radius": 400.0}], seed=0, **_BASE_KWARGS
    )
    generator = build_tomogram_generator(config).membrane_instances[0].generator
    assert generator.params["radius"] == 400.0


def test_an_explicit_target_shape_turns_the_size_cap_off() -> None:
    """`max_radius` is passed only for an auto-sized working grid."""
    config = TomogramConfig(
        membrane=[{"kind": "vesicle", "target_shape": [90, 90, 90]}],
        seed=0,
        **_BASE_KWARGS,
    )
    generator = build_tomogram_generator(config).membrane_instances[0].generator
    assert generator.target_shape == (90, 90, 90)
    # Drawn from the uncapped default range.
    assert generator.params["radius"] >= 200.0


def test_n_copies_restarts_per_entry_not_running_across_entries():
    config = TomogramConfig(
        membrane=[
            {"kind": "vesicle", "n_copies": 2},
            {"kind": "tubules", "n_copies": 2},
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
    entry = {"kind": "vesicle", "n_copies": 2}
    config = TomogramConfig(membrane=[entry], seed=0, **_BASE_KWARGS)
    build_tomogram_generator(config)
    assert entry == {"kind": "vesicle", "n_copies": 2}


@pytest.mark.parametrize(
    "membrane_entry, config_kwargs, expected_workers, expected_devices, expected_device",
    [
        (
            {"kind": "vesicle", "n_copies": 2},
            {"render_workers": 4, "device": "0,1"},
            4,
            [torch.device("cuda:0"), torch.device("cuda:1")],
            "cuda:0",
        ),
        (
            {"kind": "vesicle"},
            {"device": "cpu"},
            1,
            [torch.device("cpu")],
            "cpu",
        ),
    ],
    ids=["explicit_pool", "default_serial"],
)
def test_render_workers_and_devices_reach_membrane_tomogram_generator(
    membrane_entry: dict,
    config_kwargs: dict,
    expected_workers: int,
    expected_devices: list[torch.device],
    expected_device: str,
) -> None:
    """A comma-separated `device` pools multiple GPUs for concurrent
    per-species rendering (see specter.devices.DeviceSpec.primary_and_pool)
    -- torch.device("cuda:N") is a valid descriptor regardless of whether
    GPU N is actually present, so this doesn't need real multi-GPU hardware
    to test the wiring. Left unset, render_workers is serial; that case pins
    device="cpu" so it is about render_workers, not about which device the
    config defaults to (which is "cuda", resolved against the host at run
    time)."""
    config = TomogramConfig(membrane=[membrane_entry], **config_kwargs, **_BASE_KWARGS)
    gen = build_tomogram_generator(config)
    assert gen.render_workers == expected_workers
    assert gen.render_devices == expected_devices
    assert gen.device == expected_device
    # Every MembraneGenerator instance still gets its own default (1) --
    # transmembrane_specs is empty in _BASE_KWARGS, so there's nothing for
    # a per-instance render pass to parallelize anyway (see
    # build_tomogram_generator's own pre-render comment).
    assert all(mi.generator.render_workers == 1 for mi in gen.membrane_instances)


def test_filaments_config_builds_filament_specs():
    config = TomogramConfig(
        membrane=[{"kind": "vesicle"}],
        filaments=[{"code": "1TUB", "step": 82.0, "flex_deg": 3.0, "n_copies": 4}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.filament_specs) == 1
    spec = gen.filament_specs[0]
    assert spec.code == "1TUB"
    assert spec.step == 82.0
    assert spec.flex_deg == 3.0
    assert spec.n_copies == 4


def test_actin_flag_is_additive_to_filaments():
    config = TomogramConfig(
        membrane=[{"kind": "vesicle"}],
        filaments=[{"code": "1TUB", "step": 82.0, "flex_deg": 3.0}],
        actin=True,
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert len(gen.filament_specs) == 2
    assert gen.filament_specs[0].code == "1TUB"
    assert gen.filament_specs[1] == ACTIN_SPEC


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
        membrane=[{"kind": "vesicle"}],
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
        membrane=[{"kind": "vesicle"}],
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
        membrane=[{"kind": "tubules"}],
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
        membrane=[{"kind": "tubules", "parameterization": "lobato"}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)

    assert gen.parameterization == "kirkland"
    assert gen.membrane_instances[0].generator.parameterization == "lobato"


def test_bulk_scattering_factors_reach_carbon_and_gold():
    """Bulk materials are parameterized separately from the biomolecules.

    Shtyrov is fitted for biomolecules, so the carbon film and the gold
    fiducials take `bulk_scattering_factors` rather than the specimen's own
    `scattering_factors` -- and changing one must not move the other.
    """
    config = TomogramConfig(
        scattering_factors="shtyrov",
        bulk_scattering_factors="lobato",
        carbon_film=[{"hole_radius": 4000.0}],
        beads=[{"radius": 80.0, "n_copies": 1}],
        **_BASE_KWARGS,
    )
    gen = build_tomogram_generator(config)
    assert gen.parameterization == "shtyrov"
    assert gen.bulk_parameterization == "lobato"
