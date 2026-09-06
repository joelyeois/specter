"""The settings groups: validation, config binding, and how the micrograph
imager and its specimen generator divide them."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from specter.imagegenerator import MicrographGenerator
from specter.settings import Crowding, Ice, Packing, bundle_from_config
from specter.specimen import MicrographSpecimenGenerator


def _ctf_params():
    return {"cs": torch.tensor([2.7e7]), "dfu": torch.tensor([10000.0])}


def test_crowding_rejects_nonpositive_min_distance_and_chunk_size():
    with pytest.raises(ValueError, match="min_distance"):
        Crowding(min_distance=0.0)
    with pytest.raises(ValueError, match="chunk_size"):
        Crowding(chunk_size=0)
    assert Crowding().min_distance is None


def test_bundle_from_config_prefix_and_overrides():
    @dataclass
    class Config:
        crowd_min_distance: float = 80.0
        crowd_max_distance_z: float = 300.0
        water_air_interface: bool = True  # unprefixed on the config
        packing_backend: str = "shape"
        packing_max_retries: int = 7
        n_orientations: int = 12  # unprefixed on the config

    crowding = bundle_from_config(
        Crowding, Config(), prefix="crowd_", water_air_interface=True
    )
    assert crowding.min_distance == 80.0
    assert crowding.max_distance_z == 300.0
    assert crowding.water_air_interface is True
    # A field the config does not spell keeps the group's own default.
    assert crowding.method == "3d"

    packing = bundle_from_config(
        Packing, Config(), prefix="packing_", n_orientations=12
    )
    assert packing.backend == "shape"
    assert packing.max_retries == 7
    assert packing.n_orientations == 12
    assert packing.gap == 0.0


def test_specimen_generator_derives_nz_from_the_ice(small_volume):
    # No ice: the template's own depth.
    gen = MicrographSpecimenGenerator(small_volume, 2.0, 32, progressbars=False)
    assert gen.nz == small_volume.shape[0]
    assert gen.crowd is None
    # Ice deeper than the template grows the box.
    gen = MicrographSpecimenGenerator(
        small_volume, 2.0, 32, ice=Ice(thickness=200.0), progressbars=False
    )
    assert gen.nz == 100
    assert gen.ice_thickness == 200.0
    # An explicit nz wins, and a templateless specimen needs one.
    assert MicrographSpecimenGenerator(None, 2.0, 32, nz=48).nz == 48
    with pytest.raises(ValueError, match="nz"):
        MicrographSpecimenGenerator(None, 2.0, 32)


def test_specimen_generator_builds_its_crowd_from_the_bundles(small_volume):
    gen = MicrographSpecimenGenerator(
        small_volume,
        2.0,
        32,
        crowding=Crowding(min_distance=40.0, max_distance_z=50.0),
        packing=Packing(backend="poisson_disk", max_retries=3),
        progressbars=False,
    )
    assert gen.crowd is not None
    assert gen.crowd.max_distance_z == 50.0
    assert gen.crowd.packing_max_retries == 3
    assert gen.crowd.water_air_interface is False


def test_micrograph_generator_takes_the_specimen_generators_ice(small_volume):
    specimen = MicrographSpecimenGenerator(
        small_volume, 2.0, 32, ice=Ice(model="random"), progressbars=False
    )
    gen = MicrographGenerator(
        specimen, 32, 2.0, _ctf_params(), 300.0, 2.0, verbose=False, progressbars=False
    )
    assert gen.specimen_gen is specimen
    assert gen.ice_model == "random"
    assert gen.nz == specimen.nz


def test_micrograph_generator_rejects_ice_alongside_a_specimen_generator(
    small_volume,
):
    specimen = MicrographSpecimenGenerator(small_volume, 2.0, 32, progressbars=False)
    with pytest.raises(ValueError, match="own ice"):
        MicrographGenerator(
            specimen,
            32,
            2.0,
            _ctf_params(),
            300.0,
            2.0,
            ice=Ice(model="random"),
            verbose=False,
            progressbars=False,
        )


def test_micrograph_generator_rejects_a_mismatched_specimen(small_volume):
    specimen = MicrographSpecimenGenerator(small_volume, 2.0, 32, progressbars=False)
    with pytest.raises(ValueError, match="micrograph is 64 px"):
        MicrographGenerator(specimen, 64, 2.0, _ctf_params(), 300.0, 2.0, verbose=False)


def test_micrograph_generator_rejects_a_bare_template(small_volume):
    with pytest.raises(ValueError, match="MicrographSpecimenGenerator"):
        MicrographGenerator(
            small_volume, 32, 2.0, _ctf_params(), 300.0, 2.0, verbose=False
        )
    with pytest.raises(TypeError, match="specimen"):
        MicrographGenerator(
            "volume.mrc", 32, 2.0, _ctf_params(), 300.0, 2.0, verbose=False
        )
    with pytest.raises(RuntimeError, match="regenerate_specimen"):
        MicrographGenerator(
            small_volume.unsqueeze(0),
            32,
            2.0,
            _ctf_params(),
            300.0,
            2.0,
            verbose=False,
            progressbars=False,
        ).regenerate_specimen()
