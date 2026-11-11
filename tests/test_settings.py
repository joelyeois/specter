"""The settings groups: validation, config binding, and how a micrograph
field and its specimen generator divide them."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from specter.imaging import Field, Specimen
from specter.settings import Crowding, Ice, Packing, Propagation, bundle_from_config
from specter.specimen import MicrographSpecimenGenerator


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


# Which settings-group fields each pipeline config supplies to
# `bundle_from_config`. A field missing here silently takes the group's own
# default, so a rename on either side changes a run without failing it; this
# table makes it fail. Update it deliberately when a field is added.
_BUNDLED_FIELDS = {
    ("ParticleStackConfig", "Camera", ""): ['detector_model', 'detector_pixel_size', 'dose_weights_max_frequency', 'dose_weights_path', 'n_frames', 'noise_model'],
    ("ParticleStackConfig", "Envelopes", ""): ['cc', 'convergence_angle', 'deltaI_I', 'deltaV_V', 'dose_envelope', 'dose_envelope_target', 'energy_spread', 'temporal_coherence'],
    ("ParticleStackConfig", "Crowding", "crowd_"): ['chunk_size', 'max_distance_xy', 'max_distance_z', 'method', 'min_distance', 'n_points', 'seed'],
    ("ParticleStackConfig", "Ice", "ice_"): ['cache_dir', 'model', 'motion_variance', 'relax_steps', 'thickness'],
    ("ParticleStackConfig", "Propagation", ""): ['absorption_model', 'alpha', 'ews_curvature_sign', 'inelastic_mfp_solvent', 'inelastic_mfp_specimen', 'klim', 'oversample', 'pad_fft', 'rotate_mode', 'scattering_model'],
    ("MicrographConfig", "Camera", ""): ['detector_model', 'n_frames', 'noise_model'],
    ("MicrographConfig", "Envelopes", ""): ['cc', 'convergence_angle', 'deltaI_I', 'deltaV_V', 'dose_envelope', 'dose_envelope_target', 'energy_spread', 'temporal_coherence'],
    ("MicrographConfig", "Crowding", "crowd_"): ['chunk_size', 'max_distance_z', 'min_distance'],
    ("MicrographConfig", "Ice", "ice_"): ['cache_dir', 'model', 'motion_variance', 'profile', 'thickness'],
    ("MicrographConfig", "Packing", "packing_"): ['backend', 'gap', 'max_retries', 'seed', 'stall_patience'],
    ("MicrographConfig", "Propagation", ""): ['absorption_model', 'alpha', 'inelastic_mfp_solvent', 'inelastic_mfp_specimen', 'klim', 'oversample', 'pad_fft', 'scattering_model'],
    ("TiltSeriesConfig", "Camera", ""): ['detector_model', 'n_frames', 'noise_model'],
    ("TiltSeriesConfig", "Envelopes", ""): ['cc', 'convergence_angle', 'deltaI_I', 'deltaV_V', 'dose_envelope', 'dose_envelope_target', 'energy_spread', 'temporal_coherence'],
    ("TiltSeriesConfig", "Deformation", "deformation_"): ['burst_dose', 'correlation_length', 'height', 'hole_offset', 'hole_radius', 'model'],
    ("TiltSeriesConfig", "Ice", "ice_"): ['cache_dir', 'model', 'motion_variance', 'relax_steps'],
    ("TiltSeriesConfig", "TiltGeometry", ""): ['tilt_axis'],
    ("TiltSeriesConfig", "Propagation", ""): ['absorption_model', 'alpha', 'inelastic_mfp_solvent', 'inelastic_mfp_specimen', 'klim', 'oversample', 'pad_fft', 'scattering_model'],
    ("ReconstructionConfig", "Propagation", ""): ['alpha', 'ews_curvature_sign', 'klim', 'rotate_mode', 'scattering_model'],
}  # fmt: skip


@pytest.mark.parametrize(("config", "bundle", "prefix"), sorted(_BUNDLED_FIELDS))
def test_bundle_from_config_reads_the_pinned_fields(config, bundle, prefix):
    import dataclasses

    from specter import config as configs
    from specter import settings

    config_fields = {f.name for f in dataclasses.fields(getattr(configs, config))}
    read = sorted(
        f.name
        for f in dataclasses.fields(getattr(settings, bundle))
        if prefix + f.name in config_fields
    )
    assert read == _BUNDLED_FIELDS[(config, bundle, prefix)]


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


def test_a_field_carries_its_ice_into_every_draw(small_volume):
    """The ice library is resolved once, by `solvate`, and every draw blends
    from it into a box the specimen generator sizes."""
    field = Field(small_volume, 2.0, 32).solvate(model="random")
    assert field.icemaker is not None and field.icemaker.method == "random"
    sample = field.draw(voltage=300.0)
    engine = MicrographSpecimenGenerator(small_volume, 2.0, 32, progressbars=False)
    assert sample.nz == engine.nz
    with pytest.raises(ValueError, match="ice model"):
        Field(small_volume, 2.0, 32).solvate()


def test_a_verb_takes_its_settings_group_or_its_fields_never_both() -> None:
    """Every verb reads its group from an object or from keywords alike."""
    specimen = Specimen(torch.zeros(8, 8, 8), 1.0)
    assert (
        specimen.crowd(Crowding(min_distance=4.0)).crowding
        == specimen.crowd(min_distance=4.0).crowding
    )
    assert specimen.plan(propagation=Propagation(oversample=1)) == specimen.plan(
        oversample=1
    )
    with pytest.raises(TypeError, match="not both"):
        specimen.crowd(Crowding(min_distance=4.0), max_distance_z=2.0)
    with pytest.raises(TypeError, match="not both"):
        specimen.plan(propagation=Propagation(), oversample=1)
