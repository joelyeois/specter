"""Geometry fields can be released without changing generated outputs."""

import pytest
import torch

from specter.specimen.membrane import MembraneGenerator
from specter.specimen.tomogram import MembraneInstance, TomogramSpecimenGenerator


def make_generator(retain=None, device="cpu"):
    membrane = MembraneGenerator(
        target_shape=(24, 24, 24),
        voxel_size=5.0,
        sh_axes=(30.0, 30.0, 30.0),
        n_lipids_per_leaflet=6,
        seed=11,
        device=device,
    )
    instance = MembraneInstance(membrane)
    instance.position_xyz = (0, 0, 0)
    return TomogramSpecimenGenerator(
        membrane_instances=[instance],
        target_shape=(24, 24, 24),
        voxel_size=5.0,
        protein_specs=[],
        progressbars=False,
        seed=11,
        device=device,
        **({} if retain is None else {"retain_membrane_fields": retain}),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_released_fields_preserve_density_labels_regions_and_regeneration(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    reference = make_generator(device=device)
    compact = make_generator(False, device)
    original = reference.generate()
    actual = compact.generate()
    assert reference.membrane_instances[0].generator.field is not None
    assert compact.membrane_instances[0].generator.field is None
    assert torch.equal(actual, original)
    assert torch.equal(compact.membrane_labels, reference.membrane_labels)
    assert torch.equal(compact.instance_labels, reference.instance_labels)
    for name, mask in reference.regions.items():
        assert torch.equal(compact.regions[name], mask)
    assert torch.equal(compact.generate(), original)
    assert compact.membrane_instances[0].generator.field is None


@pytest.mark.parametrize("retain", [True, False])
def test_clipped_membrane_retention(retain, monkeypatch):
    gen = make_generator(retain)
    membrane = gen.membrane_instances[0].generator
    generate = membrane.generate

    def clipped():
        result = generate()
        membrane.clipped_at_boundary = True
        return result

    monkeypatch.setattr(membrane, "generate", clipped)
    with pytest.warns(UserWarning, match="skipped rather than compositing"):
        result = gen.generate()
    assert not bool(result.any())
    assert (membrane.field is not None) == retain
    assert (membrane.volume is not None) == retain
    assert not gen.placed_membrane_instances
