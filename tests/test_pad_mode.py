"""Particle margins contain no protein; micrograph margins extend the specimen."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from specter.imagegenerator import (
    ImageGenerator,
    ImageGeneratorFromCoordinates,
    MicrographGenerator,
)
from specter.settings import Camera, Propagation


@pytest.mark.parametrize(
    "from_coordinates", [False, True], ids=["volume", "coordinates"]
)
def test_particle_generators_zero_pad_the_protein_channel(
    monkeypatch, from_coordinates
):
    n = 8
    common = dict(
        pixel_size=2.0,
        quaternions=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        translations=torch.zeros(1, 2),
        ctf_params={"dfu": torch.tensor([5000.0])},
        voltage=300.0,
        dose_per_angstrom=2.0,
        propagation=Propagation(scattering_model="projection", pad_fft=True),
        camera=Camera(noise_model=None),
        verbose=False,
    )
    idx = torch.tensor([0])
    if from_coordinates:
        gen = ImageGeneratorFromCoordinates(
            coordinates=torch.tensor([[6.0, 0.0, 0.0], [-6.0, 1.0, 0.0]]),
            atomic_numbers=torch.tensor([6, 6]),
            nxy=n,
            **common,
        )
        unpadded = gen.potentialbuilder(gen.rotate(gen.quaternions, gen.translations))
        if unpadded.ndim == 3:
            unpadded = unpadded.unsqueeze(0)
    else:
        gen = ImageGenerator(
            scattering_potential=torch.ones(n, n, n), progressbars=False, **common
        )
        unpadded = gen.rotate(gen.quaternions, gen.translations)
    assert unpadded[..., 1:-1, 1:-1].sum() > 0
    # Observe the specimen passed to propagation, before any scattering,
    # CTF or detector operation can hide a padding difference.
    monkeypatch.setattr(gen, "process_volume", lambda volume, idx: volume)
    padded = gen(idx)
    expected = F.pad(unpadded, (n // 2,) * 4 + (0, 0), mode="constant")
    assert not torch.equal(
        expected, F.pad(unpadded, (n // 2,) * 4 + (0, 0), mode="reflect")
    )
    torch.testing.assert_close(padded, expected)
    assert padded[..., : n // 2, :].count_nonzero() == 0
    assert padded[..., -n // 2 :, :].count_nonzero() == 0
    assert padded[..., :, : n // 2].count_nonzero() == 0
    assert padded[..., :, -n // 2 :].count_nonzero() == 0


def test_micrograph_generator_reflects_the_specimen_into_the_margin(monkeypatch):
    n = 8
    volume = torch.arange(n**3, dtype=torch.float32).reshape(1, n, n, n) / 10
    gen = MicrographGenerator(
        volume.clone(),
        n,
        2.0,
        {"dfu": torch.tensor([5000.0])},
        300.0,
        2.0,
        propagation=Propagation(scattering_model="projection", pad_fft=True),
        camera=Camera(noise_model=None),
        progressbars=False,
        verbose=False,
    )
    seen = []

    def capture(volume, **kwargs):
        seen.append(volume.clone())
        return torch.ones(volume.shape[0], *volume.shape[-2:], dtype=torch.complex64)

    monkeypatch.setattr(gen.iterative_scattering, "forward", capture)
    gen(torch.tensor([0]))
    assert len(seen) == 1
    expected = F.pad(volume, (n // 2,) * 4 + (0, 0), mode="reflect")
    torch.testing.assert_close(seen[0], expected)
    assert seen[0][..., : n // 2, :].count_nonzero() > 0
