"""Particle margins contain no protein; micrograph margins extend the specimen."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from specter.imaging import Poses, Sample, Specimen
from specter.scattering import IterativeScattering


@pytest.mark.parametrize(
    "from_coordinates", [False, True], ids=["volume", "coordinates"]
)
def test_particle_specimens_zero_pad_the_protein_channel(from_coordinates):
    n = 8
    if from_coordinates:
        specimen = Specimen.from_atoms(
            torch.tensor([[6.0, 0.0, 0.0], [-6.0, 1.0, 0.0]]),
            torch.tensor([6, 6]),
            2.0,
            n,
        )
    else:
        specimen = Specimen(torch.ones(n, n, n), 2.0)
    poses = Poses(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(1, 2), torch.arange(1)
    )
    beam = dict(voltage=300.0, scattering_model="projection")
    # The specimen exactly as it would be propagated, before any scattering,
    # CTF or detector operation can hide a padding difference.
    unpadded = specimen.materialize(poses, **beam)
    padded = specimen.materialize(poses, pad_fft=True, **beam)
    assert unpadded[..., 1:-1, 1:-1].sum() > 0
    expected = F.pad(unpadded, (n // 2,) * 4 + (0, 0), mode="constant")
    assert not torch.equal(
        expected, F.pad(unpadded, (n // 2,) * 4 + (0, 0), mode="reflect")
    )
    torch.testing.assert_close(padded, expected)
    assert padded[..., : n // 2, :].count_nonzero() == 0
    assert padded[..., -n // 2 :, :].count_nonzero() == 0
    assert padded[..., :, : n // 2].count_nonzero() == 0
    assert padded[..., :, -n // 2 :].count_nonzero() == 0


def test_a_micrograph_sample_reflects_the_specimen_into_the_margin(monkeypatch):
    n = 8
    volume = torch.arange(n**3, dtype=torch.float32).reshape(1, n, n, n) / 10
    seen = []

    def capture(self, volume, *args, **kwargs):
        seen.append(volume.clone())
        return torch.ones(volume.shape[0], *volume.shape[-2:], dtype=torch.complex64)

    monkeypatch.setattr(IterativeScattering, "forward", capture)
    Sample(volume.clone(), 2.0).scatter(
        voltage=300.0, scattering_model="projection", pad_fft=True
    )
    assert len(seen) == 1
    expected = F.pad(volume, (n // 2,) * 4 + (0, 0), mode="reflect")
    torch.testing.assert_close(seen[0], expected)
    assert seen[0][..., : n // 2, :].count_nonzero() > 0
