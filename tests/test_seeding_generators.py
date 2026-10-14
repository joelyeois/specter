"""
Per-draw seeding on micrographs and tilt series.

The particle chain's version is in tests/test_particle_base.py; these pin the
same contract for the other two: a seed makes the output a function of
(seed, draw index) and nothing else.
"""

from __future__ import annotations

import torch

import specter
from specter.imaging import CTF, Exposure, Field, Poses, Sample
from specter.settings import Ice, Propagation

PROPAGATION = dict(oversample=1, alpha=0.07)


def _template(n: int = 16) -> torch.Tensor:
    z, y, x = torch.meshgrid(*[torch.arange(n) - n / 2] * 3, indexing="ij")
    return 8.0 * torch.exp(-(x**2 + y**2 + z**2) / 12.0)


def _field() -> Field:
    specter.seed(0)
    return (
        Field(_template(), 2.0, 40)
        .pack(min_distance=12.0, max_distance_z=20.0)
        .solvate(model="random", thickness=0.0)
    )


def _draw(field: Field, seed: int, index: int) -> Sample:
    return field.draw(
        voltage=300.0, seed=seed, index=index, keep_dry=True, **PROPAGATION
    )


def _image(sample: Sample, ids: list[int], seed: int) -> torch.Tensor:
    n = len(ids)
    ctf = CTF({"dfu": torch.full((n,), 10000.0), "dfv": torch.full((n,), 10000.0)})
    exposure = Exposure(50.0, n_frames=4)
    with torch.no_grad():
        waves = sample.scatter(voltage=300.0, ids=torch.tensor(ids), **PROPAGATION)
        return (
            waves.apply_ctf(ctf)
            .intensity()
            .detect(exposure, coincidence_radius=1.0, seed=seed)
            .array
        )


def test_seeded_micrograph_is_independent_of_batching():
    """Every image of a batch sees the one sample; its noise is seeded on its
    own index, so one batch of four and four singles agree bit for bit. One
    sample for both, since a re-drawn sample carries CPU threaded-accumulation
    noise at 1e-6 in its ice, enough to move a Poisson draw."""
    sample = _draw(_field(), 5, 0)
    whole = _image(sample, [0, 1, 2, 3], 5)
    singles = torch.cat([_image(sample, [i], 5) for i in range(4)])
    assert torch.equal(whole, singles)
    assert torch.equal(whole, whole.round())
    assert whole.std() > 0
    # A second draw with the same seed places the same copies: the dry
    # canvas holds the crowd alone, with no blend to carry float noise.
    again = _draw(_field(), 5, 0)
    assert again.dry is not None and sample.dry is not None
    assert torch.equal(again.dry, sample.dry)


def test_seeded_micrograph_sample_is_reproducible_and_redrawn():
    field = _field()
    a, b = _draw(field, 5, 0), _draw(field, 5, 0)
    # The draws are the contract: same placements, bit for bit. The blended
    # volume itself carries CPU threaded-accumulation noise at the 1e-6 level.
    assert a.dry is not None and b.dry is not None
    assert torch.equal(a.dry, b.dry)
    assert torch.allclose(a.volume, b.volume, atol=1e-4)
    a2, b2 = _draw(field, 5, 1), _draw(field, 5, 1)
    assert a2.dry is not None and b2.dry is not None
    assert not torch.allclose(a2.volume, a.volume, atol=1e-4)
    assert not torch.equal(a2.dry, a.dry)
    # The second draw is itself reproducible.
    assert torch.equal(a2.dry, b2.dry)
    assert torch.allclose(a2.volume, b2.volume, atol=1e-4)


def _tilt_series(seed: int):
    specter.seed(0)
    return Sample.from_volume(_template(32)[None], 2.0).tilt_series(
        Poses.tilt_series([-20.0, 0.0, 20.0]),
        Exposure.tilt_series(20.0, 3, n_frames=4),
        voltage=300.0,
        micrograph_size=32,
        ice=Ice(model="random", thickness=0.0),
        seed=seed,
        propagation=Propagation(**PROPAGATION),
        log=False,
    )


def _series_images(series, seed: int) -> torch.Tensor:
    ctf = CTF({"dfu": torch.full((1,), 10000.0), "dfv": torch.full((1,), 10000.0)})
    images = []
    with torch.no_grad():
        for i in range(len(series)):
            image = (
                series.scatter(i)
                .apply_ctf(ctf)
                .intensity()
                .detect(series.exposure[i : i + 1], coincidence_radius=1.0, seed=seed)
            )
            images.append(image.array)
    return torch.cat(images)


def test_seeded_tilt_series_is_reproducible():
    """
    A seeded series' noise is a function of (seed, series, tilt): two passes
    give the same series bit for bit. Across two series built with the same
    seed the ice DRAW is the same, but its blend carries CPU
    threaded-accumulation noise at the 1e-6 level, which re-realises a
    Poisson image, so the volumes are compared and the images are not.
    """
    series = _tilt_series(9)
    a = _series_images(series, 9)
    assert torch.equal(a, _series_images(series, 9))
    assert torch.equal(a, a.round())
    assert a.std() > 0
    # Each tilt draws its own noise.
    assert not torch.equal(a[0], a[1])
    other = _tilt_series(9)
    assert torch.allclose(series.volume, other.volume, atol=1e-4)
    assert not torch.allclose(series.volume, _tilt_series(10).volume, atol=1e-4)
