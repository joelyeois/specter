"""
Per-draw seeding on the micrograph and tilt-series generators.

The particle generators' version is in tests/test_particle_base.py; these
pin the same contract for the other two: a seed makes the output a function
of (seed, draw index) and nothing else.
"""

from __future__ import annotations

import torch

import specter
from specter.imagegenerator import MicrographGenerator, TiltSeriesGenerator
from specter.settings import Camera, Crowding, Ice, Propagation
from specter.specimen import MicrographSpecimenGenerator


def _template(n: int = 16) -> torch.Tensor:
    z, y, x = torch.meshgrid(*[torch.arange(n) - n / 2] * 3, indexing="ij")
    return 8.0 * torch.exp(-(x**2 + y**2 + z**2) / 12.0)


def _micrograph_generator(seed: int | None) -> MicrographGenerator:
    specter.seed(0)
    n_images = 4
    specimen = MicrographSpecimenGenerator(
        _template(),
        pixel_size=2.0,
        nxy=40,
        ice=Ice(model="random", thickness=0.0),
        crowding=Crowding(min_distance=12.0, max_distance_z=20.0),
        progressbars=False,
    )
    ctf = {
        "dfu": torch.full((n_images,), 10000.0),
        "dfv": torch.full((n_images,), 10000.0),
    }
    gen = MicrographGenerator(
        specimen,
        40,
        2.0,
        ctf,
        300.0,
        50.0,
        propagation=Propagation(oversample=1, alpha=0.07),
        camera=Camera(noise_model="poisson", n_frames=4),
        coincidence_radius=1.0,
        progressbars=False,
        verbose=False,
        seed=seed,
    )
    gen.eval()
    return gen


def test_seeded_micrograph_generator_is_independent_of_batching():
    """Every image in a batch sees the one specimen; its noise is seeded on
    its own index, so one batch of four and four singles agree bit for bit.
    One generator for both, since a rebuilt specimen carries CPU
    threaded-accumulation noise at 1e-6, enough to move a Poisson draw."""
    gen = _micrograph_generator(5)
    with torch.no_grad():
        whole = gen(torch.tensor([0, 1, 2, 3]))
        singles = torch.cat([gen(torch.tensor([i])) for i in range(4)])
    assert torch.equal(whole, singles)
    assert torch.equal(whole, whole.round())
    assert whole.std() > 0
    # A second generator with the same seed reproduces the draws. Its images
    # are not compared: torch.poisson consumes a data-dependent number of
    # uniforms, so the 1e-6 volume noise re-realises the whole image.
    other = _micrograph_generator(5)
    with torch.no_grad():
        again = other(torch.tensor([0, 1, 2, 3]))
    assert torch.equal(gen.specimen_gen.crowd.coords, other.specimen_gen.crowd.coords)
    assert again.shape == whole.shape


def test_seeded_micrograph_specimen_is_reproducible_and_redrawn():
    a = _micrograph_generator(5)
    b = _micrograph_generator(5)
    with torch.no_grad():
        a(torch.tensor([0]))
        b(torch.tensor([0]))
    # The draws are the contract: same placements, bit for bit. The blended
    # volume itself carries CPU threaded-accumulation noise at the 1e-6 level.
    assert torch.equal(a.specimen_gen.crowd.coords, b.specimen_gen.crowd.coords)
    assert torch.allclose(a.volume, b.volume, atol=1e-4)
    first = a.volume.clone()
    first_coords = a.specimen_gen.crowd.coords.clone()
    a.regenerate_specimen()
    with torch.no_grad():
        a(torch.tensor([0]))
    assert not torch.allclose(a.volume, first, atol=1e-4)
    assert not torch.equal(a.specimen_gen.crowd.coords, first_coords)
    # The second draw is itself reproducible.
    b.regenerate_specimen()
    with torch.no_grad():
        b(torch.tensor([0]))
    assert torch.equal(a.specimen_gen.crowd.coords, b.specimen_gen.crowd.coords)
    assert torch.allclose(a.volume, b.volume, atol=1e-4)


def _tilt_generator(seed: int | None) -> TiltSeriesGenerator:
    specter.seed(0)
    volume = _template(32)[None]
    ctf = {"dfu": torch.full((3,), 10000.0), "dfv": torch.full((3,), 10000.0)}
    gen = TiltSeriesGenerator(
        volume,
        32,
        2.0,
        ctf,
        300.0,
        torch.full((3,), 20.0),
        angles=torch.tensor([-20.0, 0.0, 20.0]),
        propagation=Propagation(oversample=1, alpha=0.07),
        camera=Camera(noise_model="poisson", n_frames=4),
        ice=Ice(model="random", thickness=0.0),
        coincidence_radius=1.0,
        progressbars=False,
        verbose=False,
        seed=seed,
    )
    gen.eval()
    return gen


def _series(gen: TiltSeriesGenerator) -> torch.Tensor:
    with torch.no_grad():
        series, _, _ = gen.generate_tilt_series(torch.tensor([0]))
    return series


def test_seeded_tilt_series_is_reproducible():
    """
    A seeded generator's noise is a function of (seed, series, tilt): two
    calls give the same series bit for bit. Across two generators built
    with the same seed the ice DRAW is the same, but its blend carries CPU
    threaded-accumulation noise at the 1e-6 level, which re-realises a
    Poisson image, so the volumes are compared and the images are not.
    """
    gen = _tilt_generator(9)
    a = _series(gen)
    assert torch.equal(a, _series(gen))
    assert torch.equal(a, a.round())
    assert a.std() > 0
    # Each tilt draws its own noise.
    assert not torch.equal(a[0, 0], a[0, 1])
    other = _tilt_generator(9)
    assert torch.allclose(gen.volume, other.volume, atol=1e-4)
    assert not torch.allclose(gen.volume, _tilt_generator(10).volume, atol=1e-4)
