"""
Tests for `ParticleGeneratorBase`: slab-wise solvation against the padded
whole-canvas reference, and the per-batch crowd stamping.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import specter
from specter import rotations
from specter.crowding import CrowdWithDuplicates
from specter.imagegenerator import ImageGenerator
from specter.imagegenerator._particle_base import _reflect_index
from specter.settings import Camera, Ice, Propagation


@pytest.mark.parametrize("n,pad", [(8, 4), (9, 4), (16, 7)])
def test_reflect_index_reproduces_reflect_padding(n, pad):
    x = torch.arange(n, dtype=torch.float32).view(1, 1, n)
    want = F.pad(x, (pad, pad), mode="reflect").flatten()
    got = x.flatten()[_reflect_index(n, pad, x.device)]
    assert torch.equal(got, want)


def _tiny_generator(pad_fft: bool, ice_model: str | None = "random") -> ImageGenerator:
    n = 16
    torch.manual_seed(0)
    V = torch.rand(n, n, n) * 3.0
    quats = rotations.random_quaternion(2)
    ctf = {
        "dfu": torch.tensor([10000.0, 12000.0]),
        "dfv": torch.tensor([10000.0, 12000.0]),
    }
    return ImageGenerator(
        V,
        1.5,
        quats,
        torch.zeros(2, 2),
        ctf,
        300.0,
        30.0,
        progressbars=False,
        verbose=False,
        propagation=Propagation(pad_fft=pad_fft),
        camera=Camera(noise_model=None),
        ice=Ice(model=ice_model, thickness=0),
    )


def _solvate_reference(model: ImageGenerator, V: torch.Tensor) -> torch.Tensor:
    """solvate as it was written before: pad the whole ice canvas, then blend
    slab by slab, subtracting the already-added ice from the halo."""
    from specter.imagegenerator._particle_base import _solvate_chunk_slices
    from specter.potential import (
        FULL_OCCUPANCY_POTENTIAL_V,
        occupancy_blur_halo_voxels,
        potential_occupancy,
    )

    ice = model.icemaker.generate_ice(batchsize=len(V)).to(V.device)
    if model.pad_fft:
        p = model.nxy // 2
        ice = F.pad(ice, (p, p, p, p, 0, 0), mode="reflect")
    nz, nxy = V.shape[1], V.shape[-1]
    chunk = _solvate_chunk_slices(nxy)
    halo = occupancy_blur_halo_voxels(model.pixel_size)
    for start in range(0, nz, chunk):
        end = min(start + chunk, nz)
        lo, hi = max(0, start - halo), min(nz, end + halo)
        src = V[:, lo:hi].clone()
        if lo < start:
            src[:, : start - lo] -= ice[:, lo:start]
        occ = potential_occupancy(
            src, model.pixel_size, full_potential=FULL_OCCUPANCY_POTENTIAL_V
        )[:, start - lo : start - lo + (end - start)]
        ice[:, start:end].mul_(occ.neg_().add_(1.0))
        V[:, start:end].add_(ice[:, start:end])
    return V


@pytest.mark.parametrize("pad_fft", [False, True])
def test_solvate_without_padded_ice_canvas_matches_padded_reference(pad_fft):
    """Reflect-indexing each slab out of the unpadded ice gives the same volume
    as padding the whole ice canvas first (the ice is RandomIcemaker's, drawn
    under the same seed for both)."""
    model = _tiny_generator(pad_fft)
    V0 = torch.rand(2, model.nz, model.pad_nxy, model.pad_nxy) * 3.0
    specter.seed(3)
    got = model.solvate(V0.clone())
    specter.seed(3)
    want = _solvate_reference(model, V0.clone())
    assert torch.allclose(got, want, atol=1e-6, rtol=0)
    assert not torch.equal(got, V0)  # ice was actually added


def test_solvate_halo_spanning_several_slabs_matches_padded_reference(monkeypatch):
    """On a very large canvas the fixed slab budget makes the slab narrower
    than the blur halo, so the halo reaches back across several slabs.
    Force that (slab 2 slices, halo 8 at 1.5 A) and compare with the padded
    reference, which reads the halo off the whole canvas."""
    import specter.imagegenerator._particle_base as pbm

    monkeypatch.setattr(pbm, "_SOLVATE_MAX_SLICES", 2)
    model = _tiny_generator(pad_fft=True)
    from specter.potential import occupancy_blur_halo_voxels

    assert occupancy_blur_halo_voxels(model.pixel_size) > pbm._solvate_chunk_slices(
        model.pad_nxy
    )
    V0 = torch.rand(2, model.nz, model.pad_nxy, model.pad_nxy) * 3.0
    specter.seed(4)
    got = model.solvate(V0.clone())
    specter.seed(4)
    want = _solvate_reference(model, V0.clone())
    assert torch.allclose(got, want, atol=1e-6, rtol=0)


def test_process_volume_keeps_no_cpu_copy_of_the_crowd_canvas():
    model = _tiny_generator(pad_fft=True, ice_model=None)
    model.crowd = CrowdWithDuplicates(
        model.V,
        model.pixel_size,
        20.0,
        nxy_out=model.pad_nxy,
        nz_out=model.nz,
        progressbars=False,
    )
    with torch.no_grad():
        model(torch.tensor([0]))
    assert not hasattr(model, "volumes")


def test_process_volume_caps_cpu_threads_once_per_batch(monkeypatch):
    """Resizing a 128-thread pool costs ~12 ms each way, and the per-chunk cap
    inside `insert_particles_into_micrograph` paid it 778 times for 64 particles
    (21 s of a 28 s run). `process_volume` caps once for the whole batch, so the
    inner caps find the pool already small: the full pool is restored exactly once."""
    from specter.cpu_threads import SMALL_OP_THREADS

    model = _tiny_generator(pad_fft=True, ice_model=None)
    model.crowd = CrowdWithDuplicates(
        model.V,
        model.pixel_size,
        20.0,
        nxy_out=model.pad_nxy,
        nz_out=model.nz,
        progressbars=False,
    )
    original = torch.get_num_threads()
    real_set = torch.set_num_threads
    calls: list[int] = []

    def counting_set(n: int) -> None:
        calls.append(n)
        real_set(n)

    real_set(max(original, 2 * SMALL_OP_THREADS))
    monkeypatch.setattr(torch, "set_num_threads", counting_set)
    try:
        with torch.no_grad():
            model(torch.tensor([0, 1]))
    finally:
        monkeypatch.undo()
        real_set(original)
    assert model.crowd.N > 1
    restores = [n for n in calls if n > SMALL_OP_THREADS]
    assert restores == [max(original, 2 * SMALL_OP_THREADS)], calls


@pytest.mark.parametrize("from_coordinates", [False, True])
def test_scalar_per_image_arguments_index_every_image_without_optics(
    from_coordinates: bool,
) -> None:
    """
    Scalar dose, coincidence radius and potential scale apply to every image.

    They are expanded to one entry per image, and the image count used to
    come only from ``ctf_params``. With ``optics=None`` there are none, so
    the scalars stayed length 1 and any index past 0 raised a CUDA
    device-side assert (an IndexError on the CPU).
    """
    from specter.imagegenerator import ImageGenerator, ImageGeneratorFromCoordinates
    from specter.settings import Camera

    n_images = 3
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_images)
    common = dict(
        quaternions=quats,
        translations=torch.zeros(n_images, 2),
        ctf_params=None,
        voltage=300.0,
        dose_per_angstrom=40.0,
        optics=None,
        camera=Camera(noise_model="none"),
        coincidence_radius=0.0,
        verbose=False,
    )
    if from_coordinates:
        g = torch.Generator().manual_seed(0)
        gen = ImageGeneratorFromCoordinates(
            coordinates=torch.randn(20, 3, generator=g) * 3.0,
            atomic_numbers=torch.full((20,), 6),
            nxy=16,
            pixel_size=1.0,
            **common,
        )
    else:
        gen = ImageGenerator(
            torch.rand(16, 16, 16),
            1.0,
            progressbars=False,
            potential_scale=1.0,
            **common,
        )
    for name in ("dose_per_angstrom", "coincidence_radius", "potential_scale"):
        assert getattr(gen, name).shape == (n_images,)
    with torch.no_grad():
        images = gen(torch.tensor([n_images - 1]))
    assert images.shape[0] == 1


# ---------------------------------------------------------------------------
# Per-particle seeding: a seeded stack is a function of (seed, particle index)
# and nothing else -- not of how the particles were batched.
# ---------------------------------------------------------------------------


def _seeded_generator(seed: int | None) -> ImageGenerator:
    """Every stochastic stage on: crowding, random ice, dose-fractionated
    Poisson noise with coincidence loss."""
    from specter.settings import Crowding

    n, n_images = 24, 6
    torch.manual_seed(0)
    z, y, x = torch.meshgrid(*[torch.arange(n) - n / 2] * 3, indexing="ij")
    V = 8.0 * torch.exp(-(x**2 + y**2 + z**2) / 20.0)
    quats = rotations.random_quaternion(n_images).reshape(n_images, 4)
    ctf = {
        "dfu": torch.full((n_images,), 10000.0),
        "dfv": torch.full((n_images,), 10000.0),
    }
    specter.seed(0)
    gen = ImageGenerator(
        V,
        2.0,
        quats,
        torch.zeros(n_images, 2),
        ctf,
        300.0,
        50.0,
        propagation=Propagation(oversample=1, alpha=0.07),
        camera=Camera(noise_model="poisson", n_frames=4),
        ice=Ice(model="random", thickness=0.0),
        crowding=Crowding(min_distance=10.0, max_distance_z=20.0),
        coincidence_radius=1.0,
        seed=seed,
        progressbars=False,
        verbose=False,
    )
    gen.eval()
    return gen


def _run_batched(seed: int | None, batches: list[list[int]]) -> torch.Tensor:
    gen = _seeded_generator(seed)
    with torch.no_grad():
        return torch.cat([gen(torch.tensor(b)) for b in batches])


def test_seeded_generator_is_independent_of_batching():
    """One batch of six, two of three and six singles give the same stack,
    bit for bit: each particle's crowd, ice and noise come from generators
    seeded on its own index, so the batch boundaries cannot move a draw from
    one particle to another. This is what lets batchsize='auto' and DDP
    reproduce a seeded run."""
    whole = _run_batched(123, [[0, 1, 2, 3, 4, 5]])
    halves = _run_batched(123, [[0, 1, 2], [3, 4, 5]])
    singles = _run_batched(123, [[i] for i in range(6)])
    assert torch.equal(whole, halves)
    assert torch.equal(whole, singles)
    # The stages really ran: integer counts (Poisson) on a non-trivial image.
    assert torch.equal(whole, whole.round())
    assert whole.std() > 0


def test_seeded_generator_depends_on_the_seed_and_the_index():
    a = _run_batched(123, [[0, 1, 2, 3, 4, 5]])
    b = _run_batched(124, [[0, 1, 2, 3, 4, 5]])
    assert not torch.equal(a, b)
    # Particle 3 drawn alone is particle 3, not "the first particle of a batch".
    alone = _run_batched(123, [[3]])
    assert torch.equal(alone[0], a[3])


def test_unseeded_generator_keeps_the_global_stream():
    """seed=None is the old behaviour: the global RNG, where batching decides
    which draw lands on which particle."""
    whole = _run_batched(None, [[0, 1, 2, 3, 4, 5]])
    singles = _run_batched(None, [[i] for i in range(6)])
    assert not torch.equal(whole, singles)
    assert _seeded_generator(None).seed is None


def test_seed_is_a_buffer_so_ddp_broadcasts_it():
    """Lightning syncs rank 0's buffers to the workers before predicting; a
    plain attribute would leave each worker deriving streams from its own
    seed."""
    gen = _seeded_generator(7)
    assert gen.seed == 7
    assert "_seed_buffer" in dict(gen.named_buffers())
    assert "_seed_buffer" not in gen.state_dict()
