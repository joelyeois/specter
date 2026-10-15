"""
Tests for the particle stages (`specter.imaging._stages`) and the particle
chain: slab-wise solvation against the padded whole-canvas reference, the
per-batch crowd stamping, and per-particle seeding.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import specter
from specter import rotations
from specter.crowding import CrowdWithDuplicates
from specter.ice import RandomIcemaker
from specter.imaging import CTF, Exposure, Poses, Specimen, simulate_particles
from specter.imaging import _stages as stages
from specter.imaging._stages import reflect_index
from specter.potential import FULL_OCCUPANCY_POTENTIAL_V


@pytest.mark.parametrize("n,pad", [(8, 4), (9, 4), (16, 7)])
def test_reflect_index_reproduces_reflect_padding(n, pad):
    x = torch.arange(n, dtype=torch.float32).view(1, 1, n)
    want = F.pad(x, (pad, pad), mode="reflect").flatten()
    got = x.flatten()[reflect_index(n, pad, x.device)]
    assert torch.equal(got, want)


NXY = 16
PIXEL = 1.5


def _solvate(V: torch.Tensor, icemaker: RandomIcemaker, pad: int) -> torch.Tensor:
    """The chain's slab-wise solvation against the fixed occupancy reference."""
    return stages.solvate(
        V,
        icemaker,
        pixel_size=PIXEL,
        nxy=NXY,
        pad=pad,
        full=torch.full((1, 1, 1, 1), FULL_OCCUPANCY_POTENTIAL_V),
    )


def _solvate_reference(
    icemaker: RandomIcemaker, V: torch.Tensor, pad: int
) -> torch.Tensor:
    """solvate as it was written before: pad the whole ice canvas, then blend
    slab by slab, subtracting the already-added ice from the halo."""
    from specter.potential import occupancy_blur_halo_voxels, potential_occupancy

    ice = icemaker.generate_ice(batchsize=len(V)).to(V.device)
    if pad:
        ice = F.pad(ice, (pad, pad, pad, pad, 0, 0), mode="reflect")
    nz, nxy = V.shape[1], V.shape[-1]
    chunk = stages.solvate_chunk_slices(nxy)
    halo = occupancy_blur_halo_voxels(PIXEL)
    for start in range(0, nz, chunk):
        end = min(start + chunk, nz)
        lo, hi = max(0, start - halo), min(nz, end + halo)
        src = V[:, lo:hi].clone()
        if lo < start:
            src[:, : start - lo] -= ice[:, lo:start]
        occ = potential_occupancy(
            src, PIXEL, full_potential=FULL_OCCUPANCY_POTENTIAL_V
        )[:, start - lo : start - lo + (end - start)]
        ice[:, start:end].mul_(occ.neg_().add_(1.0))
        V[:, start:end].add_(ice[:, start:end])
    return V


@pytest.mark.parametrize("pad_fft", [False, True])
def test_solvate_without_padded_ice_canvas_matches_padded_reference(pad_fft):
    """Reflect-indexing each slab out of the unpadded ice gives the same volume
    as padding the whole ice canvas first (the ice is RandomIcemaker's, drawn
    under the same seed for both)."""
    pad = NXY // 2 if pad_fft else 0
    icemaker = RandomIcemaker(PIXEL, NXY, NXY)
    torch.manual_seed(0)
    V0 = torch.rand(2, NXY, NXY + 2 * pad, NXY + 2 * pad) * 3.0
    specter.seed(3)
    got = _solvate(V0.clone(), icemaker, pad)
    specter.seed(3)
    want = _solvate_reference(icemaker, V0.clone(), pad)
    assert torch.allclose(got, want, atol=1e-6, rtol=0)
    assert not torch.equal(got, V0)  # ice was actually added


def test_solvate_halo_spanning_several_slabs_matches_padded_reference(monkeypatch):
    """On a very large canvas the fixed slab budget makes the slab narrower
    than the blur halo, so the halo reaches back across several slabs.
    Force that (slab 2 slices, halo 8 at 1.5 A) and compare with the padded
    reference, which reads the halo off the whole canvas."""
    from specter.potential import occupancy_blur_halo_voxels

    monkeypatch.setattr(stages, "_SOLVATE_MAX_SLICES", 2)
    pad = NXY // 2
    assert occupancy_blur_halo_voxels(PIXEL) > stages.solvate_chunk_slices(
        NXY + 2 * pad
    )
    icemaker = RandomIcemaker(PIXEL, NXY, NXY)
    torch.manual_seed(0)
    V0 = torch.rand(2, NXY, NXY + 2 * pad, NXY + 2 * pad) * 3.0
    specter.seed(4)
    got = _solvate(V0.clone(), icemaker, pad)
    specter.seed(4)
    want = _solvate_reference(icemaker, V0.clone(), pad)
    assert torch.allclose(got, want, atol=1e-6, rtol=0)


def _crowd(template: torch.Tensor) -> CrowdWithDuplicates:
    pad_nxy = NXY + (NXY // 2) * 2
    return CrowdWithDuplicates(
        template, PIXEL, 20.0, nxy_out=pad_nxy, nz_out=NXY, progressbars=False
    )


def test_crowd_stamps_the_neighbours_into_the_canvas_in_place():
    """No second canvas: the copies land in the volume handed in, item by
    item, rather than in an accumulator added once (a 2 GB host copy per
    particle at a 512-pixel box)."""
    torch.manual_seed(0)
    template = torch.rand(NXY, NXY, NXY) * 3.0
    crowd = _crowd(template)
    V = torch.zeros(2, NXY, NXY + 2 * (NXY // 2), NXY + 2 * (NXY // 2))
    storage = V.data_ptr()
    stages.crowd_into(V, crowd, seeds=[1, 2])
    assert V.data_ptr() == storage
    assert crowd.N > 1 and V[0].sum() > 0 and V[1].sum() > 0


def test_a_crowded_batch_caps_cpu_threads_once(monkeypatch):
    """Resizing a 128-thread pool costs ~12 ms each way, and the per-chunk cap
    inside `insert_particles_into_micrograph` paid it 778 times for 64 particles
    (21 s of a 28 s run). The crowd stage caps once for the whole batch, so the
    inner caps find the pool already small: the full pool is restored exactly
    once per scatter."""
    from specter.cpu_threads import SMALL_OP_THREADS

    torch.manual_seed(0)
    specimen = Specimen(torch.rand(NXY, NXY, NXY) * 3.0, PIXEL).crowd(min_distance=20.0)
    poses = Poses(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2), torch.zeros(2, 2), torch.arange(2)
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
            specimen.scatter(poses, voltage=300.0, pad_fft=True, seed=0)
    finally:
        monkeypatch.undo()
        real_set(original)
    restores = [n for n in calls if n > SMALL_OP_THREADS]
    assert restores == [max(original, 2 * SMALL_OP_THREADS)], calls


@pytest.mark.parametrize("from_coordinates", [False, True])
def test_scalar_per_image_arguments_index_every_image_without_optics(
    from_coordinates: bool,
) -> None:
    """
    Scalar dose and coincidence radius apply to every image.

    They are broadcast to one entry per image. The image count once came only
    from ``ctf_params``, so with no optics (no CTF) the scalars stayed length
    1 and any index past 0 raised a CUDA device-side assert (an IndexError on
    the CPU).
    """
    n_images = 3
    poses = Poses(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_images),
        torch.zeros(n_images, 2),
        torch.arange(n_images),
    )
    if from_coordinates:
        g = torch.Generator().manual_seed(0)
        specimen = Specimen.from_atoms(
            torch.randn(20, 3, generator=g) * 3.0, torch.full((20,), 6), 1.0, 16
        )
    else:
        specimen = Specimen(torch.rand(16, 16, 16), 1.0)
    with torch.no_grad():
        images = simulate_particles(
            specimen,
            poses,
            None,
            Exposure(40.0),
            voltage=300.0,
            batch_size=2,
            camera=dict(noise="none", coincidence_radius=0.0),
        )
    assert images.array.shape == (n_images, 16, 16)
    assert torch.isfinite(images.array[-1]).all()


# ---------------------------------------------------------------------------
# Per-particle seeding: a seeded stack is a function of (seed, particle index)
# and nothing else -- not of how the particles were batched.
# ---------------------------------------------------------------------------


def _run_batched(seed: int | None, batches: list[list[int]]) -> torch.Tensor:
    """Every stochastic stage on: crowding, random ice, dose-fractionated
    Poisson noise with coincidence loss."""
    n, n_images = 24, 6
    torch.manual_seed(0)
    z, y, x = torch.meshgrid(*[torch.arange(n) - n / 2] * 3, indexing="ij")
    V = 8.0 * torch.exp(-(x**2 + y**2 + z**2) / 20.0)
    quats = rotations.random_quaternion(n_images).reshape(n_images, 4)
    poses = Poses(quats, torch.zeros(n_images, 2), torch.arange(n_images))
    ctf = CTF(
        {
            "dfu": torch.full((n_images,), 10000.0),
            "dfv": torch.full((n_images,), 10000.0),
        }
    )
    specimen = (
        Specimen(V, 2.0)
        .crowd(min_distance=10.0, max_distance_z=20.0)
        .solvate(model="random", thickness=0.0)
    )
    exposure = Exposure(50.0, n_frames=4)
    specter.seed(0)
    with torch.no_grad():
        return torch.cat(
            [
                simulate_particles(
                    specimen,
                    poses[torch.tensor(b)],
                    ctf[torch.tensor(b)],
                    exposure,
                    voltage=300.0,
                    seed=seed,
                    batch_size=len(b),
                    scatter=dict(oversample=1, alpha=0.07),
                    camera=dict(coincidence_radius=1.0),
                ).array
                for b in batches
            ]
        )


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
    """seed=None draws from the global RNG, where batching decides which draw
    lands on which particle."""
    whole = _run_batched(None, [[0, 1, 2, 3, 4, 5]])
    singles = _run_batched(None, [[i] for i in range(6)])
    assert not torch.equal(whole, singles)


@pytest.mark.parametrize("scattering_model", ["multislice", "firstborn", "rytov"])
def test_images_without_optics_come_out_at_the_box_size(scattering_model: str) -> None:
    """With no aberration stage the detector still samples down by the
    oversample, so a wave that never left the pixel grid (first Born) has to
    be put on the image grid first; it came out a quarter of the box."""
    torch.manual_seed(0)
    V = torch.zeros(32, 32, 32)
    V[12:20, 12:20, 12:20] = 5.0
    poses = Poses(
        rotations.random_quaternion(2).reshape(2, 4), torch.zeros(2, 2), torch.arange(2)
    )
    with torch.no_grad():
        waves = Specimen(V, 1.5).scatter(
            poses, voltage=300.0, scattering_model=scattering_model
        )
        images = waves.intensity().detect(Exposure(20.0), noise=None).array
    assert images.shape == (2, 32, 32)
    assert torch.allclose(images.mean(), torch.tensor(20.0 * 1.5**2), rtol=1e-3)
