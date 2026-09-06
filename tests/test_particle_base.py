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


def test_solvate_slab_count_is_capped():
    from specter.imagegenerator._particle_base import (
        _SOLVATE_MAX_SLICES,
        _solvate_chunk_slices,
    )

    assert _solvate_chunk_slices(1024) == 64
    assert _solvate_chunk_slices(512) == 64
    assert _solvate_chunk_slices(64) == _SOLVATE_MAX_SLICES
    assert _solvate_chunk_slices(20000) == 1


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
