"""
Tests for the forward-model memory/throughput formulations, each pinned
against the simpler formulation it is equivalent to:

* `affine_sampling_grid` replaces `F.affine_grid` in every rotation grid.
* `VolumeRotator` no longer carries a persistent identity grid.
* `Scattering.multislice` complexifies per chunk instead of whole-volume.
* `gaussian_blur3d` runs as conv1d passes instead of cudnn conv3d.
* `solvate` reflect-indexes the ice per slab instead of padding a canvas.
* `CrowdWithDuplicates.forward(into=...)` stamps onto an existing canvas, and
  `process_volume` no longer keeps a CPU copy of the crowd canvas.
* `poisson_disk_neighbors_3d`'s neighbour test is one distance check.
* `potential_from_deltas(backend="auto")` picks FFT for large kernels.

The 2026-09-04 thick-ice pass is pinned next to what it changed instead:
`rotation_safe_crop` in tests/test_crowding.py, and the periodic
convolution's chunking in tests/test_potential.py.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import specter
from specter import rotations
from specter.coords import poisson_disk_neighbors_3d
from specter.crowding import CrowdWithDuplicates
from specter.fft import fftconvolve, spatial_convolve3d_same
from specter.ice import IceBank
from specter.imagegenerator import ImageGenerator
from specter.imagegenerator._particle_base import _reflect_index
from specter.potential import apply_amplitude_contrast
from specter.potential._builders import _deltas_backend, potential_from_deltas
from specter.filters import gaussian_blur3d
from specter.rotations import VolumeRotator, affine_sampling_grid
from specter.rotations._volume import _relion_rotation_grid
from specter.scattering import Scattering
from specter.settings import Camera, Ice, Propagation


def _random_theta(B: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    R = rotations.random_rotation_matrix(B).to(dtype)
    t = (torch.rand(B, 3, dtype=dtype) - 0.5) * 0.4
    return rotations.build_affine_matrix(R, t)


# ---------------------------------------------------------------------------
# rotation grids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("shape", [(9, 12, 7), (16, 16, 16)])
def test_affine_sampling_grid_matches_affine_grid(align_corners, shape):
    """The broadcast grid is F.affine_grid to float64 rounding, on
    non-cubic shapes and both align_corners conventions."""
    nz, ny, nx = shape
    theta = _random_theta(3, torch.float64)
    want = F.affine_grid(theta, [3, 1, nz, ny, nx], align_corners=align_corners)
    got = affine_sampling_grid(theta, nz, ny, nx, align_corners)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-13, rtol=0)


def _six_pass_reference(
    theta: torch.Tensor, nz: int, ny: int, nx: int, origin: str
) -> torch.Tensor:
    """The chain VolumeRotator used to apply to a cached identity grid:
    recentre, rescale, rotate, unscale, uncentre, translate -- evaluated
    entirely in float64 (the rotator's own `center_dc` buffer is float32,
    which is one of the roundings the collapsed form removes)."""
    B = theta.shape[0]
    eye = torch.eye(3, 4, dtype=torch.float64).unsqueeze(0)
    g = F.affine_grid(eye, [1, 1, nz, ny, nx], align_corners=False)
    g = g.expand(B, -1, -1, -1, -1).reshape(B, -1, 3)
    R, t = theta[..., :3], theta[..., 3]
    s = torch.tensor(
        [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2], dtype=torch.float64
    ).view(1, 1, 3)
    if origin == "relion":
        c = torch.tensor(
            [
                2 * (nx // 2 + 0.5) / nx - 1,
                2 * (ny // 2 + 0.5) / ny - 1,
                2 * (nz // 2 + 0.5) / nz - 1,
            ],
            dtype=torch.float64,
        ).view(1, 1, 3)
        g = ((g - c) * s) @ R.transpose(1, 2) / s + c + t.unsqueeze(1)
    else:
        g = (g * s) @ R.transpose(1, 2) / s + t.unsqueeze(1)
    return g.view(B, nz, ny, nx, 3)


@pytest.mark.parametrize("origin", ["relion", "center"])
def test_volume_rotator_grid_matches_six_pass_reference(origin):
    """VolumeRotator._build_grid is the collapsed form of the recentre /
    rescale / rotate / unscale / uncentre / translate chain it replaced."""
    nz, ny, nx = 10, 14, 12
    rot = VolumeRotator(nz, ny, nx, origin=origin).double()
    theta = _random_theta(2, torch.float64)
    got = rot._build_grid(theta)
    want = _six_pass_reference(theta, nz, ny, nx, origin)
    assert torch.allclose(got, want, atol=1e-13, rtol=0)


def test_volume_rotator_has_no_persistent_identity_grid():
    """A cached (nz, ny, nx, 3) identity grid would be 1.6 GB at 512^3."""
    rot = VolumeRotator(8, 8, 8)
    assert "base_grid" not in dict(rot.named_buffers())


def test_rotate_volume_relion_grid_matches_float64_reference():
    """_relion_rotation_grid, through affine_sampling_grid, against the
    six-pass chain evaluated in float64."""
    nz, ny, nx = 11, 9, 13
    theta = _random_theta(2, torch.float64)
    got = _relion_rotation_grid(theta, nz, ny, nx, False)
    want = _six_pass_reference(theta, nz, ny, nx, "relion")
    assert torch.allclose(got, want, atol=1e-13, rtol=0)


def test_rotation_grid_gradients_flow_to_pose():
    """Pose refinement in Ghostbuster differentiates through the grid."""
    q = rotations.random_quaternion(1).double().requires_grad_(True)
    t = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    import roma

    theta = rotations.build_affine_matrix(roma.unitquat_to_rotmat(q), t)
    V = torch.rand(8, 8, 8, dtype=torch.float64)
    out = rotations.rotate_volume(V, theta)
    (out * torch.rand_like(out)).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert t.grad is not None and torch.isfinite(t.grad).all()


# ---------------------------------------------------------------------------
# multislice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.07])
def test_multislice_forward_complexifies_per_chunk_bitwise(alpha):
    """forward(real V) == multislice(apply_amplitude_contrast(V)) exactly:
    the complexification is elementwise, so doing it per chunk changes
    nothing but the peak memory."""
    n, nz = 16, 13  # nz not a multiple of the chunk size
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="multislice",
        alpha=alpha,
        nz=nz,
        progressbars=False,
    )
    torch.manual_seed(0)
    V = torch.rand(2, nz, n, n) * 4.0
    got = scat(V)
    want = scat.multislice(apply_amplitude_contrast(V, alpha=alpha))
    assert torch.equal(got, want)


def test_multislice_gradient_through_real_volume_matches_complex_path():
    n, nz = 16, 8
    scat = Scattering(
        nxy=n,
        pixel_size=1.0,
        voltage=300.0,
        scattering_model="multislice",
        alpha=0.07,
        nz=nz,
        progressbars=False,
    )
    torch.manual_seed(0)
    V = torch.rand(1, nz, n, n) * 4.0

    v1 = V.clone().requires_grad_(True)
    scat(v1).real.sum().backward()
    v2 = V.clone().requires_grad_(True)
    scat.multislice(apply_amplitude_contrast(v2, alpha=0.07)).real.sum().backward()
    assert torch.equal(v1.grad, v2.grad)


# ---------------------------------------------------------------------------
# occupancy blur
# ---------------------------------------------------------------------------


def _blur_reference(V: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """The cudnn conv3d, one-axis-at-a-time formulation this replaced."""
    r = max(1, int(round(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, device=V.device, dtype=V.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    kernel = kernel / kernel.sum()
    lead = V.shape[:-3]
    out = V.reshape(-1, 1, *V.shape[-3:])
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = -1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), kernel.view(*shape))
    return out.reshape(*lead, *V.shape[-3:])


@pytest.mark.parametrize("shape", [(7, 9, 11), (2, 6, 8, 5), (2, 1, 5, 5, 5)])
def test_gaussian_blur3d_matches_conv3d_reference(shape):
    torch.manual_seed(0)
    V = torch.rand(*shape, dtype=torch.float64) * 7
    got = gaussian_blur3d(V, 2.0 / 0.731)
    want = _blur_reference(V, 2.0 / 0.731)
    assert got.shape == V.shape
    assert torch.allclose(got, want, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# solvate
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# crowding
# ---------------------------------------------------------------------------


def test_crowd_forward_into_matches_accumulate_then_add():
    torch.manual_seed(0)
    V = torch.rand(12, 12, 12)
    crowd = CrowdWithDuplicates(V, 2.0, 10.0, nxy_out=24, nz_out=12, progressbars=False)
    canvas = torch.rand(12, 24, 24)

    specter.seed(11)
    want = canvas + crowd()
    specter.seed(11)
    got = canvas.clone()
    ret = crowd(into=got)
    assert ret is got
    assert crowd.N > 1
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


# ---------------------------------------------------------------------------
# poisson-disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["origin", "random"])
def test_poisson_disk_3d_respects_min_distance_and_is_deterministic(seed):
    torch.manual_seed(5)
    pts = poisson_disk_neighbors_3d(10.0, box=(60.0, 80.0, 70.0), seed=seed)
    assert len(pts) > 20
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))
    assert d.min() >= 10.0
    assert (pts.abs() <= torch.tensor([35.0, 40.0, 30.0])).all()
    torch.manual_seed(5)
    again = poisson_disk_neighbors_3d(10.0, box=(60.0, 80.0, 70.0), seed=seed)
    assert torch.equal(pts, again)


# ---------------------------------------------------------------------------
# ice kernel convolution backend
# ---------------------------------------------------------------------------


def test_potential_from_deltas_auto_backend_selection():
    small = torch.zeros(1, 16, 16, 16)
    assert _deltas_backend(small, torch.zeros(3, 3, 3)) == "conv3d"
    assert _deltas_backend(small, torch.zeros(4, 4, 4)) == "conv3d"
    assert _deltas_backend(small, torch.zeros(5, 5, 5)) == "fftconvolve"
    # a volume too large for the FFT's complex copies (stride-0 expand, no memory)
    huge = torch.zeros(1).expand(1, 700, 700, 700)
    assert _deltas_backend(huge, torch.zeros(8, 8, 8)) == "conv3d"


@pytest.mark.parametrize("k", [3, 8])
def test_potential_from_deltas_backends_agree(k):
    torch.manual_seed(0)
    deltas = (torch.rand(2, 20, 18, 22) < 0.05).float()
    kernel = torch.rand(k, k, k)
    a = potential_from_deltas(deltas, kernel, backend="conv3d")
    b = potential_from_deltas(deltas, kernel, backend="fftconvolve")
    c = potential_from_deltas(deltas, kernel, backend="auto")
    assert torch.allclose(a, b, atol=1e-4, rtol=0)
    assert torch.equal(c, a if k == 3 else b)
    assert torch.allclose(a[0], fftconvolve(deltas[0], kernel, mode="same"), atol=1e-4)
    assert torch.equal(a, spatial_convolve3d_same(deltas, kernel))


def test_ice_bank_crop_stays_on_bank_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = IceBank().to(device)
    torch.manual_seed(0)
    ice = bank.generate_ice(n=24, dx=1.0, nz=24)
    assert ice.device.type == device
    assert bank.positions is not None and bank.positions.device.type == device
    assert math.isfinite(bank.mlbop_energy()["E_per_atom"])
