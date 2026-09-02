"""
The ice canvas must be bulk all the way to its faces.

Until 2026-09-02 ``generate_big_ice`` convolved its molecule deltas with the
water kernel as a linear 'same' convolution on an even-sized kernel. Two
things followed: a face molecule's kernel was cut off at the face, so the
face plane carried half its potential and the next plane lost the kernel's
tail; and the even kernel's origin sat between voxels, so every potential
landed half a voxel off its molecule (which is why the deficit was 50% on
the low faces and 8% on the high ones). Solvate's reflect padding then
mirrored that deficit into a dark line straddling the box boundary on
every slice, visible after the CTF as a rim and ring scaling with ice
thickness.

These tests pin the two halves of the fix: kernels are odd and centred on
their voxel, and the ice path convolves periodically.
"""

from __future__ import annotations

import pytest
import torch

from specter.ice import IceBank
from specter.ice._kernels import build_water_kernel
from specter.potential._builders import (
    build_atomic_potential_kernel,
    compute_supersampling_parameters,
    potential_from_deltas,
)


@pytest.mark.parametrize("dx", [0.05, 0.5, 0.731, 1.0, 1.5, 2.0, 4.0])
def test_kernel_grid_is_even_fine_and_odd_pooled(dx):
    ssn, ssdx, ssf = compute_supersampling_parameters(dx)
    assert ssn % 2 == 0, "fine grid must be even: no sample at the 1/r origin"
    assert ssf % 2 == 0
    assert (ssn // ssf) % 2 == 1, "pooled kernel must have a centre voxel"
    assert ssn // ssf >= 3
    assert ssdx == pytest.approx(dx / ssf)


@pytest.mark.parametrize("dx", [0.731, 1.0, 1.5])
def test_kernel_is_finite_and_centred(dx):
    k = build_atomic_potential_kernel(dx, "kirkland", atomic_number=8)
    assert torch.isfinite(k).all()
    n = k.shape[0]
    assert n % 2 == 1
    c = n // 2
    assert k[c, c, c] == k.max()
    # Symmetric about the centre voxel along every axis (float32 pooling
    # of the fine grid leaves ~1e-7 relative scatter).
    tol = 1e-5 * float(k.max())
    for axis in range(3):
        assert torch.allclose(k, k.flip(axis), atol=tol)


@pytest.mark.parametrize("backend", ["fftconvolve", "conv3d"])
@pytest.mark.parametrize("boundary", ["linear", "periodic"])
@pytest.mark.parametrize("dx", [0.731, 1.5])
def test_potential_lands_on_its_delta(backend, boundary, dx):
    """A single molecule's potential is centred on the voxel it sits in."""
    k = build_water_kernel(dx)
    n = 24
    d = torch.zeros(n, n, n)
    d[12, 12, 12] = 1.0
    p = potential_from_deltas(d, k, backend=backend, boundary=boundary)
    x = torch.arange(n, dtype=torch.float32)
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        centroid = float((p.sum(other) * x).sum() / p.sum())
        assert centroid == pytest.approx(12.0, abs=1e-4)
    assert p.sum() == pytest.approx(float(k.sum()), rel=1e-5)


@pytest.mark.parametrize("backend", ["fftconvolve", "conv3d"])
def test_periodic_wraps_a_face_delta_and_matches_linear_inside(backend):
    k = build_water_kernel(1.0)
    n = 20
    d = torch.zeros(n, n, n)
    d[0, 10, 10] = 1.0  # on the low z face
    lin = potential_from_deltas(d, k, backend=backend, boundary="linear")
    per = potential_from_deltas(d, k, backend=backend, boundary="periodic")
    # Linear loses the planes of the kernel that fall below z = 0; periodic
    # keeps them, wrapped to the top planes.
    r = k.shape[0] // 2
    lost = float(k[:r].sum())
    assert lost > 0.05 * float(k.sum())
    assert lin.sum() == pytest.approx(float(k.sum()) - lost, rel=1e-5)
    assert per.sum() == pytest.approx(float(k.sum()), rel=1e-5)
    assert per[-r:].sum() == pytest.approx(lost, rel=1e-5)
    # Away from the wrap both agree (to FFT float noise).
    tol = 1e-5 * float(k.max())
    assert torch.allclose(per[r : n - r], lin[r : n - r], atol=tol)
    # And on an interior delta the two are identical everywhere.
    d2 = torch.zeros(n, n, n)
    d2[10, 10, 10] = 1.0
    assert torch.allclose(
        potential_from_deltas(d2, k, backend=backend, boundary="periodic"),
        potential_from_deltas(d2, k, backend=backend, boundary="linear"),
        atol=tol,
    )


def test_periodic_backends_agree():
    k = build_water_kernel(0.731)
    g = torch.Generator().manual_seed(0)
    d = torch.rand(16, 18, 20, generator=g)
    a = potential_from_deltas(d, k, backend="fftconvolve", boundary="periodic")
    b = potential_from_deltas(d, k, backend="conv3d", boundary="periodic")
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4 * float(a.abs().max()))


def test_periodic_rejects_a_kernel_larger_than_the_volume():
    k = build_water_kernel(0.5)
    with pytest.raises(ValueError, match="larger than the volume"):
        potential_from_deltas(torch.zeros(4, 4, 4), k, boundary="periodic")


def test_ice_canvas_faces_are_bulk():
    """
    Every face plane of the canvas has the interior's mean potential. This
    is the assertion that would have caught the half-density face.
    """
    torch.manual_seed(0)
    bank = IceBank(device="cpu", progressbars=False)
    ice = bank.generate_big_ice(n=48, dx=1.0, nz=40, batchsize=1)[0]
    m = float(ice.mean())
    for axes in ((1, 2), (0, 2), (0, 1)):
        planes = ice.mean(dim=axes)
        interior_std = float(planes[6:-6].std())
        for face in (planes[0], planes[1], planes[-1], planes[-2]):
            assert abs(float(face) - m) < 4 * interior_std + 0.05 * m
