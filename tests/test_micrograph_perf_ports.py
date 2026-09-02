"""
Pin the 2026-09 `simulate micrograph` speedups to what they replaced.

Three changes, each exact or float-order only:

* small-op loops run under a CPU thread cap (`cpu_threads`), which changes
  nothing about the draws or the arithmetic;
* an IceBank blend into a host-resident volume is built and blended on the
  bank's device a z-slab at a time, matching the whole-canvas path;
* the periodic `conv3d` branch of `potential_from_deltas` pads a z-chunk at
  a time instead of the whole canvas, matching the FFT branch.
"""

from __future__ import annotations

import pytest
import torch

import specter
from specter.coords import poisson_disk_neighbors, poisson_disk_neighbors_3d
from specter.cpu_threads import limited_cpu_threads
from specter.ice import IceBank, blend_ice_into_volume
from specter.ice._kernels import build_water_kernel
from specter.potential import potential_from_deltas, potential_occupancy


def test_limited_cpu_threads_only_lowers_and_restores():
    before = torch.get_num_threads()
    with limited_cpu_threads(max(1, before - 1)):
        assert torch.get_num_threads() == max(1, before - 1)
    assert torch.get_num_threads() == before
    with limited_cpu_threads(before + 5):
        assert torch.get_num_threads() == before
    assert torch.get_num_threads() == before


def test_poisson_disk_samplers_are_identical_under_the_thread_cap():
    """The cap is a performance setting: the draws and the accept/reject
    arithmetic are the same at any thread count."""
    before = torch.get_num_threads()
    specter.seed(0)
    a3 = poisson_disk_neighbors_3d(30.0, box=(64.0, 128.0, 128.0))
    specter.seed(0)
    a2 = poisson_disk_neighbors(30.0, box=(128.0, 128.0))
    torch.set_num_threads(1)
    try:
        specter.seed(0)
        b3 = poisson_disk_neighbors_3d(30.0, box=(64.0, 128.0, 128.0))
        specter.seed(0)
        b2 = poisson_disk_neighbors(30.0, box=(128.0, 128.0))
    finally:
        torch.set_num_threads(before)
    assert torch.equal(a3, b3) and len(a3) > 5
    assert torch.equal(a2, b2) and len(a2) > 5
    assert torch.get_num_threads() == before


@pytest.mark.parametrize("dx", [0.731, 1.0])
def test_periodic_conv3d_chunking_matches_fft(dx):
    """The chunked circular-pad conv3d path equals the FFT path, with the
    chunk forced small enough that a volume spans several chunks."""
    from specter.potential import _builders

    k = build_water_kernel(dx)
    g = torch.Generator().manual_seed(0)
    d = torch.rand(1, 40, 24, 20, generator=g)
    a = potential_from_deltas(d, k, backend="fftconvolve", boundary="periodic")
    old = _builders._DELTAS_FFT_MAX_VOXELS
    _builders._DELTAS_FFT_MAX_VOXELS = 24 * 20 * 7
    try:
        b = potential_from_deltas(d, k, backend="conv3d", boundary="periodic")
    finally:
        _builders._DELTAS_FFT_MAX_VOXELS = old
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4 * float(a.abs().max()))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a device for the host-canvas path"
)
def test_host_volume_blend_matches_whole_canvas_blend():
    """
    A host-resident volume blended slab by slab on the device equals the
    whole-canvas blend of the same draws, to float noise, including at the
    z ends where the slab path wraps its context.
    """
    dev = torch.device("cuda")
    n, nz, dx = 96, 70, 1.0
    bank = IceBank(device=dev, progressbars=False)
    V0 = torch.zeros(1, nz, n, n)
    g = torch.Generator().manual_seed(1)
    for _ in range(8):
        c = (torch.rand(3, generator=g) * torch.tensor([nz, n, n])).long()
        zs, ys, xs = [slice(max(0, int(ci) - 8), int(ci) + 8) for ci in c]
        V0[0, zs, ys, xs] += 10.0

    # Reference: the whole-canvas path, forced by putting V on the device.
    gen = torch.Generator().manual_seed(7)
    ice = bank.generate_big_ice(
        n=n, dx=dx, nz=nz, batchsize=1, device=dev, generator=gen
    )
    ref = V0.to(dev) + ice * (1 - potential_occupancy(V0.to(dev), dx)).clamp(0, 1)

    # Slab path: V on the host, bank on the device, same draws, and a slab
    # small enough that the volume spans several.
    from specter.ice import _bank as bank_mod

    torch.manual_seed(0)
    new = V0.clone()
    gen2 = torch.Generator().manual_seed(7)
    real = bank.big_ice_positions

    def seeded_positions(*a, **kw):
        kw["generator"] = gen2
        return real(*a, **kw)

    bank.big_ice_positions = seeded_positions  # type: ignore[method-assign]
    bank_mod._blend_ice_slabwise(new, bank, dx, slab_voxels=n * n * 16)
    assert new.device.type == "cpu"
    diff = new.to(dev) - ref
    assert float(diff.abs().max()) < 1e-3 * float(ice.abs().max())
    # Plane means agree at both z ends (the wrapped context).
    pm = (new.to(dev) - ref).mean(dim=(0, 2, 3))
    assert float(pm.abs().max()) < 1e-4 * float(ice.mean())

    # And the public entry point routes a host V to this path.
    bank.big_ice_positions = real  # type: ignore[method-assign]
    out = blend_ice_into_volume(V0.clone(), bank, dx, inplace=True)
    assert out.device.type == "cpu"
    assert float((out - V0).mean()) == pytest.approx(float(ice.mean()), rel=0.05)
