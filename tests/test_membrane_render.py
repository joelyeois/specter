"""`render_bilayer`: the calibrated profile laid along a surface's normal."""

import math

import pytest
import torch

from specter.specimen.membrane._geometry import Grid
from specter.specimen.membrane._primitives import ball
from specter.specimen.membrane._profile import (
    BilayerProfile,
    build_measured_bilayer_profile,
)
from specter.specimen.membrane._render import profile_table, render_bilayer

_RADIUS = 120.0


@pytest.fixture(scope="module")
def profile() -> BilayerProfile:
    return build_measured_bilayer_profile()


def _vesicle(
    profile: BilayerProfile, voxel_size: float, block: int = 64
) -> torch.Tensor:
    """A 120 A sphere's bilayer in a 400 A box centred on the origin."""
    grid = Grid((400.0, 400.0, 400.0), lo=(-200.0, -200.0, -200.0))
    n = int(round(400.0 / voxel_size))
    return render_bilayer(
        [ball(grid, (0.0, 0.0, 0.0), _RADIUS)],
        grid,
        profile,
        (n, n, n),
        voxel_size,
        block=block,
    )


def _profile_integral(profile: BilayerProfile) -> float:
    return float(torch.trapezoid(profile.psi.cpu(), profile.distance_angstrom.cpu()))


def test_profile_table_keeps_the_integral_and_ends_at_zero(
    profile: BilayerProfile,
) -> None:
    """The table is psi(d) averaged over one voxel along the normal: a box
    average moves no potential, at any voxel size."""
    expected = _profile_integral(profile)
    for voxel_size in (2.0, 5.0, 12.0):
        table, d0, step = profile_table(profile, voxel_size)
        assert d0 < float(profile.distance_angstrom.min())
        assert d0 + step * (table.numel() - 1) > float(profile.distance_angstrom.max())
        assert float(table[0]) == 0.0 and float(table[-1]) == 0.0
        assert float(table.sum()) * step == pytest.approx(expected, rel=0.01)
    # A coarser voxel averages more: the headgroup peak drops.
    fine, _, _ = profile_table(profile, 2.0)
    coarse, _, _ = profile_table(profile, 12.0)
    assert float(coarse.max()) < float(fine.max())


def test_vesicle_shell_follows_the_profile_about_its_radius(
    profile: BilayerProfile,
) -> None:
    """Along a ray from the centre the volume is the profile table read at
    the signed distance r - R, so the shell sits on the mid-surface."""
    voxel_size = 5.0
    volume = _vesicle(profile, voxel_size)
    n = volume.shape[0]
    c = n // 2  # the voxel at the physical origin
    table, d0, step = profile_table(profile, voxel_size)

    r = torch.arange(n - c, dtype=torch.float32) * voxel_size
    index = ((r - _RADIUS - d0) / step).round().long().clamp(0, table.numel() - 1)
    expected = table[index]
    peak = float(table.max())
    for ray in (volume[c, c, c:], volume[c, c:, c], volume[c:, c, c]):
        assert float((ray - expected).abs().max()) < 0.05 * peak
        # The potential-weighted mean radius is the mid-surface, give or take
        # the profile's own asymmetry.
        mean_radius = float((ray * r).sum() / ray.sum())
        assert mean_radius == pytest.approx(_RADIUS, abs=4.0)
    assert float(volume.max()) == pytest.approx(peak, rel=0.05)
    # Empty at the centre and at the box corner, both far from the shell.
    assert float(volume[c, c, c]) == 0.0
    assert float(volume[0, 0, 0]) == 0.0


def test_integral_through_the_membrane_is_the_profiles(
    profile: BilayerProfile,
) -> None:
    voxel_size = 5.0
    volume = _vesicle(profile, voxel_size)
    c = volume.shape[0] // 2
    expected = _profile_integral(profile)
    for ray in (volume[c, c, c:], volume[c, c, :c], volume[c, c:, c], volume[c:, c, c]):
        assert float(ray.sum()) * voxel_size == pytest.approx(expected, rel=0.03)


def test_integrated_potential_per_area_does_not_depend_on_voxel_size(
    profile: BilayerProfile,
) -> None:
    """The whole shell's integrated potential is psi(d) over the spherical
    shells of radius R + d, whatever grid it is rendered on."""
    d = profile.distance_angstrom.cpu().double()
    expected = float(
        torch.trapezoid(
            profile.psi.cpu().double() * 4.0 * math.pi * (_RADIUS + d) ** 2, d
        )
    )
    totals = {v: float(_vesicle(profile, v).double().sum()) * v**3 for v in (5.0, 10.0)}
    assert totals[5.0] == pytest.approx(expected, rel=0.03)
    assert totals[10.0] == pytest.approx(expected, rel=0.03)
    assert totals[10.0] == pytest.approx(totals[5.0], rel=0.03)


def test_blocks_smaller_than_the_output_give_the_same_volume(
    profile: BilayerProfile,
) -> None:
    """The output is evaluated block by block; 50 voxels is one 64-block,
    27 16-blocks with a ragged last one."""
    whole = _vesicle(profile, 8.0, block=64)
    assert whole.shape == (50, 50, 50)
    pieces = _vesicle(profile, 8.0, block=16)
    assert torch.allclose(pieces, whole, atol=1e-4 * float(whole.max()))


def test_surfaces_add_and_origin_moves_the_window(profile: BilayerProfile) -> None:
    grid = Grid((400.0, 400.0, 400.0), lo=(-200.0, -200.0, -200.0))
    outer = ball(grid, (0.0, 0.0, 0.0), 140.0)
    inner = ball(grid, (0.0, 0.0, 0.0), 50.0)
    shape, voxel_size = (40, 40, 40), 10.0
    both = render_bilayer([outer, inner], grid, profile, shape, voxel_size)
    separate = sum(
        render_bilayer([phi], grid, profile, shape, voxel_size)
        for phi in (outer, inner)
    )
    assert isinstance(separate, torch.Tensor)
    assert torch.allclose(both, separate, atol=1e-5)

    # An explicit origin 4 voxels further along x shows the same membrane
    # shifted 4 voxels the other way.
    shifted = render_bilayer(
        [outer], grid, profile, shape, voxel_size, origin_xyz=(-160.0, -200.0, -200.0)
    )
    centred = render_bilayer([outer], grid, profile, shape, voxel_size)
    assert torch.allclose(shifted[:, :, :-4], centred[:, :, 4:], atol=1e-5)
