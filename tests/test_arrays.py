import pytest
import torch

from specter.arrays import (
    rfft_frequency_squared,
    center_crop,
    fourier_crop,
    radial_profile_2d,
    radial_profile_3d,
    soft_voxelize_coordinates,
    soft_voxelize_xy_coordinates,
    tile_volume_from_blocks_blended,
)

# ---------------------------------------------------------------------------
# soft_voxelize_coordinates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("voxelize", "seed"),
    [(soft_voxelize_coordinates, 0), (soft_voxelize_xy_coordinates, 2)],
    ids=["trilinear", "xy"],
)
def test_voxelize_conserves_mass_when_in_bounds(voxelize, seed: int) -> None:
    torch.manual_seed(seed)
    coords = (torch.rand(20, 3) - 0.5) * 5  # well within a (10,10,10) grid
    volume = voxelize(coords, (10, 10, 10), 1.0)
    assert torch.allclose(volume.sum(), torch.tensor(20.0), atol=1e-4)


@pytest.mark.parametrize(
    ("voxelize", "seed"),
    [(soft_voxelize_coordinates, 1), (soft_voxelize_xy_coordinates, 3)],
    ids=["trilinear", "xy"],
)
def test_voxelize_batched_matches_looped_unbatched(voxelize, seed: int) -> None:
    torch.manual_seed(seed)
    coords = (torch.rand(3, 15, 3) - 0.5) * 5
    batched = voxelize(coords, (10, 10, 10), 1.0)
    looped = torch.stack([voxelize(coords[b], (10, 10, 10), 1.0) for b in range(3)])
    assert torch.allclose(batched, looped, atol=1e-6)


def test_trilinear_single_coordinate_splats_to_nearest_8_voxels() -> None:
    # A coordinate exactly at a voxel corner (e.g. half-integer offset from
    # the origin) should distribute weight 0.125 to all 8 surrounding voxels.
    coords = torch.tensor([[0.5, 0.5, 0.5]])  # +0.5 voxel from center in each dim
    volume = soft_voxelize_coordinates(coords, (8, 8, 8), 1.0)
    nonzero = volume[volume > 1e-6]
    assert nonzero.numel() == 8
    assert torch.allclose(nonzero, torch.full_like(nonzero, 0.125), atol=1e-6)


@pytest.mark.parametrize(
    ("voxelize", "coord", "kwargs", "expected_sum", "atol"),
    [
        (soft_voxelize_coordinates, [100.0, 100.0, 100.0], {}, 0.0, 1e-8),
        (soft_voxelize_xy_coordinates, [100.0, 100.0, 0.0], {}, 0.0, 1e-8),
        (
            soft_voxelize_coordinates,
            [100.0, 100.0, 100.0],
            {"periodic": True},
            1.0,
            1e-4,
        ),
    ],
    ids=["trilinear_dropped", "xy_dropped", "trilinear_periodic_conserves_mass"],
)
def test_voxelize_out_of_bounds_coordinates(
    voxelize, coord, kwargs: dict, expected_sum: float, atol: float
) -> None:
    """An out-of-bounds coordinate is dropped, unless the grid is periodic,
    in which case it wraps and its whole mass is kept."""
    coords = torch.tensor([coord])
    volume = voxelize(coords, (8, 8, 8), 1.0, **kwargs)
    assert torch.allclose(volume.sum(), torch.tensor(expected_sum), atol=atol)


# ---------------------------------------------------------------------------
# soft_voxelize_xy_coordinates
# ---------------------------------------------------------------------------


def test_xy_hard_z_assignment_hits_single_z_slice() -> None:
    # All coords share the same integer z, so only one z-slice should
    # receive any weight (Z assignment is nearest-neighbor, not soft).
    coords = torch.tensor([[0.3, 0.3, 2.0], [-0.2, 0.4, 2.0]])
    volume = soft_voxelize_xy_coordinates(coords, (10, 10, 10), 1.0)
    nonzero_z_slices = (volume.sum(dim=(1, 2)) > 1e-6).sum()
    assert nonzero_z_slices == 1


# ---------------------------------------------------------------------------
# tile_volume_from_blocks_blended
# ---------------------------------------------------------------------------


def test_blended_conserves_sum() -> None:
    torch.manual_seed(1)
    blocks = torch.rand(4, 8, 8, 8)
    out = tile_volume_from_blocks_blended(blocks, (1, 24, 24, 24), conserve_sum=True)
    expected = blocks.mean() * 24 * 24 * 24
    assert torch.allclose(out.sum(), expected, atol=1e-3)


def test_blended_without_conserve_sum_preserves_a_constant_field() -> None:
    torch.manual_seed(2)
    blocks = torch.full((4, 8, 8, 8), 3.5)
    out = tile_volume_from_blocks_blended(blocks, (1, 20, 20, 20), conserve_sum=False)
    torch.testing.assert_close(out, torch.full_like(out, 3.5))


def test_blended_without_conserve_sum_stays_within_source_bounds() -> None:
    torch.manual_seed(2)
    blocks = torch.rand(4, 8, 8, 8)
    out = tile_volume_from_blocks_blended(blocks, (1, 20, 20, 20), conserve_sum=False)
    assert torch.isfinite(out).all()
    assert out.min() >= blocks.min() - 1e-6
    assert out.max() <= blocks.max() + 1e-6
    assert out.std() > 0


def test_blended_handles_non_integer_multiple_target() -> None:
    # Target size isn't a multiple of the block size (200 / 32 = 6.25), and
    # is neither square nor a multiple of block size along any axis.
    torch.manual_seed(4)
    blocks = torch.rand(8, 32, 32, 32)
    target = (2, 65, 90, 121)
    out = tile_volume_from_blocks_blended(blocks, target)
    assert out.shape == target
    assert torch.isfinite(out).all()
    expected = blocks.mean() * 65 * 90 * 121 * 2
    assert torch.allclose(out.sum(), expected, atol=1e-1)


def test_blended_tiny_overlap_has_no_zero_weight_artifact() -> None:
    # Regression test: a taper sampled at linspace's inclusive endpoints
    # degenerates to a true-zero weight on both sides of a seam simultaneously
    # when the overlap is a single voxel, producing near-black seam voxels
    # once divided by the (near-zero) weight sum. Bin-centered sampling avoids
    # this.
    torch.manual_seed(3)
    blocks = torch.rand(4, 16, 16, 16) + 1.0  # strictly positive
    out = tile_volume_from_blocks_blended(blocks, (1, 32, 32, 32), overlap_frac=1 / 16)
    assert torch.isfinite(out).all()
    assert out.min() > 0


def test_a_cpu_splat_adds_duplicates_in_a_fixed_order():
    """
    The deterministic splat must not depend on how torch threads it. Through
    ``index_put_(accumulate=True)`` the CPU added duplicate indices across
    threads in scheduling order, so the same coordinates gave a different
    volume on every call, at any thread count above one.
    """
    from specter.arrays._voxelize import _scatter_splat

    g = torch.Generator().manual_seed(0)
    indices = torch.randint(0, 8, (200_000, 3), generator=g)
    weights = torch.rand(200_000, generator=g)
    threads = torch.get_num_threads()
    volumes = []
    try:
        for n in (1, 8, 8):
            torch.set_num_threads(n)
            volume = torch.zeros(8, 8, 8)
            _scatter_splat(volume, indices, weights)
            volumes.append(volume)
        # A non-contiguous target takes the same order.
        torch.set_num_threads(8)
        transposed = torch.zeros(8, 8, 8).transpose(0, 2)
        _scatter_splat(transposed, indices[:, [2, 1, 0]], weights)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(volumes[0], volumes[1])
    assert torch.equal(volumes[1], volumes[2])
    assert torch.equal(transposed.transpose(0, 2), volumes[0])


@pytest.mark.parametrize("length", [9, 10])
@pytest.mark.parametrize("size", [3, 4])
def test_center_crop_keeps_the_fftshift_centre(length: int, size: int) -> None:
    """Index ``L // 2`` lands at ``size // 2`` for every parity pair, including
    an even axis cropped to an odd size, where a ``(L - size) // 2`` start puts
    it one index off. The detector and the inelastic forward model crop with it."""
    x = torch.arange(length * length, dtype=torch.float32).reshape(length, length)
    crop = center_crop(x, size)
    assert crop.shape == (size, size)
    assert crop[size // 2, size // 2] == x[length // 2, length // 2]


@pytest.mark.parametrize(("box", "out"), [(64, 32), (64, 33), (65, 32), (65, 33)])
def test_fourier_crop_resamples_a_band_limited_cosine(box: int, out: int) -> None:
    """A cosine below the new Nyquist comes back as the same cosine on the
    coarse grid, about its ``out // 2`` origin, for every parity pair. An
    even box cropped to an odd one used to shift the crop by one bin (error
    up to 2 on this unit cosine)."""
    pixel_size = 1.0
    cycles = 3
    j = torch.arange(box, dtype=torch.float64) - box // 2
    image = torch.cos(2 * torch.pi * cycles * j / box).expand(box, box)

    cropped, new_pixel_size = fourier_crop(image, pixel_size, pixel_size * box / out)

    assert cropped.shape == (out, out)
    assert new_pixel_size == pytest.approx(pixel_size * box / out)
    k = torch.arange(out, dtype=torch.float64) - out // 2
    expected = torch.cos(2 * torch.pi * cycles * k / out).expand(out, out)
    torch.testing.assert_close(cropped, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("shape", [(8, 9), (7, 16, 9), (6, 6, 6)])
def test_rfft_frequency_squared_is_the_half_of_the_full_spectrum(shape) -> None:
    """|k|^2 on rfftn's half spectrum is the full spectrum's, last axis cut at
    n // 2 + 1, with the absolute frequency of that axis."""
    d = 0.73
    full = sum(
        torch.fft.fftfreq(n, d=d).reshape(
            [-1 if j == i else 1 for j in range(len(shape))]
        )
        ** 2
        for i, n in enumerate(shape)
    )
    half = rfft_frequency_squared(shape, d)
    assert half.shape == (*shape[:-1], shape[-1] // 2 + 1)
    torch.testing.assert_close(half, full[..., : shape[-1] // 2 + 1])


# ---------------------------------------------------------------------------
# radial_profile_2d / radial_profile_3d
# ---------------------------------------------------------------------------


def _naive_radial_profile(data: torch.Tensor, center: tuple) -> torch.Tensor:
    """Bin each element by its rounded distance from `center`, then average."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for idx in torch.cartesian_prod(*(torch.arange(n) for n in data.shape)):
        idx = idx.reshape(-1)
        r = int(
            torch.sqrt(
                sum((float(i) - c) ** 2 for i, c in zip(idx.tolist(), center))
                * torch.ones((), dtype=torch.float32)
            ).round()
        )
        sums[r] = sums.get(r, 0.0) + float(data[tuple(idx.tolist())])
        counts[r] = counts.get(r, 0) + 1
    n = max(sums) + 1
    return torch.tensor(
        [sums.get(r, 0.0) / max(counts.get(r, 0), 1) for r in range(n)],
        dtype=data.dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("center", [None, (1, 4)])
def test_radial_profile_2d_bins_by_rounded_index_distance(dtype, center) -> None:
    # Integer values keep every sum exact, so the comparison can be bitwise.
    data = torch.arange(35, dtype=dtype).reshape(5, 7)
    profile = radial_profile_2d(data, center=center)
    assert profile.dtype == dtype
    expected = _naive_radial_profile(data, center or (5 // 2, 7 // 2))
    assert torch.allclose(profile, expected, rtol=1e-6, atol=0)
    r, again = radial_profile_2d(data, center=center, return_r=True)
    assert torch.equal(again, profile)
    assert torch.equal(r, torch.arange(len(profile)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("center", [None, (0, 1, 3)])
def test_radial_profile_3d_bins_by_rounded_index_distance(dtype, center) -> None:
    data = torch.arange(4 * 5 * 6, dtype=dtype).reshape(4, 5, 6)
    profile = radial_profile_3d(data, center=center)
    assert profile.dtype == dtype
    expected = _naive_radial_profile(data, center or (2, 2, 3))
    assert torch.allclose(profile, expected, rtol=1e-6, atol=0)
    r, again = radial_profile_3d(data, center=center, return_r=True)
    assert torch.equal(again, profile)
    assert torch.equal(r, torch.arange(len(profile)))


def test_radial_profiles_pinned_values() -> None:
    """Exact outputs, so a rewrite of either function must reproduce them."""
    assert torch.equal(
        radial_profile_2d(torch.arange(30.0).reshape(5, 6)),
        torch.tensor([15.0, 15.0, 15.0, 96.0 / 7.0, 12.0]),
    )
    assert torch.equal(
        radial_profile_3d(torch.arange(60.0).reshape(3, 4, 5)),
        torch.tensor([32.0, 32.0, 206.0 / 7.0, 22.0]),
    )


def test_radial_profiles_reject_the_wrong_rank() -> None:
    with pytest.raises(ValueError, match="2D"):
        radial_profile_2d(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="3D"):
        radial_profile_3d(torch.zeros(2, 3))
