import pytest
import torch

from specter.arrays import (
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
