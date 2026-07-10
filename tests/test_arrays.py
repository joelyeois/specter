import torch

from specter.arrays import (
    soft_voxelize_coordinates,
    soft_voxelize_xy_coordinates,
    tile_volume_from_blocks_blended,
)

# ---------------------------------------------------------------------------
# soft_voxelize_coordinates
# ---------------------------------------------------------------------------


def test_trilinear_conserves_mass_when_in_bounds() -> None:
    torch.manual_seed(0)
    coords = (torch.rand(20, 3) - 0.5) * 5  # well within a (10,10,10) grid
    volume = soft_voxelize_coordinates(coords, (10, 10, 10), 1.0)
    assert torch.allclose(volume.sum(), torch.tensor(20.0), atol=1e-4)


def test_trilinear_batched_matches_looped_unbatched() -> None:
    torch.manual_seed(1)
    coords = (torch.rand(3, 15, 3) - 0.5) * 5
    batched = soft_voxelize_coordinates(coords, (10, 10, 10), 1.0)
    looped = torch.stack(
        [soft_voxelize_coordinates(coords[b], (10, 10, 10), 1.0) for b in range(3)]
    )
    assert torch.allclose(batched, looped, atol=1e-6)


def test_trilinear_unbatched_input_returns_unbatched_output() -> None:
    coords = torch.zeros(5, 3)
    volume = soft_voxelize_coordinates(coords, (8, 8, 8), 1.0)
    assert volume.shape == (8, 8, 8)


def test_trilinear_single_coordinate_splats_to_nearest_8_voxels() -> None:
    # A coordinate exactly at a voxel corner (e.g. half-integer offset from
    # the origin) should distribute weight 0.125 to all 8 surrounding voxels.
    coords = torch.tensor([[0.5, 0.5, 0.5]])  # +0.5 voxel from center in each dim
    volume = soft_voxelize_coordinates(coords, (8, 8, 8), 1.0)
    nonzero = volume[volume > 1e-6]
    assert nonzero.numel() == 8
    assert torch.allclose(nonzero, torch.full_like(nonzero, 0.125), atol=1e-6)


def test_trilinear_out_of_bounds_coordinates_are_dropped() -> None:
    coords = torch.tensor([[100.0, 100.0, 100.0]])
    volume = soft_voxelize_coordinates(coords, (8, 8, 8), 1.0)
    assert torch.allclose(volume.sum(), torch.tensor(0.0))


def test_trilinear_periodic_conserves_mass_for_out_of_bounds() -> None:
    coords = torch.tensor([[100.0, 100.0, 100.0]])
    volume = soft_voxelize_coordinates(coords, (8, 8, 8), 1.0, periodic=True)
    assert torch.allclose(volume.sum(), torch.tensor(1.0), atol=1e-4)


# ---------------------------------------------------------------------------
# soft_voxelize_xy_coordinates
# ---------------------------------------------------------------------------


def test_xy_conserves_mass_when_in_bounds() -> None:
    torch.manual_seed(2)
    coords = (torch.rand(20, 3) - 0.5) * 5
    volume = soft_voxelize_xy_coordinates(coords, (10, 10, 10), 1.0)
    assert torch.allclose(volume.sum(), torch.tensor(20.0), atol=1e-4)


def test_xy_hard_z_assignment_hits_single_z_slice() -> None:
    # All coords share the same integer z, so only one z-slice should
    # receive any weight (Z assignment is nearest-neighbor, not soft).
    coords = torch.tensor([[0.3, 0.3, 2.0], [-0.2, 0.4, 2.0]])
    volume = soft_voxelize_xy_coordinates(coords, (10, 10, 10), 1.0)
    nonzero_z_slices = (volume.sum(dim=(1, 2)) > 1e-6).sum()
    assert nonzero_z_slices == 1


def test_xy_batched_matches_looped_unbatched() -> None:
    torch.manual_seed(3)
    coords = (torch.rand(3, 15, 3) - 0.5) * 5
    batched = soft_voxelize_xy_coordinates(coords, (10, 10, 10), 1.0)
    looped = torch.stack(
        [soft_voxelize_xy_coordinates(coords[b], (10, 10, 10), 1.0) for b in range(3)]
    )
    assert torch.allclose(batched, looped, atol=1e-6)


def test_xy_out_of_bounds_coordinates_are_dropped() -> None:
    coords = torch.tensor([[100.0, 100.0, 0.0]])
    volume = soft_voxelize_xy_coordinates(coords, (8, 8, 8), 1.0)
    assert torch.allclose(volume.sum(), torch.tensor(0.0))


# ---------------------------------------------------------------------------
# tile_volume_from_blocks_blended
# ---------------------------------------------------------------------------


def test_blended_output_shape_matches_target() -> None:
    torch.manual_seed(0)
    blocks = torch.rand(4, 8, 8, 8)
    out = tile_volume_from_blocks_blended(blocks, (2, 20, 20, 20))
    assert out.shape == (2, 20, 20, 20)


def test_blended_conserves_sum() -> None:
    torch.manual_seed(1)
    blocks = torch.rand(4, 8, 8, 8)
    out = tile_volume_from_blocks_blended(blocks, (1, 24, 24, 24), conserve_sum=True)
    expected = blocks.mean() * 24 * 24 * 24
    assert torch.allclose(out.sum(), expected, atol=1e-3)


def test_blended_without_conserve_sum_is_still_finite() -> None:
    torch.manual_seed(2)
    blocks = torch.rand(4, 8, 8, 8)
    out = tile_volume_from_blocks_blended(blocks, (1, 20, 20, 20), conserve_sum=False)
    assert torch.isfinite(out).all()


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
