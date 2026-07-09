import torch

from specter.arrays import soft_voxelize_coordinates, soft_voxelize_xy_coordinates

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
