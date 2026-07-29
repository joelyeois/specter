import numpy as np
import pytest
import roma
import torch

from specter.rotations import (
    VolumeRotator,
    _build_roi_query_points,
    _normalize_slice_indices,
    _prepare_volume_for_grid_sample,
    _resolve_roi,
    build_affine_matrix,
    rotate_volume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rot_z(angle_deg: float) -> torch.Tensor:
    """3x3 rotation matrix for CCW rotation around Z by angle_deg."""
    rad = float(np.deg2rad(angle_deg))
    c, s = np.cos(rad), np.sin(rad)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _gaussian_volume(
    coords: torch.Tensor,
    grid_shape: tuple[int, int, int],
    voxel_size: float,
    sigma_voxels: float = 3.0,
) -> torch.Tensor:
    """
    Place an isotropic Gaussian blob (sigma in voxels) at each coordinate.

    Parameters
    ----------
    coords : (N, 3) tensor in (x, y, z) Angstrom, origin at volume center.
    grid_shape : (nz, ny, nx)
    voxel_size : Angstrom per voxel.
    sigma_voxels : Gaussian sigma in voxel units.
    """
    nz, ny, nx = grid_shape
    z = torch.arange(nz, dtype=torch.float32) - nz // 2
    y = torch.arange(ny, dtype=torch.float32) - ny // 2
    x = torch.arange(nx, dtype=torch.float32) - nx // 2
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")  # (nz, ny, nx)

    vol = torch.zeros(nz, ny, nx)
    for coord in coords:
        cx = coord[0].item() / voxel_size
        cy = coord[1].item() / voxel_size
        cz = coord[2].item() / voxel_size
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        vol += torch.exp(-r2 / (2 * sigma_voxels**2))

    return vol


# ---------------------------------------------------------------------------
# Physics consistency: coordinate rotation vs volume rotation
# ---------------------------------------------------------------------------

# Asymmetric point cloud, well within a 32^3 grid at voxel_size=1 Å.
_COORDS = torch.tensor(
    [
        [5.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [-4.0, 0.0, 0.0],
        [3.0, -3.0, 0.0],
        [0.0, 0.0, 5.0],
        [2.0, 4.0, -3.0],
    ],
    dtype=torch.float32,
)

_N = 32
_VOXEL_SIZE = 1.0
_GRID = (_N, _N, _N)  # (nz, ny, nx)


@pytest.mark.parametrize("angle_deg", [90.0, 45.0, 30.0])
def test_coordinate_rotation_matches_volume_rotation(angle_deg: float) -> None:
    """
    Physics consistency test: rotating 3D atomic coordinates and then
    voxelizing must produce the same volume as voxelizing first and then
    rotating the volume.

    Uses Gaussian blobs (sigma=2 voxels) instead of delta functions so that
    both interpolation paths (trilinear splatting vs grid_sample) agree
    closely. This makes the test sensitive to convention errors (wrong axis,
    wrong sign, wrong origin) while being robust to interpolation asymmetry.

    Convention validated (matches imagegenerator.py):
      - Applying the inverse rotation R^T to coordinates (as
        ImageGenerator.rotate() does via roma.unitquat_to_rotmat(...).transpose)
        matches rotate_volume with build_affine_matrix(R), which also moves
        content by R^T. Both paths apply the same transformation, so the two
        volumes must agree.
    """
    R_matrix = _rot_z(angle_deg)
    R_inv = roma.rotmat_inverse(R_matrix)

    # Method A: rotate atomic coordinates, then build Gaussian density map.
    # Mirrors imagegenerator.py's rotate(): vectors @ R.T with R already the
    # inverse rotation matrix.
    coords_rot = _COORDS @ R_inv.T
    vol_A = _gaussian_volume(coords_rot, _GRID, _VOXEL_SIZE)

    # Method B: build Gaussian density map from original coordinates, then rotate.
    # Mirrors imagegenerator.py line ~584: build_affine_matrix(R.as_matrix(), T)
    # then self.rotator(V, theta). rotate_volume with M samples input at
    # (p-center) @ M^T + center, so content moves by M^T; passing R directly
    # moves content by R^T, consistent with Method A.
    vol_orig = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    theta = build_affine_matrix(R_matrix.unsqueeze(0))
    vol_B = rotate_volume(vol_orig, theta).squeeze(0)

    diff = (vol_A - vol_B).abs()

    # With sigma=3 Gaussians, interpolation asymmetry produces at most ~3% peak
    # error on a single voxel; the volume-mean error is <0.1% (errors cancel).
    # A convention error (wrong R) gives diffs of order 0.5-1.0, so these
    # thresholds are tight enough to catch any real mistake.
    assert (
        diff.max().item() < 0.03
    ), f"angle={angle_deg}°: max voxel diff {diff.max():.4f}"
    assert (
        diff.mean().item() < 1e-3
    ), f"angle={angle_deg}°: mean voxel diff {diff.mean():.6f}"

    # Total signal must be conserved under rotation.
    ref_mass = vol_orig.sum().item()
    assert (
        abs(vol_B.sum().item() - ref_mass) / ref_mass < 0.01
    ), "rotate_volume did not conserve total mass"


# ---------------------------------------------------------------------------
# Geometry consistency: rotate_volume vs VolumeRotator.sample_rotated_slices
# ---------------------------------------------------------------------------


def _rot_x(angle_deg: float) -> torch.Tensor:
    """3x3 rotation matrix for rotation around X by angle_deg."""
    rad = float(np.deg2rad(angle_deg))
    c, s = np.cos(rad), np.sin(rad)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32
    )


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, -30.0, 47.5])
def test_sample_rotated_slices_matches_rotate_volume(angle_deg: float) -> None:
    """
    rotate_volume and VolumeRotator.sample_rotated_slices must produce identical
    voxel values for the same rotation. Odd dimensions keep slice indices centred
    on integer locations; align_corners=True is required for the tight tolerance.
    """
    nz, ny, nx = 33, 35, 37
    torch.manual_seed(0)
    volume = torch.randn(nz, ny, nx)

    R = _rot_x(angle_deg).unsqueeze(0)
    theta = build_affine_matrix(R)

    vol_ref = rotate_volume(
        volume, theta, origin="relion", padding_mode="border", align_corners=True
    )

    rotator = VolumeRotator(
        nz=nz, ny=ny, nx=nx, origin="relion", padding_mode="border", align_corners=True
    )
    slice_indices = torch.arange(nz) - (nz // 2)
    vol_slices = rotator.sample_rotated_slices(
        volume,
        theta,
        slice_indices=slice_indices,
        roi_size=(ny, nx),
        padding_mode="border",
    )

    torch.testing.assert_close(vol_slices, vol_ref, atol=2e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# sample_rotated_slices helper functions
# ---------------------------------------------------------------------------


def test_normalize_slice_indices_accepts_int_list_and_tensor() -> None:
    assert torch.equal(
        _normalize_slice_indices(0, "cpu"), torch.as_tensor(0).unsqueeze(0)
    )
    assert torch.equal(_normalize_slice_indices([-1, 1], "cpu"), torch.tensor([-1, 1]))
    assert torch.equal(
        _normalize_slice_indices(torch.tensor([2, 3]), "cpu"), torch.tensor([2, 3])
    )


def test_resolve_roi_defaults_to_full_centered_volume() -> None:
    roi_center, roi_size = _resolve_roi(None, None, ny=10, nx=20)
    assert roi_center == (5, 10)
    assert roi_size == (10, 20)


def test_resolve_roi_preserves_explicit_values() -> None:
    roi_center, roi_size = _resolve_roi((3, 4), (6, 8), ny=10, nx=20)
    assert roi_center == (3, 4)
    assert roi_size == (6, 8)


def test_build_roi_query_points_center_pixel_is_zero() -> None:
    slice_indices = torch.tensor([0])
    points = _build_roi_query_points(
        slice_indices,
        roi_center=(5, 5),
        roi_size=(11, 11),
        ny=10,
        nx=10,
        device="cpu",
        dtype=torch.float32,
    )
    # roi is centered on (5, 5) == volume center, so its middle pixel (index
    # 5, 5) must be the (x, y, z) = (0, 0, 0) query point.
    assert torch.equal(points[0, 5, 5], torch.tensor([0.0, 0.0, 0.0]))


def test_prepare_volume_for_grid_sample_normalizes_all_ndim_variants() -> None:
    B = 3
    vol_3d = torch.rand(4, 5, 6)
    vol_4d = vol_3d.unsqueeze(0).expand(B, -1, -1, -1)
    vol_5d = vol_4d.unsqueeze(1)

    out_3d = _prepare_volume_for_grid_sample(vol_3d, B)
    out_4d = _prepare_volume_for_grid_sample(vol_4d, B)
    out_5d = _prepare_volume_for_grid_sample(vol_5d, B)

    assert out_3d.shape == out_4d.shape == out_5d.shape == (B, 1, 4, 5, 6)
    assert torch.equal(out_3d, out_4d)
    assert torch.equal(out_4d, out_5d)
