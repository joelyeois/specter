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
    random_rotation_matrix,
    rotate_volume,
    rotate_volume_fourier,
    split_affine_translation,
    translations_angstrom_to_torch,
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
    assert diff.max().item() < 0.03, (
        f"angle={angle_deg}°: max voxel diff {diff.max():.4f}"
    )
    assert diff.mean().item() < 1e-3, (
        f"angle={angle_deg}°: mean voxel diff {diff.mean():.6f}"
    )

    # Total signal must be conserved under rotation.
    ref_mass = vol_orig.sum().item()
    assert abs(vol_B.sum().item() - ref_mass) / ref_mass < 0.01, (
        "rotate_volume did not conserve total mass"
    )


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


# ---------------------------------------------------------------------------
# Fourier-space rotation: translations must be phase ramps, not resampling
# ---------------------------------------------------------------------------


def _peak_offset(a: torch.Tensor, b: torch.Tensor) -> list[int]:
    """
    Integer (x, y, z) voxel offset of `b` relative to `a`.

    Located by the peak of the circular cross-correlation, which measures where
    the density actually moved. A whole-box centroid would not: the Fourier path
    shifts circularly while the real-space path pads at the border, so the two
    treat wrapped background differently even when the density agrees.
    """
    corr = torch.fft.ifftn(torch.fft.fftn(b) * torch.fft.fftn(a).conj()).real
    peak = torch.unravel_index(corr.argmax(), corr.shape)  # (z, y, x)
    offset = [
        int(i) if int(i) <= n // 2 else int(i) - n for i, n in zip(peak, corr.shape)
    ]
    return [offset[2], offset[1], offset[0]]  # -> (x, y, z)


def test_fourier_translation_is_exact() -> None:
    """
    A pure translation must move the density, and must do so exactly.

    Applying the affine's translation column by resampling the spectrum
    modulates the volume instead of shifting it, leaving the density in place
    and driving it negative. The phase ramp is exact for an integer-voxel
    shift, so `torch.roll` is ground truth here.
    """
    vol = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    shift_angstrom = 4.0
    T = translations_angstrom_to_torch(
        torch.tensor([[shift_angstrom, 0.0]]), _N, _VOXEL_SIZE
    )
    theta = build_affine_matrix(torch.eye(3).unsqueeze(0), T)

    # Translations are subtracted, so a +4 A translation shifts by -4 voxels.
    expected = torch.roll(vol, shifts=-int(shift_angstrom / _VOXEL_SIZE), dims=2)

    rotator = VolumeRotator(_N, _N, _N, mode="fourier")
    out = rotator(vol, theta)[0]

    assert torch.allclose(out, expected, atol=1e-4)
    assert out.min() > -1e-4  # no modulation-induced sign flips


@pytest.mark.parametrize("angle_deg", [90.0, 45.0, 30.0])
def test_fourier_rotation_with_translation_matches_real(angle_deg: float) -> None:
    """
    Both rotation modes must displace the density by the same amount.

    The translation is subtracted and acts in the lab frame, so a (+4, -3) A
    translation moves the density by (-4, +3) voxels whatever the rotation is.
    """
    vol = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    R = _rot_z(angle_deg).unsqueeze(0)
    T = translations_angstrom_to_torch(torch.tensor([[4.0, -3.0]]), _N, _VOXEL_SIZE)

    for mode in ("real", "fourier"):
        rotator = VolumeRotator(_N, _N, _N, mode=mode)
        rotated = rotator(vol, build_affine_matrix(R, torch.zeros(1, 3)))[0]
        shifted = rotator(vol, build_affine_matrix(R, T))[0]
        assert _peak_offset(rotated, shifted) == [-4, 3, 0]


def test_rotate_volume_fourier_applies_translation() -> None:
    """The standalone function shares the phase-ramp path with VolumeRotator."""
    vol = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    T = translations_angstrom_to_torch(torch.tensor([[4.0, 0.0]]), _N, _VOXEL_SIZE)
    theta = build_affine_matrix(_rot_z(90.0).unsqueeze(0), T)

    out = rotate_volume_fourier(vol, theta)[0]
    reference = VolumeRotator(_N, _N, _N, mode="fourier")(vol, theta)[0]

    assert torch.allclose(out, reference, atol=1e-5)


def test_split_affine_translation_leaves_rotation_untouched() -> None:
    """Splitting must zero the translation column and preserve the rotation."""
    R = _rot_z(37.0).unsqueeze(0)
    T = translations_angstrom_to_torch(torch.tensor([[4.0, -3.0]]), _N, _VOXEL_SIZE)
    theta = build_affine_matrix(R, T)

    theta_rot, displacement = split_affine_translation(theta)

    assert torch.equal(theta_rot[..., :3], theta[..., :3])
    assert torch.all(theta_rot[..., 3] == 0)
    # displacement is -R^-1 T', and build_affine_matrix set T' = R T, so it is -T
    assert torch.allclose(displacement, -T, atol=1e-6)


# ---------------------------------------------------------------------------
# Fourier-space rotation: origin conventions
# ---------------------------------------------------------------------------


def test_fourier_origin_center_matches_real() -> None:
    """
    `origin="center"` must reproduce the real-space result.

    A 180 degree rotation is exact under both interpolation schemes, so any
    discrepancy here is the half-voxel origin offset rather than interpolation.
    The volume must be compact and well inside a roomy box: the Fourier path
    shifts circularly while the real path pads, and density reaching the edge
    would swamp the half-voxel effect under test.
    """
    n = 64
    coords = torch.tensor([[8.0, 0.0, 0.0], [-3.0, 5.0, 4.0], [0.0, -6.0, -4.0]])
    vol = _gaussian_volume(coords, (n, n, n), _VOXEL_SIZE, sigma_voxels=2.0)
    T = translations_angstrom_to_torch(torch.tensor([[2.0, -1.0]]), n, _VOXEL_SIZE)
    theta = build_affine_matrix(_rot_z(180.0).unsqueeze(0), T)

    for origin in ("relion", "center"):
        real = rotate_volume(vol, theta, origin=origin)[0]
        fourier = rotate_volume_fourier(vol, theta, origin=origin)[0]
        rotator = VolumeRotator(n, n, n, origin=origin, mode="fourier")
        assert torch.allclose(real, fourier, atol=1e-3)
        assert torch.allclose(rotator(vol, theta)[0], fourier, atol=1e-5)

    # The correction must actually do something: without it the two origins
    # would be identical, and a 180 degree rotation displaces them by 2 * 0.5 px.
    relion = rotate_volume_fourier(vol, theta, origin="relion")[0]
    center = rotate_volume_fourier(vol, theta, origin="center")[0]
    assert _peak_offset(relion, center) == [-1, -1, 0]


def test_fourier_origin_conventions_coincide_for_odd_volumes() -> None:
    """`n // 2` equals `(n - 1) / 2` for odd n, so the correction must vanish."""
    m = 33
    coords = torch.tensor([[5.0, 0.0, 0.0], [0.0, -4.0, 3.0]])
    vol = _gaussian_volume(coords, (m, m, m), _VOXEL_SIZE)
    T = translations_angstrom_to_torch(torch.tensor([[3.0, -2.0]]), m, _VOXEL_SIZE)
    theta = build_affine_matrix(_rot_z(37.0).unsqueeze(0), T)

    relion = rotate_volume_fourier(vol, theta, origin="relion")
    center = rotate_volume_fourier(vol, theta, origin="center")
    assert torch.equal(relion, center)


def test_fourier_origin_center_rejects_non_cubic() -> None:
    """The origin correction mixes axes, so a non-cubic box must be refused."""
    vol = torch.zeros(16, 16, 32)
    theta = build_affine_matrix(_rot_z(90.0).unsqueeze(0))
    with pytest.raises(ValueError, match="cubic"):
        rotate_volume_fourier(vol, theta, origin="center")


# ---------------------------------------------------------------------------
# Real vs Fourier under a combined 3D rotation and translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_plus_translation_matches_analytic_truth(seed: int) -> None:
    """
    Both rotation modes must reproduce an analytically known transform.

    Isotropic Gaussians are unchanged in shape by a rotation, so transforming
    their centres gives an exact reference volume: a feature at `p` moves to
    `R^-1 p - T`. Both modes are compared against that rather than only against
    each other, which would pass even if they shared a systematic error.
    """
    torch.manual_seed(seed)
    n = 48
    sigma = 2.5
    z = torch.arange(n, dtype=torch.float32) - n // 2
    grid = torch.stack(torch.meshgrid(z, z, z, indexing="ij")[::-1], dim=-1)

    amplitudes = torch.tensor([1.0, 0.6, 0.8])
    centres = torch.tensor([[6.0, 0.0, 0.0], [-2.4, 4.2, 3.0], [0.0, -4.8, -3.6]])

    def build(cs: torch.Tensor) -> torch.Tensor:
        vol = torch.zeros(n, n, n)
        for amp, c in zip(amplitudes, cs):
            vol += amp * torch.exp(-((grid - c) ** 2).sum(-1) / (2 * sigma**2))
        return vol

    vol = build(centres)
    peak = vol.max()

    R = random_rotation_matrix(1)
    shift_angstrom = torch.rand(1, 2) * 6.0 - 3.0
    T = translations_angstrom_to_torch(shift_angstrom, n, _VOXEL_SIZE)
    theta = build_affine_matrix(R.unsqueeze(0), T)

    shift = torch.cat([shift_angstrom[0], torch.zeros(1)])
    truth = build(torch.stack([R.T @ p - shift for p in centres]))

    real = rotate_volume(vol, theta)[0]
    fourier = rotate_volume_fourier(vol, theta)[0]

    # Interpolation error, measured at ~4% (real) and ~3% (Fourier) of peak for
    # this object; see the Fourier mode's accuracy note in `apply_fourier_
    # translation` for why the Fourier figure grows with distance from centre.
    assert (real - truth).abs().max() < 0.10 * peak
    assert (fourier - truth).abs().max() < 0.10 * peak
    assert (real - fourier).abs().max() < 0.15 * peak


# ---------------------------------------------------------------------------
# Coordinates vs map, under a batch of general 3D rotations AND translations
# ---------------------------------------------------------------------------


def test_coordinate_affine_matches_volume_affine_batched() -> None:
    """
    Transforming coordinates must match transforming the map, for full affines.

    `test_coordinate_rotation_matches_volume_rotation` covers rotation about Z
    with no translation. This extends it to a batch of general 3D rotations each
    paired with its own translation, which is what a real pose batch looks like:
    a wrong axis order, a missing pre-rotation of T, or a units slip in the
    translation all survive the rotation-only, single-axis case.
    """
    torch.manual_seed(3)
    n_poses = 3
    vol = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    peak = vol.max()

    R = random_rotation_matrix(n_poses)  # (n_poses, 3, 3)
    shift_angstrom = torch.rand(n_poses, 2) * 6.0 - 3.0
    theta = build_affine_matrix(
        R, translations_angstrom_to_torch(shift_angstrom, _N, _VOXEL_SIZE)
    )

    # One call transforms the whole batch of maps.
    volumes = rotate_volume(vol, theta)
    assert volumes.shape == (n_poses, _N, _N, _N)

    for i in range(n_poses):
        # Coordinates take the same transform: p -> R^-1 p - T, with T in A.
        shift = torch.cat([shift_angstrom[i], torch.zeros(1)])
        coords = _COORDS @ R[i] - shift
        from_coords = _gaussian_volume(coords, _GRID, _VOXEL_SIZE)

        diff = (from_coords - volumes[i]).abs()
        # Interpolation asymmetry between splatting and grid_sample is ~2% of
        # peak on isolated voxels; a convention error is of order 50-100%.
        assert diff.max() < 0.05 * peak
        assert diff.mean() < 0.002 * peak
