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
from specter import rotations
from specter.rotations import affine_sampling_grid
import torch.nn.functional as F
from specter.rotations._volume import _relion_rotation_grid


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

    volume = torch.zeros(nz, ny, nx)
    for coord in coords:
        cx = coord[0].item() / voxel_size
        cy = coord[1].item() / voxel_size
        cz = coord[2].item() / voxel_size
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        volume += torch.exp(-r2 / (2 * sigma_voxels**2))

    return volume


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
    volume_A = _gaussian_volume(coords_rot, _GRID, _VOXEL_SIZE)

    # Method B: build Gaussian density map from original coordinates, then rotate.
    # Mirrors imagegenerator.py line ~584: build_affine_matrix(R.as_matrix(), T)
    # then self.rotator(V, theta). rotate_volume with M samples input at
    # (p-center) @ M^T + center, so content moves by M^T; passing R directly
    # moves content by R^T, consistent with Method A.
    volume_orig = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    theta = build_affine_matrix(R_matrix.unsqueeze(0))
    volume_B = rotate_volume(volume_orig, theta).squeeze(0)

    diff = (volume_A - volume_B).abs()

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
    ref_mass = volume_orig.sum().item()
    assert abs(volume_B.sum().item() - ref_mass) / ref_mass < 0.01, (
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

    volume_ref = rotate_volume(
        volume, theta, origin="relion", padding_mode="border", align_corners=True
    )

    rotator = VolumeRotator(
        nz=nz, ny=ny, nx=nx, origin="relion", padding_mode="border", align_corners=True
    )
    slice_indices = torch.arange(nz) - (nz // 2)
    volume_slices = rotator.sample_rotated_slices(
        volume,
        theta,
        slice_indices=slice_indices,
        roi_size=(ny, nx),
        padding_mode="border",
    )

    torch.testing.assert_close(volume_slices, volume_ref, atol=2e-5, rtol=1e-4)


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
    volume_3d = torch.rand(4, 5, 6)
    volume_4d = volume_3d.unsqueeze(0).expand(B, -1, -1, -1)
    volume_5d = volume_4d.unsqueeze(1)

    out_3d = _prepare_volume_for_grid_sample(volume_3d, B)
    out_4d = _prepare_volume_for_grid_sample(volume_4d, B)
    out_5d = _prepare_volume_for_grid_sample(volume_5d, B)

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
    volume = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    shift_angstrom = 4.0
    T = translations_angstrom_to_torch(
        torch.tensor([[shift_angstrom, 0.0]]), _N, _VOXEL_SIZE
    )
    theta = build_affine_matrix(torch.eye(3).unsqueeze(0), T)

    # Translations are subtracted, so a +4 A translation shifts by -4 voxels.
    expected = torch.roll(volume, shifts=-int(shift_angstrom / _VOXEL_SIZE), dims=2)

    rotator = VolumeRotator(_N, _N, _N, mode="fourier")
    out = rotator(volume, theta)[0]

    assert torch.allclose(out, expected, atol=1e-4)
    assert out.min() > -1e-4  # no modulation-induced sign flips


@pytest.mark.parametrize("angle_deg", [90.0, 45.0, 30.0])
def test_fourier_rotation_with_translation_matches_real(angle_deg: float) -> None:
    """
    Both rotation modes must displace the density by the same amount.

    The translation is subtracted and acts in the lab frame, so a (+4, -3) A
    translation moves the density by (-4, +3) voxels whatever the rotation is.
    """
    volume = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    R = _rot_z(angle_deg).unsqueeze(0)
    T = translations_angstrom_to_torch(torch.tensor([[4.0, -3.0]]), _N, _VOXEL_SIZE)

    for mode in ("real", "fourier"):
        rotator = VolumeRotator(_N, _N, _N, mode=mode)
        rotated = rotator(volume, build_affine_matrix(R, torch.zeros(1, 3)))[0]
        shifted = rotator(volume, build_affine_matrix(R, T))[0]
        assert _peak_offset(rotated, shifted) == [-4, 3, 0]


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
    volume = _gaussian_volume(coords, (n, n, n), _VOXEL_SIZE, sigma_voxels=2.0)
    T = translations_angstrom_to_torch(torch.tensor([[2.0, -1.0]]), n, _VOXEL_SIZE)
    theta = build_affine_matrix(_rot_z(180.0).unsqueeze(0), T)

    for origin in ("relion", "center"):
        real = rotate_volume(volume, theta, origin=origin)[0]
        fourier = rotate_volume_fourier(volume, theta, origin=origin)[0]
        rotator = VolumeRotator(n, n, n, origin=origin, mode="fourier")
        assert torch.allclose(real, fourier, atol=1e-3)
        assert torch.allclose(rotator(volume, theta)[0], fourier, atol=1e-5)

    # The correction must actually do something: without it the two origins
    # would be identical, and a 180 degree rotation displaces them by 2 * 0.5 px.
    relion = rotate_volume_fourier(volume, theta, origin="relion")[0]
    center = rotate_volume_fourier(volume, theta, origin="center")[0]
    assert _peak_offset(relion, center) == [-1, -1, 0]


def test_fourier_origin_conventions_coincide_for_odd_volumes() -> None:
    """`n // 2` equals `(n - 1) / 2` for odd n, so the correction must vanish."""
    m = 33
    coords = torch.tensor([[5.0, 0.0, 0.0], [0.0, -4.0, 3.0]])
    volume = _gaussian_volume(coords, (m, m, m), _VOXEL_SIZE)
    T = translations_angstrom_to_torch(torch.tensor([[3.0, -2.0]]), m, _VOXEL_SIZE)
    theta = build_affine_matrix(_rot_z(37.0).unsqueeze(0), T)

    relion = rotate_volume_fourier(volume, theta, origin="relion")
    center = rotate_volume_fourier(volume, theta, origin="center")
    assert torch.equal(relion, center)


def test_fourier_origin_center_rejects_non_cubic() -> None:
    """The origin correction mixes axes, so a non-cubic box must be refused."""
    volume = torch.zeros(16, 16, 32)
    theta = build_affine_matrix(_rot_z(90.0).unsqueeze(0))
    with pytest.raises(ValueError, match="cubic"):
        rotate_volume_fourier(volume, theta, origin="center")


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
        volume = torch.zeros(n, n, n)
        for amp, c in zip(amplitudes, cs):
            volume += amp * torch.exp(-((grid - c) ** 2).sum(-1) / (2 * sigma**2))
        return volume

    volume = build(centres)
    peak = volume.max()

    R = random_rotation_matrix(1)
    shift_angstrom = torch.rand(1, 2) * 6.0 - 3.0
    T = translations_angstrom_to_torch(shift_angstrom, n, _VOXEL_SIZE)
    theta = build_affine_matrix(R.unsqueeze(0), T)

    shift = torch.cat([shift_angstrom[0], torch.zeros(1)])
    truth = build(torch.stack([R.T @ p - shift for p in centres]))

    real = rotate_volume(volume, theta)[0]
    fourier = rotate_volume_fourier(volume, theta)[0]

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
    volume = _gaussian_volume(_COORDS, _GRID, _VOXEL_SIZE)
    peak = volume.max()

    R = random_rotation_matrix(n_poses)  # (n_poses, 3, 3)
    shift_angstrom = torch.rand(n_poses, 2) * 6.0 - 3.0
    theta = build_affine_matrix(
        R, translations_angstrom_to_torch(shift_angstrom, _N, _VOXEL_SIZE)
    )

    # One call transforms the whole batch of maps.
    volumes = rotate_volume(volume, theta)
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


def _relion_rotation_grid_sixpass(
    theta: torch.Tensor, nz: int, ny: int, nx: int, align_corners: bool
) -> torch.Tensor:
    """
    The pre-fusion `_relion_rotation_grid`, kept verbatim as the reference.

    Builds a B-replicated identity grid and then centres/scales/rotates/
    unscales/uncentres/translates it in six separate full-size passes. The
    shipped version composes all of that into one affine matrix; this exists so
    that equivalence is asserted against the original arithmetic rather than
    against the fused version's own restatement of it.
    """
    import torch.nn.functional as F

    B = theta.size(0)
    null_rot = torch.zeros_like(theta)
    null_rot[:, 0, 0] = 1.0
    null_rot[:, 1, 1] = 1.0
    null_rot[:, 2, 2] = 1.0
    grid = F.affine_grid(null_rot, [B, 1, nz, ny, nx], align_corners=align_corners)

    scale = torch.tensor(
        [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2],
        device=theta.device,
        dtype=theta.dtype,
    ).view(1, 1, 3)

    cz, cy, cx = nz // 2, ny // 2, nx // 2
    center = grid[:, cz, cy, cx].unsqueeze(1)
    grid = grid.view(B, nz * ny * nx, 3)
    grid = (grid - center) * scale
    grid = grid.bmm(theta[..., :-1].transpose(1, 2))
    grid = (grid / scale) + center
    grid = grid + theta[..., -1].unsqueeze(1)
    return grid.view(B, nz, ny, nx, 3)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape", [(16, 16, 16), (17, 17, 17), (12, 20, 24), (13, 20, 25)]
)
def test_relion_rotation_grid_matches_six_pass_reference(
    shape: tuple[int, int, int], align_corners: bool
) -> None:
    """
    The fused single-`affine_grid` form reproduces the original six-pass grid.

    Run in float64 so the assertion is about the algebra, not about float32
    rounding: the two orderings are mathematically identical, and any real
    discrepancy (a transposed rotation, a wrong even/odd centre, an axis-order
    slip between (z, y, x) sizes and `affine_grid`'s (x, y, z) output) is many
    orders of magnitude larger than the 1e-15 this leaves.

    Odd and non-cubic shapes are covered because the RELION origin
    `[nz//2, ny//2, nx//2]` only coincides with the geometric centre for odd
    sizes, and the per-axis scale differs once the axes do.
    """
    from specter.rotations._volume import _relion_rotation_grid

    torch.manual_seed(0)
    nz, ny, nx = shape
    theta = build_affine_matrix(random_rotation_matrix(4)).double()
    theta[..., -1] = torch.randn(4, 3, dtype=torch.float64) * 0.05

    expected = _relion_rotation_grid_sixpass(theta, nz, ny, nx, align_corners)
    got = _relion_rotation_grid(theta, nz, ny, nx, align_corners)

    assert got.shape == expected.shape == (4, nz, ny, nx, 3)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0)


def test_rotate_volume_identity_affine_is_a_no_op() -> None:
    """
    A pure identity affine must return the volume unchanged.

    Independent of the six-pass reference: it pins the fused grid's *absolute*
    origin and scale rather than only its agreement with the previous code, so
    a centre offset that both implementations shared would still be caught.
    """
    torch.manual_seed(0)
    volume = torch.rand(16, 20, 24)
    theta = build_affine_matrix(torch.eye(3).unsqueeze(0))

    out = rotate_volume(volume, theta)

    assert out.shape == (1, 16, 20, 24)
    assert torch.allclose(out[0], volume, atol=1e-5)


# ---------------------------------------------------------------------------
# rotation grids, pinned to the reference formulations
# ---------------------------------------------------------------------------


def _random_theta(B: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    R = rotations.random_rotation_matrix(B).to(dtype)
    t = (torch.rand(B, 3, dtype=dtype) - 0.5) * 0.4
    return rotations.build_affine_matrix(R, t)


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


def test_rotate_volume_under_grad_matches_forward_only_and_batched_gradient():
    """
    With the volume requiring grad, `rotate_volume` samples one image at a
    time with a recomputed grid; the values equal the batched forward-only
    call exactly, and the gradient equals the batched kernel's to float
    accumulation order.
    """
    import torch.nn.functional as F

    from specter import rotations
    from specter.rotations._volume import _relion_rotation_grid

    torch.manual_seed(0)
    n = 20
    V = torch.randn(n, n, n)
    R = rotations.random_rotation_matrix(3)
    theta = rotations.build_affine_matrix(R)
    with torch.no_grad():
        plain = rotations.rotate_volume(V, theta)
    Vg = V.clone().requires_grad_(True)
    out = rotations.rotate_volume(Vg, theta)
    assert torch.equal(out, plain)
    (g_new,) = torch.autograd.grad((out**2).sum(), Vg)

    Vb = V.clone().requires_grad_(True)
    grid = _relion_rotation_grid(theta, n, n, n, False)
    ref = F.grid_sample(
        Vb[None, None].expand(3, 1, n, n, n),
        grid,
        align_corners=False,
        padding_mode="border",
    )[:, 0]
    (g_ref,) = torch.autograd.grad((ref**2).sum(), Vb)
    assert torch.allclose(g_new, g_ref, rtol=1e-5, atol=1e-6)


def test_volume_rotator_under_grad_matches_forward_only():
    from specter import rotations
    from specter.rotations import VolumeRotator

    torch.manual_seed(0)
    n = 20
    rot = VolumeRotator(n, n, n, origin="relion", mode="real")
    V = torch.randn(n, n, n)
    theta = rotations.build_affine_matrix(rotations.random_rotation_matrix(3))
    with torch.no_grad():
        plain = rot(V, theta)
    Vg = V.clone().requires_grad_(True)
    out = rot(Vg, theta)
    assert torch.equal(out, plain)
    (g,) = torch.autograd.grad((out**2).sum(), Vg)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
