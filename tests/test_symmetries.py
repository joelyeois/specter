import pytest
import roma
import torch

from specter.rotations import rotate_volume
from specter.symmetries import apply_symmetry, get_rotation_matrices, symmetrize_exact

# ---------------------------------------------------------------------------
# get_rotation_matrices
# ---------------------------------------------------------------------------

_GROUP_SIZES = [
    ("C1", 1),
    ("C2", 2),
    ("C4", 4),
    ("D2", 4),
    ("D3", 6),
    ("T", 12),
    ("O", 24),
    ("I", 60),
    ("I1", 60),
    ("I2", 60),
]


@pytest.mark.parametrize("sym,n_ops", _GROUP_SIZES)
def test_group_size_and_shape(sym: str, n_ops: int) -> None:
    matrices = get_rotation_matrices(sym, return_affine=False)
    assert matrices.shape == (n_ops, 3, 3)

    affine = get_rotation_matrices(sym, return_affine=True)
    assert affine.shape == (n_ops, 3, 4)
    assert torch.equal(affine[:, :, :3], matrices)
    assert torch.equal(affine[:, :, 3], torch.zeros(n_ops, 3))


@pytest.mark.parametrize("sym,_", _GROUP_SIZES)
def test_matrices_are_orthonormal(sym: str, _: int) -> None:
    matrices = get_rotation_matrices(sym, return_affine=False)
    identity = torch.eye(3).expand_as(matrices)
    assert torch.allclose(matrices @ matrices.transpose(-1, -2), identity, atol=1e-6)
    det = torch.linalg.det(matrices)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6)


def _is_closed_under_composition(matrices: torch.Tensor, atol: float = 1e-6) -> bool:
    """
    Check that composing any two operations in `matrices` yields another
    operation already present in the set.

    This is the defining property of a point group: the matrices are not
    just N arbitrary orthonormal matrices, they are the specific set that
    maps the symmetric object onto itself, so composing any two of them
    must itself be a member of the set.
    """
    products = torch.einsum("aij,bjk->abik", matrices, matrices)  # (n, n, 3, 3)
    diffs = (products.unsqueeze(2) - matrices).abs().amax(dim=(-1, -2))  # (n, n, n)
    closest_match = diffs.amin(dim=-1)  # (n, n)
    return bool((closest_match < atol).all())


@pytest.mark.parametrize("sym,_", _GROUP_SIZES)
def test_group_closed_under_composition(sym: str, _: int) -> None:
    matrices = get_rotation_matrices(sym, return_affine=False)
    assert _is_closed_under_composition(matrices)


def test_arbitrary_orthonormal_matrices_are_not_closed() -> None:
    """
    Sanity-check that closure is actually discriminating: a handful of
    random rotation matrices (not chosen to form a group) should fail it,
    so the test above isn't vacuously true.
    """
    torch.manual_seed(0)
    random_axes = torch.randn(5, 3)
    random_angles = torch.rand(5) * 2 * torch.pi
    matrices = torch.stack(
        [
            roma.rotvec_to_rotmat(angle * axis / axis.norm())
            for axis, angle in zip(random_axes, random_angles)
        ]
    )
    assert not _is_closed_under_composition(matrices)


def test_lowercase_symmetry_label_is_accepted() -> None:
    upper = get_rotation_matrices("C3", return_affine=False)
    lower = get_rotation_matrices("c3", return_affine=False)
    assert torch.equal(upper, lower)


def test_c4_matches_known_90_degree_rotation() -> None:
    matrices = get_rotation_matrices("C4", return_affine=False)
    expected_90deg = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(matrices[1], expected_90deg, atol=1e-6)


def test_d2_includes_180_degree_y_flip() -> None:
    matrices = get_rotation_matrices("D2", return_affine=False)
    expected_y_flip = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    assert torch.allclose(matrices[1], expected_y_flip, atol=1e-6)


def test_unknown_symmetry_raises() -> None:
    with pytest.raises(ValueError, match="Unknown symmetry"):
        get_rotation_matrices("X5")


# ---------------------------------------------------------------------------
# apply_symmetry
# ---------------------------------------------------------------------------


def test_apply_symmetry_c1_is_identity() -> None:
    volume = torch.rand(8, 8, 8)
    volume_sym = apply_symmetry(volume, "C1", method="real")
    assert torch.allclose(volume_sym, volume, atol=1e-5)


def test_apply_symmetry_batchsize_matches_unbatched() -> None:
    volume = torch.rand(8, 8, 8)
    unbatched = apply_symmetry(volume, "C4", batchsize=None, method="real")
    batched = apply_symmetry(volume, "C4", batchsize=2, method="real")
    assert torch.allclose(unbatched, batched, atol=1e-5)


# ---------------------------------------------------------------------------
# symmetrize_exact
# ---------------------------------------------------------------------------


def _compact_object(n: int, seed: int = 0) -> torch.Tensor:
    """Smooth Gaussian blobs inside the inscribed sphere: a band-limited particle."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(40, 3, generator=g, dtype=torch.float64)
    radii = torch.rand(40, 1, generator=g, dtype=torch.float64) ** (1 / 3) * 0.3 * n
    centres = centres / centres.norm(dim=1, keepdim=True) * radii
    z, y, x = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float64) - n // 2] * 3, indexing="ij"
    )
    v = torch.zeros(n, n, n, dtype=torch.float64)
    for cx, cy, cz in centres:
        v += torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 4.5)
    return v.float()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("shape", [(17, 17, 17), (16, 16, 16), (12, 14, 18)])
def test_symmetrize_exact_identity_returns_volume(shape: tuple[int, int, int]) -> None:
    volume = torch.randn(shape)
    assert _rel(symmetrize_exact(volume, torch.eye(3)[None]), volume) < 2e-4


def test_symmetrize_exact_c4_matches_grid_permutation() -> None:
    """A quarter turn about voxel N//2 of an odd box permutes the grid exactly, so
    the average needs no interpolation and must hold even for white noise."""
    volume = torch.randn(17, 17, 17)
    expected = torch.stack(
        [torch.rot90(volume, k, dims=(1, 2)) for k in range(4)]
    ).mean(0)
    assert _rel(symmetrize_exact(volume, get_rotation_matrices("C4")), expected) < 1e-4


@pytest.mark.parametrize("n", [24, 25])
def test_symmetrize_exact_is_idempotent_where_interpolation_is_not(n: int) -> None:
    """Averaging a symmetric map over its group again must change nothing. The
    interpolating modes attenuate it on every pass, which is the loss this
    method exists to remove."""
    group = get_rotation_matrices("I1")
    once = symmetrize_exact(_compact_object(n), group)
    assert _rel(symmetrize_exact(once, group), once) < 1e-4
    assert _rel(apply_symmetry(once, group, method="fourier"), once) > 0.05


def test_symmetrize_exact_follows_rotate_volume_convention() -> None:
    """One operator samples the input at R x, as rotate_volume does, not at R^T x."""
    volume = _compact_object(32)
    rotation = roma.rotvec_to_rotmat(torch.tensor([0.3, -0.7, 0.5]))
    affine = torch.cat([rotation, torch.zeros(3, 1)], 1)[None]
    interpolated = rotate_volume(volume, affine)[0]
    assert _rel(symmetrize_exact(volume, rotation[None]), interpolated) < 0.1
    assert _rel(symmetrize_exact(volume, rotation.T[None]), interpolated) > 0.5


def test_apply_symmetry_defaults_to_exact() -> None:
    volume = _compact_object(16)
    expected = symmetrize_exact(volume, get_rotation_matrices("D3"))
    assert torch.allclose(apply_symmetry(volume, "D3"), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_symmetrize_exact_cuda_matches_cpu() -> None:
    volume = _compact_object(20)
    group = get_rotation_matrices("D7")
    on_gpu = symmetrize_exact(volume.cuda(), group)
    assert on_gpu.device.type == "cuda"
    assert _rel(on_gpu.cpu(), symmetrize_exact(volume, group)) < 1e-4
