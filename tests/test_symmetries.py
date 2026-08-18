import pytest
import roma
import torch

from specter.symmetries import apply_symmetry, get_rotation_matrices

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
    assert torch.allclose(matrices @ matrices.transpose(-1, -2), identity, atol=1e-4)
    det = torch.linalg.det(matrices)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-4)


def _is_closed_under_composition(matrices: torch.Tensor, atol: float = 1e-4) -> bool:
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
