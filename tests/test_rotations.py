import numpy as np
import pytest
import torch

from cryosim.rotations import (
    Rotation,
    build_affine_matrix,
    translations_angstrom_to_torch,
)


def test_rotation_from_quat_identity():
    # Identity quaternion (xyzw)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    R = Rotation.from_quat(quat)
    matrix = R.as_matrix()
    expected = torch.eye(3, dtype=torch.float32)
    torch.testing.assert_close(matrix.squeeze(), expected)


def test_rotation_apply():
    # 90 degree rotation around Z
    # q = [0, 0, sin(45), cos(45)]
    s45 = float(np.sin(np.pi / 4))
    c45 = float(np.cos(np.pi / 4))
    quat = torch.tensor([0.0, 0.0, s45, c45], dtype=torch.float32)
    R = Rotation.from_quat(quat)

    vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    rotated = R.apply(vec, inverse=False)

    # [1, 0, 0] rotated 90 deg around Z is [0, 1, 0]
    expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(rotated, expected, atol=1e-6, rtol=1e-6)


def test_translations_angstrom_to_torch():
    T = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    n = 100
    voxel_size = 1.0
    # Formula: T * 2 / n / voxel_size
    T_norm = translations_angstrom_to_torch(T, n, voxel_size)
    expected = torch.tensor([[0.2, 0.4, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(T_norm, expected)


def test_build_affine_matrix_identity():
    R = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    T = torch.zeros(1, 3, dtype=torch.float32)
    theta = build_affine_matrix(R, T)

    expected = torch.zeros(1, 3, 4, dtype=torch.float32)
    expected[0, :3, :3] = torch.eye(3, dtype=torch.float32)
    torch.testing.assert_close(theta, expected)


def test_rotate_volume_ball():
    from cryosim.rotations import rotate_volume

    # Construct a 3D ball in a 32x32x32 volume
    nz, ny, nx = 32, 32, 32
    z = torch.linspace(-1, 1, nz)
    y = torch.linspace(-1, 1, ny)
    x = torch.linspace(-1, 1, nx)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    dist = torch.sqrt(zz**2 + yy**2 + xx**2)
    vol = (dist < 0.5).float()

    # Rotate by 45 degrees around Y
    angle = torch.tensor([45.0])
    rad = torch.deg2rad(angle)
    c, s = torch.cos(rad), torch.sin(rad)
    R = torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32
    ).unsqueeze(0)
    theta = build_affine_matrix(R)

    vol_rotated = rotate_volume(vol, theta).squeeze(0)

    # Verify symmetry (mostly) - interpolation at edges will cause small diffs
    # Using larger tolerance due to nearest/linear interpolation on a binary ball
    diff = torch.abs(vol_rotated - vol)
    assert diff.mean() < 0.02
    assert (vol_rotated > 0.5).sum() == pytest.approx(vol.sum().item(), rel=0.05)
