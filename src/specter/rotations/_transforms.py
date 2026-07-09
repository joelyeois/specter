from __future__ import annotations

from typing import Literal

import torch

from ._rotation import Rotation


def rotate_coordinates(vectors: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors by quaternions.

    Parameters
    ----------
    vectors : torch.Tensor
        (N, 3) or (3,) coordinates.
    quat : torch.Tensor
        (B, 4) or (4,) quaternions.

    Returns
    -------
    rotated : torch.Tensor
        Rotated coordinates.
    """
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    R = Rotation.from_quat(quat)
    rotated = R.apply(vectors, inverse=False)
    if rotated.ndim == 3 and rotated.shape[1] == 1:
        rotated = rotated.squeeze(1)
    return rotated


def random_quaternion(
    batchsize: int = 1,
    convention: Literal["xyzw", "wxyz"] = "xyzw",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Generate uniformly random unit quaternions using Shoemake's method.

    Parameters
    ----------
    batchsize : int, optional
        Number of quaternions to generate. Default 1.
    convention : str, optional
        Quaternion convention ('xyzw' or 'wxyz'). Default 'xyzw'.
    device : str or torch.device, optional
        Device for output tensor. Default 'cpu'.

    Returns
    -------
    quats : torch.Tensor
        Random unit quaternions. Shape (batchsize, 4) or (4,) if batchsize=1.
    """

    u1, u2, u3 = torch.rand(3, batchsize, device=device)

    sqrt_u1 = torch.sqrt(u1)
    sqrt_1u1 = torch.sqrt(1 - u1)

    q_w = sqrt_1u1 * torch.sin(2 * torch.pi * u2)
    q_x = sqrt_1u1 * torch.cos(2 * torch.pi * u2)
    q_y = sqrt_u1 * torch.sin(2 * torch.pi * u3)
    q_z = sqrt_u1 * torch.cos(2 * torch.pi * u3)

    if convention == "xyzw":
        quats = torch.stack([q_x, q_y, q_z, q_w], dim=-1)
    elif convention == "wxyz":
        quats = torch.stack([q_w, q_x, q_y, q_z], dim=-1)
    else:
        raise ValueError("convention must be 'xyzw' or 'wxyz'")

    if batchsize == 1:
        return quats.squeeze(0)
    return quats


def random_rotvec(
    batchsize: int = 1, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Generate uniformly random rotation vectors using the Rotation3D class.

    Parameters
    ----------
    batchsize : int
        Number of rotation vectors to generate.
    device : str or torch.device
        Device for the output tensor.

    Returns
    -------
    rotvecs : torch.Tensor
        Tensor of shape (batchsize, 3) with rotation vectors (axis * angle).
    """
    quats = random_quaternion(batchsize=batchsize, convention="xyzw", device=device)
    R = Rotation.from_quat(quats)
    rotvecs = R.as_rotvec()

    if batchsize == 1:
        return rotvecs.squeeze(0)
    return rotvecs


def random_rotation_matrix(
    batchsize: int = 1, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Generate uniformly random 3x3 rotation matrices using the Rotation3D class.

    Parameters
    ----------
    batchsize : int
        Number of rotation matrices to generate.
    device : str or torch.device
        Device for the output tensor.

    Returns
    -------
    rotmats : torch.Tensor
        Tensor of shape (batchsize, 3, 3) containing rotation matrices.
    """
    quats = random_quaternion(batchsize=batchsize, convention="xyzw", device=device)
    R = Rotation.from_quat(quats)
    rotmats = R.as_matrix()

    if batchsize == 1:
        return rotmats.squeeze(0)
    return rotmats


def rotations_angular_difference(
    r1: torch.Tensor,
    r2: torch.Tensor,
    rotation_representation: Literal["quaternion", "rotvec"] = "rotvec",
) -> torch.Tensor:
    """
    Compute the smallest angular difference between two batches of rotations.

    Parameters
    ----------
    r1 : torch.Tensor
        Batch of rotation representations, shape (N, ...).
    r2 : torch.Tensor
        Batch of rotation representations, shape (N, ...).
    rotation_representation : str, optional
        Input representation: 'quaternion' or 'rotvec'. Default is 'rotvec'.

    Returns
    -------
    angles : torch.Tensor
        Smallest angular difference in degrees, shape (N,).

    References
    ----------
    https://math.stackexchange.com/a/4001635
    """
    if rotation_representation == "rotvec":
        r1 = Rotation.from_rotvec(r1)
        r2 = Rotation.from_rotvec(r2)
    elif rotation_representation == "quaternion":
        r1 = Rotation.from_quat(r1)
        r2 = Rotation.from_quat(r2)
    else:
        raise ValueError(
            f"Unknown rotation_representation '{rotation_representation}'. Must be 'quaternion' or 'rotvec'."
        )

    re_m = r1.inv().as_matrix() @ r2.as_matrix()
    trace = torch.diagonal(re_m, dim1=-2, dim2=-1).sum(-1)
    angles_rad = torch.arccos(((trace - 1) / 2).clamp(-1.0, 1.0))
    return angles_rad / torch.pi * 180
