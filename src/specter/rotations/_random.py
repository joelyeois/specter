from __future__ import annotations

from typing import Literal

import roma
import torch


def random_quaternion(
    batchsize: int = 1,
    convention: Literal["xyzw", "wxyz"] = "xyzw",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Generate uniformly random unit quaternions.

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
    quats = roma.random_unitquat(size=(batchsize,), device=device)  # xyzw

    if convention == "wxyz":
        quats = roma.quat_xyzw_to_wxyz(quats)
    elif convention != "xyzw":
        raise ValueError("convention must be 'xyzw' or 'wxyz'")

    if batchsize == 1:
        return quats.squeeze(0)
    return quats


def random_rotvec(
    batchsize: int = 1, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Generate uniformly random rotation vectors.

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
    rotvecs = roma.random_rotvec(size=(batchsize,), device=device)

    if batchsize == 1:
        return rotvecs.squeeze(0)
    return rotvecs


def random_rotation_matrix(
    batchsize: int = 1, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Generate uniformly random 3x3 rotation matrices.

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
    rotmats = roma.random_rotmat(size=(batchsize,), device=device)

    if batchsize == 1:
        return rotmats.squeeze(0)
    return rotmats


def random_rotation_matrix_from_generator(
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw one uniformly random 3D rotation matrix from a given generator.

    QR decomposition of a random Gaussian matrix (Mezzadri, 2007), sign- and
    determinant-corrected for a proper rotation (no reflection).

    A second draw beside :func:`random_rotation_matrix` on purpose: the
    roma-based one takes no generator, and roma's ``random_rotmat``/``random_unitquat`` only forwards
    the given ``generator`` to one of its three underlying random draws (the
    other two always fall back to the global RNG -- confirmed empirically,
    not documented), so it cannot give the fully generator-reproducible
    output :meth:`IceBank._extract_crop` needs. This QR-based path's only
    random draw is the single ``torch.randn(..., generator=generator)``
    below, so it is.

    Parameters
    ----------
    generator : torch.Generator, optional
        RNG to draw from. Default is the global RNG.

    Returns
    -------
    torch.Tensor
        Shape (3, 3), ``det(R) == 1``.
    """
    A = torch.randn(3, 3, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


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
    """
    if rotation_representation == "rotvec":
        R1 = roma.rotvec_to_rotmat(r1)
        R2 = roma.rotvec_to_rotmat(r2)
    elif rotation_representation == "quaternion":
        R1 = roma.unitquat_to_rotmat(r1)
        R2 = roma.unitquat_to_rotmat(r2)
    else:
        raise ValueError(
            f"Unknown rotation_representation '{rotation_representation}'. Must be 'quaternion' or 'rotvec'."
        )

    angles_rad = roma.rotmat_geodesic_distance(R1, R2)
    return angles_rad / torch.pi * 180
