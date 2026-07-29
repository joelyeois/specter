from __future__ import annotations

import roma
import torch
import torch.nn.functional as F


def rotate_coordinates(vectors: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors by quaternions.

    Parameters
    ----------
    vectors : torch.Tensor
        (N, 3) or (3,) coordinates.
    quat : torch.Tensor
        (B, 4) or (4,) quaternions, xyzw convention.

    Returns
    -------
    rotated : torch.Tensor
        Rotated coordinates.
    """
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    R = roma.unitquat_to_rotmat(quat)  # (3,3) or (B,3,3)
    N = vectors.shape[0]
    if R.ndim == 2:  # single rotation
        rotated = vectors @ R.T
    else:  # batch of rotations
        B = R.shape[0]
        vectors_exp = vectors.unsqueeze(0).expand(B, N, 3)
        rotated = torch.einsum("bij,bkj->bki", R, vectors_exp)
    if rotated.ndim == 3 and rotated.shape[1] == 1:
        rotated = rotated.squeeze(1)
    return rotated


def translate_coordinates(
    vectors: torch.Tensor, T: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """
    Apply translation to points, with broadcasting.

    Parameters
    ----------
    vectors : torch.Tensor
        Shape: (N,3) or (B,N,3)
    T : torch.Tensor
        Shape: (3,), (1,3), or (B,3). Last dim can be 2 or 3.
    inverse : bool
        If True, subtract translation instead of adding.

    Returns
    -------
    translated : torch.Tensor
        Same shape as vectors
    """
    # Pad z=0 if last dim is 2
    if T.shape[-1] == 2:
        T_full = F.pad(T, (0, 1))  # pad last dim with 1 zero
    elif T.shape[-1] == 3:
        T_full = T
    else:
        raise ValueError("T must have last dimension 2 or 3")

    sign = -1 if inverse else 1

    if vectors.ndim == 2:  # single rotation
        if T_full.ndim == 1:
            T_full = T_full.unsqueeze(0)  # (1,3)
        return vectors + sign * T_full
    else:  # batch rotation
        B, N, _ = vectors.shape
        if T_full.ndim == 2:
            if T_full.shape[0] == B:
                T_full = T_full[:, None, :]
            elif T_full.shape[0] == 1:
                T_full = T_full
            else:
                raise ValueError("T shape not compatible with batch size")
        return vectors + sign * T_full
