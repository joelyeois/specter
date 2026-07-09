from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from ..fft import fft3, ifft3


def _relion_rotation_grid(
    theta: torch.Tensor,
    nz: int,
    ny: int,
    nx: int,
    align_corners: bool,
) -> torch.Tensor:
    """
    Build a sampling grid for RELION-convention rotation about [nz//2, ny//2, nx//2].

    Parameters
    ----------
    theta : torch.Tensor
        Batch of affine matrices, Bx3x4.
    nz, ny, nx : int
        Volume dimensions.
    align_corners : bool
        Passed to affine_grid.

    Returns
    -------
    grid : torch.Tensor
        Sampling grid, shape (B, nz, ny, nx, 3).
    """
    B = theta.size(0)

    null_rot = torch.zeros_like(theta)
    null_rot[:, 0, 0] = 1.0
    null_rot[:, 1, 1] = 1.0
    null_rot[:, 2, 2] = 1.0
    grid = F.affine_grid(null_rot, (B, 1, nz, ny, nx), align_corners=align_corners)

    # scale coordinates by (nx-1)/2, (ny-1)/2, (nz-1)/2 to make them isotropic
    scale = torch.tensor(
        [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2],
        device=theta.device,
        dtype=theta.dtype,
    ).view(1, 1, 3)

    # rotate the grid around the center defined by RELION
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    center = grid[:, cz, cy, cx]  # (B, 3)
    center = center.unsqueeze(1)
    grid = grid.view(B, nz * ny * nx, 3)

    # Apply isotropic rotation
    grid = (grid - center) * scale
    grid = grid.bmm(theta[..., :-1].transpose(1, 2))
    grid = (grid / scale) + center

    # translate the grid
    dx = theta[..., -1].unsqueeze(1)  # (B, 1, 3)
    grid = grid + dx
    return grid.view(B, nz, ny, nx, 3)


def rotate_volume(
    V: torch.Tensor,
    theta: torch.Tensor,
    origin: Literal["relion", "center"] = "relion",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Rotates a single 3D volume based on the batch of 3x4 affine transform matrices.

    Written by Tristan Bepler.

    Parameters
    ----------
    V : torch.Tensor
        The volume to be rotated, must be real-valued. Shape (Z, Y, X).
    theta : torch.Tensor
        Batch of affine matrices, Bx3x4.
        Concatenates a 3x3 rotation matrix and a 3x1 translation vector.
    origin : str, optional
        Convention for the index of the origin of rotation. "relion" defines the
        origin to be at [nz//2, ny//2, nx//2], whereas "center" sets it to
        [(nz + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]. Default "relion".
    padding_mode : str, optional
        Padding mode for grid_sample. Default "border".

    Returns
    -------
    V_rotated : torch.Tensor
         The rotated volumes. Shape (B, Z, Y, X).
    """

    B = theta.size(0)
    nz, ny, nx = V.size()

    # create the coordinate grid depending on origin convention.
    if origin == "relion":
        grid = _relion_rotation_grid(theta, nz, ny, nx, align_corners)
    elif origin == "center":
        grid = F.affine_grid(theta, (B, 1, nz, ny, nx), align_corners=align_corners)

    # transform the volume
    V = V.unsqueeze(0).unsqueeze(1)  # (1 x 1 x Z x Y x X)
    V = V.expand(B, 1, nz, ny, nx)
    V_ = F.grid_sample(V, grid, align_corners=align_corners, padding_mode=padding_mode)

    return V_.squeeze(1)  # B x Z x Y x X


def rotate_volume_fourier(
    V: torch.Tensor,
    theta: torch.Tensor,
    origin: Literal["relion", "center"] = "relion",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Rotate a 3D volume by interpolating in Fourier space.

    Transforms to Fourier space, rotates real and imaginary parts separately
    using :func:`rotate_volume`, then transforms back.

    Parameters
    ----------
    V : torch.Tensor
        Volume to rotate, shape (Z, Y, X).
    theta : torch.Tensor
        Batch of affine matrices, shape (B, 3, 4).
    origin : str, optional
        Rotation origin convention ('relion' or 'center'). Default is 'relion'.
    padding_mode : str, optional
        Padding mode for grid sampling. Default is 'border'.
    align_corners : bool, optional
        Passed to affine_grid and grid_sample. Default is False.

    Returns
    -------
    V_rot : torch.Tensor
        Rotated volume, shape (B, Z, Y, X).
    """
    # Fourier domain
    V_f = fft3(V, shift=True)  # Z x X x Y

    # rotate real and imag parts
    V_f_rot_real = rotate_volume(
        V_f.real, theta, origin="relion", padding_mode="border", align_corners=False
    )
    V_f_rot_imag = rotate_volume(
        V_f.imag, theta, origin="relion", padding_mode="border", align_corners=False
    )
    V_f_rot = torch.complex(V_f_rot_real, V_f_rot_imag)
    V_rot = ifft3(V_f_rot, shift=True)
    return V_rot.real  # B x Z x X x Y


def translations_angstrom_to_torch(
    T: torch.Tensor, n: int, voxel_size: float
) -> torch.Tensor:
    """
    Builds a batch of normalized translation vectors from rlnOriginXAngst and rlnOriginYAngst.

    Torch affine matrix uses a normalized coordinate system, where the coordinates
    of each axis ranges from [-1, 1]. Therefore, we need to do a coordinate
    transformation to match this Torch coordinate system for translations.

    Parameters
    ----------
    T : torch.Tensor
        Translation vector of shape (N, 2), built from [rlnOriginXAngst, rlnOriginYAngst]. In Å.
    n : int
        Number of pixels in x/y direction.
    voxel_size : float
        Voxel size in Å.

    Returns
    -------
    T_norm : torch.Tensor
        Batch of Torch normalized translation vectors with shape (N, 3).
    """
    num = len(T)
    if T.shape[-1] == 2:
        tz = torch.zeros(num, device=T.device)
        T = torch.concat([T, tz[..., None]], dim=-1)
    T_norm = T * 2 / n / voxel_size
    return T_norm


def build_affine_matrix(R: torch.Tensor, T: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build a batch of Torch affine matrices (N, 3, 4) from rotation matrices and translations.

    CryoSPARC performs shifts before rotations. The affine matrix performs rotations
    before shifts, so the translation vector is pre-rotated:
    T_i' = sum_j R_ij * T_j

    Parameters
    ----------
    R : torch.Tensor
        Batch of rotation matrices, shape (N, 3, 3).
    T : torch.Tensor, optional
        Batch of Torch-normalized translation vectors, shape (N, 3).
        Must be normalized via :func:`translations_angstrom_to_torch` first.
        If None, zero translations are used.

    Returns
    -------
    theta : torch.Tensor
        Batch of affine matrices, shape (N, 3, 4).
    """
    if R.ndim == 2:
        R = R.unsqueeze(0)
    if T is not None and T.ndim == 1:
        T = T.unsqueeze(0)
    if T is None:
        T = R.new_zeros(R.shape[0], 3)

    Tprime = torch.zeros_like(T)
    Tprime[:, 0] = R[:, 0, 0] * T[:, 0] + R[:, 0, 1] * T[:, 1] + R[:, 0, 2] * T[:, 2]
    Tprime[:, 1] = R[:, 1, 0] * T[:, 0] + R[:, 1, 1] * T[:, 1] + R[:, 1, 2] * T[:, 2]
    Tprime[:, 2] = R[:, 2, 0] * T[:, 0] + R[:, 2, 1] * T[:, 1] + R[:, 2, 2] * T[:, 2]
    theta = torch.concat([R, Tprime.unsqueeze(2)], dim=-1)
    return theta
