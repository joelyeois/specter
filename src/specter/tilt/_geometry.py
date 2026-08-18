from __future__ import annotations

import math
from typing import Sequence

import roma
import torch
import torch.nn.functional as F

from ..filters import cosine_taper_window


def estimate_required_nxy(desired_nxy: int, nz: int, max_tilt_angle_deg: float) -> int:
    """Minimum XY size so the tilted projection still covers ``desired_nxy`` pixels."""
    theta_rad = math.radians(max_tilt_angle_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    return math.ceil((desired_nxy + nz * sin_t) / cos_t)


def estimate_max_allowed_nxy(
    available_nxy: int, nz: int, max_tilt_angle_deg: float
) -> int:
    """Maximum output XY achievable given the available volume at this tilt."""
    theta_rad = math.radians(max_tilt_angle_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    return math.ceil(available_nxy * cos_t - nz * sin_t)


def estimate_max_allowed_tilt_deg(
    desired_nxy: int, nz: int, available_nxy: int
) -> float:
    """Largest tilt (degrees) such that the available volume still covers ``desired_nxy``."""
    thetas_deg = torch.linspace(0.0, 89.9, 4000)
    thetas_rad = torch.deg2rad(thetas_deg)
    spans = available_nxy * torch.cos(thetas_rad) - nz * torch.sin(thetas_rad)
    valid = spans >= desired_nxy
    if not bool(valid.any()):
        return 0.0
    return float(thetas_deg[valid][-1].item())


def infer_max_tilt_from_inputs(
    angles: torch.Tensor | Sequence[float] | None = None,
    quaternions: torch.Tensor | None = None,
) -> float:
    """Infer max tilt magnitude in degrees from provided poses."""
    if angles is not None:
        return float(torch.as_tensor(angles).abs().max())
    if quaternions is not None:
        rotvecs = roma.unitquat_to_rotvec(torch.as_tensor(quaternions))
        max_angle_rad = torch.linalg.norm(rotvecs, dim=-1).max()
        return float(max_angle_rad * (180.0 / torch.pi))
    return 0.0


def pad_volume_xy_for_tilt(
    volume: torch.Tensor, required_nxy: int, available_nxy: int
) -> torch.Tensor:
    """
    Pad ``volume`` symmetrically in XY using reflect mode to reach ``required_nxy``.

    Parameters
    ----------
    volume : torch.Tensor
        Volume of shape (..., Z, Y, X).
    required_nxy : int
        Target XY size after padding.
    available_nxy : int
        Current XY extent of ``volume``.

    Returns
    -------
    volume : torch.Tensor
        Reflect-padded volume.
    """
    pad_each_side = (required_nxy - available_nxy + 1) // 2
    return F.pad(
        volume,
        (pad_each_side, pad_each_side, pad_each_side, pad_each_side, 0, 0),
        mode="reflect",
    )


def apply_volume_cosine_taper(
    volume: torch.Tensor, taper_xy: int = 0, taper_z: int = 0
) -> torch.Tensor:
    """
    Apply a cosine taper to the XY and/or Z edges of a volume.

    Parameters
    ----------
    volume : torch.Tensor
        Volume of shape (..., Z, Y, X).
    taper_xy : int
        Taper width in XY pixels. 0 to skip.
    taper_z : int
        Taper width in Z pixels. 0 to skip.

    Returns
    -------
    volume : torch.Tensor
        Volume with taper applied.
    """
    if taper_xy <= 0 and taper_z <= 0:
        return volume

    nz, ny, nx = volume.shape[-3], volume.shape[-2], volume.shape[-1]
    device, dtype = volume.device, volume.dtype
    mask = torch.ones(1, device=device, dtype=dtype)

    if taper_xy > 0:
        win_y = cosine_taper_window(ny, taper_xy, device, dtype)
        win_x = cosine_taper_window(nx, taper_xy, device, dtype)
        mask = mask * win_y[:, None] * win_x[None, :]

    if taper_z > 0:
        win_z = cosine_taper_window(nz, taper_z, device, dtype)
        mask = win_z[:, None, None] * mask if mask.ndim == 2 else win_z[:, None, None]

    return volume * mask


def nz_tilt_for_pose(volume_shape: tuple[int, ...], theta_matrix: torch.Tensor) -> int:
    """
    Number of Z slices needed to fully cover a volume once rotated by ``theta_matrix``.

    A pure function of ``volume_shape`` and the pose -- shared by
    :class:`~specter.imagegenerator.TiltSeriesGenerator` (forward model) and
    :class:`~specter.ghostbuster.TomogramReconstructor` (inverse problem),
    both of which need it to compute the multislice propagation depth and
    the resulting defocus shift (see :func:`shift_ctf_defocus_for_tilt`).

    Parameters
    ----------
    volume_shape : tuple of int
        Volume shape ``(B, Z, Y, X)``. Only the extents matter, not the data.
    theta_matrix : torch.Tensor
        Affine transformation matrix of shape (B, 3, 4) or (B, 4, 4).

    Returns
    -------
    nz_new : int
        Number of slices.
    """
    _, Z, Y, X = volume_shape
    device, dtype = theta_matrix.device, theta_matrix.dtype
    R = theta_matrix[:, :3, :3]

    corners = torch.tensor(
        [
            [-X / 2, -Y / 2, -Z / 2],
            [X / 2, -Y / 2, -Z / 2],
            [-X / 2, Y / 2, -Z / 2],
            [X / 2, Y / 2, -Z / 2],
            [-X / 2, -Y / 2, Z / 2],
            [X / 2, -Y / 2, Z / 2],
            [-X / 2, Y / 2, Z / 2],
            [X / 2, Y / 2, Z / 2],
        ],
        device=device,
        dtype=dtype,
    ).t()  # (3, 8)

    rotated_corners = torch.bmm(
        R.transpose(1, 2), corners.unsqueeze(0).expand(R.shape[0], -1, -1)
    )
    z_min = rotated_corners[:, 2, :].min(dim=1).values
    z_max = rotated_corners[:, 2, :].max(dim=1).values
    nz_new = int(torch.ceil((z_max - z_min).max()).item())
    return max(1, nz_new)


def shift_ctf_defocus_for_tilt(
    ctf_params: dict[str, torch.Tensor],
    volume_shape: tuple[int, ...],
    theta_matrix: torch.Tensor,
    nz: int,
    pixel_size: float,
) -> dict[str, torch.Tensor]:
    """
    Shift ``dfu``/``dfv`` to account for the extra Z depth multislice propagates
    through at this tilt.

    Multislice propagates ``nz_tilt_for_pose(volume_shape, theta_matrix)`` slices
    instead of the untilted ``nz``, shifting the effective specimen centre by
    ``(nz_new - nz) * pixel_size / 2``. Returns a new dict; ``ctf_params``
    itself is left untouched.

    Parameters
    ----------
    ctf_params : dict[str, torch.Tensor]
        Per-tilt CTF parameters for this tilt (batch dimension already sliced).
    volume_shape : tuple of int
        Volume shape ``(B, Z, Y, X)`` passed to the propagator.
    theta_matrix : torch.Tensor
        Affine transformation matrix of shape (B, 3, 4) or (B, 4, 4).
    nz : int
        Untilted number of Z slices.
    pixel_size : float
        Pixel/voxel size in Å.

    Returns
    -------
    dict[str, torch.Tensor]
        ``ctf_params`` with ``dfu``/``dfv`` (if present) shifted.
    """
    nz_new = nz_tilt_for_pose(volume_shape, theta_matrix)
    z_offset = (nz_new - nz) * pixel_size / 2.0
    shifted = dict(ctf_params)
    if "dfu" in shifted:
        shifted["dfu"] = shifted["dfu"] - z_offset
    if "dfv" in shifted:
        shifted["dfv"] = shifted["dfv"] - z_offset
    return shifted
