"""
Tilt-series geometry: the XY coverage a tilt range needs, edge tapers, the
number of slices under a pose, and the defocus shift across a tilted
specimen.
"""

from __future__ import annotations

import math
from typing import Sequence

import roma
import torch
import torch.nn.functional as F

from .._defaults import FFT_PAD_MARGIN
from ..filters import cosine_taper_window
from ..options import TiltAxis


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


def estimate_required_extent(
    desired_nxy: int, nz: int, rotations: torch.Tensor
) -> tuple[int, int]:
    """
    The Y and X extent a set of poses samples of a slab.

    A pose's image is the projection of the points ``p = R q`` of the
    volume, with ``q`` running over the square frame and down the beam. Only
    the points inside the slab matter, ``|p_z| <= nz / 2``, which bounds
    each lateral coordinate by a sum of three terms linear in the frame's
    half-width and the slab's half-depth. For a tilt of ``theta`` about x
    this is ``(desired_nxy + nz * sin(theta)) / cos(theta)`` along y, as
    `estimate_required_nxy` gives, and ``desired_nxy`` along x.

    Parameters
    ----------
    desired_nxy : int
        Size of the square frame, in pixels.
    nz : int
        Depth of the slab, in voxels.
    rotations : torch.Tensor
        ``(N, 3, 3)`` rotation matrices, as `build_affine_matrix` takes them.

    Returns
    -------
    ny, nx : int
        The extent the poses need along Y and along X, in voxels.
    """
    R = torch.as_tensor(rotations).double().reshape(-1, 3, 3)
    lateral, beam = R[:, :2, :], R[:, 2, :]
    slope = lateral[:, :, 2] / beam[:, 2:]
    across = lateral[:, :, :2] - slope[:, :, None] * beam[:, None, :2]
    extent = desired_nxy * across.abs().sum(-1) + nz * slope.abs()
    # A part in 1e9 below the ceiling, so that an extent which is a whole
    # number of voxels does not round up on its last bit.
    nx, ny = torch.ceil(extent.amax(0) - 1e-9).tolist()
    return int(ny), int(nx)


def volume_shape_for_tilt(
    field_shape: tuple[int, int, int],
    max_tilt_angle_deg: float,
    tilt_axis: str = "y",
    edge_margin: int = 8,
    canvas_margin: int = FFT_PAD_MARGIN,
) -> tuple[int, int, int]:
    """
    The volume a field of view needs around it to be imaged up to a tilt.

    Across the tilt axis the volume holds everything a view of the field at
    `max_tilt_angle_deg` passes through (`estimate_required_nxy`) plus the
    edge margin. Along the tilt axis a tilt needs nothing more of the
    specimen, so the volume only spans the propagation canvas, the field
    and its margin, plus the same edge margin. A tilt series of a volume
    this size therefore pads nothing by reflection
    (`specter.imaging.TiltSeries`). Each axis grows by the same whole number
    of voxels on both sides, so the field is the centred crop of the volume.

    Parameters
    ----------
    field_shape : tuple of int
        The field of view, ``(Z, Y, X)`` voxels.
    max_tilt_angle_deg : float
        The largest tilt of the series, in degrees.
    tilt_axis : str, optional
        ``"x"`` or ``"y"``, the axis the series tilts about. Default ``"y"``.
    edge_margin : int, optional
        `TiltSeries`' ``edge_margin``. Default 8.
    canvas_margin : int, optional
        `TiltSeries`' ``fft_pad_margin``. Default 32.

    Returns
    -------
    tuple of int
        The volume's shape, ``(Z, Y, X)`` voxels.
    """
    if tilt_axis not in ("x", "y"):
        raise ValueError(f"tilt_axis={tilt_axis!r} must be 'x' or 'y'")
    nz, ny, nx = field_shape
    along, across = (nx, ny) if tilt_axis == "x" else (ny, nx)
    grow_across = (
        estimate_required_nxy(across, nz, abs(max_tilt_angle_deg))
        + 2 * edge_margin
        - across
    )
    across += 2 * math.ceil(grow_across / 2)
    along += 2 * (canvas_margin + edge_margin)
    return (nz, across, along) if tilt_axis == "x" else (nz, along, across)


def pad_volume_xy_for_tilt(
    volume: torch.Tensor,
    required_nxy: int | tuple[int, int],
    available_nxy: int | None = None,
) -> torch.Tensor:
    """
    Pad ``volume`` symmetrically in XY using reflect mode to reach ``required_nxy``.

    Parameters
    ----------
    volume : torch.Tensor
        Volume of shape (..., Z, Y, X).
    required_nxy : int or tuple of int
        Target XY size after padding, or ``(ny, nx)`` for a target per axis.
        An axis already that large is left alone.
    available_nxy : int, optional
        The extent both axes are taken to have. Default: each axis's own.

    Returns
    -------
    volume : torch.Tensor
        Reflect-padded volume.
    """
    if isinstance(required_nxy, int):
        required_nxy = (required_nxy, required_nxy)
    pad_y, pad_x = (
        max(0, (required - (n if available_nxy is None else available_nxy) + 1) // 2)
        for required, n in zip(required_nxy, volume.shape[-2:], strict=True)
    )
    if pad_y == 0 and pad_x == 0:
        return volume
    return F.pad(volume, (pad_x, pad_x, pad_y, pad_y, 0, 0), mode="reflect")


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
    :class:`~specter.imaging.TiltSeries` (forward model) and
    :class:`~specter.ghostbuster.TomogramReconstructor` (inverse problem),
    both of which need it to compute the multislice propagation depth and
    the resulting defocus shift (``imaging._tilt.tilt_defocus_shift``).

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


def turn_to_tilt_axis(
    quaternions: torch.Tensor, axis: TiltAxis = "x"
) -> tuple[torch.Tensor, float]:
    """
    Pose a tilt series in a volume frame whose ``axis`` is the tilt axis.

    A pose ``R`` takes beam coordinates to the volume's, so the beam runs
    along ``R e_z`` in the volume. In a single-axis series those directions
    lie in the plane perpendicular to the tilt axis. The frame is turned
    about the volume's z by the rotation ``M`` that brings that axis onto
    ``axis``, and the poses become ``M R``. A volume reconstructed with the
    returned poses holds the point ``r`` of the original frame at ``M r``.

    With the tilt axis along a volume axis, every ray keeps its coordinate
    along it, the volume that covers every ray
    (`TomogramReconstructor.covering_shape`) grows across the axis only, and
    the volume can be solved in strips along it
    (`specter.ghostbuster._tomogram_solver.solve_tilt_series_in_strips`).

    Parameters
    ----------
    quaternions : torch.Tensor
        Per-tilt rotation quaternions, ``(N_tilts, 4)``, xyzw convention.
    axis : {"x", "y"}
        The volume axis to bring the tilt axis onto. Default ``"x"``.

    Returns
    -------
    quaternions : torch.Tensor
        The poses in the turned frame, ``(N_tilts, 4)``.
    angle : float
        The angle of ``M`` in degrees, counter-clockwise about z: the turn
        of at most 90 degrees that brings the tilt axis onto ``axis``.

    Notes
    -----
    The tilt axis is the in-plane direction ``n`` minimising
    ``sum_i (n . R_i e_z)^2``, the least-squares axis of a series that is
    not exactly single-axis. An untilted series has no axis, and any frame
    serves; it is returned unchanged.
    """
    rot = roma.unitquat_to_rotmat(torch.as_tensor(quaternions).double())
    beam = rot[:, :2, 2]  # in-plane part of R e_z, per tilt
    scatter = beam.T @ beam
    if float(scatter.trace()) < 1e-12:
        return torch.as_tensor(quaternions), 0.0
    n = torch.linalg.eigh(scatter).eigenvectors[:, 0]  # smallest eigenvalue
    lead = 0 if str(axis).lower() == "x" else 1
    if n[lead] < 0:
        n = -n
    # Angle from `axis` to the tilt axis; M undoes it.
    angle = math.atan2(float(n[1]), float(n[0]))
    if lead == 1:
        angle -= math.pi / 2
    c, s_ = math.cos(-angle), math.sin(-angle)
    turn = torch.tensor(
        [[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    turned = roma.rotmat_to_unitquat(turn @ rot)
    return turned.to(torch.as_tensor(quaternions).dtype), math.degrees(-angle)
