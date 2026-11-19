"""Rotation grids, and the frames a point group leaves equivalent."""

from __future__ import annotations

import math

import torch

from ..symmetries import get_rotation_matrices

_TWO_PI = 2.0 * math.pi


def rotvec_to_rotmat(v: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrices of rotation vectors, differentiable at zero.

    Parameters
    ----------
    v : torch.Tensor
        Rotation vectors (..., 3): the axis scaled by the angle in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrices (..., 3, 3).
    """
    k = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    k[..., 0, 1], k[..., 0, 2] = -v[..., 2], v[..., 1]
    k[..., 1, 0], k[..., 1, 2] = v[..., 2], -v[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -v[..., 1], v[..., 0]
    return torch.linalg.matrix_exp(k)


def _is_normalised_by(group: torch.Tensor, candidates: torch.Tensor) -> bool:
    """Whether every candidate rotation conjugates the group onto itself."""
    conj = torch.einsum("nij,gjk,nlk->ngil", candidates, group, candidates)
    d = (conj[:, :, None] - group[None, None]).abs().amax((-1, -2)).amin(-1)
    return bool((d < 1e-4).all())


def frame_group(sym: str) -> torch.Tensor | None:
    """
    Rotations that turn one aligned frame of a point group into another.

    A volume aligned to the group's axes stays aligned when it is turned by
    any element of the group's normaliser in SO(3), so an alignment is
    determined only up to that set. It is D_2n for D_n (n >= 3), O for D2, T
    and O, and I for I.

    Parameters
    ----------
    sym : str
        Point-group symbol, as `specter.symmetries.get_rotation_matrices`
        takes it.

    Returns
    -------
    torch.Tensor or None
        Rotation matrices (N, 3, 3), float64, the identity first. None for
        C_n, whose normaliser is continuous: any turn about the axis, and the
        flip of the axis.
    """
    s = sym.upper()
    group = get_rotation_matrices(s, return_affine=False).double()
    if s.startswith("C"):
        return None
    if s == "D2":
        cand = get_rotation_matrices("O", return_affine=False).double()
    elif s.startswith("D"):
        cand = get_rotation_matrices(f"D{2 * int(s[1:])}", return_affine=False).double()
    elif s == "T":
        # T and its coset under a quarter turn about one of its twofold axes.
        half_turns = (group.diagonal(dim1=-2, dim2=-1).sum(-1) + 1).abs() < 1e-6
        two = group[half_turns][0]
        w, v = torch.linalg.eigh((two + two.T) / 2)
        quarter = rotvec_to_rotmat(v[:, w.argmax()] * (math.pi / 2))
        cand = torch.cat([group, group @ quarter])
    else:
        cand = group
    frames = cand if _is_normalised_by(group, cand) else group
    eye = torch.eye(3, dtype=torch.float64)
    first = (frames - eye).abs().amax((-1, -2)).argmin()
    keep = torch.arange(frames.shape[0]) != first
    return torch.cat([frames[first][None], frames[keep]])


def so3_grid(step: float) -> torch.Tensor:
    """
    Near-uniform rotations from a super-Fibonacci spiral.

    Parameters
    ----------
    step : float
        Target spacing between neighbouring rotations, in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrices (N, 3, 3), float64, with ``N = 8 pi^2 / step^3``.

    References
    ----------
    .. [1] Alexa, M. (2022). Super-Fibonacci spirals: Fast, low-discrepancy
       sampling of SO(3). *2022 IEEE/CVF Conference on Computer Vision and
       Pattern Recognition (CVPR)*, 8281–8290.
       doi:10.1109/CVPR52688.2022.00811
    """
    n = max(1, math.ceil(8 * math.pi**2 / step**3))
    i = torch.arange(n, dtype=torch.float64) + 0.5
    s = i / n
    a = _TWO_PI * i / math.sqrt(2.0)
    b = _TWO_PI * i / 1.533751168755204288118041
    r, rr = s.sqrt(), (1 - s).sqrt()
    x, y, z, w = r * a.sin(), r * a.cos(), rr * b.sin(), rr * b.cos()
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1
            ),
            torch.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1
            ),
            torch.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def tilt_to(axis: torch.Tensor) -> torch.Tensor:
    """
    The smallest rotation taking z to each given direction.

    Parameters
    ----------
    axis : torch.Tensor
        Unit vectors (..., 3) with ``z > -1``.

    Returns
    -------
    torch.Tensor
        Rotation matrices (..., 3, 3) whose third column is `axis`.
    """
    x, y, z = axis.unbind(-1)
    f = 1 / (1 + z)
    return torch.stack(
        [
            torch.stack([1 - f * x * x, -f * x * y, x], -1),
            torch.stack([-f * x * y, 1 - f * y * y, y], -1),
            torch.stack([-x, -y, z], -1),
        ],
        -2,
    )


def axis_grid(step: float) -> torch.Tensor:
    """
    Rotations taking z to near-uniform directions on the upper hemisphere.

    The grid a cyclic group needs: its score does not depend on the spin
    about the axis, nor on the axis's sign.

    Parameters
    ----------
    step : float
        Target spacing between neighbouring directions, in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrices (N, 3, 3), float64, without spin about z.
    """
    n = max(1, math.ceil(2 * math.pi / step**2))
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - i / n
    phi = _TWO_PI * i / ((1 + math.sqrt(5.0)) / 2)
    rho = (1 - z * z).clamp_min(0).sqrt()
    return tilt_to(torch.stack([rho * phi.cos(), rho * phi.sin(), z], -1))


def nearest_frame(rot: torch.Tensor, frames: torch.Tensor | None) -> torch.Tensor:
    """
    Of the equivalent aligned frames, the one closest to the identity.

    Parameters
    ----------
    rot : torch.Tensor
        Rotation matrices (..., 3, 3).
    frames : torch.Tensor or None
        `frame_group`'s output. None is the cyclic case: the axis is flipped
        to the upper hemisphere and the spin about it removed.

    Returns
    -------
    torch.Tensor
        ``rot @ n`` for the equivalent frame ``n`` that turns the volume
        least.
    """
    if frames is not None:
        tr = torch.einsum("...ij,nji->...n", rot, frames)
        return rot @ frames[tr.argmax(-1)]
    flip = torch.diag(
        torch.tensor([1.0, -1.0, -1.0], dtype=rot.dtype, device=rot.device)
    )
    rot = torch.where((rot[..., 2, 2] < 0)[..., None, None], rot @ flip, rot)
    phi = torch.atan2(rot[..., 0, 1] - rot[..., 1, 0], rot[..., 0, 0] + rot[..., 1, 1])
    c, s, o, i = phi.cos(), phi.sin(), torch.zeros_like(phi), torch.ones_like(phi)
    spin = torch.stack(
        [
            torch.stack([c, -s, o], -1),
            torch.stack([s, c, o], -1),
            torch.stack([o, o, i], -1),
        ],
        -2,
    )
    return rot @ spin


def frame_distance(
    a: torch.Tensor, b: torch.Tensor, frames: torch.Tensor | None
) -> torch.Tensor:
    """
    Angle between two aligned frames, up to the equivalent choices.

    Parameters
    ----------
    a, b : torch.Tensor
        Rotation matrices (..., 3, 3).
    frames : torch.Tensor or None
        `frame_group`'s output. None is the cyclic case, where the distance
        is the angle between the two axes as undirected lines.

    Returns
    -------
    torch.Tensor
        Angles in radians, (...).
    """
    if frames is None:
        za, zb = a[..., :, 2], b[..., :, 2]
        return torch.atan2(
            torch.linalg.cross(za, zb).norm(dim=-1), (za * zb).sum(-1).abs()
        )
    # The chord length rather than the trace: acos loses half the digits near zero.
    chord = (
        (a[..., None, :, :] - b[..., None, :, :] @ frames)
        .flatten(-2)
        .norm(dim=-1)
        .amin(-1)
    )
    return 2 * torch.asin((chord / (2 * math.sqrt(2.0))).clamp(max=1))
