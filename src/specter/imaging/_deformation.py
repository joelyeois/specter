"""
Beam-induced bending of a thin specimen, as the map a tilt samples its
volume through.

A `~specter.settings.Deformation` describes the bend as a deflection
``w(x, y)`` of the specimen's midplane. Every point of the specimen moves
with it under thin-plate (Kirchhoff) kinematics: a point at ``X = (x, y, z)``,
``z`` above the midplane, moves to ``X + u(X)`` with

    u(X) = (-z dw/dx, -z dw/dy, w).

The volume is rendered undeformed, so a sample point ``s`` of the deformed
specimen reads the undeformed volume where its material came from, the
solution ``X`` of ``s = X + u(X)``, and the value read is divided by
``det(I + grad u)`` so that the deformation conserves the integrated
potential. Both happen inside the sampling a tilt already does
(`VolumeRotator.sample_rotated_slices`), so the bend costs no interpolation
of its own.

Beyond the volume's faces there is no material, and the field holds the
faces' height there: the vacuum a steep tilt samples far above the film
moves with the film's surface rather than with a lever arm that grows
with its distance. That bounds the map's gradient by the slope plus the
bending strain at the faces, `Deformation.max_gradient`.

The kinematics are linear in that gradient, which leaves an error of order
its square: a material line normal to the film keeps its length to first
order only. `settings.MAX_DEFORMATION_GRADIENT` bounds it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..settings import Deformation

__all__ = [
    "Warp",
    "burst_fraction",
    "dome_curvature",
    "dome_deflection",
    "specimen_warp",
]

#: Maps sample points ``(..., 3)``, ``(x, y, z)`` in voxels from the volume's
#: centre, to the points of the undeformed volume they read and the weight
#: each value read is multiplied by.
Warp = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

#: How closely the pull-back is solved, in voxels.
_PULL_BACK_TOLERANCE = 1e-3


def burst_fraction(dose: float, burst_dose: float) -> float:
    """
    The fraction of the full bend reached at an accumulated `dose`.

    Parameters
    ----------
    dose : float
        Exposure the specimen has received, in e-/A^2.
    burst_dose : float
        Exposure over which the burst completes, in e-/A^2.

    Returns
    -------
    float
        ``1 - exp(-dose / burst_dose)``.
    """
    return -math.expm1(-max(dose, 0.0) / burst_dose)


def dome_deflection(
    x: torch.Tensor, y: torch.Tensor, height: float, radius: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A clamped circular film's deflection and slope.

    ``w = height * (1 - r^2 / radius^2)^2`` for ``r <= radius`` and zero
    outside: the deflection of a circular plate clamped at its rim, so the
    film meets the support with no slope and the displacement is continuous
    across the rim.

    Parameters
    ----------
    x, y : torch.Tensor
        Positions from the hole's centre, in Angstrom.
    height : float
        Central deflection in Angstrom.
    radius : float
        Hole radius in Angstrom.

    Returns
    -------
    tuple of torch.Tensor
        ``(w, dw/dx, dw/dy)``, each shaped like `x`.
    """
    a2 = radius * radius
    q = (1.0 - (x * x + y * y) / a2).clamp(min=0.0)
    c = (-4.0 * height / a2) * q
    return height * q * q, c * x, c * y


def dome_curvature(
    x: torch.Tensor, y: torch.Tensor, height: float, radius: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The second derivatives of :func:`dome_deflection`.

    Returns
    -------
    tuple of torch.Tensor
        ``(d2w/dx2, d2w/dxdy, d2w/dy2)``, zero over the support.
    """
    a2 = radius * radius
    q = 1.0 - (x * x + y * y) / a2
    inside = (q >= 0.0).to(x.dtype)
    c = (-4.0 * height / a2) * inside
    return (
        c * (q - 2.0 * x * x / a2),
        -2.0 * c * x * y / a2,
        c * (q - 2.0 * y * y / a2),
    )


def _iterations(lipschitz: float, displacement: float) -> int:
    """Fixed-point steps contracting `displacement` voxels to the tolerance."""
    if displacement <= _PULL_BACK_TOLERANCE or lipschitz <= 0.0:
        return 1
    target = _PULL_BACK_TOLERANCE * (1.0 - lipschitz) / displacement
    return max(1, math.ceil(math.log(target) / math.log(lipschitz)))


def specimen_warp(
    deformation: Deformation, dose: float, voxel_size: float, half_depth: float
) -> Warp | None:
    """
    The map a tilt recorded at `dose` samples its volume through.

    Parameters
    ----------
    deformation : Deformation
        The bend.
    dose : float
        The exposure the specimen had received at this tilt, in e-/A^2;
        `TiltSeries` passes the middle of the tilt's own exposure.
    voxel_size : float
        In Angstrom.
    half_depth : float
        Half the volume's depth in Angstrom: the height of its faces above
        the midplane, where the field stops growing with ``z``.

    Returns
    -------
    Warp or None
        None for a rigid specimen, or one with no bend yet.
    """
    if deformation.model is None or deformation.height is None:
        return None
    fraction = burst_fraction(dose, deformation.burst_dose)
    height = deformation.height * fraction
    if height == 0.0:
        return None
    radius = deformation.hole_radius
    ox, oy = deformation.hole_offset
    # The map's Lipschitz constant and largest displacement scale with the
    # deflection, so a bend still growing needs fewer steps.
    steps = _iterations(
        deformation.max_gradient(half_depth) * fraction,
        (abs(height) + half_depth * deformation.max_slope * fraction) / voxel_size,
    )

    def warp(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sx = points[..., 0] * voxel_size - ox
        sy = points[..., 1] * voxel_size - oy
        sz = points[..., 2] * voxel_size
        # Solve s = X + u(X) for X by fixed-point iteration from X = s.
        x, y, z = sx, sy, sz
        for _ in range(steps):
            w, w_x, w_y = dome_deflection(x, y, height, radius)
            lever = z.clamp(-half_depth, half_depth)
            x = sx + lever * w_x
            y = sy + lever * w_y
            z = sz - w
        w, w_x, w_y = dome_deflection(x, y, height, radius)
        w_xx, w_xy, w_yy = dome_curvature(x, y, height, radius)
        lever = z.clamp(-half_depth, half_depth)
        # det(I + grad u), in closed form. Inside the volume the lever grows
        # with z, which couples the slope into the determinant; beyond its
        # faces it is held, and those terms drop.
        a = 1.0 - lever * w_xx
        b = 1.0 - lever * w_yy
        det = a * b - (lever * w_xy) ** 2
        within = (z.abs() <= half_depth).to(z.dtype)
        det = det + within * (a * w_y**2 + b * w_x**2 + 2.0 * lever * w_xy * w_x * w_y)
        source = torch.stack([x + ox, y + oy, z], dim=-1) / voxel_size
        return source, 1.0 / det

    return warp
