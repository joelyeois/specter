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
bending strain at the faces, `Deflection.max_gradient`.

The kinematics are linear in that gradient, which leaves an error of order
its square: a material line normal to the film keeps its length to first
order only. `settings.MAX_DEFORMATION_GRADIENT` bounds it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..settings import MAX_DEFORMATION_GRADIENT, Deformation

__all__ = [
    "Deflection",
    "DomeDeflection",
    "RandomDeflection",
    "Warp",
    "burst_fraction",
    "deflection_field",
    "displace",
    "material_rotation",
    "specimen_warp",
]

#: Maps sample points ``(..., 3)``, ``(x, y, z)`` in voxels from the volume's
#: centre, to the points of the undeformed volume they read and the weight
#: each value read is multiplied by.
Warp = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

#: How closely the pull-back is solved, in voxels.
_PULL_BACK_TOLERANCE = 1e-3

#: Random Fourier features summed for a random deflection: enough that the
#: field is Gaussian to a few percent in its one-point statistics.
_RANDOM_MODES = 512

#: Grid nodes per correlation length the random deflection is tabulated at.
#: Bicubic Hermite interpolation between them is accurate to ~(1/16)^4 of the
#: field.
_NODES_PER_CORRELATION_LENGTH = 16

#: Largest tabulation grid, per axis; a field far wider than its
#: correlation length is tabulated more coarsely rather than without bound.
_MAX_NODES = 513

#: Largest dense table a random deflection is read from, per axis.
_MAX_DENSE_NODES = 2049


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


class Deflection:
    """
    A midplane deflection ``w(x, y)`` once the burst is over.

    Positions are ``(x, y)`` in Angstrom from the volume's centre, and the
    deflection is in Angstrom. Subclasses give the deflection, its slope and
    its curvature, and bounds on each over the specimen.

    Attributes
    ----------
    max_height : float
        Largest ``|w|``, in Angstrom.
    max_slope : float
        Largest ``|grad w|``.
    max_curvature : float
        Largest eigenvalue magnitude of the Hessian of ``w``, in 1/Angstrom.
    """

    max_height: float
    max_slope: float
    max_curvature: float

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(w, dw/dx, dw/dy)`` at `x`, `y`."""
        raise NotImplementedError

    def curvature(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(d2w/dx2, d2w/dxdy, d2w/dy2)`` at `x`, `y`."""
        raise NotImplementedError

    def max_gradient(self, half_depth: float) -> float:
        """
        A bound on the displacement gradient over a specimen `half_depth`
        Angstrom either side of its midplane: the slope plus the bending
        strain at its faces.
        """
        return self.max_slope + half_depth * self.max_curvature

    def check_depth(self, half_depth: float) -> None:
        """
        Refuse a bend the thin-plate kinematics cannot carry through a
        specimen `half_depth` Angstrom either side of its midplane.

        Raises
        ------
        ValueError
            If `max_gradient` exceeds `settings.MAX_DEFORMATION_GRADIENT`.
        """
        gradient = self.max_gradient(half_depth)
        if gradient > MAX_DEFORMATION_GRADIENT:
            raise ValueError(
                f"this bend through a specimen {2 * half_depth:g} A deep gives "
                f"a displacement gradient of up to {gradient:.3f} (slope "
                f"{self.max_slope:.3f}, face strain "
                f"{half_depth * self.max_curvature:.3f}); the thin-plate "
                f"kinematics hold to {MAX_DEFORMATION_GRADIENT}, and measured "
                "domes over thin ice stay below 0.07"
            )


class DomeDeflection(Deflection):
    """
    A clamped circular film's deflection.

    ``w = height * (1 - r^2 / radius^2)^2`` for ``r <= radius`` and zero
    outside, ``r`` measured from the hole's centre at `offset`: the
    deflection of a circular plate clamped at its rim, so the film meets the
    support with no slope and the displacement is continuous across the rim.

    Parameters
    ----------
    height : float
        Central deflection in Angstrom.
    radius : float
        Hole radius in Angstrom.
    offset : tuple of float
        The hole's centre ``(x, y)`` from the volume's centre, in Angstrom.
    """

    def __init__(
        self, height: float, radius: float, offset: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        self.height = height
        self.radius = radius
        self.offset = offset
        self.max_height = abs(height)
        # |dw/dr| peaks at r = radius / sqrt(3); the radial curvature at the rim.
        self.max_slope = 8.0 * abs(height) / (3.0 * math.sqrt(3.0) * radius)
        self.max_curvature = 8.0 * abs(height) / radius**2

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x - self.offset[0]
        y = y - self.offset[1]
        a2 = self.radius * self.radius
        q = (1.0 - (x * x + y * y) / a2).clamp(min=0.0)
        c = (-4.0 * self.height / a2) * q
        return self.height * q * q, c * x, c * y

    def curvature(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x - self.offset[0]
        y = y - self.offset[1]
        a2 = self.radius * self.radius
        q = 1.0 - (x * x + y * y) / a2
        inside = (q >= 0.0).to(x.dtype)
        c = (-4.0 * self.height / a2) * inside
        return (
            c * (q - 2.0 * x * x / a2),
            -2.0 * c * x * y / a2,
            c * (q - 2.0 * y * y / a2),
        )


class RandomDeflection(Deflection):
    """
    A smooth random deflection: a stationary Gaussian process with a
    squared-exponential covariance.

    ``<w(r) w(r')> = rms^2 exp(-|r - r'|^2 / (2 correlation_length^2))``,
    drawn as a sum of random Fourier features and
    tabulated, with its exact derivatives, on a grid over the specimen's
    footprint that bicubic Hermite interpolation reads.

    Parameters
    ----------
    rms : float
        Root-mean-square deflection in Angstrom.
    correlation_length : float
        In Angstrom.
    half_width : float
        Half the side of the square footprint the field is tabulated over,
        in Angstrom, centred on the volume; beyond it the edge values hold.
    resolution : float
        Spacing in Angstrom of the dense table the field is read from,
        usually the voxel size.
    generator : torch.Generator, optional
        The draw's source; the global stream when None.

    Notes
    -----
    The field is evaluated exactly on a coarse grid of
    `_NODES_PER_CORRELATION_LENGTH` nodes per correlation length, then
    interpolated by bicubic Hermite onto a dense table at `resolution`
    (or a sixty-fourth of the correlation length, if finer), which every
    tilt reads bilinearly in one fused `grid_sample`. Bilinear reads of a
    table that fine keep the slope consistent with the deflection's own
    derivative to ``resolution * |d2w|``, which bounds the error of the
    weight `specimen_warp` applies.
    """

    def __init__(
        self,
        rms: float,
        correlation_length: float,
        half_width: float,
        resolution: float,
        generator: torch.Generator | None = None,
    ) -> None:
        f64 = torch.float64
        k = torch.randn(_RANDOM_MODES, 2, generator=generator, dtype=f64)
        k /= correlation_length
        phase = torch.rand(_RANDOM_MODES, generator=generator, dtype=f64)
        phase *= 2.0 * math.pi
        amplitude = rms * math.sqrt(2.0 / _RANDOM_MODES)

        spacing = max(
            correlation_length / _NODES_PER_CORRELATION_LENGTH,
            2.0 * half_width / (_MAX_NODES - 1),
        )
        n = int(math.ceil(2.0 * half_width / spacing)) + 1
        self.half_width = (n - 1) * spacing / 2.0
        axis = torch.linspace(-self.half_width, self.half_width, n, dtype=f64)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        nodes = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

        # w and its derivatives at every node, exactly, a chunk at a time.
        values = torch.empty(6, n * n, dtype=f64)
        kx, ky = k[:, 0], k[:, 1]
        for start in range(0, n * n, 4096):
            chunk = nodes[start : start + 4096]
            arg = chunk @ k.T + phase
            c, s = amplitude * torch.cos(arg), amplitude * torch.sin(arg)
            values[:, start : start + 4096] = torch.stack(
                [
                    c.sum(-1),
                    -(s * kx).sum(-1),
                    -(s * ky).sum(-1),
                    -(c * kx * kx).sum(-1),
                    -(c * kx * ky).sum(-1),
                    -(c * ky * ky).sum(-1),
                ]
            )
        w, w_x, w_y, w_xx, w_xy, w_yy = values
        self.max_height = float(w.abs().max())
        self.max_slope = float(torch.sqrt(w_x**2 + w_y**2).max())
        mean = 0.5 * (w_xx + w_yy)
        spread = torch.sqrt((0.5 * (w_xx - w_yy)) ** 2 + w_xy**2)
        self.max_curvature = float((mean.abs() + spread).max())
        # What the Hermite interpolation reads: w, w_x, w_y, w_xy.
        self.table = values[[0, 1, 2, 4]].contiguous()
        self.nodes = n
        self.spacing = spacing

        fine = max(
            min(resolution, correlation_length / 64.0),
            2.0 * self.half_width / (_MAX_DENSE_NODES - 1),
        )
        m = int(math.ceil(2.0 * self.half_width / fine)) + 1
        axis = torch.linspace(-self.half_width, self.half_width, m, dtype=f64)
        dense = torch.empty(6, m, m, dtype=f64)
        rows = max(1, 2**20 // m)
        for start in range(0, m, rows):
            gy, gx = torch.meshgrid(axis[start : start + rows], axis, indexing="ij")
            dense[:, start : start + rows] = torch.stack(self._hermite(gx, gy))
        self.dense = dense[None]
        self._read: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _hermite(self, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        """
        ``w`` and its first and second derivatives at `x`, `y`.

        Bicubic Hermite interpolation of the nodes' exact ``w``, ``w_x``,
        ``w_y`` and ``w_xy``, differentiated analytically, so the slope and
        curvature returned are those of the deflection returned. (Keys'
        cubic convolution, `grid_sample`'s bicubic mode, is not: with its
        ``a = -0.75`` its slope at a node is 1.5 times the central
        difference.)
        """
        table = self.table.to(x.device, x.dtype)
        n, h = self.nodes, self.spacing
        u = ((x + self.half_width) / h).clamp(0.0, n - 1.0)
        v = ((y + self.half_width) / h).clamp(0.0, n - 1.0)
        i = u.floor().clamp(max=n - 2)
        j = v.floor().clamp(max=n - 2)
        t, s = u - i, v - j
        index = (j * n + i).long().reshape(-1)

        def basis(t: torch.Tensor) -> list[list[torch.Tensor]]:
            # Value, slope and value-at-the-other-end Hermite functions per
            # end, with their first and second derivatives in t.
            t2, t3 = t * t, t * t * t
            return [
                [2 * t3 - 3 * t2 + 1, 6 * t2 - 6 * t, 12 * t - 6],
                [t3 - 2 * t2 + t, 3 * t2 - 4 * t + 1, 6 * t - 4],
                [-2 * t3 + 3 * t2, -6 * t2 + 6 * t, -12 * t + 6],
                [t3 - t2, 3 * t2 - 2 * t, 6 * t - 2],
            ]

        bt, bs = basis(t.reshape(-1)), basis(s.reshape(-1))
        # Per end a in t: value function bt[2a], slope function bt[2a + 1].
        out = [torch.zeros_like(index, dtype=x.dtype) for _ in range(6)]
        derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        for a in (0, 1):
            for b in (0, 1):
                f, f_x, f_y, f_xy = table[:, index + b * n + a]
                for k, (dt, ds) in enumerate(derivatives):
                    pt, mt = bt[2 * a][dt], bt[2 * a + 1][dt] * h
                    ps, ms = bs[2 * b][ds], bs[2 * b + 1][ds] * h
                    term = f * pt * ps + f_x * mt * ps + f_y * pt * ms
                    out[k] = out[k] + (term + f_xy * mt * ms) / h ** (dt + ds)
        return [o.reshape(x.shape) for o in out]

    def _sample(
        self, x: torch.Tensor, y: torch.Tensor, channels: slice
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Three of the dense table's fields at `x`, `y`, read bilinearly."""
        key = (x.device, x.dtype)
        if key not in self._read:
            self._read[key] = self.dense.to(x.device, x.dtype)
        table = self._read[key][:, channels]
        grid = torch.stack([x, y], dim=-1).reshape(1, 1, -1, 2) / self.half_width
        sampled = torch.nn.functional.grid_sample(
            table, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).reshape(3, *x.shape)
        return sampled[0], sampled[1], sampled[2]

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._sample(x, y, slice(0, 3))

    def curvature(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._sample(x, y, slice(3, 6))


def deflection_field(
    deformation: Deformation,
    half_width: float,
    resolution: float,
    generator: torch.Generator | None = None,
) -> Deflection | None:
    """
    The deflection a `Deformation` describes, over a specimen.

    Parameters
    ----------
    deformation : Deformation
        The bend.
    half_width : float
        Half the side of the specimen's square footprint, in Angstrom; a
        random deflection is drawn over it.
    resolution : float
        Spacing in Angstrom a random deflection is tabulated at, usually
        the voxel size.
    generator : torch.Generator, optional
        Source of a random deflection's draw.

    Returns
    -------
    Deflection or None
        None for a rigid specimen.
    """
    if deformation.model is None or deformation.height is None:
        return None
    if deformation.height == 0.0:
        return None
    if deformation.model == "dome":
        return DomeDeflection(
            deformation.height, deformation.hole_radius, deformation.hole_offset
        )
    return RandomDeflection(
        abs(deformation.height),
        deformation.correlation_length,
        half_width,
        resolution,
        generator,
    )


def _iterations(lipschitz: float, displacement: float) -> int:
    """Fixed-point steps contracting `displacement` voxels to the tolerance."""
    if displacement <= _PULL_BACK_TOLERANCE or lipschitz <= 0.0:
        return 1
    target = _PULL_BACK_TOLERANCE * (1.0 - lipschitz) / displacement
    return max(1, math.ceil(math.log(target) / math.log(lipschitz)))


def displace(
    deflection: Deflection, fraction: float, points: torch.Tensor, half_depth: float
) -> torch.Tensor:
    """
    Where the bend carries specimen points: the forward map ``X + u(X)``.

    Parameters
    ----------
    deflection : Deflection
        The full bend.
    fraction : float
        The fraction of it reached (`burst_fraction`).
    points : torch.Tensor
        ``(..., 3)``, ``(x, y, z)`` in Angstrom from the volume's centre,
        ``z`` from its midplane.
    half_depth : float
        Half the volume's depth in Angstrom, past which the lever is held.

    Returns
    -------
    torch.Tensor
        The displaced points, shaped and in units like `points`.
    """
    x, y, z = points.unbind(-1)
    w, w_x, w_y = deflection(x, y)
    lever = fraction * z.clamp(-half_depth, half_depth)
    return torch.stack([x - lever * w_x, y - lever * w_y, z + fraction * w], dim=-1)


def material_rotation(
    deflection: Deflection, fraction: float, points: torch.Tensor
) -> torch.Tensor:
    """
    How the bend turns the material at specimen points.

    The rotation part of ``I + grad u``: about ``x`` by ``dw/dy`` and about
    ``y`` by ``-dw/dx``, so a particle on the film turns with its local
    slope. The plate kinematics give the angle to first order, as they give
    the displacement.

    Parameters
    ----------
    deflection : Deflection
        The full bend.
    fraction : float
        The fraction of it reached.
    points : torch.Tensor
        ``(..., 3)`` positions in the undeformed specimen, in Angstrom.

    Returns
    -------
    torch.Tensor
        ``(..., 3, 3)`` rotation matrices, applied to a vector on the left.
    """
    import roma

    x, y, _ = points.unbind(-1)
    _, w_x, w_y = deflection(x, y)
    rotvec = fraction * torch.stack([w_y, -w_x, torch.zeros_like(w_x)], dim=-1)
    return roma.rotvec_to_rotmat(rotvec)


def specimen_warp(
    deflection: Deflection | None,
    fraction: float,
    voxel_size: float,
    half_depth: float,
) -> Warp | None:
    """
    The map a tilt samples its volume through, at a fraction of the bend.

    Parameters
    ----------
    deflection : Deflection or None
        The full bend; None for a rigid specimen.
    fraction : float
        The fraction of it reached when the tilt was recorded
        (`burst_fraction` of its mid-exposure dose).
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
    if deflection is None or fraction == 0.0:
        return None
    # The map's Lipschitz constant and largest displacement scale with the
    # deflection, so a bend still growing needs fewer steps.
    steps = _iterations(
        deflection.max_gradient(half_depth) * fraction,
        (deflection.max_height + half_depth * deflection.max_slope)
        * fraction
        / voxel_size,
    )

    def warp(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sx = points[..., 0] * voxel_size
        sy = points[..., 1] * voxel_size
        sz = points[..., 2] * voxel_size
        # Solve s = X + u(X) for X by fixed-point iteration from X = s.
        x, y, z = sx, sy, sz
        for _ in range(steps):
            w, w_x, w_y = deflection(x, y)
            lever = fraction * z.clamp(-half_depth, half_depth)
            x = sx + lever * w_x
            y = sy + lever * w_y
            z = sz - fraction * w
        w, w_x, w_y = deflection(x, y)
        w_xx, w_xy, w_yy = deflection.curvature(x, y)
        w_x, w_y = fraction * w_x, fraction * w_y
        w_xx, w_xy, w_yy = fraction * w_xx, fraction * w_xy, fraction * w_yy
        lever = z.clamp(-half_depth, half_depth)
        # det(I + grad u), in closed form. Inside the volume the lever grows
        # with z, which couples the slope into the determinant; beyond its
        # faces it is held, and those terms drop.
        a = 1.0 - lever * w_xx
        b = 1.0 - lever * w_yy
        det = a * b - (lever * w_xy) ** 2
        within = (z.abs() <= half_depth).to(z.dtype)
        det = det + within * (a * w_y**2 + b * w_x**2 + 2.0 * lever * w_xy * w_x * w_y)
        source = torch.stack([x, y, z], dim=-1) / voxel_size
        return source, 1.0 / det

    return warp
