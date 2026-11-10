"""
Membrane shape primitives and the operations that combine them.

Every primitive returns the signed distance of a solid on a `Grid` (negative
inside); a membrane is the boundary of a solid after `regularise`. Three
skeletons cover the shapes membranes take: a point (`ball`, `ellipsoid`), a
curve (`tube`) and a surface patch (`sheet`, `cap`), each thickened by a
radius that may vary along it. `union`, `subtract`, `nested` and `undulate`
combine and perturb them. All coordinates are physical ``(x, y, z)`` in Å.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from ._geometry import (
    Grid,
    redistance,
    regularise,
    skeleton_solid,
    smooth_noise,
)

Outline = Callable[[float], float]


def curve(
    ctrl: np.ndarray | Sequence[Sequence[float]], spacing: float = 3.0
) -> np.ndarray:
    """
    Points along a natural cubic spline through control points.

    Parameters
    ----------
    ctrl : array-like
        Control points, shape ``(N, 3)``, Å.
    spacing : float, optional
        Approximate distance between returned points, Å. Default 3.0.

    Returns
    -------
    numpy.ndarray
        Shape ``(M, 3)``.
    """
    c = np.asarray(ctrl, dtype=float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    cs = CubicSpline(s, c, bc_type="natural")
    return np.asarray(cs(np.linspace(0, s[-1], int(s[-1] / spacing) + 2)))


def ball(grid: Grid, centre: Sequence[float], radius: float) -> torch.Tensor:
    """
    A solid sphere.

    Parameters
    ----------
    grid : Grid
        Working grid.
    centre : sequence of float
        Centre, Å.
    radius : float
        Radius, Å.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``.
    """
    d2 = sum((a - float(c)) ** 2 for a, c in zip(grid.axes(), centre))
    assert isinstance(d2, torch.Tensor)
    return d2.sqrt() - radius


def ellipsoid(
    grid: Grid,
    centre: Sequence[float],
    radii: Sequence[float],
    rot: np.ndarray | None = None,
) -> torch.Tensor:
    """
    A solid ellipsoid.

    Parameters
    ----------
    grid : Grid
        Working grid.
    centre : sequence of float
        Centre, Å.
    radii : sequence of float
        Semi-axes along the ellipsoid's own axes, Å.
    rot : numpy.ndarray, optional
        Rotation matrix whose columns are the ellipsoid's axes. Default the
        coordinate axes.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``.
    """
    d = [a - float(c) for a, c in zip(grid.axes(), centre)]
    if rot is not None:
        m = np.asarray(rot)
        d = [
            float(m[0, k]) * d[0] + float(m[1, k]) * d[1] + float(m[2, k]) * d[2]
            for k in range(3)
        ]
    q = (
        (d[0] / radii[0]) ** 2 + (d[1] / radii[1]) ** 2 + (d[2] / radii[2]) ** 2
    ).sqrt()
    return redistance((q - 1.0) * min(radii), grid)


def tube(
    grid: Grid,
    ctrl: np.ndarray | Sequence[Sequence[float]],
    radius: float | Callable[[np.ndarray], np.ndarray],
) -> torch.Tensor:
    """
    A solid of given radius around a spline.

    Parameters
    ----------
    grid : Grid
        Working grid.
    ctrl : array-like
        Control points of the centreline, shape ``(N, 3)``, Å.
    radius : float or callable
        Radius, Å, or a function of the arc fraction in ``[0, 1]`` returning
        it, which is how a tube is pearled or tapered.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``.
    """
    pts = curve(ctrl)
    r = radius(np.linspace(0, 1, len(pts))) if callable(radius) else radius
    return skeleton_solid(grid, pts, r)


def sheet(
    grid: Grid,
    centre: Sequence[float],
    rot: np.ndarray,
    outline: Outline,
    half: float | Callable[[np.ndarray], np.ndarray],
    height: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    holes: Sequence[Sequence[float]] = (),
    extent: float | None = None,
    spacing: float = 4.0,
) -> torch.Tensor:
    """
    A solid of given half-thickness around a surface patch.

    The patch is ``(u, v, height(u, v))`` for ``rho < outline(theta)`` in
    its own plane, minus circular holes, rotated by `rot` and moved to
    `centre`.

    Parameters
    ----------
    grid : Grid
        Working grid.
    centre : sequence of float
        Position of the patch's origin, Å.
    rot : numpy.ndarray
        Rotation matrix taking the patch's frame to the grid's.
    outline : callable
        Radius of the patch's edge as a function of the in-plane angle, Å.
    half : float or callable
        Half-thickness, Å, or a function of ``rho / outline(theta)``
        returning it, which is how a rim is thickened.
    height : callable, optional
        Out-of-plane displacement ``height(u, v)``, Å. Default flat.
    holes : sequence of (u, v, radius), optional
        Circular fenestrations in the patch's plane, Å.
    extent : float, optional
        Half-width of the sampled square, Å. Default 1.3 times the largest
        outline radius.
    spacing : float, optional
        Sample spacing on the patch, Å. Default 4.0.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``.
    """
    if extent is None:
        extent = 1.3 * max(outline(t) for t in np.linspace(0, 2 * np.pi, 90))
    a = np.arange(-extent, extent + spacing, spacing)
    u, v = np.meshgrid(a, a, indexing="ij")
    rho, th = np.hypot(u, v), np.arctan2(v, u)
    frac = rho / np.vectorize(outline)(th)
    keep = frac < 1
    for u0, v0, r0 in holes:
        keep &= np.hypot(u - u0, v - v0) > r0
    w = height(u, v) if height is not None else np.zeros_like(u)
    pts = np.stack([u[keep], v[keep], w[keep]], 1) @ np.asarray(rot).T
    r = half(frac[keep]) if callable(half) else half
    return skeleton_solid(grid, pts + np.asarray(centre, dtype=float), r)


def cap(
    grid: Grid,
    centre: Sequence[float],
    rot: np.ndarray,
    radius: float,
    polar_max: float,
    half: float,
    spacing: float = 4.0,
) -> torch.Tensor:
    """
    A solid around a spherical cap: a cup.

    Parameters
    ----------
    grid : Grid
        Working grid.
    centre : sequence of float
        Centre of the cap's sphere, Å.
    rot : numpy.ndarray
        Rotation matrix taking +z to the cap's pole.
    radius : float
        Radius of the cap's sphere, Å.
    polar_max : float
        The cap covers polar angles below this, radians.
    half : float
        Half-thickness, Å.
    spacing : float, optional
        Sample spacing on the cap, Å. Default 4.0.

    Returns
    -------
    torch.Tensor
        Signed distance, shape ``grid.shape``.
    """
    n = int(4 * math.pi * radius**2 / spacing**2)
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    ph = i * math.pi * (3 - math.sqrt(5))
    s = np.sqrt(1 - z**2)
    pts = radius * np.stack([s * np.cos(ph), s * np.sin(ph), z], 1)
    pts = pts[z > math.cos(polar_max)] @ np.asarray(rot).T
    return skeleton_solid(grid, pts + np.asarray(centre, dtype=float), half)


def half_space(
    grid: Grid,
    z0: float,
    height: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Everything below ``z0 + height(x, y)``: a cell under its plasma membrane.

    Parameters
    ----------
    grid : Grid
        Working grid.
    z0 : float
        Height of the surface, Å.
    height : callable, optional
        Displacement of the surface as a function of the grid's ``x`` and
        ``y`` tensors, Å. Default flat.

    Returns
    -------
    torch.Tensor
        Signed field, shape ``grid.shape``; a distance only where the
        surface is flat, so `regularise` it before use.
    """
    x, y, z = grid.axes()
    h = height(x, y) if height is not None else 0.0
    return (z - z0 - h).expand(grid.shape).contiguous()


def union(*phis: torch.Tensor) -> torch.Tensor:
    """Solid covered by any of the given solids."""
    out = phis[0]
    for p in phis[1:]:
        out = torch.minimum(out, p)
    return out


def subtract(a: torch.Tensor, b: torch.Tensor, gap: float = 0.0) -> torch.Tensor:
    """
    Solid `a` with `b`, grown by `gap`, removed.

    `b` must be a signed distance when `gap` is nonzero.
    """
    return torch.maximum(a, gap - b)


def undulate(
    phi: torch.Tensor,
    grid: Grid,
    rng: np.random.Generator,
    amplitude: float,
    wavelength: float,
) -> torch.Tensor:
    """
    Displace a solid's surface by a smooth random field.

    Parameters
    ----------
    phi : torch.Tensor
        Signed distance, shape ``grid.shape``.
    grid : Grid
        Working grid.
    rng : numpy.random.Generator
        Source of the field.
    amplitude : float
        Standard deviation of the displacement, Å.
    wavelength : float
        Typical wavelength of the displacement, Å.

    Returns
    -------
    torch.Tensor
        Displaced field; `regularise` it before use.
    """
    return phi + amplitude * smooth_noise(grid, wavelength, rng)


def nested(phi: torch.Tensor, grid: Grid, gap: float) -> torch.Tensor:
    """
    The regularised solid whose surface lies `gap` inside that of `phi`.

    `phi` must be a signed distance, which makes the offset exact.
    """
    return regularise(phi + gap, grid)


def circle(r: float) -> Outline:
    """A circular outline of radius `r` for `sheet`."""
    return lambda th: r


def wobbly(
    r: float, rng: np.random.Generator, modes: int = 4, amp: float = 0.18
) -> Outline:
    """
    An irregular closed outline of mean radius `r` for `sheet`.

    Parameters
    ----------
    r : float
        Mean radius, Å.
    rng : numpy.random.Generator
        Source of the mode amplitudes and phases.
    modes : int, optional
        Angular modes, starting from the second. Default 4.
    amp : float, optional
        Relative amplitude of the first mode. Default 0.18.

    Returns
    -------
    callable
        Outline radius as a function of the in-plane angle.
    """
    a = rng.normal(size=modes) * amp / np.arange(1, modes + 1)
    p = rng.uniform(0, 2 * np.pi, modes)

    def outline(th: float) -> float:
        return float(
            r * (1 + sum(a[m] * np.cos((m + 2) * th + p[m]) for m in range(modes)))
        )

    return outline


def rot_to(axis: Sequence[float]) -> np.ndarray:
    """Rotation matrix taking +z to `axis`."""
    v = np.asarray(axis, dtype=float)
    r, _ = Rotation.align_vectors([v / np.linalg.norm(v)], [[0, 0, 1]])
    return np.asarray(r.as_matrix())


def random_walk(
    rng: np.random.Generator,
    start: Sequence[float] | np.ndarray,
    step: float,
    n: int,
    persistence: float = 0.75,
    lo: Sequence[float] | np.ndarray | None = None,
    hi: Sequence[float] | np.ndarray | None = None,
    direction: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Control points of a persistent random walk, for `tube`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the walk.
    start : sequence of float
        First point, Å.
    step : float
        Distance between points, Å.
    n : int
        Steps taken.
    persistence : float, optional
        Fraction of each step's direction kept from the last, in
        ``[0, 1]``. Default 0.75.
    lo, hi : sequence of float, optional
        Corners of a box the walk reflects off, Å. Default unbounded.
    direction : sequence of float, optional
        Initial direction. Default random.

    Returns
    -------
    numpy.ndarray
        Shape ``(n + 1, 3)``.
    """
    d = rng.normal(size=3) if direction is None else np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    pts = [np.asarray(start, dtype=float)]
    for _ in range(n):
        d = persistence * d + (1 - persistence) * rng.normal(size=3)
        d /= np.linalg.norm(d)
        nxt = pts[-1] + step * d
        if lo is not None and hi is not None:
            for k in range(3):
                if not lo[k] < nxt[k] < hi[k]:
                    d[k] = -d[k]
            nxt = pts[-1] + step * d
        pts.append(nxt)
    return np.array(pts)


__all__ = [
    "ball",
    "cap",
    "circle",
    "curve",
    "ellipsoid",
    "half_space",
    "nested",
    "random_walk",
    "rot_to",
    "sheet",
    "subtract",
    "tube",
    "undulate",
    "union",
    "wobbly",
]
