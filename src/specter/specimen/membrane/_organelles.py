"""
Organelle kinds: recipes that compose the primitives into a membrane system.

A kind is a short recipe over `_primitives`. Its settings are the
quantities a tomogram shows (a radius, a length, a lumen width), each a
scalar or a ``[low, high]`` range drawn once per membrane. `plan_organelle`
draws them and reports the extent the result can reach, which is known
before any geometry is built; `OrganellePlan.build` then composes the
solid on a `Grid` centred on the origin. The remaining randomness
(undulation, a tubule's path, a cisterna's outline) is drawn at build time,
so a build that fails its reach check is redrawn without changing the
membrane's size.

All lengths are in Å and all coordinates physical ``(x, y, z)``, with the
organelle centred on the origin and its thin direction along ``z``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ...config import parse_scalar_or_range
from ...options import MembraneKind
from ._geometry import Grid, regularise
from ._primitives import (
    ellipsoid,
    nested,
    random_walk,
    rot_to,
    sheet,
    subtract,
    tube,
    undulate,
    union,
    wobbly,
)

Build = Callable[[Grid, np.random.Generator], list[torch.Tensor]]

# The settings of each kind and their defaults. A two-element value is a
# range a membrane's own value is drawn from uniformly.
ORGANELLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "vesicle": {
        "radius": [200.0, 340.0],
        "elongation": [1.0, 1.5],
        "undulation": [5.0, 45.0],
        "layers": 1,
    },
    "tubules": {
        "radius": [110.0, 170.0],
        "length": [1200.0, 1800.0],
        "branches": [2, 4],
    },
    "cisterna": {
        "diameter": [800.0, 1120.0],
        "lumen_width": [100.0, 160.0],
        "fenestrations": [0, 1],
    },
    "mitochondrion": {
        "radius": [470.0, 560.0],
        "length": [900.0, 1440.0],
        "crista_spacing": [230.0, 330.0],
        "membrane_gap": 130.0,
    },
}

# Settings drawn as whole numbers.
_INTEGER_FIELDS = frozenset({"layers", "branches", "fenestrations"})
# Smallest value each setting may take, A (or a count).
_MINIMUM: dict[str, float] = {
    "radius": 60.0,
    "elongation": 1.0,
    "undulation": 0.0,
    "layers": 1,
    "length": 300.0,
    "branches": 1,
    "diameter": 300.0,
    "lumen_width": 70.0,
    "fenestrations": 0,
    "crista_spacing": 180.0,
    "membrane_gap": 90.0,
}
# Larger minima of one kind: a mitochondrion's matrix must hold a crista.
_KIND_MINIMUM: dict[str, dict[str, float]] = {"mitochondrion": {"radius": 300.0}}
# The settings that set an organelle's overall size, shrunk to fit a box.
_SIZE_FIELDS = frozenset({"radius", "length", "diameter"})
# Distance between the mid-surfaces of a multilamellar vesicle's layers, A.
_LAYER_SPACING = 90.0
# Step of a tubule's path between control points, A.
_TUBULE_STEP = 300.0


@dataclass(frozen=True)
class OrganellePlan:
    """
    One membrane system, drawn but not yet built.

    Parameters
    ----------
    kind : str
        The organelle kind.
    params : dict
        The value drawn for each of the kind's settings.
    half_extent : numpy.ndarray
        Largest ``|x|``, ``|y|`` and ``|z|`` the membrane can reach, Å.
    build : callable
        ``build(grid, rng)`` returns the signed distance of each membrane
        mid-surface on `grid`, outermost first.
    bounding_radius : float
        Radius of a sphere about the origin that holds the membrane, Å.
    """

    kind: str
    params: dict[str, float]
    half_extent: np.ndarray
    build: Build
    bounding_radius: float


def _minimum(kind: str, name: str) -> float:
    """Smallest value setting `name` of `kind` may take."""
    return _KIND_MINIMUM.get(kind, {}).get(name, _MINIMUM[name])


def _draw(
    kind: str, fields: Mapping[str, Any], rng: np.random.Generator
) -> dict[str, float]:
    """Draw one value per setting of `kind`, validating names and bounds."""
    defaults = ORGANELLE_DEFAULTS[kind]
    unknown = sorted(set(fields) - set(defaults))
    if unknown:
        raise ValueError(
            f"membrane kind {kind!r} has no setting {unknown[0]!r}; "
            f"its settings are {sorted(defaults)}"
        )
    out: dict[str, float] = {}
    for name, default in defaults.items():
        value = fields.get(name, default)
        lo, hi = parse_scalar_or_range(default if value is None else value)
        if hi < lo:
            raise ValueError(f"membrane {name} range [{lo}, {hi}] is descending")
        minimum = _minimum(kind, name)
        if lo < minimum:
            raise ValueError(
                f"{kind} membrane {name} = {lo:g} is below the smallest "
                f"supported value, {minimum:g}"
            )
        if name in _INTEGER_FIELDS:
            out[name] = float(rng.integers(int(round(lo)), int(round(hi)) + 1))
        else:
            out[name] = float(rng.uniform(lo, hi))
    return out


def _plan_vesicle(p: dict[str, float], rng: np.random.Generator) -> OrganellePlan:
    # A prolate spheroid of the drawn mean radius, randomly oriented.
    e = p["elongation"]
    radii = p["radius"] * np.array([e ** (2 / 3), e ** (-1 / 3), e ** (-1 / 3)])
    rot = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    amplitude = min(p["undulation"], 0.15 * float(radii.min()))
    layers = int(p["layers"])

    def build(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
        phi = ellipsoid(grid, (0, 0, 0), tuple(radii), rot)
        if amplitude > 0:
            phi = undulate(phi, grid, rng, amplitude, float(rng.uniform(350, 700)))
        surfaces = [regularise(phi, grid)]
        for k in range(1, layers):
            # Stop once the next layer would no longer hold a lumen.
            if float(radii.min()) - k * _LAYER_SPACING < 2 * grid.r_min:
                break
            surfaces.append(nested(surfaces[0], grid, k * _LAYER_SPACING))
        return surfaces

    reach = float(np.sqrt(((rot * radii) ** 2).sum(1)).max()) + 3 * amplitude + 20
    return OrganellePlan("vesicle", p, np.full(3, reach), build, reach)


def _plan_tubules(p: dict[str, float], rng: np.random.Generator) -> OrganellePlan:
    radius, length = p["radius"], p["length"]
    branches = int(p["branches"])
    steps = max(2, int(round(length / _TUBULE_STEP)))
    # The paths reflect off this box, flat along z as a lamella is.
    bound = np.array([0.4 * length, 0.4 * length, max(0.2 * length, radius)])

    def build(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
        start = rng.uniform(-0.5, 0.5, 3) * bound
        parts = []
        for _ in range(branches):
            ctrl = random_walk(rng, start, _TUBULE_STEP, steps, 0.8, -bound, bound)
            parts.append(tube(grid, ctrl, radius))
            if rng.random() < 0.5:
                start = ctrl[int(rng.integers(1, len(ctrl) - 1))]
        return [regularise(union(*parts), grid)]

    half_extent = bound + radius + 40
    return OrganellePlan(
        "tubules", p, half_extent, build, float(np.linalg.norm(half_extent))
    )


def _plan_cisterna(p: dict[str, float], rng: np.random.Generator) -> OrganellePlan:
    r, half = p["diameter"] / 2, p["lumen_width"] / 2
    n_holes = int(p["fenestrations"])
    sag = 0.3 * r

    def build(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
        k = np.clip(rng.normal(size=3), -1.5, 1.5)

        def bend(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            bowl = (k[0] * u**2 + k[1] * v**2) * sag / (3 * r**2)
            return np.asarray(bowl + 0.1 * sag * np.sin(2 * k[2] * u / r + 1))

        holes = [
            (*rng.uniform(-0.45 * r, 0.45 * r, 2), float(rng.uniform(130, 180)))
            for _ in range(n_holes)
        ]
        rot = rot_to([rng.normal() * 0.2, rng.normal() * 0.2, 1])
        outline = wobbly(r, rng, amp=0.22)
        phi = sheet(grid, (0, 0, 0), rot, outline, half, bend, holes)
        return [regularise(phi, grid)]

    lateral = 1.3 * r + half + 20
    half_extent = np.array([lateral, lateral, 0.2 * lateral + 0.6 * sag + half + 30])
    return OrganellePlan(
        "cisterna", p, half_extent, build, float(np.hypot(lateral, half_extent[2]))
    )


def _plan_mitochondrion(p: dict[str, float], rng: np.random.Generator) -> OrganellePlan:
    radius, half_length = p["radius"], p["length"] / 2
    spacing, gap = p["crista_spacing"], p["membrane_gap"]
    azimuth = float(rng.uniform(0, 2 * math.pi))
    # Long axis in the xy plane, so the organelle lies in the lamella.
    frame = Rotation.from_euler("z", azimuth).as_matrix()
    crista_radius = radius - gap - 2 * 30.0

    def build(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
        bow = np.clip(rng.normal(size=2), -1.5, 1.5) * 90
        ctrl = np.array(
            [[-half_length, 0, 0], [0, bow[0], bow[1]], [half_length, 0, 0]]
        )
        body = tube(grid, ctrl @ frame.T, radius)
        outer = regularise(undulate(body, grid, rng, 20.0, 600.0), grid)
        parts = []
        for x in np.arange(-half_length + 60, half_length, spacing):
            tilt = rot_to([1, rng.normal() * 0.2, rng.normal() * 0.2])
            ang = rng.uniform(0, 2 * np.pi)
            e = np.array([math.cos(ang), math.sin(ang), 0])
            ctr = np.array([x, 0, 0]) + tilt @ (e * rng.uniform(60, 160))
            lamella = wobbly(max(crista_radius - rng.uniform(0, 70), 90), rng, amp=0.15)
            parts.append(sheet(grid, frame @ ctr, frame @ tilt, lamella, 72.0))
            # The crista junction: a tube from the lamella to the boundary.
            junction = [ctr + tilt @ (e * f * radius) for f in (0.25, 0.75, 1.3)]
            parts.append(tube(grid, np.array(junction) @ frame.T, 78.0))
        boundary = outer + gap
        if parts:
            boundary = subtract(boundary, union(*parts))
        return [outer, regularise(boundary, grid)]

    # A capsule: the body's radius, its bow and its undulation about the axis.
    across = radius + 135 + 60
    c, s = abs(math.cos(azimuth)), abs(math.sin(azimuth))
    half_extent = np.array([half_length * c + across, half_length * s + across, across])
    return OrganellePlan("mitochondrion", p, half_extent, build, half_length + across)


_PLANNERS: dict[
    str, Callable[[dict[str, float], np.random.Generator], OrganellePlan]
] = {
    "vesicle": _plan_vesicle,
    "tubules": _plan_tubules,
    "cisterna": _plan_cisterna,
    "mitochondrion": _plan_mitochondrion,
}


def plan_organelle(
    kind: MembraneKind | str,
    rng: np.random.Generator,
    max_radius: float | None = None,
    **fields: Any,
) -> OrganellePlan:
    """
    Draw one membrane system of an organelle kind.

    Parameters
    ----------
    kind : str
        ``"vesicle"``, ``"tubules"``, ``"cisterna"`` or
        ``"mitochondrion"``.
    rng : numpy.random.Generator
        Source of the draw.
    max_radius : float, optional
        Largest bounding radius the organelle may have, Å. A draw that
        exceeds it has the size settings left at their defaults (radius,
        length, diameter) scaled down until it fits or they reach their
        minima; a setting the caller gave is never changed. Default None:
        no limit.
    **fields
        The kind's settings (`ORGANELLE_DEFAULTS` lists them), each a scalar
        or a ``[low, high]`` range, Å or a count. An omitted setting takes
        its default.

    Returns
    -------
    OrganellePlan
        The drawn values, the extent the membrane can reach and the
        function that builds it.

    Raises
    ------
    ValueError
        If `kind` or a setting is unknown, or a value is below what the
        geometry supports.
    """
    if kind not in _PLANNERS:
        raise ValueError(
            f"unknown membrane kind {kind!r}; choose from {sorted(_PLANNERS)}"
        )
    params = _draw(kind, fields, rng)
    # The planner's own draws (an orientation) are repeated for every size
    # tried, so shrinking changes the size alone.
    sub_seed = int(rng.integers(1 << 62))
    plan = _PLANNERS[kind](dict(params), np.random.default_rng(sub_seed))
    free = [n for n in params if n in _SIZE_FIELDS and fields.get(n) is None]
    if max_radius is None or not free:
        return plan
    for _ in range(8):
        if plan.bounding_radius <= max_radius:
            break
        factor = 0.97 * max_radius / plan.bounding_radius
        shrunk = {n: max(params[n] * factor, _minimum(kind, n)) for n in free}
        if all(shrunk[n] == params[n] for n in free):
            break
        params.update(shrunk)
        plan = _PLANNERS[kind](dict(params), np.random.default_rng(sub_seed))
    return plan


__all__ = ["ORGANELLE_DEFAULTS", "OrganellePlan", "plan_organelle"]
