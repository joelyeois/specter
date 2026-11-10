"""Signed distances of the membrane solids and their set operations."""

from collections.abc import Sequence

import numpy as np
import pytest
import torch

from specter.specimen.membrane._geometry import Grid
from specter.specimen.membrane._primitives import (
    ball,
    ellipsoid,
    half_space,
    nested,
    subtract,
    tube,
    union,
)


def _grid() -> Grid:
    """A 400 A box centred on the origin, whose nodes sit on multiples of 10 A."""
    return Grid((400.0, 400.0, 400.0), lo=(-200.0, -200.0, -200.0))


def _at(phi: torch.Tensor, grid: Grid, point: Sequence[float]) -> float:
    """`phi` at a physical point that is a grid node."""
    index = (torch.tensor(point, dtype=torch.float32) - grid.origin.cpu()) / grid.h
    assert torch.allclose(index, index.round(), atol=1e-4), "not a grid node"
    i, j, k = (int(v) for v in index.round().tolist())
    return float(phi[i, j, k])


def test_ball_is_the_distance_to_a_sphere() -> None:
    grid = _grid()
    phi = ball(grid, (20.0, 0.0, -10.0), 90.0)
    assert phi.shape == grid.shape
    assert _at(phi, grid, (20.0, 0.0, -10.0)) == pytest.approx(-90.0, abs=1e-3)
    assert _at(phi, grid, (110.0, 0.0, -10.0)) == pytest.approx(0.0, abs=1e-3)
    assert _at(phi, grid, (20.0, 150.0, -10.0)) == pytest.approx(60.0, abs=1e-3)
    assert _at(phi, grid, (20.0, 30.0, 30.0)) == pytest.approx(-40.0, abs=1e-3)


def test_ellipsoid_along_its_axes() -> None:
    grid = _grid()
    phi = ellipsoid(grid, (0.0, 0.0, 0.0), (120.0, 80.0, 80.0))
    assert _at(phi, grid, (0.0, 0.0, 0.0)) == pytest.approx(-80.0, abs=1.0)
    assert _at(phi, grid, (120.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1.0)
    assert _at(phi, grid, (150.0, 0.0, 0.0)) == pytest.approx(30.0, abs=1.5)
    assert _at(phi, grid, (0.0, 100.0, 0.0)) == pytest.approx(20.0, abs=1.5)
    assert _at(phi, grid, (0.0, 0.0, -80.0)) == pytest.approx(0.0, abs=1.0)


def test_ellipsoid_rotation_turns_its_long_axis() -> None:
    """The columns of `rot` are the ellipsoid's axes."""
    grid = _grid()
    # Long axis (the first semi-axis) along +y.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    phi = ellipsoid(grid, (0.0, 0.0, 0.0), (120.0, 80.0, 80.0), rot=rot)
    assert _at(phi, grid, (0.0, 120.0, 0.0)) == pytest.approx(0.0, abs=1.0)
    assert _at(phi, grid, (80.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1.0)
    assert _at(phi, grid, (120.0, 0.0, 0.0)) > 30.0


def test_tube_is_the_distance_to_its_centreline() -> None:
    grid = _grid()
    phi = tube(grid, [(-100.0, 0.0, 0.0), (0.0, 0.0, 0.0), (100.0, 0.0, 0.0)], 40.0)
    assert _at(phi, grid, (0.0, 0.0, 0.0)) == pytest.approx(-40.0, abs=1.0)
    assert _at(phi, grid, (50.0, 40.0, 0.0)) == pytest.approx(0.0, abs=1.0)
    assert _at(phi, grid, (0.0, 70.0, 0.0)) == pytest.approx(30.0, abs=1.0)
    assert _at(phi, grid, (-30.0, 0.0, 20.0)) == pytest.approx(-20.0, abs=1.0)
    # The ends are round: 40 A past the last control point is on the surface.
    assert _at(phi, grid, (140.0, 0.0, 0.0)) == pytest.approx(0.0, abs=3.0)
    assert _at(phi, grid, (180.0, 0.0, 0.0)) == pytest.approx(40.0, abs=3.0)


def test_tube_radius_can_vary_along_the_arc() -> None:
    grid = _grid()
    phi = tube(
        grid,
        [(-100.0, 0.0, 0.0), (0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
        lambda t: 30.0 + 30.0 * t,
    )
    # Thin at the start of the arc, thick at its end.
    assert _at(phi, grid, (-100.0, 40.0, 0.0)) > 0.0
    assert _at(phi, grid, (100.0, 40.0, 0.0)) < 0.0


def test_half_space_is_negative_below_its_plane() -> None:
    grid = _grid()
    phi = half_space(grid, 30.0)
    assert _at(phi, grid, (0.0, 0.0, 0.0)) == pytest.approx(-30.0, abs=1e-3)
    assert _at(phi, grid, (50.0, -70.0, 100.0)) == pytest.approx(70.0, abs=1e-3)


def test_union_and_subtract_signs() -> None:
    grid = _grid()
    left = ball(grid, (-60.0, 0.0, 0.0), 80.0)
    right = ball(grid, (60.0, 0.0, 0.0), 80.0)

    both = union(left, right)
    assert _at(both, grid, (-100.0, 0.0, 0.0)) < 0.0  # in the left ball only
    assert _at(both, grid, (100.0, 0.0, 0.0)) < 0.0  # in the right ball only
    assert _at(both, grid, (0.0, 0.0, 0.0)) < 0.0  # in both
    assert _at(both, grid, (0.0, 100.0, 0.0)) > 0.0  # in neither
    assert torch.equal(both, torch.minimum(left, right))

    cut = subtract(left, right)
    assert _at(cut, grid, (-100.0, 0.0, 0.0)) < 0.0  # left, not right: kept
    assert _at(cut, grid, (0.0, 0.0, 0.0)) > 0.0  # in the overlap: removed
    assert _at(cut, grid, (100.0, 0.0, 0.0)) > 0.0  # right only: never there

    # With a gap, the removed solid is grown: a point 10 A outside the right
    # ball, inside the left one, goes too.
    assert _at(cut, grid, (-30.0, 0.0, 0.0)) < 0.0
    assert _at(subtract(left, right, gap=20.0), grid, (-30.0, 0.0, 0.0)) > 0.0


def test_nested_lies_the_gap_inside() -> None:
    """A ball nested by 40 A is the concentric ball 40 A smaller."""
    grid = _grid()
    outer = ball(grid, (0.0, 0.0, 0.0), 130.0)
    inner = nested(outer, grid, 40.0)
    assert _at(inner, grid, (0.0, 0.0, 0.0)) < 0.0
    assert _at(inner, grid, (110.0, 0.0, 0.0)) > 0.0  # inside outer, outside inner
    # Everywhere inside the inner solid is inside the outer one.
    assert bool((outer[inner < 0] < 0).all())
    near = inner.abs() < 30.0
    assert float((inner - (outer + 40.0))[near].abs().max()) < 3.0
