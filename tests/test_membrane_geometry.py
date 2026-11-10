"""The reach guarantee of the membrane geometry: `regularise` and `reach`."""

import pytest
import torch

from specter.specimen.membrane._geometry import (
    Grid,
    ReachError,
    reach,
    regularise,
    surface_points,
)
from specter.specimen.membrane._primitives import ball, tube, union

# What a bilayer needs on each side of its mid-surface for the two leaflets
# not to cross: `regularise`'s default `min_reach`.
_MIN_REACH = 25.0


def _two_balls(grid: Grid) -> torch.Tensor:
    """Two overlapping balls: their union has a concave crease."""
    return union(
        ball(grid, (140.0, 150.0, 150.0), 90.0), ball(grid, (260.0, 150.0, 150.0), 90.0)
    )


def test_grid_layout() -> None:
    """Nodes are `h` apart, indexed (x, y, z), and the box sits inside them."""
    grid = Grid((400.0, 300.0, 200.0), lo=(-200.0, -150.0, -100.0))
    x, y, z = (a.flatten() for a in grid.axes())
    assert grid.shape == (x.numel(), y.numel(), z.numel())
    assert grid.n_nodes == x.numel() * y.numel() * z.numel()
    for axis, low, extent in (
        (x, -200.0, 400.0),
        (y, -150.0, 300.0),
        (z, -100.0, 200.0),
    ):
        assert torch.allclose(axis.diff(), torch.full_like(axis.diff(), grid.h))
        assert float(axis[0]) <= low and float(axis[-1]) >= low + extent
    assert tuple(float(c) for c in grid.centre) == pytest.approx((0.0, 0.0, 0.0))
    assert float(x[0]) == pytest.approx(float(grid.origin[0]))


def test_reach_of_a_ball_is_not_limited_by_its_surface() -> None:
    """A sphere's medial axis is its centre: the reach inside is the radius
    (capped at the search length), and outside nothing is ever met."""
    grid = Grid((300.0, 300.0, 300.0))
    points, normals, outer, inner = reach(
        ball(grid, (150.0, 150.0, 150.0), 100.0), grid
    )
    assert points.shape[0] > 100 and points.shape == normals.shape
    radius = (points - torch.tensor([150.0, 150.0, 150.0])).norm(dim=1)
    assert float((radius - 100.0).abs().max()) < 1.0
    # Unit outward normals.
    outward = (points - 150.0) / radius[:, None]
    assert float((normals * outward).sum(1).min()) > 0.99
    assert float(outer.min()) > 60.0
    assert float(inner.min()) > 60.0


def test_regularise_keeps_a_ball() -> None:
    """A shape that already has the reach comes back where it was: the
    surface of a 100 A ball stays within a fraction of the 10 A spacing."""
    grid = Grid((300.0, 300.0, 300.0))
    phi = regularise(ball(grid, (150.0, 150.0, 150.0), 100.0), grid)
    found = surface_points(phi, grid)
    assert found is not None
    radius = (found[1] - torch.tensor([150.0, 150.0, 150.0])).norm(dim=1)
    assert float(radius.mean()) == pytest.approx(100.0, abs=2.5)
    assert float((radius - radius.mean()).abs().max()) < 2.5
    # And the result is a signed distance: its centre is one radius deep.
    assert float(phi.min()) == pytest.approx(-float(radius.mean()), abs=5.0)


def test_regularise_removes_the_crease_between_two_balls() -> None:
    grid = Grid((400.0, 300.0, 300.0))
    raw = _two_balls(grid)
    _, _, raw_outer, raw_inner = reach(raw, grid)
    # The crease: on the outer side the normals cross almost at once.
    assert float(raw_outer.min()) < _MIN_REACH

    phi = regularise(raw, grid)
    _, _, outer, inner = reach(phi, grid)
    assert float(outer.min()) >= _MIN_REACH
    assert float(inner.min()) >= _MIN_REACH

    # The repair is local: both ball centres are still deep inside, and a
    # point well outside both is still outside.
    def index(point: tuple[float, float, float]) -> tuple[int, ...]:
        return tuple(
            int(round((c - float(o)) / grid.h)) for c, o in zip(point, grid.origin)
        )

    assert float(phi[index((140.0, 150.0, 150.0))]) < -60.0
    assert float(phi[index((260.0, 150.0, 150.0))]) < -60.0
    assert float(phi[index((200.0, 290.0, 150.0))]) > 30.0


def test_regularise_rounds_a_ball_and_tube_junction() -> None:
    """A tube leaving a ball meets it in a sharp concave ring."""
    grid = Grid((400.0, 300.0, 300.0))
    raw = union(
        ball(grid, (130.0, 150.0, 150.0), 90.0),
        tube(
            grid,
            [(130.0, 150.0, 150.0), (230.0, 150.0, 150.0), (320.0, 150.0, 150.0)],
            45.0,
        ),
    )
    _, _, raw_outer, _ = reach(raw, grid)
    assert float(raw_outer.min()) < _MIN_REACH

    _, _, outer, inner = reach(regularise(raw, grid), grid)
    assert float(outer.min()) >= _MIN_REACH
    assert float(inner.min()) >= _MIN_REACH


def test_regularise_refuses_what_it_was_not_allowed_to_repair() -> None:
    """With no repair rounds and no smoothing, the crease survives and the
    reach check reports it instead of handing back a self-crossing bilayer."""
    grid = Grid((400.0, 300.0, 300.0))
    with pytest.raises(ReachError, match="below 25.0 A"):
        regularise(_two_balls(grid), grid, rounds=0, r=5.0, smooth=0.0)
    assert issubclass(ReachError, ValueError)
