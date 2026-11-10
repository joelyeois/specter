"""`plan_organelle`: what a membrane kind draws, refuses, and builds."""

from typing import Any

import numpy as np
import pytest
import torch

from specter.specimen.membrane._geometry import Grid, reach
from specter.specimen.membrane._organelles import (
    ORGANELLE_DEFAULTS,
    OrganellePlan,
    plan_organelle,
)

# The smallest organelle of each kind, so a build stays a second or two.
_SMALLEST: dict[str, dict[str, Any]] = {
    "vesicle": dict(radius=100.0, elongation=1.2, undulation=5.0),
    "tubules": dict(radius=60.0, length=300.0, branches=1),
    "cisterna": dict(diameter=300.0, lumen_width=70.0, fenestrations=0),
    "mitochondrion": dict(
        radius=300.0, length=300.0, crista_spacing=180.0, membrane_gap=90.0
    ),
}
# A mitochondrion's grid is ~2 million nodes even at its smallest: built on
# the GPU, skipped without one.
_KINDS = [
    pytest.param(
        kind,
        marks=pytest.mark.skipif(
            kind == "mitochondrion" and not torch.cuda.is_available(),
            reason="a mitochondrion is built on the GPU",
        ),
    )
    for kind in _SMALLEST
]


def _build(plan: OrganellePlan, seed: int = 1) -> tuple[Grid, list[torch.Tensor]]:
    """Build a plan on a grid centred on the organelle, as the generator does."""
    half = np.asarray(plan.half_extent, dtype=float)
    device = "cuda" if plan.kind == "mitochondrion" else "cpu"
    grid = Grid(tuple(2 * half), lo=tuple(-half), device=device)
    return grid, plan.build(grid, np.random.default_rng(seed))


def test_every_kind_has_defaults() -> None:
    assert set(ORGANELLE_DEFAULTS) == set(_SMALLEST)


@pytest.mark.parametrize("kind", _KINDS)
def test_every_kind_plans_and_builds_inside_its_bounding_radius(kind: str) -> None:
    plan = plan_organelle(kind, np.random.default_rng(0), **_SMALLEST[kind])
    assert plan.kind == kind
    assert set(plan.params) == set(ORGANELLE_DEFAULTS[kind])
    assert plan.bounding_radius > 0 and np.all(np.asarray(plan.half_extent) > 0)

    grid, surfaces = _build(plan)
    assert len(surfaces) >= 1
    x, y, z = grid.axes()
    distance = (x**2 + y**2 + z**2).sqrt().expand(grid.shape)
    for phi in surfaces:
        assert phi.shape == grid.shape
        inside = phi < 0
        # A closed solid with a surface: neither empty nor the whole grid.
        assert bool(inside.any()) and not bool(inside.all())
        # `bounding_radius` bounds the solid about the organelle's centre.
        assert float(distance[inside].max()) <= plan.bounding_radius
        # And the surface keeps the reach a bilayer needs on both sides.
        _, _, outer, inner = reach(phi, grid)
        assert float(outer.min()) >= 25.0 - 2.0
        assert float(inner.min()) >= 25.0 - 2.0


def test_mitochondrion_has_an_inner_membrane_inside_the_outer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("a mitochondrion is built on the GPU")
    plan = plan_organelle(
        "mitochondrion", np.random.default_rng(0), **_SMALLEST["mitochondrion"]
    )
    _, (outer, inner) = _build(plan)
    assert bool((outer[inner < 0] < 0).all())
    # The cristae fold the inner membrane: its solid is well short of the
    # outer one's volume.
    assert int((inner < 0).sum()) < 0.9 * int((outer < 0).sum())


def test_vesicle_layers_nest() -> None:
    """A second layer lies 90 A inside the first, when the lumen has room."""
    plan = plan_organelle(
        "vesicle",
        np.random.default_rng(0),
        radius=170.0,
        elongation=1.0,
        undulation=0.0,
        layers=2,
    )
    _, surfaces = _build(plan)
    assert len(surfaces) == 2
    outer, inner = surfaces
    assert bool((outer[inner < 0] < 0).all())
    near = inner.abs() < 20.0
    assert float((inner - (outer + 90.0))[near].abs().max()) < 4.0

    # Too small to hold a second lumen: one layer, not a degenerate inner one.
    small = plan_organelle(
        "vesicle",
        np.random.default_rng(0),
        radius=100.0,
        elongation=1.0,
        undulation=0.0,
        layers=2,
    )
    assert len(_build(small)[1]) == 1


def test_a_scalar_is_used_exactly_and_a_range_is_drawn_from() -> None:
    radii = set()
    for seed in range(20):
        plan = plan_organelle(
            "vesicle",
            np.random.default_rng(seed),
            radius=[120.0, 180.0],
            elongation=1.25,
            undulation=[0.0, 10.0],
        )
        assert 120.0 <= plan.params["radius"] <= 180.0
        assert plan.params["elongation"] == 1.25
        assert 0.0 <= plan.params["undulation"] <= 10.0
        assert plan.params["layers"] == 1
        radii.add(plan.params["radius"])
    assert len(radii) == 20  # a draw per seed, not one value

    counts = {
        plan_organelle("tubules", np.random.default_rng(seed), branches=[1, 3]).params[
            "branches"
        ]
        for seed in range(30)
    }
    assert counts == {1.0, 2.0, 3.0}  # integer settings draw whole counts


@pytest.mark.parametrize("kind", list(_SMALLEST))
def test_defaults_draw_inside_their_ranges(kind: str) -> None:
    plan = plan_organelle(kind, np.random.default_rng(3))
    for name, default in ORGANELLE_DEFAULTS[kind].items():
        low, high = (default, default) if np.isscalar(default) else default
        assert low <= plan.params[name] <= high, name


def test_same_seed_plans_the_same_organelle() -> None:
    first = plan_organelle("tubules", np.random.default_rng(5))
    second = plan_organelle("tubules", np.random.default_rng(5))
    other = plan_organelle("tubules", np.random.default_rng(6))
    assert first.params == second.params
    assert first.bounding_radius == second.bounding_radius
    assert np.array_equal(first.half_extent, second.half_extent)
    assert first.params != other.params


def test_same_seed_builds_the_same_surface() -> None:
    fields = _SMALLEST["vesicle"]
    one = _build(plan_organelle("vesicle", np.random.default_rng(2), **fields), seed=4)
    two = _build(plan_organelle("vesicle", np.random.default_rng(2), **fields), seed=4)
    assert torch.equal(one[1][0], two[1][0])


@pytest.mark.parametrize(
    ("kind", "fields", "match"),
    [
        ("vesicle", dict(radius=40.0), "below the smallest supported value"),
        ("vesicle", dict(radius=[40.0, 200.0]), "below the smallest supported value"),
        ("vesicle", dict(elongation=0.8), "below the smallest supported value"),
        ("mitochondrion", dict(radius=200.0), "below the smallest supported value"),
        ("cisterna", dict(lumen_width=50.0), "below the smallest supported value"),
        ("tubules", dict(branches=0), "below the smallest supported value"),
        ("vesicle", dict(radius=[200.0, 100.0]), "descending"),
        ("vesicle", dict(axes=[100.0, 100.0, 100.0]), "has no setting 'axes'"),
        ("tubules", dict(elongation=1.2), "has no setting 'elongation'"),
        ("golgi", {}, "unknown membrane kind"),
    ],
)
def test_bad_settings_are_refused(
    kind: str, fields: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        plan_organelle(kind, np.random.default_rng(0), **fields)


@pytest.mark.parametrize("kind", ["vesicle", "tubules", "cisterna"])
def test_max_radius_shrinks_default_sizes_to_fit(kind: str) -> None:
    for seed in range(5):
        free = plan_organelle(kind, np.random.default_rng(seed))
        limit = 0.6 * free.bounding_radius
        fitted = plan_organelle(kind, np.random.default_rng(seed), max_radius=limit)
        assert fitted.bounding_radius <= limit
        # Only the size settings moved, and only downwards.
        for name, value in free.params.items():
            if name in ("radius", "length", "diameter"):
                assert fitted.params[name] < value
            else:
                assert fitted.params[name] == value
        # A limit the draw already meets changes nothing.
        loose = plan_organelle(
            kind, np.random.default_rng(seed), max_radius=2 * free.bounding_radius
        )
        assert loose.params == free.params


def test_max_radius_never_changes_an_explicit_size() -> None:
    plan = plan_organelle(
        "vesicle", np.random.default_rng(0), max_radius=100.0, radius=250.0
    )
    assert plan.params["radius"] == 250.0
    assert plan.bounding_radius > 100.0

    # Tubules given a radius but not a length: the length alone shrinks.
    free = plan_organelle("tubules", np.random.default_rng(0), radius=150.0)
    fitted = plan_organelle(
        "tubules",
        np.random.default_rng(0),
        radius=150.0,
        max_radius=0.6 * free.bounding_radius,
    )
    assert fitted.params["radius"] == 150.0
    assert fitted.params["length"] < free.params["length"]


def test_max_radius_stops_at_the_minimum_size() -> None:
    """A limit nothing can meet leaves the sizes at their minima instead of
    below them; the caller sees the bounding radius it got."""
    plan = plan_organelle("vesicle", np.random.default_rng(0), max_radius=30.0)
    assert plan.params["radius"] == 60.0
    assert plan.bounding_radius > 30.0
