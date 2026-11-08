"""
How much room a membrane of a given shape needs.

One question -- *how big is this organelle* -- asked from several places for
different reasons: `MembraneGenerator` sizes its own working grid with it,
and `specimen.tomogram` uses it as a collision radius when placing instances.
Answering it means branching on the shape's kind, and that branch was written
out at each site, so a third backend would have meant finding them all and
two of them silently disagreeing if it did not.

Only the *measurement* lives here. What callers do with it deliberately does
not: `MembraneGenerator` divides by a safety margin before choosing a grid,
`specimen.tomogram` takes it raw, and the tomogram pipeline caps a draw range
against the box's limiting axis using `clip_axes` -- tomogram-level knowledge
a membrane has no business holding. Those are different questions that happen
to start from the same number.
"""

from __future__ import annotations

from ._shape import (
    DEFAULT_SH_AXES_RANGE_ANGSTROM,
    DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_ANGSTROM,
    DEFAULT_SWEPT_TUBE_RADIUS_RANGE_ANGSTROM,
    MembraneShape,
    SphericalHarmonicsShape,
    SweptSplineShape,
)

__all__ = [
    "DEFAULT_SH_AXES_RANGE_ANGSTROM",
    "DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_ANGSTROM",
    "DEFAULT_SWEPT_TUBE_RADIUS_RANGE_ANGSTROM",
    "membrane_bounding_radius",
]


def membrane_bounding_radius(shape: MembraneShape) -> float:
    """
    Conservative bounding-sphere radius for a membrane's resolved size, in A.

    Deliberately generous rather than exact. For a swept spline a wandering
    path's true bounding box is smaller than its contour length, so treating
    the full contour as if it were straight overestimates the reach -- which
    is the safe direction for both sizing a grid and rejecting a colliding
    placement.

    Parameters
    ----------
    shape : SphericalHarmonicsShape or SweptSplineShape
        The membrane's shape, its sizes resolved (`resolve_sizes`).

    Returns
    -------
    float
        Bounding-sphere radius in A.

    Raises
    ------
    ValueError
        If a size the shape needs is still None, or the shape is of an
        unknown kind. A new backend must be added here rather than
        defaulting to another one's formula.
    """
    if isinstance(shape, SphericalHarmonicsShape):
        if shape.axes is None:
            raise ValueError("a SphericalHarmonicsShape needs its axes resolved")
        return max(shape.axes)
    if isinstance(shape, SweptSplineShape):
        if shape.total_length is None or shape.tube_radius is None:
            raise ValueError(
                "a SweptSplineShape needs its total_length and tube_radius resolved"
            )
        return 0.5 * shape.total_length + shape.tube_radius
    raise ValueError(f"Unknown membrane shape: {shape!r}")
