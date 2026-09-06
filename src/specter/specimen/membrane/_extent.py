"""
How much room a membrane of a given shape needs.

One question -- *how big is this organelle* -- asked from several places for
different reasons: `MembraneGenerator` sizes its own working grid with it,
and `specimen.tomogram` uses it as a collision radius when placing instances.
Answering it means branching on `shape_backend`, and that branch was written
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

from collections.abc import Sequence
from specter.options import ShapeBackend

#: `MembraneGenerator`'s default size-draw ranges, in Angstrom. Named here
#: rather than left as bare signature defaults because a second reader needs
#: them: `pipelines._tomogram` caps an auto-sized organelle against the
#: tomogram box and has to write a full range to do it, so it needs the lower
#: bound the generator would have drawn from. Retyped there, the two could
#: drift with nothing failing -- silently, and only for a membrane whose size
#: the user left entirely unspecified.
#:
#: The lower bounds are biology-motivated organelle sizes and are never
#: shrunk by a cap; only the upper bound is clamped to fit the box.
DEFAULT_SH_AXES_RANGE_ANGSTROM = (150.0, 450.0)
DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_ANGSTROM = (1500.0, 2500.0)
DEFAULT_SWEPT_TUBE_RADIUS_RANGE_ANGSTROM = (150.0, 400.0)


def membrane_bounding_radius(
    shape_backend: ShapeBackend,
    *,
    sh_axes: Sequence[float] | None = None,
    swept_total_length: float | None = None,
    swept_tube_radius: float | None = None,
) -> float:
    """
    Conservative bounding-sphere radius for a membrane's resolved size, in A.

    Deliberately generous rather than exact. For ``"swept_spline"`` a
    wandering path's true bounding box is smaller than its contour length, so
    treating the full contour as if it were straight overestimates the reach
    -- which is the safe direction for both sizing a grid and rejecting a
    colliding placement.

    Parameters
    ----------
    shape_backend : {"spherical_harmonics", "swept_spline"}
        Which shape model the membrane uses.
    sh_axes : sequence of float, optional
        Semi-axes in A. Required for ``"spherical_harmonics"``.
    swept_total_length, swept_tube_radius : float, optional
        Contour length and tube radius in A. Required for ``"swept_spline"``.

    Returns
    -------
    float
        Bounding-sphere radius in A.

    Raises
    ------
    ValueError
        If the arguments the named backend needs are missing, or the backend
        is unknown. A new backend must be added here rather than defaulting
        to another one's formula.
    """
    if shape_backend == "spherical_harmonics":
        if sh_axes is None:
            raise ValueError("shape_backend='spherical_harmonics' requires sh_axes")
        return max(sh_axes)
    if shape_backend == "swept_spline":
        if swept_total_length is None or swept_tube_radius is None:
            raise ValueError(
                "shape_backend='swept_spline' requires swept_total_length and "
                "swept_tube_radius"
            )
        return 0.5 * swept_total_length + swept_tube_radius
    raise ValueError(f"Unknown shape_backend: {shape_backend!r}")
