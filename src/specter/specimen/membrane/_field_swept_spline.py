"""
Centerline-spline-swept membrane shape backend.

For elongated, wandering TUBE topology (ER-tubule-like) -- NOT the
star-convex "radius function of direction" family
``_field_spherical_harmonics.py`` belongs to; this is the complementary case
that backend structurally cannot represent (a tube that curls back near
itself is crossed more than once by some rays from any single center).

Reuses ``_field.py``'s existing metaball machinery almost entirely: a swept
tube's signed field IS a smooth-min union of many small sphere SDFs whose
centers walk along a smooth path -- exactly what ``blend_field`` already
computes for the ``"metaball"`` backend's randomly-SCATTERED sources. The
only new piece here is the source SAMPLING: a correlated ("persistent")
random walk instead of ``sample_metaball_sources``' independent uniform
scatter in a box.

Because ``phi`` comes directly from ``blend_field``'s analytic smooth-min of
exact sphere SDFs, it is well-behaved (Eikonal property, ``|grad(phi)| ~=
1``) BY CONSTRUCTION, the same category ``generate_membrane_field``
(metaball) is in -- unlike the EDT-derived backends
(``_field_alpha.py``/``_field_spherical_harmonics.py``), there is no
boolean-mask/``distance_transform_edt`` step here, and
correspondingly no ``_MIN_RELIABLE_VOXELS_PER_RADIUS``-style reliability
warning is needed (metaball itself has none, for the same reason).
Performance: ``blend_field`` loops one smooth-min pass per source
sequentially (Python-level loop, each pass a full elementwise grid op) --
plausibly a concern for a dense tube needing dozens-to-hundreds of sources,
but measured directly rather than assumed: ~1.3s for 121 sources on a
150^3 grid, and ~3.9s for a realistic ~9.8M-voxel (214^3) working grid at
this module's own default source count (~34, from `total_length_a=500`/
`step_length_a=15`) -- comparable to the other fast backends, not a
bottleneck in practice at these scales.

``cap_curvature`` IS still applied, though, exactly as metaball does by
default: an analytic smooth-min blend can still have locally sharp concave
curvature, and a sharply-bending (low ``flexibility``) tube is exactly the
case most at risk of violating the bilayer self-intersection guarantee the
later ``+-thickness/2`` leaflet offset depends on.

Path construction
------------------
A "persistent" (direction-correlated) random walk: each step's direction is
a weighted blend of the previous step's direction and a fresh random unit
vector (``flexibility`` controls the blend weight -- low values give long,
gently-curving paths, high values give tightly wandering ones), then
re-normalized. The resulting point sequence is:

1. Recentered (bounding-box midpoint to the origin) -- unlike the other
   backends' sources, which are constructed to already radiate from the
   origin, a random walk drifts and is not naturally centered.
2. Smoothed with ``scipy.ndimage.gaussian_filter1d(..., axis=0)`` --
   smooths along the array's own PATH ORDER, not spatial proximity. This
   matters: ``_cts_membrane.py``'s own ``_smooth_points`` (k-nearest-
   neighbor averaging, correct for an unordered surface point cloud, its
   actual use there) would be WRONG here -- a sufficiently sinuous tube can
   curl close to itself in physical space, and k-NN smoothing would then
   average together points that are near in space but far apart along the
   path, corrupting exactly the self-approaching topology this backend
   exists to produce.

Beading risk: same-radius spheres spaced ``step_length_a`` apart only fuse
into a smooth continuous tube (rather than a visible string of beads) if
``step_length_a`` stays well under ``2 * tube_radius_a`` and
``blend_sharpness_a`` is on the order of ``tube_radius_a`` itself --
``generate_membrane_field``'s own default ``blend_sharpness_a`` (tuned for a
handful of sparse, independent blobs) would under-blend a dense chain if
reused verbatim, so this module computes its own default instead. Warns if
``step_length_a > tube_radius_a``, a real, previously observed failure mode
(visible beading), not a hypothetical one.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy import ndimage

from ._field import (
    MembraneField,
    MetaballSource,
    _grid_points_xyz,
    _warn_if_clipped_at_boundary,
    blend_field,
    cap_curvature,
)


def _sample_wandering_path(
    n_points: int,
    step_length_a: float,
    flexibility: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a persistent (direction-correlated) random walk.

    Parameters
    ----------
    n_points : int
        Number of path points (``>= 2``).
    step_length_a : float
        Distance between consecutive points, Angstrom.
    flexibility : float
        In ``(0, 1]``. At each step, the new direction is
        ``normalize((1 - flexibility) * previous_direction + flexibility *
        random_unit_vector)`` -- low values give long, gently-curving paths;
        values near 1 give a nearly memoryless (tightly wandering) walk.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape ``(n_points, 3)``, physical ``(x, y, z)`` path points,
        Angstrom, NOT yet recentered or smoothed.
    """
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    positions = np.zeros((n_points, 3), dtype=np.float64)
    for i in range(1, n_points):
        random_unit = rng.normal(size=3)
        random_unit /= np.linalg.norm(random_unit)
        direction = (1.0 - flexibility) * direction + flexibility * random_unit
        direction /= np.linalg.norm(direction)
        positions[i] = positions[i - 1] + step_length_a * direction
    return positions


def generate_membrane_field_swept_spline(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    total_length_a: float = 500.0,
    step_length_a: float = 15.0,
    tube_radius_a: float = 25.0,
    flexibility: float = 0.15,
    blend_sharpness_a: float | None = None,
    path_smoothing_sigma_points: float = 1.5,
    curvature_iterations: int = 15,
    curvature_step_fraction: float = 0.15,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> MembraneField:
    """
    Generate an elongated, wandering tube-shaped membrane mid-surface field
    by sweeping a sphere along a smooth random path, in the same
    :class:`MembraneField` representation the other backends produce.

    Parameters
    ----------
    shape_zyx : tuple of int
        Working grid shape, ``(Z, Y, X)``.
    spacing_a : float
        Working grid voxel spacing, Angstrom.
    total_length_a : float, optional
        Approximate path CONTOUR length (not bounding-box extent -- a
        wandering path's bounding box is typically much smaller than its
        contour length), Angstrom. Default 500.0.
    step_length_a : float, optional
        Distance between consecutive metaball source centers along the
        path, Angstrom. Must stay well under ``2 * tube_radius_a`` (see
        module docstring's beading-risk warning). Default 15.0.
    tube_radius_a : float, optional
        Tube radius, Angstrom. Default 25.0.
    flexibility : float, optional
        In ``(0, 1]`` -- see ``_sample_wandering_path``. Default 0.15,
        picked from a direct visual sweep (0.05/0.15/0.35, see ``dev/
        swept_spline_sweep.py``): 0.05 was nearly a straight rod, 0.35
        produced a sharp, almost folded-back bend (a good "very flexible"
        stress case, not a good default); 0.15 gave a gently organic,
        clearly non-straight tube with no beading at this module's other
        defaults.
    blend_sharpness_a : float, optional
        Smooth-min blend radius, Angstrom (see
        :func:`~specter.specimen.membrane._field.blend_field`). Default
        ``0.5 * tube_radius_a`` -- deliberately NOT
        ``generate_membrane_field``'s own default (tuned for a handful of
        sparse, independent blobs, which under-blends a dense chain of
        sources into visible beading).
    path_smoothing_sigma_points : float, optional
        ``scipy.ndimage.gaussian_filter1d`` sigma, in PATH POINTS (not
        Angstrom or voxels) -- see module docstring for why this is
        order-aware rather than spatial. Default 1.5.
    curvature_iterations : int, optional
        Curvature-capping relaxation steps (see
        :func:`~specter.specimen.membrane._field.cap_curvature`). Default
        15 -- lower than metaball's own default 30, since a swept tube's
        curvature is already gentler by construction than a compact blob's.
    curvature_step_fraction : float, optional
        See :func:`~specter.specimen.membrane._field.cap_curvature`.
        Default 0.15.
    device : str or torch.device, optional
        Device for the returned field. Default ``"cpu"``.
    seed : int, optional
        Random seed. Default ``None``.

    Returns
    -------
    MembraneField
    """
    if step_length_a > tube_radius_a:
        warnings.warn(
            f"generate_membrane_field_swept_spline: step_length_a "
            f"({step_length_a:.1f} A) exceeds tube_radius_a ({tube_radius_a:.1f} A) "
            "-- consecutive sphere sources are spaced too far apart relative to "
            "their own radius to fuse into a smooth tube (a real, previously "
            "observed failure mode: visible beading/segmentation along the path "
            "rather than a continuous surface). Decrease step_length_a, or "
            "increase tube_radius_a.",
            stacklevel=2,
        )
    if not 0.0 < flexibility <= 1.0:
        raise ValueError(f"flexibility must be in (0, 1], got {flexibility}")

    n_points = max(2, round(total_length_a / step_length_a) + 1)
    rng = np.random.default_rng(seed)
    positions = _sample_wandering_path(n_points, step_length_a, flexibility, rng)

    bbox_mid = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    positions = positions - bbox_mid
    positions = ndimage.gaussian_filter1d(
        positions, sigma=path_smoothing_sigma_points, axis=0, mode="nearest"
    )

    if blend_sharpness_a is None:
        blend_sharpness_a = 0.5 * tube_radius_a

    positions_t = torch.as_tensor(positions, dtype=torch.float32, device=device)
    sources = [
        MetaballSource(center_xyz=positions_t[i], radius=tube_radius_a)
        for i in range(n_points)
    ]

    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a

    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device)
    phi = blend_field(sources, points_xyz, blend_sharpness_a)
    phi = cap_curvature(phi, spacing_a, curvature_iterations, curvature_step_fraction)

    _warn_if_clipped_at_boundary(phi)

    return MembraneField(phi=phi, spacing_a=spacing_a, origin_xyz=origin_xyz.to(device))


__all__ = ["generate_membrane_field_swept_spline"]
