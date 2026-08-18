"""
Centerline-spline-swept membrane shape backend.

For elongated, wandering TUBE topology (ER-tubule-like) -- NOT the
star-convex "radius function of direction" family
``_field_spherical_harmonics.py`` belongs to; this is the complementary case
that backend structurally cannot represent (a tube that curls back near
itself is crossed more than once by some rays from any single center).

Reuses ``_field.py``'s shared smooth-min blend machinery (``SphereSource``/
``blend_field``/``cap_curvature`` -- the only other consumer of these was a
deprecated ``shape_backend="metaball"`` backend, isotropically-SCATTERED
sources rather than a swept path, since deleted): a swept tube's signed
field IS a smooth-min union of many small sphere SDFs whose centers walk
along a smooth path -- exactly what ``blend_field`` computes given any
list of sources, regardless of how they were sampled. The only new piece
here is the source SAMPLING: a correlated ("persistent") random walk
through space, rather than an independent uniform scatter in a box.

Because ``phi`` comes directly from ``blend_field``'s analytic smooth-min of
exact sphere SDFs, it is well-behaved (Eikonal property, ``|grad(phi)| ~=
1``) BY CONSTRUCTION -- unlike the EDT-derived ``_field_spherical_harmonics.py``
backend, there is no boolean-mask/``distance_transform_edt`` step here, and
correspondingly no ``_MIN_RELIABLE_VOXELS_PER_RADIUS``-style reliability
warning is needed. Performance: ``blend_field`` loops one smooth-min pass
per source sequentially (Python-level loop, each pass a full elementwise
grid op) -- plausibly a concern for a dense tube needing dozens-to-hundreds
of sources, but measured directly rather than assumed: ~1.3s for 121
sources on a 150^3 grid, and ~3.9s for a realistic ~9.8M-voxel (214^3)
working grid at this module's own default source count (~34, from
`total_length_a=500`/`step_length_a=15`) -- comparable to the other fast
backend, not a bottleneck in practice at these scales.

``cap_curvature`` IS still applied: an analytic smooth-min blend can still
have locally sharp concave curvature, and a sharply-bending (low
``flexibility``) tube is exactly the case most at risk of violating the
bilayer self-intersection guarantee the later ``+-thickness/2`` leaflet
offset depends on.

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
   matters: a k-nearest-neighbor averaging smoother (correct for an
   unordered surface point cloud, but not what this is) would be WRONG
   here -- a sufficiently sinuous tube can curl close to itself in physical
   space, and k-NN smoothing would then average together points that are
   near in space but far apart along the path, corrupting exactly the
   self-approaching topology this backend exists to produce.

Beading risk: same-radius spheres spaced ``step_length_a`` apart only fuse
into a smooth continuous tube (rather than a visible string of beads) if
``step_length_a`` stays well under ``2 * tube_radius_a`` and
``blend_sharpness_a`` is on the order of ``tube_radius_a`` itself -- a
default tuned for a handful of sparse, independent blobs would under-blend
a dense chain, so this module computes its own default instead. Warns if
``step_length_a`` exceeds the (mean, see "Radius variation" below) tube
radius, a real, previously observed failure mode (visible beading), not a
hypothetical one.

Radius variation
-----------------
``radius_variation`` (default 0, off) draws a per-source radius instead of
reusing ``tube_radius_a`` for every sphere: i.i.d. Gaussian noise, one value
per path point, smoothed with ``gaussian_filter1d`` along path order (the
SAME tool and reasoning as the path-smoothing step above, reused for a
second, independent field) at ``radius_variation_sigma_points``, then
rescaled to unit RMS and applied multiplicatively --
``tube_radius_a * clip(1 + radius_variation * noise, _MIN_RADIUS_FRACTION,
None)`` -- the same amplitude-normalized-perturbation pattern
``_field_spherical_harmonics.py`` uses for its own random radius function,
not a fresh scheme.

Deliberately NOT a second persistent random walk (like the path direction
itself): direction lives on a bounded manifold (the unit sphere), so a
persistent walk there just wanders in place indefinitely, but radius is
unbounded -- an actual random walk in radius would drift to implausible
values over a long path. Smoothed Gaussian noise stays anchored to
``tube_radius_a`` regardless of path length.

This also changes what the beading check (above) should mean: with
non-zero ``radius_variation``, the LOCAL radius will occasionally dip below
``step_length_a`` wherever the (smooth, non-periodic) noise is low, which
is intentional -- it produces sparse, irregularly-spaced constrictions
along the tube, not the mechanically-repeating beading pattern the check
exists to catch. Warning on every such local dip would suppress this
effect entirely, so the check instead compares ``step_length_a`` against
the MEAN drawn radius: it fires only when beading would be the norm along
the whole tube, not the occasional exception.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy import ndimage

from ._field import (
    MembraneField,
    SphereSource,
    _grid_points_xyz,
    _warn_if_clipped_at_boundary,
    blend_field,
    cap_curvature,
)

# Floor for radius_variation's multiplicative perturbation -- keeps even an
# aggressive draw from collapsing a source's radius toward zero (which would
# pinch the tube toward disconnection). Mirrors _field_spherical_harmonics.py's
# own floor on its radius perturbation (0.05), but expressed relative to THIS
# module's radius scale (tube_radius_a, default 25 A) rather than that
# module's unit-sphere convention -- the two floors are not directly
# comparable numbers.
_MIN_RADIUS_FRACTION = 0.25


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
        Distance between consecutive points, Å.
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
        Å, NOT yet recentered or smoothed.
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
    radius_variation: float = 0.0,
    radius_variation_sigma_points: float = 2.0,
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
        Working grid voxel spacing, Å.
    total_length_a : float, optional
        Approximate path CONTOUR length (not bounding-box extent -- a
        wandering path's bounding box is typically much smaller than its
        contour length), Å. Default 500.0.
    step_length_a : float, optional
        Distance between consecutive blended sphere source centers along
        the path, Å. Must stay well under ``2 * tube_radius_a`` (see
        module docstring's beading-risk warning). Default 15.0.
    tube_radius_a : float, optional
        Tube radius, Å. Default 25.0.
    flexibility : float, optional
        In ``(0, 1]`` -- see ``_sample_wandering_path``. Default 0.15,
        picked from a direct visual sweep (0.05/0.15/0.35, see ``dev/
        swept_spline_sweep.py``): 0.05 was nearly a straight rod, 0.35
        produced a sharp, almost folded-back bend (a good "very flexible"
        stress case, not a good default); 0.15 gave a gently organic,
        clearly non-straight tube with no beading at this module's other
        defaults.
    radius_variation : float, optional
        RMS fractional variation in tube radius along the path
        (dimensionless, relative to `tube_radius_a`) -- see module
        docstring's "Radius variation" section. Default 0.0 (constant
        radius, this function's behaviour before this parameter existed).
    radius_variation_sigma_points : float, optional
        ``scipy.ndimage.gaussian_filter1d`` sigma for the radius noise, in
        PATH POINTS. Default 2.0, picked from a direct visual sweep (1/2/3,
        see ``dev/swept_spline_radius_variation_sweep.py``) -- NOT the
        larger value (4.0) that seemed intuitive going in: at this
        function's own default path length (~34 points), sigma=4 leaves
        too few effectively-independent points for more than one broad
        swell to fit, and ``gaussian_filter1d``'s ``nearest`` edge padding
        then biases that single swell into looking like a monotonic taper
        (a systematic effect, reproduced across multiple seeds, not
        coincidence) -- the opposite of the several-irregular-bumps look
        this parameter exists to produce. sigma=1-2 reliably gives multiple
        organic-looking swells instead. Only affects the field when
        `radius_variation > 0`.
    blend_sharpness_a : float, optional
        Smooth-min blend radius, Å (see
        :func:`~specter.specimen.membrane._field.blend_field`). Default
        ``0.5 * tube_radius_a`` -- a default tuned for a handful of sparse,
        independent blobs would under-blend a dense chain of sources into
        visible beading, so this module computes its own instead.
    path_smoothing_sigma_points : float, optional
        ``scipy.ndimage.gaussian_filter1d`` sigma, in PATH POINTS (not
        Å or voxels) -- see module docstring for why this is
        order-aware rather than spatial. Default 1.5.
    curvature_iterations : int, optional
        Curvature-capping relaxation steps (see
        :func:`~specter.specimen.membrane._field.cap_curvature`). Default
        15 -- a swept tube's curvature is already gentler by construction
        than a compact blob's, so fewer relaxation steps are needed.
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
    if not 0.0 < flexibility <= 1.0:
        raise ValueError(f"flexibility must be in (0, 1], got {flexibility}")
    if radius_variation < 0.0:
        raise ValueError(f"radius_variation must be >= 0, got {radius_variation}")

    n_points = max(2, round(total_length_a / step_length_a) + 1)
    rng = np.random.default_rng(seed)
    positions = _sample_wandering_path(n_points, step_length_a, flexibility, rng)

    if radius_variation > 0.0:
        noise = rng.normal(size=n_points)
        noise = ndimage.gaussian_filter1d(
            noise, sigma=radius_variation_sigma_points, mode="nearest"
        )
        # Center BEFORE scaling to unit RMS, not just divide by std: at
        # small n_points, a smoothed sequence's own mean is generically
        # nonzero (not guaranteed near zero the way the raw i.i.d. noise
        # was), and dividing an off-center sequence by its own (possibly
        # small) std amplifies that offset along with the genuine spread --
        # confirmed directly to occasionally shift EVERY point several
        # std past the floor clip (e.g. n_points=18, seed=3 above: raw
        # smoothed mean -0.43 vs std 0.06, an uncentered divide put every
        # normalized value below -5, collapsing the whole radius sequence
        # to the floor instead of the intended mild, varying perturbation).
        noise = noise - noise.mean()
        noise_std = float(noise.std())
        if noise_std > 0.0:
            noise = noise / noise_std
        radii = tube_radius_a * np.clip(
            1.0 + radius_variation * noise, _MIN_RADIUS_FRACTION, None
        )
    else:
        radii = np.full(n_points, tube_radius_a)

    if step_length_a > float(radii.mean()):
        warnings.warn(
            f"generate_membrane_field_swept_spline: step_length_a "
            f"({step_length_a:.1f} A) exceeds the mean tube radius "
            f"({radii.mean():.1f} A) -- consecutive sphere sources are spaced "
            "too far apart relative to their own radius to fuse into a smooth "
            "tube ON AVERAGE (a real, previously observed failure mode: "
            "visible beading/segmentation along the path rather than a "
            "continuous surface). Decrease step_length_a, or increase "
            "tube_radius_a. (With radius_variation > 0, occasional local "
            "narrowing below step_length_a is expected -- it produces sparse, "
            "irregular constrictions rather than this failure mode -- so this "
            "check only fires when beading would be the norm, not the "
            "exception.)",
            stacklevel=2,
        )

    bbox_mid = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    positions = positions - bbox_mid
    positions = ndimage.gaussian_filter1d(
        positions, sigma=path_smoothing_sigma_points, axis=0, mode="nearest"
    )

    if blend_sharpness_a is None:
        blend_sharpness_a = 0.5 * tube_radius_a

    positions_t = torch.as_tensor(positions, dtype=torch.float32, device=device)
    sources = [
        SphereSource(center_xyz=positions_t[i], radius=float(radii[i]))
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

    clipped = _warn_if_clipped_at_boundary(phi)

    return MembraneField(
        phi=phi,
        spacing_a=spacing_a,
        origin_xyz=origin_xyz.to(device),
        clipped_at_boundary=clipped,
    )


__all__ = ["generate_membrane_field_swept_spline"]
