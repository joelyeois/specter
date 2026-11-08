"""
The two organic-shape models a `MembraneGenerator` can draw, as settings
groups.

Each backend's parameters are a frozen dataclass of their own, so a
parameter can only be given to the backend that reads it:
`SphericalHarmonicsShape` perturbs an ellipsoid, `SweptSplineShape` sweeps a
sphere along a wandering path. A size left as None is drawn from its range
by `resolve_sizes`, which returns the same group with every size concrete.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from specter.options import ShapeBackend

#: `MembraneGenerator`'s default size-draw ranges, in Angstrom. Named here
#: rather than left as bare field defaults because a second reader needs
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


@dataclass(frozen=True)
class SphericalHarmonicsShape:
    """
    A star-convex organelle: an ellipsoid perturbed by a random real
    spherical-harmonic expansion
    (:func:`~specter.specimen.membrane._field_spherical_harmonics.generate_membrane_field_spherical_harmonics`).

    Smooth by construction, with a physically motivated (Helfrich
    bending-mode) undulation spectrum, but a radius function of direction,
    so restricted to star-convex topology: vesicles, nuclei and roughly
    spherical or ellipsoidal mitochondria, not branching or self-occluding
    shapes.

    Attributes
    ----------
    max_degree : int, optional
        Highest spherical-harmonic degree in the random surface
        perturbation; higher values add finer surface detail. Default 8.
    axes : tuple of float, optional
        Semi-axes ``(a_x, a_y, a_z)`` of the base ellipsoid, Å. Isotropic
        axes give a roughly spherical organelle, anisotropic ones an
        elongated or flattened one. Default None: each axis is drawn
        independently, uniformly, from `axes_range` (mild natural anisotropy
        for free) with a `seed`-derived generator independent of the
        shape's own randomness.
    axes_range : tuple of float, optional
        ``(low, high)`` semi-axis draw range, Å, used only when `axes` is
        None. Default ``(150.0, 450.0)``: vesicle and small-organelle scale.
        Synaptic vesicles run ~25-60 nm in diameter and general or endosomal
        vesicles up to ~100-300 nm in cryo-ET (PNAS
        10.1073/pnas.2403136121; PMID 24455109); this range (30-90 nm
        diameter) sits inside that population without reaching
        mitochondrial scale (~200-700 nm diameter), which is different
        enough that one default range cannot cover both. Pass `axes`
        explicitly for that.
    amplitude : float, optional
        RMS fractional radius perturbation (dimensionless), picked from a
        direct visual sweep. Default 0.15.
    spectrum_power : float, optional
        Exponent ``p`` in ``Var(a_lm) ~ 1 / [l (l + 1)]**p``. 2.0 matches
        the Helfrich (1973) [1]_ thermal bending-mode spectrum of a lipid
        bilayer at equilibrium, in the large-``l``, tensionless limit of
        the vesicle spectrum of Milner & Safran (1987) [2]_. Default 2.0.

    References
    ----------
    .. [1] Helfrich, W. (1973). Elastic properties of lipid bilayers: theory and
       possible experiments. *Zeitschrift für Naturforschung C*, 28(11–12),
       693–703. https://doi.org/10.1515/znc-1973-11-1209
    .. [2] Milner, S. T., & Safran, S. A. (1987). Dynamical fluctuations of droplet
       microemulsions and vesicles. *Physical Review A*, 36(9), 4371–4379.
       https://doi.org/10.1103/PhysRevA.36.4371
    """

    backend: ClassVar[ShapeBackend] = "spherical_harmonics"

    max_degree: int = 8
    axes: tuple[float, float, float] | None = None
    axes_range: tuple[float, float] = DEFAULT_SH_AXES_RANGE_ANGSTROM
    amplitude: float = 0.15
    spectrum_power: float = 2.0


@dataclass(frozen=True)
class SweptSplineShape:
    """
    An elongated, tube-shaped organelle (ER-tubule-like): a sphere swept
    along a smoothly wandering random path
    (:func:`~specter.specimen.membrane._field_swept_spline.generate_membrane_field_swept_spline`).

    Unlike `SphericalHarmonicsShape` it can approach and loop back near
    itself.

    Attributes
    ----------
    total_length : float, optional
        Approximate path CONTOUR length (not bounding-box extent), Å.
        Default None: drawn uniformly from `total_length_range`.
    total_length_range : tuple of float, optional
        ``(low, high)`` contour-length draw range, Å, used only when
        `total_length` is None. Default ``(1500.0, 2500.0)``: long enough to
        read as a tube even at `tube_radius_range`'s upper bound, while the
        worst case (max radius, max length) auto-sizes to ~540 voxels per
        axis at 5 Å (~0.6 GB float32). A 4000 Å upper bound instead reaches
        ~1130 voxels per axis (~5.8 GB) and measurably slow generation, for
        organelles larger than a typical tomogram crop.
    step_length_angstrom : float, optional
        Distance between consecutive blended sphere centres along the path,
        Å. It must stay well under ``2 * tube_radius`` or the tube shows
        visible beading (warned about). Default None:
        ``0.5 * tube_radius``, which stays under the beading threshold
        whichever radius is drawn.
    tube_radius : float, optional
        Tube radius, Å. Default None: drawn uniformly from
        `tube_radius_range`.
    tube_radius_range : tuple of float, optional
        ``(low, high)`` tube-radius draw range, Å, used only when
        `tube_radius` is None. Default ``(150.0, 400.0)`` (30-80 nm
        diameter), ER tubule scale: neuronal ER tubules measure ~20 nm in
        diameter by EM, and general ER tubules up to ~88 nm by STORM (PNAS
        10.1073/pnas.2117559119; PMC5705721).
    flexibility : float, optional
        In (0, 1], picked from a direct visual sweep: 0.05 is nearly a
        straight rod, 0.35 and above produce sharp hooks and loops, 0.15 a
        gently organic tube. Default None: drawn uniformly from
        `flexibility_range`.
    flexibility_range : tuple of float, optional
        ``(low, high)`` flexibility draw range, used only when
        `flexibility` is None. Default ``(0.08, 0.25)``, inside the gently
        organic zone of that sweep.
    radius_variation : float, optional
        RMS fractional variation of the tube radius along the path. Default
        None: drawn uniformly from `radius_variation_range`.
    radius_variation_range : tuple of float, optional
        ``(low, high)`` radius-variation draw range, used only when
        `radius_variation` is None. Default ``(0.1, 0.3)``: mild caliber
        variation (real ER tubules show local varicosities and
        constrictions) rather than a perfectly uniform tube.
    radius_variation_sigma_points : float, optional
        Path-order smoothing of the radius noise, in PATH POINTS; 2.0 was
        picked from a direct visual sweep. Only affects the field when
        `radius_variation` is above 0. Default 2.0.
    blend_sharpness_angstrom : float, optional
        Smooth-min blend radius, Å. Default None: ``0.5 * tube_radius``. A
        default tuned for a few sparse blobs would under-blend a dense chain
        of sources into visible beading.
    path_smoothing_sigma_points : float, optional
        Path-order-aware smoothing (``scipy.ndimage.gaussian_filter1d``), in
        PATH POINTS. Default 1.5.
    curvature_iterations : int, optional
        Curvature-capping relaxation steps; an analytic smooth-min blend
        can still be locally sharply concave. Default 15.
    curvature_step_fraction : float, optional
        See :func:`~specter.specimen.membrane._field.cap_curvature`.
        Default 0.15.
    """

    backend: ClassVar[ShapeBackend] = "swept_spline"

    total_length: float | None = None
    total_length_range: tuple[float, float] = DEFAULT_SWEPT_TOTAL_LENGTH_RANGE_ANGSTROM
    step_length_angstrom: float | None = None
    tube_radius: float | None = None
    tube_radius_range: tuple[float, float] = DEFAULT_SWEPT_TUBE_RADIUS_RANGE_ANGSTROM
    flexibility: float | None = None
    flexibility_range: tuple[float, float] = (0.08, 0.25)
    radius_variation: float | None = None
    radius_variation_range: tuple[float, float] = (0.1, 0.3)
    radius_variation_sigma_points: float = 2.0
    blend_sharpness_angstrom: float | None = None
    path_smoothing_sigma_points: float = 1.5
    curvature_iterations: int = 15
    curvature_step_fraction: float = 0.15


MembraneShape = SphericalHarmonicsShape | SweptSplineShape

#: Each backend's settings group, by its `shape_backend` name.
SHAPE_GROUPS: dict[str, type[SphericalHarmonicsShape] | type[SweptSplineShape]] = {
    "spherical_harmonics": SphericalHarmonicsShape,
    "swept_spline": SweptSplineShape,
}


def membrane_shape(
    shape: MembraneShape | None,
    shape_backend: ShapeBackend | None,
    settings: dict[str, Any],
) -> MembraneShape:
    """
    The shape group a `MembraneGenerator` was given: `shape` itself, or one
    built from its fields as keywords for `shape_backend` (default
    ``"spherical_harmonics"``). Never both, and a field of the other
    backend is refused rather than ignored.
    """
    if shape is not None:
        if settings:
            raise TypeError(
                f"pass a {type(shape).__name__} or its fields as keywords, not "
                f"both (got {sorted(settings)})"
            )
        if shape_backend is not None and shape_backend != shape.backend:
            raise ValueError(
                f"shape_backend={shape_backend!r} does not match the "
                f"{type(shape).__name__} given as shape"
            )
        return shape
    backend = shape_backend or "spherical_harmonics"
    if backend not in SHAPE_GROUPS:
        raise ValueError(
            "shape_backend must be 'spherical_harmonics' or 'swept_spline', "
            f"got {backend!r}"
        )
    group = SHAPE_GROUPS[backend]
    known = {f.name for f in dataclasses.fields(group)}
    unknown = sorted(set(settings) - known)
    if unknown:
        raise TypeError(
            f"MembraneGenerator got {unknown}, which are not {group.__name__} "
            f"fields (shape_backend={backend!r}); its fields are {sorted(known)}"
        )
    return group(**settings)


def _resolve_range(
    name: str, range_ab: tuple[float, float], min_low: float
) -> tuple[float, float]:
    low, high = range_ab
    if low > high or low < min_low:
        raise ValueError(
            f"{name} must be (low, high) with {min_low} <= low <= high, "
            f"got {range_ab!r}"
        )
    return low, high


def _draw_uniform(rng: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high, generator=rng).item())


def resolve_sizes(shape: MembraneShape, seed: int | None) -> MembraneShape:
    """
    `shape` with every size left as None drawn from its range.

    The draws come from a generator seeded from ``seed`` alone, separate
    from whatever RNG the shape backend uses internally, so they never
    consume that stream.
    """
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    if isinstance(shape, SphericalHarmonicsShape):
        if shape.axes is None:
            lo, hi = _resolve_range("axes_range", shape.axes_range, min_low=1e-6)
            axes = (
                _draw_uniform(rng, lo, hi),
                _draw_uniform(rng, lo, hi),
                _draw_uniform(rng, lo, hi),
            )
        else:
            # A tuple even when given a list (every TOML-sourced value), so
            # `_fit_shape_to_box`'s clamp check compares like with like.
            axes = (float(shape.axes[0]), float(shape.axes[1]), float(shape.axes[2]))
        return dataclasses.replace(shape, axes=axes)

    # Three draws a swept spline does not use. Both backends' sizes were
    # once drawn from one stream, ellipsoid axes first, and skipping them
    # would move every seeded swept-spline membrane.
    for _ in range(3):
        _draw_uniform(rng, *DEFAULT_SH_AXES_RANGE_ANGSTROM)
    tube_radius = shape.tube_radius
    if tube_radius is None:
        lo, hi = _resolve_range(
            "tube_radius_range", shape.tube_radius_range, min_low=1e-6
        )
        tube_radius = _draw_uniform(rng, lo, hi)
    total_length = shape.total_length
    if total_length is None:
        lo, hi = _resolve_range(
            "total_length_range", shape.total_length_range, min_low=1e-6
        )
        total_length = _draw_uniform(rng, lo, hi)
    flexibility = shape.flexibility
    if flexibility is None:
        lo, hi = _resolve_range(
            "flexibility_range", shape.flexibility_range, min_low=1e-6
        )
        if hi > 1.0:
            raise ValueError(
                "flexibility_range must have high <= 1, got "
                f"{shape.flexibility_range!r}"
            )
        flexibility = _draw_uniform(rng, lo, hi)
    radius_variation = shape.radius_variation
    if radius_variation is None:
        lo, hi = _resolve_range(
            "radius_variation_range", shape.radius_variation_range, min_low=0.0
        )
        radius_variation = _draw_uniform(rng, lo, hi)
    step_length = shape.step_length_angstrom
    if step_length is None:
        # Half the resolved tube radius: always under the 2 * tube_radius
        # beading threshold whichever value was drawn, unlike a fixed
        # absolute default tuned for one radius.
        step_length = 0.5 * tube_radius
    return dataclasses.replace(
        shape,
        tube_radius=tube_radius,
        total_length=total_length,
        flexibility=flexibility,
        radius_variation=radius_variation,
        step_length_angstrom=step_length,
    )
