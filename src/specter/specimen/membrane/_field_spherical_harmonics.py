"""
Spherical-harmonic membrane shape backend.

``_field_swept_spline.py``'s smooth-min sphere blend produces elongated,
tube-like topology but cannot represent a closed, roughly-spherical
organelle without stringing many overlapping sources together. This module
offers a purpose-built construction for that case instead, specifically for
CLOSED, STAR-CONVEX organelles (vesicles, nuclei, roughly-spherical
mitochondria): every ray from the organelle's center crosses the surface
exactly once, so the surface is fully described by a radius function of
direction, ``R(theta, phi)``, expanded in real spherical harmonics. This is
smooth by construction (no faceting to clean up) and gives direct,
physically-motivated control over the undulation spectrum (see
``_sample_sh_coefficients``'s Helfrich-spectrum docstring) -- at the cost
of being unable to represent non-star-convex topology (branching tubules,
self-occluding folds) at all; ``"swept_spline"`` (``_field_swept_spline.py``)
covers elongated, tube-like organelles instead, though not full branching
topology either.

Only the SHAPE (a solid to compute a signed-distance field from) is
constructed here; the calibrated bilayer profile (``_profile.py``/
``_raster.py``) and transmembrane surface-site sampling (``_placement.py``)
are completely shape-backend-agnostic already, reading the result only
through :class:`~specter.specimen.membrane._field.MembraneField`'s public
contract.

Algorithm
---------
1. Draw random real spherical-harmonic coefficients ``a_lm`` for
   ``l = 2 .. sh_max_degree`` (``l=0`` is a fixed baseline radius and ``l=1``
   is a pure translation -- both carry no shape information, so neither is
   randomized), with variance falling off as a power of ``l(l+1)`` (see
   ``_sample_sh_coefficients``).
2. Synthesize the resulting harmonic perturbation on a small, regular
   ``(n_theta, n_phi)`` angular grid rather than at every working-grid voxel
   directly -- evaluating ``scipy.special.sph_harm_y_all`` at every voxel's
   own direction (one voxel = one distinct, essentially never-repeated
   direction) does not reuse anything between voxels and was measured to
   dominate `generate()`'s cost, ~70s at a realistic ~10M-voxel working grid.
   Since the perturbation itself only contains information up to degree
   `sh_max_degree`, it is fully resolved by a MUCH smaller number of angular
   samples (see ``_angular_grid_resolution``); every voxel's own value is
   then obtained by bilinear interpolation of that small grid
   (``_interpolate_angular_grid``, via ``torch.nn.functional.grid_sample``,
   with a manual phi-periodic pad since ``grid_sample`` has no circular
   mode). Measured directly: interpolation error is ~0.17% of the
   perturbation's own RMS scale at the default resolution -- an order of
   magnitude below the ``distance_transform_edt`` discretization noise step
   3 already introduces -- while cutting wall time by 30-150x (more at
   larger voxel counts, since the angular synthesis cost is now fixed
   regardless of how many voxels read from it).
3. For every working-grid point, transform into a unit-sphere frame scaled by
   the organelle's physical semi-axes ``sh_axes_a`` (anisotropic axes give
   elongated/flattened organelles), and test -- using the interpolated
   perturbation from step 2 -- whether the point's own radius in that frame
   is less than the harmonic-perturbed surface radius at the point's own
   direction.
4. Feed the resulting boolean solid mask through
   ``scipy.ndimage.distance_transform_edt`` (via ``_signed_distance_
   transform``) to get a real, physical-Angstrom, Eikonal-
   respecting (``|grad(phi)| ~= 1``) signed distance field -- never an
   analytic-but-approximate radial-residual SDF, whose error grows with the
   surface's local slope and would distort the physically-calibrated
   ``bilayer_thickness_a`` precisely in the high-frequency, biologically
   interesting regions.

No mesh is ever constructed.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import sph_harm_y, sph_harm_y_all

from ._field import MembraneField, _grid_points_xyz, _signed_distance_transform

# _placement.py's Newton projection (`x - phi(x) * normal(x)`) only converges
# reliably if phi closely satisfies the Eikonal property near the surface --
# true by construction for distance_transform_edt output, but only as
# well-resolved as the grid.
_MIN_RELIABLE_VOXELS_PER_RADIUS = 8.0


def _real_from_complex(y: np.ndarray, order: int) -> np.ndarray:
    """Standard real combination of a complex ``Y_l^m`` value for order ``m``:
    ``m=0`` -> real part as-is; ``m>0`` -> ``sqrt(2)*(-1)^m*Re``; the ``m<0``
    case is handled by callers passing in ``Y_l^{|m|}`` and taking the
    imaginary part via this same ``(-1)^m`` sign (see ``_real_spherical_harmonic``
    and ``_real_spherical_harmonics_grid``, which both funnel through here so
    there is exactly one place this convention is encoded)."""
    if order == 0:
        return np.real(y)
    if order > 0:
        return np.sqrt(2.0) * (-1.0) ** order * np.real(y)
    return np.sqrt(2.0) * (-1.0) ** order * np.imag(y)


def _real_spherical_harmonic(
    degree: int, order: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Real-valued, orthonormal-on-S^2 spherical harmonic ``Y_l^m``.

    Built from a single ``scipy.special.sph_harm_y`` call using the standard
    real combination of the +-m complex pair (see ``_real_from_complex``).
    Verified numerically (not just taken from a remembered formula) against
    the closed form ``Y_2^0(theta) = (1/4) * sqrt(5/pi) * (3*cos(theta)**2 - 1)``
    (max absolute error ~1e-16) and against Monte-Carlo orthonormality on the
    sphere.

    Intended for one-off/test use. Direct per-point evaluation like this is
    NOT what ``generate_membrane_field_spherical_harmonics`` uses on its
    working grid: even the batched ``_real_spherical_harmonics_grid`` (a
    single ``scipy.special.sph_harm_y_all`` call for the whole degree/order
    grid, ~5x faster than one ``sph_harm_y`` call per ``(l, m)`` pair) still
    costs O(voxel count), which dominates `generate()` at realistic working-
    grid sizes (~70s measured at ~10M voxels). ``generate_membrane_field_
    spherical_harmonics`` instead evaluates the harmonic sum on a small
    angular grid once and interpolates (see ``_synthesize_angular_grid``/
    ``_interpolate_angular_grid``), a further 30-150x faster. This function
    and ``_real_spherical_harmonics_grid`` remain the ground truth those are
    tested against.

    Parameters
    ----------
    degree : int
        Degree ``l >= 0``.
    order : int
        Order ``m``, ``-l <= m <= l``.
    theta : np.ndarray
        Polar angle, radians, ``0`` at the ``+z`` pole, range ``[0, pi]``.
    phi : np.ndarray
        Azimuthal angle, radians, range ``[0, 2*pi)``.

    Returns
    -------
    np.ndarray
        Real spherical harmonic values, same shape as ``theta``/``phi``.

    Notes
    -----
    ``scipy.special.sph_harm_y(n, m, theta, phi)`` (the current, non-
    deprecated API) takes ``theta`` as the POLAR angle and ``phi`` as the
    azimuthal angle -- verified here via ``Y_0^0 == 1/sqrt(4*pi)`` and
    ``Y_1^0(theta, phi) / cos(theta)`` being the constant
    ``sqrt(3 / (4*pi))``. This is the OPPOSITE argument convention from the
    older, deprecated ``scipy.special.sph_harm(m, n, theta, phi)``, where
    ``theta`` was azimuthal -- a real trap if the two are confused.
    """
    m_abs = order if order >= 0 else -order
    return _real_from_complex(sph_harm_y(degree, m_abs, theta, phi), order)


def _real_spherical_harmonics_grid(
    max_degree: int, theta: np.ndarray, phi: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    """
    Real spherical harmonics for every ``(l, m)`` with ``2 <= l <= max_degree``,
    ``-l <= m <= l``, computed in a single batched
    ``scipy.special.sph_harm_y_all`` call rather than one ``sph_harm_y`` call
    per ``(l, m)`` pair -- ``sph_harm_y`` itself has large fixed per-call
    overhead independent of array size (measured directly: a lone call over
    512,000 points takes ~0.07s in isolation, but ~0.25s/call when called 77
    times back-to-back for varying degree, ~19s total), whereas
    ``sph_harm_y_all`` computes the whole degree/order grid via one shared
    recurrence (~3.5s total for the same `max_degree=8` grid -- verified
    directly, not assumed, and cross-checked element-by-element against
    ``sph_harm_y`` for every `(l, m)` up to `max_degree=8`, exact match).

    ``sph_harm_y_all(n, m, theta, phi)`` returns a dense
    ``(n+1, 2*m+1, ...)`` array whose order axis is NOT simple ascending
    ``-m..m`` -- it is FFT-style wraparound order: columns ``0..m`` hold
    orders ``0..m``, and columns ``m+1..2*m`` hold orders ``-m..-1``
    (verified numerically here, not documented explicitly by scipy). This
    function never reads those wraparound (negative-order) columns, though:
    exactly like ``_real_spherical_harmonic``, the real-SH combination for a
    NEGATIVE order ``m`` is built from the POSITIVE-order complex value
    ``Y_l^{|m|}`` (see ``_real_from_complex``), not from ``Y_l^{-|m|}``
    itself -- those differ by an extra ``(-1)^m`` (a real, previously caught
    bug here: using the array's own negative-order column instead gave every
    negative-order real harmonic the wrong sign). So only the trivially
    ascending non-negative columns (index ``== order``) are ever indexed.

    Parameters
    ----------
    max_degree : int
        Highest harmonic degree to include, ``>= 2``.
    theta : np.ndarray
        Polar angle, radians.
    phi : np.ndarray
        Azimuthal angle, radians.

    Returns
    -------
    dict of (int, int) to np.ndarray
        Maps ``(degree, order)`` to its real spherical harmonic values, same
        shape as ``theta``/``phi``.
    """
    all_complex = sph_harm_y_all(max_degree, max_degree, theta, phi)
    real: dict[tuple[int, int], np.ndarray] = {}
    for degree in range(2, max_degree + 1):
        for order in range(-degree, degree + 1):
            m_abs = order if order >= 0 else -order
            real[(degree, order)] = _real_from_complex(
                all_complex[degree, m_abs], order
            )
    return real


def _angular_grid_resolution(max_degree: int) -> tuple[int, int]:
    """
    ``(n_theta, n_phi)`` for the coarse angular grid the harmonic
    perturbation is synthesized on before interpolation (see
    ``_synthesize_angular_grid``).

    ``16 * max_degree`` (minimum 32) polar samples, doubled for azimuthal --
    a ~16x oversample over the bare Nyquist floor for exact spectral
    reconstruction (``max_degree + 1`` polar samples). The extra margin is
    for plain bilinear interpolation's own accuracy (it is not an exact
    spectral resynthesis), verified directly: at ``max_degree=8`` this gives
    a 128x256 grid, measured to have ~0.17% max error relative to direct
    per-point evaluation -- an order of magnitude below the
    ``distance_transform_edt`` discretization noise already present
    downstream.
    """
    n_theta = max(32, 16 * max_degree)
    return n_theta, 2 * n_theta


def _synthesize_angular_grid(
    coefficients: list[tuple[int, int, float]],
    max_degree: int,
    n_theta: int,
    n_phi: int,
) -> np.ndarray:
    """
    Evaluate the harmonic perturbation ``sum(a_lm * Y_l^m)`` on a regular
    ``(n_theta, n_phi)`` lat-lon grid (``theta`` in ``[0, pi]``, ``phi`` in
    ``[0, 2*pi)``), returning it right-padded with a copy of the ``phi=0``
    column at ``phi=2*pi`` -- shape ``(n_theta, n_phi + 1)`` -- so the
    result is ready for non-periodic bilinear interpolation across the
    wraparound seam (see ``_interpolate_angular_grid``).
    """
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tt, pp = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    real_sh = _real_spherical_harmonics_grid(max_degree, tt.ravel(), pp.ravel())
    coarse = np.zeros(tt.shape, dtype=np.float64)
    for degree, order, coeff in coefficients:
        coarse += coeff * real_sh[(degree, order)].reshape(tt.shape)
    return np.concatenate([coarse, coarse[:, :1]], axis=1)


def _interpolate_angular_grid(
    coarse_padded: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    device: str | torch.device,
) -> np.ndarray:
    """
    Bilinearly interpolate ``_synthesize_angular_grid``'s ``(n_theta,
    n_phi + 1)`` output at arbitrary ``(theta, phi)`` points via
    ``torch.nn.functional.grid_sample`` -- ``phi`` is wrapped into
    ``[0, 2*pi)`` first (the padding already makes that range
    interpolate correctly across the seam); ``theta`` clamps at the poles
    via ``padding_mode="border"`` (exact at ``theta`` exactly 0/pi, and
    the grid already spans the full ``[0, pi]`` range so this only matters
    for floating-point boundary roundoff).
    """
    coarse_t = torch.as_tensor(coarse_padded, dtype=torch.float32, device=device)[
        None, None
    ]
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    phi_t = torch.as_tensor(
        np.mod(phi, 2.0 * np.pi), dtype=torch.float32, device=device
    )
    norm_theta = (theta_t / np.pi) * 2.0 - 1.0
    norm_phi = (phi_t / (2.0 * np.pi)) * 2.0 - 1.0
    grid = torch.stack([norm_phi, norm_theta], dim=-1).reshape(1, -1, 1, 2)

    sampled = F.grid_sample(
        coarse_t, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.reshape(-1).cpu().numpy()


def _sample_sh_coefficients(
    max_degree: int, spectrum_power: float, rng: np.random.Generator
) -> list[tuple[int, int, float]]:
    """
    Draw random real spherical-harmonic coefficients with a physically
    motivated undulation spectrum.

    ``a_lm ~ Normal(0, 1 / [l*(l+1)]**spectrum_power)`` for
    ``l = 2 .. max_degree``, ``m = -l .. l`` -- ``spectrum_power=2.0``
    reproduces the Helfrich (1973) thermal equilibrium bending-mode
    spectrum of a near-equilibrium lipid bilayer, where long-wavelength
    modes dominate and short-wavelength wrinkles are naturally suppressed.
    ``l=0`` (a fixed baseline radius) and ``l=1`` (a pure translation, no
    shape information) are excluded -- both would be wasted degrees of
    freedom for shape control.

    The full coefficient set is then rescaled so ``sum(a_lm**2) == 1``
    exactly: since the ``Y_l^m`` are orthonormal on the sphere, this makes
    the RMS of the resulting radius perturbation over the sphere exactly 1,
    deterministically (Parseval), regardless of ``max_degree`` or
    ``spectrum_power`` -- so a caller's amplitude parameter alone controls
    the perturbation's overall scale.

    Parameters
    ----------
    max_degree : int
        Highest harmonic degree to include, ``>= 2``.
    spectrum_power : float
        Exponent ``p`` in ``Var(a_lm) ~ 1 / [l*(l+1)]**p``.
    rng : np.random.Generator

    Returns
    -------
    list of (int, int, float)
        ``(degree, order, coefficient)`` triples.
    """
    coeffs: list[tuple[int, int, float]] = []
    for degree in range(2, max_degree + 1):
        sigma = 1.0 / (degree * (degree + 1)) ** (spectrum_power / 2.0)
        for order in range(-degree, degree + 1):
            coeffs.append((degree, order, float(rng.normal(0.0, sigma))))
    sum_of_squares = sum(a * a for _, _, a in coeffs)
    norm = float(np.sqrt(sum_of_squares)) if sum_of_squares > 0 else 1.0
    return [(degree, order, a / norm) for degree, order, a in coeffs]


def generate_membrane_field_spherical_harmonics(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    sh_max_degree: int = 8,
    sh_axes_a: tuple[float, float, float] = (300.0, 300.0, 300.0),
    sh_amplitude: float = 0.15,
    sh_spectrum_power: float = 2.0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> MembraneField:
    """
    Generate an organic membrane mid-surface field from a random spherical-
    harmonic-perturbed ellipsoid, in the same :class:`MembraneField`
    representation every other shape backend produces.

    Parameters
    ----------
    shape_zyx : tuple of int
        Working grid shape, ``(Z, Y, X)``. Should cover the intended
        membrane instance's bounding box with margin, matching the other
        backends' own convention.
    spacing_a : float
        Working grid voxel spacing, Angstrom. Independent of any downstream
        output voxel size.
    sh_max_degree : int, optional
        Highest spherical-harmonic degree included in the random surface
        perturbation. Higher values add finer surface detail (see
        ``_sample_sh_coefficients``). Default 8.
    sh_axes_a : tuple of float, optional
        Physical semi-axes ``(a_x, a_y, a_z)`` of the base ellipsoid,
        Angstrom -- isotropic (equal) axes give a roughly spherical
        organelle, anisotropic axes give an elongated/flattened one (e.g.
        a stretched mitochondrion). Default ``(300.0, 300.0, 300.0)``.
    sh_amplitude : float, optional
        RMS fractional radius perturbation (dimensionless, relative to the
        local semi-axis) -- exactly defined via the Parseval-normalized
        coefficient set in ``_sample_sh_coefficients``, so this parameter
        alone controls the perturbation's scale independent of
        `sh_max_degree`/`sh_spectrum_power`. Default 0.15, chosen from a
        direct visual sweep (0.10/0.15/0.25/0.40, see ``dev/
        sh_membrane_examples.py``): 0.10 was barely distinguishable from a
        sphere, 0.40 produced visible concave dimples (still a valid,
        non-degenerate star-convex surface -- `R_prime`'s positive floor
        holds even there -- but an unusually irregular-looking organelle);
        0.15 was the smallest value that read as clearly organic/non-
        spherical without any concavity artifacts.
    sh_spectrum_power : float, optional
        Exponent ``p`` in ``Var(a_lm) ~ 1 / [l*(l+1)]**p``. ``2.0`` matches
        the Helfrich (1973) thermal bending-mode spectrum of a lipid
        bilayer at equilibrium (long-wavelength modes dominate). Lower
        values give relatively more high-frequency (finer, less physically
        "membrane-like") roughness. Default 2.0.
    device : str or torch.device, optional
        Device for the returned field. Default ``"cpu"``.
    seed : int, optional
        Random seed. Default ``None``.

    Notes
    -----
    No blur-then-rethreshold surface-smoothing step is needed here: the
    harmonic-perturbed surface is smooth by construction (no faceting is
    ever introduced), unlike a point-cloud/mesh-based shape construction.

    The harmonic perturbation itself is synthesized once on a small
    ``_angular_grid_resolution(sh_max_degree)`` angular grid and bilinearly
    interpolated at every working-grid voxel (see this module's own
    Algorithm docstring) rather than evaluated directly per voxel -- ~0.17%
    interpolation error relative to the perturbation's own RMS scale at the
    default resolution, well below the ``distance_transform_edt``
    discretization noise already present, for a measured 30-150x wall-time
    reduction depending on working-grid size.

    The same Eikonal/grid-resolution caveat applies to
    `_placement.sample_surface_sites`'s Newton-projection surface search --
    see this function's own warning below.

    Returns
    -------
    MembraneField
    """
    if sh_max_degree < 2:
        raise ValueError(f"sh_max_degree must be >= 2, got {sh_max_degree}")

    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a

    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    query = points_xyz.reshape(-1, 3).numpy()

    axes = np.asarray(sh_axes_a, dtype=np.float64)
    p_prime = query / axes
    r_prime = np.linalg.norm(p_prime, axis=-1)
    r_safe = np.clip(r_prime, 1e-12, None)
    theta = np.arccos(np.clip(p_prime[:, 2] / r_safe, -1.0, 1.0))
    phi_ang = np.arctan2(p_prime[:, 1], p_prime[:, 0])

    rng = np.random.default_rng(seed)
    coefficients = _sample_sh_coefficients(sh_max_degree, sh_spectrum_power, rng)

    n_theta, n_phi = _angular_grid_resolution(sh_max_degree)
    coarse_padded = _synthesize_angular_grid(
        coefficients, sh_max_degree, n_theta, n_phi
    )
    perturbation = _interpolate_angular_grid(
        coarse_padded, theta, phi_ang, device="cpu"
    )

    r_surface = np.clip(1.0 + sh_amplitude * perturbation, 0.05, None)
    inside = (r_prime < r_surface).reshape(shape_zyx)

    # See _signed_distance_transform's own docstring for why this is
    # GPU-accelerated (optionally) rather than always scipy.
    phi = _signed_distance_transform(inside, spacing_a, device)

    voxels_per_radius = min(sh_axes_a) / spacing_a
    if voxels_per_radius < _MIN_RELIABLE_VOXELS_PER_RADIUS:
        warnings.warn(
            f"generate_membrane_field_spherical_harmonics: min(sh_axes_a) "
            f"({min(sh_axes_a):.1f} A) is only {voxels_per_radius:.1f} "
            f"working-grid voxels at spacing_a={spacing_a:.2f} A -- below the "
            f"{_MIN_RELIABLE_VOXELS_PER_RADIUS:.0f} voxels/radius verified reliable "
            "for sample_surface_sites' Newton-projection surface sampling. "
            "Transmembrane placement may silently find zero sites at this "
            "resolution. Increase sh_axes_a, or decrease spacing_a (e.g. a "
            "smaller v_size on the owning MembraneGenerator), to raise this "
            "ratio.",
            stacklevel=2,
        )

    clipped = (
        inside[0].any()
        or inside[-1].any()
        or inside[:, 0].any()
        or inside[:, -1].any()
        or inside[:, :, 0].any()
        or inside[:, :, -1].any()
    )
    if clipped:
        warnings.warn(
            "generate_membrane_field_spherical_harmonics: the organelle's solid "
            "interior touches the working grid's boundary -- the shape is being "
            "hard-clipped by shape_zyx rather than tapering to zero inside it "
            "(same failure mode the other shape backends' own boundary checks "
            "guard against). Increase shape_zyx (or the owning MembraneGenerator's "
            "target_shape_zyx/v_size), or reduce sh_axes_a/sh_amplitude.",
            stacklevel=2,
        )

    return MembraneField(
        phi=phi,
        spacing_a=spacing_a,
        origin_xyz=origin_xyz.to(device),
        clipped_at_boundary=clipped,
    )


__all__ = ["generate_membrane_field_spherical_harmonics"]
