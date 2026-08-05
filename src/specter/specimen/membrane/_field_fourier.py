"""
Fourier-surface membrane shape backend.

Same star-convex "radius function of direction" family as
``_field_spherical_harmonics.py``, but with a cheap sin/cos basis instead of
real spherical harmonics ``Y_l^m`` -- no ``scipy.special`` call at all, pure
``numpy``/``torch`` elementwise trig, aimed at being the fastest of the
star-convex backends. Trades away real spherical harmonics' exact
orthonormality and its direct physical (Helfrich) spectral interpretation
for that speed.

Pole regularity
----------------
A naive 2D Fourier series in ``(theta, phi)`` is NOT automatically smooth at
the poles (``theta = 0, pi``): every azimuthal angle ``phi`` maps to the same
physical point there, so any term depending on ``phi`` must vanish at the
poles, and vanish to the RIGHT ORDER, or it leaves a visible pinch/dimple
artifact. This mirrors exactly why the associated Legendre functions inside
real spherical harmonics carry a ``sin(theta)^|m|`` factor. Verified by local
Cartesian expansion near a pole (`u = theta*cos(phi)`, `v = theta*sin(phi)`):
a term ``sin(theta)**abs(f_phi) * cos(f_phi*phi + phase)`` reduces to a
homogeneous degree-``|f_phi|`` polynomial in ``(u, v)`` -- exactly the
"solid harmonic" (``r^l * Y_l^m``) smoothness argument, and genuinely
smooth at the origin. A flat ``sin(theta)`` (power 1 regardless of
``f_phi``) does NOT have this property for ``|f_phi| >= 2`` and leaves a
real, angle-dependent artifact right at the pole -- confirmed by symbolic
expansion, not just reasoned about.

Each term is therefore SEPARABLE (theta-only factor times a pole-damped
phi-only factor), not one additive ``cos(f_theta*theta + f_phi*phi)``
argument -- mirrors real spherical harmonics' own
``P_l^m(cos theta) * trig(m*phi)`` structure and avoids a residual
curvature kink at the poles that the additive form still has even after
correct damping (traced via angle-addition expansion: the cross term
reduces to ``theta * (smooth function)``, and ``theta = sqrt(u**2+v**2)``
itself isn't smooth at the origin, only continuous).

``f_phi`` MUST be an integer (else a hard discontinuity at the
``phi=0/2*pi`` seam, verified via the boundary-continuity regression test in
``tests/test_membrane_generator.py``); ``f_theta`` has no periodicity
constraint since ``theta`` only spans ``[0, pi]`` once, no wraparound.
Zonal terms (``f_phi == 0``) need no pole damping at all -- automatically
regular there regardless of ``f_theta``.

Normalization
-------------
Unlike real spherical harmonics, this basis is NOT exactly orthonormal on
the sphere (random phases, and the ``sin(theta)^|f_phi|`` damping factor
itself breaks the exact orthonormality real ``Y_l^m`` has) -- so, unlike
``_sample_sh_coefficients``'s exact Parseval renormalization, this module
normalizes the perturbation's scale EMPIRICALLY: evaluate it on a modest
Monte-Carlo sample of random directions, measure the actual RMS, rescale.
Honest for a non-orthonormal basis rather than claiming an analytic
guarantee that doesn't hold here.

Same exact star-convex boolean solid test + ``scipy.ndimage.
distance_transform_edt`` construction as ``_field_spherical_harmonics.py``
-- see that module's own docstring for why (an exact test, never an
approximate radial-residual SDF, to avoid distorting the calibrated
``bilayer_thickness_a`` near the surface).

Direct-vs-interpolated evaluation
----------------------------------
Measured directly (not assumed, despite ``cos``/``sin`` having none of
``scipy.special.sph_harm_y``'s fixed per-call overhead): direct per-voxel
evaluation of `fourier_n_terms=20` terms took ~14s at a realistic ~9.8M-
voxel working grid -- transcendental functions are simply not free at that
scale, over ~100 elementwise passes. So this module reuses
``_field_spherical_harmonics.py``'s coarse-angular-grid-synthesis +
bilinear-interpolation trick after all (``_interpolate_angular_grid`` is
imported directly from there rather than duplicated -- unlike the small
per-module constants/warnings this package otherwise duplicates by
convention, this is non-trivial, easy-to-get-subtly-wrong numerical
machinery (a wraparound-indexing bug was caught here during the SH
backend's own development), so a second independent copy would be a real
regression risk for no benefit; only the basis EVALUATION itself, which is
genuinely backend-specific, is implemented locally).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy import ndimage

from ._field import MembraneField, _grid_points_xyz
from ._field_spherical_harmonics import _interpolate_angular_grid

# Same underlying cause as _field_spherical_harmonics.py's own identically-
# named constant -- duplicated rather than shared, matching this package's
# established fully-independent-module convention.
_MIN_RELIABLE_VOXELS_PER_RADIUS = 8.0


def _sample_fourier_terms(
    n_terms: int,
    max_theta_freq: float,
    max_phi_freq: int,
    spectrum_power: float,
    rng: np.random.Generator,
) -> list[tuple[float, int, float, float, float]]:
    """
    Draw random Fourier terms with a power-law undulation spectrum.

    Each term is ``(f_theta, f_phi, phase_theta, phase_phi, coeff)``.
    ``f_theta`` is drawn uniformly in ``[0, max_theta_freq]`` (continuous --
    no periodicity constraint). ``f_phi`` is drawn as a uniform random
    integer in ``[-max_phi_freq, max_phi_freq]`` (must be integer, see
    module docstring). ``coeff ~ Normal(0, 1 / (1 + sqrt(f_theta**2 +
    f_phi**2))**spectrum_power)`` -- higher combined frequency terms get
    smaller typical amplitude, the Fourier analogue of
    ``_sample_sh_coefficients``'s ``1 / [l*(l+1)]**p`` falloff.

    Parameters
    ----------
    n_terms : int
        Number of random terms to draw.
    max_theta_freq : float
        Upper bound on ``f_theta``.
    max_phi_freq : int
        Upper bound (absolute value) on the integer ``f_phi``.
    spectrum_power : float
        Power-law falloff exponent.
    rng : np.random.Generator

    Returns
    -------
    list of (float, int, float, float, float)
    """
    terms: list[tuple[float, int, float, float, float]] = []
    for _ in range(n_terms):
        f_theta = float(rng.uniform(0.0, max_theta_freq))
        f_phi = int(rng.integers(-max_phi_freq, max_phi_freq + 1))
        phase_theta = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_phi = float(rng.uniform(0.0, 2.0 * np.pi))
        combined_freq = float(np.sqrt(f_theta**2 + f_phi**2))
        sigma = 1.0 / (1.0 + combined_freq) ** spectrum_power
        coeff = float(rng.normal(0.0, sigma))
        terms.append((f_theta, f_phi, phase_theta, phase_phi, coeff))
    return terms


def _evaluate_fourier_perturbation(
    terms: list[tuple[float, int, float, float, float]],
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Evaluate ``sum(coeff_i * term_i(theta, phi))`` directly at every given
    point -- see module docstring for the separable, pole-damped form of
    each term.

    Used directly for the Monte-Carlo RMS-normalization probe
    (``_empirical_rms_normalize``) and for synthesizing the coarse angular
    grid (``_synthesize_angular_grid``) -- NOT called per working-grid voxel
    (see module docstring's "Direct-vs-interpolated evaluation" section for
    why: measured directly to be too slow at realistic voxel counts).

    Parameters
    ----------
    terms : list of (float, int, float, float, float)
        Output of ``_sample_fourier_terms``.
    theta : np.ndarray
        Polar angle, radians.
    phi : np.ndarray
        Azimuthal angle, radians.

    Returns
    -------
    np.ndarray
        Perturbation values, same shape as ``theta``/``phi``.
    """
    total = np.zeros_like(theta)
    for f_theta, f_phi, phase_theta, phase_phi, coeff in terms:
        theta_factor = np.cos(f_theta * theta + phase_theta)
        if f_phi == 0:
            total += coeff * theta_factor
        else:
            pole_damping = np.sin(theta) ** abs(f_phi)
            phi_factor = np.cos(f_phi * phi + phase_phi)
            total += coeff * theta_factor * pole_damping * phi_factor
    return total


def _angular_grid_resolution(
    max_theta_freq: float, max_phi_freq: int
) -> tuple[int, int]:
    """
    ``(n_theta, n_phi)`` for the coarse angular grid the perturbation is
    synthesized on before interpolation -- mirrors
    ``_field_spherical_harmonics._angular_grid_resolution``'s own ``16x``
    oversample-over-Nyquist reasoning, using the larger of the two
    frequency bounds as the effective "degree".
    """
    max_freq = max(max_theta_freq, max_phi_freq)
    n_theta = max(32, int(16 * max_freq))
    return n_theta, 2 * n_theta


def _synthesize_angular_grid(
    terms: list[tuple[float, int, float, float, float]],
    n_theta: int,
    n_phi: int,
) -> np.ndarray:
    """
    Evaluate ``_evaluate_fourier_perturbation`` on a regular ``(n_theta,
    n_phi)`` lat-lon grid, right-padded with a copy of the ``phi=0`` column
    at ``phi=2*pi`` -- shape ``(n_theta, n_phi + 1)`` -- ready for
    ``_interpolate_angular_grid``.
    """
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tt, pp = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    coarse = _evaluate_fourier_perturbation(terms, tt.ravel(), pp.ravel()).reshape(
        tt.shape
    )
    return np.concatenate([coarse, coarse[:, :1]], axis=1)


def _empirical_rms_normalize(
    terms: list[tuple[float, int, float, float, float]],
    rng: np.random.Generator,
    n_probe: int = 20_000,
) -> list[tuple[float, int, float, float, float]]:
    """
    Rescale ``terms``' coefficients so the perturbation has RMS 1 over the
    sphere, measured empirically (Monte-Carlo probe directions) rather than
    assumed analytically -- see module docstring for why this basis isn't
    exactly orthonormal like real spherical harmonics.
    """
    probe_theta = np.arccos(rng.uniform(-1.0, 1.0, n_probe))
    probe_phi = rng.uniform(0.0, 2.0 * np.pi, n_probe)
    values = _evaluate_fourier_perturbation(terms, probe_theta, probe_phi)
    rms = float(np.sqrt(np.mean(values**2)))
    if rms <= 1e-12:
        return terms
    return [
        (f_theta, f_phi, phase_theta, phase_phi, coeff / rms)
        for f_theta, f_phi, phase_theta, phase_phi, coeff in terms
    ]


def generate_membrane_field_fourier(
    shape_zyx: tuple[int, int, int],
    spacing_a: float,
    fourier_axes_a: tuple[float, float, float] = (200.0, 200.0, 200.0),
    fourier_n_terms: int = 20,
    fourier_amplitude: float = 0.15,
    fourier_spectrum_power: float = 2.0,
    fourier_max_theta_freq: float = 6.0,
    fourier_max_phi_freq: int = 6,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> MembraneField:
    """
    Generate an organic membrane mid-surface field from a random Fourier-
    perturbed ellipsoid, in the same :class:`MembraneField` representation
    ``generate_membrane_field_spherical_harmonics`` produces.

    Parameters
    ----------
    shape_zyx : tuple of int
        Working grid shape, ``(Z, Y, X)``.
    spacing_a : float
        Working grid voxel spacing, Angstrom.
    fourier_axes_a : tuple of float, optional
        Physical semi-axes ``(a_x, a_y, a_z)`` of the base ellipsoid,
        Angstrom. Default ``(200.0, 200.0, 200.0)``.
    fourier_n_terms : int, optional
        Number of random Fourier terms. Default 20.
    fourier_amplitude : float, optional
        RMS fractional radius perturbation (dimensionless), empirically
        normalized (see ``_empirical_rms_normalize``). Default 0.15, picked
        from a direct visual sweep (0.08/0.15/0.25, see ``dev/
        fourier_sweep.py``) matching the same discipline used for the SH
        backend's `sh_amplitude`: 0.08 was barely distinguishable from a
        sphere, 0.25 clipped the working grid at this module's own default
        `fourier_axes_a` scale (a margin issue, not a sign of an invalid
        surface); 0.15 was clearly organic/non-spherical with no clipping.
    fourier_spectrum_power : float, optional
        Power-law falloff exponent for term amplitude vs. combined
        frequency. Default 2.0.
    fourier_max_theta_freq : float, optional
        Upper bound on each term's polar frequency. Default 6.0.
    fourier_max_phi_freq : int, optional
        Upper bound (absolute value) on each term's integer azimuthal
        frequency. Default 6.
    device : str or torch.device, optional
        Device for the returned field. Default ``"cpu"``.
    seed : int, optional
        Random seed. Default ``None``.

    Returns
    -------
    MembraneField
    """
    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a

    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    query = points_xyz.reshape(-1, 3).numpy()

    axes = np.asarray(fourier_axes_a, dtype=np.float64)
    p_prime = query / axes
    r_prime = np.linalg.norm(p_prime, axis=-1)
    r_safe = np.clip(r_prime, 1e-12, None)
    theta = np.arccos(np.clip(p_prime[:, 2] / r_safe, -1.0, 1.0))
    phi_ang = np.arctan2(p_prime[:, 1], p_prime[:, 0])

    rng = np.random.default_rng(seed)
    terms = _sample_fourier_terms(
        fourier_n_terms,
        fourier_max_theta_freq,
        fourier_max_phi_freq,
        fourier_spectrum_power,
        rng,
    )
    terms = _empirical_rms_normalize(terms, rng)

    n_theta, n_phi = _angular_grid_resolution(
        fourier_max_theta_freq, fourier_max_phi_freq
    )
    coarse_padded = _synthesize_angular_grid(terms, n_theta, n_phi)
    perturbation = _interpolate_angular_grid(
        coarse_padded, theta, phi_ang, device="cpu"
    )

    r_surface = np.clip(1.0 + fourier_amplitude * perturbation, 0.05, None)
    inside = (r_prime < r_surface).reshape(shape_zyx)

    dist_out = ndimage.distance_transform_edt(~inside, sampling=spacing_a)
    dist_in = ndimage.distance_transform_edt(inside, sampling=spacing_a)
    phi_np = dist_out - dist_in

    phi = torch.as_tensor(phi_np, dtype=torch.float32, device=device)

    voxels_per_radius = min(fourier_axes_a) / spacing_a
    if voxels_per_radius < _MIN_RELIABLE_VOXELS_PER_RADIUS:
        warnings.warn(
            f"generate_membrane_field_fourier: min(fourier_axes_a) "
            f"({min(fourier_axes_a):.1f} A) is only {voxels_per_radius:.1f} "
            f"working-grid voxels at spacing_a={spacing_a:.2f} A -- below the "
            f"{_MIN_RELIABLE_VOXELS_PER_RADIUS:.0f} voxels/radius verified reliable "
            "(see generate_membrane_field_alpha_shape's own Notes for the same "
            "underlying Eikonal/grid-resolution reasoning) for "
            "sample_surface_sites' Newton-projection surface sampling. Increase "
            "fourier_axes_a, or decrease spacing_a, to raise this ratio.",
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
            "generate_membrane_field_fourier: the organelle's solid interior "
            "touches the working grid's boundary -- the shape is being hard-"
            "clipped by shape_zyx rather than tapering to zero inside it. "
            "Increase shape_zyx, or reduce fourier_axes_a/fourier_amplitude.",
            stacklevel=2,
        )

    return MembraneField(phi=phi, spacing_a=spacing_a, origin_xyz=origin_xyz.to(device))


__all__ = ["generate_membrane_field_fourier"]
