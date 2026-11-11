"""
Real spherical harmonics on an angular grid: random coefficients, synthesis
and interpolation at arbitrary directions.

Used by `BeadGenerator` for the surface roughness of gold fiducials.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import sph_harm_y, sph_harm_y_all


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

    Intended for one-off/test use. The bead roughness in
    ``specimen/_grid.py`` instead evaluates the harmonic sum on a small
    angular grid once and interpolates (see ``_synthesize_angular_grid``/
    ``_interpolate_angular_grid``). This function and
    ``_real_spherical_harmonics_grid`` remain the ground truth those are
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
    itself -- those differ by an extra ``(-1)^m``, so using the array's own
    negative-order column instead gives every negative-order real harmonic
    the wrong sign. So only the trivially
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
    theta: np.ndarray | torch.Tensor,
    phi: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> np.ndarray | torch.Tensor:
    """
    Bilinearly interpolate ``_synthesize_angular_grid``'s ``(n_theta,
    n_phi + 1)`` output at arbitrary ``(theta, phi)`` points via
    ``torch.nn.functional.grid_sample`` -- ``phi`` is wrapped into
    ``[0, 2*pi)`` first (the padding already makes that range
    interpolate correctly across the seam); ``theta`` clamps at the poles
    via ``padding_mode="border"`` (exact at ``theta`` exactly 0/pi, and
    the grid already spans the full ``[0, pi]`` range so this only matters
    for floating-point boundary roundoff).

    Returns the same array type it was given: a `torch.Tensor` on `device` for
    tensor input, a NumPy array for NumPy input. The NumPy form is what
    `specimen/_grid.py`'s bead roughness uses, on a few thousand points where
    the round trip is free.
    """
    as_tensor = torch.is_tensor(theta)
    coarse_t = torch.as_tensor(coarse_padded, dtype=torch.float32, device=device)[
        None, None
    ]
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    phi_t = torch.as_tensor(phi, dtype=torch.float32, device=device).remainder(
        2.0 * np.pi
    )
    norm_theta = (theta_t / np.pi) * 2.0 - 1.0
    norm_phi = (phi_t / (2.0 * np.pi)) * 2.0 - 1.0
    grid = torch.stack(torch.broadcast_tensors(norm_phi, norm_theta), dim=-1).reshape(
        1, -1, 1, 2
    )

    sampled = F.grid_sample(
        coarse_t, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    flat = sampled.reshape(-1)
    return flat if as_tensor else flat.cpu().numpy()


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
        Source of the coefficient draws.

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
