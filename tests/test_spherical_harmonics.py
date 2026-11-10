"""The real spherical-harmonic helpers behind the gold beads' roughness."""

import numpy as np
import torch


def test_real_spherical_harmonic_matches_known_l2_m0_and_is_orthonormal():
    """Standalone check of the real-SH basis convention itself, independent
    of the field-generation pipeline: scipy.special.sph_harm_y's argument
    order/convention (theta=polar, phi=azimuthal in this scipy version) is
    easy to get backwards against the older, deprecated scipy.special.sph_harm
    (theta=azimuthal there) -- verify against a closed form and via
    Monte-Carlo orthonormality rather than trusting the convention blindly."""
    from specter.specimen._spherical_harmonics import (
        _real_spherical_harmonic,
    )

    theta = torch.tensor([0.3, 0.7, 1.2, 2.0]).numpy()
    phi = torch.tensor([0.1, 1.0, 2.5, 4.0]).numpy()
    y20 = _real_spherical_harmonic(2, 0, theta, phi)
    expected = 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * np.cos(theta) ** 2 - 1.0)
    assert np.allclose(y20, expected, atol=1e-10)

    rng = np.random.default_rng(0)
    n = 200_000
    mc_theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    mc_phi = rng.uniform(0.0, 2.0 * np.pi, n)
    y21 = _real_spherical_harmonic(2, 1, mc_theta, mc_phi)
    y2neg1 = _real_spherical_harmonic(2, -1, mc_theta, mc_phi)
    # <Y|Y'> = (1/4pi) * integral over solid angle -- Monte Carlo estimate
    # via the mean over uniformly-sampled directions.
    assert abs(np.mean(y21 * y21) * 4.0 * np.pi - 1.0) < 0.05
    assert abs(np.mean(y21 * y2neg1) * 4.0 * np.pi) < 0.05


def test_real_spherical_harmonics_grid_matches_single_call_for_every_l_m():
    """Regression test for the batched sph_harm_y_all optimization: callers use
    _real_spherical_harmonics_grid (one scipy.special.sph_harm_y_all call for
    the whole degree/order grid, ~5x faster) instead of one
    _real_spherical_harmonic (scipy.special.sph_harm_y) call per (l, m) --
    sph_harm_y_all's order axis uses FFT-style wraparound indexing, not
    simple ascending -m..m, which _real_spherical_harmonics_grid's _column
    helper decodes. Verify the two paths agree exactly for every (l, m) up
    to a representative max_degree, not just spot-checked ones."""
    from specter.specimen._spherical_harmonics import (
        _real_spherical_harmonic,
        _real_spherical_harmonics_grid,
    )

    rng = np.random.default_rng(1)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 500))
    phi = rng.uniform(0.0, 2.0 * np.pi, 500)
    max_degree = 6

    batched = _real_spherical_harmonics_grid(max_degree, theta, phi)
    for degree in range(2, max_degree + 1):
        for order in range(-degree, degree + 1):
            single = _real_spherical_harmonic(degree, order, theta, phi)
            assert np.allclose(batched[(degree, order)], single, atol=1e-10)


def test_coarse_angular_grid_interpolation_matches_direct_evaluation():
    """Regression test for the coarse-angular-grid + bilinear-interpolation
    optimization: a harmonic perturbation is not evaluated directly at every
    voxel (O(voxel count)) -- it is synthesized once on a small
    _angular_grid_resolution(max_degree) angular grid and bilinearly
    interpolates (torch.nn.functional.grid_sample, with a manual phi-
    periodic pad). Verify the interpolated result stays close to direct
    per-point evaluation (the ground truth _real_spherical_harmonics_grid
    itself is separately verified above) -- measured directly at
    max_degree=8's default 128x256 grid: ~0.17% max error relative to the
    perturbation's own RMS=1 scale. Assert a generous bound (1%) so this
    catches a real regression (e.g. a broken periodic wrap or a badly
    undersized grid) without being sensitive to incidental resampling noise."""
    from specter.specimen._spherical_harmonics import (
        _angular_grid_resolution,
        _interpolate_angular_grid,
        _real_spherical_harmonics_grid,
        _sample_sh_coefficients,
        _synthesize_angular_grid,
    )

    max_degree = 8
    rng = np.random.default_rng(2)
    coefficients = _sample_sh_coefficients(max_degree, 2.0, rng)

    n_check = 20_000
    theta = np.arccos(rng.uniform(-1.0, 1.0, n_check))
    phi = rng.uniform(0.0, 2.0 * np.pi, n_check)

    real_sh = _real_spherical_harmonics_grid(max_degree, theta, phi)
    direct = np.zeros(n_check)
    for degree, order, coeff in coefficients:
        direct += coeff * real_sh[(degree, order)]

    n_theta, n_phi = _angular_grid_resolution(max_degree)
    coarse_padded = _synthesize_angular_grid(coefficients, max_degree, n_theta, n_phi)
    interpolated = _interpolate_angular_grid(coarse_padded, theta, phi, device="cpu")

    max_err = np.abs(interpolated - direct).max()
    assert max_err < 0.01, f"interpolation max error {max_err} exceeds 1% bound"


def test_interpolate_angular_grid_returns_tensor_for_tensor_input() -> None:
    """
    `_interpolate_angular_grid` returns the array type it was handed.

    A caller may pass torch tensors so a large grid never leaves the GPU;
    `specimen/_grid.py`'s bead roughness passes NumPy. Pin the contract rather
    than the current caller.
    """
    from specter.specimen._spherical_harmonics import (
        _angular_grid_resolution,
        _interpolate_angular_grid,
        _sample_sh_coefficients,
        _synthesize_angular_grid,
    )

    rng = np.random.default_rng(0)
    coefficients = _sample_sh_coefficients(6, 2.0, rng)
    n_theta, n_phi = _angular_grid_resolution(6)
    coarse = _synthesize_angular_grid(coefficients, 6, n_theta, n_phi)

    theta = rng.uniform(0.0, np.pi, 512)
    phi = rng.uniform(-np.pi, 3.0 * np.pi, 512)  # spans the wrap on both sides

    from_numpy = _interpolate_angular_grid(coarse, theta, phi, device="cpu")
    from_torch = _interpolate_angular_grid(
        coarse, torch.as_tensor(theta), torch.as_tensor(phi), device="cpu"
    )

    assert isinstance(from_numpy, np.ndarray)
    assert torch.is_tensor(from_torch)
    assert torch.allclose(
        torch.as_tensor(from_numpy, dtype=torch.float32), from_torch, atol=1e-6
    )
