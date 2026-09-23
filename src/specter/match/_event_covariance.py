"""Detector-event noise estimated independently of specimen signal spectra.

An exclusion cell contains one uniformly localized event with probability q.
Its normalized noise power is 1-q*|W(k)|², where W is the cell transform.
A Gaussian event footprint adds exp(-4*pi²*sigma²*k²). Cell orientation is
averaged: this is an isotropic second-moment model, not an event simulation.
"""

import numpy as np
from scipy.optimize import least_squares

from ._movie_calibration import fit_one


def event_noise_power(k, occupancy, cell_side_A, event_sigma_A=0.0):
    """Unnormalized NPS shape; k in inverse Å, lengths in Å, 0 <= q < 1."""
    k = np.asarray(k, dtype=float)
    if (
        k.ndim != 1
        or not np.isfinite(k).all()
        or np.any(k < 0)
        or not 0 <= occupancy < 1
        or not np.isfinite([cell_side_A, event_sigma_A]).all()
        or cell_side_A <= 0
        or event_sigma_A < 0
    ):
        raise ValueError("invalid detector-event covariance parameters")
    angle = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    cell = np.mean(
        np.sinc(k[:, None] * cell_side_A * np.cos(angle)) ** 2
        * np.sinc(k[:, None] * cell_side_A * np.sin(angle)) ** 2,
        axis=1,
    )
    return (1 - occupancy * cell) * np.exp(-((2 * np.pi * event_sigma_A * k) ** 2))


def exclusion_cell_efficiency(occupancy):
    """DC DQE / low-flux QE for Poisson arrivals and one-event cells.

    q=1-exp(-lambda). DQE0/QE=lambda/(exp(lambda)-1). This assumes
    independent exclusion cells and does not establish a detector's QE.
    """
    if not 0 <= occupancy < 1:
        raise ValueError("occupancy must be in [0,1)")
    if occupancy < 1e-8:
        return 1.0 - occupancy / 2
    return -np.log1p(-occupancy) * (1 - occupancy) / occupancy


def estimate_detector_noise(files, motion_variance):
    """Estimate event parameters from calibration movie covariance only.

    The diagonal independent-noise component is separated from temporally
    correlated solvent/protein components. Only event parameters are exported
    for simulation; neither specimen spectral amplitude is exported. The
    returned measured NPS is diagnostic and not a simulation lookup curve.
    """
    items = [dict(np.load(p, allow_pickle=False)) for p in files]
    if not items or any(str(d["split"]) != "calibration" for d in items):
        raise ValueError("only nonempty calibration movie inputs are allowed")
    edges, k = items[0]["dose_edges"], items[0]["k"]
    for d in items:
        np.testing.assert_allclose(d["dose_edges"], edges)
        np.testing.assert_allclose(d["k"], k)
    cov = np.average(
        [d["cov_outer"] for d in items],
        axis=0,
        weights=[int(d["n_particles"]) for d in items],
    )
    nf = len(edges) - 1
    if nf > 40:
        if nf % 40:
            raise ValueError("frame count must divide into 40 equal-dose groups")
        np.testing.assert_allclose(np.diff(edges), np.diff(edges)[0])
        cov = cov.reshape(len(k), 40, nf // 40, 40, nf // 40).mean((2, 4))
        edges = edges[:: nf // 40]
    noise = np.array(
        [fit_one(c, edges, f, motion_variance)[0][2] for c, f in zip(cov, k)]
    )
    selected = (k > 0.04) & (k < 0.65) & ~((k > 1 / 3.9) & (k < 1 / 3.5)) & (noise > 0)
    if selected.sum() < 8:
        raise ValueError("too few independent noise bands for event calibration")

    def residual(x):
        amplitude, q, side, sigma = x
        prediction = amplitude * event_noise_power(k[selected], q, side, sigma)
        return np.log(prediction / noise[selected])

    amplitude = float(noise[selected].max())
    fit = least_squares(
        residual,
        [amplitude, 0.2, 3, 0.2],
        bounds=([amplitude * 0.1, 0, 0.2, 0], [amplitude * 10, 0.85, 15, 2]),
    )
    if not fit.success:
        raise RuntimeError(f"detector noise calibration failed: {fit.message}")
    amp, q, side, sigma = fit.x
    report = dict(
        noise_amplitude=float(amp),
        cell_occupancy=float(q),
        cell_side_A=float(side),
        event_sigma_A=float(sigma),
        noise_relative_rms=float(np.mean(np.expm1(residual(fit.x)) ** 2) ** 0.5),
        calibration_movies=len(items),
        specimen_spectra_exported=False,
        scope="Effective event covariance; detector response and processing can both contribute.",
    )
    return report, k, noise
