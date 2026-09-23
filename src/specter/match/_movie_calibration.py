"""Estimate solvent mobility from movie covariance, independently of stack PSD.

Spectral amplitudes are nuisance variables in this temporal estimator. They
are discarded: SPECTER ice and atomic potentials supply all simulated spectra.
"""

import itertools
import numpy as np
from scipy.optimize import minimize_scalar


def solvent_background_distance(sim, exp, pixel_size: float, radius_A: float) -> float:
    """Native-resolution solvent PSD distance, withholding 3.9–3.5 Å.

    Inputs are particle stacks, not learned spectra. Unit variance normalization
    means the resulting thickness estimate is conditional on detector and
    specimen models, not an independent absolute-thickness measurement.
    """
    import torch

    n = sim.shape[-1]
    xy = (torch.arange(n, device=sim.device) - n / 2) * pixel_size
    radius = torch.hypot(xy[:, None], xy[None, :])
    from scipy.signal.windows import tukey

    taper = torch.as_tensor(
        np.outer(tukey(n, 0.1), tukey(n, 0.1)), device=sim.device, dtype=sim.dtype
    )
    window = taper * ((radius - radius_A) / 20).clamp(0, 1)
    k = torch.hypot(
        torch.fft.fftfreq(n, pixel_size, device=sim.device)[:, None],
        torch.fft.rfftfreq(n, pixel_size, device=sim.device)[None, :],
    )
    idx = (k / 0.01).long()

    def profile(images):
        images = images.to(sim.device)
        images = (images - images.mean((-2, -1), keepdim=True)) / images.std(
            (-2, -1), keepdim=True, correction=0
        )
        images = (
            images - (images * window).sum((-2, -1), keepdim=True) / window.sum()
        ) * window
        power = torch.fft.rfft2(images).abs().square().mean(0) / window.square().sum()
        counts = torch.bincount(idx.flatten()).clamp_min(1)
        return torch.bincount(idx.flatten(), weights=power.flatten()) / counts

    a, b = profile(sim), profile(exp)
    centers = (torch.arange(len(a), device=sim.device) + 0.5) * 0.01
    selected = (
        (centers > 0.06)
        & (centers < 0.60)
        & ~((centers > 1 / 3.9) & (centers < 1 / 3.5))
    )
    return float(torch.log(a[selected] / b[selected]).square().mean())


def ice_covariance(edges, k, s2):
    """Exact interval-average Brownian covariance for nonoverlapping bins."""
    rate = 2 * np.pi**2 * k * k * s2
    widths = np.diff(edges)
    z = rate * widths
    g = -np.expm1(-z) / z
    gaps = np.maximum(edges[:-1, None] - edges[None, 1:], 0)
    gaps += gaps.T
    c = np.exp(-rate * gaps) * g[:, None] * g[None, :]
    diag = np.where(
        z < 1e-3, 1 - z / 3 + z * z / 12 - z**3 / 60, 2 * (z + np.expm1(-z)) / (z * z)
    )
    np.fill_diagonal(c, diag)
    return c


def bases(edges, k, s2):
    ne = 0.245 * k ** (-1.665) + 2.81
    q = (
        2
        * ne
        / np.diff(edges)
        * (np.exp(-edges[:-1] / (2 * ne)) - np.exp(-edges[1:] / (2 * ne)))
    )
    return np.stack(
        [ice_covariance(edges, k, s2), np.outer(q, q), np.diag(1 / np.diff(edges))]
    )


def solve_nonnegative(gram, rhs):
    """Three-component least squares, enumerating the seven active sets."""
    best = np.zeros(3)
    obj = 0.0
    for count in (1, 2, 3):
        for subset in itertools.combinations(range(3), count):
            idx = np.array(subset)
            try:
                v = np.linalg.solve(gram[np.ix_(idx, idx)], rhs[idx])
            except np.linalg.LinAlgError:
                continue
            if np.min(v) < 0:
                continue
            score = v @ gram[np.ix_(idx, idx)] @ v - 2 * v @ rhs[idx]
            if score < obj:
                obj = score
                best = np.zeros(3)
                best[idx] = v
    return best


def fit_one(cov, edges, k, s2):
    b = bases(edges, k, s2)
    n = len(edges) - 1
    ii, jj = np.triu_indices(n)
    # Covariance sampling errors scale with sqrt(Pii Pjj), not with signal.
    diagonal = np.diag(cov).clip(1e-20)
    err = np.sqrt(diagonal[ii] * diagonal[jj])
    err *= np.where(ii == jj, 1, 1 / np.sqrt(2))
    B = b[:, ii, jj] / err
    Y = cov[ii, jj] / err
    coef = solve_nonnegative(B @ B.T, B @ Y)
    residual = coef @ B - Y
    return coef, float(residual @ residual), b


def estimate_solvent_motion(files):
    """Fit calibration movies only; report the effective per-axis displacement rate."""
    items = [dict(np.load(p, allow_pickle=False)) for p in files]
    if not items:
        raise ValueError("at least one calibration movie is required")
    edges = items[0]["dose_edges"]
    k = items[0]["k"]
    for d in items:
        np.testing.assert_allclose(d["dose_edges"], edges)
        np.testing.assert_allclose(d["k"], k)
        if str(d["split"]) != "calibration":
            raise ValueError("validation movies must not enter calibration")
    sizes = np.array([int(d["n_particles"]) for d in items])
    cov = np.average([d["cov_outer"] for d in items], axis=0, weights=sizes)
    # Temporal averaging bounds cost for high-fraction-count movies. The
    # input acquisition here uses equal native doses when aggregation is needed.
    group = max(1, (len(edges) - 1) // 40)
    if group > 1:
        np.testing.assert_allclose(np.diff(edges), np.diff(edges)[0])
        n = (len(edges) - 1) // group
        if n * group != len(edges) - 1:
            raise ValueError("frame count must divide into equal groups")
        cov = cov.reshape(len(k), n, group, n, group).mean((2, 4))
        edges = edges[::group]
    select = np.flatnonzero((k > 0.21) & (k < 0.34))

    def objective(log_s2):
        return sum(
            fit_one(cov[b], edges, k[b], np.exp(log_s2))[1] * items[0]["counts"][b]
            for b in select
        )

    bounds = (np.log(0.005), np.log(5.0))
    opt = minimize_scalar(objective, bounds=bounds, method="bounded")
    s2 = float(np.exp(opt.x))
    return dict(
        motion_variance=s2,
        units="A^2 per (electron/A^2), one coordinate",
        n_movies=len(items),
        n_particles=int(sizes.sum()),
        covariance_files=[str(p) for p in files],
        objective=float(opt.fun),
        scope="effective solvent motion; molecular rearrangement and residual alignment are not separately identified",
        spectral_amplitudes_exported=False,
    )
