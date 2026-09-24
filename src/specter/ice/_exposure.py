"""Effective solvent exposure: how much ice structure survives a summed movie.

Each frame of an exposure sees proper water, but a different arrangement of
it: the beam kicks the molecules and the network re-forms. Ice structure at
spatial frequency k therefore loses coherence with dose, and a summed image
keeps only the fraction a^T C(k) a of its power, C being the correlation
between the ice's Fourier amplitudes at different doses and a the frame
weights. That fraction is applied here as an amplitude filter on one ice
realisation, which reproduces the summed image's second-order statistics
without simulating the frames.

This is a second-moment approximation, exact for projected linear scattering.
It is not an atomistic dynamics integrator or an exact multislice exposure.
The underlying solvent structure factor still comes from the generated ice.

Two coherence models are available (:func:`solvent_coherence`).

``"gaussian"`` is McMullan et al. (2015): each molecule takes an independent
Gaussian step of variance sigma0^2 per axis per e-/A^2, so the correlation is
exp(-2 pi^2 k^2 sigma0^2 |D - D'|).

``"relaxed"``, the default, is measured on explicit trajectories: periodic
256 A boxes of the bundled ice library evolved by Gaussian kicks of 0.272 A^2
per axis per step, each followed by the library's own S(k) plus ML-BOP
relaxation so that every state is water. The coherence was measured without a
grid, from the structure factor of all 527,178 molecules summed at
reciprocal-lattice vectors of the box, at lags of 1-60 steps over 181 steps
of four chains. It is not a single exponential: the decay is slow over the
first step or two and faster afterwards, and every shell from 90 A to 1.4 A
is fitted within 0.012 in rho by a compressed exponential,
exp(-(x / x0(k))^beta(k)) in accumulated kick variance x, with beta ~1.1.
A single-exponential fit to the first step overstates how coherent the ice
stays over a long exposure by 17-37 %.

sigma0^2 is defined by a measurement at the 3.7 A water ring, the decay of
its amplitude over many frames read through the Gaussian model. The relaxed
curves therefore supply only the shape, in k and in dose, and their time
scale is fixed so that the ring's correlation has the same area, int rho dD,
as the Gaussian model's for the same sigma0^2; that area is what a
multi-frame fit determines. The shape departs from the Gaussian in the
direction liquid dynamics predicts. Relaxed ice resists compression, so its
long-wavelength density fluctuations decorrelate far faster than
independent kicks would: the correlation area is 1/33 of the Gaussian's at
20 A, 1/12 at 10 A and 1/3 at 5 A. Near the peaks of its structure factor it
decorrelates more slowly (de Gennes narrowing, 0.83x the Gaussian rate at
2.1 A), and at short wavelengths the two approach each other.

Validation, against the explicit trajectories imaged frame by frame (620 A
ice, 300 kV): with the static ice power cancelled, the growth of summed ice
power from single frames to blocks of 8 and 120 frames is predicted within
0-16 % from 2.5 to 20 A, the residual consistent with the trajectories'
first ~50 steps, taken while the ice was still settling from library
structure. The trajectories' frame-to-frame decorrelation is the same at
0.25 A and 1.04 A pixels. The Gaussian model overstates long-exposure ice
power 2.4-30x from 5 to 20 A, and gives a summed spectrum of pure ice a
bright low-frequency disc that McMullan et al.'s Fig. 1(a) does not show.
The relaxation step is a structure-matching optimisation, not molecular
dynamics, so the long-wavelength time scales are the least certain part;
scaling them with sigma0^2 away from the measured kick is untested.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from ..fft import apply_radial_envelope_

CoherenceModel = Literal["relaxed", "gaussian"]

#: Where sigma0^2 is measured: McMullan et al.'s 1/3.7 A water ring.
_RING_K = 1.0 / 3.7

#: Compressed-exponential fits to the relaxed trajectories' coherence,
#: rho(x) = exp(-(x / x0)^beta), with x the kick variance accumulated between
#: the two states in A^2 per axis. One entry per reciprocal-lattice shell with
#: at least 40 vectors and a one-step coherence above 0.15, 1/90.5 to
#: 1/1.39 A^-1; fitted where rho > 0.03. Outside that range x0 scales as
#: k^-2 at fixed beta: diffusive relaxation at long wavelengths, and at short
#: ones the self-motion limit, where one sub-step already decorrelates.
_RELAXED_K: tuple[float, ...] = (
    0.01105,
    0.01408,
    0.02104,
    0.02501,
    0.0262,
    0.03221,
    0.03428,
    0.04078,
    0.04243,
    0.04863,
    0.05078,
    0.05497,
    0.05859,
    0.06089,
    0.06466,
    0.06922,
    0.07128,
    0.0736,
    0.07504,
    0.07871,
    0.081,
    0.08387,
    0.08717,
    0.09052,
    0.0952,
    0.09905,
    0.10141,
    0.10409,
    0.11069,
    0.1122,
    0.11777,
    0.12408,
    0.12825,
    0.13131,
    0.1386,
    0.14024,
    0.1489,
    0.15174,
    0.15862,
    0.16374,
    0.17031,
    0.17469,
    0.18113,
    0.18754,
    0.19644,
    0.20466,
    0.21144,
    0.22358,
    0.22981,
    0.23457,
    0.24581,
    0.2533,
    0.2634,
    0.27416,
    0.28406,
    0.29905,
    0.30479,
    0.31643,
    0.3315,
    0.34609,
    0.35532,
    0.37025,
    0.38454,
    0.3955,
    0.40555,
    0.43289,
    0.4421,
    0.46036,
    0.47787,
    0.49073,
    0.51735,
    0.52756,
    0.54818,
    0.56866,
    0.59379,
    0.61258,
    0.63423,
    0.66136,
    0.71986,
)
_RELAXED_X0: tuple[float, ...] = (
    26.6858,
    11.0004,
    4.5916,
    3.0798,
    2.3467,
    1.9251,
    1.4722,
    1.2186,
    1.0139,
    0.9062,
    0.8039,
    0.7381,
    0.6692,
    0.661,
    0.6726,
    0.6578,
    0.6515,
    0.6373,
    0.597,
    0.6058,
    0.5798,
    0.5802,
    0.5427,
    0.5669,
    0.5465,
    0.5582,
    0.5508,
    0.5273,
    0.5397,
    0.5168,
    0.5079,
    0.5133,
    0.5018,
    0.4759,
    0.4907,
    0.4788,
    0.4715,
    0.4703,
    0.4755,
    0.4783,
    0.4721,
    0.4916,
    0.4964,
    0.5247,
    0.5408,
    0.5618,
    0.6193,
    0.6566,
    0.7392,
    0.7934,
    0.8821,
    0.9525,
    0.9662,
    0.9065,
    0.8294,
    0.7503,
    0.6671,
    0.5878,
    0.5309,
    0.5085,
    0.4619,
    0.4434,
    0.4209,
    0.4111,
    0.4065,
    0.4117,
    0.3963,
    0.388,
    0.3529,
    0.32,
    0.2782,
    0.2423,
    0.2035,
    0.183,
    0.1609,
    0.1562,
    0.1529,
    0.1402,
    0.1295,
)
_RELAXED_BETA: tuple[float, ...] = (
    1.199,
    1.157,
    1.235,
    1.177,
    1.202,
    1.1,
    1.194,
    1.186,
    1.171,
    1.108,
    1.154,
    1.147,
    1.136,
    1.159,
    1.134,
    1.124,
    1.155,
    1.114,
    1.108,
    1.073,
    1.138,
    1.121,
    1.114,
    1.07,
    1.143,
    1.086,
    1.126,
    1.14,
    1.107,
    1.081,
    1.104,
    1.079,
    1.115,
    1.098,
    1.053,
    1.08,
    1.076,
    1.054,
    1.069,
    1.075,
    1.046,
    1.068,
    1.08,
    1.075,
    1.089,
    1.1,
    1.057,
    1.089,
    1.087,
    1.072,
    1.065,
    1.062,
    1.041,
    1.109,
    1.104,
    1.069,
    1.121,
    1.118,
    1.132,
    1.114,
    1.116,
    1.095,
    1.111,
    1.105,
    1.135,
    1.101,
    1.084,
    1.068,
    1.068,
    1.023,
    1.0,
    0.971,
    0.932,
    0.914,
    0.864,
    0.923,
    0.961,
    0.9,
    0.845,
)

#: Grid for the cumulative integrals of rho, in units of its own time scale:
#: geometric from 1e-7 (rho = 1 to working precision below) to 40, past which
#: exp(-t^beta) is below 1e-17 for any tabulated beta.
_T_GRID_POINTS = 4096
_T_MAX = 40.0


def _relaxed_shape(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fitted time scale x0 (A^2 of kick variance) and shape beta at |k|."""
    kt = torch.tensor(_RELAXED_K, dtype=k.dtype, device=k.device)
    x0t = torch.tensor(_RELAXED_X0, dtype=k.dtype, device=k.device)
    bt = torch.tensor(_RELAXED_BETA, dtype=k.dtype, device=k.device)
    kc = k.clamp(min=kt[0], max=kt[-1])
    i = torch.searchsorted(kt, kc.contiguous(), right=True).clamp(1, len(kt) - 1)
    frac = (kc - kt[i - 1]) / (kt[i] - kt[i - 1])
    # the time scale interpolated in log, since it spans three decades
    x0 = torch.exp(torch.lerp(x0t[i - 1].log(), x0t[i].log(), frac))
    beta = torch.lerp(bt[i - 1], bt[i], frac)
    safe_k = k.clamp(min=torch.finfo(k.dtype).tiny)
    x0 = torch.where(k < kt[0], x0t[0] * (kt[0] / safe_k).square(), x0)
    beta = torch.where(k < kt[0], bt[0], beta)
    x0 = torch.where(k > kt[-1], x0t[-1] * (kt[-1] / safe_k).square(), x0)
    beta = torch.where(k > kt[-1], bt[-1], beta)
    return x0, beta


def _relaxed_time_scale(
    k: torch.Tensor, displacement_variance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The relaxed correlation's dose scale tau_c (e-/A^2) and shape beta.

    rho(D) = exp(-(D / tau_c)^beta). The kick variance per e-/A^2 that makes
    the ring's correlation area x0_r Gamma(1 + 1/beta_r) / (c sigma0^2) equal
    the Gaussian model's 1 / (2 pi^2 k_r^2 sigma0^2) is c sigma0^2, with
    c = 2 pi^2 k_r^2 x0_r Gamma(1 + 1/beta_r).
    """
    x0, beta = _relaxed_shape(k)
    ring = torch.tensor([_RING_K], dtype=torch.float64)
    x0_r, beta_r = _relaxed_shape(ring)
    c = 2 * math.pi**2 * _RING_K**2 * float(x0_r) * math.gamma(1 + 1 / float(beta_r))
    if displacement_variance == 0:
        return torch.full_like(k, math.inf), beta
    return x0 / (c * displacement_variance), beta


def solvent_coherence(
    k: torch.Tensor,
    dose_lag: float | torch.Tensor,
    displacement_variance: float,
    model: CoherenceModel = "relaxed",
) -> torch.Tensor:
    """
    Correlation of the ice's Fourier amplitudes a given dose apart.

    Parameters
    ----------
    k : torch.Tensor
        Spatial frequency magnitude in 1/A.
    dose_lag : float or torch.Tensor
        Dose between the two states, e-/A^2; broadcast against `k`.
    displacement_variance : float
        McMullan's sigma0^2 in A^2 per e-/A^2, as measured from the 3.7 A
        ring's decorrelation. Both models give the same correlation area at
        that ring.
    model : {"relaxed", "gaussian"}, optional
        See the module docstring. Default "relaxed".

    Returns
    -------
    torch.Tensor
        rho in [0, 1].
    """
    lag = torch.as_tensor(dose_lag, dtype=k.dtype, device=k.device).abs()
    if model == "gaussian":
        return torch.exp(-2 * math.pi**2 * k.square() * displacement_variance * lag)
    if model != "relaxed":
        raise ValueError(f"unknown coherence model {model!r}")
    tau_c, beta = _relaxed_time_scale(k, displacement_variance)
    return torch.exp(-((lag / tau_c) ** beta))


def _cumulative_moments(
    tau_c: torch.Tensor, beta: torch.Tensor, doses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    I0(D) = int_0^D rho and I1(D) = int_0^D tau rho for rho = exp(-(tau/tau_c)^beta).

    `tau_c` and `beta` have shape (K,), `doses` (Q,); returns (K, Q) each.
    Computed on a grid in t = tau / tau_c, so it is equally accurate whether
    the ice decorrelates far faster or far slower than the doses asked for.
    """
    dt = tau_c.dtype
    t = torch.cat(
        [
            torch.zeros(1, dtype=dt, device=tau_c.device),
            torch.logspace(
                -7, math.log10(_T_MAX), _T_GRID_POINTS, dtype=dt, device=tau_c.device
            ),
        ]
    )
    f = torch.exp(-(t[None, :] ** beta[:, None]))  # (K, G)
    h = t.diff()[None, :]
    j0 = torch.cat(
        [torch.zeros_like(f[:, :1]), torch.cumsum(0.5 * h * (f[:, 1:] + f[:, :-1]), 1)],
        1,
    )
    tf = t[None, :] * f
    j1 = torch.cat(
        [
            torch.zeros_like(f[:, :1]),
            torch.cumsum(0.5 * h * (tf[:, 1:] + tf[:, :-1]), 1),
        ],
        1,
    )
    finite = torch.isfinite(tau_c)
    tc = torch.where(finite, tau_c, torch.ones_like(tau_c))
    tq = (doses[None, :] / tc[:, None]).clamp(max=_T_MAX)
    idx = torch.searchsorted(t, tq.contiguous(), right=True).clamp(1, len(t) - 1)
    t_lo, t_hi = t[idx - 1], t[idx]
    w = (tq - t_lo) / (t_hi - t_lo)
    J0 = torch.lerp(j0.gather(1, idx - 1), j0.gather(1, idx), w)
    J1 = torch.lerp(j1.gather(1, idx - 1), j1.gather(1, idx), w)
    I0 = tc[:, None] * J0
    I1 = tc[:, None] ** 2 * J1
    # frozen ice: rho = 1 at every lag
    frozen = ~finite[:, None]
    I0 = torch.where(frozen, doses[None, :].expand_as(I0), I0)
    I1 = torch.where(frozen, 0.5 * doses[None, :].square().expand_as(I1), I1)
    return I0, I1


def _relaxed_exposure_power(
    k: torch.Tensor,
    dose: float,
    displacement_variance: float,
    a: torch.Tensor | None,
    n_frames: int,
) -> torch.Tensor:
    """a^T C a for the relaxed model, frames integrated continuously."""
    shape = k.shape
    kf = k.reshape(-1).to(torch.float64)
    out = torch.empty_like(kf)
    d = dose / n_frames
    chunk = 4096
    for s in range(0, kf.numel(), chunk):
        kk = kf[s : s + chunk]
        tau_c, beta = _relaxed_time_scale(kk, displacement_variance)
        if a is None:
            # equal weights: the continuous average over the whole exposure,
            # (2 / D^2) int_0^D (D - tau) rho(tau) dtau, whatever n is
            D = torch.tensor([dose], dtype=kk.dtype, device=kk.device)
            I0, I1 = _cumulative_moments(tau_c, beta, D)
            out[s : s + chunk] = (2 / dose**2) * (dose * I0[:, 0] - I1[:, 0])
            continue
        doses = torch.arange(n_frames + 1, dtype=kk.dtype, device=kk.device) * d
        I0, I1 = _cumulative_moments(tau_c, beta, doses)  # (K, n+1)
        # c_0: a frame with itself; c_L: two frames L apart, a triangle-weighted
        # average of rho over lags (L-1)d .. (L+1)d
        c = torch.empty(kk.numel(), n_frames, dtype=kk.dtype, device=kk.device)
        c[:, 0] = (2 / d**2) * (d * I0[:, 1] - I1[:, 1])
        if n_frames > 1:
            L = torch.arange(1, n_frames, dtype=kk.dtype, device=kk.device)
            i0m, i0c, i0p = I0[:, :-2], I0[:, 1:-1], I0[:, 2:]
            i1m, i1c, i1p = I1[:, :-2], I1[:, 1:-1], I1[:, 2:]
            c[:, 1:] = (
                (i1c - i1m)
                - (L - 1) * d * (i0c - i0m)
                + (L + 1) * d * (i0p - i0c)
                - (i1p - i1c)
            ) / d**2
        ak = a.reshape(n_frames, -1)[:, s : s + chunk].to(kk.dtype)  # (n, K)
        total = c[:, 0] * ak.square().sum(0)
        for lag in range(1, n_frames):
            total = total + 2 * c[:, lag] * (ak[:-lag] * ak[lag:]).sum(0)
        out[s : s + chunk] = total
    return out.reshape(shape).to(k.dtype).clamp(0, 1)


def solvent_decorrelation_rate(
    k: torch.Tensor,
    displacement_variance: float,
    model: CoherenceModel = "relaxed",
) -> torch.Tensor:
    """
    The rate of an exponential with the same correlation area, per e-/A^2.

    1 / int_0^inf rho(D) dD. For the Gaussian model this is its exact rate,
    2 pi^2 k^2 sigma0^2; for the relaxed model a summary of a curve that is
    not exponential, equal to the Gaussian rate at the 3.7 A ring by
    construction. It fixes the long-exposure limit, where the surviving
    fraction tends to 2 / (Gamma D).
    """
    if model == "gaussian":
        return 2 * math.pi**2 * k.square() * displacement_variance
    if model != "relaxed":
        raise ValueError(f"unknown coherence model {model!r}")
    tau_c, beta = _relaxed_time_scale(k, displacement_variance)
    area = tau_c * torch.exp(torch.lgamma(1 + 1 / beta))
    return 1 / area


def solvent_exposure_power(
    k: torch.Tensor,
    dose: float,
    displacement_variance: float,
    n_frames: int,
    weights: torch.Tensor | None = None,
    weights_max_frequency: float | None = None,
    model: CoherenceModel = "relaxed",
) -> torch.Tensor:
    """Return a^T C a for equal-dose fractions, including within-frame motion.

    ``displacement_variance`` is per-axis Å² per (electron/Å²). The
    correlation between instantaneous Fourier amplitudes is
    :func:`solvent_coherence` under `model`; each frame averages it over its
    own dose. Normalized amplitude weights a sum to one at each frequency.
    For the Gaussian model an O(N) recursion evaluates all pairs without a
    dense covariance; for the relaxed model the pair terms are
    triangle-weighted averages of rho, from its cumulative integrals, and
    with equal weights the result is the continuous average over the whole
    exposure, independent of the frame count.
    """
    if dose < 0 or displacement_variance < 0 or n_frames < 1:
        raise ValueError("dose/motion must be nonnegative and n_frames positive")
    if weights is not None:
        if weights.shape[0] != n_frames or weights_max_frequency is None:
            raise ValueError("weights require matching frames and a frequency axis")
        if weights_max_frequency <= 0 or torch.any(weights < 0):
            raise ValueError("invalid weight axis or negative weights")
        idx = (k / weights_max_frequency * (weights.shape[1] - 1)).round().long()
        idx = idx.clamp(0, weights.shape[1] - 1)
        a = weights.to(k)[:, idx]
        total = a.sum(0)
        if torch.any(total <= 0):
            raise ValueError("frame weights must have positive sum")
        a = a / total
    else:
        a = None
    if model == "relaxed":
        if dose == 0 or displacement_variance == 0:
            return torch.ones_like(k)
        return _relaxed_exposure_power(k, dose, displacement_variance, a, n_frames)
    if model != "gaussian":
        raise ValueError(f"unknown coherence model {model!r}")
    if a is None:
        a = torch.ones((n_frames, *k.shape), device=k.device, dtype=k.dtype) / n_frames
    z = (
        solvent_decorrelation_rate(k, displacement_variance, "gaussian")
        * dose
        / n_frames
    )
    safe = z.clamp_min(1e-5)
    g = torch.where(z < 1e-3, 1 - z / 2 + z.square() / 6, -torch.expm1(-safe) / safe)
    diagonal = torch.where(
        z < 1e-2,
        1 - z / 3 + z.square() / 12 - z.pow(3) / 60,
        2 * (safe + torch.expm1(-safe)) / safe.square(),
    )
    history = torch.zeros_like(k)
    pairs = torch.zeros_like(k)
    decay = torch.exp(-z)
    for j in range(1, n_frames):
        history = a[j - 1] + decay * history
        pairs += a[j] * history
    return (diagonal * a.square().sum(0) + 2 * g.square() * pairs).clamp(0, 1)


def apply_solvent_exposure(
    ice: torch.Tensor,
    pixel_size: float,
    dose: float,
    displacement_variance: float,
    n_frames: int,
    weights: torch.Tensor | None,
    weights_max_frequency: float | None,
    model: CoherenceModel = "relaxed",
) -> None:
    """Filter generated ice in place; DC/mean potential is preserved.

    A radial 3D extension supplies a rotationally invariant effective volume.
    Its projected power has the specified temporal average. Applying this
    before multislice is an approximation.
    """

    def envelope(k: torch.Tensor) -> torch.Tensor:
        return solvent_exposure_power(
            k,
            dose,
            displacement_variance,
            n_frames,
            weights,
            weights_max_frequency,
            model,
        ).sqrt()

    for volume in ice:
        apply_radial_envelope_(volume, pixel_size, envelope)
