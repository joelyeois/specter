"""Image-domain comparisons between a simulated and an experimental particle stack.

Every function takes stacks that share particle index -- simulated particle
``i`` was rendered at the pose and CTF of experimental particle ``i`` -- and
returns plain floats or small tensors, so `specter.pipelines.run_match` can
tabulate them and the report can plot them. Nothing here touches CryoSPARC.

The comparisons fall into two groups. The first are *screens* for failure
modes that rotationally averaged statistics miss: poses refined against a reference that is not the model
(`matched_index_correlation`), a fixed pattern shared by every simulated
particle (`edge_band_means`), and a neighbour or background mismatch
(`annulus_std_profile`). The second are the two *moments* the match is
judged on, both of which need the pose correspondence: the matched-pose
signal-to-noise ratio per frequency band (`matched_pose_snr`), which says
how much cleaner the simulation is than the experiment at each scale, and
the twin test (`twin_test`), which asks whether an experimental particle is
any further from its simulated twin than a second simulated realisation is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

from specter.aberrations import b_envelope
from specter.filters import butter

#: Frequency bands in 1/Å, as (low, high): coarser than 33 Å, 33-12, 12-6.7,
#: 6.7-4, 4-3 Å. The last two are empty at pixel sizes above 1.5 Å.
BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 1 / 33),
    (1 / 33, 1 / 12),
    (1 / 12, 1 / 6.7),
    (1 / 6.7, 1 / 4),
    (1 / 4, 1 / 3),
)

BAND_LABELS: tuple[str, ...] = (">33 Å", "33-12 Å", "12-6.7 Å", "6.7-4 Å", "4-3 Å")


def _zscore(a: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    return a / a.std(dim=(-2, -1), keepdim=True).clamp(min=1e-12)


def _kgrid(n: int, pixel_size: float) -> torch.Tensor:
    k = torch.fft.fftfreq(n, d=pixel_size)
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    return torch.sqrt(kx**2 + ky**2)


def _radial(power: torch.Tensor, k: torch.Tensor, pixel_size: float) -> torch.Tensor:
    n = power.shape[-1]
    nb = n // 2
    kb = (k / (0.5 / pixel_size) * nb).long().clamp(max=nb - 1)
    num = torch.bincount(kb.flatten(), weights=power.flatten(), minlength=nb)
    den = torch.bincount(kb.flatten(), minlength=nb).clamp(min=1)
    return num / den


def _kbins(n: int, pixel_size: float) -> torch.Tensor:
    return (torch.arange(n // 2) + 0.5) / (n // 2) * (0.5 / pixel_size)


def _band_mean(
    profile: torch.Tensor, kk: torch.Tensor, band: tuple[float, float]
) -> float:
    sel = (kk >= band[0]) & (kk < band[1])
    if not bool(sel.any()):
        return float("nan")
    return float(profile[sel].mean())


def _xcorr_max(a: torch.Tensor, b: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Peak of the circular cross-correlation of each pair within ``max_shift``."""
    fa, fb = torch.fft.fft2(a), torch.fft.fft2(b)
    cc = torch.fft.ifft2(fa * fb.conj()).real / a[0].numel()
    cc = torch.fft.fftshift(cc, dim=(-2, -1))
    c = a.shape[-1] // 2
    return (
        cc[..., c - max_shift : c + max_shift + 1, c - max_shift : c + max_shift + 1]
        .flatten(1)
        .max(1)
        .values
    )


def _bandpass(
    a: torch.Tensor, pixel_size: float, low_angstrom: float, high_angstrom: float
) -> torch.Tensor:
    k = _kgrid(a.shape[-1], pixel_size)
    mask = ((k >= 1 / low_angstrom) & (k < 1 / high_angstrom)).to(a.dtype)
    out = torch.fft.ifft2(torch.fft.fft2(a) * mask).real
    return _zscore(out)


@dataclass
class PoseAlignmentResult:
    """
    Outcome of `matched_index_correlation`.

    Attributes
    ----------
    matched : float
        Mean peak correlation of the matched pairs.
    shuffled : float
        Mean of the same statistic over random pairings, the null.
    fraction_above : float
        Of the comparisons between a matched pair and a mismatched one sharing
        its simulated image, the fraction the matched pair wins; 0.5 under the
        null.
    z_score : float
        ``(matched - shuffled)`` in units of the null's own spread: the
        standard deviation of the mean over random pairings.
    """

    matched: float
    shuffled: float
    fraction_above: float
    z_score: float

    @property
    def passed(self) -> bool:
        """Matched pairs beat shuffled pairs decisively (z >= 5, >60% of pairs)."""
        return self.z_score >= 5.0 and self.fraction_above > 0.6


def _pair_correlations(
    s: torch.Tensor, e: torch.Tensor, max_shift: int, chunk: int = 8
) -> torch.Tensor:
    """``C[i, j]``: peak cross-correlation of ``s[i]`` with ``e[j]``, ``(n, n)``."""
    n = s.shape[0]
    fs, fe = torch.fft.fft2(s), torch.fft.fft2(e)
    c = s.shape[-1] // 2
    window = slice(c - max_shift, c + max_shift + 1)
    out = torch.empty(n, n)
    for start in range(0, n, chunk):
        cc = torch.fft.ifft2(fs[start : start + chunk, None] * fe[None].conj()).real
        cc = torch.fft.fftshift(cc / s[0].numel(), dim=(-2, -1))
        out[start : start + chunk] = cc[..., window, window].flatten(2).max(2).values
    return out


def matched_index_correlation(
    sim: torch.Tensor,
    exp: torch.Tensor,
    pixel_size: float,
    band_angstrom: tuple[float, float] = (60.0, 15.0),
    max_shift: int = 12,
    seed: int = 0,
    n_shuffles: int = 1000,
) -> PoseAlignmentResult:
    """
    Test whether simulated particle ``i`` shows the same view as experimental
    particle ``i``.

    The peak cross-correlation of each matched pair, band-passed to
    ``band_angstrom`` and allowed a small shift, is compared with the same
    statistic over random pairings. Poses aligned to the model put the matched
    value well above the shuffled one; poses from a refinement that was never
    aligned to the model (no Align 3D step against it) put the two at the
    same level. No rotationally averaged statistic can see this: on
    EMPIAR-11377, sixty classifications missed that every simulated view
    was rotated 142 degrees from its experimental one.

    The null is a permutation test. Every simulated image is correlated with
    every experimental one, and the matched mean is placed in the
    distribution of means over `n_shuffles` random pairings, whose spread is
    the standard error the z-score divides by. A single shuffled pairing is
    one draw from that distribution: dividing by the matched pairs' standard
    error alone took the draw for the null's exact mean, which at 64 pairs
    passed unaligned poses at some shuffle seeds and failed them at others.

    Parameters
    ----------
    sim, exp : torch.Tensor
        Stacks of shape (N, n, n) sharing particle index.
    pixel_size : float
        Pixel size in Å.
    band_angstrom : (float, float), optional
        Band-pass in Å, (low-resolution edge, high-resolution edge). The
        default 60-15 Å carries view-dependent structure for any particle
        and little noise.
    max_shift : int, optional
        Shift search radius in pixels.
    seed : int, optional
        Seed of the random pairings.
    n_shuffles : int, optional
        Random pairings the null is estimated from. Default 1000.

    Returns
    -------
    PoseAlignmentResult
    """
    n = sim.shape[0]
    s = _bandpass(sim.float().cpu(), pixel_size, *band_angstrom)
    e = _bandpass(exp.float().cpu(), pixel_size, *band_angstrom)
    pairs = _pair_correlations(s, e, max_shift)
    matched = pairs.diagonal()
    generator = torch.Generator().manual_seed(seed)
    shuffles = torch.stack(
        [torch.randperm(n, generator=generator) for _ in range(n_shuffles)]
    )
    null = pairs[torch.arange(n), shuffles].mean(dim=1)
    spread = float(null.std())
    others = ~torch.eye(n, dtype=torch.bool)
    wins = (matched[:, None] > pairs)[others].float().mean() if n > 1 else 0.0
    return PoseAlignmentResult(
        matched=float(matched.mean()),
        shuffled=float(null.mean()),
        fraction_above=float(wins),
        z_score=(
            float((matched.mean() - null.mean()) / spread)
            if spread > 0
            else float("inf")
        ),
    )


def edge_band_means(
    stack: torch.Tensor, bands_px: tuple[int, ...] = (2, 4, 8, 16)
) -> list[float]:
    """
    Mean of the stack-averaged, z-scored image in concentric bands from the
    box edge inward: 0-2, 2-4, 4-8, 8-16 px.

    A deterministic pattern shared by every simulated particle -- the
    Fresnel ring an ice slab truncated at the field of view produces --
    shows up here as a mean well away from zero, while every
    variance-based statistic reports nothing. The experiment sits at a few
    thousandths.
    """
    n = stack.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    edge = torch.as_tensor(
        np.minimum(np.minimum(yy, xx), np.minimum(n - 1 - yy, n - 1 - xx))
    )
    mean_img = _zscore(stack.float()).mean(0)
    out, lo = [], 0
    for hi in bands_px:
        out.append(float(mean_img[(edge >= lo) & (edge < hi)].mean()))
        lo = hi
    return out


def band_power_ratio(
    sim: torch.Tensor, exp: torch.Tensor, pixel_size: float
) -> list[float]:
    """
    Ratio of simulated to experimental radial power in each of `BANDS`,
    both stacks z-scored per image, so the ratio compares the *fraction* of
    each image's variance at each scale.

    Descriptive only. This was the one statistic that did not predict
    mixing: a stack with half the experiment's low-band fraction mixed well
    and one with 1.8x mixed worse. It stays in the report because it is
    the view users know how to read, next to the ratio that makes a 2x
    mismatch visible.
    """
    n = sim.shape[-1]
    k = _kgrid(n, pixel_size)
    kk = _kbins(n, pixel_size)
    ps = _radial(
        (torch.fft.fft2(_zscore(sim.float())).abs() ** 2).mean(0), k, pixel_size
    )
    pe = _radial(
        (torch.fft.fft2(_zscore(exp.float())).abs() ** 2).mean(0), k, pixel_size
    )
    out = []
    for band in BANDS:
        sel = (kk >= band[0]) & (kk < band[1])
        out.append(
            float(ps[sel].sum() / pe[sel].sum()) if bool(sel.any()) else float("nan")
        )
    return out


def radial_power_spectra(
    sim: torch.Tensor, exp: torch.Tensor, pixel_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frequency axis (1/Å) and the z-scored radial power of each stack, for plotting."""
    n = sim.shape[-1]
    k = _kgrid(n, pixel_size)
    ps = _radial(
        (torch.fft.fft2(_zscore(sim.float())).abs() ** 2).mean(0), k, pixel_size
    )
    pe = _radial(
        (torch.fft.fft2(_zscore(exp.float())).abs() ** 2).mean(0), k, pixel_size
    )
    return _kbins(n, pixel_size), ps, pe


def water_ring_excess(stack: torch.Tensor, pixel_size: float) -> float:
    """
    Log-power excess of the 3.7 Å water ring over a linear baseline drawn
    between 0.22-0.24 and 0.30-0.32 1/Å. NaN when the pixel size does not
    reach 3.1 Å.
    """
    n = stack.shape[-1]
    if 0.5 / pixel_size < 0.32:
        return float("nan")
    k = _kgrid(n, pixel_size)
    kk = _kbins(n, pixel_size)
    lp = torch.log(
        _radial(
            (torch.fft.fft2(_zscore(stack.float())).abs() ** 2).mean(0), k, pixel_size
        ).clamp(min=1e-12)
    )
    lo = _band_mean(lp, kk, (0.22, 0.24))
    hi = _band_mean(lp, kk, (0.30, 0.32))
    base = lo + (hi - lo) * ((0.27 - 0.23) / (0.31 - 0.23))
    return _band_mean(lp, kk, (0.26, 0.28)) - base


def annulus_std_per_particle(
    stack: torch.Tensor, radii_px: tuple[int, ...] = (20, 40, 60, 80, 100, 128)
) -> torch.Tensor:
    """
    Per-particle standard deviation of the low-passed, z-scored image inside
    concentric annuli (0-20, 20-40, ... px) plus the corners beyond the last
    radius.

    Returns
    -------
    torch.Tensor
        ``(n_particles, len(radii_px) + 1)``, NaN in an annulus the box does
        not reach.
    """
    n = stack.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    rr = torch.as_tensor(np.hypot(yy - n // 2, xx - n // 2)).float()
    a = _zscore(torch.as_tensor(butter(stack.float())))
    columns, lo = [], 0.0
    for hi in (*radii_px, None):
        sel = (rr >= lo) if hi is None else (rr >= lo) & (rr < hi)
        columns.append(
            a[:, sel].std(dim=1)
            if bool(sel.any())
            else torch.full((a.shape[0],), float("nan"))
        )
        lo = float(hi) if hi is not None else lo
    return torch.stack(columns, dim=1)


def annulus_std_profile(
    stack: torch.Tensor, radii_px: tuple[int, ...] = (20, 40, 60, 80, 100, 128)
) -> list[float]:
    """
    Standard deviation of the low-passed, z-scored image inside concentric
    annuli (0-20, 20-40, ... px) plus the corners beyond the last radius,
    averaged over particles.

    Outside the particle this measures background structure: neighbours,
    ice, thickness gradients. It is what sets the ice thickness and
    neighbour spacing, and what exposes a close-packed monolayer (CCMV)
    the simulator cannot reproduce.
    """
    return [float(v) for v in annulus_std_per_particle(stack, radii_px).mean(dim=0)]


def _profile_rms(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")


@dataclass(frozen=True)
class ProbeChoice:
    """
    The outcome of choosing one setting from a probe's candidates.

    Attributes
    ----------
    value : float
        The chosen candidate.
    best : float
        The candidate with the lowest score.
    distinguished : bool
        Whether the data separate the best candidate from the fallback (the
        first candidate): False when the fallback was kept because it is
        not significantly worse than the best.
    scores : list of float
        Each candidate's score, in candidate order.
    errors : list of float
        Each score's bootstrap standard error.
    p_fallback_not_worse : float
        Fraction of bootstrap replicates in which the fallback scored at
        least as well as the best candidate.
    """

    value: float
    best: float
    distinguished: bool
    scores: list[float]
    errors: list[float]
    p_fallback_not_worse: float


def choose_probe(
    candidates: list[float],
    sim_stacks: list[torch.Tensor],
    exp: torch.Tensor,
    first_bin: int,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> ProbeChoice:
    """
    Choose a probe setting, keeping the first candidate unless the data
    reject it.

    Each candidate is scored by the RMS difference between its low-passed
    annulus-std profile outside the particle and the experiment's. The
    score's noise is estimated by a paired bootstrap over particles (one
    index draw per replicate, shared by every candidate, since candidate
    stacks are the same particles at the same poses, and an independent
    draw over the experimental particles). The best-scoring candidate
    replaces the first one only if it beats it in at least ``1 - alpha`` of
    the replicates.

    The rule exists because the scores are noisy at the default probe size:
    on EMPIAR-11377 one setting scored 0.083 and 0.097 on two runs with the
    same seed and device, while the candidates differed by at most 0.03, so
    a plain minimum chose a different ice thickness on every run.

    Parameters
    ----------
    candidates : list of float
        The settings, fallback first.
    sim_stacks : list of torch.Tensor
        One ``(n, Y, X)`` probe stack per candidate, the same particles in
        the same order.
    exp : torch.Tensor
        The experimental probe stack.
    first_bin : int
        First annulus outside the particle.
    n_boot : int, optional
        Bootstrap replicates. Default 400.
    alpha : float, optional
        Required confidence that the best beats the fallback. Default 0.05.
    seed : int, optional
        Seed of the bootstrap draws. Default 0.

    Returns
    -------
    ProbeChoice
    """
    exp_f = annulus_std_per_particle(exp)[:, first_bin:].double().numpy()
    sims = [
        annulus_std_per_particle(s)[:, first_bin:].double().numpy() for s in sim_stacks
    ]
    exp_profile = exp_f.mean(axis=0)
    scores = [_profile_rms(s.mean(axis=0), exp_profile) for s in sims]
    rng = np.random.default_rng(seed)
    n_sim, n_exp = sims[0].shape[0], exp_f.shape[0]
    boot = np.empty((n_boot, len(sims)))
    for r in range(n_boot):
        i = rng.integers(0, n_sim, n_sim)
        e = exp_f[rng.integers(0, n_exp, n_exp)].mean(axis=0)
        for c, s in enumerate(sims):
            boot[r, c] = _profile_rms(s[i].mean(axis=0), e)
    errors = [float(v) for v in boot.std(axis=0, ddof=1)]
    if len(candidates) == 1 or all(np.isnan(scores)):
        # Nothing to choose between, or nothing to score with (no annulus
        # lies outside the particle in this box): the fallback stands, and
        # with more than one candidate it is not a measurement.
        return ProbeChoice(
            value=float(candidates[0]),
            best=float(candidates[0]),
            distinguished=len(candidates) == 1,
            scores=[float(v) for v in scores],
            errors=errors,
            p_fallback_not_worse=1.0,
        )
    best = int(np.nanargmin(scores))
    p = float(np.mean(boot[:, 0] <= boot[:, best])) if best != 0 else 1.0
    keep_fallback = best == 0 or p >= alpha
    return ProbeChoice(
        value=float(candidates[0] if keep_fallback else candidates[best]),
        best=float(candidates[best]),
        distinguished=best == 0 or not keep_fallback,
        scores=[float(v) for v in scores],
        errors=errors,
        p_fallback_not_worse=p,
    )


#: Fit window for the residual envelope, in 1/Angstrom (10-4 A).
_ENVELOPE_BAND = (0.10, 0.25)
#: Coarse bands the window is split into before any division is done.
_ENVELOPE_N_BANDS = 8
#: A band's denominator must exceed this many standard errors to be used.
_ENVELOPE_SIGMA = 3.0
#: Two-sided significance level the fitted B must clear against its own
#: standard error. Compared through Student's t at the fit's realised degrees
#: of freedom, not as a fixed multiple: with eight bands and two parameters
#: there are six, where a flat |t| > 2 is only p = 0.09 and lets one noise
#: realisation in eleven through.
_ENVELOPE_ALPHA = 0.05


def _residual_envelope(
    kk: torch.Tensor, s_es: torch.Tensor, s_ss: torch.Tensor
) -> tuple[float, float]:
    r"""
    Guinier slope of the experiment/simulation amplitude ratio, as a B-factor.

    The ratio is formed from band MEANS rather than averaged over per-bin
    ratios, and the fit is reported only when it is statistically significant.
    Both matter, and the earlier per-bin version had neither.

    `s_ss` is a cross-power between two independent simulation seeds, so it is
    an unbiased estimate of the signal that goes NEGATIVE wherever there is no
    signal left. Dividing by it per bin, with a ``clamp(min=1e-9)`` to avoid a
    zero denominator, turns "no correlation" into an enormous positive ratio:
    on EMPIAR-10254 one bin at 7.64 A came out at 2.7e11, and since the fit is
    on ``log(a)`` that single point carried ~50x the leverage of any other and
    moved the answer from 74 to 275 A^2. Filtering on ``a > 0`` does not help --
    it keeps the upward noise excursions and drops the downward ones, which
    biases the slope whenever `a` is consistent with zero.

    Smoothing the ratio is not a fix either: a Gaussian on ``a(k)`` spreads such
    a spike over its neighbours and made that dataset worse (275 -> 921 A^2). A
    median is robust to it but still left 145-187, because the fit was still
    driven by bands carrying no signal.

    Aggregating first removes the blow-up structurally, since band denominators
    are far from zero. The fit itself is then a direct nonlinear least squares
    of ``A * exp(-B k^2 / 4)`` to the band ratios, NOT a straight line through
    ``log(a)``. Taking a log cannot represent a non-positive ratio, so a log fit
    has to discard those bands -- and discarding them is the same selection bias
    as the per-bin form's ``a > 0``, one level up: upward noise excursions are
    kept and downward ones dropped. Measured over 200 synthetic realisations,
    that bias runs one way and it is not small. A true B = 200 A^2 at high noise
    came back as 176 (12% low) from the log fit against 206 from this one, and
    the log fit reported an envelope for 9.0% of pure-noise inputs against 0.5%
    here. Fitting the exponential directly also makes a strongly damped band a
    datum rather than a deletion, which is what a refusal rule keyed on dropped
    bands got wrong: it refused hardest exactly where the envelope was real and
    strong (detection fell to 28% at B = 200).

    EMPIAR-10254 fits B = 44 +- 97 A^2, |t| = 0.46, and is correctly reported as
    no measurable envelope.

    Parameters
    ----------
    kk : torch.Tensor
        Radial frequency bin centres, 1/Angstrom.
    s_es, s_ss : torch.Tensor
        Experiment/simulation and simulation/simulation radial cross-powers.

    Returns
    -------
    tuple of float
        The B-factor in Angstrom^2 and its standard error, or ``(nan, nan)``
        when too few bands survive or the slope is not significant.
    """
    lo, hi = _ENVELOPE_BAND
    edges = np.linspace(lo, hi, _ENVELOPE_N_BANDS + 1)
    ks, ratios = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (kk >= a) & (kk < b)
        if int(m.sum()) < 2:
            continue
        den = float(s_ss[m].mean())
        err = float(s_ss[m].std()) / float(int(m.sum())) ** 0.5
        # The denominator must be real signal, not a noise excursion. The
        # NUMERATOR is not filtered: a negative ratio is a measurement that
        # the envelope has run out of amplitude, and dropping it would bias
        # the fit upward.
        if not (den > 0.0 and den > _ENVELOPE_SIGMA * err):
            continue
        ks.append(0.5 * (a + b))
        ratios.append(float(s_es[m].mean()) / den)
    n_bands = len(ks)
    if n_bands < 4:
        return float("nan"), float("nan")
    k = np.asarray(ks, dtype=float)
    a_k = np.asarray(ratios, dtype=float)

    def model(kx: np.ndarray, amplitude: float, bfactor: float) -> np.ndarray:
        envelope = b_envelope(
            torch.as_tensor(kx**2, dtype=torch.float64),
            torch.tensor(bfactor, dtype=torch.float64),
        )
        return amplitude * envelope.numpy()

    try:
        params, cov = curve_fit(
            model, k, a_k, p0=[max(float(a_k.max()), 1e-3), 50.0], maxfev=20000
        )
    except (RuntimeError, ValueError, TypeError):
        return float("nan"), float("nan")
    if not np.all(np.isfinite(cov)):
        return float("nan"), float("nan")
    bfactor = float(params[1])
    bfactor_se = float(np.sqrt(cov[1, 1]))
    dof = n_bands - 2
    if not (bfactor_se > 0.0 and dof >= 1):
        return float("nan"), float("nan")
    crit = float(student_t.ppf(1.0 - _ENVELOPE_ALPHA / 2.0, dof))
    if abs(bfactor) <= crit * bfactor_se:
        # Consistent with a flat envelope: report the uncertainty, not a value.
        return float("nan"), bfactor_se
    return bfactor, bfactor_se


#: Bootstrap resamples of the particles behind each band's interval.
_SNR_BOOTSTRAP = 500
#: Two-sided coverage of the reported interval (16th to 84th percentile).
_SNR_INTERVAL = (0.16, 0.84)


@dataclass
class MatchedPoseSNR:
    """Outcome of `matched_pose_snr`."""

    snr_sim: list[float]
    snr_exp: list[float]
    ratio: list[float]
    ratio_lo: list[float]  # 16th percentile of `ratio` over a particle bootstrap
    ratio_hi: list[float]  # 84th percentile; inf when the experiment may hold no signal
    signal_plateau: float  # exp/sim signal amplitude ratio at 33-12 Å (z-scored units)
    residual_bfactor: float  # Å², Guinier slope of the exp/sim signal ratio, 10-4 Å
    residual_bfactor_stderr: float  # Å², 1 sigma on the above; NaN when not fitted


def _band_spectra(
    sim: torch.Tensor, sim2: torch.Tensor, exp: torch.Tensor, pixel_size: float
) -> torch.Tensor:
    """
    Per-particle sums over each of `BANDS` of the four spectra the matched-pose
    SNR is built from: ``Re(S1 S2*)``, ``Re(E S1*)``, ``|E|^2`` and ``|S1|^2``.

    Returns
    -------
    torch.Tensor
        Shape (n_particles, len(BANDS), 4), float64.
    """
    n = sim.shape[-1]
    k = _kgrid(n, pixel_size).flatten()
    member = torch.zeros(k.numel(), len(BANDS), dtype=torch.float32)
    for i, (lo, hi) in enumerate(BANDS):
        member[(k >= lo) & (k < hi), i] = 1.0
    fe, fs, fs2 = (torch.fft.fft2(_zscore(x.float())) for x in (exp, sim, sim2))
    spectra = (
        (fs * fs2.conj()).real,
        (fe * fs.conj()).real,
        fe.abs() ** 2,
        fs.abs() ** 2,
    )
    return torch.stack(
        [s.reshape(s.shape[0], -1) @ member for s in spectra], dim=-1
    ).double()


def _band_snr(terms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SNR of both stacks and their ratio per band, from particle-averaged band sums.

    A band whose simulated cross-power is not positive has no simulated signal
    to compare against and is NaN. A band where the experiment's coherent
    signal does not exceed its own noise has an unbounded ratio and is inf.
    """
    s_ss, s_es, p_e, p_s = terms.mean(0).unbind(-1)
    signal_e = s_es**2 / s_ss
    snr_e = signal_e / (p_e - signal_e)
    snr_s = s_ss / (p_s - s_ss)
    ratio = snr_s / snr_e
    ratio = torch.where(snr_e > 0, ratio, torch.full_like(ratio, float("inf")))
    nan = torch.full_like(ratio, float("nan"))
    ratio = torch.where(s_ss > 0, ratio, nan)
    return torch.where(s_ss > 0, snr_s, nan), torch.where(s_ss > 0, snr_e, nan), ratio


def matched_pose_snr(
    sim: torch.Tensor, sim2: torch.Tensor, exp: torch.Tensor, pixel_size: float
) -> MatchedPoseSNR:
    """
    Signal-to-noise ratio of both stacks in each of `BANDS`, from matched-pose
    cross-spectra, and the residual envelope between them.

    With poses shared, every experimental image is a noisy realisation of a
    specific simulated one, ``exp_i = a(k) * signal_i + noise``. Two
    simulated seeds give the simulated signal power directly,
    ``<S1 S2*>``, and the experiment's coherent signal comes from
    ``<E S1*>``; what each stack has left over is its noise. The ratio of
    the two SNRs per band says how much cleaner the simulation is than the
    experiment at each scale. The Guinier slope of ``a(k)`` between 10 and
    4 Å is the envelope the experiment's particle carries beyond the
    simulation, reported as a residual and never applied.

    Every spectrum is summed over a band's pixels, and over the particles,
    BEFORE any division. Dividing per radial shell and averaging the shell
    ratios instead, as this function did until 2026-10-06, fails wherever a
    CTF zero falls inside a band for most of the stack: ``<S1 S2*>`` is then
    near zero in one shell, that shell's ratio is unbounded, and it decides
    the band. On EMPIAR-11752, whose defocus is near 5400 Å throughout, it
    reported two of five bands as undefined, a negative experimental SNR,
    and 0.00 for a control stack whose correct ratio is exactly 1; the band
    sums give 0.97-1.05 on that control. The same holds for the 33-12 Å
    signal amplitude, which read -1.2e11 there.

    The interval is a particle bootstrap of the same estimator. It matters
    most in the finest bands, where a few hundred particles carry little
    experimental signal: on EMPIAR-11377 the 4-3 Å ratio of 454 had an
    interval of 118 to 4500.

    Parameters
    ----------
    sim, sim2 : torch.Tensor
        Two simulated realisations (different seeds) at the same poses.
    exp : torch.Tensor
        Experimental stack, same particle order.
    pixel_size : float
        Pixel size in Å.

    Returns
    -------
    MatchedPoseSNR

    References
    ----------
    .. [1] Rosenthal, P. B., & Henderson, R. (2003). Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy. *Journal of Molecular
       Biology*, 333(4), 721–745. https://doi.org/10.1016/j.jmb.2003.07.013
    """
    terms = _band_spectra(sim, sim2, exp, pixel_size)
    snr_s, snr_e, ratio = _band_snr(terms)
    # Seeded, so the same stacks always give the same report.
    gen = torch.Generator().manual_seed(0)
    n_particles = terms.shape[0]
    draws = torch.randint(0, n_particles, (_SNR_BOOTSTRAP, n_particles), generator=gen)
    boot = torch.stack([_band_snr(terms[i])[2] for i in draws])
    # "nearest", not linear: a resample with no experimental signal is inf,
    # and interpolating between two infs gives NaN.
    lo, hi = (
        torch.nanquantile(boot, q, dim=0, interpolation="nearest")
        for q in _SNR_INTERVAL
    )
    s_ss_band, s_es_band = terms[:, 1, 0].mean(), terms[:, 1, 1].mean()
    plateau = float(s_es_band / s_ss_band) if s_ss_band > 0 else float("nan")

    n = sim.shape[-1]
    k = _kgrid(n, pixel_size)
    kk = _kbins(n, pixel_size)
    fe, fs, fs2 = (torch.fft.fft2(_zscore(x.float())) for x in (exp, sim, sim2))
    s_ss = _radial((fs * fs2.conj()).real.mean(0), k, pixel_size)
    s_es = _radial((fe * fs.conj()).real.mean(0), k, pixel_size)
    bfac, bfac_se = _residual_envelope(kk, s_es, s_ss)
    return MatchedPoseSNR(
        snr_sim=snr_s.tolist(),
        snr_exp=snr_e.tolist(),
        ratio=ratio.tolist(),
        ratio_lo=lo.tolist(),
        ratio_hi=hi.tolist(),
        signal_plateau=plateau,
        residual_bfactor=bfac,
        residual_bfactor_stderr=bfac_se,
    )


@dataclass
class TwinResult:
    """Outcome of `twin_test`."""

    exp_vs_sim: float
    sim_vs_sim: float
    cohen_d: float
    exp_vs_sim_values: torch.Tensor
    sim_vs_sim_values: torch.Tensor


def twin_test(
    sim: torch.Tensor, sim2: torch.Tensor, exp: torch.Tensor, max_shift: int = 8
) -> TwinResult:
    """
    Is an experimental particle any further from its simulated twin than a
    second simulated realisation is?

    Low-passed peak correlation of ``exp_i`` with ``sim_i``, against that of
    ``sim2_i`` with ``sim_i``. If the two distributions overlap (Cohen's d
    near 0) a classifier has nothing to separate; a d of 2-3 means the
    experiment is systematically unlike its twin.
    """
    s, s2, e = (_zscore(torch.as_tensor(butter(x.float()))) for x in (sim, sim2, exp))
    c_es = _xcorr_max(s, e, max_shift)
    c_ss = _xcorr_max(s, s2, max_shift)
    pooled = torch.sqrt(0.5 * (c_ss.var() + c_es.var())).clamp(min=1e-12)
    return TwinResult(
        exp_vs_sim=float(c_es.mean()),
        sim_vs_sim=float(c_ss.mean()),
        cohen_d=float((c_ss.mean() - c_es.mean()) / pooled),
        exp_vs_sim_values=c_es,
        sim_vs_sim_values=c_ss,
    )
