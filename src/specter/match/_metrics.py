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
    a: torch.Tensor, pixel_size: float, low_A: float, high_A: float
) -> torch.Tensor:
    k = _kgrid(a.shape[-1], pixel_size)
    mask = ((k >= 1 / low_A) & (k < 1 / high_A)).to(a.dtype)
    out = torch.fft.ifft2(torch.fft.fft2(a) * mask).real
    return _zscore(out)


@dataclass
class PoseAlignmentResult:
    """Outcome of `matched_index_correlation`."""

    matched: float
    shuffled: float
    fraction_above: float
    z_score: float

    @property
    def passed(self) -> bool:
        """Matched pairs beat shuffled pairs decisively (z >= 5, >60% of pairs)."""
        return self.z_score >= 5.0 and self.fraction_above > 0.6


def matched_index_correlation(
    sim: torch.Tensor,
    exp: torch.Tensor,
    pixel_size: float,
    band_A: tuple[float, float] = (60.0, 15.0),
    max_shift: int = 12,
    seed: int = 0,
) -> PoseAlignmentResult:
    """
    Test whether simulated particle ``i`` shows the same view as experimental
    particle ``i``.

    The peak cross-correlation of each matched pair, band-passed to
    ``band_A`` and allowed a small shift, is compared with the same statistic
    for a random pairing. Poses aligned to the model put the matched value
    well above the shuffled one; poses from a refinement that was never
    aligned to the model (no Align 3D step against it) put the two at the
    same level. This is the check that the EMPIAR-11377 sweep of
    2026-09-01/02 lacked: sixty classifications could not see that every
    simulated view was rotated 142 degrees from its experimental one.

    Parameters
    ----------
    sim, exp : torch.Tensor
        Stacks of shape (N, n, n) sharing particle index.
    pixel_size : float
        Pixel size in Å.
    band_A : (float, float), optional
        Band-pass in Å, (low-resolution edge, high-resolution edge). The
        default 60-15 Å carries view-dependent structure for any particle
        and little noise.
    max_shift : int, optional
        Shift search radius in pixels.
    seed : int, optional
        Seed of the shuffled pairing.

    Returns
    -------
    PoseAlignmentResult
    """
    n = sim.shape[0]
    s = _bandpass(sim.float(), pixel_size, *band_A)
    e = _bandpass(exp.float(), pixel_size, *band_A)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    m = _xcorr_max(s, e, max_shift)
    mp = _xcorr_max(s, e[perm], max_shift)
    sem = float(m.std() / n**0.5) if n > 1 else float("inf")
    return PoseAlignmentResult(
        matched=float(m.mean()),
        shuffled=float(mp.mean()),
        fraction_above=float((m > mp).float().mean()),
        z_score=float((m.mean() - mp.mean()) / sem) if sem > 0 else float("inf"),
    )


def edge_band_means(
    stack: torch.Tensor, bands_px: tuple[int, ...] = (2, 4, 8, 16)
) -> list[float]:
    """
    Mean of the stack-averaged, z-scored image in concentric bands from the
    box edge inward: 0-2, 2-4, 4-8, 8-16 px.

    A deterministic pattern shared by every simulated particle -- the
    Fresnel ring an ice slab truncated at the field of view produced until
    2026-09-02 -- shows up here as a mean well away from zero, while every
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


def annulus_std_profile(
    stack: torch.Tensor, radii_px: tuple[int, ...] = (20, 40, 60, 80, 100, 128)
) -> list[float]:
    """
    Standard deviation of the low-passed, z-scored image inside concentric
    annuli (0-20, 20-40, ... px) plus the corners beyond the last radius,
    averaged over particles.

    Outside the particle this measures background structure: neighbours,
    ice, thickness gradients. It is what set the ice thickness and neighbour
    spacing in the 2026-09-03 verification, and what exposed a close-packed
    CCMV monolayer the simulator cannot reproduce.
    """
    n = stack.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    rr = torch.as_tensor(np.hypot(yy - n // 2, xx - n // 2)).float()
    a = _zscore(torch.as_tensor(butter(stack.float())))
    out, lo = [], 0.0
    for hi in radii_px:
        sel = (rr >= lo) & (rr < hi)
        out.append(
            float(a[:, sel].std(dim=1).mean()) if bool(sel.any()) else float("nan")
        )
        lo = float(hi)
    sel = rr >= lo
    out.append(float(a[:, sel].std(dim=1).mean()) if bool(sel.any()) else float("nan"))
    return out


@dataclass
class MatchedPoseSNR:
    """Outcome of `matched_pose_snr`."""

    snr_sim: list[float]
    snr_exp: list[float]
    ratio: list[float]
    signal_plateau: float  # exp/sim signal amplitude ratio at 33-12 Å (z-scored units)
    residual_bfactor: float  # Å², Guinier slope of the exp/sim signal ratio, 10-4 Å


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
    the two SNRs per band is the number that separated the datasets that
    mixed in CryoSPARC from the ones that did not: flat near 1-5 on the
    energy-filtered sets, growing to 40-100x with frequency on the
    unfiltered ones. The Guinier slope of ``a(k)`` between 10 and 4 Å is
    the envelope the experiment carries beyond the simulation, applied as
    a B-factor when it is clearly positive.

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
    """
    n = sim.shape[-1]
    k = _kgrid(n, pixel_size)
    kk = _kbins(n, pixel_size)
    fe, fs, fs2 = (torch.fft.fft2(_zscore(x.float())) for x in (exp, sim, sim2))
    s_ss = _radial((fs * fs2.conj()).real.mean(0), k, pixel_size)
    s_es = _radial((fe * fs.conj()).real.mean(0), k, pixel_size)
    p_e = _radial((fe.abs() ** 2).mean(0), k, pixel_size)
    p_s = _radial((fs.abs() ** 2).mean(0), k, pixel_size)
    a = s_es / s_ss.clamp(min=1e-9)
    snr_e = (a**2 * s_ss) / (p_e - a**2 * s_ss).clamp(min=1e-9)
    snr_s = s_ss / (p_s - s_ss).clamp(min=1e-9)
    ratio = []
    for band in BANDS:
        se, ss = _band_mean(snr_e, kk, band), _band_mean(snr_s, kk, band)
        ratio.append(ss / se if se > 0 else float("nan"))
    sel = (kk >= 0.1) & (kk <= 0.25) & (a > 0)
    if int(sel.sum()) >= 3:
        slope, _ = np.polyfit((kk[sel] ** 2).numpy(), np.log(a[sel].numpy()), 1)
        bfac = float(-4.0 * slope)
    else:
        bfac = float("nan")
    return MatchedPoseSNR(
        snr_sim=[_band_mean(snr_s, kk, b) for b in BANDS],
        snr_exp=[_band_mean(snr_e, kk, b) for b in BANDS],
        ratio=ratio,
        signal_plateau=_band_mean(a, kk, BANDS[1]),
        residual_bfactor=bfac,
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
