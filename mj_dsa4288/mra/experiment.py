"""
Sweep drivers reproducing the two headline plots of Perry et al. (2019).

* :func:`snr_sweep` -- Sigworth-style: reconstruction SNR versus data SNR at a
  fixed (or per-SNR) sample size. Slopes should read 1 at high SNR, 3 at low.
* :func:`sample_complexity_curve` -- the paper's own definition: for each SNR,
  the smallest ``n`` (doubling search) such that ``rho(theta_tilde, theta) <= eps``.
  Should scale as ``SNR^{-1}`` at high SNR and ``SNR^{-3}`` at low SNR.

Rows are plain dicts so they can be dumped to JSON and re-plotted later.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from mj_dsa4288.mra.em import em_mra
from mj_dsa4288.mra.jennrich import homojen
from mj_dsa4288.mra.metrics import reconstruction_snr, rho, two_regime_slopes
from mj_dsa4288.mra.model import SNRConvention, sample_mra, sigma_for_snr

Estimator = Literal["em", "jennrich"]


@dataclass
class SweepConfig:
    """Settings for :func:`snr_sweep` and :func:`sample_complexity_curve`."""

    snr_values: list[float] = field(default_factory=lambda: list(np.logspace(-1, 2, 7)))
    n_samples: int | list[int] = 2000
    n_repeats: int = 1
    estimator: Estimator = "em"
    snr_convention: SNRConvention = "paper"
    rotations: bool = False
    seed: int = 0
    em_iters: int = 100
    em_batch_size: int = 2048
    em_oracle_init: bool = False
    # sample-complexity search
    eps: float = 0.2
    n_min: int = 32
    n_max: int = 1_000_000
    device: str = "cpu"


def _estimate(
    y: torch.Tensor,
    sigma: float,
    cfg: SweepConfig,
    theta: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    if cfg.estimator == "em":
        init = theta if cfg.em_oracle_init else None
        return em_mra(
            y,
            sigma,
            n_iter=cfg.em_iters,
            init=init,
            rotations=cfg.rotations,
            batch_size=cfg.em_batch_size,
            generator=gen,
        ).theta
    if cfg.estimator == "jennrich":
        if y.ndim != 2:
            raise ValueError(
                "the Jennrich estimator is implemented for 1-D signals only"
            )
        return homojen(y, sigma, generator=gen)
    raise ValueError(f"unknown estimator {cfg.estimator!r}")


def snr_sweep(theta: torch.Tensor, cfg: SweepConfig) -> list[dict[str, Any]]:
    """
    Reconstruction SNR versus data SNR (Fig. 2 of the paper / Sigworth Fig. 2C).

    Parameters
    ----------
    theta : torch.Tensor
        Clean template, ``[d]`` or ``[H, W]``.
    cfg : SweepConfig
        Sweep settings. ``n_samples`` may be a list with one entry per SNR.

    Returns
    -------
    list[dict]
        One row per (SNR, repeat) with keys ``snr, sigma, n, rho, rec_snr,
        estimator, seconds``.
    """
    theta = theta.to(cfg.device)
    rows: list[dict[str, Any]] = []
    n_list = (
        cfg.n_samples
        if isinstance(cfg.n_samples, list)
        else [cfg.n_samples] * len(cfg.snr_values)
    )
    for snr, n in zip(cfg.snr_values, n_list):
        sigma = sigma_for_snr(theta, snr, cfg.snr_convention)
        for rep in range(cfg.n_repeats):
            gen = torch.Generator().manual_seed(
                cfg.seed + 1000 * rep + int(1e6 * snr) % 997
            )
            t0 = time.perf_counter()
            y, _, _ = sample_mra(
                theta, n, sigma, generator=gen, rotations=cfg.rotations
            )
            est = _estimate(y, sigma, cfg, theta, gen)
            err = rho(est, theta, rotations=cfg.rotations)
            rows.append(
                {
                    "snr": float(snr),
                    "sigma": float(sigma),
                    "n": int(n),
                    "repeat": rep,
                    "rho": err,
                    "rec_snr": reconstruction_snr(est, theta, rotations=cfg.rotations),
                    "estimator": cfg.estimator,
                    "snr_convention": cfg.snr_convention,
                    "seconds": time.perf_counter() - t0,
                }
            )
    return rows


def sample_complexity_curve(
    theta: torch.Tensor, cfg: SweepConfig
) -> list[dict[str, Any]]:
    """
    Smallest ``n`` reaching ``rho <= eps`` at each SNR (doubling search).

    Parameters
    ----------
    theta : torch.Tensor
        Clean template.
    cfg : SweepConfig
        Uses ``eps``, ``n_min``, ``n_max`` and the estimator settings.

    Returns
    -------
    list[dict]
        One row per SNR with keys ``snr, sigma, n_required, rho, reached``.
    """
    theta = theta.to(cfg.device)
    rows: list[dict[str, Any]] = []
    for snr in cfg.snr_values:
        sigma = sigma_for_snr(theta, snr, cfg.snr_convention)
        n = cfg.n_min
        reached = False
        err = float("nan")
        while n <= cfg.n_max:
            gen = torch.Generator().manual_seed(cfg.seed + n)
            y, _, _ = sample_mra(
                theta, n, sigma, generator=gen, rotations=cfg.rotations
            )
            est = _estimate(y, sigma, cfg, theta, gen)
            err = rho(est, theta, rotations=cfg.rotations)
            if err <= cfg.eps:
                reached = True
                break
            n *= 2
        rows.append(
            {
                "snr": float(snr),
                "sigma": float(sigma),
                "n_required": int(n if reached else cfg.n_max),
                "rho": err,
                "reached": reached,
                "estimator": cfg.estimator,
                "eps": cfg.eps,
            }
        )
    return rows


def summarise_slopes(
    rows: list[dict[str, Any]], split: float, key: str = "rec_snr"
) -> dict[str, float]:
    """Mean ``key`` per SNR, then log-log slopes below/above ``split``."""
    snrs = sorted({r["snr"] for r in rows})
    means = [np.mean([r[key] for r in rows if r["snr"] == s]) for s in snrs]
    y = np.asarray(means, float)
    if key == "n_required":
        y = 1.0 / y  # so that the expected slopes are +1 and +3, like rec_snr
    low, high = two_regime_slopes(np.asarray(snrs), y, split)
    return {"low_snr_slope": low, "high_snr_slope": high, "split": split}


def save_rows(
    rows: list[dict[str, Any]], path: Path, cfg: SweepConfig | None = None
) -> None:
    """Dump rows (and the config) to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(cfg) if cfg else None, "rows": rows}
    path.write_text(json.dumps(payload, indent=2, default=float))


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load rows written by :func:`save_rows`."""
    return json.loads(path.read_text())["rows"]


def plot_sigworth(
    rows: list[dict[str, Any]], path: Path, split: float | None = None, title: str = ""
) -> None:
    """Log-log reconstruction SNR versus data SNR with slope-1 and slope-3 guides."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array(sorted({r["snr"] for r in rows}))
    means = np.array(
        [np.mean([r["rec_snr"] for r in rows if r["snr"] == s]) for s in snrs]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(snrs, means, "o-", label="reconstruction SNR")
    anchor = means[-1] / snrs[-1]
    ax.loglog(snrs, anchor * snrs, "--", c="gray", label="slope 1")
    ax.loglog(snrs, anchor * snrs[-1] ** -2 * snrs**3, ":", c="gray", label="slope 3")
    if split is not None:
        lo, hi = two_regime_slopes(snrs, means, split)
        ax.set_title(f"{title} slopes: low={lo:.2f}, high={hi:.2f}")
    elif title:
        ax.set_title(title)
    ax.set_xlabel("data SNR")
    ax.set_ylabel("reconstruction SNR")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_complexity(
    rows: list[dict[str, Any]], path: Path, title: str = ""
) -> None:
    """Log-log ``n_required`` versus SNR with ``SNR^-1`` and ``SNR^-3`` guides."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array([r["snr"] for r in rows])
    n_req = np.array([r["n_required"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(snrs, n_req, "o-", label="n required")
    ax.loglog(snrs, n_req[-1] * snrs[-1] / snrs, "--", c="gray", label="SNR^-1")
    ax.loglog(snrs, n_req[-1] * (snrs[-1] / snrs) ** 3, ":", c="gray", label="SNR^-3")
    ax.set_xlabel("data SNR")
    ax.set_ylabel("samples to reach eps")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
