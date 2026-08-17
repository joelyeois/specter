"""
Generate the figures for docs/concepts/detector.md.

Calls `specter.detectors`' bundled MTF/DQE(0) functions and
`specter.microscope.Detector`'s coincidence-loss machinery directly, so
the curves plotted are exactly what `BaseImager._init_detector_mtf` and
`Detector.apply_coincidence` produce for a real imaging run.

Run with: uv run python docs-figures/detector.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from specter.arrays import radial_profile_2d
from specter.detectors import (
    DQE0,
    dqe0_for_detector,
    falcon4i_200kv,
    falcon4i_300kv,
    k3_200kv,
    k3_300kv,
    perfect_detector,
)
from specter.microscope import Detector
from specter.plots import _deep_palette

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"

DEVICE = "cpu"


def figure_mtf_overlay() -> None:
    n, dx = 512, 1.0
    curves = [
        ("K3, 200 kV", k3_200kv, "-"),
        ("K3, 300 kV", k3_300kv, "-"),
        ("Falcon 4i, 200 kV", falcon4i_200kv, "--"),
        ("Falcon 4i, 300 kV", falcon4i_300kv, "--"),
        ("Perfect (pixel sinc)", perfect_detector, ":"),
    ]
    palette = _deep_palette(len(curves))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    for color, (label, fn, ls) in zip(palette, curves):
        k, mtf = fn(n, dx, DEVICE, return1d=True)
        k_nyquist = 1 / (2 * dx)
        ax.plot(
            k.cpu().numpy() / k_nyquist,
            mtf.cpu().numpy(),
            color=color,
            linestyle=ls,
            label=label,
        )
    ax.set_xlabel("Spatial frequency (fraction of Nyquist)")
    ax.set_ylabel("MTF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "detector-mtf-overlay.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_dqe0_bar() -> None:
    models = ["k3_200kv", "k3_300kv", "falcon4i_200kv", "falcon4i_300kv", "perfect"]
    labels = [
        "K3\n200 kV",
        "K3\n300 kV",
        "Falcon 4i\n200 kV",
        "Falcon 4i\n300 kV",
        "Perfect",
    ]
    values = [dqe0_for_detector(m) for m in models]
    has_source = [m in DQE0 for m in models]
    palette = _deep_palette(2)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    colors = [palette[0] if s else "lightgray" for s in has_source]
    bars = ax.bar(labels, values, color=colors)
    for bar, s in zip(bars, has_source):
        if not s:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                "no published\nvalue -> 1.0",
                ha="center",
                fontsize=7.5,
                color="gray",
            )
    ax.set_ylabel("DQE(0) -- zero-frequency counting efficiency")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
    fig.tight_layout()
    path = OUT_DIR / "detector-dqe0-bar.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_coincidence_loss() -> None:
    """(a) Detected-electron efficiency vs. incident dose rate: pure Poisson
    stays at 1.0 by construction, while coincidence loss grows with rate --
    this is the effect being calibrated against real Falcon 4i data (see
    Detector.apply_coincidence's docstring). (b) The radially averaged
    power spectrum of a coincidence-suppressed frame vs. a plain-Poisson
    frame at the same (moderately high) dose, both under otherwise-uniform
    illumination so the only structure in the spectrum is the noise
    statistics themselves."""
    pixel_size = 1.0
    h = w = 96
    coinc_radius = 2.394  # px, Falcon 4i calibration (see Detector.apply_coincidence)
    intensity_map = torch.ones(h, w) / (h * w)
    detector = Detector(pixel_size)

    doses = torch.logspace(-1, 2, 14)
    efficiency = []
    for dose in doses:
        out = detector.apply_detector_physics(
            intensity_map, pixel_size, float(dose), coinc_radius_pixels=coinc_radius
        )
        expected = float(dose) * pixel_size**2 * h * w
        efficiency.append(out.sum().item() / expected)

    palette = _deep_palette(2)
    fig, (ax_eff, ax_ps) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=190)

    ax_eff.semilogx(
        doses.numpy(), efficiency, color=palette[0], marker="o", markersize=4
    )
    ax_eff.axhline(1.0, color="gray", linestyle=":", label="Ideal (no coincidence)")
    ax_eff.set_xlabel("Incident dose (e⁻/Å²/frame)")
    ax_eff.set_ylabel("Detected / incident electrons")
    ax_eff.set_ylim(0, 1.05)
    ax_eff.set_title(f"Coincidence radius = {coinc_radius:g} px", fontsize=10)
    ax_eff.legend(fontsize=8)
    ax_eff.grid(True, alpha=0.3)

    dose_demo = 8.0
    n_realizations = 40
    torch.manual_seed(0)
    lam = intensity_map * dose_demo * pixel_size**2 * h * w
    k_nyquist = 1 / (2 * pixel_size)

    for color, label, use_coincidence in [
        (palette[0], "Plain Poisson", False),
        (palette[1], "With coincidence loss", True),
    ]:
        power_sum = None
        for _ in range(n_realizations):
            if use_coincidence:
                frame = detector.apply_detector_physics(
                    intensity_map,
                    pixel_size,
                    dose_demo,
                    coinc_radius_pixels=coinc_radius,
                )
            else:
                frame = torch.poisson(lam)
            power = torch.abs(torch.fft.fft2(frame - frame.mean())) ** 2
            power = torch.fft.fftshift(power)
            power_sum = power if power_sum is None else power_sum + power
        r, profile = radial_profile_2d(power_sum / n_realizations, return_r=True)
        # Normalized by the high-frequency plateau so the two curves --
        # which differ hugely in absolute power once coincidence loss has
        # thrown away most of the incident electrons -- are compared by
        # shape (is there a dip approaching k=0?), not by scale.
        plateau = profile[len(profile) // 2 :].mean()
        ax_ps.plot(
            (r.float() / (h * pixel_size) / k_nyquist).numpy(),
            (profile / plateau).numpy(),
            color=color,
            label=label,
        )
    ax_ps.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
    ax_ps.set_xlabel("Spatial frequency (fraction of Nyquist)")
    ax_ps.set_ylabel("Radial noise power (normalized to high-k plateau)")
    ax_ps.set_title(
        f"Dose = {dose_demo:g} e⁻/Å²/frame, {n_realizations} realizations averaged",
        fontsize=10,
    )
    ax_ps.legend(fontsize=8)
    ax_ps.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "detector-coincidence-loss.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    figure_mtf_overlay()
    figure_dqe0_bar()
    figure_coincidence_loss()


if __name__ == "__main__":
    main()
