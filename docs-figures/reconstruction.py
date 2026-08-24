"""
Generate the figures for docs/user-guide/reconstruction.md.

Calls `specter.plots.plot_halfmap_fsc` -- the same function
`specter.pipelines._reconstruct._compute_and_save_gold_standard_fsc` calls at
the end of a real gold-standard run -- on a synthetic band-limited pair, so
the curve shape is genuine even though no reconstruction was actually run.

Run with: uv run python docs-figures/reconstruction.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from specter.plots import plot_halfmap_fsc

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"


def _band_limited_pair(
    n: int, voxel_size: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two noisy observations of one band-limited signal: an FSC that falls
    off smoothly with resolution, the way a real half-map pair's does,
    rather than a synthetic curve with no physical shape at all."""
    torch.manual_seed(seed)
    kf = torch.fft.fftfreq(n, d=voxel_size)
    k_mag = torch.sqrt(sum(g**2 for g in torch.meshgrid(kf, kf, kf, indexing="ij")))
    signal = torch.fft.ifftn(
        torch.fft.fftn(torch.randn(n, n, n)) * torch.exp(-((k_mag / 0.1) ** 2))
    ).real
    signal = signal / signal.std()
    return (
        signal + 0.5 * torch.randn(n, n, n),
        signal + 0.5 * torch.randn(n, n, n),
    )


def figure_gold_standard_fsc() -> None:
    n, voxel_size = 64, 1.5
    vol_a, vol_b = _band_limited_pair(n, voxel_size, seed=0)
    fig, resolutions = plot_halfmap_fsc(
        [vol_a],
        [vol_b],
        voxel_size=voxel_size,
        labels=["gold-standard (synthetic example)"],
        show=False,
    )
    assert fig is not None
    fig.savefig(OUT_DIR / "reconstruction-gold-standard-fsc.png", bbox_inches="tight")
    plt.close(fig)
    print(f"gold-standard synthetic example resolution: {resolutions[0]}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_gold_standard_fsc()
