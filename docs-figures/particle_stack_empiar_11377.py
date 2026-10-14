"""
Generate the EMPIAR-11377 2D-classification figure for
docs/user-guide/particle-stack.md: CryoSPARC's 2D classification of a mixed
stack of 2000 real and 2000 SPECTER-simulated EMPIAR-11377 particles.

Reads docs-figures/data/empiar-11377-mixed-2dclasses.csv, the per-particle
2D-class assignment from that CryoSPARC job. It is not reproducible by this
script alone, since it requires running CryoSPARC; the simulated half was
generated at the refinement's poses in the 8B0X frame (J205). If simulated
particles are physically realistic, CryoSPARC's own 2D classification should
sort them into the same classes as the real particles, in similar
proportion.

The side-by-side comparison of individual real and simulated particles lives
in docs/tutorials/real-dataset.md (docs-figures/tutorials/real_dataset.py),
which uses the refinement's poses in the model's frame.

Run with: uv run python docs-figures/particle_stack_empiar_11377.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specter.plots import _deep_palette

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"


def _figure_2d_classes() -> None:
    df = pd.read_csv(DATA_DIR / "empiar-11377-mixed-2dclasses.csv")
    n_classes = int(df["class_2d"].max()) + 1
    sim_counts = np.bincount(
        df.loc[df["source"] == "SPECTER", "class_2d"], minlength=n_classes
    )
    real_counts = np.bincount(
        df.loc[df["source"] == "EMPIAR-11377", "class_2d"], minlength=n_classes
    )
    order = np.argsort(sim_counts + real_counts)[::-1]
    color_real, color_sim = _deep_palette(4)[0], _deep_palette(4)[3]

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
    xpos = np.arange(n_classes)
    ax.bar(xpos, real_counts[order], color=color_real, label="EMPIAR-11377")
    ax.bar(
        xpos,
        sim_counts[order],
        bottom=real_counts[order],
        color=color_sim,
        label="SPECTER",
    )
    # All 50 class IDs don't fit legibly in the plot width regardless of font
    # size -- they're arbitrary CryoSPARC labels, not information the reader
    # needs per-bar. Thinning to every 5th tick leaves the descending-count
    # pattern intact while keeping what remains actually readable.
    ax.set_xticks(xpos[::5])
    ax.set_xticklabels(order[::5], fontsize=11)
    ax.set_xlabel(
        "CryoSPARC 2D class (sorted by total particle count, descending)", fontsize=12
    )
    ax.set_ylabel("particle count", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    path = OUT_DIR / "particle-stack-empiar-11377-2d-classes.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    _figure_2d_classes()


if __name__ == "__main__":
    main()
