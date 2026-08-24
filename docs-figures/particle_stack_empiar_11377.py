"""
Generate the EMPIAR-11377 comparison figures for
docs/user-guide/particle-stack.md, showing that `specter simulate particles`
driven from a real CryoSPARC .cs file (``cs_path``) produces images that
resemble real experimental data both qualitatively (raw images side by side)
and quantitatively (CryoSPARC 2D classification of a mixed real+simulated
stack).

Reads three fixtures, all under docs-figures/data/ -- none of these are
pytest fixtures (the pipeline's cs_path *code path* is covered separately,
against a fabricated minimal .cs, by
tests/test_pipelines.py::test_run_particle_stack_from_csfile; this script's
job is a real-data demonstration, not test coverage):

- empiar-11377-passthrough-5particles.cs
    First 5 rows of the real CryoSPARC passthrough .cs from EMPIAR-11377
    (https://www.ebi.ac.uk/empiar/EMPIAR-11377/), job J214 -- supplies real
    pose/CTF/pixel_size/voltage/alpha for 5 particles of PDB 8b0x (a
    translating 70S ribosome), read via `extract_parameters_from_csfile`.
- empiar-11377-real-5particles.mrcs
    The 5 real experimental particle images those poses/CTF came from
    (restacked, same row order as the .cs file above).
- empiar-11377-mixed-2dclasses.csv
    Per-particle 2D-class assignment from a CryoSPARC 2D Classification job
    run on a *mixed* stack of 2000 real EMPIAR-11377 particles + 2000
    SPECTER-simulated particles (same pdb_code/physics params as this
    script, generated in bulk offline). This one isn't reproducible by
    this script alone -- it requires actually running CryoSPARC -- but
    it's the stronger evidence: if simulated particles are physically
    realistic, CryoSPARC's own 2D classification should sort them into the
    same classes as the real particles they're mixed with, roughly in
    proportion to each class's real/simulated split.

Run with: uv run python docs-figures/particle_stack_empiar_11377.py
Saves PNGs directly into docs/assets/images/ (consumed by
docs/user-guide/particle-stack.md). Needs a GPU with a few GB free for the
512^3 multislice potential/propagation; falls back to CPU (slow) otherwise.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import torch

from specter.config import ParticleStackConfig
from specter.image import normalize_particles
from specter.pipelines import run_particle_stack
from specter.plots import _deep_palette

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"

PDB_CODE = "8b0x"  # translating 70S ribosome -- EMPIAR-11377's particle
NUM_PIXELS = 512
DOSE = "40"  # e-/A^2, from EMDB's Experiment tab for this deposition
COINCIDENCE_RADIUS = "2"  # pixels, matches the contrast of the real data
POTENTIAL_SCALE = "0.5"  # approximates a thicker ice layer than the box


def _generate_specter_particles(device: str) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as tmp:
        config = ParticleStackConfig(
            pdb_code=PDB_CODE,
            n_pixels=NUM_PIXELS,
            cs_path=str(DATA_DIR / "empiar-11377-passthrough-5particles.cs"),
            n_particles=5,
            dose=DOSE,
            scattering_model="multislice",
            aberration_model="holography",
            noise_model="poisson",
            ice_model="gd",
            coincidence_radius=COINCIDENCE_RADIUS,
            potential_scale=POTENTIAL_SCALE,
            detector_model="none",
            normalize_particles=True,
            device=device,
            batchsize=1,
            output_dir=tmp,
            filename="empiar_11377_sim",
        )
        run_particle_stack(config)
        with mrcfile.open(Path(tmp) / "empiar_11377_sim.mrcs") as mrc:
            return torch.as_tensor(np.asarray(mrc.data).copy())


def _load_real_particles() -> torch.Tensor:
    with mrcfile.open(DATA_DIR / "empiar-11377-real-5particles.mrcs") as mrc:
        images = torch.as_tensor(np.asarray(mrc.data).copy())
    particles, _means, _stds = normalize_particles(images)
    return particles


def _figure_comparison(sim: torch.Tensor, real: torch.Tensor) -> None:
    n = sim.shape[0]
    fig, axes = plt.subplots(2, n, dpi=200, constrained_layout=True, figsize=(2 * n, 4))
    for ax in axes.ravel():
        ax.set(xticks=[], yticks=[])
    for i in range(n):
        axes[0, i].imshow(sim[i], cmap="gray")
        axes[1, i].imshow(real[i], cmap="gray")
    axes[0, 0].set_ylabel("SPECTER")
    axes[1, 0].set_ylabel("EMPIAR-11377")
    path = OUT_DIR / "particle-stack-empiar-11377-comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


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
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    sim = _generate_specter_particles(device)
    real = _load_real_particles()
    _figure_comparison(sim, real)
    _figure_2d_classes()


if __name__ == "__main__":
    main()
