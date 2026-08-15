"""
Generate the figures for docs/concepts/cryoet-specimen/carbon-film.md,
which explains the carbon support film component of
`specter build tomogram` (``specter.specimen._carbon``).

Calls the shipped `CarbonFilmGenerator`/`edge_hole_center` directly rather
than reimplementing the alpha-shape geometry, so the figures cannot
silently drift from what's actually shipped.

Run with: uv run python docs-figures/cryoet_specimen_carbon_film.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from specter.specimen._carbon import (
    QUANTIFOIL_R1_2_HOLE_RADIUS,
    CarbonFilmGenerator,
    edge_hole_center,
)

OUT_DIR = "docs/assets/images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHAPE_ZYX = (40, 400, 400)
VOXEL_SIZE = 5.0
THICKNESS = 150.0
SEED = 4


def _film(
    edge_fraction: float = 0.25,
    edge_roughness: float = 60.0,
    side: str = "bottom",
    seed: int = SEED,
) -> np.ndarray:
    gen = CarbonFilmGenerator(voxel_size=VOXEL_SIZE, seed=seed, device=DEVICE)
    center = edge_hole_center(
        SHAPE_ZYX, VOXEL_SIZE, QUANTIFOIL_R1_2_HOLE_RADIUS, edge_fraction, side
    )
    inst = gen.generate(
        SHAPE_ZYX,
        thickness=THICKNESS,
        hole_radius=QUANTIFOIL_R1_2_HOLE_RADIUS,
        edge_roughness=edge_roughness,
        hole_center=center,
    )
    return inst.density.cpu().numpy()


def figure_hero() -> None:
    """Page-top hero: the film summed along Z -- a thin strip of carbon
    intruding from one frame edge, with the rough, genuinely 3D rim the
    alpha shape produces."""
    density = _film()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(density.sum(axis=0), cmap="gray_r", origin="lower")
    ax.axis("off")
    plt.tight_layout(pad=0)
    path = f"{OUT_DIR}/cryoet-carbon-hero.png"
    plt.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def figure_hole_geometry() -> None:
    """Why `edge_hole_center` exists: a real Quantifoil R1.2/1.3 hole is
    0.6 micron in radius, far bigger than a tomogram's field of view, so
    the realistic case is a nearly-straight boundary crossing the frame --
    not a small hole sitting inside it. Left, to scale; right, the
    resulting projections at three `edge_fraction` values."""
    nz, ny, nx = SHAPE_ZYX
    half_x, half_y = nx * VOXEL_SIZE / 2, ny * VOXEL_SIZE / 2

    fig = plt.figure(figsize=(11.0, 3.9))
    ax = fig.add_subplot(1, 4, 1)
    for edge_fraction, color in zip(
        [0.1, 0.25, 0.4], ["#1b7f7f", "#c25e00", "#5b4b8a"]
    ):
        cx, cy = edge_hole_center(
            SHAPE_ZYX, VOXEL_SIZE, QUANTIFOIL_R1_2_HOLE_RADIUS, edge_fraction, "bottom"
        )
        ax.add_patch(
            Circle(
                (cx, cy),
                QUANTIFOIL_R1_2_HOLE_RADIUS,
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                label=f"edge_fraction={edge_fraction}",
            )
        )
    ax.add_patch(
        Rectangle(
            (-half_x, -half_y),
            2 * half_x,
            2 * half_y,
            fill=False,
            edgecolor="k",
            linewidth=1.6,
        )
    )
    ax.set_xlim(-7500, 7500)
    ax.set_ylim(-7000, 8000)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7.5, loc="upper center")
    ax.set_title("hole vs. field of view", fontsize=10)

    for i, edge_fraction in enumerate([0.1, 0.25, 0.4]):
        ax = fig.add_subplot(1, 4, i + 2)
        ax.imshow(
            _film(edge_fraction=edge_fraction).sum(axis=0),
            cmap="gray_r",
            origin="lower",
        )
        ax.set_title(f"edge_fraction = {edge_fraction}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-carbon-hole-geometry.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_roughness() -> None:
    """`edge_roughness` sweep, cropped to the rim. 0 gives the bare alpha
    shape of an unperturbed seed cloud (already not a clean circle, since
    the seeds are a Poisson cloud rather than a boundary parameterisation);
    60 is the shipped default; 150 pinches off islands."""
    values = [0.0, 60.0, 150.0]
    fig, axes = plt.subplots(1, len(values), figsize=(3.4 * len(values), 3.4))
    for ax, edge_roughness in zip(axes, values):
        density = _film(edge_fraction=0.25, edge_roughness=edge_roughness)
        projection = density.sum(axis=0)
        ax.imshow(projection[:150, :], cmap="gray_r", origin="lower", aspect="auto")
        ax.set_title(f"edge_roughness = {edge_roughness:g} A", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-carbon-roughness.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_slices() -> None:
    """The rim is 3D, not a swept 2D outline: three z-slices through the
    same film have different boundaries, with detached islands and
    overhanging lips appearing at some heights and not others."""
    density = _film(edge_fraction=0.25)
    nz = density.shape[0]
    film_rows = np.flatnonzero(density.sum(axis=(1, 2)) > 0)
    z_lo, z_hi = int(film_rows[0]), int(film_rows[-1])
    picks = [z_lo + 1, (z_lo + z_hi) // 2, z_hi - 1]

    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 3.2))
    for ax, z in zip(axes, picks):
        ax.imshow(density[z][:150, :] > 0, cmap="gray_r", origin="lower", aspect="auto")
        ax.set_title(f"z slice {z} of {nz}", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-carbon-slices.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_calibration() -> None:
    """The film's bulk mean inner potential, against the 7.8-9.1 V range
    reported for amorphous carbon -- and the reason the placed density is
    0.7x bulk rather than bulk: the independent-atom model carries no
    bonding correction, so full bulk density overshoots."""
    fractions = [0.5, 0.7, 1.0]
    gen = CarbonFilmGenerator(voxel_size=VOXEL_SIZE, seed=SEED, device=DEVICE)
    per_atom = gen.atom_potential_integral
    from specter.specimen._carbon import CARBON_DENSITY_G_CM3, CARBON_MOLAR_MASS
    from specter.specimen._grid import _number_density_per_a3

    bulk = _number_density_per_a3(CARBON_DENSITY_G_CM3, CARBON_MOLAR_MASS)
    values = [f * bulk * per_atom for f in fractions]

    density = _film(edge_fraction=0.4)
    interior = density[density > 0]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    ax.axhspan(7.8, 9.1, color="0.85", label="literature, amorphous C")
    bars = ax.bar(
        [f"{f:g}x bulk" for f in fractions], values, color="#4c6e7a", width=0.55
    )
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.3,
            f"{v:.1f} V",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel("mean inner potential (V)")
    ax.set_ylim(0, 14)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("placed density calibration", fontsize=11)

    ax = axes[1]
    ax.hist(interior, bins=60, color="#4c6e7a")
    ax.axvline(
        gen.mean_inner_potential,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label=f"MIP = {gen.mean_inner_potential:.2f} V",
    )
    ax.set_xlabel("voxel potential (V)")
    ax.set_ylabel("voxels")
    ax.legend(fontsize=9)
    ax.set_title(f"rendered film, {VOXEL_SIZE:g} A/voxel", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-carbon-calibration.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_hero()
    figure_hole_geometry()
    figure_roughness()
    figure_slices()
    figure_calibration()
