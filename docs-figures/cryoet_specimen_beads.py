"""
Generate the figures for docs/concepts/cryoet-specimen/beads.md, which
explains the gold fiducial bead component of `specter build tomogram`
(``specter.specimen._grid``).

Calls the same private helpers the real code path uses (`_BeadShape`,
`BeadGenerator.generate`, `_mean_inner_potential`) rather than
reimplementing any of the physics, so the figures cannot silently drift
from what's actually shipped.

Run with: uv run python docs-figures/cryoet_specimen_beads.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _render import TEAL, isosurface_axes
from specter.specimen._grid import (
    GOLD_DENSITY_G_CM3,
    GOLD_MOLAR_MASS,
    _BeadShape,
    _mean_inner_potential,
    _number_density_per_a3,
    BeadGenerator,
)

OUT_DIR = "docs/assets/images"

GOLD = np.array([0.72, 0.53, 0.04])
RADIUS_ANGSTROM = 100.0
SEED = 3


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _shape_field(roughness: float, seed: int, n: int = 96) -> tuple[np.ndarray, float]:
    """Signed field for a `_BeadShape` boundary: negative inside, zero on
    the boundary -- the same `contains` test the atom fill and the
    segmentation mask both use, just evaluated as a continuous field so
    marching cubes can pull a surface out of it."""
    shape = _BeadShape(RADIUS_ANGSTROM, roughness, _generator(seed))
    spacing = 2.2 * shape.half_extent / (n - 1)
    idx = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * spacing
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    limit = shape.scale * (1.0 + shape.roughness * shape._field(points))
    phi = (points.norm(dim=-1) - limit).reshape(n, n, n)
    return phi.numpy(), spacing


def figure_hero() -> None:
    """Page-top hero: the irregular quasi-sphere boundary a single bead
    realisation actually gets, at the default roughness."""
    phi, spacing = _shape_field(0.12, SEED)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    isosurface_axes(ax, phi, spacing, GOLD)
    ax.view_init(elev=18, azim=35)
    plt.tight_layout(pad=0)
    path = f"{OUT_DIR}/cryoet-bead-hero.png"
    plt.savefig(path, dpi=250, transparent=True)
    plt.close(fig)
    print(f"saved {path}")


def figure_roughness_sweep() -> None:
    """roughness sweep at one fixed seed: 0 is an exact sphere, the default
    0.12 reads as a genuinely irregular colloidal particle, 0.24 is badly
    misshapen. Every panel is volume-matched to the same nominal radius."""
    values = [0.0, 0.06, 0.12, 0.24]
    fig, axes = plt.subplots(
        1,
        len(values),
        figsize=(3.0 * len(values), 3.2),
        subplot_kw={"projection": "3d"},
    )
    for ax, roughness in zip(axes, values):
        phi, spacing = _shape_field(roughness, SEED)
        isosurface_axes(ax, phi, spacing, GOLD)
        ax.view_init(elev=18, azim=35)
        ax.set_title(f"roughness = {roughness:.2f}", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-bead-roughness-sweep.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_lattice() -> None:
    """Central slice of the rendered potential at 1 A/voxel (fcc lattice
    fringes resolved, running in a direction set by this bead's own random
    crystal orientation) and at 5 A/voxel (a realistic tomogram voxel size,
    where the same bead reads as a solid slab of gold)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, voxel_size in zip(axes, [1.0, 5.0]):
        gen = BeadGenerator(voxel_size=voxel_size)
        bead = gen.generate(RADIUS_ANGSTROM, _generator(SEED))
        mid = bead.density.shape[0] // 2
        sl = bead.density[mid].numpy()
        im = ax.imshow(sl, cmap="gray_r", origin="lower")
        ax.set_title(
            f"{voxel_size:g} A/voxel  (peak {sl.max():.0f} V)",
            fontsize=11,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="potential (V)")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-bead-lattice.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_volume_match() -> None:
    """Two invariants worth trusting the generator on.

    Left: total integrated potential across roughness values -- the
    irregular boundary is volume-matched to the nominal sphere, so a lumpy
    bead carries the same projected signal as a round one of the same
    nominal radius. Right: the bead's mean interior potential against
    gold's bulk mean inner potential, computed independently from bulk
    mass density (and against the fcc lattice's own number density)."""
    roughness_values = [0.0, 0.06, 0.12, 0.18, 0.24]
    voxel_size = 2.0
    totals = []
    for roughness in roughness_values:
        gen = BeadGenerator(voxel_size=voxel_size, roughness=roughness)
        per_seed = []
        for seed in range(4):
            bead = gen.generate(RADIUS_ANGSTROM, _generator(seed))
            per_seed.append(float(bead.density.sum()) * voxel_size**3)
        totals.append(per_seed)
    totals_arr = np.array(totals)
    gen = BeadGenerator(voxel_size=voxel_size)
    nominal = (4 / 3) * np.pi * RADIUS_ANGSTROM**3 * gen.mean_inner_potential

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    for row, roughness in zip(totals_arr, roughness_values):
        ax.scatter([roughness] * len(row), row / nominal, color=TEAL, s=22)
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("roughness")
    ax.set_ylabel("integrated potential / nominal sphere")
    ax.set_ylim(0.9, 1.1)
    ax.set_title("volume matching (4 seeds each)", fontsize=11)

    ax = axes[1]
    number_density = _number_density_per_a3(GOLD_DENSITY_G_CM3, GOLD_MOLAR_MASS)
    fcc_density = 4 / 4.0782**3
    reference_bead = gen.generate(RADIUS_ANGSTROM, _generator(SEED))
    bars = {
        "bulk mass\ndensity route": gen.mean_inner_potential,
        "fcc lattice\nnumber density": fcc_density
        * _mean_inner_potential(voxel_size, 1.0, 79, "kirkland"),
        "rendered bead\ninterior mean": float(
            reference_bead.density[reference_bead.mask].mean()
        ),
    }
    ax.bar(list(bars), list(bars.values()), color=GOLD, width=0.55)
    for i, v in enumerate(bars.values()):
        ax.text(i, v + 0.6, f"{v:.1f} V", ha="center", fontsize=10)
    ax.set_ylabel("mean inner potential (V)")
    ax.set_ylim(0, 36)
    ax.set_title(f"gold MIP, {number_density:.4f} atoms/A^3", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-bead-calibration.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_hero()
    figure_roughness_sweep()
    figure_lattice()
    figure_volume_match()
