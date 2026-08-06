"""
Generate the step-by-step figures for
docs/concepts/membrane-shape/swept-spline.md, which explains the
``swept_spline`` membrane shape backend
(``specter.specimen.membrane._field_swept_spline``).

Calls the same private helpers the real code path uses
(`_sample_wandering_path`, `blend_field`, `cap_curvature`) rather than
reimplementing the math independently, so the figures cannot silently
drift from what's actually shipped.

Run with: uv run python docs-figures/membrane_shape_swept_spline.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from _render import TEAL, isosurface_axes
from specter.specimen.membrane._field import (
    MetaballSource,
    _grid_points_xyz,
    blend_field,
    cap_curvature,
)
from specter.specimen.membrane._field_swept_spline import _sample_wandering_path
from specter.specimen.membrane._generator import MembraneGenerator

OUT_DIR = "docs/assets/images"

# Reference instance, matching generate_membrane_field_swept_spline's own
# defaults, plus a fixed seed chosen for a clearly wandering (non-straight),
# non-self-touching path.
SHAPE_ZYX = (170, 170, 170)
SPACING_A = 3.5
TOTAL_LENGTH_A = 500.0
STEP_LENGTH_A = 15.0
TUBE_RADIUS_A = 25.0
FLEXIBILITY = 0.15
BLEND_SHARPNESS_A = 0.5 * TUBE_RADIUS_A
PATH_SIGMA_POINTS = 1.5
CURVATURE_ITERATIONS = 15
CURVATURE_STEP_FRACTION = 0.15
SEED = 1


def _reference_instance() -> dict:
    """Reproduce generate_membrane_field_swept_spline's own steps, keeping
    the raw and processed path arrays the public function doesn't expose."""
    rng = np.random.default_rng(SEED)
    n_points = max(2, round(TOTAL_LENGTH_A / STEP_LENGTH_A) + 1)
    raw = _sample_wandering_path(n_points, STEP_LENGTH_A, FLEXIBILITY, rng)

    bbox_mid = 0.5 * (raw.min(axis=0) + raw.max(axis=0))
    centered = raw - bbox_mid
    smoothed = ndimage.gaussian_filter1d(
        centered, sigma=PATH_SIGMA_POINTS, axis=0, mode="nearest"
    )

    positions_t = torch.as_tensor(smoothed, dtype=torch.float32)
    sources = [
        MetaballSource(center_xyz=positions_t[i], radius=TUBE_RADIUS_A)
        for i in range(n_points)
    ]

    extent_a = (
        torch.tensor([SHAPE_ZYX[2], SHAPE_ZYX[1], SHAPE_ZYX[0]], dtype=torch.float32)
        * SPACING_A
    )
    origin_xyz = -0.5 * extent_a
    points_xyz = _grid_points_xyz(SHAPE_ZYX, SPACING_A, origin_xyz, device="cpu")
    phi_raw = blend_field(sources, points_xyz, BLEND_SHARPNESS_A)
    phi_capped = cap_curvature(
        phi_raw, SPACING_A, CURVATURE_ITERATIONS, CURVATURE_STEP_FRACTION
    )

    return dict(
        raw=raw,
        centered=centered,
        smoothed=smoothed,
        phi_capped=phi_capped.numpy(),
    )


def figure_hero(ref: dict) -> None:
    """Page-top hero: shaded 3D isosurface of the reference tube. No EDT
    involved here (blend_field is already an analytic smooth-min of exact
    sphere SDFs), so this is smooth at ordinary working-grid resolution --
    unlike the spherical-harmonics backend's hero shot, no extra-fine grid
    is needed."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    isosurface_axes(ax, ref["phi_capped"], SPACING_A, TEAL)
    ax.view_init(elev=15, azim=25)
    plt.tight_layout(pad=0)
    path = f"{OUT_DIR}/membrane-swept-hero.png"
    plt.savefig(path, dpi=250, transparent=True)
    plt.close(fig)
    print(f"saved {path}")


def figure_path(ref: dict) -> None:
    """Raw persistent random walk vs. the recentered + path-order-smoothed
    version actually used to place sphere sources."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), subplot_kw={"projection": "3d"})
    for ax, pts, title in [
        (axes[0], ref["centered"], "raw random walk"),
        (axes[1], ref["smoothed"], "recentered + smoothed"),
    ]:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=TEAL, linewidth=1.5)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="k", s=15)
        ax.set_title(title, fontsize=11)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-swept-path.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def _straight_chain_phi(step_length_a: float, n_points: int = 8) -> np.ndarray:
    """Sphere centers evenly spaced along a straight line -- isolates the
    smooth-min union / beading behavior from the random walk's own shape."""
    centers = torch.zeros((n_points, 3), dtype=torch.float32)
    centers[:, 0] = torch.arange(n_points, dtype=torch.float32) * step_length_a
    sources = [
        MetaballSource(center_xyz=centers[i], radius=TUBE_RADIUS_A)
        for i in range(n_points)
    ]
    shape_zyx = (40, 60, 220)
    spacing_a = 2.0
    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = torch.tensor([-20.0, -0.5 * extent_a[1], -0.5 * extent_a[2]])
    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    phi = blend_field(sources, points_xyz, BLEND_SHARPNESS_A)
    return phi.numpy(), spacing_a, origin_xyz.numpy()


def figure_beading(ref: dict) -> None:
    """Longitudinal slice through a straight chain of sphere sources,
    smooth-min blended -- step_length_a well under tube_radius_a fuses into
    a continuous tube; step_length_a exceeding it visibly beads, the real
    failure mode generate_membrane_field_swept_spline's own step_length_a
    check warns about."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.4))
    for ax, step_length_a, title in [
        (
            axes[0],
            15.0,
            f"step_length_a=15 (< tube_radius_a={TUBE_RADIUS_A:.0f}): smooth",
        ),
        (
            axes[1],
            45.0,
            f"step_length_a=45 (> tube_radius_a={TUBE_RADIUS_A:.0f}): beaded",
        ),
    ]:
        phi, spacing_a, origin_xyz = _straight_chain_phi(step_length_a)
        z_mid = phi.shape[0] // 2
        sl = phi[z_mid] < 0
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-swept-beading.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_flexibility_sweep() -> None:
    """flexibility sweep: low values give long, gently-curving tubes; values
    near 1 give a tightly wandering, near-self-touching walk."""
    configs = [0.05, 0.15, 0.35]
    fig, axes = plt.subplots(1, len(configs), figsize=(3.4 * len(configs), 3.6))
    for ax, flexibility in zip(axes, configs):
        gen = MembraneGenerator(
            target_shape_zyx=SHAPE_ZYX,
            v_size=SPACING_A,
            shape_backend="swept_spline",
            swept_total_length_a=TOTAL_LENGTH_A,
            swept_step_length_a=STEP_LENGTH_A,
            swept_tube_radius_a=TUBE_RADIUS_A,
            swept_flexibility=flexibility,
            n_lipids_per_leaflet=1,
            seed=SEED,
        )
        gen.generate()
        mip = (gen.field.phi < 0).numpy().max(axis=0)
        ax.imshow(mip, cmap="gray", origin="lower")
        default_tag = "  (default)" if flexibility == FLEXIBILITY else ""
        ax.set_title(f"flexibility={flexibility:g}{default_tag}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-swept-flexibility-sweep.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_curvature_capping() -> None:
    """A synthetic tight semicircular bend (path radius of curvature only
    modestly larger than tube_radius_a) -- deliberately NOT the random-walk
    reference instance, so the bend's sharpness and orientation (all in one
    plane, chosen to align with a plotted slice) are fully controlled.

    Overlays the mid-surface (phi=0) contour before and after cap_curvature
    at the bend's concave inner corner. Verified numerically (not just
    assumed) before plotting: cap_curvature's Laplacian relaxation makes phi
    LESS negative (surface pulled outward, corner filled in / rounded) on
    the concave inner side and MORE negative (surface pulled inward) on the
    adjacent convex outer bulge -- both directions reduce local curvature,
    consistent with mean-curvature-flow-like smoothing, not just a
    plausible-looking difference."""
    arc_radius_a = 1.3 * TUBE_RADIUS_A
    n_points = 24
    theta = np.linspace(0.0, np.pi, n_points)
    positions = np.stack(
        [
            arc_radius_a * np.cos(theta),
            arc_radius_a * np.sin(theta),
            np.zeros_like(theta),
        ],
        axis=-1,
    )
    positions_t = torch.as_tensor(positions, dtype=torch.float32)
    sources = [
        MetaballSource(center_xyz=positions_t[i], radius=TUBE_RADIUS_A)
        for i in range(n_points)
    ]

    shape_zyx = (24, 130, 130)
    spacing_a = 1.5
    extent_a = (
        torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=torch.float32)
        * spacing_a
    )
    origin_xyz = -0.5 * extent_a
    points_xyz = _grid_points_xyz(shape_zyx, spacing_a, origin_xyz, device="cpu")
    phi_raw = blend_field(sources, points_xyz, BLEND_SHARPNESS_A).numpy()
    phi_capped = cap_curvature(
        torch.as_tensor(phi_raw),
        spacing_a,
        CURVATURE_ITERATIONS,
        CURVATURE_STEP_FRACTION,
    ).numpy()

    z_mid = shape_zyx[0] // 2
    half = 0.5 * shape_zyx[1] * spacing_a
    extent = [-half, half, -half, half]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(
        phi_capped[z_mid] < 0, cmap="Greys", alpha=0.35, origin="lower", extent=extent
    )
    ax.contour(
        phi_raw[z_mid],
        levels=[0.0],
        colors="0.3",
        linestyles="--",
        linewidths=1.8,
        extent=extent,
        origin="lower",
    )
    ax.contour(
        phi_capped[z_mid],
        levels=[0.0],
        colors=[TEAL],
        linewidths=1.8,
        extent=extent,
        origin="lower",
    )
    ax.plot([], [], color="0.3", linestyle="--", label="before cap_curvature")
    ax.plot([], [], color=TEAL, label="after cap_curvature")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xlim(-25, 45)
    ax.set_ylim(-15, 45)
    ax.axis("off")
    ax.set_title("Mid-surface at a tight concave bend", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-swept-curvature-capping.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    ref = _reference_instance()
    figure_hero(ref)
    figure_path(ref)
    figure_beading(ref)
    figure_flexibility_sweep()
    figure_curvature_capping()
    print("done")
