"""
Generate the figures for docs/concepts/cryoet-specimen/microtubules.md,
which explains the microtubule component of `specter build tomogram`
(``specter.specimen.filament._lattice``/``_tube``/``_tubulin``).

Every figure comes from the shipped code path -- `solve_tube_lattice`,
`place_microtubules` and `TomogramSpecimenGenerator` itself -- so the
figures cannot drift from the implementation. The measured numbers
annotated on them are measured off the rendered volume at plot time, not
copied in.

Run with: uv run python docs-figures/cryoet_specimen_microtubules.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specter.specimen.filament import (
    DIMER_REPEAT,
    MONOMER_RISE,
    MicrotubuleSpec,
    measure_source_lattice,
    microtubule_axis_path,
    solve_tube_lattice,
    thermal_flex_deg,
)
from specter.specimen.tomogram import TomogramSpecimenGenerator

OUT_DIR = "docs/assets/images"
VOXEL_SIZE = 8.0
ALPHA_COLOR = "#4C72B0"
BETA_COLOR = "#DD8452"
SEAM_COLOR = "#C44E52"


def _render(
    spec: MicrotubuleSpec,
    shape: tuple[int, int, int] = (64, 224, 224),
    seed: int = 0,
) -> tuple[torch.Tensor, TomogramSpecimenGenerator]:
    generator = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=shape,
        voxel_size=VOXEL_SIZE,
        protein_specs=[],
        microtubule_specs=[spec],
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        progressbars=False,
    )
    return generator.generate().cpu(), generator


def _radial_profile(
    volume: torch.Tensor, axis_xyz: torch.Tensor, r_max: float = 200.0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean density vs distance from the tube axis."""
    nz, ny, nx = volume.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    points = (
        torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], 1).float() * VOXEL_SIZE
    )
    radius = torch.cdist(points, axis_xyz).min(dim=1).values
    density = volume.flatten()

    edges = torch.arange(0.0, r_max, 4.0)
    index = torch.bucketize(radius, edges)
    total = torch.zeros(len(edges) + 1).scatter_add_(0, index, density)
    count = torch.zeros(len(edges) + 1).scatter_add_(0, index, torch.ones_like(density))
    profile = (total / count.clamp_min(1))[1 : len(edges)]
    return (edges[:-1] + 2.0).numpy(), profile.numpy()


def _cross_section(
    volume: torch.Tensor,
    axis_xyz: torch.Tensor,
    half_width: float = 190.0,
    depth: float = DIMER_REPEAT,
    n: int = 121,
) -> np.ndarray:
    """
    Resample the volume on the plane **perpendicular to the tube axis**.

    A plain z-slice would only be a cross-section for a tube running along
    z; a microtubule confined to thin ice lies mostly in-plane, so slicing
    along z cuts lengthwise and the result is not a ring at all. Sample on
    the local parallel-transport frame instead, averaging over one dimer
    repeat along the axis so the protofilaments read clearly.
    """
    from scipy.ndimage import map_coordinates

    from specter.specimen.filament import parallel_transport_frames

    mid_index = len(axis_xyz) // 2
    frame = parallel_transport_frames(axis_xyz)[mid_index]
    normal, binormal, tangent = frame[:, 0], frame[:, 1], frame[:, 2]
    centre = axis_xyz[mid_index]

    grid = torch.linspace(-half_width, half_width, n)
    uu, vv = torch.meshgrid(grid, grid, indexing="ij")
    depths = torch.linspace(-depth / 2, depth / 2, 9)

    accumulated = np.zeros((n, n))
    for w in depths:
        points = (
            centre
            + uu.unsqueeze(-1) * normal
            + vv.unsqueeze(-1) * binormal
            + w * tangent
        ) / VOXEL_SIZE
        # map_coordinates indexes (z, y, x); points are (x, y, z).
        coords = np.stack(
            [
                points[..., 2].numpy(),
                points[..., 1].numpy(),
                points[..., 0].numpy(),
            ]
        )
        accumulated += map_coordinates(volume.numpy(), coords, order=1, cval=0.0)
    return accumulated / len(depths)


# --------------------------------------------------------------------------
def fig_hero() -> None:
    """A rendered microtubule: projection plus a true cross-section."""
    spec = MicrotubuleSpec(n_copies=1, length=1400.0, confine_to_slab=True)
    volume, generator = _render(spec, shape=(48, 224, 224))
    tube = generator.microtubule_instances[0]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    projection = volume.sum(dim=0).numpy()
    ax0.imshow(
        projection,
        cmap="gray_r",
        origin="lower",
        extent=[
            0,
            projection.shape[1] * VOXEL_SIZE / 10,
            0,
            projection.shape[0] * VOXEL_SIZE / 10,
        ],
    )
    ax0.plot(
        tube.axis_xyz[:, 0].numpy() / 10,
        tube.axis_xyz[:, 1].numpy() / 10,
        color=SEAM_COLOR,
        lw=0.9,
        ls="--",
        label="tube axis",
    )
    ax0.set(xlabel="x (nm)", ylabel="y (nm)", title="projection along z")
    ax0.legend(fontsize=8, loc="upper right")

    half_width = 190.0
    cross = _cross_section(volume, tube.axis_xyz, half_width=half_width)
    extent = np.array([-1, 1, -1, 1]) * half_width / 10
    ax1.imshow(cross.T, cmap="gray_r", origin="lower", extent=extent)
    circle = plt.Circle(
        (0, 0), tube.lattice.radius / 10, fill=False, color=SEAM_COLOR, lw=1.0, ls="--"
    )
    ax1.add_patch(circle)
    ax1.set(
        xlabel="x (nm)",
        ylabel="y (nm)",
        title=f"cross-section — {tube.lattice.n_protofilaments} protofilaments "
        "around an empty lumen",
    )

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cryoet-microtubule-hero.png", dpi=150)
    plt.close(fig)


def fig_wall_profile() -> None:
    """Radial density of the rendered tube against the real microtubule."""
    spec = MicrotubuleSpec(n_copies=1, length=1400.0)
    volume, generator = _render(spec, shape=(48, 224, 224))
    tube = generator.microtubule_instances[0]
    radii, profile = _radial_profile(volume, tube.axis_xyz)

    peak = profile.max()
    above = radii[profile > 0.1 * peak]
    inner, outer = float(above.min()), float(above.max())
    half_max = radii[profile > 0.5 * peak]
    fwhm_inner, fwhm_outer = float(half_max.min()), float(half_max.max())

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(radii, profile, color=ALPHA_COLOR, lw=1.8)
    ax.axvline(
        tube.lattice.radius,
        color=SEAM_COLOR,
        ls="--",
        lw=1.2,
        label=f"model pf radius {tube.lattice.radius:.1f} Å",
    )
    ax.axvspan(0, inner, color="0.85", alpha=0.6)
    ax.text(
        inner / 2, peak * 0.55, "lumen\n(empty)", ha="center", fontsize=9, color="0.35"
    )
    ax.annotate(
        "",
        xy=(inner, peak * 1.06),
        xytext=(outer, peak * 1.06),
        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.3),
    )
    ax.text(
        (inner + outer) / 2,
        peak * 1.09,
        f"wall {outer - inner:.0f} Å",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    ax.axvspan(
        fwhm_inner,
        fwhm_outer,
        color=BETA_COLOR,
        alpha=0.15,
        label=f"FWHM {2 * fwhm_inner:.0f}–{2 * fwhm_outer:.0f} Å (Ø)",
    )
    ax.set(
        xlabel="distance from tube axis (Å)",
        ylabel="mean rendered density",
        title=f"rendered: Ø {2 * fwhm_inner:.0f}–{2 * fwhm_outer:.0f} Å at half "
        f"maximum   |   real microtubule: ~170 Å lumen, ~250 Å outer",
        xlim=(0, 200),
        ylim=(0, peak * 1.2),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cryoet-microtubule-wall.png", dpi=150)
    plt.close(fig)
    print(
        f"  wall profile: peak at {radii[profile.argmax()]:.0f} A, "
        f"outer D {2 * outer:.0f} A, lumen D {2 * inner:.0f} A"
    )


def fig_lattice_and_seam() -> None:
    """Unrolled lattice: the stagger, and the one junction that doesn't fit."""
    lattice = solve_tube_lattice(13)
    n_pf = lattice.n_protofilaments
    spacing = 2 * np.pi * lattice.radius / n_pf

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    phi = 2 * np.pi * np.arange(n_pf) / n_pf
    x, y = lattice.radius * np.cos(phi), lattice.radius * np.sin(phi)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax0.plot(
        lattice.radius * np.cos(theta),
        lattice.radius * np.sin(theta),
        ls="--",
        lw=1.0,
        color="0.6",
    )
    ax0.scatter(x, y, s=300, color=ALPHA_COLOR, edgecolor="k", lw=0.6, zorder=3)
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax0.annotate(
            str(i),
            (xi, yi),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            zorder=4,
        )
    ax0.scatter(
        [x[0]], [y[0]], s=300, facecolor="none", edgecolor=SEAM_COLOR, lw=2.5, zorder=5
    )
    ax0.set_aspect("equal")
    ax0.set(
        xlabel="x (Å)",
        ylabel="y (Å)",
        title=f"{n_pf} protofilaments, R = {lattice.radius:.1f} Å",
    )

    def draw(px: float, z0: float, **kwargs) -> None:
        for m in range(10):
            z = z0 + m * MONOMER_RISE
            if not -20 < z < 400:
                continue
            is_alpha = m % 2 == 0
            ax1.scatter(
                px,
                z,
                s=150,
                marker="o" if is_alpha else "s",
                color=ALPHA_COLOR if is_alpha else BETA_COLOR,
                zorder=3,
                **kwargs,
            )

    for p in range(n_pf):
        draw(p * spacing, p * lattice.stagger, edgecolor="0.25", linewidth=0.6)
    wrap = n_pf * spacing
    draw(wrap - 9, n_pf * lattice.stagger, edgecolor=SEAM_COLOR, linewidth=2.2)
    draw(wrap + 9, 0.0, edgecolor="0.25", linewidth=0.6, alpha=0.4)

    ax1.axvline((n_pf - 0.5) * spacing, color=SEAM_COLOR, ls="--", lw=1.4)
    ax1.annotate(
        "",
        xy=(wrap + 26, 246),
        xytext=(wrap + 26, 246 + MONOMER_RISE),
        arrowprops=dict(arrowstyle="<->", color=SEAM_COLOR, lw=1.6),
    )
    ax1.text(
        wrap + 32,
        248,
        f"{lattice.seam_offset:.0f} Å = 1 monomer\nα meets β → A-lattice SEAM",
        color=SEAM_COLOR,
        fontsize=8.5,
        va="bottom",
    )
    ax1.scatter([], [], marker="o", color=ALPHA_COLOR, label="α")
    ax1.scatter([], [], marker="s", color=BETA_COLOR, label="β")
    ax1.plot(
        [],
        [],
        "o",
        mfc="none",
        mec=SEAM_COLOR,
        mew=2,
        label="pf 0 predicted by 13 lateral bonds",
    )
    ax1.plot([], [], "o", color="0.6", label="pf 0 actual register")
    ax1.set(
        xlabel="arc position around the tube (Å)",
        ylabel="axial position z (Å)",
        title=f"unrolled lattice — stagger {lattice.stagger:.2f} Å per pf",
        ylim=(-20, 400),
        xlim=(-40, wrap + 150),
    )
    ax1.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cryoet-microtubule-lattice.png", dpi=150)
    plt.close(fig)


def fig_protofilament_number() -> None:
    """Rendered cross-sections at three protofilament numbers, plus the
    model checked against the structures its constants came from."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))

    for ax, n_pf in zip(axes[:3], (11, 13, 14)):
        spec = MicrotubuleSpec(n_copies=1, n_protofilaments=n_pf, length=700.0)
        volume, generator = _render(spec, shape=(48, 160, 160), seed=1)
        tube = generator.microtubule_instances[0]
        half_width = 190.0
        cross = _cross_section(volume, tube.axis_xyz, half_width=half_width)
        extent = np.array([-1, 1, -1, 1]) * half_width
        ax.imshow(cross.T, cmap="gray_r", origin="lower", extent=extent)
        ax.set_title(
            f"{n_pf} protofilaments\nR = {tube.lattice.radius:.0f} Å", fontsize=10
        )
        ax.set_xlabel("x (Å)")
        ax.set_xticks([-150, 0, 150])
        ax.set_yticks([-150, 0, 150])

    ax = axes[3]
    ns = list(range(10, 17))
    ax.plot(
        ns,
        [solve_tube_lattice(n).radius for n in ns],
        "o-",
        color=ALPHA_COLOR,
        label="model  R = N·a/2π",
    )
    for source, n_pf in (("3JAL", 13), ("6DPU", 14)):
        try:
            measured = measure_source_lattice(source)
        except Exception as exc:  # offline: skip the overlay, keep the curve
            print(f"  (skipping {source} overlay: {exc})")
            continue
        ax.scatter([n_pf], [measured["radius"]], s=140, color=SEAM_COLOR, zorder=4)
        ax.annotate(
            source,
            (n_pf, measured["radius"]),
            fontsize=9,
            color=SEAM_COLOR,
            textcoords="offset points",
            xytext=(8, -12),
        )
    ax.scatter([], [], s=140, color=SEAM_COLOR, label="measured (deposited)")
    ax.set(
        xlabel="protofilaments N",
        ylabel="radius of pf centres (Å)",
        title="radius vs protofilament number",
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cryoet-microtubule-protofilaments.png", dpi=150)
    plt.close(fig)


def fig_bending() -> None:
    """Thermal stiffness vs a mechanical bend, both at true scale."""
    lattice = solve_tube_lattice(13)
    diameter = 2 * lattice.radius + 44.0
    n_rings = 74  # ~600 nm

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.0))
    flex = thermal_flex_deg()

    for ax, (spec, label, colour) in zip(
        axes,
        [
            (
                MicrotubuleSpec(length=6000.0, confine_to_slab=False),
                f"thermal walk — derived flex {flex:.2f}° "
                "(L_p = 1 mm): essentially straight",
                ALPHA_COLOR,
            ),
            (
                MicrotubuleSpec(
                    length=6000.0, bend_radius=3.0e4, confine_to_slab=False
                ),
                "bend_radius = 3 µm — the smooth curvature of a "
                "mechanically constrained microtubule",
                BETA_COLOR,
            ),
        ],
    ):
        axis = microtubule_axis_path(
            spec,
            n_rings,
            torch.zeros(3),
            torch.tensor([1.0, 0.0, 0.0]),
            generator=torch.Generator().manual_seed(3),
        )
        path = axis - axis[0]
        chord = path[-1] / path[-1].norm()
        up = torch.tensor([0.0, 0.0, 1.0])
        if abs(float(up @ chord)) > 0.9:
            up = torch.tensor([0.0, 1.0, 0.0])
        e2 = torch.linalg.cross(chord, up)
        e2 = e2 / e2.norm()
        along = (path @ chord).numpy() / 10
        across = (path @ torch.linalg.cross(e2, chord)).numpy()

        ax.fill_between(
            along, across - diameter / 2, across + diameter / 2, color=colour, alpha=0.3
        )
        ax.plot(along, across, color=colour, lw=1.3)
        ax.set(ylabel="transverse (Å)", title=label, xlim=(-5, 615))
        ax.set_aspect(1 / 10)  # x in nm, y in A -> equal physical scale
        ax.title.set_size(9)

    axes[1].set_xlabel("distance along the microtubule (nm)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cryoet-microtubule-bending.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("rendering microtubule figures...")
    fig_lattice_and_seam()
    print("  lattice + seam")
    fig_hero()
    print("  hero")
    fig_wall_profile()
    fig_protofilament_number()
    print("  protofilament sweep")
    fig_bending()
    print("  bending")
    print(f"done -> {OUT_DIR}/")
