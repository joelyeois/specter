"""
Generate the figures for docs/concepts/specimen-deformation.md, which
explains `settings.Deformation`: the beam-induced bending of a tilt
series' specimen.

Draws the deflections through the same `deflection_field` the tilt series
uses, and the tilt images through `Sample.tilt_series`, so the figures
cannot drift from what ships.

Run with: uv run python docs-figures/specimen_deformation.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specter.imaging import Exposure, Poses, Sample
from specter.imaging._deformation import (
    RigidBodies,
    burst_fraction,
    deflection_field,
)
from specter.settings import Deformation, Propagation

OUT_DIR = "docs/assets/images"
ACCENT = "#c4572e"
BLUE = "#4c72b0"


def shapes() -> None:
    """The two deflections, over a field of view and in cross-section."""
    fov = 6000.0  # A, the field a tilt series images
    dome = Deformation(
        model="dome", height=150.0, hole_radius=6000.0, hole_offset=(-4000.0, 0.0)
    )
    random = Deformation(model="random", height=40.0, correlation_length=10000.0)
    grid = torch.linspace(-fov / 2, fov / 2, 241, dtype=torch.float64)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    fig, axs = plt.subplots(
        1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.6, 1, 1]}
    )

    hole = torch.linspace(-12000.0, 12000.0, 601, dtype=torch.float64)
    field = deflection_field(dome, 12000.0, 50.0)
    assert field is not None
    w = field(hole, torch.zeros_like(hole))[0]
    ax = axs[0]
    for z, label in [(1000.0, "top face"), (-1000.0, "bottom face")]:
        # The faces of a 200 nm film, each displaced as the plate kinematics say.
        _, w_x, _ = field(hole, torch.zeros_like(hole))
        ax.plot((hole - z * w_x) / 10, (z + w) / 10, color=BLUE, lw=1.2)
        ax.text(-1180, z / 10 + 8, label, fontsize=9, color=BLUE)
    ax.plot(hole / 10, w / 10, color=ACCENT, lw=1.2, ls="--", label="midplane")
    ax.axvspan(-fov / 20, fov / 20, color="0.9", zorder=0, label="field of view")
    for rim in (-1000, 200):  # the hole's rim, its centre 400 nm to the left
        ax.axvline(rim, color="0.4", lw=0.8)
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-130, 150)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(
        "Dome over a 1.2 µm hole in a 200 nm film\n(h = 150 Å; z stretched 4x)"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_aspect(4.0)

    extent = [-fov / 20, fov / 20, -fov / 20, fov / 20]
    for ax, deformation, title in [
        (axs[1], dome, "Dome, field 400 nm from the hole's centre"),
        (axs[2], random, "Random, RMS 40 Å, correlation length 1 µm"),
    ]:
        field = deflection_field(
            deformation, fov / 2, 25.0, torch.Generator().manual_seed(4)
        )
        assert field is not None
        w = field(gx, gy)[0].numpy()
        limit = float(np.abs(w).max())
        image = ax.imshow(
            w, origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        ax.contour(
            w, levels=8, colors="k", linewidths=0.4, origin="lower", extent=extent
        )
        plt.colorbar(image, ax=ax, fraction=0.046, label="w (Å)")
        ax.set_title(title)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/specimen-deformation-shapes.png", dpi=130)
    plt.close(fig)


def burst() -> None:
    """How much of the bend each tilt of a dose-symmetric-length series sees."""
    dose_per_tilt, n_tilts = 3.0, 41
    doses = np.linspace(0.0, dose_per_tilt * n_tilts, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for burst_dose, color in [(2.0, ACCENT), (10.0, BLUE)]:
        ax.plot(
            doses,
            [burst_fraction(d, burst_dose) for d in doses],
            color=color,
            lw=1.4,
            label=f"burst_dose = {burst_dose:g} e⁻/Å²",
        )
        mid = dose_per_tilt * np.arange(n_tilts) + dose_per_tilt / 2
        ax.plot(
            mid,
            [burst_fraction(d, burst_dose) for d in mid],
            "o",
            color=color,
            ms=3,
        )
    ax.set_xlabel("accumulated exposure (e⁻/Å²)")
    ax.set_ylabel("fraction of the full bend")
    ax.set_title(
        f"Each tilt sees the bend at its mid-exposure (dots: {dose_per_tilt:g} e⁻/Å² per tilt)"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/specimen-deformation-burst.png", dpi=130)
    plt.close(fig)


PIX, NXY, NZ = 10.0, 320, 40
ANGLES = [0.0, 30.0, 60.0]
SPACING, LAYER = 48, 150.0


def _lattice() -> torch.Tensor:
    """Two layers of Gaussian beads, 150 A above and below the midplane."""
    axes = [(torch.arange(n) - n // 2) * PIX for n in (NZ, NXY, NXY)]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    volume = torch.zeros(NZ, NXY, NXY)
    centres = np.arange(-3, 3) * SPACING * PIX + SPACING * PIX / 4
    for zc, shift in [(LAYER, 0.0), (-LAYER, SPACING * PIX / 2)]:
        for cx in centres:
            for cy in centres:
                r2 = (x - cx - shift) ** 2 + (y - cy - shift) ** 2 + (z - zc) ** 2
                volume += torch.exp(-r2 / (2 * (1.5 * PIX) ** 2))
    return volume[None]


def _images(deformation: Deformation) -> list[torch.Tensor]:
    series = Sample.from_volume(_lattice(), PIX).tilt_series(
        Poses.tilt_series(ANGLES),
        Exposure.tilt_series(3.0, len(ANGLES)),
        voltage=300.0,
        micrograph_size=NXY,
        deformation=deformation,
        pad_volume=False,
        log=False,
        propagation=Propagation(
            scattering_model="projection", alpha=0.0, oversample=1, pad_fft=False
        ),
    )
    return [
        torch.angle(series.scatter(i).pixels()[0]).double() for i in range(len(ANGLES))
    ]


def _peaks(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = torch.nn.functional.max_pool2d(image[None, None], 7, 1, 3)[0, 0]
    ys, xs = torch.nonzero(
        (image == pooled) & (image > 0.3 * float(image.max())), as_tuple=True
    )
    keep = (ys > 8) & (ys < NXY - 9) & (xs > 8) & (xs < NXY - 9)
    return ys[keep], xs[keep]


def _centroid(image: torch.Tensor, y: int, x: int, r: int = 4) -> tuple[float, float]:
    window = image[y - r : y + r + 1, x - r : x + r + 1].clamp(min=0)
    yy, xx = torch.meshgrid(
        torch.arange(y - r, y + r + 1), torch.arange(x - r, x + r + 1), indexing="ij"
    )
    return (
        float((window * xx).sum() / window.sum()),
        float((window * yy).sum() / window.sum()),
    )


def tilts() -> None:
    """Simulated tilts of beads on a dome's flank, against a rigid specimen."""
    dome = Deformation(
        model="dome",
        height=150.0,
        hole_radius=6000.0,
        hole_offset=(-4000.0, 0.0),
        burst_dose=1e-3,
    )
    rigid, bent = _images(Deformation()), _images(dome)
    extent = np.array([-NXY / 2, NXY / 2, NXY / 2, -NXY / 2]) * PIX / 10
    scale = 3.0
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, angle, r_img, b_img in zip(axs, ANGLES, rigid, bent, strict=True):
        ax.imshow(b_img.numpy(), cmap="gray_r", extent=extent)
        ys, xs = _peaks(r_img)
        bys, bxs = _peaks(b_img)
        shifts = []
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
            d2 = (bys - y) ** 2 + (bxs - x) ** 2
            j = int(d2.argmin())
            if float(d2[j]) > 14**2:
                continue
            cr = _centroid(r_img, y, x)
            cb = _centroid(b_img, int(bys[j]), int(bxs[j]))
            dx, dy = cb[0] - cr[0], cb[1] - cr[1]
            shifts.append(math.hypot(dx, dy) * PIX)
            px, py = (cr[0] - NXY / 2) * PIX / 10, (cr[1] - NXY / 2) * PIX / 10
            ax.annotate(
                "",
                xy=(px + dx * scale * PIX / 10, py + dy * scale * PIX / 10),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4),
            )
        ax.set_title(
            f"tilt {angle:+.0f}°: median {np.median(shifts):.1f} Å, "
            f"max {np.max(shifts):.1f} Å"
        )
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        print(
            f"tilt {angle:+.0f}: median {np.median(shifts):.1f} A, max {np.max(shifts):.1f} A"
        )
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/specimen-deformation-tilts.png", dpi=130)
    plt.close(fig)


R_PIX, R_NXY, R_NZ = 4.0, 128, 250
R_HEIGHT, R_RADIUS, R_PITCH = 300.0, 180.0, 60.0


def _particle() -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """A ball of beads on a cubic lattice, 300 A above the midplane, and its label."""
    axes = [(torch.arange(n) - n // 2) * R_PIX for n in (R_NZ, R_NXY, R_NXY)]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    volume = torch.zeros(R_NZ, R_NXY, R_NXY)
    nodes = np.arange(-3, 4) * R_PITCH
    columns = []
    for cx in nodes:
        for cy in nodes:
            for cz in nodes:
                if cx**2 + cy**2 + cz**2 > R_RADIUS**2:
                    continue
                r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - R_HEIGHT - cz) ** 2
                volume += torch.exp(-r2 / (2 * 7.0**2))
            if cx**2 + cy**2 <= R_RADIUS**2:
                columns.append((cx, cy))
    ball = x**2 + y**2 + (z - R_HEIGHT) ** 2 <= (R_RADIUS + 20.0) ** 2
    return volume[None], ball.numpy().astype(np.int32), np.array(columns)


def rigid() -> None:
    """One particle at a dome's apex, strained with the film and carried rigidly."""
    volume, labels, columns = _particle()
    dome = Deformation(model="dome", height=23.0, hole_radius=800.0, burst_dose=1e-3)
    images = []
    for bodies in (None, RigidBodies.from_labels(labels, R_PIX)):
        series = Sample.from_volume(volume, R_PIX).tilt_series(
            Poses.tilt_series([0.0]),
            Exposure.tilt_series(3.0, 1),
            voltage=300.0,
            micrograph_size=R_NXY,
            deformation=dome,
            rigid_bodies=bodies,
            pad_volume=False,
            log=False,
            propagation=Propagation(
                scattering_model="projection", alpha=0.0, oversample=1, pad_fft=False
            ),
        )
        images.append(torch.angle(series.scatter(0).pixels()[0]).double())
    extent = np.array([-R_NXY / 2, R_NXY / 2, R_NXY / 2, -R_NXY / 2]) * R_PIX / 10
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.2))
    titles = ("strained with the film", "carried as a rigid body")
    for ax, image, title in zip(axs, images, titles, strict=True):
        ax.imshow(image.numpy(), cmap="gray_r", extent=extent)
        ax.scatter(
            columns[:, 0] / 10,
            columns[:, 1] / 10,
            s=260,
            facecolors="none",
            edgecolors=ACCENT,
            linewidths=1.2,
        )
        moved = []
        for cx, cy in columns:
            px, py = (
                int(round(cx / R_PIX)) + R_NXY // 2,
                int(round(cy / R_PIX)) + R_NXY // 2,
            )
            window = image[py - 6 : py + 7, px - 6 : px + 7].clamp(min=0)
            yy, xx = torch.meshgrid(
                torch.arange(py - 6, py + 7),
                torch.arange(px - 6, px + 7),
                indexing="ij",
            )
            mx = float((window * xx).sum() / window.sum()) - R_NXY // 2
            my = float((window * yy).sum() / window.sum()) - R_NXY // 2
            moved.append(math.hypot(mx * R_PIX - cx, my * R_PIX - cy))
        ax.set_title(f"{title}: beads off by up to {max(moved):.1f} Å")
        ax.set_xlim(-23, 23)
        ax.set_ylim(23, -23)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        print(f"{title}: max {max(moved):.2f} A, median {np.median(moved):.2f} A")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/specimen-deformation-rigid.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    shapes()
    burst()
    tilts()
    rigid()
