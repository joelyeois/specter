"""
Generate the figures for docs/concepts/membrane-shape/ (index.md,
geometry.md, organelles.md): how `specter.specimen.membrane` builds a
membrane as the boundary of a solid.

Every panel is drawn from the shipped code: `MembraneGenerator` for the four
organelle kinds, and `Grid`, the primitives, `regularise`, `reach` and
`render_bilayer` for the geometry and composition figures. The numbers the
pages quote (reach before and after regularisation, node counts, build
times) are the ones this script prints.

Run with: uv run python docs-figures/membrane_shape.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import math
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _render import TEAL, isosurface_axes
from specter.specimen.membrane import (
    ORGANELLE_DEFAULTS,
    Grid,
    MembraneGenerator,
    reach,
    regularise,
    render_bilayer,
)
from specter.specimen.membrane._geometry import MIN_REACH, redistance
from specter.specimen.membrane._primitives import (
    ball,
    half_space,
    nested,
    rot_to,
    sheet,
    subtract,
    tube,
    undulate,
    union,
    wobbly,
)
from specter.specimen.membrane._profile import build_measured_bilayer_profile

OUT_DIR = "docs/assets/images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 3
VOXEL_SIZE = 10.0
ORANGE = "#d98218"


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize(torch.device(DEVICE))


def figure_kinds() -> None:
    """The four organelle kinds at their default settings: the outermost
    membrane's mid-surface above, a central slice of the rendered potential
    below."""
    # The plane each kind is cut in: the cisterna is a sheet lying in xy,
    # so only a section along z shows its two faces.
    cut_axis = {"vesicle": 0, "tubules": 0, "cisterna": 1, "mitochondrion": 0}
    fig = plt.figure(figsize=(13.0, 6.6))
    for col, kind in enumerate(ORGANELLE_DEFAULTS):
        _sync()
        start = time.perf_counter()
        gen = MembraneGenerator(
            voxel_size=VOXEL_SIZE, kind=kind, device=DEVICE, seed=SEED
        )
        volume = gen.generate()
        _sync()
        elapsed = time.perf_counter() - start
        volume = volume.cpu()
        params = ", ".join(f"{k}={v:g}" for k, v in gen.params.items())
        print(
            f"{kind}: {params}; bounding radius {gen.bounding_radius:.0f} A, "
            f"volume {tuple(volume.shape)} at {VOXEL_SIZE:g} A, "
            f"coarse field {tuple(gen.field.phi.shape)}, {elapsed:.1f} s on {DEVICE}"
        )

        ax = fig.add_subplot(2, 4, col + 1, projection="3d")
        phi = gen.field.phi.cpu().numpy().transpose(2, 1, 0)
        isosurface_axes(ax, phi, gen.field.spacing_angstrom, TEAL)
        ax.view_init(elev=28, azim=-60)
        ax.set_title(kind, fontsize=12)

        ax = fig.add_subplot(2, 4, col + 5)
        axis = cut_axis[kind]
        section = volume.select(axis, volume.shape[axis] // 2).numpy()
        ax.imshow(section, cmap="gray_r", origin="lower")
        ax.set_title("xy section" if axis == 0 else "xz section", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-shape-kinds.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def _min_reach(phi: torch.Tensor, grid: Grid) -> tuple[float, float]:
    _, _, outer, inner = reach(phi, grid)
    return float(outer.min()), float(inner.min())


def figure_regularise() -> None:
    """A union of primitives before and after `regularise`: the crease
    where two balls meet, a spike thinner than a bilayer and a slot
    narrower than one are all replaced by surfaces a bilayer fits on."""
    grid = Grid((1000.0, 800.0, 400.0), device=DEVICE)
    c = grid.centre
    left, right = c + [-170.0, 0.0, 0.0], c + [190.0, 30.0, 0.0]
    raw = union(
        ball(grid, left, 230.0),
        ball(grid, right, 190.0),
        # A spike of 12 A radius, thinner than the bilayer itself.
        tube(
            grid,
            [right, right + [150.0, 150.0, 0.0], right + [260.0, 260.0, 0.0]],
            12.0,
        ),
    )
    # A slot 30 A wide cut into the left ball.
    slot = sheet(
        grid, left + [-150.0, 0.0, 0.0], rot_to([0.0, 1.0, 0.0]), lambda th: 160.0, 15.0
    )
    raw = redistance(subtract(raw, slot), grid)
    regular = regularise(raw, grid)

    raw_outer, raw_inner = _min_reach(raw, grid)
    reg_outer, reg_inner = _min_reach(regular, grid)
    print(
        f"regularise: grid {grid.shape} = {grid.n_nodes} nodes at {grid.h:g} A; "
        f"smallest reach before {min(raw_outer, raw_inner):.1f} A "
        f"(outer {raw_outer:.1f}, inner {raw_inner:.1f}), "
        f"after {min(reg_outer, reg_inner):.1f} A "
        f"(outer {reg_outer:.1f}, inner {reg_inner:.1f}); "
        f"accepted minimum {MIN_REACH:g} A"
    )

    k = grid.shape[2] // 2
    x = grid.axes()[0].flatten().cpu().numpy()
    y = grid.axes()[1].flatten().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
    for ax, phi, title in zip(
        axes,
        (raw, regular),
        ("union of primitives", "after regularise (rolling ball of 30 A)"),
    ):
        section = phi[:, :, k].cpu().numpy().T
        ax.contourf(x, y, section, levels=[-1e9, 0.0], colors=["#dcecec"])
        ax.contour(x, y, section, levels=[0.0], colors=[TEAL], linewidths=1.6)
        # The two leaflets of a 37 A bilayer.
        ax.contour(
            x,
            y,
            section,
            levels=[-18.5, 18.5],
            colors=[ORANGE],
            linewidths=0.7,
            linestyles="solid",
        )
        ax.set_aspect("equal")
        ax.set_xlim(grid.lo[0], grid.lo[0] + grid.box[0])
        ax.set_ylim(grid.lo[1], grid.lo[1] + grid.box[1])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (A)")
    axes[0].set_ylabel("y (A)")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-shape-regularise.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def _bud(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
    c = grid.centre
    a, b = c + [-200.0, 0.0, 0.0], c + [330.0, 40.0, 0.0]
    phi = union(
        ball(grid, a, 300.0),
        ball(grid, b, 170.0),
        tube(grid, [a, (a + b) / 2, b], 62.0),
    )
    return [regularise(phi, grid)]


def _multilamellar(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
    outer = regularise(
        undulate(ball(grid, grid.centre, 350.0), grid, rng, 25.0, 500.0), grid
    )
    return [outer] + [nested(outer, grid, 90.0 * k) for k in range(1, 3)]


def _sheet_with_tubules(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
    c = grid.centre
    flat = sheet(
        grid,
        c + [-450.0, 0.0, 0.0],
        np.eye(3),
        wobbly(480.0, rng, amp=0.12),
        62.0,
    )
    upper = tube(
        grid,
        [c + [-250, 150, 0], c + [250, 330, 0], c + [550, 250, 0], c + [850, 420, 0]],
        120.0,
    )
    lower = tube(
        grid,
        [
            c + [-250, -200, 0],
            c + [230, -420, 0],
            c + [500, -330, 0],
            c + [850, -560, 0],
        ],
        120.0,
    )
    rung = tube(grid, [c + [500, -330, 0], c + [540, -40, 0], c + [550, 250, 0]], 120.0)
    return [regularise(union(flat, upper, lower, rung), grid)]


def _plasma_membrane(grid: Grid, rng: np.random.Generator) -> list[torch.Tensor]:
    def wave(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 45.0 * torch.sin(x / 330.0) * torch.cos(y / 410.0 + 1.0)

    cell = half_space(grid, 420.0, wave)
    pit = np.array([450.0, 600.0, 200.0])
    bud = np.array([1250.0, 600.0, 760.0])
    cell = subtract(
        cell,
        union(
            ball(grid, pit, 170.0),
            tube(grid, [pit, pit + [0, 0, 150], pit + [0, 0, 400]], 75.0),
        ),
    )
    cell = union(
        cell,
        ball(grid, bud, 150.0),
        tube(grid, [bud, bud - [0, 0, 200], bud - [0, 0, 450]], 75.0),
    )
    return [regularise(cell, grid)]


def figure_compositions() -> None:
    """Shapes outside the four kinds, composed from the same primitives and
    rendered with the same bilayer profile."""
    profile = build_measured_bilayer_profile(device=DEVICE)
    cases = [
        ("union(ball, ball, tube)", (1300.0, 1000.0, 600.0), None, _bud, 0),
        ("nested(...) twice", (1000.0, 1000.0, 1000.0), None, _multilamellar, 0),
        (
            "union(sheet, tube, tube, tube)",
            (2200.0, 1600.0, 500.0),
            None,
            _sheet_with_tubules,
            0,
        ),
        (
            "subtract and union on half_space",
            (1800.0, 1200.0, 1000.0),
            200.0,
            _plasma_membrane,
            1,
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.9))
    for ax, (title, box, pad, build, axis) in zip(axes, cases):
        rng = np.random.default_rng(SEED)
        grid = Grid(box, pad=pad, device=DEVICE)
        _sync()
        start = time.perf_counter()
        surfaces = build(grid, rng)
        shape_zyx = tuple(int(math.ceil(box[k] / VOXEL_SIZE)) for k in (2, 1, 0))
        volume = render_bilayer(
            surfaces, grid, profile, shape_zyx, VOXEL_SIZE, origin_xyz=(0.0, 0.0, 0.0)
        )
        _sync()
        print(
            f"{title}: {len(surfaces)} surface(s), grid {grid.shape}, "
            f"output {shape_zyx}, {time.perf_counter() - start:.1f} s on {DEVICE}"
        )
        section = volume.select(axis, volume.shape[axis] // 2).cpu().numpy()
        ax.imshow(section, cmap="gray_r", origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/membrane-shape-compositions.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_kinds()
    figure_regularise()
    figure_compositions()
