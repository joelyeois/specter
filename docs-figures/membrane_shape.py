"""
Generate the figures for docs/concepts/membrane-shape/ (index.md,
geometry.md, organelles.md): how `specter.specimen.membrane` builds a
membrane as the boundary of a solid.

Every panel is drawn from the shipped code: `MembraneGenerator` for the four
organelle kinds, and `Grid`, the primitives, `regularise`, `reach` and
`render_bilayer` for the regularisation figure and the gallery of twenty
composed shapes. The numbers the pages quote (reach before and after
regularisation, node counts, build times) are the ones this script prints.

Run with: uv run python docs-figures/membrane_shape.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import specter.specimen.membrane as prim
from _render import TEAL, isosurface_axes
from specter.specimen.membrane import (
    ORGANELLE_DEFAULTS,
    BilayerProfile,
    Grid,
    MembraneGenerator,
    build_measured_bilayer_profile,
    reach,
    regularise,
    render_bilayer,
)
from specter.specimen.membrane._geometry import MIN_REACH, redistance

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
    raw = prim.union(
        prim.ball(grid, left, 230.0),
        prim.ball(grid, right, 190.0),
        # A spike of 12 A radius, thinner than the bilayer itself.
        prim.tube(
            grid,
            [right, right + [150.0, 150.0, 0.0], right + [260.0, 260.0, 0.0]],
            12.0,
        ),
    )
    # A slot 30 A wide cut into the left ball.
    slot = prim.sheet(
        grid,
        left + [-150.0, 0.0, 0.0],
        prim.rot_to([0.0, 1.0, 0.0]),
        lambda th: 160.0,
        15.0,
    )
    raw = redistance(prim.subtract(raw, slot), grid)
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


# ---------------------------------------------------------------- gallery
#
# Twenty shapes composed from the public primitives. Each recipe returns a
# `_Shape`: its grid, its regularised surfaces, the view of its 3D panel and
# the axis its section is cut normal to.

GALLERY_SEED = 1
GALLERY_VOXEL_SIZE = 4.0
# Thickness of the shell the 3D panels draw, A: the bilayer's 37 A.
SHELL_THICKNESS = 37.0
OUTER_FACE = (0.60, 0.72, 0.86)
LUMEN_FACE = (0.93, 0.74, 0.52)
CUT_FACE = (0.16, 0.18, 0.22)


@dataclass
class _Shape:
    name: str
    grid: Grid
    surfaces: list[torch.Tensor]
    view: dict[str, Any] = field(default_factory=dict)
    section_axis: int = 2


def _grid(box: tuple[float, float, float], pad: float | None = None) -> Grid:
    return Grid(box, pad=pad, device=DEVICE)


def _vesicle(rng: np.random.Generator) -> _Shape:
    g = _grid((1000.0, 1000.0, 1000.0))
    return _Shape("vesicle", g, [regularise(prim.ball(g, g.centre, 330.0), g)])


def _undulating_vesicle(rng: np.random.Generator) -> _Shape:
    g = _grid((1000.0, 1000.0, 1000.0))
    phi = prim.undulate(prim.ball(g, g.centre, 320.0), g, rng, 45.0, 450.0)
    return _Shape("undulating vesicle", g, [regularise(phi, g)])


def _prolate(rng: np.random.Generator) -> _Shape:
    g = _grid((1200.0, 1000.0, 1000.0))
    rot = Rotation.from_euler("zy", [25, 15], degrees=True).as_matrix()
    phi = prim.ellipsoid(g, g.centre, (470.0, 230.0, 230.0), rot)
    phi = prim.undulate(phi, g, rng, 15.0, 500.0)
    return _Shape("prolate / elongated", g, [regularise(phi, g)])


def _discocyte(rng: np.random.Generator) -> _Shape:
    g = _grid((1000.0, 1000.0, 1000.0))

    def half(f: np.ndarray) -> np.ndarray:
        # Thicker towards the rim, which is what makes the disc biconcave.
        return 42.0 + 70.0 * np.clip(f, 0.0, 1.0) ** 3

    phi = prim.sheet(
        g, g.centre, prim.rot_to([0.3, -0.2, 1.0]), prim.circle(330.0), half
    )
    return _Shape("discocyte (biconcave)", g, [regularise(phi, g)], {"cut": "half"}, 1)


def _stomatocyte(rng: np.random.Generator) -> _Shape:
    g = _grid((1000.0, 1000.0, 1000.0))
    phi = prim.cap(
        g,
        g.centre + [0.0, 0.0, -30.0],
        prim.rot_to([0.25, 0.1, -1.0]),
        270.0,
        math.radians(118),
        85.0,
    )
    return _Shape("stomatocyte (cup)", g, [regularise(phi, g)], {"cut": "half"}, 1)


def _bud(rng: np.random.Generator) -> _Shape:
    g = _grid((1300.0, 1000.0, 1000.0))
    a, b = g.centre + [-200.0, 0.0, 0.0], g.centre + [330.0, 40.0, 30.0]
    phi = prim.union(
        prim.ball(g, a, 300.0),
        prim.ball(g, b, 170.0),
        prim.tube(g, [a, (a + b) / 2, b], 62.0),
    )
    view = {"cut": (0.0, 1.0, 0.0), "az": 65.0}
    return _Shape("bud on a neck", g, [regularise(phi, g)], view)


def _pearled_tubule(rng: np.random.Generator) -> _Shape:
    g = _grid((1600.0, 800.0, 800.0))
    ctrl = [
        g.centre + [x, 60.0 * math.sin(x / 300.0), 0.0]
        for x in np.linspace(-950, 950, 9)
    ]
    phi = prim.tube(g, ctrl, lambda s: 110.0 + 50.0 * np.sin(2 * np.pi * 5.5 * s))
    return _Shape("pearled tubule", g, [regularise(phi, g)])


def _curved_tubule(rng: np.random.Generator) -> _Shape:
    g = _grid((1600.0, 1200.0, 1200.0))
    ctrl = [
        [-200, 350, 500],
        [350, 420, 650],
        [800, 800, 700],
        [700, 1000, 380],
        [1100, 700, 350],
        [1500, 800, 600],
        [1900, 700, 700],
    ]
    return _Shape("curved tubule", g, [regularise(prim.tube(g, ctrl, 140.0), g)])


def _tubule_network(rng: np.random.Generator) -> _Shape:
    g = _grid((1800.0, 1800.0, 1000.0))
    nodes = np.array([[500, 600, 500], [1200, 500, 420], [950, 1250, 600]], float)
    ends = np.array(
        [
            [-200, 300, 450],
            [2000, 150, 300],
            [1000, 2000, 700],
            [-200, 1300, 600],
            [2000, 1300, 500],
        ],
        float,
    )
    pairs = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[0]),
        (nodes[0], ends[0]),
        (nodes[1], ends[1]),
        (nodes[2], ends[2]),
        (nodes[2], ends[3]),
        (nodes[1], ends[4]),
    ]
    parts = []
    for a, b in pairs:
        mid = (a + b) / 2 + rng.normal(size=3) * 70.0
        parts.append(prim.tube(g, [a, mid, b], 130.0))
    return _Shape("three-way tubule network", g, [regularise(prim.union(*parts), g)])


def _flat_cisterna(rng: np.random.Generator) -> _Shape:
    g = _grid((1500.0, 1500.0, 700.0))

    def bend(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return 60.0 * np.sin(u / 380.0) * np.cos(v / 420.0)

    phi = prim.sheet(
        g, g.centre, prim.rot_to([0.05, 0.1, 1.0]), prim.wobbly(520.0, rng), 62.0, bend
    )
    view = {"cut": (0.0, 1.0, 0.0), "az": 60.0, "el": 30.0}
    return _Shape("flat cisterna", g, [regularise(phi, g)], view, 1)


def _cisterna_stack(rng: np.random.Generator) -> _Shape:
    g = _grid((1500.0, 1500.0, 1100.0))
    surfaces = []
    for k in range(5):

        def bend(u: np.ndarray, v: np.ndarray, k: int = k) -> np.ndarray:
            return u**2 / (2 * (1500.0 + 260.0 * k)) + 25.0 * np.sin(v / 300.0 + k)

        outline = prim.wobbly(560.0 - 45.0 * k, rng, amp=0.1)
        centre = g.centre + [0.0, 0.0, -430.0 + 190.0 * k]
        surfaces.append(
            regularise(prim.sheet(g, centre, np.eye(3), outline, 52.0, bend), g)
        )
    view = {"cut": (0.0, 1.0, 0.0), "az": 60.0, "el": 20.0}
    return _Shape("curved cisterna stack", g, surfaces, view, 1)


def _fenestrated_sheet(rng: np.random.Generator) -> _Shape:
    g = _grid((1600.0, 1600.0, 700.0))
    holes: list[np.ndarray] = []
    for _ in range(4000):
        h = np.r_[rng.uniform(-520, 520, 2), rng.uniform(120, 170)]
        clear = all(np.hypot(*(h[:2] - o[:2])) > h[2] + o[2] + 70 for o in holes)
        if clear and np.hypot(*h[:2]) < 500:
            holes.append(h)

    def bend(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return 50.0 * np.sin(u / 350.0 + 1.0) * np.cos(v / 300.0)

    outline = prim.wobbly(680.0, rng, amp=0.1)
    phi = prim.sheet(g, g.centre, np.eye(3), outline, 60.0, bend, holes)
    return _Shape("fenestrated sheet", g, [regularise(phi, g)], {"el": 40.0})


def _sheet_to_tubules(rng: np.random.Generator) -> _Shape:
    g = _grid((1800.0, 1600.0, 900.0))
    c = g.centre
    flat = prim.sheet(
        g,
        c + [-350.0, 0.0, 0.0],
        np.eye(3),
        prim.wobbly(480.0, rng, amp=0.12),
        62.0,
        lambda u, v: 40.0 * np.sin(v / 300.0),
    )
    upper = prim.tube(
        g,
        [
            c + [-150, 150, 0],
            c + [350, 330, 90],
            c + [700, 250, 180],
            c + [1100, 420, 150],
        ],
        120.0,
    )
    lower = prim.tube(
        g,
        [
            c + [-150, -200, 0],
            c + [330, -420, -80],
            c + [650, -330, -150],
            c + [1100, -560, -100],
        ],
        120.0,
    )
    rung = prim.tube(
        g, [c + [650, -330, -150], c + [690, -40, 0], c + [700, 250, 180]], 120.0
    )
    phi = regularise(prim.union(flat, upper, lower, rung), g)
    return _Shape("sheet-to-tubule transition", g, [phi])


def _multilamellar(rng: np.random.Generator) -> _Shape:
    g = _grid((1000.0, 1000.0, 1000.0))
    outer = regularise(
        prim.undulate(prim.ball(g, g.centre, 350.0), g, rng, 25.0, 500.0), g
    )
    surfaces = [outer] + [prim.nested(outer, g, 78.0 * k) for k in range(1, 4)]
    return _Shape("multilamellar vesicle", g, surfaces, {"cut": "half"})


def _multivesicular(rng: np.random.Generator) -> _Shape:
    g = _grid((1100.0, 1100.0, 1100.0))
    c = g.centre
    outer = regularise(prim.undulate(prim.ball(g, c, 400.0), g, rng, 30.0, 500.0), g)
    inner: list[tuple[np.ndarray, float]] = []
    tries = 0
    while len(inner) < 9 and tries < 5000:
        tries += 1
        p, r = c + rng.uniform(-330, 330, 3), float(rng.uniform(70, 125))
        fits = np.linalg.norm(p - c) + r < 310
        if fits and all(np.linalg.norm(p - q) > r + s + 60 for q, s in inner):
            inner.append((p, r))
    surfaces = [outer] + [regularise(prim.ball(g, p, r), g) for p, r in inner]
    return _Shape("multivesicular body", g, surfaces, {"cut": "half"})


def _mitochondrion(
    g: Grid,
    rng: np.random.Generator,
    ctrl: list[np.ndarray],
    radius: float,
    cristae: torch.Tensor,
    gap: float = 130.0,
) -> list[torch.Tensor]:
    """An outer membrane round a tube, and an inner one `gap` inside it with
    the cristae cut into it."""
    outer = regularise(
        prim.undulate(prim.tube(g, ctrl, radius), g, rng, 20.0, 600.0), g
    )
    inner = regularise(prim.subtract(outer + gap, cristae), g)
    return [outer, inner]


_MITO_VIEW = {"cut": (0.0, 1.0, 0.0), "az": 65.0, "el": 18.0}


def _mito_lamellar(rng: np.random.Generator) -> _Shape:
    g = _grid((2600.0, 1700.0, 1700.0))
    c = g.centre
    radius = 640.0
    parts = []
    for x in np.arange(-780, 800, 260):
        x = x + rng.uniform(-30, 30)
        tilt = prim.rot_to([1.0, rng.normal() * 0.12, rng.normal() * 0.12])
        ang = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(40, 130)
        ctr = (
            c
            + [x, 0.0, 0.0]
            + tilt @ np.array([off * math.cos(ang), off * math.sin(ang), 0.0])
        )
        parts.append(
            prim.sheet(
                g,
                ctr,
                tilt,
                prim.wobbly(radius - 290.0, rng, amp=0.08),
                75.0,
                lambda u, v: 18.0 * np.sin(u / 200.0),
            )
        )
        for j in range(int(rng.integers(1, 3))):
            a = ang + j * np.pi + rng.normal() * 0.3
            e = np.array([math.cos(a), math.sin(a), 0.0])
            ends = [
                ctr + tilt @ (e * 200),
                ctr + tilt @ (e * 420),
                ctr + tilt @ (e * 700),
            ]
            parts.append(prim.tube(g, ends, 78.0))
    ctrl = [c + [-900.0, 0.0, 0.0], c, c + [900.0, 0.0, 0.0]]
    surfaces = _mitochondrion(g, rng, ctrl, radius, prim.union(*parts))
    return _Shape("mitochondrion, lamellar cristae", g, surfaces, _MITO_VIEW)


def _mito_tubular(rng: np.random.Generator) -> _Shape:
    g = _grid((2400.0, 1700.0, 1700.0))
    c = g.centre
    radius = 620.0
    parts = []
    for _ in range(11):
        x = rng.uniform(-850, 850)
        a = rng.uniform(0, 2 * np.pi)
        e = np.array([0.0, math.cos(a), math.sin(a)])
        start = c + [x, 0.0, 0.0] + e * 760
        walk = prim.random_walk(
            rng, start, 170.0, int(rng.integers(3, 6)), 0.7, direction=-e
        )
        parts.append(prim.tube(g, walk, 85.0))
    ctrl = [c + [-830.0, -40.0, 0.0], c + [0.0, 40.0, 20.0], c + [830.0, -30.0, 0.0]]
    surfaces = _mitochondrion(g, rng, ctrl, radius, prim.union(*parts))
    return _Shape("mitochondrion, tubular cristae", g, surfaces, _MITO_VIEW)


def _plasma_membrane(rng: np.random.Generator) -> _Shape:
    g = _grid((1800.0, 1800.0, 1100.0), pad=200.0)

    def wave(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 45.0 * torch.sin(x / 330.0) * torch.cos(y / 410.0 + 1.0) + 0.04 * (
            x - 900.0
        )

    cell = prim.half_space(g, 520.0, wave)
    pit, bud = np.array([520.0, 620.0, 250.0]), np.array([1250.0, 1150.0, 880.0])
    cell = prim.subtract(
        cell,
        prim.union(
            prim.ball(g, pit, 190.0),
            prim.tube(g, [pit, pit + [0, 0, 180], pit + [0, 0, 420]], 75.0),
        ),
    )
    cell = prim.union(
        cell,
        prim.ball(g, bud, 170.0),
        prim.tube(g, [bud, bud - [0, 0, 200], bud - [0, 0, 480]], 75.0),
    )
    cell = prim.subtract(cell, prim.ball(g, [1300.0, 450.0, 560.0], 210.0))
    return _Shape(
        "plasma membrane: pits and a bud", g, [regularise(cell, g)], {"el": 28.0}, 1
    )


def _double_envelope(rng: np.random.Generator) -> _Shape:
    g = _grid((2800.0, 2800.0, 900.0), pad=420.0)
    pores = [
        (-650, -500, 600),
        (700, -620, 600),
        (60, 620, 600),
        (-1250, 900, 600),
        (1330, 760, 600),
    ]

    def bend(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (u**2 + v**2) / (2 * 9000.0) - 120.0

    phi = prim.sheet(
        g,
        g.centre,
        np.eye(3),
        prim.circle(4000.0),
        170.0,
        bend,
        pores,
        extent=1900.0,
        spacing=5.0,
    )
    return _Shape(
        "double envelope with pores", g, [regularise(phi, g)], {"el": 32.0}, 1
    )


def _contact(rng: np.random.Generator) -> _Shape:
    g = _grid((1500.0, 1100.0, 1100.0))
    a = prim.ball(g, g.centre + [-240.0, 0.0, 0.0], 380.0)
    b = prim.ball(g, g.centre + [300.0, 60.0, 0.0], 330.0)
    gap = 110.0
    # Each solid is cut by the plane halfway between them, `gap` apart.
    fa = regularise(torch.maximum(a, (a - b) / 2 + gap / 2), g)
    fb = regularise(torch.maximum(b, (b - a) / 2 + gap / 2), g)
    view = {"cut": (0.0, 1.0, 0.0), "az": 70.0}
    return _Shape("flattened contact", g, [fa, fb], view)


GALLERY = [
    _vesicle,
    _undulating_vesicle,
    _prolate,
    _discocyte,
    _stomatocyte,
    _bud,
    _pearled_tubule,
    _curved_tubule,
    _tubule_network,
    _flat_cisterna,
    _cisterna_stack,
    _fenestrated_sheet,
    _sheet_to_tubules,
    _multilamellar,
    _multivesicular,
    _mito_lamellar,
    _mito_tubular,
    _plasma_membrane,
    _double_envelope,
    _contact,
]


def _trilinear(vol: torch.Tensor, grid: Grid, pts: torch.Tensor) -> torch.Tensor:
    """`vol` ``(C, X, Y, Z)`` on `grid`, sampled at physical points ``(N, 3)``."""
    size = torch.tensor(grid.shape, device=vol.device, dtype=torch.float32)
    g = (
        (((pts - grid.origin) / grid.h) / (size - 1) * 2 - 1)
        .flip(-1)
        .view(1, -1, 1, 1, 3)
    )
    out = F.grid_sample(
        vol[None], g, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out[0, :, :, 0, 0]


def _raymarch(
    shape: _Shape,
    az: float = 32.0,
    el: float = 22.0,
    cut: str | tuple[float, float, float] | None = None,
    res: int = 560,
    zoom: float = 1.2,
) -> np.ndarray:
    """
    Sphere-traced RGB view ``(res, res, 3)`` of the bilayer shell.

    The shell is ``|phi| - t/2`` of the nearest surface, clipped to the box.
    A face looking away from the lumen is blue, one looking into it orange,
    and wherever the shell is cut (by the box, or by the plane through the
    centre normal to `cut`, ``"half"`` meaning the camera direction) dark.
    """
    grid = shape.grid
    dev = grid.device
    stack = torch.stack(shape.surfaces)
    nearest = stack.gather(0, stack.abs().argmin(0, keepdim=True))
    del stack
    vol = torch.cat([nearest.abs() - SHELL_THICKNESS / 2, nearest])
    box = torch.tensor(grid.box, device=dev)
    centre = torch.tensor(grid.centre, device=dev, dtype=torch.float32)
    a, e = math.radians(az), math.radians(el)
    to_cam = torch.tensor(
        [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)], device=dev
    )
    fwd = -to_cam
    right = F.normalize(
        torch.linalg.cross(fwd, torch.tensor([0.0, 0.0, 1.0], device=dev)), dim=0
    )
    up = torch.linalg.cross(right, fwd)
    rb = 0.5 * float(box.norm())
    if cut is None:
        cut_n = None
    elif isinstance(cut, str):
        cut_n = to_cam
    else:
        cut_n = F.normalize(torch.tensor(cut, device=dev, dtype=torch.float32), dim=0)
    s = torch.linspace(-1, 1, res, device=dev) * rb * 0.82 / zoom
    yy, xx = torch.meshgrid(-s, s, indexing="ij")
    o = centre + to_cam * rb + xx.reshape(-1, 1) * right + yy.reshape(-1, 1) * up

    def sdf(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = _trilinear(vol[:1], grid, p)[0]
        d_box = ((p - centre).abs() - box / 2).max(1).values
        d = torch.maximum(v, d_box)
        flat = d_box > v
        if cut_n is not None:
            d_cut = ((p - centre) * cut_n).sum(1)
            flat |= d_cut > d
            d = torch.maximum(d, d_cut)
        return d, flat

    t = torch.zeros(o.shape[0], device=dev)
    for _ in range(260):
        d, _ = sdf(o + t[:, None] * fwd)
        t = t + torch.where(d < 0.3, torch.zeros_like(d), d.clamp(min=0.6) * 0.85)
        t.clamp_(max=2 * rb)
    p = o + t[:, None] * fwd
    d, flat = sdf(p)
    hit = (d < 1.5) & (t < 2 * rb - 1)
    eye = torch.eye(3, device=dev)
    grad = torch.stack(
        [
            _trilinear(vol[:1], grid, p + 4.0 * eye[k])[0]
            - _trilinear(vol[:1], grid, p - 4.0 * eye[k])[0]
            for k in range(3)
        ],
        1,
    )
    n = F.normalize(grad, dim=1)
    light = F.normalize(to_cam + 0.55 * up - 0.45 * right, dim=0)
    lambert = (n * light).sum(1).clamp(0, 1)
    occlusion = (_trilinear(vol[:1], grid, p + n * 60.0)[0] / 60.0).clamp(0, 1)
    outside = (_trilinear(vol[1:], grid, p)[0] > 0)[:, None]
    base = torch.where(
        outside,
        torch.tensor(OUTER_FACE, device=dev),
        torch.tensor(LUMEN_FACE, device=dev),
    )
    col = base * ((0.32 + 0.68 * lambert) * (0.45 + 0.55 * occlusion))[:, None]
    col = torch.where(
        flat[:, None], torch.tensor(CUT_FACE, device=dev).expand_as(col), col
    )
    depth = ((t - 0.2 * rb) / (1.8 * rb)).clamp(0, 1)[:, None]
    col = col * (1 - 0.25 * depth) + 0.25 * depth
    img = torch.where(hit[:, None], col, torch.ones_like(col))
    return img.view(res, res, 3).clamp(0, 1).cpu().numpy()


def _central_section(
    shape: _Shape, profile: BilayerProfile, voxel_size: float
) -> np.ndarray:
    """Central section of the rendered potential normal to the shape's
    section axis, rendered in a slab of at most 64 voxels around it; rows
    are the higher remaining axis."""
    grid, axis = shape.grid, shape.section_axis
    n_xyz = [int(round(b / voxel_size)) for b in grid.box]
    i = n_xyz[axis] // 2
    depth = min(64, n_xyz[axis])
    start = min(max(i - depth // 2, 0), n_xyz[axis] - depth)
    origin = [voxel_size / 2] * 3
    origin[axis] += start * voxel_size
    n_xyz[axis] = depth
    volume = render_bilayer(
        shape.surfaces,
        grid,
        profile,
        (n_xyz[2], n_xyz[1], n_xyz[0]),
        voxel_size,
        origin_xyz=origin,
    )
    return volume.select(2 - axis, i - start).cpu().numpy()


def _min_reach_all(shape: _Shape) -> float:
    return min(min(_min_reach(phi, shape.grid)) for phi in shape.surfaces)


def figure_gallery() -> None:
    """Twenty shapes composed from the primitives: the bilayer shell in 3D
    above, a central section of the rendered potential at 4 A per voxel
    below."""
    profile = build_measured_bilayer_profile(device=DEVICE)
    ncol = 5
    nrow = math.ceil(len(GALLERY) / ncol)
    fig, axes = plt.subplots(
        2 * nrow, ncol, figsize=(3.3 * ncol, 6.3 * nrow), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    total = 0.0
    for j, recipe in enumerate(GALLERY):
        r, c = divmod(j, ncol)
        _sync()
        start = time.perf_counter()
        shape = recipe(np.random.default_rng(GALLERY_SEED))
        _sync()
        built = time.perf_counter() - start
        total += built
        reach_min = _min_reach_all(shape)
        print(
            f"{shape.name}: {len(shape.surfaces)} surface(s), grid {shape.grid.shape}, "
            f"min reach {reach_min:.1f} A, built in {built:.1f} s on {DEVICE}"
        )
        axes[2 * r, c].imshow(_raymarch(shape, **shape.view))
        axes[2 * r, c].set_title(shape.name, fontsize=11)
        section = _central_section(shape, profile, GALLERY_VOXEL_SIZE)
        axes[2 * r + 1, c].imshow(section, cmap="gray_r", origin="lower")
        axes[2 * r + 1, c].set_title(
            f"min reach {reach_min:.0f} Å · built in {built:.1f} s",
            fontsize=8,
            color="0.35",
        )
        del shape
    print(f"gallery: {len(GALLERY)} shapes built in {total:.1f} s on {DEVICE}")
    fig.suptitle(
        "Top: outer face blue, lumen face orange, cut bilayer dark. "
        "Bottom: central section of the rendered potential (4 Å voxels).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = f"{OUT_DIR}/membrane-shape-gallery.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_kinds()
    figure_regularise()
    figure_gallery()
