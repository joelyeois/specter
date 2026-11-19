"""
Figure and printed output for docs/tutorials/membrane-gallery.md.

Run with: uv run python docs-figures/tutorials/membrane_gallery.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("membrane_gallery")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt
import numpy as np
import torch

import specter.specimen.membrane as membrane
from specter.specimen.membrane import Grid, regularise

device = "cuda"

# Each shape returns its grid (a box in Å, ordered x, y, z) and a list of
# regularised signed-distance fields, one per membrane.


def discocyte(rng):
    g = Grid((1000.0, 1000.0, 1000.0), device=device)
    phi = membrane.sheet(
        g,
        g.centre,
        membrane.rot_to([0.3, -0.2, 1.0]),
        membrane.circle(330.0),
        lambda f: 42.0 + 70.0 * np.clip(f, 0, 1) ** 3,  # half-thickness to the rim
    )
    return g, [regularise(phi, g)]


def stomatocyte(rng):
    g = Grid((1000.0, 1000.0, 1000.0), device=device)
    phi = membrane.cap(
        g,
        g.centre + [0.0, 0.0, -30.0],
        membrane.rot_to([0.25, 0.1, -1.0]),
        radius=270.0,
        polar_max=np.radians(118.0),
        half=85.0,
    )
    return g, [regularise(phi, g)]


def budding_vesicle(rng):
    g = Grid((1300.0, 1000.0, 1000.0), device=device)
    a = g.centre + [-200.0, 0.0, 0.0]
    b = g.centre + [330.0, 40.0, 30.0]
    phi = membrane.union(
        membrane.ball(g, a, 300.0),
        membrane.ball(g, b, 170.0),
        membrane.tube(g, [a, (a + b) / 2, b], 62.0),
    )
    return g, [regularise(phi, g)]


def pearled_tubule(rng):
    g = Grid((1600.0, 800.0, 800.0), device=device)
    path = [
        g.centre + [x, 60.0 * np.sin(x / 300.0), 0.0] for x in np.linspace(-950, 950, 9)
    ]
    phi = membrane.tube(g, path, lambda s: 110.0 + 50.0 * np.sin(2 * np.pi * 5.5 * s))
    return g, [regularise(phi, g)]


def tubule_network(rng):
    g = Grid((1800.0, 1800.0, 1000.0), device=device)
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
    pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]
    pairs += [(nodes[0], ends[0]), (nodes[1], ends[1]), (nodes[2], ends[2])]
    pairs += [(nodes[2], ends[3]), (nodes[1], ends[4])]
    tubes = []
    for start, end in pairs:
        middle = (start + end) / 2 + rng.normal(size=3) * 70.0
        tubes.append(membrane.tube(g, [start, middle, end], 130.0))
    return g, [regularise(membrane.union(*tubes), g)]


def cisterna_stack(rng):
    g = Grid((1500.0, 1500.0, 1100.0), device=device)
    sacs = []
    for k in range(5):

        def bend(u, v, k=k):
            return u**2 / (2 * (1500.0 + 260.0 * k)) + 25.0 * np.sin(v / 300.0 + k)

        phi = membrane.sheet(
            g,
            g.centre + [0.0, 0.0, -430.0 + 190.0 * k],
            np.eye(3),
            membrane.wobbly(560.0 - 45.0 * k, rng, amp=0.1),
            52.0,
            bend,
        )
        sacs.append(regularise(phi, g))
    return g, sacs


def multilamellar_vesicle(rng):
    g = Grid((1000.0, 1000.0, 1000.0), device=device)
    ball = membrane.ball(g, g.centre, 350.0)
    outer = regularise(membrane.undulate(ball, g, rng, 25.0, 500.0), g)
    return g, [outer] + [membrane.nested(outer, g, 78.0 * k) for k in range(1, 4)]


def plasma_membrane(rng):
    g = Grid((1800.0, 1800.0, 1100.0), pad=200.0, device=device)

    def wave(x, y):  # torch tensors: the grid's x and y axes
        ripple = 45.0 * torch.sin(x / 330.0) * torch.cos(y / 410.0 + 1.0)
        return ripple + 0.04 * (x - 900.0)

    cell = membrane.half_space(g, 520.0, wave)
    pit, bud = np.array([520.0, 900.0, 250.0]), np.array([1250.0, 900.0, 880.0])
    invagination = membrane.union(
        membrane.ball(g, pit, 190.0),
        membrane.tube(g, [pit, pit + [0, 0, 180.0], pit + [0, 0, 420.0]], 75.0),
    )
    cell = membrane.subtract(cell, invagination)
    cell = membrane.union(
        cell,
        membrane.ball(g, bud, 170.0),
        membrane.tube(g, [bud, bud - [0, 0, 200.0], bud - [0, 0, 480.0]], 75.0),
    )
    dimple = membrane.ball(g, np.array([1300.0, 450.0, 560.0]), 210.0)
    return g, [regularise(membrane.subtract(cell, dimple), g)]


# The shapes, and the axis of (Z, Y, X) each is cut and projected along.
shapes = [
    (discocyte, 1),
    (stomatocyte, 1),
    (budding_vesicle, 0),
    (pearled_tubule, 0),
    (tubule_network, 0),
    (cisterna_stack, 1),
    (multilamellar_vesicle, 0),
    (plasma_membrane, 1),
]
voxel_size = 10.0
profile = membrane.build_measured_bilayer_profile(
    thickness_angstrom=37.0, device=device
)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for index, (shape, axis) in enumerate(shapes):
    g, surfaces = shape(np.random.default_rng(0))
    shape_zyx = tuple(round(extent / voxel_size) for extent in g.box[::-1])
    volume = membrane.render_bilayer(
        surfaces, g, profile, shape_zyx, voxel_size, origin_xyz=tuple(g.lo)
    )
    weakest = min(
        float(min(outer.min(), inner.min()))
        for _, _, outer, inner in (membrane.reach(s, g) for s in surfaces)
    )
    print(f"{shape.__name__}: {len(surfaces)} surface(s), reach {weakest:.0f} Å")

    central = volume.select(axis, volume.shape[axis] // 2)
    projection = volume.sum(axis) * voxel_size
    row, column = 2 * (index // 4), index % 4
    for ax, image in zip(axes[row : row + 2, column], [central, projection]):
        ax.imshow(image.cpu().numpy(), cmap="gray_r", origin="lower")
        ax.set(xticks=[], yticks=[])
    axes[row, column].set_title(shape.__name__.replace("_", " "))
    if column == 0:
        axes[row, 0].set_ylabel("central slice")
        axes[row + 1, 0].set_ylabel("projection")

fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "membrane_gallery")

thumbnail(as_numpy(central)[:, 60:170], "membrane_gallery")

transcript.close()
