"""
Generate the figures for docs/concepts/cryoet-specimen/filaments.md, which
explains the filament component of `specter build tomogram`
(``specter.specimen.filament``).

Calls the same functions the real code path uses
(`generate_filament_path`, `filament_orientations`, the generator's own
filament renderer, and `TomogramSpecimenGenerator` itself for the hero)
rather than reimplementing the geometry, so the figures cannot silently
drift from what's actually shipped.

Run with: uv run python docs-figures/cryoet_specimen_filaments.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import os
from dataclasses import replace

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _render import TEAL
from specter.rotations import rotation_aligning
from specter.specimen.filament import (
    ACTIN_SPEC,
    PROTOFILAMENT_SPEC,
    FilamentInstance,
    FilamentSpec,
)
from specter.specimen.filament._path import generate_filament_path
from specter.specimen.filament._placement import filament_orientations
from specter.specimen.tomogram import TomogramSpecimenGenerator

OUT_DIR = "docs/assets/images"
# Expanded: passed through unexpanded, a literal "~" directory gets
# created wherever the script is run from, duplicating the cache.
PDB_CACHE = os.path.expanduser("~/.cache/specter/pdb")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _lay_along_x(positions_xyz: torch.Tensor) -> torch.Tensor:
    """Rigidly rotate a path so its end-to-end vector points along +x, and
    recenter it. Purely a framing convenience for the figures --
    `place_filaments` itself starts every filament in a uniformly random
    direction, which would put a path anywhere in the box."""
    end_to_end = positions_xyz[-1] - positions_xyz[0]
    rotation = rotation_aligning(end_to_end, torch.tensor([1.0, 0.0, 0.0]))
    rotated = positions_xyz @ rotation.T
    return rotated - rotated.mean(dim=0)


def _render_segment(
    spec: FilamentSpec,
    positions_xyz: torch.Tensor,
    rotations: torch.Tensor,
    voxel_size: float,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Render one filament's monomers into a box through
    ``TomogramSpecimenGenerator._render_filament_instances``, the renderer
    `_stamp_filaments` itself calls: one template per species, prepared by
    the spec's `frame` (an F-actin subunit or a tubulin dimer lifted out of
    a deposited model, or a monomer's longest principal axis aligned to
    +Z), then one rotated copy inserted per monomer.

    `positions_xyz` is box-centered here; the generator's instances are
    corner-relative, so they are shifted by half the box on the way in.
    """
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=target_shape,
        voxel_size=voxel_size,
        protein_specs=[],
        filament_specs=[spec],
        pdb_cache_dir=PDB_CACHE,
        device=DEVICE,
        progressbars=False,
    )
    extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * voxel_size
    instances = [
        FilamentInstance(
            code=spec.code,
            filament_id=0,
            monomer_index=i,
            position_xyz=position + extent_xyz / 2,
            rotation_matrix=rotation,
        )
        for i, (position, rotation) in enumerate(zip(positions_xyz, rotations))
    ]
    volume, _, _ = gen._render_filament_instances(
        torch.zeros(target_shape, device=DEVICE),
        torch.zeros(target_shape, dtype=torch.int32, device=DEVICE),
        1,
        voxel_size,
        instances,
    )
    return volume.cpu()


def figure_hero() -> None:
    """Page-top hero: a filaments-only specimen built by the real
    `TomogramSpecimenGenerator`, summed along Z."""
    voxel_size = 5.0
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=(60, 340, 340),
        voxel_size=voxel_size,
        protein_specs=[],
        filament_specs=[replace(ACTIN_SPEC, n_copies=14, n_monomers=(40, 90))],
        pdb_cache_dir=PDB_CACHE,
        seed=SEED,
        device=DEVICE,
        progressbars=False,
    )
    volume = gen.generate().cpu()
    projection = volume.sum(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    # Clipped at a high percentile: a handful of crossing points where two
    # filaments overlap in projection carry several times the density of a
    # single strand and would otherwise set the whole scale.
    ax.imshow(
        projection, cmap="gray_r", origin="lower", vmax=np.percentile(projection, 99.7)
    )
    ax.axis("off")
    plt.tight_layout(pad=0)
    path = f"{OUT_DIR}/cryoet-filament-hero.png"
    plt.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path} ({len(gen.filament_instances)} monomers)")


def figure_flex_sweep() -> None:
    """flex_deg sweep at a fixed seed: 0 is a perfectly straight rod, 3 is
    CTS's microtubule-protofilament value, 12 its actin value, 30 a
    tangled worm. Each path is the same length (same monomer count and
    axial rise), so only the curvature differs."""
    values = [0.0, 3.0, 12.0, 30.0]
    n_monomers = 90
    paths = [
        generate_filament_path(
            n_monomers, ACTIN_SPEC.step, flex_deg, generator=_generator(SEED)
        ).numpy()
        for flex_deg in values
    ]
    # One common span for every panel, so the shrinking end-to-end extent
    # at high flex_deg reads as what it is (the same contour length, more
    # tangled) rather than being zoomed away panel by panel.
    span = 0.5 * max((p.max(axis=0) - p.min(axis=0)).max() for p in paths)
    fig, axes = plt.subplots(
        1,
        len(values),
        figsize=(3.0 * len(values), 3.4),
        subplot_kw={"projection": "3d"},
    )
    for ax, flex_deg, pts in zip(axes, values, paths):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=TEAL, linewidth=2.0)
        ax.scatter(*pts[0], color="k", s=18)
        mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setter(c - span, c + span)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"flex_deg = {flex_deg:g}", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-filament-flex-sweep.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


def figure_twist() -> None:
    """Same straight chain of F-actin subunits, rendered with and without
    the helical twist. Without it every subunit presents the same face from
    the same side of the axis, so the strand reads as a smooth ridge; with
    F-actin's measured -166.6 deg per-subunit twist (a left-handed one-start
    helix) the projection picks up the crossover pattern a real F-actin
    filament shows."""
    voxel_size = 2.0
    n_monomers = 34
    shape = (56, 56, 380)
    positions = torch.zeros((n_monomers, 3))
    positions[:, 0] = (
        torch.arange(n_monomers, dtype=torch.float32) - (n_monomers - 1) / 2
    ) * ACTIN_SPEC.step

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 3.4))
    for ax, twist_deg in zip(axes, [0.0, ACTIN_SPEC.twist_deg]):
        rotations = filament_orientations(positions, twist_deg)
        volume = _render_segment(ACTIN_SPEC, positions, rotations, voxel_size, shape)
        ax.imshow(
            volume.sum(dim=0).numpy(), cmap="gray_r", origin="lower", aspect="auto"
        )
        ax.set_title(f"twist_deg = {twist_deg:g}", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-filament-twist.png"
    plt.savefig(path, dpi=190)
    plt.close(fig)
    print(f"saved {path}")


def figure_presets() -> None:
    """The two bundled presets rendered at the same scale: F-actin's 27.5 A
    rise packs monomers into a continuous strand, while a microtubule
    protofilament's 82 A tubulin-dimer repeat leaves the individual
    subunits resolvable, and its flex_deg=3 keeps it far straighter."""
    voxel_size = 4.0
    specs = [
        ("F-actin (7BT7 subunit)", ACTIN_SPEC),
        ("Microtubule protofilament (3JAL dimer)", PROTOFILAMENT_SPEC),
    ]
    shape = (40, 200, 480)
    fig, axes = plt.subplots(len(specs), 1, figsize=(9.0, 4.2))
    for ax, (title, spec) in zip(axes, specs):
        n_monomers = int(round(1600 / spec.step))
        positions = generate_filament_path(
            n_monomers, spec.step, spec.flex_deg, generator=_generator(SEED)
        )
        positions = _lay_along_x(positions)
        rotations = filament_orientations(positions, spec.twist_deg)
        volume = _render_segment(spec, positions, rotations, voxel_size, shape)
        ax.imshow(
            volume.sum(dim=0).numpy(), cmap="gray_r", origin="lower", aspect="auto"
        )
        ax.set_title(
            f"{title} -- step {spec.step:g} A, flex {spec.flex_deg:g} deg", fontsize=11
        )
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-filament-presets.png"
    plt.savefig(path, dpi=190)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_flex_sweep()
    figure_twist()
    figure_presets()
    figure_hero()
