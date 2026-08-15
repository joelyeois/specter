"""
Generate the figures for docs/concepts/cryoet-specimen/packing.md, which
explains region classification and the RSA hard-sphere packing behind
`specter build tomogram`'s protein fill (``specter.specimen.packing``,
``specter.specimen.tomogram._regions``,
``specter.specimen.cytosolic_filler``).

Calls the shipped `pack_hard_spheres_3d`/`draw_species_pool`/
`classify_membrane_regions` and, for the hero, a real
`TomogramSpecimenGenerator` run, rather than reimplementing any of the
placement logic.

Run with: uv run python docs-figures/cryoet_specimen_packing.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from specter.specimen.cytosolic_filler import (
    CRYOETSIM_PARTICLE_TABLE,
    PEI2016_CROWDING_TABLE,
)
from specter.specimen.membrane import MembraneGenerator
from specter.specimen.packing import draw_species_pool, pack_hard_spheres_3d
from specter.specimen.tomogram import (
    MembraneInstance,
    TomogramSpecimenGenerator,
    TomogramProteinSpec,
)
from specter.specimen.tomogram.generator import _build_sphere_exclusion_field

OUT_DIR = "docs/assets/images"
PDB_CACHE = "specter-data/pdb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHELL_COLOR = "#0f7373"
CYTOSOL_COLOR = "#d98218"
LUMEN_COLOR = "#6b4fa0"
SEED = 5

# Three vesicles, placed by hand rather than auto-packed, purely so the
# figure is reproducible and all three sit at mid-Z where a single slab
# projection catches their lumens.
VOXEL_SIZE = 6.0
SHAPE_ZYX = (120, 300, 300)
MEMBRANE_POSITIONS = [(-520.0, -430.0, 0.0), (430.0, 300.0, 0.0), (-120.0, 620.0, 0.0)]


def _membrane_instances() -> list[MembraneInstance]:
    return [
        MembraneInstance(
            generator=MembraneGenerator(
                voxel_size=VOXEL_SIZE,
                shape_backend="spherical_harmonics",
                sh_axes_range=(160.0, 200.0),
                membrane_scale_range=(1.0, 1.0),
                bilayer_thickness=30.0,
                device=DEVICE,
                seed=SEED + i,
            ),
            position_xyz=position,
        )
        for i, position in enumerate(MEMBRANE_POSITIONS)
    ]


def figure_regions_hero() -> None:
    """Page-top hero: a real region-gated run. Left, the density; right,
    the same slab painted by where each placed instance was allowed to go
    -- the cytosol species never enters a vesicle, the lumen species never
    leaves one, and neither ever overlaps the bilayer shell."""
    gen = TomogramSpecimenGenerator(
        membrane_instances=_membrane_instances(),
        target_shape=SHAPE_ZYX,
        voxel_size=VOXEL_SIZE,
        protein_specs=[
            TomogramProteinSpec("6QZP", location="cytosol", ratio=1.0),
            TomogramProteinSpec("1BXN", location="lumen", ratio=1.0),
        ],
        occupancy_fraction=0.3,
        pdb_cache_dir=PDB_CACHE,
        seed=SEED,
        device=DEVICE,
        clip_axes=(False, True, True),
        progressbars=False,
    )
    volume = gen.generate().cpu()
    labels = gen.instance_labels.cpu()
    shell = gen.regions["shell"].cpu()

    z0, z1 = SHAPE_ZYX[0] // 2 - 12, SHAPE_ZYX[0] // 2 + 12
    density = volume[z0:z1].sum(dim=0).numpy()

    lumen_ids = torch.tensor(
        [p.instance_id for p in gen.placements if p.location == "lumen"]
    )
    slab_labels = labels[z0:z1]
    painted = np.zeros(density.shape + (3,), dtype=float)
    shell_2d = shell[z0:z1].any(dim=0).numpy()
    protein_2d = (slab_labels > 0).any(dim=0).numpy()
    lumen_2d = torch.isin(slab_labels, lumen_ids).any(dim=0).numpy()
    for mask, color in [
        (shell_2d, SHELL_COLOR),
        (protein_2d & ~lumen_2d, CYTOSOL_COLOR),
        (lumen_2d, LUMEN_COLOR),
    ]:
        painted[mask] = matplotlib.colors.to_rgb(color)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    axes[0].imshow(density, cmap="magma", origin="lower")
    axes[0].set_title("density (24-voxel slab, summed)", fontsize=11)
    axes[1].imshow(painted, origin="lower")
    axes[1].set_title("ground-truth labels", fontsize=11)
    for ax in axes:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=c, label=label)
        for c, label in [
            (SHELL_COLOR, "membrane shell"),
            (CYTOSOL_COLOR, "cytosol species (6QZP)"),
            (LUMEN_COLOR, "lumen species (1BXN)"),
        ]
    ]
    axes[1].legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-hero.png"
    plt.savefig(path, dpi=190)
    plt.close(fig)
    print(f"saved {path}: {len(gen.placements)} instances")


def _achieved_occupancy(
    radii: torch.Tensor, box: tuple[float, float, float], gap: float
) -> float:
    coords, accepted = pack_hard_spheres_3d(radii, box, gap=gap, seed=SEED)
    volume = float((4 / 3) * np.pi * (radii[accepted] ** 3).sum())
    return volume / (box[0] * box[1] * box[2])


def figure_rsa_limit() -> None:
    """RSA jams: past a certain requested occupancy, drawing a bigger
    candidate pool stops adding placed volume. The ceiling is higher for a
    polydisperse pool (small spheres fit into gaps large ones leave) than a
    monodisperse one -- which is why `filler_occupancy_fraction` is a
    budget rather than a promise, and why raising it past ~0.5 does
    nothing."""
    box = (2000.0, 2000.0, 2000.0)
    box_volume = box[0] * box[1] * box[2]
    requested = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    mono_radii = torch.tensor([60.0])
    poly_radii = torch.tensor([40.0, 60.0, 90.0, 130.0])
    ratios = torch.ones(len(poly_radii))

    curves = {}
    for name, species_radii, species_ratios in [
        ("monodisperse (r = 60 A)", mono_radii, torch.ones(1)),
        ("polydisperse (r = 40-130 A)", poly_radii, ratios),
    ]:
        achieved = []
        for fraction in requested:
            radii, _ = draw_species_pool(
                species_radii, species_ratios, fraction, box_volume, seed=SEED
            )
            achieved.append(_achieved_occupancy(radii, box, gap=5.0))
        curves[name] = achieved

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for (name, achieved), color in zip(curves.items(), [SHELL_COLOR, CYTOSOL_COLOR]):
        ax.plot(requested, achieved, "o-", color=color, label=name)
    ax.plot(
        [0, 1], [0, 1], "--", color="0.6", linewidth=1, label="requested = achieved"
    )
    ax.set_xlabel("occupancy_fraction (requested)")
    ax.set_ylabel("achieved bare-sphere occupancy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=9)
    ax.set_title("RSA saturates well below the requested budget", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-rsa-limit.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_staging() -> None:
    """Placement runs largest-radius-first, in stages. Acceptance rate
    still falls with radius -- a big sphere has more potential conflict
    partners -- but every species gets its crack at the box while it is
    still mostly empty, instead of competing against an already-placed
    cloud of small ones."""
    box = (2000.0, 2000.0, 2000.0)
    species_radii = torch.tensor([40.0, 60.0, 90.0, 130.0])
    ratios = torch.ones(len(species_radii))
    radii, species_idx = draw_species_pool(
        species_radii, ratios, 0.5, box[0] * box[1] * box[2], seed=SEED
    )
    _, accepted = pack_hard_spheres_3d(radii, box, gap=5.0, seed=SEED)

    drawn = np.bincount(species_idx.numpy(), minlength=len(species_radii))
    placed = np.bincount(species_idx[accepted].numpy(), minlength=len(species_radii))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.arange(len(species_radii))
    axes[0].bar(x - 0.2, drawn, width=0.4, color="0.75", label="drawn")
    axes[0].bar(x + 0.2, placed, width=0.4, color=SHELL_COLOR, label="placed")
    axes[0].set_xticks(x, [f"{r:.0f} A" for r in species_radii])
    axes[0].set_ylabel("instances")
    axes[0].legend(fontsize=9)
    axes[0].set_title("candidate pool vs. accepted", fontsize=11)

    axes[1].bar(x, placed / np.maximum(drawn, 1), color=CYTOSOL_COLOR, width=0.55)
    axes[1].set_xticks(x, [f"{r:.0f} A" for r in species_radii])
    axes[1].set_ylabel("acceptance rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("acceptance rate by species radius", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-staging.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_exclusion_field() -> None:
    """One mechanism covers both obstacle avoidance and region
    restriction: a distance field to the nearest forbidden voxel. A
    candidate is rejected unless the field at its center exceeds its own
    radius plus `gap`, so the whole sphere clears the forbidden region --
    here, the membrane shell plus everything already placed."""
    membrane = MembraneGenerator(
        voxel_size=VOXEL_SIZE,
        shape_backend="spherical_harmonics",
        sh_axes_range=(160.0, 200.0),
        membrane_scale_range=(1.0, 1.0),
        bilayer_thickness=30.0,
        device=DEVICE,
        seed=SEED,
    )
    density = membrane.generate().cpu()
    # Padded into a roomier box so there is real cytosol around the vesicle
    # to place into -- the membrane's own auto-sized local grid hugs it.
    pad = 45
    density = torch.nn.functional.pad(density, (pad,) * 6)
    shape = tuple(density.shape)

    from specter.specimen.tomogram._regions import classify_membrane_regions

    regions = classify_membrane_regions(density)
    from scipy import ndimage

    shell_field = ndimage.distance_transform_edt(
        ~regions["shell"].numpy(), sampling=(VOXEL_SIZE,) * 3
    )
    mid = shape[0] // 2

    radius, gap = 90.0, 5.0
    box = tuple(s * VOXEL_SIZE for s in shape)
    coords, accepted = pack_hard_spheres_3d(
        torch.full((25,), radius),
        box,
        gap=gap,
        seed=SEED,
        exclusion_distance_field=torch.from_numpy(shell_field).float(),
        field_voxel_size=VOXEL_SIZE,
        sampling_mask=regions["cytosol"],
    )
    placed_field = _build_sphere_exclusion_field(
        coords, torch.full((coords.shape[0],), radius), shape, VOXEL_SIZE
    ).numpy()
    combined = np.minimum(shell_field, placed_field)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    titles = [
        "distance to the membrane shell",
        f"after placing {coords.shape[0]} cytosol spheres",
        "combined (elementwise minimum)",
    ]
    rejected_cmap = ListedColormap([(0.85, 0.1, 0.1, 0.55)])
    for ax, data, title in zip(axes, [shell_field, placed_field, combined], titles):
        slice_ = data[mid]
        im = ax.imshow(slice_, cmap="viridis", origin="lower", vmin=0, vmax=200)
        ax.imshow(
            np.ma.masked_where(slice_ >= radius + gap, np.ones_like(slice_)),
            cmap=rejected_cmap,
            origin="lower",
        )
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="distance (A)")
    fig.suptitle(
        f"red: centres rejected for a {radius:.0f} A sphere "
        f"(field < radius + gap = {radius + gap:.0f} A)",
        fontsize=10,
        y=0.04,
    )
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-exclusion-field.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_filler_tables() -> None:
    """The two bundled reference tables, by mass. PEI2016 carries the
    source paper's own relative-abundance weighting; the CryoETSim table is
    broader and categorised."""
    pei_mw = [e["mw_kda"] for e in PEI2016_CROWDING_TABLE]
    pei_freq = [e["occurrence_freq"] for e in PEI2016_CROWDING_TABLE]
    cts_mw = [e["mw_kda"] for e in CRYOETSIM_PARTICLE_TABLE]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    axes[0].scatter(pei_mw, pei_freq, color=SHELL_COLOR, s=32)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("molecular weight (kDa)")
    axes[0].set_ylabel("occurrence_freq (relative)")
    axes[0].set_title(f"PEI2016_CROWDING_TABLE ({len(pei_mw)} species)", fontsize=11)

    bins = np.logspace(
        np.log10(min(cts_mw + pei_mw)), np.log10(max(cts_mw + pei_mw)), 22
    )
    axes[1].hist(pei_mw, bins=bins, alpha=0.75, color=SHELL_COLOR, label="PEI2016")
    axes[1].hist(cts_mw, bins=bins, alpha=0.65, color=CYTOSOL_COLOR, label="CryoETSim")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("molecular weight (kDa)")
    axes[1].set_ylabel("species")
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"mass coverage (CryoETSim: {len(cts_mw)} species)", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-filler-tables.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_rsa_limit()
    figure_staging()
    figure_filler_tables()
    figure_exclusion_field()
    figure_regions_hero()
