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

import os

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
from specter.pdb import PDB
from specter.specimen.packing import (
    build_species_mask,
    draw_species_pool,
    pack_hard_spheres_3d,
    pack_shapes_3d,
)
from specter.specimen.tomogram import (
    MembraneInstance,
    TomogramSpecimenGenerator,
    TomogramProteinSpec,
)
from specter.specimen.tomogram.generator import _build_sphere_exclusion_field

OUT_DIR = "docs/assets/images"
# Expanded: passed through unexpanded, a literal "~" directory gets
# created wherever the script is run from, duplicating the cache.
PDB_CACHE = os.path.expanduser("~/.cache/specter/pdb")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHELL_COLOR = "#0f7373"
CYTOSOL_COLOR = "#d98218"
LUMEN_COLOR = "#6b4fa0"
SEED = 5

# Three vesicles, collision-rejecting auto-placed (seeded for
# reproducibility). The density/label slab used below is derived from
# where they actually landed, rather than assumed to be mid-Z.
VOXEL_SIZE = 6.0
SHAPE_ZYX = (120, 300, 300)


def _membrane_instances() -> list[MembraneInstance]:
    return [
        MembraneInstance(
            generator=MembraneGenerator(
                voxel_size=VOXEL_SIZE,
                shape_backend="spherical_harmonics",
                sh_axes_range=(160.0, 200.0),
                bilayer_thickness=30.0,
                device=DEVICE,
                seed=SEED + i,
            ),
        )
        for i in range(3)
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
        # All-False (not just Z): a membrane instance clipped against an
        # X/Y wall has its shell broken there, which can disconnect its
        # lumen from being an enclosed compartment at all -- this figure's
        # whole point is showing an intact lumen, so every instance must
        # fit fully inside the box on every axis.
        clip_axes=(False, False, False),
        progressbars=False,
    )
    volume = gen.generate().cpu()
    labels = gen.instance_labels.cpu()
    shell = gen.regions["shell"].cpu()
    membrane = gen.membrane_labels.cpu()

    # Slab spanning wherever the (auto-placed) vesicles actually landed in
    # Z, rather than an assumed mid-Z window -- a thin slab still catches
    # their lumens as cross-sections rather than collapsing each vesicle
    # into a filled disk under a full-depth projection.
    z_present = torch.nonzero((membrane > 0).any(dim=2).any(dim=1)).flatten()
    z0, z1 = int(z_present.min()), int(z_present.max()) + 1
    density = volume[z0:z1].sum(dim=0).numpy()

    lumen_ids = torch.tensor(
        [p.instance_id for p in gen.placements if p.location == "lumen"]
    )
    slab_labels = labels[z0:z1]
    # White background, matching the inverted-grey density panel beside it.
    painted = np.ones(density.shape + (3,), dtype=float)
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
    axes[0].imshow(density, cmap="gray_r", origin="lower")
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


def figure_backends() -> None:
    """The choice that dominates achievable density: what each backend
    collides. Real species, one box, sweeping the requested budget. Volume
    fraction is measured as placed molecular mass over box volume, the
    same quantity crowding concentrations are quoted in, so the
    physiological band and CryoTomoSim's own output can sit on the same
    axis."""
    codes = ["1S3X", "1TUB", "1A1S", "1MBO", "1C3W", "1FA2"]
    voxel, box_a = 5.0, (400.0, 400.0, 400.0)
    grid = tuple(int(round(b / voxel)) for b in box_a)
    box_volume = float(np.prod(box_a))
    da_per_a3 = 1.35e-24 / 1.66054e-24
    mass_per_z = {6: 12.011, 7: 14.007, 8: 15.999, 15: 30.974, 16: 32.06}
    implicit_h = {6: 1.3, 7: 1.1, 8: 0.2, 16: 0.6}

    pdbs, masses, radii = [], [], []
    for code in codes:
        pdb = PDB(code, pdb_cache_dir=PDB_CACHE, verbose=False)
        z = pdb.atomic_numbers.numpy().astype(int)
        masses.append(
            sum(
                int((z == k).sum()) * (mass_per_z[k] + implicit_h.get(k, 0) * 1.008)
                for k in np.unique(z)
                if k in mass_per_z
            )
        )
        radii.append(float(pdb.max_diameter) / 2.0)
        pdbs.append(pdb)

    masks = [build_species_mask(p.coordinates, voxel, gap=0.0) for p in pdbs]
    mask_volumes = torch.tensor([float(m.sum()) * voxel**3 for m in masks])
    radii_t, ratios = torch.tensor(radii), torch.ones(len(pdbs))
    requested = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    def volume_fraction(species_of_accepted) -> float:
        placed = float(sum(masses[int(i)] for i in species_of_accepted))
        return placed / box_volume / da_per_a3

    sphere, shape = [], []
    for fraction in requested:
        pool_r, pool_s = draw_species_pool(
            radii_t, ratios, fraction, box_volume, seed=SEED
        )
        _, accepted = pack_hard_spheres_3d(pool_r, box_a, gap=0.0, seed=SEED)
        sphere.append(volume_fraction(pool_s[accepted]))

        _, pool_s2 = draw_species_pool(
            radii_t,
            ratios,
            fraction,
            box_volume,
            seed=SEED,
            species_volumes=mask_volumes,
        )
        _, _, accepted2, _ = pack_shapes_3d(
            masks, pool_s2, grid, voxel, seed=SEED, n_orientations=128
        )
        shape.append(volume_fraction(pool_s2[accepted2]))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhspan(0.20, 0.30, color="0.88", zorder=0)
    ax.text(0.035, 0.252, "crowded cytoplasm", fontsize=8, color="0.35")
    ax.axhline(0.240, ls=":", lw=1.2, color="0.45")
    ax.text(0.60, 0.213, "CryoTomoSim (0.240)", fontsize=8, color="0.45")
    ax.plot(requested, shape, "o-", color=SHELL_COLOR, label='packing_backend="shape"')
    ax.plot(
        requested, sphere, "o-", color=CYTOSOL_COLOR, label='packing_backend="sphere"'
    )
    ax.set_xlabel("occupancy_fraction (requested)")
    ax.set_ylabel("achieved macromolecule volume fraction")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(0.32, max(shape) * 1.15))
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        "What gets collided sets the ceiling\n"
        "six species at 5 A; absolute values move with the species mix",
        fontsize=10,
    )
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-backends.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_shape_jamming() -> None:
    """The shape backend jams too, and polydispersity still raises the
    ceiling -- the same mechanism the sphere curve shows, measured in
    macromolecule volume fraction on real molecules rather than in
    bare-sphere occupancy on synthetic radii."""
    voxel, box_a = 4.0, (280.0, 280.0, 280.0)
    grid = tuple(int(round(b / voxel)) for b in box_a)
    box_volume = float(np.prod(box_a))
    da_per_a3 = 1.35e-24 / 1.66054e-24
    mass_per_z = {6: 12.011, 7: 14.007, 8: 15.999, 15: 30.974, 16: 32.06}
    implicit_h = {6: 1.3, 7: 1.1, 8: 0.2, 16: 0.6}

    def load(codes):
        pdbs, masses = [], []
        for code in codes:
            pdb = PDB(code, pdb_cache_dir=PDB_CACHE, verbose=False)
            z = pdb.atomic_numbers.numpy().astype(int)
            masses.append(
                sum(
                    int((z == k).sum()) * (mass_per_z[k] + implicit_h.get(k, 0) * 1.008)
                    for k in np.unique(z)
                    if k in mass_per_z
                )
            )
            pdbs.append(pdb)
        return pdbs, masses

    pools = {
        "monodisperse (1MBO only)": ["1MBO"],
        "polydisperse (5 species)": ["1C3W", "1MBO", "1S3X", "1TUB", "1A1S"],
    }
    requested = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    curves = {}
    for label, codes in pools.items():
        pdbs, masses = load(codes)
        masks = [build_species_mask(p.coordinates, voxel, gap=0.0) for p in pdbs]
        volumes = torch.tensor([float(m.sum()) * voxel**3 for m in masks])
        radii = torch.tensor([float(p.max_diameter) / 2.0 for p in pdbs])
        ratios = torch.ones(len(pdbs))
        achieved = []
        for fraction in requested:
            _, pool_s = draw_species_pool(
                radii,
                ratios,
                fraction,
                box_volume,
                seed=SEED,
                species_volumes=volumes,
            )
            _, _, accepted, _ = pack_shapes_3d(
                masks, pool_s, grid, voxel, seed=SEED, n_orientations=128
            )
            placed = float(sum(masses[int(i)] for i in pool_s[accepted]))
            achieved.append(placed / box_volume / da_per_a3)
        curves[label] = achieved

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for (label, achieved), color in zip(curves.items(), [CYTOSOL_COLOR, SHELL_COLOR]):
        ax.plot(requested, achieved, "o-", color=color, label=label)
    ax.axhspan(0.20, 0.30, color="0.88", zorder=0)
    ax.text(0.035, 0.283, "crowded cytoplasm", fontsize=8, color="0.35")
    ax.set_xlabel("occupancy_fraction (requested)")
    ax.set_ylabel("achieved macromolecule volume fraction")
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Shape packing saturates later, and higher", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-shape-jamming.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_occupancy_grid() -> None:
    """The shape backend's answer to obstacles and regions: one boolean
    grid. Everything forbidden is stamped into it -- the region's own
    complement, the membrane, and every instance already placed -- and a
    candidate is rejected if its rotated footprint meets any set voxel. No
    distance field is involved."""
    from specter.specimen.filament import FilamentSpec

    # Filaments earn their place here: with only a membrane, the shell is
    # already outside the cytosol region, so "stamp the obstacles in" would
    # be a step that visibly does nothing.
    filaments = [
        FilamentSpec(
            code="1J6Z",
            step=27.3,
            flex_deg=12.0,
            twist_deg=166.15,
            n_copies=6,
            n_monomers=(30, 50),
        ),
    ]
    common = dict(
        target_shape=SHAPE_ZYX,
        voxel_size=VOXEL_SIZE,
        filament_specs=filaments,
        packing_backend="shape",
        pdb_cache_dir=PDB_CACHE,
        seed=SEED,
        device=DEVICE,
        progressbars=False,
    )

    # Two runs at one seed: the obstacles alone, then the same specimen
    # with protein fill. Deriving the obstacles by subtracting labels from
    # the full run does not work -- an instance label is thresholded at 1%
    # of its template's peak, so a placed protein's soft tail falls outside
    # its own label and would be counted as an obstacle.
    obstacles_only = TomogramSpecimenGenerator(
        membrane_instances=_membrane_instances()[:1], protein_specs=[], **common
    )
    obstacle_volume = np.asarray(obstacles_only.generate().cpu())
    # Relative threshold, not > 0: the membrane field carries a small
    # nonzero value across its own bounding box, which a bare > 0 renders
    # as a square slab.
    obstacles = obstacle_volume > 0.01 * obstacle_volume.max()
    cytosol = np.asarray(obstacles_only.regions["cytosol"].cpu(), dtype=bool)

    full = TomogramSpecimenGenerator(
        membrane_instances=_membrane_instances()[:1],
        protein_specs=[TomogramProteinSpec(pdb_source="1MBO", location="cytosol")],
        occupancy_fraction=0.35,
        **common,
    )
    full.generate()
    labels = (full.instance_labels > 0).cpu().numpy()
    assert np.array_equal(
        cytosol, np.asarray(full.regions["cytosol"].cpu(), dtype=bool)
    ), "the two runs must share a specimen for the panels to compose"

    z = SHAPE_ZYX[0] // 2
    panels = [
        (~cytosol, "seeded: the region's own complement"),
        (~cytosol | obstacles, "+ membrane and filaments stamped in"),
        (~cytosol | obstacles | labels, "+ every protein placed"),
    ]
    cmap = ListedColormap(["white", SHELL_COLOR])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel[z], cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "One occupancy grid, no distance field: a candidate is rejected if "
        "its footprint meets any set voxel",
        fontsize=10,
    )
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-packing-occupancy-grid.png"
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
    figure_backends()
    figure_shape_jamming()
    figure_occupancy_grid()
    figure_staging()
    figure_filler_tables()
    figure_exclusion_field()
    figure_regions_hero()
