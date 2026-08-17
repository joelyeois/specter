"""
Generate the figures for docs/concepts/cryoet-specimen/index.md, the
overview of `specter build tomogram`'s specimen assembly
(``specter.specimen.tomogram.TomogramSpecimenGenerator``).

Runs the real generator once with every component switched on -- carbon
film, membranes with transmembrane proteins, filaments, microtubules, gold
fiducial beads, exact-count targets and ratio-weighted filler -- and draws
every figure from that single run's own outputs (`volume`, `regions`,
`membrane_labels`, `instance_labels`, `placements`, `bead_instances`).

Run with: uv run python docs-figures/cryoet_specimen_overview.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specter.specimen._carbon import CarbonFilmSpec
from specter.specimen.cytosolic_filler import (
    PEI2016_CROWDING_TABLE,
    build_filler_pool_specs,
)
from specter.specimen.filament import ACTIN_SPEC, FilamentSpec, MicrotubuleSpec
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.tomogram import (
    MembraneInstance,
    TomogramSpecimenGenerator,
    TomogramBeadSpec,
    TomogramProteinSpec,
)

OUT_DIR = "docs/assets/images"
PDB_CACHE = "specter-data/pdb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 11

VOXEL_SIZE = 8.0
SHAPE_ZYX = (110, 440, 440)  # 880 x 3520 x 3520 Angstrom

CARBON_COLOR = "#8c8c8c"
MEMBRANE_COLOR = "#0f7373"
FILAMENT_COLOR = "#c0392b"
MICROTUBULE_COLOR = "#4C72B0"
BEAD_COLOR = "#b8860b"
TARGET_COLOR = "#d98218"
FILLER_COLOR = "#6b4fa0"


def _membrane_instances() -> list[MembraneInstance]:
    positions = [(-950.0, -800.0, 0.0), (700.0, 450.0, 0.0), (-250.0, 1050.0, 0.0)]
    return [
        MembraneInstance(
            generator=MembraneGenerator(
                voxel_size=VOXEL_SIZE,
                shape_backend="spherical_harmonics",
                sh_axes_range=(220.0, 300.0),
                membrane_scale_range=(1.0, 1.0),
                bilayer_thickness=30.0,
                transmembrane_specs=[
                    TransmembraneSpec(pdb_source="1C3W", frequency=25)
                ],
                pdb_cache_dir=PDB_CACHE,
                device=DEVICE,
                seed=SEED + i,
            ),
            position_xyz=position,
        )
        for i, position in enumerate(positions)
    ]


def build() -> tuple[TomogramSpecimenGenerator, torch.Tensor]:
    filler = [
        TomogramProteinSpec(entry["pdb_source"], location="cytosol", ratio=1.0)
        for entry in build_filler_pool_specs(PEI2016_CROWDING_TABLE, min_mw_kda=100.0)
    ]
    gen = TomogramSpecimenGenerator(
        membrane_instances=_membrane_instances(),
        target_shape=SHAPE_ZYX,
        voxel_size=VOXEL_SIZE,
        protein_specs=[
            TomogramProteinSpec("6QZP", location="cytosol", n_copies=25),
            TomogramProteinSpec("7VD8", location="cytosol", n_copies=25),
            TomogramProteinSpec("1BXN", location="lumen", ratio=1.0),
            *filler,
        ],
        filament_specs=[
            FilamentSpec(
                code=ACTIN_SPEC.code,
                step=ACTIN_SPEC.step,
                flex_deg=ACTIN_SPEC.flex_deg,
                twist_deg=ACTIN_SPEC.twist_deg,
                n_copies=10,
                n_monomers=(40, 90),
            )
        ],
        microtubule_specs=[
            MicrotubuleSpec(
                n_copies=2,
                # length=None spans the volume's own diagonal, so each tube
                # crosses the whole field of view rather than appearing as
                # a stub. bend_radius=3e4 (3 um) gives the smooth,
                # mechanically-constrained curvature real microtubules show
                # in cellular tomograms, instead of the near-straight
                # default thermal walk.
                bend_radius=3.0e4,
            )
        ],
        carbon_film_spec=CarbonFilmSpec(
            thickness=150.0, edge_fraction=0.08, edge_side="bottom"
        ),
        bead_specs=[TomogramBeadSpec(radius=[80.0, 120.0], count=8)],
        bead_roughness=[0.0, 0.2],
        occupancy_fraction=0.35,
        clip_axes=(False, True, True),
        pdb_cache_dir=PDB_CACHE,
        seed=SEED,
        device=DEVICE,
        render_workers="auto",
        accumulator_device="auto",
        chunk_size=64,
        progressbars=False,
    )
    volume = gen.generate().cpu()
    return gen, volume


def _component_masks(gen: TomogramSpecimenGenerator) -> dict[str, np.ndarray]:
    """Split the run's own ground truth into one 3D boolean mask per
    component. Instance ids are handed out in placement order -- filament
    monomers first (one id each), then microtubules (one id per tube), then
    beads, then cytosol/lumen proteins -- so the protein and bead ids the
    generator reports back, plus the filament/microtubule instance counts,
    are enough to identify every remaining labelled voxel."""
    labels = gen.instance_labels.cpu()
    membrane = gen.membrane_labels.cpu() > 0
    shell = gen.regions["shell"].cpu()

    n_filament_ids = len(gen.filament_instances)
    n_microtubule_ids = len(gen.microtubule_instances)
    filament_ids = torch.arange(1, 1 + n_filament_ids)
    microtubule_ids = torch.arange(
        1 + n_filament_ids, 1 + n_filament_ids + n_microtubule_ids
    )
    bead_ids = torch.tensor([b.instance_id for b in gen.bead_instances])
    target_ids = torch.tensor(
        [p.instance_id for p in gen.placements if p.role == "target"]
    )
    filler_ids = torch.tensor(
        [p.instance_id for p in gen.placements if p.role == "filler"]
    )

    return {
        # Carbon reads as "shell" to the region classifier, the same bucket
        # a bilayer occupies -- everything shell-classified that is not a
        # labelled membrane instance is film.
        "carbon film": (shell & ~membrane).numpy(),
        "membrane": membrane.numpy(),
        "filament": torch.isin(labels, filament_ids).numpy(),
        "microtubule": torch.isin(labels, microtubule_ids).numpy(),
        "gold bead": torch.isin(labels, bead_ids).numpy(),
        "target protein": torch.isin(labels, target_ids).numpy(),
        "filler protein": torch.isin(labels, filler_ids).numpy(),
    }


def figure_hero(volume: torch.Tensor) -> None:
    """Page-top hero: the whole specimen, summed along Z."""
    projection = volume.sum(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(
        projection,
        cmap="gray_r",
        origin="lower",
        vmax=np.percentile(projection, 99.5),
    )
    ax.axis("off")
    plt.tight_layout(pad=0)
    path = f"{OUT_DIR}/cryoet-tomogram-hero.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    print(f"saved {path}")


def figure_components(gen: TomogramSpecimenGenerator) -> None:
    """The same specimen, painted by component from the run's own
    ground-truth label volumes -- nothing here is inferred from the
    density."""
    masks = _component_masks(gen)
    colors = {
        "carbon film": CARBON_COLOR,
        "filler protein": FILLER_COLOR,
        "target protein": TARGET_COLOR,
        "filament": FILAMENT_COLOR,
        "microtubule": MICROTUBULE_COLOR,
        "membrane": MEMBRANE_COLOR,
        "gold bead": BEAD_COLOR,
    }
    # A thin mid-Z slab, not the full-depth projection: at this crowding a
    # projection through the whole 880 A of specimen is solid filler, and
    # every membrane collapses from a ring to a filled disk.
    z = SHAPE_ZYX[0] // 2
    shape = tuple(gen.target_shape[1:])
    painted = np.ones(shape + (3,))
    for name, color in colors.items():
        painted[masks[name][z - 1 : z + 2].any(axis=0)] = matplotlib.colors.to_rgb(
            color
        )

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.imshow(painted, origin="lower")
    ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=c, label=n)
        for n, c in colors.items()
    ]
    ax.legend(
        handles=handles,
        fontsize=9,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-tomogram-components.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved {path}")


def figure_ground_truth(gen: TomogramSpecimenGenerator) -> None:
    """What a run writes out alongside the density: the region map, the
    per-instance membrane labels, and the per-instance protein labels."""
    z = SHAPE_ZYX[0] // 2
    regions = gen.regions
    region_map = torch.zeros(SHAPE_ZYX, dtype=torch.int32)
    region_map[regions["shell"].cpu()] = 1
    region_map[regions["lumen"].cpu()] = 2

    def _unlabelled_masked(labels: torch.Tensor) -> np.ndarray:
        """Draw id 0 ("nothing here") as background rather than as a
        colour, so a label volume doesn't read as one huge instance."""
        data = labels.cpu()[z].numpy()
        return np.ma.masked_where(data == 0, data)

    panels = [
        (region_map[z].numpy(), "regions (0 cytosol / 1 shell / 2 lumen)", "viridis"),
        (_unlabelled_masked(gen.membrane_labels), "membrane instance labels", "tab10"),
        (
            _unlabelled_masked(gen.instance_labels),
            "protein/filament/bead labels",
            "nipy_spectral",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    for ax, (data, title, cmap) in zip(axes, panels):
        colormap = matplotlib.colormaps[cmap].with_extremes(bad="white")
        ax.imshow(data, cmap=colormap, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-tomogram-ground-truth.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    generator, volume = build()
    print(
        f"{len(generator.placements)} protein instances, "
        f"{len(generator.filament_instances)} filament monomers, "
        f"{len(generator.microtubule_instances)} microtubules, "
        f"{len(generator.bead_instances)} beads"
    )
    figure_hero(volume)
    figure_components(generator)
    figure_ground_truth(generator)
