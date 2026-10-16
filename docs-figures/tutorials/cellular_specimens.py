"""
Figures and printed output for docs/tutorials/cellular-specimens.md.

Run with: uv run python docs-figures/tutorials/cellular_specimens.py
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from _common import PALETTE, Transcript, as_numpy, save, show_image

transcript = Transcript("cellular_specimens")

transcript.start("setup")
# --8<-- [start:setup]
from dataclasses import replace

from specter.specimen import (
    PEI2016_CROWDING_TABLE,
    CarbonFilmSpec,
    build_filler_pool_specs,
)
from specter.specimen.filament import ACTIN_SPEC, MicrotubuleSpec
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.tomogram import (
    MembraneInstance,
    TomogramBeadSpec,
    TomogramProteinSpec,
    TomogramSpecimenGenerator,
)

voxel_size = 8.0  # Å
common = dict(
    target_shape=(96, 384, 384),  # (Z, Y, X): 768 x 3072 x 3072 Å
    voxel_size=voxel_size,
    clip_axes=(False, True, True),  # instances may run off the field's edges
    seed=3,
    device="cuda",
    progressbars=False,
)
# --8<-- [end:setup]

transcript.start("membranes")


# --8<-- [start:membranes]
def membranes() -> list[MembraneInstance]:
    """Three vesicles, each studded with bacteriorhodopsin."""
    return [
        MembraneInstance(
            MembraneGenerator(
                voxel_size=voxel_size,
                sh_axes_range=(200.0, 320.0),  # semi-axes, Å
                bilayer_thickness=37.0,
                transmembrane_specs=[TransmembraneSpec("1C3W", frequency=20)],
                device="cuda",
                seed=i,
            )
        )
        for i in range(3)
    ]


carbon = CarbonFilmSpec(edge_fraction=0.1, edge_side="bottom")

gen = TomogramSpecimenGenerator(
    membranes(), protein_specs=[], carbon_film_spec=carbon, **common
)
volumes = {"membranes, carbon": gen.generate().cpu()}
print(f"vesicles placed: {len(gen.placed_membrane_instances)} of 3")
print(f"transmembrane proteins: {len(gen.transmembrane_placements)}")
# --8<-- [end:membranes]

transcript.start("cytoskeleton")
# --8<-- [start:cytoskeleton]
cytoskeleton = dict(
    filament_specs=[replace(ACTIN_SPEC, n_copies=8, n_monomers=(40, 90))],
    microtubule_specs=[MicrotubuleSpec(n_copies=2)],
)
gen = TomogramSpecimenGenerator(
    membranes(), protein_specs=[], carbon_film_spec=carbon, **cytoskeleton, **common
)
volumes["+ actin, microtubules"] = gen.generate().cpu()

n_filaments = len({f.filament_id for f in gen.filament_instances})
print(f"actin: {n_filaments} filaments, {len(gen.filament_instances)} monomers")
print(
    f"microtubules: {len(gen.microtubule_instances)} tubes, "
    f"{len(gen.microtubule_dimer_instances)} tubulin dimers"
)
# --8<-- [end:cytoskeleton]

transcript.start("beads")
# --8<-- [start:beads]
fiducials = dict(bead_specs=[TomogramBeadSpec(radius=[80.0, 110.0], count=6)])
gen = TomogramSpecimenGenerator(
    membranes(),
    protein_specs=[],
    carbon_film_spec=carbon,
    **cytoskeleton,
    **fiducials,
    **common,
)
volumes["+ gold beads"] = gen.generate().cpu()
radii = sorted(round(b.radius) for b in gen.bead_instances)
print(f"gold beads: {len(gen.bead_instances)}, radii {radii} Å")
# --8<-- [end:beads]

transcript.start("proteins")
# --8<-- [start:proteins]
targets = [
    TomogramProteinSpec("6QZP", n_copies=12),  # ribosome
    TomogramProteinSpec("7VD8", n_copies=12),  # apoferritin
]
filler = [
    TomogramProteinSpec(entry["pdb_source"])  # location="cytosol", ratio=1
    for entry in build_filler_pool_specs(PEI2016_CROWDING_TABLE, min_mw_kda=300.0)
]
lumen = [TomogramProteinSpec("1BXN", location="lumen")]  # RuBisCO

gen = TomogramSpecimenGenerator(
    membranes(),
    protein_specs=targets + filler + lumen,
    carbon_film_spec=carbon,
    occupancy_fraction=0.35,
    **cytoskeleton,
    **fiducials,
    **common,
)
volume = gen.generate().cpu()
volumes["+ proteins"] = volume

roles = Counter((p.role, p.location) for p in gen.placements)
species = Counter(p.species_id for p in gen.placements if p.role == "target")
print(f"targets: {dict(species)}")
print(f"filler: {roles['filler', 'cytosol']} in the cytosol from {len(filler)} species")
print(f"filler: {roles['filler', 'lumen']} in the vesicle lumens")
# --8<-- [end:proteins]

titles = list(volumes)
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.95))
for ax, title in zip(axes, titles):
    projection = as_numpy(volumes[title].sum(0)) * voxel_size
    show_image(
        ax,
        projection,
        cmap="gray_r",
        title=title,
        vmin=float(np.percentile(projection, 0.5)),
        vmax=float(np.percentile(projection, 99.7)),
        pixel_size=voxel_size,
        scalebar=500,
    )
fig.tight_layout()
save(fig, "cellular_specimens-buildup")

transcript.start("ground-truth")
# --8<-- [start:ground-truth]
regions = {name: mask.cpu() for name, mask in gen.regions.items()}
for name, mask in regions.items():
    print(f"{name:8s} {mask.float().mean():6.1%} of the volume")

instances = gen.instance_labels.cpu()
membrane_labels = gen.membrane_labels.cpu()
print(
    f"instance ids: {int(instances.max())}, membrane ids: {int(membrane_labels.max())}"
)

picks_dir = Path(tempfile.mkdtemp())
picks = gen.export_picks(picks_dir)
for name, path in sorted(picks.items()):
    n = sum(1 for _ in open(path))
    print(f"{path.name:45s} {n:4d} picks")
# --8<-- [end:ground-truth]

# A slice through the first vesicle, where all three regions meet.
first = gen.placed_membrane_instances[0]
z = common["target_shape"][0] // 2 + round(first.position_xyz[2] / voxel_size)
z = int(np.clip(z, 0, common["target_shape"][0] - 1))

semantic = np.zeros(regions["shell"].shape[1:], dtype=np.int64)
semantic[as_numpy(regions["cytosol"][z]) > 0] = 1
semantic[as_numpy(regions["lumen"][z]) > 0] = 2
semantic[as_numpy(regions["shell"][z]) > 0] = 3
region_cmap = ListedColormap(["white", "#e8e2d4", PALETTE[0], "0.2"])

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
density = as_numpy(volume[z])
show_image(
    axes[0],
    density,
    cmap="gray_r",
    title=f"Density, z slice {z}",
    vmin=float(np.percentile(density, 0.5)),
    vmax=float(np.percentile(density, 99.7)),
    pixel_size=voxel_size,
    scalebar=500,
)
axes[1].imshow(semantic, cmap=region_cmap, vmin=0, vmax=3, interpolation="nearest")
axes[1].set_title("Regions: cytosol, lumen, shell")
mem = as_numpy(membrane_labels[z]).astype(float)
mem[mem == 0] = np.nan
axes[2].imshow(np.zeros_like(mem), cmap="gray_r", vmin=0, vmax=1)
axes[2].imshow(mem, cmap=ListedColormap(PALETTE[:3]), interpolation="nearest")
axes[2].set_title("membrane_labels, one id per vesicle")
for ax in axes[1:]:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
save(fig, "cellular_specimens-regions")

# Instance labels in random colours, and the target picks over the projection.
rng = np.random.default_rng(0)
n_ids = int(instances.max()) + 1
colors = rng.uniform(0.15, 0.95, size=(n_ids, 3))
colors[0] = 1.0
label_cmap = ListedColormap(colors)
label_slice = as_numpy(instances[z]).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
axes[0].imshow(
    label_slice, cmap=label_cmap, vmin=0, vmax=n_ids - 1, interpolation="nearest"
)
axes[0].set_title(f"instance_labels, z slice {z}")
projection = as_numpy(volume.sum(0))
show_image(
    axes[1],
    projection,
    cmap="gray_r",
    title="Target picks over the projection",
    vmin=float(np.percentile(projection, 0.5)),
    vmax=float(np.percentile(projection, 99.7)),
)
ny, nx = projection.shape
for color, (code, name) in zip(
    PALETTE[1:], [("6QZP", "ribosome"), ("7VD8", "apoferritin")]
):
    xy = torch.stack(
        [p.position_xyz for p in gen.placements if p.species_id == code]
    ).cpu()
    axes[1].scatter(
        xy[:, 0] / voxel_size + nx / 2,
        xy[:, 1] / voxel_size + ny / 2,
        s=40,
        facecolors="none",
        edgecolors=color,
        linewidths=1.3,
        label=name,
    )
axes[1].legend(loc="upper right", fontsize=7, frameon=True)
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.tight_layout()
save(fig, "cellular_specimens-labels")

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(
    label_slice[64:320, 64:320],
    cmap=label_cmap,
    vmin=0,
    vmax=n_ids - 1,
    interpolation="nearest",
)
ax.set_axis_off()
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "cellular_specimens-thumbnail")

transcript.start("tilt-series")
# --8<-- [start:tilt-series]
from specter.imaging import CTF, Exposure, Poses, Sample
from specter.settings import Ice

angles = [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0]  # dose-symmetric
exposure = Exposure.tilt_series(3.0, len(angles))  # e⁻/Å² per tilt
series = Sample.from_volume(volume[None], voxel_size).tilt_series(
    Poses.tilt_series(angles),
    exposure,
    voltage=300.0,
    micrograph_size=176,  # 1408 Å: the widest field the volume covers at 45°
    ice=Ice(model="gd"),
    seed=0,
    log=False,
)
ctf = CTF.from_values(defocus=40000.0, n=len(angles))
noiseless, tilts = [], []
for i in range(len(angles)):
    image = series.scatter(i, device="cuda").apply_ctf(ctf[i : i + 1]).intensity()
    noiseless.append(image)
    tilts.append(image.detect(exposure[i : i + 1], seed=0))
print(f"{len(tilts)} tilts of {tuple(tilts[0].array.shape[1:])} px")
print(
    f"coverage: {series.coverage.max_allowed_tilt_deg_for_volume:.1f}° max for this field"
)
# --8<-- [end:tilt-series]


def unpadded(image) -> torch.Tensor:
    """The first image on its pixel grid, the FFT margin cropped."""
    pixels = image.pixels()[0]
    m = (image.pad_nxy - image.nxy) // 2
    return pixels[m : m + image.nxy, m : m + image.nxy]


def lowpass(x: torch.Tensor, resolution: float) -> torch.Tensor:
    """A Gaussian low-pass to `resolution` Å, for display only; mirror-padded
    first, so a bead at one edge does not ring in at the opposite one."""
    pad = x.shape[-1] // 4
    x = torch.nn.functional.pad(x.float()[None, None], (pad,) * 4, mode="reflect")[0, 0]
    k = torch.fft.fftfreq(x.shape[-1], d=voxel_size, device=x.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    x = torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(-k2 * resolution**2)).real
    return x[pad:-pad, pad:-pad]


shown = [angles.index(a) for a in (-45.0, 0.0, 45.0)]
fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.5))
for col, i in enumerate(shown):
    clean = as_numpy(lowpass(unpadded(noiseless[i]), 20.0))
    noisy = as_numpy(lowpass(tilts[i].array[0], 20.0))
    for row, data in enumerate([clean, noisy]):
        show_image(
            axes[row, col],
            data,
            vmin=float(np.percentile(data, 1)),
            vmax=float(np.percentile(data, 99)),
            title=f"{angles[i]:+.0f}°" if row == 0 else None,
            pixel_size=voxel_size if col == 0 else None,
            scalebar=500 if col == 0 else None,
        )
axes[0, 0].set_ylabel("Noiseless")
axes[1, 0].set_ylabel("Detected, 3 e⁻/Å²")
fig.tight_layout()
save(fig, "cellular_specimens-tilts")

transcript.close()
