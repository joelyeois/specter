"""
Generate the figures for docs/concepts/cryoet-specimen/bilayer.md, which
explains how a membrane's shape becomes scattering potential
(``specter.specimen.membrane._profile``, ``._raster``, ``._placement``).

Calls the shipped `build_analytic_bilayer_profile`/
`estimate_bilayer_peak_amplitude`/`rasterize_membrane_density` and a real
`MembraneGenerator` run rather than reimplementing any of them.

Run with: uv run python docs-figures/cryoet_specimen_bilayer.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _render import TEAL
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.membrane._profile import (
    build_analytic_bilayer_profile,
    build_reference_lipid_patch,
    compute_bilayer_profile,
    estimate_bilayer_peak_amplitude,
)
from specter.specimen.membrane._raster import rasterize_membrane_density

OUT_DIR = "docs/assets/images"
PDB_CACHE = "~/.cache/specter/pdb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 5
ORANGE = "#d98218"
PURPLE = "#6b4fa0"


def figure_profile() -> None:
    """How psi(d) is built. Left: the schematic atomic lipid patch's own
    laterally-averaged profile against the two-Gaussian analytic profile
    the generator actually ships -- same twin-peak structure, but the
    analytic one cannot grow a competing hump between the leaflets.
    Right: the two knobs that shape it."""
    atomic_numbers, coordinates = build_reference_lipid_patch(seed=SEED, device=DEVICE)
    atomic = compute_bilayer_profile(atomic_numbers, coordinates, device=DEVICE)
    amplitude = estimate_bilayer_peak_amplitude(atomic_numbers, coordinates)
    analytic = build_analytic_bilayer_profile(amplitude=amplitude)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    ax = axes[0]
    atomic_psi = atomic.psi.cpu().numpy()
    ax.plot(
        atomic.distance_a.cpu().numpy(),
        atomic_psi / atomic_psi.max(),
        color=ORANGE,
        label="atomic lipid patch (shape only)",
    )
    analytic_psi = analytic.psi.numpy()
    ax.plot(
        analytic.distance_a.numpy(),
        analytic_psi / analytic_psi.max(),
        color=TEAL,
        linewidth=2,
        label="analytic two-Gaussian (shipped)",
    )
    ax.set_xlabel("signed distance from mid-plane, d (A)")
    ax.set_ylabel("psi(d), normalised")
    ax.set_xlim(-30, 30)
    ax.legend(fontsize=8.5)
    ax.set_title(f"peak calibrated to {amplitude:.1f} V", fontsize=11)

    ax = axes[1]
    for thickness, style in [(25.0, ":"), (30.0, "-"), (35.0, "--")]:
        profile = build_analytic_bilayer_profile(thickness_a=thickness)
        ax.plot(
            profile.distance_a.numpy(),
            profile.psi.numpy(),
            style,
            color=TEAL,
            label=f"thickness_a = {thickness:g}",
        )
    for sigma, style in [(0.5, ":"), (2.0, "--")]:
        profile = build_analytic_bilayer_profile(layer_sigma_a=sigma)
        ax.plot(
            profile.distance_a.numpy(),
            profile.psi.numpy(),
            style,
            color=PURPLE,
            label=f"layer_sigma_a = {sigma:g}",
        )
    ax.set_xlabel("signed distance from mid-plane, d (A)")
    ax.set_ylabel("psi(d), uncalibrated")
    ax.set_xlim(-30, 30)
    ax.legend(fontsize=8)
    ax.set_title("bilayer_thickness and layer_sigma", fontsize=11)
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-bilayer-profile.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def _vesicle(transmembrane: bool = False) -> MembraneGenerator:
    specs = (
        [TransmembraneSpec(pdb_source="1C3W", frequency=60)] if transmembrane else None
    )
    gen = MembraneGenerator(
        voxel_size=4.0,
        shape_backend="spherical_harmonics",
        sh_axes_range=(200.0, 240.0),
        membrane_scale_range=(1.0, 1.0),
        bilayer_thickness=30.0,
        transmembrane_specs=specs,
        pdb_cache_dir=PDB_CACHE,
        device=DEVICE,
        seed=SEED,
    )
    gen.generate()
    return gen


def figure_antialias() -> None:
    """The same membrane field rasterized at three output voxel sizes,
    with and without the anti-aliasing pre-filter, along a line through
    the vesicle wall. Point-sampling a few-Angstrom-wide feature at a
    coarse voxel size makes the leaflet separation wander with voxel size
    -- an artifact, since the physical spacing is fixed. Filtering first
    gives the physically right answer instead: the two leaflets merge into
    one broad peak as resolution drops."""
    gen = _vesicle()
    field, profile = gen.field, gen.profile
    extent_a = 700.0

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.2), sharex=True, sharey=True)
    for col, voxel_size in enumerate([4.0, 8.0, 12.0]):
        n = int(round(extent_a / voxel_size)) // 2 * 2
        shape = (n, n, n)
        for row, antialias in enumerate([0.0, None]):
            volume = rasterize_membrane_density(
                field, profile, shape, voxel_size, antialias_sigma_a=antialias
            ).cpu()
            line = volume[n // 2, n // 2].numpy()
            x = (np.arange(n) - n / 2 + 0.5) * voxel_size
            ax = axes[row, col]
            ax.plot(x, line, "o-", color=TEAL if row else ORANGE, markersize=3)
            ax.set_xlim(-320, -140)
            if row == 0:
                ax.set_title(f"{voxel_size:g} A/voxel", fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    "point-sampled\npotential (V)"
                    if row == 0
                    else "anti-aliased\npotential (V)",
                    fontsize=10,
                )
            ax.set_xlabel("x (A)" if row == 1 else "")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-bilayer-antialias.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def figure_transmembrane() -> None:
    """Transmembrane placement. Left: sampled surface sites with the local
    field gradient (the exact surface normal) drawn at each. Right: the
    rendered result -- every protein's own axis follows the local normal,
    so none of them lie flat against the bilayer."""
    gen = _vesicle(transmembrane=True)
    placements = gen.place_transmembrane(min_spacing_a=40.0)
    sites = torch.stack([p.center_xyz for p in placements]).cpu().numpy()
    normals = gen.field.gradient(torch.stack([p.center_xyz for p in placements]))
    normals = normals.cpu().numpy()

    fig = plt.figure(figsize=(9.8, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.quiver(
        sites[:, 0],
        sites[:, 1],
        sites[:, 2],
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
        length=70.0,
        color=ORANGE,
        linewidth=0.9,
        arrow_length_ratio=0.3,
    )
    ax.scatter(sites[:, 0], sites[:, 1], sites[:, 2], color=TEAL, s=9)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"{len(placements)} sites + surface normals", fontsize=11)

    volume = gen.volume.cpu()
    mid = volume.shape[0] // 2
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(
        volume[mid - 4 : mid + 4].sum(dim=0).numpy(), cmap="gray_r", origin="lower"
    )
    ax.set_title("rendered slab (1C3W in the bilayer)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    path = f"{OUT_DIR}/cryoet-bilayer-transmembrane.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    figure_profile()
    figure_antialias()
    figure_transmembrane()
