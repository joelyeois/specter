"""
Generate the figures for docs/concepts/cryoet-specimen/bilayer.md, which
explains how a membrane's shape becomes scattering potential
(``specter.specimen.membrane._profile``, ``._raster``, ``._placement``).

Calls the shipped `build_measured_bilayer_profile`/
`compute_bilayer_profile`/`rasterize_membrane_density` and a real
`MembraneGenerator` run rather than reimplementing any of them. The one
exception is `_superseded_two_gaussian`, the profile specter rendered with
until 2026-08-31: it is reconstructed here, not imported, because the
left-hand panel exists to show what replacing it bought.

Run with: uv run python docs-figures/cryoet_specimen_bilayer.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _render import TEAL
from specter.specimen.membrane import MembraneGenerator, TransmembraneSpec
from specter.specimen.membrane._profile import (
    CALIBRATION_N_LIPIDS_PER_LEAFLET,
    build_measured_bilayer_profile,
    build_reference_lipid_patch,
    compute_bilayer_profile,
    native_bilayer_thickness_a,
)
from specter.specimen.membrane._raster import rasterize_membrane_density

OUT_DIR = "docs/assets/images"
# Expanded: passed through unexpanded, a literal "~" directory gets
# created wherever the script is run from, duplicating the cache.
PDB_CACHE = os.path.expanduser("~/.cache/specter/pdb")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 5
ORANGE = "#d98218"
PURPLE = "#6b4fa0"


def _superseded_two_gaussian(
    thickness_a: float = 30.0, layer_sigma_a: float = 1.25, amplitude: float = 20.2
) -> tuple[np.ndarray, np.ndarray]:
    """The profile specter shipped until 2026-08-31: two Gaussians on
    vacuum, scaled by an amplitude fitted from an isolated atom's peak.

    Reconstructed here rather than imported because it was deleted --
    modelling a bilayer as two peaks with nothing between them describes
    how one LOOKS under CTF, not what it is, and discarding the acyl core
    cost 4.8x of the integrated potential. Kept only so the figure can
    show that difference.
    """
    d = np.linspace(-40.0, 40.0, 481)
    half = thickness_a / 2.0
    peaks = np.exp(-0.5 * ((d - half) / layer_sigma_a) ** 2) + np.exp(
        -0.5 * ((d + half) / layer_sigma_a) ** 2
    )
    return d, amplitude * peaks


def figure_profile() -> None:
    """How psi(d) is built. Left: the measured profile the generator
    renders, against the two-Gaussian form it replaced -- the difference
    is the acyl core, which is most of the integrated potential. Right:
    what bilayer_thickness does to it."""
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=CALIBRATION_N_LIPIDS_PER_LEAFLET, seed=SEED, device=DEVICE
    )
    measured = compute_bilayer_profile(atomic_numbers, coordinates, device=DEVICE)
    m_d = measured.distance_a.cpu().numpy()
    m_psi = measured.psi.cpu().numpy()
    old_d, old_psi = _superseded_two_gaussian()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    ax = axes[0]
    ax.plot(m_d, m_psi, color=ORANGE, linewidth=2, label="measured psi(d) (shipped)")
    ax.fill_between(m_d, 0, m_psi, color=ORANGE, alpha=0.12)
    ax.plot(
        old_d, old_psi, color=TEAL, linewidth=1.4, label="two-Gaussian (superseded)"
    )
    ax.axhline(4.6, ls="--", lw=1.0, color="#888")
    ax.text(-38, 4.9, "amorphous ice, 4.6 V", fontsize=8, color="#666")
    integral = float(np.trapezoid(m_psi[np.abs(m_d) < 40], m_d[np.abs(m_d) < 40]))
    ax.set_xlabel("signed distance from mid-plane, d (A)")
    ax.set_ylabel("psi(d), volts")
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, max(m_psi.max(), old_psi.max()) * 1.08)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(
        f"integral psi dz = {integral:.0f} V*A, against 53 before", fontsize=10
    )

    ax = axes[1]
    native = native_bilayer_thickness_a()
    for thickness, style in [(30.0, ":"), (38.0, "-"), (46.0, "--")]:
        profile = build_measured_bilayer_profile(thickness_a=thickness)
        psi = profile.psi.cpu().numpy()
        dist = profile.distance_a.cpu().numpy()
        ax.plot(
            dist,
            psi,
            style,
            color=TEAL,
            linewidth=1.8 if thickness == 38.0 else 1.2,
            label=f"bilayer_thickness = {thickness:g}",
        )
    ax.set_xlabel("signed distance from mid-plane, d (A)")
    ax.set_ylabel("psi(d), volts")
    ax.set_xlim(-40, 40)
    ax.legend(fontsize=8)
    ax.set_title(
        f"rescaled in z at fixed amplitude (template: {native:.0f} A)", fontsize=10
    )

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
