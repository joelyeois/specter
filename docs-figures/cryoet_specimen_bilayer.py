"""
Generate the figures for docs/concepts/cryoet-specimen/bilayer.md, which
explains how a membrane's shape becomes scattering potential
(``specter.specimen.membrane._profile``, ``._render``, ``._placement``).

Calls the shipped `build_measured_bilayer_profile`/
`compute_bilayer_profile`/`profile_table` and a real
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
    CALIBRATION_SEED,
    CALIBRATION_VOXEL_SIZE_ANGSTROM,
    build_measured_bilayer_profile,
    build_reference_lipid_patch,
    compute_bilayer_profile,
    native_bilayer_thickness_angstrom,
)
from specter.specimen.membrane._render import profile_table

OUT_DIR = "docs/assets/images"
# Expanded: passed through unexpanded, a literal "~" directory gets
# created wherever the script is run from, duplicating the cache.
PDB_CACHE = os.path.expanduser("~/.cache/specter/pdb")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 5
ORANGE = "#d98218"
PURPLE = "#6b4fa0"


def _superseded_two_gaussian(
    thickness_angstrom: float = 30.0,
    layer_sigma_angstrom: float = 1.25,
    amplitude: float = 20.2,
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
    half = thickness_angstrom / 2.0
    peaks = np.exp(-0.5 * ((d - half) / layer_sigma_angstrom) ** 2) + np.exp(
        -0.5 * ((d + half) / layer_sigma_angstrom) ** 2
    )
    return d, amplitude * peaks


def figure_profile() -> None:
    """How psi(d) is built. Left: the measured profile the generator
    renders, against the two-Gaussian form it replaced -- the difference
    is the acyl core, which is most of the integrated potential. Right:
    what bilayer_thickness does to it."""
    # The shipped profile's own patch and grid, so the integral in the title
    # is the one the page quotes.
    atomic_numbers, coordinates = build_reference_lipid_patch(
        n_lipids_per_leaflet=CALIBRATION_N_LIPIDS_PER_LEAFLET,
        seed=CALIBRATION_SEED,
        device=DEVICE,
    )
    measured = compute_bilayer_profile(
        atomic_numbers,
        coordinates,
        voxel_size=CALIBRATION_VOXEL_SIZE_ANGSTROM,
        device=DEVICE,
    )
    m_d = measured.distance_angstrom.cpu().numpy()
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
    native = native_bilayer_thickness_angstrom()
    for thickness, style in [(30.0, ":"), (37.0, "-"), (46.0, "--")]:
        profile = build_measured_bilayer_profile(thickness_angstrom=thickness)
        psi = profile.psi.cpu().numpy()
        dist = profile.distance_angstrom.cpu().numpy()
        ax.plot(
            dist,
            psi,
            style,
            color=TEAL,
            linewidth=1.8 if thickness == 37.0 else 1.2,
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
        kind="vesicle",
        radius=[200.0, 240.0],
        bilayer_thickness=30.0,
        transmembrane_specs=specs,
        pdb_cache_dir=PDB_CACHE,
        device=DEVICE,
        seed=SEED,
    )
    gen.generate()
    return gen


def figure_voxel_size() -> None:
    """The same vesicle rendered at three output voxel sizes, along a line
    through its wall. `profile_table` averages psi(d) over one voxel along
    the membrane normal before the lookup, so the two leaflets lose
    amplitude and merge into one broad peak as the voxel grows, and the
    integral across the wall stays the same."""
    profile = build_measured_bilayer_profile(thickness_angstrom=37.0)
    fine_d = profile.distance_angstrom.cpu().numpy()
    fine_psi = profile.psi.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    for ax, voxel_size in zip(axes, [4.0, 8.0, 12.0]):
        gen = MembraneGenerator(
            voxel_size=voxel_size,
            kind="vesicle",
            radius=220.0,
            elongation=1.0,
            undulation=0.0,
            device=DEVICE,
            seed=SEED,
        )
        volume = gen.generate().cpu()
        nz, ny, nx = volume.shape
        line = volume[nz // 2, ny // 2].numpy()
        x = (np.arange(nx) - nx // 2) * voxel_size
        table, d0, step = profile_table(profile, voxel_size, DEVICE)
        table_d = d0 + step * np.arange(table.numel())
        integral = float(np.trapezoid(table.cpu().numpy(), table_d))
        print(
            f"voxel {voxel_size:g} A: peak along the line {line.max():.2f} V, "
            f"voxel-averaged profile peak {float(table.max()):.2f} V, "
            f"integral {integral:.1f} V*A"
        )
        ax.plot(-220.0 + fine_d, fine_psi, color="#bbbbbb", linewidth=1.0)
        ax.plot(x, line, "o-", color=TEAL, markersize=3)
        ax.set_xlim(-270, -170)
        ax.set_title(f"{voxel_size:g} A/voxel", fontsize=11)
        ax.set_xlabel("x (A)")
    axes[0].set_ylabel("potential (V)")
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
    placements = gen.place_transmembrane(min_spacing_angstrom=40.0)
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
    figure_voxel_size()
    figure_transmembrane()
