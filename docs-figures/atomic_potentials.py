"""
Generate the figures for docs/concepts/atomic-potentials.md.

Two groups of figures:

1. Reproductions of Kirkland, *Advanced Computing in Electron Microscopy*
   (2nd ed.), Ch. 5 worked examples for the standard five-element test row
   (C, Si, Cu, Au, U) -- the same computation as
   demo-notebooks/compare-atomic-potentials-with-kirkland.ipynb, factored
   out here so the docs site gets a clean PNG instead of a notebook
   screenshot. ``atomic-potential-3d-kirkland.png`` and
   ``projected-atomic-potential-2d-kirkland.png`` are SPECTER's own
   output; ``coherent-bright-field-linescan-kirkland.png`` places it next
   to a scan of the textbook figure it reproduces (images/*.png, already
   in the repo) for direct visual comparison.
2. New figures explaining the parameterizations themselves: the
   Lorentzian + Gaussian term decomposition behind Kirkland's fit, and an
   overlay of the Kirkland/Lobato/Peng curves showing three independently
   fitted parameterizations agree.

Run with: uv run python docs-figures/atomic_potentials.py
Saves PNGs directly into docs/assets/images/ (consumed by
docs/concepts/atomic-potentials.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch

from specter.atom import atom_number
from specter.atom.atomic_potentials import (
    kirkland_atomic_potential_2d,
    kirkland_atomic_potential_3d,
    load_kirkland_parameters,
    lobato_atomic_potential_3d,
    peng_atomic_potential_3d,
)
from specter.fft import fft2, ifft2
from specter.plots import _deep_palette

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"
TEXTBOOK_SCAN = REPO_ROOT / "images" / "coherent-bright-field-linescan-kirkland.png"

ELEMENTS = ["C", "Si", "Cu", "Au", "U"]
ATOMIC_NUMBERS = [int(atom_number(el)) for el in ELEMENTS]
PALETTE = _deep_palette(len(ELEMENTS))


def _radial_grid_3d(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-0.5, 0.5, n)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="xy")
    return x, torch.sqrt(X**2 + Y**2 + Z**2)


def _radial_grid_2d(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-0.5, 0.5, n)
    X, Y = torch.meshgrid(x, x, indexing="xy")
    return x, torch.sqrt(X**2 + Y**2)


def figure_3d_potential() -> None:
    """Kirkland Fig. 5.4: 3D atomic potential vs. radius, per element."""
    n = 200
    x, R = _radial_grid_3d(n)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    for color, el, z in zip(PALETTE, ELEMENTS, ATOMIC_NUMBERS):
        pot3d = kirkland_atomic_potential_3d(z, R)
        ax.plot(
            x[n // 2 :], pot3d[n // 2, n // 2, n // 2 :] / 1e3, color=color, label=el
        )
    ax.set_xlabel("Radius (Angstroms)")
    ax.set_ylabel("Atomic potential (kV)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "atomic-potential-3d-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_2d_potential() -> None:
    """Kirkland Fig. 5.5: 2D projected atomic potential vs. radius."""
    n = 800
    x, R_xy = _radial_grid_2d(n)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    for color, el, z in zip(PALETTE, ELEMENTS, ATOMIC_NUMBERS):
        pot2d = kirkland_atomic_potential_2d(z, R_xy)
        ax.plot(x[n // 2 :], pot2d[n // 2, n // 2 :], color=color, label=el)
    ax.set_xlabel("Radius (Angstroms)")
    ax.set_ylabel("Projected atomic potential (V·Å)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 5000)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "projected-atomic-potential-2d-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _bright_field_linescan() -> tuple[torch.Tensor, torch.Tensor]:
    """Kirkland Figs. 5.11-5.13: transmission function -> Scherzer image -> linescan."""
    rest_mass_energy = torch.tensor(511.0e3)  # [eV]
    hc = torch.tensor(12.398e3)  # [eV * Å]

    def energy_to_wavelength(voltage: torch.Tensor) -> torch.Tensor:
        ev = voltage * 1e3
        return hc / torch.sqrt(ev * (ev + 2.0 * rest_mass_energy))

    def interaction_parameter(voltage: torch.Tensor) -> torch.Tensor:
        w = energy_to_wavelength(voltage)
        ev = voltage * 1e3
        return (
            2.0
            * torch.pi
            / (w * ev)
            * ((ev + rest_mass_energy) / (ev + 2.0 * rest_mass_energy))
        )

    voltage = torch.tensor(200.0)  # [kV]
    wavelength = energy_to_wavelength(voltage)
    sigma = interaction_parameter(voltage)

    element_locations = [5, 15, 25, 35, 45]  # Angstroms
    n = 512
    x = torch.linspace(0, 50, n)
    dx = x[1] - x[0]
    y = torch.linspace(-25, 25, n) - dx / 2

    pots_atom_2d = torch.zeros(n, n)
    for el, z in zip(element_locations, ATOMIC_NUMBERS):
        X, Y = torch.meshgrid(x, y, indexing="xy")
        R_xy = torch.sqrt((X - el) ** 2 + Y**2)
        pots_atom_2d = pots_atom_2d + kirkland_atomic_potential_2d(z, R_xy)

    transmission = torch.exp(1j * sigma * pots_atom_2d)

    cs = 1.3e7  # [Å]
    df = 700.0  # [Å]
    kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
    ky = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
    kxx, kyy = torch.meshgrid(kx, ky, indexing="ij")
    k2 = kxx**2 + kyy**2

    # Scherzer condition, Kirkland Eq. 3.17.
    kmax = 1 / (0.64 * (cs * wavelength**3) ** 0.25)
    aperture = (k2.sqrt() < kmax).to(transmission.dtype)

    gamma = torch.pi * wavelength * k2 * (0.5 * cs * wavelength**2 * k2 - df)
    propagated = ifft2(
        fft2(transmission, shift=True) * torch.exp(-1j * gamma) * aperture, shift=True
    )
    image = torch.abs(propagated**2)
    return x, image[n // 2, :]


def figure_bright_field_linescan_comparison() -> None:
    """SPECTER's own linescan next to the scanned textbook figure (Fig. 5.13)."""
    x, linescan = _bright_field_linescan()
    element_locations = [5, 15, 25, 35, 45]

    fig, (ax_ours, ax_book) = plt.subplots(
        2, 1, figsize=(7, 8), dpi=200, constrained_layout=True
    )
    ax_ours.plot(x, linescan, color=PALETTE[0])
    for loc, el in zip(element_locations, ELEMENTS):
        ax_ours.text(loc, 1.05, el, ha="center")
    ax_ours.set_xlim(0, 50)
    ax_ours.set_ylim(0.5, 1.1)
    ax_ours.set_xlabel("position x (Å)")
    ax_ours.set_ylabel("Image intensity")
    ax_ours.set_title("SPECTER")
    ax_ours.grid(True, alpha=0.4)

    ax_book.imshow(mpimg.imread(TEXTBOOK_SCAN))
    ax_book.set_axis_off()
    ax_book.set_title("Kirkland (2010), Fig. 5.13")

    path = OUT_DIR / "coherent-bright-field-linescan-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_lorentzian_gaussian_terms() -> None:
    """Decompose Kirkland's fit for one element into its 3 Lorentzian
    (Yukawa-like, real-space) + 3 Gaussian terms, plus their sum."""
    element, z = "Au", int(atom_number("Au"))
    n = 400
    x, R = _radial_grid_3d(n)
    r = x[n // 2 :]

    a0 = 0.529  # Bohr radius, [Å]
    e = 14.4  # electron charge, [V·Å]
    c1 = 2 * (torch.pi**2) * a0 * e
    c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

    P = load_kirkland_parameters()[z]  # (3, 4): [a_i, b_i, c_i, d_i]
    r_line = R[n // 2, n // 2, n // 2 :].clamp(min=1e-4)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    lorentzian_palette = _deep_palette(6)
    for i in range(3):
        a_i, b_i, c_i, d_i = P[i]
        lorentzian = c1 * a_i / r_line * torch.exp(-2 * torch.pi * r_line * b_i.sqrt())
        gaussian = (
            c2 * c_i * d_i ** (-1.5) * torch.exp(-(torch.pi**2) * r_line**2 / d_i)
        )
        ax.plot(
            r,
            lorentzian / 1e3,
            color=lorentzian_palette[i],
            linestyle="--",
            label=f"Lorentzian {i + 1} ($a_{i + 1}$={a_i:.3g}, $b_{i + 1}$={b_i:.3g})",
        )
        ax.plot(
            r,
            gaussian / 1e3,
            color=lorentzian_palette[i + 3],
            linestyle=":",
            label=f"Gaussian {i + 1} ($c_{i + 1}$={c_i:.3g}, $d_{i + 1}$={d_i:.3g})",
        )

    total = kirkland_atomic_potential_3d(z, R)[n // 2, n // 2, n // 2 :]
    ax.plot(r, total / 1e3, color="black", linewidth=2, label=f"Total ({element})")

    ax.set_xlabel("Radius (Angstroms)")
    ax.set_ylabel("Atomic potential (kV)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = OUT_DIR / "atomic-potential-lorentzian-gaussian-terms-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_parameterization_comparison() -> None:
    """Overlay Kirkland, Lobato, and Peng (independent-atom-model) real-space
    potentials for carbon: three independently fitted parameterizations
    agreeing away from the near-singularity at r=0."""
    element, z = "C", int(atom_number("C"))
    n = 400
    x, R = _radial_grid_3d(n)
    r = x[n // 2 :]

    kirkland = kirkland_atomic_potential_3d(z, R)[n // 2, n // 2, n // 2 :]
    lobato = lobato_atomic_potential_3d(z, R)[n // 2, n // 2, n // 2 :]
    peng = peng_atomic_potential_3d(z, R)[n // 2, n // 2, n // 2 :]

    palette = _deep_palette(3)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    ax.plot(r, kirkland / 1e3, color=palette[0], label="Kirkland")
    ax.plot(r, lobato / 1e3, color=palette[1], linestyle="--", label="Lobato")
    ax.plot(r, peng / 1e3, color=palette[2], linestyle=":", label="Peng (gemmi c4322)")
    ax.set_xlabel("Radius (Angstroms)")
    ax.set_ylabel("Atomic potential (kV)")
    ax.set_title(f"{element} (Z={z}): three independent parameterizations")
    ax.set_xlim(0.02, 0.5)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "atomic-potential-parameterization-comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    figure_3d_potential()
    figure_2d_potential()
    figure_bright_field_linescan_comparison()
    figure_lorentzian_gaussian_terms()
    figure_parameterization_comparison()


if __name__ == "__main__":
    main()
