"""
Generate the figures for docs/concepts/atomic-potentials.md.

Two groups of figures:

1. Reproductions of Kirkland, *Advanced Computing in Electron Microscopy*
   (2nd ed.), Ch. 5 worked examples for the standard five-element test row
   (C, Si, Cu, Au, U) -- the same computation as
   demo-notebooks/compare-atomic-potentials-with-kirkland.ipynb, factored
   out here so the docs site gets a clean PNG instead of a notebook
   screenshot. Each of the five (``atomic-potential-3d-kirkland.png``,
   ``projected-atomic-potential-2d-kirkland.png``,
   ``transmission-function-linescan-kirkland.png``,
   ``coherent-bright-field-image-kirkland.png``,
   ``coherent-bright-field-linescan-kirkland.png``) places SPECTER's own
   output next to a scan of the textbook figure it reproduces
   (images/*.png, already in the repo) for direct visual comparison.
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
SCAN_DIR = REPO_ROOT / "images"
POTENTIAL_3D_SCAN = SCAN_DIR / "atomic-potential-3d-kirkland.png"
POTENTIAL_2D_SCAN = SCAN_DIR / "projected-atomic-potential-2d-kirkland.png"
TEXTBOOK_SCAN = SCAN_DIR / "coherent-bright-field-linescan-kirkland.png"
TRANSMISSION_SCAN = SCAN_DIR / "line-scan-transmission-kirkland.png"
IMAGE_SCAN = SCAN_DIR / "coherent-bright-field-kirkland.png"

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


def _matched_grid(
    n_rows: int,
    n_cols: int,
    box_w: float,
    box_h: float,
    *,
    left_margin: float = 0.9,
    right_margin: float = 0.3,
    bottom_margin: float = 0.6,
    top_margin: float = 0.45,
    col_gap: float = 0.6,
    row_gap: float = 0.55,
    dpi: int = 200,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """A grid of axes that are all an identical, explicit box_w x box_h size
    (in inches), placed via add_axes rather than plt.subplots + a layout
    engine. A layout engine (e.g. constrained_layout) sizes each axes
    independently around its own labels/title, so a shared box_aspect alone
    only matches panel *shape*, not size -- see
    figure_bright_field_linescan_comparison's git history for the ~13%
    mismatch that produced."""
    fig_w = left_margin + n_cols * box_w + (n_cols - 1) * col_gap + right_margin
    fig_h = bottom_margin + n_rows * box_h + (n_rows - 1) * row_gap + top_margin
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    axes = []
    for row in range(n_rows):
        y0 = bottom_margin + (n_rows - 1 - row) * (box_h + row_gap)
        row_axes = []
        for col in range(n_cols):
            x0 = left_margin + col * (box_w + col_gap)
            ax = fig.add_axes((x0 / fig_w, y0 / fig_h, box_w / fig_w, box_h / fig_h))
            row_axes.append(ax)
        axes.append(row_axes)
    return fig, axes


def _crop(img: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Trim a scan down to just its spine frame -- no axis ticks/number
    labels, no printed figure caption. Real matplotlib ticks/labels are
    drawn on top instead, on the same data range as the original scan, so
    SPECTER's panel and the scan share one consistent typography and axes
    boxes that match exactly rather than just approximately (an inset crop,
    imshow'd with aspect="auto" into an identically-sized box, does not by
    itself make the *spine* sizes match -- the spine sits inset from the
    crop edges by however much tick-label margin the crop still carries)."""
    left, top, right, bottom = box
    return img[top:bottom, left:right]


POTENTIAL_3D_SCAN_CROP = (188, 24, 660, 396)
POTENTIAL_2D_SCAN_CROP = (209, 17, 669, 381)


def figure_3d_potential_comparison() -> None:
    """SPECTER's own 3D atomic potential next to the scanned textbook figure
    (Fig. 5.4)."""
    n = 200
    x, R = _radial_grid_3d(n)

    book_img = mpimg.imread(POTENTIAL_3D_SCAN)
    book_img = _crop(book_img, POTENTIAL_3D_SCAN_CROP)
    box_aspect = book_img.shape[0] / book_img.shape[1]

    # Sized close to the scan crop's own native resolution (472x372 px) at
    # 200 dpi -- see figure_transmission_function_comparison for why a
    # smaller, near-native box avoids blurring the scan's printed labels.
    box_w, box_h = 2.5, 2.5 * box_aspect
    fig, ((ax_ours, ax_book),) = _matched_grid(1, 2, box_w, box_h)

    for color, el, z in zip(PALETTE, ELEMENTS, ATOMIC_NUMBERS):
        pot3d = kirkland_atomic_potential_3d(z, R)
        ax_ours.plot(
            x[n // 2 :], pot3d[n // 2, n // 2, n // 2 :] / 1e3, color=color, label=el
        )
    ax_ours.set_xlabel("Radius (Å)")
    ax_ours.set_ylabel("Atomic potential (kV)")
    ax_ours.set_xlim(0, 0.5)
    ax_ours.set_ylim(0, 20)
    ax_ours.grid(True, alpha=0.4)
    ax_ours.legend(fontsize=7)
    ax_ours.set_title("SPECTER")

    ax_book.imshow(book_img, extent=(0, 0.5, 0, 20), aspect="auto")
    ax_book.set_xlim(0, 0.5)
    ax_book.set_ylim(0, 20)
    ax_book.set_xlabel("Radius (Å)")
    ax_book.set_title("Kirkland (2010), Fig. 5.4")

    path = OUT_DIR / "atomic-potential-3d-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_2d_potential_comparison() -> None:
    """SPECTER's own 2D projected atomic potential next to the scanned
    textbook figure (Fig. 5.5)."""
    n = 800
    x, R_xy = _radial_grid_2d(n)

    book_img = mpimg.imread(POTENTIAL_2D_SCAN)
    book_img = _crop(book_img, POTENTIAL_2D_SCAN_CROP)
    box_aspect = book_img.shape[0] / book_img.shape[1]

    box_w, box_h = 2.5, 2.5 * box_aspect
    fig, ((ax_ours, ax_book),) = _matched_grid(1, 2, box_w, box_h)

    for color, el, z in zip(PALETTE, ELEMENTS, ATOMIC_NUMBERS):
        pot2d = kirkland_atomic_potential_2d(z, R_xy)
        ax_ours.plot(x[n // 2 :], pot2d[n // 2, n // 2 :], color=color, label=el)
    ax_ours.set_xlabel("Radius (Å)")
    ax_ours.set_ylabel("Projected atomic potential (V·Å)")
    ax_ours.set_xlim(0, 0.5)
    ax_ours.set_ylim(0, 5000)
    ax_ours.grid(True, alpha=0.4)
    ax_ours.legend(fontsize=7)
    ax_ours.set_title("SPECTER")

    ax_book.imshow(book_img, extent=(0, 0.5, 0, 5000), aspect="auto")
    ax_book.set_xlim(0, 0.5)
    ax_book.set_ylim(0, 5000)
    ax_book.set_xlabel("Radius (Å)")
    ax_book.set_title("Kirkland (2010), Fig. 5.5")

    path = OUT_DIR / "projected-atomic-potential-2d-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _bright_field_simulation() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Kirkland Figs. 5.11-5.13: transmission function -> Scherzer image.

    Returns
    -------
    x : (n,) tensor
        Position along the atom row, Angstroms, 0 to 50.
    y : (n,) tensor
        Position across the atom row, Angstroms, -25 to 25.
    transmission : (n, n) complex tensor
        t(x, y) = exp(i * sigma * V_2D(x, y)), row n // 2 runs through the
        atom centers (Fig. 5.11).
    image : (n, n) tensor
        The Scherzer-condition coherent bright-field image, |propagated|^2
        (Fig. 5.12); row n // 2 is its line scan (Fig. 5.13).
    """
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
    return x, y, transmission, image


TEXTBOOK_SCAN_CROP = (153, 17, 807, 534)
# These two crops start below the scan's own printed "C Si Cu Au U" labels
# (row 16+26 / 284+26, skipping the top spine and the text band beneath it),
# not at the spine like the other scans -- see figure_transmission_function_
# comparison for why: those labels are only 14px tall natively, too small to
# survive being shared with SPECTER's own larger vector labels at any one
# box size. Real matplotlib text is drawn over the resulting blank strip
# instead, at the same position/fontsize as SPECTER's panel.
TRANSMISSION_SCAN_CROP_REAL = (134, 42, 723, 212)
TRANSMISSION_SCAN_CROP_IMAG = (134, 310, 723, 480)
# Data-coordinate y value the crop's new top row corresponds to (the label
# band it replaces spans the rest of the way up to each panel's real ylim).
TRANSMISSION_CROP_YTOP_REAL = 1.2 * (1 - 26 / 196)
TRANSMISSION_CROP_YTOP_IMAG = 1.0 * (1 - 26 / 196)
IMAGE_SCAN_CROP = (146, 10, 664, 529)


def figure_transmission_function_comparison() -> None:
    """SPECTER's own transmission-function line scan next to the scanned
    textbook figure (Fig. 5.11): real part on top, imaginary part below."""
    x, _, transmission, _ = _bright_field_simulation()
    n = x.shape[0]
    real_part = transmission[n // 2, :].real
    imag_part = transmission[n // 2, :].imag
    element_locations = [5, 15, 25, 35, 45]

    book_img = mpimg.imread(TRANSMISSION_SCAN)
    real_crop = _crop(book_img, TRANSMISSION_SCAN_CROP_REAL)
    imag_crop = _crop(book_img, TRANSMISSION_SCAN_CROP_IMAG)
    # The panel's own box shape comes from the *full* spine box (0 to each
    # panel's ylim), not the content-only crop above (which excludes the
    # label band, and so is shorter) -- box_aspect must match ax_*_ours'
    # actual shape, independent of how much of the box the image fills.
    box_aspect = 196 / 589

    box_w, box_h = 4.6, 4.6 * box_aspect
    (
        fig,
        (
            (ax_real_ours, ax_real_book),
            (ax_imag_ours, ax_imag_book),
        ),
    ) = _matched_grid(2, 2, box_w, box_h)

    ax_real_ours.plot(x, real_part, color=PALETTE[0])
    for loc, el in zip(element_locations, ELEMENTS):
        ax_real_ours.text(loc, 1.08, el, ha="center")
    ax_real_ours.set_xlim(0, 50)
    ax_real_ours.set_ylim(0, 1.2)
    ax_real_ours.set_ylabel("Real part")
    ax_real_ours.set_title("SPECTER")
    ax_real_ours.grid(True, alpha=0.4)

    ax_real_book.imshow(
        real_crop, extent=(0, 50, 0, TRANSMISSION_CROP_YTOP_REAL), aspect="auto"
    )
    for loc, el in zip(element_locations, ELEMENTS):
        ax_real_book.text(loc, 1.08, el, ha="center")
    ax_real_book.set_xlim(0, 50)
    ax_real_book.set_ylim(0, 1.2)
    ax_real_book.set_title("Kirkland (2010), Fig. 5.11")

    ax_imag_ours.plot(x, imag_part, color=PALETTE[0])
    for loc, el in zip(element_locations, ELEMENTS):
        ax_imag_ours.text(loc, 0.90, el, ha="center")
    ax_imag_ours.set_xlim(0, 50)
    ax_imag_ours.set_ylim(0, 1.0)
    ax_imag_ours.set_xlabel("position x (Å)")
    ax_imag_ours.set_ylabel("Imag. part")
    ax_imag_ours.grid(True, alpha=0.4)

    ax_imag_book.imshow(
        imag_crop, extent=(0, 50, 0, TRANSMISSION_CROP_YTOP_IMAG), aspect="auto"
    )
    for loc, el in zip(element_locations, ELEMENTS):
        ax_imag_book.text(loc, 0.90, el, ha="center")
    ax_imag_book.set_xlim(0, 50)
    ax_imag_book.set_ylim(0, 1.0)
    ax_imag_book.set_xlabel("position x (Å)")

    path = OUT_DIR / "transmission-function-linescan-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_bright_field_image_comparison() -> None:
    """SPECTER's own coherent bright-field image next to the scanned textbook
    figure (Fig. 5.12), both grayscale on the same intensity scale (0.72 to
    1.03 -- the black/white levels the original caption itself states)."""
    _, _, _, image = _bright_field_simulation()

    book_img = mpimg.imread(IMAGE_SCAN)
    book_img = _crop(book_img, IMAGE_SCAN_CROP)
    box_aspect = book_img.shape[0] / book_img.shape[1]

    box_w, box_h = 4.6, 4.6 * box_aspect
    fig, ((ax_ours, ax_book),) = _matched_grid(1, 2, box_w, box_h)

    ax_ours.imshow(
        image,
        extent=(0, 50, -25, 25),
        origin="lower",
        cmap="gray",
        vmin=0.72,
        vmax=1.03,
    )
    ax_ours.set_xlabel("position x (Å)")
    ax_ours.set_ylabel("position y (Å)")
    ax_ours.set_title("SPECTER")

    ax_book.imshow(book_img, extent=(0, 50, -25, 25), aspect="auto")
    ax_book.set_xlabel("position x (Å)")
    ax_book.set_title("Kirkland (2010), Fig. 5.12")

    path = OUT_DIR / "coherent-bright-field-image-kirkland.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_bright_field_linescan_comparison() -> None:
    """SPECTER's own linescan next to the scanned textbook figure (Fig. 5.13)."""
    x, _, _, image = _bright_field_simulation()
    linescan = image[x.shape[0] // 2, :]
    element_locations = [5, 15, 25, 35, 45]

    book_img = mpimg.imread(TEXTBOOK_SCAN)
    book_img = _crop(book_img, TEXTBOOK_SCAN_CROP)
    box_aspect = book_img.shape[0] / book_img.shape[1]

    box_w, box_h = 4.6, 4.6 * box_aspect
    fig, ((ax_ours, ax_book),) = _matched_grid(1, 2, box_w, box_h)

    ax_ours.plot(x, linescan, color=PALETTE[0])
    for loc, el in zip(element_locations, ELEMENTS):
        ax_ours.text(loc, 1.05, el, ha="center")
    ax_ours.set_xlim(0, 50)
    ax_ours.set_ylim(0.5, 1.1)
    ax_ours.set_xlabel("position x (Å)")
    ax_ours.set_ylabel("Image intensity")
    ax_ours.set_title("SPECTER")
    ax_ours.grid(True, alpha=0.4)

    ax_book.imshow(book_img, extent=(0, 50, 0.5, 1.1), aspect="auto")
    ax_book.set_xlim(0, 50)
    ax_book.set_ylim(0.5, 1.1)
    ax_book.set_xlabel("position x (Å)")
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
    figure_3d_potential_comparison()
    figure_2d_potential_comparison()
    figure_transmission_function_comparison()
    figure_bright_field_image_comparison()
    figure_bright_field_linescan_comparison()
    figure_lorentzian_gaussian_terms()
    figure_parameterization_comparison()


if __name__ == "__main__":
    main()
