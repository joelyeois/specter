"""
Generate the figures for docs/concepts/aberrations.md.

Calls `specter.aberrations.Aberration.transfer_function` and the
`specter.aberrations._envelopes` functions directly, at physically
representative 300 kV parameters, rather than reimplementing chi(k).

Run with: uv run python docs-figures/aberrations.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from specter.aberrations import Aberration
from specter.aberrations import _envelopes as env
from specter.arrays import radial_profile_2d
from specter.plots import _deep_palette

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"

VOLTAGE = 300.0  # kV
N_PIXELS = 420
PIXEL_SIZE = 1.0  # Angstrom -> Nyquist 0.5 1/A
N_PIXELS_1D = 900  # finer radial-bin spacing (1/(N*d)) for the smooth 1D CTF curve
DFU = 20000.0  # Angstrom (2 um), typical single-particle target defocus
CS = 2.7e7  # Angstrom (2.7 mm), typical Titan Krios spherical aberration


def _aberration() -> Aberration:
    return Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")


def _radial(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Radial profile of a native-FFT-order 2D field, in 1/Angstrom."""
    shifted = torch.fft.fftshift(field)
    r, profile = radial_profile_2d(shifted, return_r=True)
    return r.float() / (N_PIXELS * PIXEL_SIZE), profile


def figure_ctf_1d() -> None:
    """Classic radial CTF curve: Re/Im of the transfer function at a single
    representative defocus, with the Im part's zero crossings marked --
    the frequencies at which phase-contrast transfer changes sign, i.e.
    where Thon rings in a power spectrum go through zero. Uses a larger
    grid than the other figures purely for smoother radial-bin spacing
    (1/(N*d)) at high k, where chi(k) oscillates fastest."""
    ab = Aberration(N_PIXELS_1D, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    tf = ab.transfer_function({"dfu": torch.tensor([DFU]), "cs": torch.tensor([CS])})[0]
    shifted_re, shifted_im = torch.fft.fftshift(tf.real), torch.fft.fftshift(tf.imag)
    k, re = radial_profile_2d(shifted_re, return_r=True)
    _, im = radial_profile_2d(shifted_im, return_r=True)
    k = k.float() / (N_PIXELS_1D * PIXEL_SIZE)

    palette = _deep_palette(2)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(
        k.numpy(), re.numpy(), color=palette[0], label=r"Re[$T(k)$] = $\cos\chi(k)$"
    )
    ax.plot(
        k.numpy(), im.numpy(), color=palette[1], label=r"Im[$T(k)$] = $-\sin\chi(k)$"
    )
    ax.axhline(0, color="gray", linewidth=0.8)

    sign_changes = torch.where(torch.diff(torch.sign(im)) != 0)[0]
    for idx in sign_changes[:8]:
        ax.axvline(
            k[idx].item(), color=palette[1], linestyle=":", linewidth=0.7, alpha=0.6
        )

    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("Transfer function amplitude")
    ax.set_title(
        f"dfu = dfv = {DFU / 1e4:g} µm, Cs = {CS / 1e7:g} mm, {VOLTAGE:g} kV",
        fontsize=10,
    )
    ax.set_xlim(0, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "aberrations-ctf-1d.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _panel(ax, field2d: torch.Tensor, title: str) -> None:
    shifted = torch.fft.fftshift(field2d).numpy()
    ax.imshow(shifted, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_aberration_modes_2d() -> None:
    """Im[T(k)] over the full 2D frequency plane for four aberration terms
    in isolation: isotropic defocus (concentric Thon rings), astigmatism
    (elliptical rings), trefoil, and tetrafoil (both non-rotationally-
    symmetric, breaking the ring pattern into lobes)."""
    ab = _aberration()

    isotropic = ab.transfer_function(
        {"dfu": torch.tensor([DFU]), "cs": torch.tensor([CS])}
    )[0].imag
    astigmatic = ab.transfer_function(
        {
            "dfu": torch.tensor([DFU - 5000.0]),
            "dfv": torch.tensor([DFU + 5000.0]),
            "dfang": torch.tensor([30.0]),
            "cs": torch.tensor([CS]),
        }
    )[0].imag
    trefoil = ab.transfer_function(
        {
            "dfu": torch.tensor([5000.0]),
            "cs": torch.tensor([CS]),
            "trefoil1": torch.tensor([150.0]),
            "trefoil2": torch.tensor([0.0]),
        }
    )[0].imag
    tetrafoil = ab.transfer_function(
        {
            "dfu": torch.tensor([5000.0]),
            "cs": torch.tensor([CS]),
            "tetrafoil3": torch.tensor([1200.0]),
            "tetrafoil4": torch.tensor([0.0]),
        }
    )[0].imag

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=180)
    _panel(axes[0], isotropic, "Defocus\n(dfu = dfv)")
    _panel(axes[1], astigmatic, "Astigmatism\n(dfu ≠ dfv, dfang = 30°)")
    _panel(axes[2], trefoil, "Trefoil\n(3-fold)")
    _panel(axes[3], tetrafoil, "Tetrafoil\n(4-fold)")
    fig.suptitle("Im[T(k)] over the 2D frequency plane, isolated per term", y=1.05)
    fig.tight_layout()
    path = OUT_DIR / "aberrations-modes-2d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_envelopes() -> None:
    """The three continuous-parameter envelopes (B-factor, spatial
    coherence from Cs, temporal coherence from Cc) and the dose envelope,
    overlaid, plus their combined effect damping the same CTF curve from
    figure_ctf_1d -- the mechanism by which each sets a practical
    information limit well inside the detector's Nyquist frequency."""
    n = 400
    pixel_size = 0.5  # finer grid: push out past 1 1/A to see the falloff
    k = torch.fft.fftfreq(n, pixel_size)
    kxx, kyy = torch.meshgrid(k, k, indexing="ij")
    k2 = kxx**2 + kyy**2
    k_mag = torch.sqrt(k2)
    wavelength = float(Aberration(4, 1.0, VOLTAGE).wavelength)

    bfactor = 60.0  # Angstrom^2
    convergence_angle = 0.02  # mrad
    cc = 2.7e7  # Angstrom (2.7 mm)
    dose_values = [20.0, 60.0]

    b_env = env.b_envelope(k2, torch.tensor(bfactor))
    cs_env = env.cs_envelope(
        k_mag, wavelength, torch.tensor(CS), torch.tensor(float(DFU)), convergence_angle
    )
    cc_env = env.cc_envelope(k2, wavelength, cc, VOLTAGE * 1e3, 0.7, 0.06e-6, 0.01e-6)

    r, b_env_r = _radial_1d(b_env, n, pixel_size)
    _, cs_env_r = _radial_1d(cs_env, n, pixel_size)
    _, cc_env_r = _radial_1d(cc_env, n, pixel_size)
    combined_r = b_env_r * cs_env_r * cc_env_r

    palette = _deep_palette(4 + len(dose_values))
    fig, (ax_env, ax_ctf) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=190)

    ax_env.plot(r, b_env_r, color=palette[0], label=f"B-factor ({bfactor:g} Å²)")
    ax_env.plot(
        r,
        cs_env_r,
        color=palette[1],
        label=f"Cs coherence ({convergence_angle:g} mrad)",
    )
    ax_env.plot(r, cc_env_r, color=palette[2], label=f"Cc coherence ({cc / 1e7:g} mm)")
    ax_env.plot(
        r, combined_r, color="black", linewidth=2, label="Combined (B × Cs × Cc)"
    )
    for i, dose in enumerate(dose_values):
        dose_env = env.dose_envelope(k_mag, torch.tensor(dose))
        _, dose_env_r = _radial_1d(dose_env, n, pixel_size)
        ax_env.plot(
            r,
            dose_env_r,
            color=palette[3 + i],
            linestyle="--",
            label=f"Dose envelope ({dose:g} e⁻/Å²)",
        )
    ax_env.set_xlabel("Spatial frequency (1/Å)")
    ax_env.set_ylabel("Envelope amplitude")
    ax_env.set_xlim(0, 1.0)
    ax_env.set_ylim(0, 1.02)
    ax_env.legend(fontsize=7.5)
    ax_env.grid(True, alpha=0.3)
    ax_env.set_title("Envelopes in isolation", fontsize=10)

    ab = Aberration(
        n,
        pixel_size,
        VOLTAGE,
        aberration_model="nonlinear",
        bfactor=bfactor,
        convergence_angle=convergence_angle,
        cc=cc,
    )
    tf_damped = ab.transfer_function(
        {"dfu": torch.tensor([DFU]), "cs": torch.tensor([CS])}
    )[0]
    r2, im_damped = _radial_1d(tf_damped.imag, n, pixel_size)
    ab_bare = _aberration()
    tf_bare = ab_bare.transfer_function(
        {"dfu": torch.tensor([DFU]), "cs": torch.tensor([CS])}
    )[0]
    r3, im_bare = _radial(tf_bare.imag)

    ax_ctf.plot(
        r3.numpy(), im_bare.numpy(), color="gray", alpha=0.5, label="No envelope"
    )
    ax_ctf.plot(r2, im_damped, color="black", label="With B/Cs/Cc envelopes")
    ax_ctf.set_xlabel("Spatial frequency (1/Å)")
    ax_ctf.set_ylabel(r"Im[$T(k)$]")
    ax_ctf.set_xlim(0, 0.5)
    ax_ctf.legend(fontsize=8)
    ax_ctf.grid(True, alpha=0.3)
    ax_ctf.set_title("Effect on the CTF curve", fontsize=10)

    fig.tight_layout()
    path = OUT_DIR / "aberrations-envelopes.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _radial_1d(field: torch.Tensor, n: int, pixel_size: float):
    shifted = torch.fft.fftshift(field)
    r, profile = radial_profile_2d(shifted, return_r=True)
    return (r.float() / (n * pixel_size)).numpy(), profile.numpy()


def main() -> None:
    figure_ctf_1d()
    figure_aberration_modes_2d()
    figure_envelopes()


if __name__ == "__main__":
    main()
