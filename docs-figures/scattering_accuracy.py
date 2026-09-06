"""
Generate the figures for docs/concepts/scattering/multislice.md,
docs/concepts/scattering/rytov.md, and docs/concepts/scattering/other-modes.md.

All figures are built from one real potential volume -- a `RandomIcemaker`
vitreous-ice slab (the same class `ImageGenerator`/`MicrographGenerator` use
for the fast/low-fidelity ice path) -- and call `specter.scattering.Scattering`
directly with each `scattering_model`, so the numbers plotted are exactly
what the real forward model produces, not a reimplementation.

Two groups of figures:

1. Multislice-specific: the per-slice transmit/propagate recursion traced
   at several depths through the slab, and the effect of Kirkland's `klim`
   antialiasing bandlimit on the exit wave's radial power spectrum.
2. An accuracy-vs-thickness sweep comparing every approximate mode
   (`rytov`, `firstborn`, `kinematic`, `projection`) against `multislice`
   as the reference, saved twice with different emphasis for rytov.md
   (rytov highlighted) and other-modes.md (firstborn/kinematic/projection
   highlighted, rytov shown faint for context).

Run with: uv run python docs-figures/scattering_accuracy.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from specter.arrays import radial_profile_2d
from specter.fft import fft2, ifft2
from specter.ice import RandomIcemaker
from specter.plots import _deep_palette
from specter.scattering import Scattering

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 11

PIXEL_SIZE = 2.0  # Angstrom, dz == pixel_size (Scattering's convention)
NXY = 64
NZ_MAX = 160  # -> 320 Angstrom slab, spanning typical single-particle ice thickness
VOLTAGE = 300.0
THICKNESS_STEPS_NZ = [4, 8, 16, 24, 32, 48, 64, 96, 128, 160]

APPROX_MODELS = ["rytov", "firstborn", "kinematic", "projection"]


def _ice_slab() -> torch.Tensor:
    """One real vitreous-ice potential volume, (1, NZ_MAX, NXY, NXY)."""
    torch.manual_seed(SEED)
    icemaker = RandomIcemaker(
        dx=PIXEL_SIZE, n=NXY, nz=NZ_MAX, parameterization="kirkland"
    )
    return icemaker.generate_ice(batchsize=1, device=DEVICE)


def figure_multislice_trace(V: torch.Tensor) -> None:
    """Exit-wave intensity after 1, 1/4, 1/2, 3/4, and all of the slab's
    slices, tracing the same transmit-then-propagate recursion
    `Scattering.multislice` runs internally (its own registered Fresnel
    propagator and sigma are reused directly, not recomputed)."""
    scat = Scattering(NXY, PIXEL_SIZE, VOLTAGE, scattering_model="multislice").to(
        DEVICE
    )
    F = scat.F_real + 1j * scat.F_imag
    V_flipped = torch.flip(V, dims=(1,))  # ews_curvature_sign="negative" default
    nz = V.shape[1]
    checkpoints = sorted({1, nz // 4, nz // 2, (3 * nz) // 4, nz})

    exitwave: torch.Tensor | None = None
    frames = {}
    for i in range(nz):
        t = torch.exp(1j * scat.sigma * scat.pixel_size * V_flipped[:, i])
        wv = t if exitwave is None else t * exitwave
        exitwave = ifft2(fft2(wv) * F)
        if (i + 1) in checkpoints:
            frames[i + 1] = torch.abs(exitwave[0]) ** 2

    fig, axes = plt.subplots(1, len(frames), figsize=(3.0 * len(frames), 3.2), dpi=180)
    # |psi|^2 is a weak perturbation that fluctuates symmetrically around a
    # unit incident baseline (a thin phase object barely changes intensity)
    # -- both brighter and darker specks appear as the recursion progresses.
    # Plotting the deviation magnitude |frame - 1| instead of raw |psi|^2
    # puts that baseline exactly at 0 (white, under gray_r), with departures
    # in *either* direction darkening it, rather than raw intensity sitting
    # mid-scale and reading as a flat, hard-to-read gray. A shared vmax
    # across panels (rather than autoscaling each independently) keeps the
    # comparison honest: the thin/early panels correctly stay close to flat
    # white, and only the deepest panel shows strong dark speckle.
    deviations = {depth: (frame - 1.0).abs() for depth, frame in frames.items()}
    vmax = max(d.max().item() for d in deviations.values())
    for ax, (depth, deviation) in zip(axes, deviations.items()):
        ax.imshow(deviation.cpu().numpy(), cmap="gray_r", vmin=0, vmax=vmax)
        ax.set_title(f"{depth} / {nz} slices\n({depth * PIXEL_SIZE:g} Å)", fontsize=14)
        ax.axis("off")
    fig.suptitle(
        "Exit-wave contrast ||ψ|² − 1| through the multislice recursion",
        y=1.03,
        fontsize=16,
    )
    fig.tight_layout()
    path = OUT_DIR / "multislice-recursion-trace.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_klim_bandlimit(V: torch.Tensor) -> None:
    """Radial power spectrum of the exit wave with and without Kirkland's
    klim antialiasing bandlimit, at the full slab thickness -- aliasing
    energy accumulates slice to slice, so it is most visible at the
    thickest specimen in the sweep."""
    palette = _deep_palette(2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    for color, klim, label in zip(
        palette, [None, 0.66], ["No bandlimit", "klim = 0.66 (Kirkland default)"]
    ):
        scat = Scattering(
            NXY, PIXEL_SIZE, VOLTAGE, scattering_model="multislice", klim=klim
        ).to(DEVICE)
        exitwave = scat(V)
        power = torch.abs(torch.fft.fft2(exitwave[0])) ** 2
        power = torch.fft.fftshift(power)
        r, profile = radial_profile_2d(power.cpu(), return_r=True)
        k_nyquist = 1 / (2 * PIXEL_SIZE)
        ax.semilogy(
            r.numpy() / (NXY * PIXEL_SIZE) / k_nyquist,
            profile.numpy(),
            color=color,
            label=label,
        )
    ax.axvline(0.66, color="gray", linestyle=":", linewidth=1)
    ax.text(0.665, ax.get_ylim()[0] * 3, "klim cutoff", fontsize=8, color="gray")
    ax.set_xlabel("Spatial frequency (fraction of Nyquist)")
    ax.set_ylabel("Radial power spectrum |ψ̂(k)|² (a.u., log scale)")
    ax.set_xlim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "multislice-klim-bandlimit.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _accuracy_sweep(
    V: torch.Tensor,
) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]]]:
    """Relative L1 error in exit-wave intensity vs. multislice, and mean
    intensity (energy conservation -- should stay 1.0, since the exit wave
    is unit-modulus before absorption), per model, as a function of slab
    thickness."""
    thickness_angstrom = []
    errors: dict[str, list[float]] = {m: [] for m in APPROX_MODELS}
    means: dict[str, list[float]] = {m: [] for m in APPROX_MODELS}

    for nz in THICKNESS_STEPS_NZ:
        V_slice = V[:, :nz].contiguous()
        thickness_angstrom.append(nz * PIXEL_SIZE)

        ref_scat = Scattering(
            NXY, PIXEL_SIZE, VOLTAGE, scattering_model="multislice", progressbars=False
        ).to(DEVICE)
        ref_intensity = torch.abs(ref_scat(V_slice)) ** 2

        for model in APPROX_MODELS:
            scat = Scattering(
                NXY,
                PIXEL_SIZE,
                VOLTAGE,
                scattering_model=model,
                nz=nz,
                progressbars=False,
            ).to(DEVICE)
            intensity = torch.abs(scat(V_slice)) ** 2
            rel_err = (
                torch.abs(intensity - ref_intensity).mean() / ref_intensity.mean()
            ).item()
            errors[model].append(rel_err)
            means[model].append(intensity.mean().item())

    return thickness_angstrom, errors, means


def figure_accuracy_vs_thickness(
    thickness_angstrom: list[float], errors: dict[str, list[float]]
) -> None:
    palette = dict(zip(APPROX_MODELS, _deep_palette(len(APPROX_MODELS))))

    def _plot(path_name: str, highlight: list[str], title: str) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
        for model in APPROX_MODELS:
            emphasized = model in highlight
            ax.semilogy(
                thickness_angstrom,
                errors[model],
                color=palette[model],
                linewidth=2.2 if emphasized else 1.2,
                alpha=1.0 if emphasized else 0.35,
                label=model,
                marker="o" if emphasized else None,
                markersize=4,
            )
        ax.set_xlabel("Specimen thickness (Å)")
        ax.set_ylabel("Relative error in |ψ|² vs. multislice")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = OUT_DIR / path_name
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}")

    _plot(
        "scattering-accuracy-vs-thickness-rytov.png",
        ["rytov"],
        "Rytov vs. multislice, as a function of ice thickness",
    )
    _plot(
        "scattering-accuracy-vs-thickness-other-modes.png",
        ["firstborn", "kinematic", "projection"],
        "First Born, kinematic and projection vs. multislice",
    )


def figure_mean_intensity_vs_thickness(
    thickness_angstrom: list[float], means: dict[str, list[float]]
) -> None:
    """Mean exit-wave intensity vs. thickness, per model. A properly
    normalized exit wave conserves total intensity (mean |psi|^2 == 1,
    before absorption/alpha); this figure exposes which approximations
    actually preserve that, and which quietly stop conserving energy as
    the specimen thickens -- the real breakdown of the linearized
    single-scattering models, which the relative-error figure above
    reports as "error" without saying *why*."""
    palette = dict(zip(APPROX_MODELS, _deep_palette(len(APPROX_MODELS))))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    ax.axhline(1.0, color="black", linewidth=1.0, label="multislice (reference)")
    for model in APPROX_MODELS:
        ax.plot(
            thickness_angstrom,
            means[model],
            color=palette[model],
            marker="o",
            markersize=4,
            label=model,
        )
    ax.set_xlabel("Specimen thickness (Å)")
    ax.set_ylabel("Mean exit-wave intensity ⟨|ψ|²⟩")
    ax.set_title("Energy conservation vs. thickness", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "scattering-mean-intensity-vs-thickness.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_mode_intensity_maps(V: torch.Tensor) -> None:
    """Exit-wave intensity maps at full slab thickness, side by side, for
    multislice (reference), rytov, firstborn, and projection -- the direct
    visual counterpart to the mean-intensity and relative-error figures.
    Shows *what* each approximation actually produces, not just a scalar
    error against the reference."""
    nz = THICKNESS_STEPS_NZ[-1]
    V_slice = V[:, :nz].contiguous()
    models = ["multislice", "rytov", "firstborn", "projection"]

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4), dpi=180)
    for ax, model in zip(axes, models):
        kwargs = {"nz": nz} if model in ("firstborn", "kinematic", "rytov") else {}
        scat = Scattering(
            NXY,
            PIXEL_SIZE,
            VOLTAGE,
            scattering_model=model,
            progressbars=False,
            **kwargs,
        ).to(DEVICE)
        intensity = torch.abs(scat(V_slice)) ** 2
        ax.imshow(intensity[0].cpu().numpy(), cmap="gray_r", vmin=0.8, vmax=1.7)
        ax.set_title(
            f"{model}\nmean={intensity.mean().item():.3f}, "
            f"std={intensity.std().item():.3f}",
            fontsize=12,
        )
        ax.axis("off")
    fig.suptitle(
        f"Exit-wave intensity at {nz * PIXEL_SIZE:g} Å, per model", y=1.05, fontsize=14
    )
    fig.tight_layout()
    path = OUT_DIR / "scattering-mode-intensity-maps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _rytov_theta(V_slice: torch.Tensor, nz: int) -> torch.Tensor:
    """The complex quantity Theta = sigma*dz*sum_z[F_z * V_z] such that
    rytov's exit wave is exp(i*Theta) and firstborn's is 1 + i*Theta --
    i.e. firstborn is the linearization of rytov in this exact variable.
    Re(Theta) is the ordinary (propagation-independent-at-low-k) projected
    phase; Im(Theta) is what genuinely drives |exp(i*Theta))|^2 = exp(-2*
    Im(Theta)), the true intensity. Recomputed directly (not read off
    Scattering.rytov's output) so both real and imaginary parts are
    available -- Scattering only returns the already-exponentiated wave."""
    scat = Scattering(
        NXY, PIXEL_SIZE, VOLTAGE, scattering_model="rytov", nz=nz, progressbars=False
    ).to(DEVICE)
    F = scat.F_real + 1j * scat.F_imag
    V_flipped = torch.flip(V_slice, dims=(1,))
    E = ifft2(fft2(V_flipped) * F).sum(dim=1)[0]
    return scat.sigma * scat.pixel_size * E


def figure_pattern_correlation_vs_thickness(V: torch.Tensor) -> None:
    """Correlation of each model's exit-wave intensity fluctuation with
    multislice's true fluctuation pattern, vs. thickness. `projection` is
    excluded: its intensity has exactly zero variance (see other-modes.md),
    so a correlation coefficient against it is undefined. Unlike the
    mean-intensity bias (which grows smoothly from thickness zero),
    firstborn/kinematic's *pattern* correlation is already near zero at
    the thinnest slab tested -- this figure is what motivates
    `_rytov_theta`'s Re/Im decomposition in the docs page."""
    palette = dict(zip(APPROX_MODELS, _deep_palette(len(APPROX_MODELS))))
    models = ["rytov", "firstborn", "kinematic"]
    thickness_angstrom = []
    correlations: dict[str, list[float]] = {m: [] for m in models}

    for nz in THICKNESS_STEPS_NZ:
        V_slice = V[:, :nz].contiguous()
        thickness_angstrom.append(nz * PIXEL_SIZE)
        ref_scat = Scattering(
            NXY, PIXEL_SIZE, VOLTAGE, scattering_model="multislice", progressbars=False
        ).to(DEVICE)
        ref_intensity = torch.abs(ref_scat(V_slice)) ** 2
        ref_c = (ref_intensity - ref_intensity.mean()).flatten()

        for model in models:
            scat = Scattering(
                NXY,
                PIXEL_SIZE,
                VOLTAGE,
                scattering_model=model,
                nz=nz,
                progressbars=False,
            ).to(DEVICE)
            intensity = torch.abs(scat(V_slice)) ** 2
            ic = (intensity - intensity.mean()).flatten()
            corr = torch.corrcoef(torch.stack([ic, ref_c]))[0, 1].item()
            correlations[model].append(corr)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    for model in models:
        ax.plot(
            thickness_angstrom,
            correlations[model],
            color=palette[model],
            marker="o",
            markersize=4,
            label=model,
        )
    ax.set_xlabel("Specimen thickness (Å)")
    ax.set_ylabel("Correlation with multislice's true intensity pattern")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Spatial pattern fidelity vs. thickness", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "scattering-pattern-correlation-vs-thickness.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_theta_real_imag_split(V: torch.Tensor) -> None:
    """Std. dev. of Re(Theta) and Im(Theta) vs. thickness. Theta is the
    complex quantity common to both rytov (exp(i*Theta)) and firstborn
    (1 + i*Theta) -- see `_rytov_theta`. Re(Theta) (ordinary projected
    phase) dominates Im(Theta) (the part that actually sets true
    intensity) at every thickness tested here, which is why firstborn's
    |1+i*Theta|^2 = 1 - 2*Im(Theta) + |Theta|^2 is swamped by the spurious
    Re(Theta)^2 term with no counterpart in the true |exp(i*Theta)|^2 =
    exp(-2*Im(Theta))."""
    palette = _deep_palette(2)
    thickness_angstrom = []
    std_re, std_im = [], []
    for nz in THICKNESS_STEPS_NZ:
        V_slice = V[:, :nz].contiguous()
        thickness_angstrom.append(nz * PIXEL_SIZE)
        theta = _rytov_theta(V_slice, nz)
        std_re.append(theta.real.std().item())
        std_im.append(theta.imag.std().item())

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    ax.plot(
        thickness_angstrom,
        std_re,
        color=palette[0],
        marker="o",
        markersize=4,
        label="Re(Θ)",
    )
    ax.plot(
        thickness_angstrom,
        std_im,
        color=palette[1],
        marker="o",
        markersize=4,
        label="Im(Θ)",
    )
    ax.set_xlabel("Specimen thickness (Å)")
    ax.set_ylabel("Standard deviation")
    ax.set_title("Re(Θ) vs. Im(Θ): phase dominates absorption", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "scattering-theta-real-imag-split.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    V = _ice_slab()
    figure_multislice_trace(V)
    figure_klim_bandlimit(V)
    thickness_angstrom, errors, means = _accuracy_sweep(V)
    figure_accuracy_vs_thickness(thickness_angstrom, errors)
    figure_mean_intensity_vs_thickness(thickness_angstrom, means)
    figure_mode_intensity_maps(V)
    figure_pattern_correlation_vs_thickness(V)
    figure_theta_real_imag_split(V)


if __name__ == "__main__":
    main()
