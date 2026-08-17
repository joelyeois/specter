"""
Generate the protein-based robustness-check figure for
docs/concepts/scattering/other-modes.md.

Every other figure on that page uses a `RandomIcemaker` ice slab --
amorphous, roughly homogeneous density. Real single-particle targets are
the opposite: sparse, sharply peaked atomic density with large empty
gaps. This script reruns the same error / mean-intensity-bias /
pattern-correlation sweep from `scattering_accuracy.py` on a real protein
potential (myoglobin, PDB 1mbo) built via the same `PotentialBuilder`
path `run_particle_stack` uses, to check whether the ice-slab conclusions
generalize to the specimen type SPECTER actually simulates most.

Run with: uv run python docs-figures/scattering_protein_comparison.py
Saves a PNG directly into docs/assets/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from specter.pdb import PDB
from specter.plots import _deep_palette
from specter.potential import PotentialBuilder
from specter.scattering import Scattering
from specter.specimen.membrane._placement import align_principal_axis_to_z
from specter.specimen.packing import estimate_protein_box_size

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"
PDB_CACHE = "specter-data/pdb"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOLTAGE = 300.0
PIXEL_SIZE = 2.0  # Angstrom, same as the ice-slab sweep for comparability
PDB_CODE = "1mbo"  # myoglobin: small, compact, real single-particle target

APPROX_MODELS = ["rytov", "firstborn", "kinematic", "projection"]


def _protein_volume() -> torch.Tensor:
    """Myoglobin's potential, principal axis aligned to Z, shape (1, Z, Y, X)."""
    pdb = PDB(PDB_CODE, savefolder=PDB_CACHE, verbose=False)
    n = estimate_protein_box_size(pdb.max_diameter, PIXEL_SIZE)
    builder = PotentialBuilder(
        n_xyz=n,
        dx=PIXEL_SIZE,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
    ).to(DEVICE)
    template = builder.forward(
        align_principal_axis_to_z(pdb.coordinates), method="analytic"
    ).to(DEVICE)
    return template.unsqueeze(0)  # (1, Z, Y, X)


def figure_protein_comparison() -> None:
    V = _protein_volume()
    nxy = V.shape[-1]
    nz_full = V.shape[1]
    # Denser sampling than the ice sweep since the protein's own extent is
    # much smaller (tens, not hundreds, of Angstrom).
    thickness_steps_nz = sorted(
        {max(2, round(nz_full * f)) for f in (0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0)}
    )

    thickness_A: list[float] = []
    errors: dict[str, list[float]] = {m: [] for m in APPROX_MODELS}
    means: dict[str, list[float]] = {m: [] for m in APPROX_MODELS}
    correlations: dict[str, list[float]] = {
        m: [] for m in ["rytov", "firstborn", "kinematic"]
    }

    for nz in thickness_steps_nz:
        V_slice = V[:, :nz].contiguous()
        thickness_A.append(nz * PIXEL_SIZE)

        ref_scat = Scattering(
            nxy, PIXEL_SIZE, VOLTAGE, scattering_model="multislice", progressbars=False
        ).to(DEVICE)
        ref_intensity = torch.abs(ref_scat(V_slice)) ** 2
        ref_c = (ref_intensity - ref_intensity.mean()).flatten()

        for model in APPROX_MODELS:
            scat = Scattering(
                nxy,
                PIXEL_SIZE,
                VOLTAGE,
                scattering_model=model,
                nz=nz,
                progressbars=False,
            ).to(DEVICE)
            intensity = torch.abs(scat(V_slice)) ** 2
            errors[model].append(
                (
                    torch.abs(intensity - ref_intensity).mean() / ref_intensity.mean()
                ).item()
            )
            means[model].append(intensity.mean().item())
            if model in correlations:
                ic = (intensity - intensity.mean()).flatten()
                c = (
                    torch.corrcoef(torch.stack([ic, ref_c]))[0, 1].item()
                    if ic.std() > 0
                    else float("nan")
                )
                correlations[model].append(c)

    palette = dict(zip(APPROX_MODELS, _deep_palette(len(APPROX_MODELS))))
    fig, (ax_err, ax_mean, ax_corr) = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=190)

    for model in APPROX_MODELS:
        ax_err.semilogy(
            thickness_A,
            errors[model],
            color=palette[model],
            marker="o",
            markersize=4,
            label=model,
        )
    ax_err.set_xlabel("Protein thickness (Å)")
    ax_err.set_ylabel("Relative error in |ψ|² vs. multislice")
    ax_err.set_title("Error vs. thickness", fontsize=10)
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    ax_mean.axhline(1.0, color="black", linewidth=1.0, label="multislice (ref)")
    for model in APPROX_MODELS:
        ax_mean.plot(
            thickness_A,
            means[model],
            color=palette[model],
            marker="o",
            markersize=4,
            label=model,
        )
    ax_mean.set_xlabel("Protein thickness (Å)")
    ax_mean.set_ylabel("Mean exit-wave intensity ⟨|ψ|²⟩")
    ax_mean.set_title("Energy conservation vs. thickness", fontsize=10)
    ax_mean.legend(fontsize=8)
    ax_mean.grid(True, alpha=0.3)

    ax_corr.axhline(0.0, color="gray", linewidth=0.8)
    for model in correlations:
        ax_corr.plot(
            thickness_A,
            correlations[model],
            color=palette[model],
            marker="o",
            markersize=4,
            label=model,
        )
    ax_corr.set_xlabel("Protein thickness (Å)")
    ax_corr.set_ylabel("Correlation with multislice's true pattern")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.set_title("Pattern fidelity vs. thickness", fontsize=10)
    ax_corr.legend(fontsize=8)
    ax_corr.grid(True, alpha=0.3)

    fig.suptitle(
        f"Same sweep, real protein instead of ice (myoglobin, {PDB_CODE.upper()}, "
        f"{nz_full * PIXEL_SIZE:g} Å full depth)",
        y=1.03,
    )
    fig.tight_layout()
    path = OUT_DIR / "scattering-protein-comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    for model in APPROX_MODELS:
        print(
            f"  {model:10s} final: error={errors[model][-1]:.4f} "
            f"mean={means[model][-1]:.4f}"
        )
    for model in correlations:
        print(f"  {model:10s} final corr={correlations[model][-1]:+.4f}")


if __name__ == "__main__":
    figure_protein_comparison()
