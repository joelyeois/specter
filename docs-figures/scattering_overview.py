"""
Generate the figure for docs/concepts/scattering/index.md.

Validates `specter.constants.energy_to_wavelength` and
`interaction_parameter` against the standard physical checkpoint
(lambda(300 kV) ~= 1.969 pm), and shows how both quantities behave across
the accelerating voltages SPECTER is actually used at.

Run with: uv run python docs-figures/scattering_overview.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from specter.constants import energy_to_wavelength, interaction_parameter
from specter.plots import _deep_palette

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"

CHECKPOINTS = [100.0, 200.0, 300.0]


def figure_sigma_wavelength_vs_voltage() -> None:
    palette = _deep_palette(2)
    voltage = torch.linspace(60.0, 400.0, 400)
    wavelength = energy_to_wavelength(voltage) * 100  # Angstrom -> pm
    sigma = torch.as_tensor([interaction_parameter(float(v)) for v in voltage])

    fig, ax_lambda = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    ax_sigma = ax_lambda.twinx()

    ax_lambda.plot(voltage, wavelength, color=palette[0], label=r"$\lambda$")
    ax_sigma.plot(voltage, sigma, color=palette[1], linestyle="--", label=r"$\sigma$")

    for v in CHECKPOINTS:
        lam = float(energy_to_wavelength(v)) * 100
        ax_lambda.plot([v], [lam], marker="o", color=palette[0], zorder=5)
        ax_lambda.annotate(
            f"{v:g} kV\n{lam:.3f} pm",
            (v, lam),
            textcoords="offset points",
            xytext=(8, 10),
            fontsize=8,
            color=palette[0],
        )

    ax_lambda.set_xlabel("Accelerating voltage (kV)")
    ax_lambda.set_ylabel(r"Wavelength $\lambda$ (pm)", color=palette[0])
    ax_sigma.set_ylabel(
        r"Interaction parameter $\sigma$ (rad / V$\cdot$Å)", color=palette[1]
    )
    ax_lambda.tick_params(axis="y", colors=palette[0])
    ax_sigma.tick_params(axis="y", colors=palette[1])
    ax_lambda.grid(True, alpha=0.4)
    ax_lambda.set_xlim(60, 400)
    fig.tight_layout()
    path = OUT_DIR / "scattering-sigma-wavelength-vs-voltage.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")

    for v in CHECKPOINTS:
        lam_pm = float(energy_to_wavelength(v)) * 100
        sig = interaction_parameter(v)
        print(f"  {v:g} kV: lambda = {lam_pm:.4f} pm, sigma = {sig:.6f} rad/(V*A)")


def main() -> None:
    figure_sigma_wavelength_vs_voltage()


if __name__ == "__main__":
    main()
