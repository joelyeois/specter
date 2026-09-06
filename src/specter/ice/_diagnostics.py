"""
Quality-check figures for an ice library: ML-BOP energy and structure per
cached config, and every config's S(k) profile against the MD target.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.figure
import torch


from ..arrays import (
    soft_voxelize_coordinates,
)
from ._energy import MLBOP

if TYPE_CHECKING:
    from ._bank import IceBank


def plot_ice_bank_diagnostics(
    bank: IceBank, save_path: str | None = None, show: bool = True
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """See :meth:`IceBank.plot_diagnostics`."""
    import matplotlib.pyplot as plt

    from ..arrays import radial_profile_3d
    from ..fft import fft3
    from ._kernels import compute_native_target

    model = MLBOP(device=bank.device)
    labels, energies, sk_profiles, dks = [], [], [], []
    for path, config in zip(bank._config_paths, bank._configs):
        pos, box_L, n, dx = (
            config["positions"],
            config["box_L"],
            config["n"],
            config["dx"],
        )
        labels.append(os.path.basename(path))
        with torch.no_grad():
            result = model.compute_energy(
                pos.to(bank.device), box_size=(box_L,) * 3, pbc=True
            )
        energies.append({k: v.item() for k, v in result.items()})
        vox = soft_voxelize_coordinates(
            pos, grid_shape=(n, n, n), voxel_size=dx, periodic=True
        )
        amp = torch.abs(fft3(vox, shift=True)) / (pos.shape[0] ** 0.5)
        _, prof = radial_profile_3d(amp, return_r=True)
        sk_profiles.append(prof)
        dks.append(1 / n / dx)

    ref_n, ref_dx = bank._configs[0]["n"], bank._configs[0]["dx"]
    native_k, native_f = compute_native_target(n=ref_n, dx=ref_dx)

    fig1, axes = plt.subplots(1, 3, figsize=(max(6, 0.4 * len(labels) + 4), 4))
    x = list(range(len(labels)))
    for ax, key, title, target in [
        (axes[0], "E_per_atom", "E/atom (eV)", -0.413),
        (axes[1], "rij_var", "O-O distance variance", None),
        (axes[2], "theta_var", "O-O-O angle variance", None),
    ]:
        ax.bar(x, [e[key] for e in energies])
        if target is not None:
            ax.axhline(target, ls="--", color="k", lw=1, label=f"MD target ({target})")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=6, rotation=90)
    fig1.suptitle(f"IceBank cache diagnostics -- {len(labels)} configs")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for i, (prof, dk) in enumerate(zip(sk_profiles, dks)):
        k_axis = torch.arange(len(prof)) * dk
        ax2.plot(k_axis[1:], prof[1:], alpha=0.5, lw=1, color="C0")
    ax2.plot(
        native_k[native_k > 0],
        native_f[native_k > 0],
        "k--",
        lw=2,
        label="native MD target",
    )
    ax2.set_xlabel("k (1/Å)")
    ax2.set_ylabel("radial |F(k)|")
    ax2.set_xlim(0, 1.0)
    ax2.legend()
    ax2.set_title(f"S(k) radial profile -- {len(labels)} cached configs")
    fig2.tight_layout()

    if save_path is not None:
        fig1.savefig(f"{save_path}_energy.png", dpi=130)
        fig2.savefig(f"{save_path}_sk.png", dpi=130)
    if show:
        plt.show()
    return fig1, fig2
