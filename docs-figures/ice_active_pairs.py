"""Reproduce the production-scale ice Fourier and timing comparison.

By default render the committed spectra and measured timings on CPU:
    uv run python docs-figures/ice_active_pairs.py

To recompute spectra from the six original validation cells on GPU:
    uv run python docs-figures/ice_active_pairs.py --recompute \
        --input /tmp/specter-ice-validation --device cuda:1

No optimization is rerun. Amplitude is mean(|F|)/sqrt(N); power is
independently averaged mean(|F|**2)/N, not squared mean amplitude.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def compute(args: argparse.Namespace) -> None:
    """Decode saved coordinates and recompute full 3D spectra on a GPU."""
    import torch

    from specter.arrays import soft_voxelize_coordinates
    from specter.ice._gradient import GradientSKIcemaker
    from specter.ice._library import decode_positions, default_ice_cache_dir

    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    gd = GradientSKIcemaker(n=256, dx=1.0, device=args.device, progressbars=False)
    k = np.arange(gd._n_rbins) * gd.dk
    target = gd._f_target_rad_1d.cpu().numpy() / np.sqrt(gd.n_molecules)
    bins = gd._r_bins
    count = gd._bin_count
    results = {}

    def read(path, keep_images=False):
        data = torch.load(path, map_location="cpu", weights_only=False)
        coords = data["positions"]
        if data.get("coord_encoding") == "int16_fixed":
            coords = decode_positions(coords, data["box_L"])
        else:
            coords = coords.float()
        with torch.no_grad():
            pos = coords.to(args.device)
            loss, amp = gd._sk_loss(pos, rep_strength=0, mlbop_strength=0)
            radial = (
                torch.zeros(gd._n_rbins, device=args.device).scatter_add_(
                    0, bins, amp.flatten()
                )
                / count
                / np.sqrt(len(pos))
            )
            power = (
                torch.zeros(gd._n_rbins, device=args.device).scatter_add_(
                    0, bins, amp.square().flatten()
                )
                / count
                / len(pos)
            )
            result = {
                "amplitude": radial.cpu().numpy(),
                "power": power.cpu().numpy(),
                "recomputed_loss": float(loss),
            }
            if keep_images:
                image = amp[128].square().cpu().numpy() / len(pos)
                image[128, 128] = np.nan  # omit DC in the displayed spectrum only
                vox = soft_voxelize_coordinates(
                    pos,
                    grid_shape=(256, 256, 256),
                    voxel_size=1.0,
                    device=args.device,
                    periodic=True,
                )
                # Same central 4 A slab and 80 A square for all saved cells.
                result["density"] = vox[126:130, 88:168, 88:168].mean(0).cpu().numpy()
                result["slice"] = image
        return result

    shipped = []
    for path in sorted(Path(default_ice_cache_dir()).glob("config_*.pt")):
        shipped.append(read(path))
    print(
        f"Computed full 256-cubed spectra for {len(shipped)} shipped cells", flush=True
    )
    shipped_amp = np.stack([x["amplitude"] for x in shipped])
    shipped_power = np.stack([x["power"] for x in shipped])
    records = {}
    for seed in (1000, 1001, 1002):
        for variant in ("original", "filtered"):
            name = f"{variant}-{seed}"
            results[name] = read(args.input / f"{name}.pt", keep_images=True)
            records[name] = json.loads((args.input / f"{name}.json").read_text())
            recorded = records[name]["metadata"]["sk_loss"]
            actual = results[name]["recomputed_loss"]
            if not np.isclose(recorded, actual, rtol=0.01, atol=2e-6):
                raise ValueError(f"Saved loss mismatch: {name}: {recorded} vs {actual}")
            print(
                f"{name}: saved loss {recorded:.8g}, recomputed {actual:.8g}",
                flush=True,
            )

    kernel = json.loads((args.input / "kernel.json").read_text())
    shipped_quality = json.loads((args.input / "shipped-quality.json").read_text())
    np.savez_compressed(
        args.output / "spectra.npz",
        k=k,
        target_amplitude=target,
        shipped_amplitude=shipped_amp,
        shipped_power=shipped_power,
        **{
            f"{name}_{key}": value
            for name, r in results.items()
            for key, value in r.items()
            if isinstance(value, np.ndarray)
        },
    )
    (args.output / "measurements.json").write_text(
        json.dumps(
            {
                "builds": records,
                "kernel": kernel,
                "shipped_quality": shipped_quality,
                "recomputed_losses": {
                    name: r["recomputed_loss"] for name, r in results.items()
                },
            },
            indent=2,
        )
        + "\n"
    )


def render(args: argparse.Namespace) -> None:
    """Render plots and CSVs from the committed measurements, without CUDA."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm

    with np.load(args.output / "spectra.npz") as arrays:
        data = {key: arrays[key] for key in arrays.files}
    measurements = json.loads((args.output / "measurements.json").read_text())
    k, target = data["k"], data["target_amplitude"]
    shipped_amp, shipped_power = data["shipped_amplitude"], data["shipped_power"]
    records = measurements["builds"]
    kernel = measurements["kernel"]
    shipped_quality = measurements["shipped_quality"]
    results = {
        name: {
            key: data[f"{name}_{key}"]
            for key in ("amplitude", "power", "density", "slice")
        }
        for name in records
    }
    colors = {"original": "#2563a6", "filtered": "#dd7029"}
    labels = {"original": "Before", "filtered": "After"}
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    mask = k > 0
    full = k > 0
    zoom = (k > 0) & (k <= 0.5)
    figures = []
    # Exact production spectral observable, residuals and independently averaged power.
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, layout="constrained")
    for col, seed in enumerate((1000, 1001, 1002)):
        for row, key in ((0, "amplitude"), (2, "power")):
            bank = shipped_amp if key == "amplitude" else shipped_power
            axes[row, col].fill_between(
                k[mask],
                bank.min(0)[mask],
                bank.max(0)[mask],
                color="#bbc2cb",
                alpha=0.40,
                label="20 shipped cells: min–max",
            )
        axes[0, col].plot(
            k[mask],
            target[mask],
            color="black",
            ls="--",
            lw=1.5,
            label="Production MD target",
        )
        for variant in ("original", "filtered"):
            r = results[f"{variant}-{seed}"]
            loss = records[f"{variant}-{seed}"]["metadata"]["sk_loss"]
            axes[0, col].plot(
                k[mask],
                r["amplitude"][mask],
                color=colors[variant],
                lw=1.3,
                label=f"{labels[variant]} (saved loss {loss:.5f})",
            )
            error = 100 * (r["amplitude"] - target) / np.maximum(target, 1e-8)
            axes[1, col].plot(k[mask], error[mask], color=colors[variant], lw=1.3)
            axes[2, col].plot(k[mask], r["power"][mask], color=colors[variant], lw=1.3)
        axes[0, col].set_title(f"Seed {seed}", weight="bold")
        axes[0, col].legend(fontsize=8, loc="upper right")
        axes[1, col].axhline(0, color="black", lw=0.8)
        axes[2, col].set_xlabel(r"Spatial frequency |k| (Å$^{-1}$)")
        for ax in axes[:, col]:
            ax.axvline(0.5, color="#666666", ls=":", lw=1)
            ax.axvspan(0.5, k[-1], color="#bbbbbb", alpha=0.12)
            ax.grid(alpha=0.16)
            ax.set_xlim(0, k[-1])
    axes[0, 0].set_ylabel(r"Shell-mean amplitude ⟨|F|⟩ / √N")
    axes[1, 0].set_ylabel("Amplitude deviation from target (%)")
    axes[2, 0].set_ylabel(r"Shell-mean power ⟨|F|²⟩ / N")
    for row in (0, 2):
        lim = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(0, lim)
    lim = max(max(abs(v) for v in ax.get_ylim()) for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(-lim, lim)
    fig.suptitle(
        "Saved production ice: Fourier spectra before and after\n"
        "256³ at 1 Å/voxel · 527,178 molecules · DC omitted from plots · shaded region: shells extend beyond axial Nyquist",
        fontsize=15,
    )
    fig.savefig(args.output / "fourier_profiles.png", dpi=160)
    figures.append(fig)

    # Matching display scales, unaltered pixels; density and central Fourier plane.
    all_density = np.concatenate([r["density"].flatten() for r in results.values()])
    all_slices = np.concatenate([r["slice"].flatten() for r in results.values()])
    density_max = np.quantile(all_density, 0.995)
    finite = all_slices[np.isfinite(all_slices) & (all_slices > 0)]
    power_min, power_max = np.quantile(finite, [0.01, 0.995])
    for seed in (1000, 1001, 1002):
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
        for col, variant in enumerate(("original", "filtered")):
            r = results[f"{variant}-{seed}"]
            density_artist = axes[0, col].imshow(
                r["density"],
                origin="lower",
                cmap="gray",
                vmin=0,
                vmax=density_max,
                extent=[-40, 40, -40, 40],
            )
            spectrum_artist = axes[1, col].imshow(
                r["slice"],
                origin="lower",
                cmap="magma",
                norm=LogNorm(vmin=power_min, vmax=power_max),
                extent=[
                    -0.5 - 0.5 / 256,
                    0.5 - 0.5 / 256,
                    -0.5 - 0.5 / 256,
                    0.5 - 0.5 / 256,
                ],
            )
            axes[0, col].set_title(
                f"{labels[variant]}: 4 Å mean-density slab", weight="bold"
            )
            axes[1, col].set_title(f"{labels[variant]}: Fourier power at k_z = 0")
            axes[0, col].set_xlabel("x (Å)")
            axes[0, col].set_ylabel("y (Å)")
            axes[1, col].set_xlabel(r"k_x (Å$^{-1}$)")
            axes[1, col].set_ylabel(r"k_y (Å$^{-1}$)")
        fig.colorbar(
            density_artist, ax=list(axes[0]), shrink=0.85, label="Molecules / Å³"
        )
        fig.colorbar(
            spectrum_artist,
            ax=list(axes[1]),
            shrink=0.85,
            label=r"Fourier power |F|² / N (log color scale)",
        )
        fig.suptitle(
            f"Seed {seed}: saved ice cells before and after\n"
            "Common scales across all seeds · same spatial slab / Fourier plane · different optimizer trajectories",
            fontsize=14,
        )
        fig.savefig(args.output / f"ice_and_fourier_seed_{seed}.png", dpi=150)
        figures.append(fig)

    # Actual measured build wall times, iteration counts and identical-input kernel trials.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    seeds = (1000, 1001, 1002)
    x = np.arange(3)
    width = 0.34
    for variant, offset in (("original", -width / 2), ("filtered", width / 2)):
        metas = [records[f"{variant}-{seed}"]["metadata"] for seed in seeds]
        values = [m["wall_time"] for m in metas]
        bars = axes[0, 0].bar(
            x + offset, values, width, color=colors[variant], label=labels[variant]
        )
        axes[0, 0].bar_label(bars, fmt="%.1f s", fontsize=9, padding=3)
        bars = axes[0, 1].bar(
            x + offset,
            [m["n_steps_actual"] for m in metas],
            width,
            color=colors[variant],
        )
        axes[0, 1].bar_label(bars, fontsize=9, padding=3)
        times = [c["variants"][variant]["median_s"] * 1000 for c in kernel["cases"]]
        bars = axes[1, 0].bar(x + offset, times, width, color=colors[variant])
        axes[1, 0].bar_label(bars, fmt="%.2f ms", fontsize=9, padding=3)
    axes[0, 0].set_title("Complete build wall time (different stopping points)")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 650)
    axes[0, 1].set_title("Completed L-BFGS steps (ceiling: 250)")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].set_ylim(0, 280)
    axes[1, 0].set_title("Same-input ML-BOP energy + backward")
    axes[1, 0].set_ylabel("Milliseconds (median of 5 warmed trials)")
    axes[1, 0].set_ylim(0, 90)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xticks(x, [str(seed) for seed in seeds])
        ax.set_xlabel("Seed")
    axes[1, 0].set_xticks(x, ["Shipped 1000", "Shipped 1001", "Random start"])
    losses = np.array([r["sk_loss"] for r in shipped_quality["configs"]])
    axes[1, 1].scatter(
        losses, np.zeros(len(losses)), color="#8d96a3", s=28, label="20 shipped"
    )
    for row, variant in ((1, "original"), (2, "filtered")):
        for seed in seeds:
            loss = records[f"{variant}-{seed}"]["metadata"]["sk_loss"]
            axes[1, 1].scatter(loss, row, color=colors[variant], s=55)
            axes[1, 1].annotate(
                str(seed),
                (loss, row),
                xytext=(0, 22 if seed == 1002 else 8),
                textcoords="offset points",
                fontsize=9,
                ha="center",
            )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yticks([0, 1, 2], ["Shipped", "Before", "After"])
    axes[1, 1].set_ylim(-0.4, 2.7)
    axes[1, 1].set_title("Saved-coordinate quality (lower loss is better)")
    axes[1, 1].set_xlabel("Production spectral MSE (log axis)")
    for ax in axes.flatten():
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Measured NVIDIA L40 performance at production scale\n"
        "256³ · dx = 1 Å · 527,178 molecules · unchanged dtypes and optimization recipe",
        fontsize=15,
    )
    fig.savefig(args.output / "timing_and_quality.png", dpi=160)
    figures.append(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    for col, seed in enumerate((1000, 1001, 1002)):
        top, bottom = axes[:, col]
        top.fill_between(
            k[full],
            shipped_amp.min(0)[full],
            shipped_amp.max(0)[full],
            color="#bbc2cb",
            alpha=0.5,
            label="20 shipped cells: min–max",
        )
        top.plot(
            k[full], target[full], color="black", ls="--", lw=1.5, label="MD target"
        )
        for variant in ("original", "filtered"):
            curve = results[f"{variant}-{seed}"]["amplitude"]
            top.plot(
                k[full],
                curve[full],
                color=colors[variant],
                lw=1.3,
                label=labels[variant],
            )
            residual = 100 * (curve[zoom] - target[zoom]) / target[zoom]
            bottom.plot(k[zoom], residual, color=colors[variant], lw=1.2)
        top.set_title(f"Seed {seed}", weight="bold")
        top.legend(fontsize=8)
        top.set_xlim(0, k[-1])
        top.set_ylim(0, 0.88)
        top.axvline(0.5, color="#666666", ls=":", lw=1)
        top.axvspan(0.5, k[-1], color="#bbbbbb", alpha=0.12)
        top.set_xlabel(r"|k| (Å$^{-1}$); shaded: incomplete spherical shells")
        bottom.set_xlim(0, 0.5)
        bottom.set_ylim(-0.025, 0.025)
        bottom.axhline(0, color="black", lw=0.8)
        bottom.set_xlabel(r"|k| (Å$^{-1}$); zoom to complete spherical shells")
        for ax in (top, bottom):
            ax.grid(alpha=0.17)
    axes[0, 0].set_ylabel(r"Radial amplitude ⟨|F|⟩ / √N")
    axes[1, 0].set_ylabel("Deviation from MD target (%)")
    fig.suptitle(
        "Before / after Fourier spectra of the saved production ice\n"
        "256³ at 1 Å/voxel · 527,178 molecules · DC omitted · residuals magnified below",
        fontsize=15,
    )
    fig.savefig(args.output / "fourier_comparison.png", dpi=160)
    with PdfPages(args.output / "fourier_comparison.pdf") as pdf:
        pdf.savefig(fig)
    figures.insert(0, fig)

    with PdfPages(args.output / "ice_comparison.pdf") as pdf:
        for fig in figures:
            pdf.savefig(fig)
    for fig in figures:
        plt.close(fig)

    csv = [
        "seed,before_seconds,after_seconds,speedup,before_steps,after_steps,before_saved_loss,after_saved_loss"
    ]
    for seed in seeds:
        a, b = [records[f"{v}-{seed}"]["metadata"] for v in ("original", "filtered")]
        csv.append(
            f"{seed},{a['wall_time']},{b['wall_time']},{a['wall_time'] / b['wall_time']},"
            f"{a['n_steps_actual']},{b['n_steps_actual']},{a['sk_loss']},{b['sk_loss']}"
        )
    (args.output / "timings.csv").write_text("\n".join(csv) + "\n")

    # Tabular spectral data for inspection without Python's NPZ loader.
    columns = [k, target, shipped_amp.min(0), shipped_amp.max(0)]
    headers = [
        "k_A^-1",
        "target_mean_amplitude_sqrtN",
        "shipped_min_mean_amplitude_sqrtN",
        "shipped_max_mean_amplitude_sqrtN",
    ]
    for name, r in results.items():
        columns += [r["amplitude"], r["power"]]
        headers += [name + "_mean_amplitude_sqrtN", name + "_mean_power_N"]
    np.savetxt(
        args.output / "spectra.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
    print(f"Saved figures, PDF, measurements and CSVs to {args.output}", flush=True)

    rows = [
        "variant,seed,max_target_relative_amplitude_error_pct_k_le_0p5,squared_residual_fraction_k_gt_0p5"
    ]
    for seed in (1000, 1001, 1002):
        for variant in ("original", "filtered"):
            residual = results[f"{variant}-{seed}"]["amplitude"] - target
            maximum = float(100 * np.max(abs(residual[zoom] / target[zoom])))
            fraction = float(np.sum(residual[k > 0.5] ** 2) / np.sum(residual**2))
            rows.append(f"{variant},{seed},{maximum},{fraction}")
    (args.output / "spectral_summary.csv").write_text("\n".join(rows) + "\n")


def main() -> None:
    """Render committed data, optionally refreshing spectra from saved cells."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/specter-ice-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("/tmp/specter-ice-validation")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/ice-active-pairs",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute spectra from original saved cells; otherwise use committed data.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.recompute:
        compute(args)
    render(args)


if __name__ == "__main__":
    main()
