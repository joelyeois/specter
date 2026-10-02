"""Plot complete gold reconstructions, Fourier spectra, FSCs and measurements."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch
from scipy.fft import fft2, fftshift

from specter.fft import fourier_shell_correlation
from specter.plots import fsc_resolution


def read_json(path):
    return json.loads(path.read_text())


def read_volume(path):
    with mrcfile.open(path) as m:
        return torch.from_numpy(m.data.copy()), float(m.voxel_size.x)


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=170)
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference-512", default="/scratch/loh/joel/empiar-11377/8b0x.mrc")
    p.add_argument("--boxes", nargs="+", choices=["256", "512"], default=["256", "512"])
    a = p.parse_args()
    torch.set_num_threads(4)
    root = Path(a.data)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("256", "before-256-123-clean", "after-256-123", 10000, None),
        (
            "512",
            "before-512-123-clean",
            "after-512-123",
            2000,
            a.reference_512,
        ),
    ]
    cases = [case for case in cases if case[0] in a.boxes]
    fig, axes = plt.subplots(
        len(cases),
        6,
        figsize=(20, 4.2 * len(cases)),
        constrained_layout=True,
        squeeze=False,
    )
    summary = {"cases": []}
    colors = ["#386CB0", "#F28E2B"]
    for row, (box, old_name, new_name, count, reference) in enumerate(cases):
        folders = [root / old_name, root / new_name]
        jobs = [next((f / "reconstructions").glob("J*")) for f in folders]
        metrics = [read_json(f / "measurement.json") for f in folders]
        halves = [
            [read_volume(j / f"volume_{h}.mrc")[0] for h in ["A", "B"]] for j in jobs
        ]
        dx = read_volume(jobs[0] / "volume_A.mrc")[1]
        means = [(h[0] + h[1]) / 2 for h in halves]
        diff = means[1] - means[0]
        errors = []
        for h in range(2):
            before, after = halves[0][h], halves[1][h]
            errors.append(
                dict(
                    halfset="AB"[h],
                    bitwise_equal=torch.equal(before, after),
                    relative_l2=float(
                        torch.linalg.vector_norm((after - before).double())
                        / torch.linalg.vector_norm(before.double())
                    ),
                    max_abs_V=float((after - before).abs().max()),
                )
            )
        vmax = max(float(m[m.shape[0] // 2].abs().quantile(0.995)) for m in means)
        for col, m in enumerate(means):
            axes[row, col].imshow(
                m[m.shape[0] // 2], cmap="coolwarm", vmin=-vmax, vmax=vmax
            )
            axes[row, col].set_title(
                f"{'Before' if col == 0 else 'After'}: central XY slice"
            )
            axes[row, col].axis("off")
        axes[row, 0].text(
            -0.1,
            0.5,
            f"{count:,} particles\n{box}³ map",
            rotation=90,
            ha="center",
            va="center",
            transform=axes[row, 0].transAxes,
        )
        peak = float(diff.abs().max())
        im = axes[row, 2].imshow(
            diff[diff.shape[0] // 2],
            cmap="coolwarm",
            vmin=-max(peak, 1e-12),
            vmax=max(peak, 1e-12),
        )
        axes[row, 2].set_title(f"After − before (own scale)\nmax |ΔV|={peak:.2g} V")
        axes[row, 2].axis("off")
        fig.colorbar(im, ax=axes[row, 2], shrink=0.7)
        spectra = []
        for m in means:
            # The Fourier plane kz=0, up to a shared normalization, is the
            # FFT of the complete map's mean-Z projection. Remove DC only
            # for display; the reconstructions themselves are unchanged.
            projection = m.mean(0).numpy()
            spectra.append(
                np.log10(
                    np.abs(fftshift(fft2(projection - projection.mean(), workers=4)))
                    ** 2
                    + 1e-20
                )
            )
        freq = np.fft.fftshift(np.fft.fftfreq(int(box), dx))
        ky, kx = np.meshgrid(freq, freq, indexing="ij")
        inside = kx**2 + ky**2 <= (1 / (2 * dx)) ** 2
        lo, hi = np.percentile(np.concatenate([s[inside] for s in spectra]), [3, 99.7])
        spectrum_cmap = plt.get_cmap("magma").copy()
        spectrum_cmap.set_bad("black")
        for col, spectrum in enumerate(spectra, start=3):
            spectrum_im = axes[row, col].imshow(
                np.ma.masked_where(~inside, spectrum),
                cmap=spectrum_cmap,
                vmin=lo,
                vmax=hi,
                extent=(-1 / (2 * dx), 1 / (2 * dx), -1 / (2 * dx), 1 / (2 * dx)),
            )
            axes[row, col].set_title(
                f"{'Before' if col == 3 else 'After'} Fourier power, kz=0"
            )
            axes[row, col].set_xlabel("kx (Å⁻¹)")
            axes[row, col].set_ylabel("ky (Å⁻¹)")
        fig.colorbar(
            spectrum_im,
            ax=list(axes[row, 3:5]),
            shrink=0.7,
            label="log₁₀ Fourier power (shared scale)",
        )
        fsc_curves = []
        resolutions = []
        for col, h in enumerate(halves):
            k, fsc = fourier_shell_correlation(h[0], h[1], pixel_size=dx)
            resolution = fsc_resolution(k, fsc, 0.143, k_max=1 / (2 * dx))
            resolutions.append(resolution)
            axes[row, 5].plot(
                k,
                fsc,
                label=f"{'Before' if col == 0 else 'After'}: {resolution}",
                color=colors[col],
                linestyle="-" if col == 0 else "--",
            )
            fsc_curves.append(fsc[k <= 1 / (2 * dx)])
        axes[row, 5].axhline(0.143, color="gray", linestyle=":", label="FSC 0.143")
        axes[row, 5].set(
            xlim=(0, 1 / (2 * dx)),
            ylim=(-0.05, 1.05),
            xlabel="Spatial frequency (Å⁻¹)",
            ylabel="Unmasked halfmap FSC",
        )
        axes[row, 5].legend(fontsize=8)
        reference_resolutions = []
        if reference:
            ref, refdx = read_volume(Path(reference))
            assert abs(refdx - dx) < 1e-5
            for m in means:
                k, fsc = fourier_shell_correlation(m, ref, pixel_size=dx)
                reference_resolutions.append(
                    fsc_resolution(k, fsc, 0.5, k_max=1 / (2 * dx))
                )
        worker_metrics = [
            [read_json(f / f"measurement_{h}.json") for h in ["A", "B"]]
            for f in folders
        ]
        loading = [
            read_json(root / f"loading-{variant}-{box}.json")
            for variant in ["before", "after"]
        ]
        hashes_equal = all(
            loading[0][key] == loading[1][key]
            for key in loading[0]
            if key.endswith("sha256")
        )
        summary["cases"].append(
            dict(
                box=int(box),
                particles=count,
                pixel_size_A=dx,
                halfmaps=errors,
                halfmap_resolution=resolutions,
                map_to_model_resolution=reference_resolutions,
                max_halfmap_fsc_difference=float(
                    (fsc_curves[0] - fsc_curves[1]).abs().max()
                ),
                preprocessed_images_and_parameters_bitwise_equal=hashes_equal,
                cli_seconds=[m["cli_seconds"] for m in metrics],
                cli_speedup=metrics[0]["cli_seconds"] / metrics[1]["cli_seconds"],
                tree_peak_pss_GiB=[
                    m["host_peak"]["tree_pss_bytes"] / 2**30 for m in metrics
                ],
                worker_peak_gpu_allocated_GiB=[
                    max(h["peak_allocated_bytes"] for h in group) / 2**30
                    for group in worker_metrics
                ],
                worker_peak_gpu_reserved_GiB=[
                    max(h["peak_reserved_bytes"] for h in group) / 2**30
                    for group in worker_metrics
                ],
                worker_peak_driver_GiB=[
                    max(h["driver_peak_bytes"] for h in group) / 2**30
                    for group in worker_metrics
                ],
                worker_peak_loading_rss_GiB=[
                    max(h["loading_peak_rss_bytes"] for h in group) / 2**30
                    for group in worker_metrics
                ],
                fit_seconds=[
                    sum(h["fit_seconds"] for h in group) for group in worker_metrics
                ],
                training_step_median_ms=[
                    float(np.median([s for h in group for s in h["step_seconds"]]))
                    * 1000
                    for group in worker_metrics
                ],
                loading_all_particles_peak_rss_GiB=[
                    m["loading_peak_rss_bytes"] / 2**30 for m in loading
                ],
                measurement_count="One complete before/after gold run per box; no repeated whole-CLI speedup claim.",
            )
        )
    save(fig, output, "maps_fourier_and_fsc")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    labels = [f"{c['particles']:,} × {c['box']}" for c in summary["cases"]]
    x = np.arange(len(cases))
    w = 0.36
    for i, variant in enumerate(["Before", "After"]):
        offset = (i - 0.5) * w
        for col, (key, label) in enumerate(
            [
                ("cli_seconds", "Complete gold CLI (minutes)"),
                ("tree_peak_pss_GiB", "Peak host memory, process-tree PSS (GiB)"),
                (
                    "worker_peak_gpu_allocated_GiB",
                    "Peak GPU allocation per halfset (GiB)",
                ),
            ]
        ):
            values = [
                c[key][i] / 60 if key == "cli_seconds" else c[key][i]
                for c in summary["cases"]
            ]
            bars = axes[col].bar(x + offset, values, w, label=variant, color=colors[i])
            axes[col].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
            axes[col].set(ylabel=label, xticks=x, xticklabels=labels)
            axes[col].set_ylim(
                0,
                max(
                    values
                    + [
                        c[key][0] / 60 if key == "cli_seconds" else c[key][0]
                        for c in summary["cases"]
                    ]
                )
                * 1.2,
            )
    axes[0].legend()
    fig.suptitle(
        "Five epochs, gold halfsets, batch 3, 16-mixed; same GPU per paired case"
    )
    save(fig, output, "timing_and_memory")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
