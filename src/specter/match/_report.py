"""Render the outcome of `specter match particles` as a figure and a Markdown page."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import torch

from specter.filters import butter

from ._metrics import (
    BAND_LABELS,
    MatchedPoseSNR,
    PoseAlignmentResult,
    TwinResult,
    radial_power_spectra,
)


@dataclass
class DerivedValue:
    """One derived simulation parameter and where it came from."""

    name: str
    value: Any
    source: str  # "metadata" | "detector table" | "probe" | "measured" | "fallback" | "fixed"
    note: str = ""


@dataclass
class MatchReport:
    """Everything `run_match` decided and measured, in one place."""

    pose: PoseAlignmentResult
    derived: list[DerivedValue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    band_ratio: list[float] = field(default_factory=list)
    edge_sim: list[float] = field(default_factory=list)
    edge_exp: list[float] = field(default_factory=list)
    annulus_sim: list[float] = field(default_factory=list)
    annulus_exp: list[float] = field(default_factory=list)
    ring_sim: float = float("nan")
    ring_exp: float = float("nan")
    snr: MatchedPoseSNR | None = None
    twin: TwinResult | None = None
    probe_scores: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    n_battery: int = 0
    pixel_size: float = float("nan")

    @property
    def verdict(self) -> str:
        """One line a reader can act on."""
        if not self.pose.passed:
            return "FAIL: poses are not aligned to the model; nothing downstream is meaningful."
        if self.snr is None:
            return "incomplete"
        excess = [r for r in self.snr.ratio[:3] if not math.isnan(r)]
        worst = max(excess) if excess else float("nan")
        if worst <= 3.0:
            return "match: simulated and experimental SNR agree within 3x at every scale coarser than 6.7 Å."
        return (
            f"residual: the simulation is {worst:.0f}x cleaner than the experiment in at least one "
            "band coarser than 6.7 Å. No parameter of the forward model closes this; see Warnings."
        )

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary for job.json."""
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "pose_alignment": vars(self.pose),
            "derived": {
                d.name: {"value": d.value, "source": d.source} for d in self.derived
            },
            "warnings": list(self.warnings),
            "band_power_ratio": self.band_ratio,
            "edge_band_means": {"sim": self.edge_sim, "exp": self.edge_exp},
            "annulus_std": {"sim": self.annulus_sim, "exp": self.annulus_exp},
            "water_ring_excess": {"sim": self.ring_sim, "exp": self.ring_exp},
            "probe_scores": self.probe_scores,
        }
        if self.snr is not None:
            out["matched_pose_snr"] = {
                "ratio_sim_over_exp": self.snr.ratio,
                "snr_sim": self.snr.snr_sim,
                "snr_exp": self.snr.snr_exp,
                "signal_plateau": self.snr.signal_plateau,
                "residual_bfactor": self.snr.residual_bfactor,
            }
        if self.twin is not None:
            out["twin"] = {
                "exp_vs_sim": self.twin.exp_vs_sim,
                "sim_vs_sim": self.twin.sim_vs_sim,
                "cohen_d": self.twin.cohen_d,
            }
        return out


def _f(x: float, nd: int = 2) -> str:
    return (
        "n/a"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.{nd}f}"
    )


def render_markdown(report: MatchReport, toml_name: str = "matched.toml") -> str:
    """The report as Markdown."""
    lines = ["# specter match particles", "", f"**Verdict:** {report.verdict}", ""]
    if report.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in report.warnings] + [""]
    lines += [
        "## Pose alignment",
        "",
        f"Matched-pair correlation {_f(report.pose.matched, 3)} against {_f(report.pose.shuffled, 3)} "
        f"for a shuffled pairing; {report.pose.fraction_above:.0%} of pairs above shuffled, "
        f"z = {_f(report.pose.z_score, 1)}. {'PASS' if report.pose.passed else 'FAIL'}.",
        "",
        "## Derived parameters",
        "",
        "| parameter | value | source | note |",
        "|---|---|---|---|",
    ]
    for d in report.derived:
        lines.append(f"| `{d.name}` | {d.value} | {d.source} | {d.note} |")
    lines += ["", f"Written to `{toml_name}`.", ""]
    if report.snr is not None:
        lines += [
            "## Matched-pose signal-to-noise ratio (simulated / experimental)",
            "",
            "| band | " + " | ".join(BAND_LABELS) + " |",
            "|---|" + "---|" * len(BAND_LABELS),
            "| ratio | " + " | ".join(_f(r) for r in report.snr.ratio) + " |",
            "| SNR sim | " + " | ".join(_f(r, 3) for r in report.snr.snr_sim) + " |",
            "| SNR exp | " + " | ".join(_f(r, 3) for r in report.snr.snr_exp) + " |",
            "",
            f"Residual envelope (Guinier, 10-4 Å): B = {_f(report.snr.residual_bfactor, 0)} Å². "
            f"Experimental signal amplitude at 33-12 Å relative to simulated: {_f(report.snr.signal_plateau)}.",
            "",
        ]
    if report.twin is not None:
        lines += [
            "## Twin test",
            "",
            f"Correlation of each experimental particle with its simulated twin {_f(report.twin.exp_vs_sim, 3)}, "
            f"of a second simulated seed with the same twin {_f(report.twin.sim_vs_sim, 3)}; "
            f"Cohen's d = {_f(report.twin.cohen_d)} (0 means indistinguishable).",
            "",
        ]
    lines += [
        "## Screens",
        "",
        "| statistic | simulated | experimental |",
        "|---|---|---|",
        f"| power fraction ratio per band | {' / '.join(_f(r) for r in report.band_ratio)} | 1 |",
        f"| edge-band means (0-2, 2-4, 4-8, 8-16 px) | {' / '.join(_f(v, 3) for v in report.edge_sim)} | {' / '.join(_f(v, 3) for v in report.edge_exp)} |",
        f"| low-passed std by annulus | {' / '.join(_f(v) for v in report.annulus_sim)} | {' / '.join(_f(v) for v in report.annulus_exp)} |",
        f"| 3.7 Å water-ring excess | {_f(report.ring_sim, 3)} | {_f(report.ring_exp, 3)} |",
        "",
    ]
    if report.probe_scores:
        lines += ["## Probes", ""]
        for name, scores in report.probe_scores.items():
            lines.append(
                f"- {name}: " + ", ".join(f"{v:g} -> {s:.3f}" for v, s in scores)
            )
        lines.append("")
    return "\n".join(lines)


def render_figure(
    report: MatchReport,
    sim: torch.Tensor,
    exp: torch.Tensor,
    path: str,
    n_gallery: int = 5,
) -> None:
    """One-page figure: matched-pair gallery, spectra, SNR ratio, twin histograms."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(16, 10), dpi=130)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.15], hspace=0.3)
    top = gridspec.GridSpecFromSubplotSpec(
        2, n_gallery + 1, subplot_spec=outer[0], wspace=0.05, hspace=0.08
    )
    lp_exp, lp_sim = butter(exp[:n_gallery].float()), butter(sim[:n_gallery].float())
    for row, (stack, label, full) in enumerate(
        ((lp_exp, "experimental", exp), (lp_sim, "simulated", sim))
    ):
        for col in range(n_gallery):
            ax = fig.add_subplot(top[row, col])
            ax.imshow(stack[col], cmap="gray")
            ax.set(xticks=[], yticks=[])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"particle {col}", fontsize=8)
        ax = fig.add_subplot(top[row, n_gallery])
        ax.imshow(full.float().mean(0), cmap="gray")
        ax.set(xticks=[], yticks=[])
        if row == 0:
            ax.set_title("mean of stack", fontsize=8)

    bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.5)
    kk, ps, pe = radial_power_spectra(sim, exp, report.pixel_size)
    ax = fig.add_subplot(bottom[0])
    ax.loglog(kk[1:], pe[1:], "k", label="experimental")
    ax.loglog(kk[1:], ps[1:], "r", label="simulated")
    ax2 = ax.twinx()
    ax2.semilogx(kk[1:], (ps[1:] / pe[1:]).numpy(), color="tab:blue", lw=1)
    ax2.axhline(1, color="tab:blue", ls="--", lw=0.8)
    ax2.set_ylim(0, 3)
    ax2.set_ylabel("ratio", color="tab:blue")
    ax.set_xlabel("spatial frequency (1/Å)")
    ax.set_ylabel("radial power (z-scored images)")
    ax.set_title("power spectra and their ratio")
    ax.legend(fontsize=8, loc="lower left")

    ax = fig.add_subplot(bottom[1])
    if report.snr is not None:
        keep = [
            i for i, r in enumerate(report.snr.ratio) if not math.isnan(r) and r > 0
        ]
        ax.bar(range(len(keep)), [report.snr.ratio[i] for i in keep], color="tab:red")
        ax.axhspan(0.5, 2.0, color="tab:green", alpha=0.15, label="within 2x")
        ax.set_yscale("log")
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels([BAND_LABELS[i] for i in keep], fontsize=8)
        ax.set_ylabel("SNR simulated / experimental")
        ax.legend(fontsize=8)
    ax.set_title("matched-pose SNR ratio per band")

    ax = fig.add_subplot(bottom[2])
    if report.twin is not None:
        ax.hist(
            report.twin.sim_vs_sim_values.numpy(),
            bins=30,
            alpha=0.6,
            label="sim vs sim (2nd seed)",
        )
        ax.hist(
            report.twin.exp_vs_sim_values.numpy(),
            bins=30,
            alpha=0.6,
            label="exp vs sim twin",
        )
        ax.set_xlabel("low-passed peak correlation")
        ax.legend(fontsize=8)
        ax.set_title(f"twin test, Cohen's d = {report.twin.cohen_d:.2f}")
    fig.suptitle(report.verdict, fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_report(
    report: MatchReport, sim: torch.Tensor, exp: torch.Tensor, out_dir: str
) -> tuple[str, str]:
    """Write ``match_report.md`` and ``match_report.png`` into ``out_dir``."""
    import os

    md_path = os.path.join(out_dir, "match_report.md")
    png_path = os.path.join(out_dir, "match_report.png")
    with open(md_path, "w") as fh:
        fh.write(render_markdown(report))
    render_figure(report, sim, exp, png_path)
    return md_path, png_path
