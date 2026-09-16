"""
Phase A: SPECTER template (single view, no physics) + cyclic shifts + Gaussian noise.

Usage::

    python mj_dsa4288/scripts/run_phase_a.py [--config mj_dsa4288/configs/phase_a.toml] [--quick]
"""

from __future__ import annotations

import argparse

import torch
from _common import RESULTS, default_config, load_toml

from mj_dsa4288.mra.experiment import (
    SweepConfig,
    plot_sample_complexity,
    plot_sigworth,
    sample_complexity_curve,
    save_rows,
    snr_sweep,
    summarise_slopes,
)
from mj_dsa4288.mra.template import synthetic_template, template_from_pdb


def build_template(t: dict, quick: bool) -> torch.Tensor:
    """Template from a PDB entry via SPECTER, or a synthetic generic image."""
    n = int(t["num_pixels"]) if not quick else 16
    if t.get("source", "pdb") == "pdb":
        return template_from_pdb(
            t["pdb_id"],
            num_pixels=n,
            pixel_size=float(t["pixel_size"]),
            quaternion=torch.tensor(t["quaternion"], dtype=torch.float32),
            voltage=float(t.get("voltage", 300.0)),
            parameterization=t.get("parameterization", "kirkland"),
        )
    gen = torch.Generator().manual_seed(int(t.get("seed", 0)))
    return synthetic_template((n, n), generator=gen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config("phase_a")))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_toml(args.config)

    theta = build_template(cfg["template"], args.quick)
    out = RESULTS / "phase_a"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(theta, out / "template.pt")

    s = cfg["sweep"]
    sweep = SweepConfig(
        snr_values=[float(v) for v in s["snr_values"]],
        n_samples=int(s["n_samples"]) if not args.quick else 300,
        n_repeats=int(s["n_repeats"]) if not args.quick else 1,
        estimator=s["estimator"],
        snr_convention=s["snr_convention"],
        rotations=bool(s.get("rotations", False)),
        seed=int(s["seed"]),
        em_iters=int(s["em_iters"]) if not args.quick else 15,
        em_batch_size=int(s["em_batch_size"]),
        device=s["device"],
        eps=float(cfg["sample_complexity"]["eps"]),
        n_min=int(cfg["sample_complexity"]["n_min"]),
        n_max=int(cfg["sample_complexity"]["n_max"]),
    )
    rows = snr_sweep(theta, sweep)
    save_rows(rows, out / "sweep.json", sweep)
    plot_sigworth(rows, out / "sweep.png", s["split_snr"], "Phase A")
    print("rec-SNR slopes:", summarise_slopes(rows, s["split_snr"]))
    if cfg["sample_complexity"]["enabled"] and not args.quick:
        sc_rows = sample_complexity_curve(theta, sweep)
        save_rows(sc_rows, out / "sample_complexity.json", sweep)
        plot_sample_complexity(sc_rows, out / "sample_complexity.png", "Phase A")
        print(
            "n-required slopes:",
            summarise_slopes(sc_rows, s["split_snr"], "n_required"),
        )
    print("results in", out)


if __name__ == "__main__":
    main()
