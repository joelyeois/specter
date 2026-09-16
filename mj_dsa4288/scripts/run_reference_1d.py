"""
Reproduce the paper's 1-D result: reconstruction SNR and sample complexity vs SNR.

Usage (from the repo root, inside ``.venv``)::

    python mj_dsa4288/scripts/run_reference_1d.py [--config mj_dsa4288/configs/reference_1d.toml] [--quick]
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
from mj_dsa4288.mra.template import synthetic_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config("reference_1d")))
    parser.add_argument(
        "--quick", action="store_true", help="tiny run for smoke testing"
    )
    args = parser.parse_args()
    cfg = load_toml(args.config)

    t = cfg["template"]
    gen = torch.Generator().manual_seed(int(t.get("seed", 0)))
    theta = synthetic_template((int(t["d"]),), t["corr_length"], t["mean_offset"], gen)

    s = cfg["sweep"]
    out = RESULTS / "reference_1d"
    for estimator in s["estimators"]:
        sweep = SweepConfig(
            snr_values=[float(v) for v in s["snr_values"]],
            n_samples=int(s["n_samples"]) if not args.quick else 500,
            n_repeats=int(s["n_repeats"]) if not args.quick else 1,
            estimator=estimator,
            snr_convention=s["snr_convention"],
            seed=int(s["seed"]),
            em_iters=int(s["em_iters"]) if not args.quick else 20,
            device=s["device"],
            eps=float(cfg["sample_complexity"]["eps"]),
            n_min=int(cfg["sample_complexity"]["n_min"]),
            n_max=int(cfg["sample_complexity"]["n_max"]) if not args.quick else 2000,
        )
        rows = snr_sweep(theta, sweep)
        save_rows(rows, out / f"sweep_{estimator}.json", sweep)
        plot_sigworth(
            rows, out / f"sweep_{estimator}.png", s["split_snr"], f"1-D {estimator}"
        )
        print(estimator, "rec-SNR slopes:", summarise_slopes(rows, s["split_snr"]))
        if cfg["sample_complexity"]["enabled"]:
            sc_rows = sample_complexity_curve(theta, sweep)
            save_rows(sc_rows, out / f"sample_complexity_{estimator}.json", sweep)
            plot_sample_complexity(
                sc_rows, out / f"sample_complexity_{estimator}.png", f"1-D {estimator}"
            )
            print(
                estimator,
                "n-required slopes:",
                summarise_slopes(sc_rows, s["split_snr"], "n_required"),
            )
    print("results in", out)


if __name__ == "__main__":
    main()
