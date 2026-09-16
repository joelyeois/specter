"""Re-plot any saved sweep JSON: ``python mj_dsa4288/scripts/plot_results.py results/phase_a/sweep.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_toml  # noqa: F401  (ensures sys.path setup)

from mj_dsa4288.mra.experiment import (
    load_rows,
    plot_sample_complexity,
    plot_sigworth,
    summarise_slopes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument("--split", type=float, default=1.0)
    args = parser.parse_args()
    rows = load_rows(args.json)
    png = args.json.with_suffix(".png")
    if "n_required" in rows[0]:
        plot_sample_complexity(rows, png, args.json.stem)
        print(summarise_slopes(rows, args.split, "n_required"))
    else:
        plot_sigworth(rows, png, args.split, args.json.stem)
        print(summarise_slopes(rows, args.split))
    print("wrote", png)


if __name__ == "__main__":
    main()
