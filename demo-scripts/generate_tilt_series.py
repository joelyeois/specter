"""
Generate a simulated cryo-ET tilt series end-to-end, in one command: build a
specimen volume with `specter build tomogram` (MembraneTomogramGenerator),
then simulate the tilted acquisition with TiltSeriesGenerator via
`run_tilt_series`. Saves the result as .mrcs + .star.

This is a thin convenience wrapper chaining the same two pipeline functions
`specter build tomogram`/`specter simulate tiltseries` already use -- for
full control over every field (species, membranes, filaments, CTF,
envelopes, etc.), run those two CLI commands directly instead, or edit
--tomogram_config/--tilt_series_config themselves:

    specter build tomogram --config configs/tomogram.toml
    specter simulate tiltseries --config configs/tilt_series.toml \\
        --volume_path output/tomogram.mrc

Usage:
    python demo-scripts/generate_tilt_series.py

    python demo-scripts/generate_tilt_series.py \\
        --tomogram_config configs/tomogram.toml \\
        --tilt_series_config configs/tilt_series.toml \\
        --device cuda:0
"""

import argparse
import os
import time

from rich.console import Console
from rich.rule import Rule

_console = Console()


def parse_args() -> argparse.Namespace:
    from specter.config import REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Build a specimen volume (specter build tomogram) and "
        "simulate a tilt series from it (specter simulate tiltseries), "
        "back to back.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tomogram_config",
        type=str,
        default=str(REPO_ROOT / "configs" / "tomogram.toml"),
        help="TOML config for the specimen-building stage (TomogramConfig). "
        "Edit this file directly for species/membrane/filament changes -- "
        "there's no per-field CLI flag here, use `specter build tomogram` "
        "directly for that.",
    )
    parser.add_argument(
        "--tilt_series_config",
        type=str,
        default=str(REPO_ROOT / "configs" / "tilt_series.toml"),
        help="TOML config for the imaging stage (TiltSeriesConfig). Its own "
        "volume_path is overridden automatically to point at whatever the "
        "specimen-building stage just wrote.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help="Device for BOTH stages, e.g. cpu | cuda | cuda:0. Overrides "
        "each config's own `device` field.",
    )
    return parser.parse_args()


def _section(msg: str) -> None:
    _console.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def main() -> None:
    from specter.config import TiltSeriesConfig, TomogramConfig, load_config
    from specter.pipelines import run_build_tomogram, run_tilt_series

    args = parse_args()
    tomo_config = load_config(args.tomogram_config, TomogramConfig)
    tilt_config = load_config(args.tilt_series_config, TiltSeriesConfig)
    if hasattr(args, "device"):
        tomo_config.device = args.device
        tilt_config.device = args.device

    t_start = time.perf_counter()

    _section("Building specimen volume (specter build tomogram)")
    run_build_tomogram(tomo_config)

    _section("Simulating tilt series (specter simulate tiltseries)")
    tilt_config.volume_path = os.path.join(
        tomo_config.output_dir, tomo_config.filename + ".mrc"
    )
    run_tilt_series(tilt_config)

    elapsed = time.perf_counter() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        time_str = f"{h}h {m}m {s}s"
    elif m > 0:
        time_str = f"{m}m {s}s"
    else:
        time_str = f"{s}s"
    _console.print(f"\n[bold]Total time:[/bold] {time_str}")


if __name__ == "__main__":
    main()
