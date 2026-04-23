"""
Generate a simulated cryo-ET tilt series from a pre-built tomogram volume and save as an .mrcs stack.

The input MRC file is treated as the scattering potential volume (e.g. from Polnet or
TomogramGenerator). Ice is optionally generated and blended in before simulation.

Usage:
    python demo-scripts/generate_tilt_series.py --mrc_path /path/to/tomo.mrc --voxel_size 3.0

Device options:
    --device cpu          Single CPU
    --device cuda         Single GPU (default: cuda)
    --device cuda:0       Specific GPU

Example (HPC):
    python demo-scripts/generate_tilt_series.py \\
        --mrc_path /path/to/tomo.mrc \\
        --voxel_size 3.0 \\
        --micrograph_size 2100 \\
        --energy 300 \\
        --dose_per_tilt 3.0 \\
        --min_tilt_angle -45 \\
        --max_tilt_angle 45 \\
        --n_tilts 61 \\
        --defocus 22000 \\
        --cs 2.7 \\
        --alpha 0.1 \\
        --tilt_axis y \\
        --scattering_model multislice \\
        --noise_model poisson \\
        --coincidence_radius 1.5 \\
        --num_frames 10 \\
        --add_ice True \\
        --algorithm_dx 0.5 \\
        --n_ice_blocks 8 \\
        --tomo_to_ice_ratio 0.75 \\
        --save_exitwaves True \\
        --device cuda:0 \\
        --output_dir ./output/ \\
        --filename tilt_series
"""

import argparse
import os
import time

from rich.console import Console
from rich.rule import Rule

_console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a cryo-ET tilt series from a tomogram MRC and save as .mrcs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input volume ---
    parser.add_argument(
        "--mrc_path",
        type=str,
        required=True,
        help="Path to the input MRC volume (Z, Y, X). (required)",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=3.0,
        help="Voxel size in Ångstrom.",
    )
    parser.add_argument(
        "--micrograph_size",
        type=int,
        default=None,
        help="Output micrograph size in pixels. Defaults to the XY dimension of the volume.",
    )

    # --- Microscope / physics ---
    parser.add_argument(
        "--energy",
        type=float,
        default=300.0,
        help="Electron beam energy in keV.",
    )
    parser.add_argument(
        "--dose_per_tilt",
        type=float,
        default=3.0,
        help="Dose per tilt angle in e⁻/Å².",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of movie frames per tilt.",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=2.0,
        help="Spherical aberration in mm (1–3 mm typical).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Amplitude contrast ratio.",
    )

    # --- Defocus ---
    parser.add_argument(
        "--defocus",
        type=float,
        default=22000.0,
        help="Defocus in Ångstrom (positive = underfocus).",
    )

    # --- Tilt geometry ---
    parser.add_argument(
        "--min_tilt_angle",
        type=float,
        default=-45.0,
        help="Minimum tilt angle in degrees.",
    )
    parser.add_argument(
        "--max_tilt_angle",
        type=float,
        default=45.0,
        help="Maximum tilt angle in degrees.",
    )
    parser.add_argument(
        "--n_tilts",
        type=int,
        default=61,
        help="Number of tilt angles (evenly spaced from min to max).",
    )
    parser.add_argument(
        "--tilt_axis",
        type=str,
        default="y",
        choices=["x", "y"],
        help="Tilt axis.",
    )

    # --- Models ---
    parser.add_argument(
        "--scattering_model",
        type=str,
        default="multislice",
        choices=["multislice", "firstborn", "projection", "ctf"],
        help="Scattering model.",
    )
    parser.add_argument(
        "--noise_model",
        type=str,
        default="poisson",
        choices=["poisson", "none"],
        help="Noise model. Use 'none' for no noise.",
    )
    parser.add_argument(
        "--coincidence_radius",
        type=float,
        default=1.5,
        help="Coincidence radius in Ångstrom for direct-detector modelling.",
    )

    # --- Ice ---
    parser.add_argument(
        "--add_ice",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="True|False",
        help="Generate and blend amorphous ice into the volume.",
    )
    parser.add_argument(
        "--algorithm_dx",
        type=float,
        default=0.5,
        help="Voxel size in Å at which to run the ice generation algorithm (most stable at 0.5 Å).",
    )
    parser.add_argument(
        "--n_ice_blocks",
        type=int,
        default=8,
        help="Number of unique ice blocks to generate and tile.",
    )
    parser.add_argument(
        "--tomo_to_ice_ratio",
        type=float,
        default=0.75,
        help="Scale factor for tomogram intensity relative to ice.",
    )

    # --- Post-processing ---
    parser.add_argument(
        "--normalize",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Normalize each tilt image to zero mean and unit std.",
    )
    parser.add_argument(
        "--save_exitwaves",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Save exit wave magnitude and phase as separate .mrcs files.",
    )

    # --- Compute ---
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use. Options: cpu | cuda | cuda:0.",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="tilt_series",
        help="Base name for output files (no extension).",
    )

    return parser.parse_args()


def _section(msg: str) -> None:
    _console.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def main() -> None:
    import logging

    import mrcfile
    import torch

    import specter
    from specter.cryosparc import create_micrograph_starfile
    from specter.imagegenerator import TiltSeriesGenerator
    from specter.icemaker import Icemaker

    args = parse_args()
    specter.set_verbosity(logging.INFO)

    t_start = time.perf_counter()

    # --- Load volume ---
    _section("Loading MRC volume")
    with mrcfile.open(args.mrc_path) as mrc:
        tomo = torch.as_tensor(mrc.data[...].copy())
    _console.print(f"  Volume shape: {tuple(tomo.shape)}  (Z, Y, X)")

    dx = args.voxel_size
    micrograph_size = (
        args.micrograph_size if args.micrograph_size is not None else tomo.shape[-1]
    )

    # --- Ice ---
    if args.add_ice:
        _section("Generating amorphous ice")
        _console.print(
            f"  algorithm_dx = {args.algorithm_dx:.2f} Å,  target_dx = {dx:.2f} Å,  n_blocks = {args.n_ice_blocks}"
        )
        im = Icemaker(n=256, dx=dx)
        ice = torch.squeeze(
            im.generate_big_ice_interpolate(
                (tomo.shape[0], tomo.shape[1], tomo.shape[2]),
                n_blocks=args.n_ice_blocks,
                algorithm_dx=args.algorithm_dx,
            )
        )
        ratio = args.tomo_to_ice_ratio
        tomo = tomo * torch.max(ice) * ratio + ice * (tomo < 0.1)

    # --- Build tilt series generator ---
    _section("Building TiltSeriesGenerator")
    cs_angstrom = args.cs * 1e7
    angles = torch.linspace(args.min_tilt_angle, args.max_tilt_angle, args.n_tilts)
    _console.print(
        f"  Tilt angles: {args.n_tilts} steps from {args.min_tilt_angle}° to {args.max_tilt_angle}°"
    )

    noise_model = None if args.noise_model == "none" else args.noise_model

    ctf_params = {
        "cs": torch.tensor([cs_angstrom]),
        "dfu": torch.tensor([args.defocus]),
    }

    model = TiltSeriesGenerator(
        tomo[None, ...],  # add batch dim: (1, Z, Y, X)
        micrograph_size,
        dx,
        ctf_params,
        args.energy,
        args.dose_per_tilt,
        angles=angles,
        scattering_model=args.scattering_model,
        noise_model=noise_model,
        alpha=args.alpha,
        tilt_axis=args.tilt_axis,
        coincidence_radius=args.coincidence_radius,
        num_frames=args.num_frames,
    ).to(args.device)

    # --- Generate ---
    _section(f"Generating tilt series on {args.device}")
    with torch.no_grad():
        images, exitwaves, cleans = model.generate_tilt_series(torch.tensor([0]))
    # images: (1, n_tilts, H, W)
    images = images[0].cpu()  # (n_tilts, H, W)
    exitwaves = exitwaves[0].cpu()  # (n_tilts, H, W) complex

    # --- Post-processing ---
    if args.normalize:
        _section("Normalizing")
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / std.clamp(min=1e-8)

    # --- Save ---
    _section("Saving")
    os.makedirs(args.output_dir, exist_ok=True)

    mrcs_path = os.path.join(args.output_dir, args.filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mrcs_path}")

    ctf_params_broadcast = {
        "cs": torch.full((args.n_tilts,), cs_angstrom),
        "dfu": torch.full((args.n_tilts,), args.defocus),
    }
    create_micrograph_starfile(
        n=args.n_tilts,
        energy=args.energy,
        pixel_size=dx,
        alpha=args.alpha,
        ctf_params=ctf_params_broadcast,
        folderpath=args.output_dir,
        filename=args.filename,
        dose_per_angstrom=args.dose_per_tilt,
        coincidence_radius=args.coincidence_radius,
        tilt_angles=angles,
    )

    if args.save_exitwaves:
        ew_prefix = "exitwave" if args.add_ice else "clean_exitwave"
        _section(f"Saving {ew_prefix.replace('_', ' ')}")
        mag_path = os.path.join(
            args.output_dir, f"{args.filename}_{ew_prefix}_magnitude.mrcs"
        )
        phase_path = os.path.join(
            args.output_dir, f"{args.filename}_{ew_prefix}_phase.mrcs"
        )
        with mrcfile.new(mag_path, overwrite=True) as mrc:
            mrc.set_data(exitwaves.abs().numpy().astype("float32"))
        _console.print(f"  [green]✓[/green] {mag_path}")
        with mrcfile.new(phase_path, overwrite=True) as mrc:
            mrc.set_data(exitwaves.angle().numpy().astype("float32"))
        _console.print(f"  [green]✓[/green] {phase_path}")

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
