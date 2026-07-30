"""
Generate a simulated cryo-ET tilt series end-to-end: place proteins/membranes with
CryoETSpecimenGenerator (polnet-driven placement + specter's own PotentialBuilder for
density), then blend in ice and simulate the tilted acquisition with TiltSeriesGenerator.
Saves the result as an .mrcs stack.

Mirrors demo-notebooks/create_tilt_series/tilt-series-generator.ipynb.

Usage:
    python demo-scripts/generate_tilt_series.py --config configs/tilt_series.toml

Parameters are loaded from --config (default: configs/tilt_series.toml); any flag
below overrides the corresponding scalar field in that file. protein_specs and
membrane_specs are TOML-only (list-of-tables): edit --config directly to change
species/populations, there is no per-species CLI flag.

Device options:
    --device cpu          Single CPU
    --device cuda         Single GPU (default device)
    --device cuda:0       Specific single GPU

Example (HPC):
    python demo-scripts/generate_tilt_series.py \\
        --config configs/tilt_series.toml \\
        --target_shape 184 630 630 \\
        --target_v_size 5.0 \\
        --voltage 300 \\
        --dose_per_tilt 3.0 \\
        --min_tilt_angle -45 \\
        --max_tilt_angle 45 \\
        --n_tilts 61 \\
        --defocus 22000 \\
        --cs 2.0 \\
        --alpha 0.1 \\
        --tilt_axis y \\
        --scattering_model multislice \\
        --noise_model poisson \\
        --coincidence_radius 1.5 \\
        --num_frames 10 \\
        --ice_model gd \\
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
    from specter.config import REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Simulate a cryo-ET tilt series (polnet placement + specter "
        "density + ice + tilted acquisition) and save as .mrcs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "tilt_series.toml"),
        help="Path to a TOML config file. Every other flag below overrides a field in it.",
    )

    # --- Specimen placement (CryoETSpecimenGenerator) ---
    parser.add_argument(
        "--pdb_savefolder",
        type=str,
        default=argparse.SUPPRESS,
        help="Folder to cache downloaded PDB files. Overrides --config.",
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=argparse.SUPPRESS,
        help="Output specimen volume shape in voxels. Overrides --config.",
    )
    parser.add_argument(
        "--target_v_size",
        type=float,
        default=argparse.SUPPRESS,
        help="Target voxel size in Ångstrom. Overrides --config.",
    )
    parser.add_argument(
        "--low_res_v_size",
        type=float,
        default=argparse.SUPPRESS,
        help="Voxel size used for polnet's low-resolution placement pass, in Ångstrom. "
        "Overrides --config.",
    )
    parser.add_argument(
        "--membrane_potential_scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Membrane potential scale factor, on top of the auto-calibrated "
        "reference. Overrides --config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Random seed for polnet's placement. Overrides --config.",
    )
    parser.add_argument(
        "--filler_occupancy",
        type=float,
        default=argparse.SUPPRESS,
        help="Total occupancy of generic cytosolic filler added on top of "
        "protein_specs. Set to 0 or omit from --config to disable. Overrides --config.",
    )

    # --- Microscope / physics ---
    parser.add_argument(
        "--micrograph_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Output tilt-image size in pixels (square). Defaults to the XY "
        "dimension of target_shape. Overrides --config.",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=argparse.SUPPRESS,
        help="Electron beam accelerating voltage in kV. Overrides --config.",
    )
    parser.add_argument(
        "--dose_per_tilt",
        type=float,
        default=argparse.SUPPRESS,
        help="Dose per tilt angle in e⁻/Å². Overrides --config.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of movie frames per tilt. Overrides --config.",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=argparse.SUPPRESS,
        help="Spherical aberration in mm (1-3 mm typical). Overrides --config.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Amplitude contrast ratio. Overrides --config.",
    )

    # --- Envelopes ---
    parser.add_argument(
        "--convergence_angle",
        type=float,
        default=argparse.SUPPRESS,
        help="Beam convergence semi-angle in mrad, for the Cs (spatial coherence) "
        "envelope. None disables it. Overrides --config.",
    )
    parser.add_argument(
        "--cc",
        type=float,
        default=argparse.SUPPRESS,
        help="Chromatic aberration coefficient in mm, for the Cc (temporal "
        "coherence) envelope. None disables it. Overrides --config.",
    )
    parser.add_argument(
        "--energy_spread",
        type=float,
        default=argparse.SUPPRESS,
        help="FWHM of the beam voltage spread in eV, used by the Cc envelope. "
        "Overrides --config.",
    )
    parser.add_argument(
        "--deltaV_V",
        type=float,
        default=argparse.SUPPRESS,
        help="Relative high-voltage instability, used by the Cc envelope. "
        "Overrides --config.",
    )
    parser.add_argument(
        "--deltaI_I",
        type=float,
        default=argparse.SUPPRESS,
        help="Relative objective-lens current instability, used by the Cc envelope. "
        "Overrides --config.",
    )
    parser.add_argument(
        "--dose_envelope",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Apply the Grant & Grigorieff (2015) cumulative-dose envelope. "
        "Overrides --config.",
    )

    # --- Defocus ---
    parser.add_argument(
        "--defocus",
        type=float,
        default=argparse.SUPPRESS,
        help="Defocus in Ångstrom (positive = underfocus). Overrides --config.",
    )

    # --- Tilt geometry ---
    parser.add_argument(
        "--min_tilt_angle",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum tilt angle in degrees. Overrides --config.",
    )
    parser.add_argument(
        "--max_tilt_angle",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum tilt angle in degrees. Overrides --config.",
    )
    parser.add_argument(
        "--n_tilts",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of tilt angles (evenly spaced from min to max). Overrides --config.",
    )
    parser.add_argument(
        "--tilt_axis",
        type=str,
        default=argparse.SUPPRESS,
        choices=["x", "y"],
        help="Tilt axis. Overrides --config.",
    )

    # --- Models ---
    parser.add_argument(
        "--scattering_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["multislice", "firstborn", "projection", "ctf"],
        help="Scattering model. Overrides --config.",
    )
    parser.add_argument(
        "--aberration_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["holography", "ctf"],
        help="Aberration model. Overrides --config.",
    )
    parser.add_argument(
        "--noise_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["poisson", "none"],
        help="Noise model. Use 'none' for no noise. Overrides --config.",
    )
    parser.add_argument(
        "--coincidence_radius",
        type=float,
        default=argparse.SUPPRESS,
        help="Coincidence radius in pixels for direct-detector modelling. "
        "Overrides --config.",
    )
    parser.add_argument(
        "--ice_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["gd", "random", "none"],
        help="Ice generation algorithm: 'gd' (samples from the pre-generated "
        "IceBank cache, default), 'random' (instant, cheap RandomIcemaker "
        "placement), or 'none'. Overrides --config.",
    )
    parser.add_argument(
        "--ice_cache_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Directory of cached ice configs for ice_model='gd'. Defaults to "
        "the bundled ice-data/ice_cache. Overrides --config.",
    )
    parser.add_argument(
        "--ice_relax_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Local MLBOP relaxation steps used to heal ice tile seams "
        "(ice_model='gd' only). Overrides --config.",
    )
    parser.add_argument(
        "--pad_fft",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Pad volume for FFT to avoid multislice edge-wraparound artifacts "
        "under tilt. Overrides --config.",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["none", "perfect", "k3_300kv", "k3_200kv"],
        help="Detector model. Overrides --config.",
    )

    # --- Post-processing ---
    parser.add_argument(
        "--normalize_tilt_series",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Normalize each tilt image to zero mean and unit std. Overrides --config.",
    )
    parser.add_argument(
        "--save_exitwaves",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Save exit wave magnitude and phase as separate .mrcs files. "
        "Overrides --config.",
    )

    # --- Compute ---
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help="Device to use. Options: cpu | cuda | cuda:0. Overrides --config.",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Directory to save output files. Overrides --config.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=argparse.SUPPRESS,
        help="Base name for output files (no extension). Overrides --config.",
    )

    return parser.parse_args()


def _section(msg: str) -> None:
    _console.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def main() -> None:
    import logging

    import mrcfile
    import torch

    import specter
    from specter.config import TiltSeriesConfig, apply_overrides, load_config
    from specter.io import create_micrograph_starfile
    from specter.imagegenerator import TiltSeriesGenerator
    from specter.specimen.cryoet import CryoETSpecimenGenerator
    from specter.specimen.cytosolic_filler import build_filler_protein_specs

    args = parse_args()
    config = load_config(args.config, TiltSeriesConfig)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    apply_overrides(config, overrides)
    specter.set_verbosity(logging.INFO)

    t_start = time.perf_counter()

    # --- Specimen placement + density (polnet placement, specter density) ---
    _section("Building specimen (polnet placement + PotentialBuilder density)")
    protein_specs = list(config.protein_specs)
    if config.filler_occupancy is not None:
        protein_specs += build_filler_protein_specs(
            total_occupancy=config.filler_occupancy
        )

    gen = CryoETSpecimenGenerator(
        protein_specs=protein_specs,
        membrane_specs=config.membrane_specs,
        target_shape=tuple(config.target_shape),
        target_v_size=config.target_v_size,
        low_res_v_size=config.low_res_v_size,
        pdb_cache_dir=config.pdb_savefolder,
        membrane_potential_scale=config.membrane_potential_scale,
        seed=config.seed,
    )
    volume = gen.generate()
    _console.print(f"  Volume shape: {tuple(volume.shape)}  (Z, Y, X)")

    # --- Build tilt series generator ---
    # TiltSeriesGenerator resolves ice_model/ice_cache_dir/ice_relax_steps into a
    # concrete icemaker itself and blends it into the volume before simulation --
    # no separate ice-generation step needed here, unlike volumes sourced from
    # outside specter (e.g. raw Polnet/TomogramGenerator MRCs) whose intensity
    # scale isn't guaranteed to already match specter's own potential units.
    _section("Building TiltSeriesGenerator")
    dx = config.target_v_size
    micrograph_size = (
        config.micrograph_size
        if config.micrograph_size is not None
        else int(volume.shape[-1])
    )
    angles = torch.linspace(
        config.min_tilt_angle, config.max_tilt_angle, config.n_tilts
    )
    _console.print(
        f"  Tilt angles: {config.n_tilts} steps from {config.min_tilt_angle}° "
        f"to {config.max_tilt_angle}°"
    )

    cs_angstrom = config.cs * 1e7
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None

    ctf_params = {
        "cs": torch.tensor([cs_angstrom]),
        "dfu": torch.tensor([config.defocus]),
    }

    ice_model = None if config.ice_model == "none" else config.ice_model
    noise_model = None if config.noise_model == "none" else config.noise_model
    detector_model = None if config.detector_model == "none" else config.detector_model

    model = TiltSeriesGenerator(
        volume.unsqueeze(0),  # add batch dim: (1, Z, Y, X)
        micrograph_size,
        dx,
        ctf_params,
        config.voltage,
        config.dose_per_tilt,
        angles=angles,
        ice_model=ice_model,
        ice_cache_dir=config.ice_cache_dir,
        ice_relax_steps=config.ice_relax_steps,
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        alpha=config.alpha,
        pad_fft=config.pad_fft,
        tilt_axis=config.tilt_axis,
        coincidence_radius=config.coincidence_radius,
        num_frames=config.num_frames,
        convergence_angle=config.convergence_angle,
        cc=cc_angstrom,
        energy_spread=config.energy_spread,
        deltaV_V=config.deltaV_V,
        deltaI_I=config.deltaI_I,
        dose_envelope=config.dose_envelope,
        detector_model=detector_model,
    ).to(config.device)

    # --- Generate ---
    _section(f"Generating tilt series on {config.device}")
    with torch.no_grad():
        images, exitwaves, cleans = model.generate_tilt_series(torch.tensor([0]))
    # images: (1, n_tilts, H, W)
    images = images[0].cpu()  # (n_tilts, H, W)
    exitwaves = exitwaves[0].cpu()  # (n_tilts, H, W) complex

    # --- Post-processing ---
    if config.normalize_tilt_series:
        _section("Normalizing")
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    _section("Saving")
    os.makedirs(config.output_dir, exist_ok=True)

    mrcs_path = os.path.join(config.output_dir, config.filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mrcs_path}")

    ctf_params_broadcast = {
        "cs": torch.full((config.n_tilts,), cs_angstrom),
        "dfu": torch.full((config.n_tilts,), config.defocus),
    }
    create_micrograph_starfile(
        n=config.n_tilts,
        voltage=config.voltage,
        pixel_size=dx,
        alpha=config.alpha,
        ctf_params=ctf_params_broadcast,
        folderpath=config.output_dir,
        filename=config.filename,
        dose_per_angstrom=config.dose_per_tilt,
        coincidence_radius=config.coincidence_radius,
        tilt_angles=angles,
    )

    if config.save_exitwaves:
        ew_prefix = "exitwave" if ice_model is not None else "clean_exitwave"
        _section(f"Saving {ew_prefix.replace('_', ' ')}")
        mag_path = os.path.join(
            config.output_dir, f"{config.filename}_{ew_prefix}_magnitude.mrcs"
        )
        phase_path = os.path.join(
            config.output_dir, f"{config.filename}_{ew_prefix}_phase.mrcs"
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
