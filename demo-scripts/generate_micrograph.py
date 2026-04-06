"""
Generate simulated cryo-EM micrographs and save as an .mrcs stack.

The particle volume, ice, and crowding are assembled once at initialisation;
each forward pass applies a different CTF (defocus drawn uniformly at random).

Usage:
    python demo-scripts/generate_micrograph.py --pdb_code 6bdf --output_dir /path/to/output

Device options:
    --device cpu          Single CPU
    --device cuda         Single GPU (default)
    --device cuda:0       Specific GPU

Example (HPC):
    python demo-scripts/generate_micrograph.py \\
        --pdb_code 6bdf \\
        --n_micrographs 10 \\
        --num_pixels 256 \\
        --pixel_size 1.056 \\
        --micrograph_size 4096 \\
        --energy 300 \\
        --dose 53 \\
        --defocus_min 5000 \\
        --defocus_max 15000 \\
        --cs 2.7 \\
        --alpha 0.07 \\
        --scattering_model multislice \\
        --aberration_model holography \\
        --noise_model poisson \\
        --coincidence_radius 2.1 \\
        --ice_model iterative \\
        --ice_thickness 500 \\
        --chunk_size 8 \\
        --normalize_micrographs True \\
        --device cuda:0 \\
        --batchsize 1 \\
        --output_dir ./output/ \\
        --filename micrographs
"""

import argparse
import os
import time

from rich.console import Console
from rich.rule import Rule

_console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate cryo-EM micrographs and save as .mrcs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- PDB / potential ---
    parser.add_argument(
        "--pdb_code",
        type=str,
        required=True,
        help="PDB accession code or path to a local .cif/.pdb file. (required)",
    )
    parser.add_argument(
        "--assembly",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="True|False",
        help="Fetch biological assembly.",
    )
    parser.add_argument(
        "--pdb_savefolder",
        type=str,
        default="../pdb-data/",
        help="Folder to cache downloaded PDB files.",
    )
    parser.add_argument(
        "--num_pixels",
        type=int,
        default=256,
        help="Particle box size in pixels (used to build the scattering potential).",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=1.0,
        help="Pixel size in Ångstrom.",
    )
    parser.add_argument(
        "--micrograph_size",
        type=int,
        default=4096,
        help="Micrograph size in pixels (square).",
    )

    # --- Microscope / physics ---
    parser.add_argument(
        "--energy",
        type=float,
        default=300.0,
        help="Electron beam energy in keV.",
    )
    parser.add_argument(
        "--dose",
        type=float,
        default=20.0,
        help="Electron dose in e⁻/Å².",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames. Defaults to int(dose) if not set.",
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
        "--defocus_min",
        type=float,
        default=5000.0,
        help="Minimum defocus in Ångstrom.",
    )
    parser.add_argument(
        "--defocus_max",
        type=float,
        default=15000.0,
        help="Maximum defocus in Ångstrom.",
    )

    # --- Dataset size ---
    parser.add_argument(
        "--n_micrographs",
        type=int,
        default=1,
        help="Number of micrographs to simulate.",
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
        "--aberration_model",
        type=str,
        default="holography",
        choices=["holography", "ctf"],
        help="Aberration model.",
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
        default=1.8,
        help="Coincidence loss radius in Ångstrom. Set to 0 for default Poisson.",
    )
    parser.add_argument(
        "--ice_model",
        type=str,
        default="iterative",
        choices=["iterative", "randomchoice", "none"],
        help="Ice model.",
    )
    parser.add_argument(
        "--ice_thickness",
        type=float,
        default=500.0,
        help="Ice thickness in Ångstrom.",
    )
    parser.add_argument(
        "--crowd_min_distance",
        type=float,
        default=None,
        help="Min distance between crowded particles in Ångstrom. Defaults to pdb.max_diameter. Set to 0 to disable crowding.",
    )
    parser.add_argument(
        "--crowd_max_distance_z",
        type=float,
        default=None,
        help="Max z-distance between crowded particles in Ångstrom. Default: None (uses ice thickness).",
    )
    parser.add_argument(
        "--water_air_interface",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="True|False",
        help="Simulate water-air interface.",
    )
    parser.add_argument(
        "--pad_fft",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Pad volume for FFT to avoid edge artifacts.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Slice chunk size for specimen generation. Set if GPU memory is limited (e.g. 8).",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="none",
        choices=["none", "perfect", "k3_300kv", "k3_200kv"],
        help="Detector model.",
    )

    # --- Post-processing ---
    parser.add_argument(
        "--normalize_micrographs",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Normalize micrographs to zero mean and unit std.",
    )
    parser.add_argument(
        "--save_exitwaves",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Save icy exit wave magnitude and phase as separate .mrcs files.",
    )
    parser.add_argument(
        "--save_clean_exitwaves",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Save clean (no-ice) exit wave magnitude and phase. Runs scattering twice per micrograph.",
    )

    # --- Compute ---
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
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
        default="micrographs",
        help="Base name for output files (no extension).",
    )

    return parser.parse_args()


def _section(msg: str) -> None:
    """Print a full-width titled rule as a section separator."""
    _console.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def _crop_center(t, nxy: int):
    """Center-crop a (..., H, W) tensor to (..., nxy, nxy). Matches Detector.forward crop."""
    H, W = t.shape[-2], t.shape[-1]
    if H == nxy and W == nxy:
        return t
    cy, cx = H // 2, W // 2
    half = nxy // 2
    return t[..., cy - half : cy + half + (nxy % 2), cx - half : cx + half + (nxy % 2)]


def main() -> None:
    import logging

    import torch

    import cryosim
    from cryosim.imagegenerator import MicrographGenerator
    from cryosim.pdbtools import PDB
    from cryosim.potential import PotentialBuilder
    from cryosim.progress import track

    args = parse_args()
    cryosim.set_verbosity(logging.INFO)

    t_start = time.perf_counter()

    # --- Building 3D scattering potential ---
    _section("Building 3D scattering potential")
    pdb = PDB(args.pdb_code, assembly=args.assembly, savefolder=args.pdb_savefolder)

    cs_angstrom = args.cs * 1e7

    pb = PotentialBuilder(args.num_pixels, args.pixel_size, pdb.atomic_numbers).to(
        "cpu"
    )
    with torch.no_grad():
        V = pb(pdb.coordinates).clone()

    n = args.n_micrographs

    noise_model = None if args.noise_model == "none" else args.noise_model
    ice_model = None if args.ice_model == "none" else args.ice_model
    detector_model = None if args.detector_model == "none" else args.detector_model
    crowd_min_distance = (
        None
        if args.crowd_min_distance == 0
        else args.crowd_min_distance
        if args.crowd_min_distance is not None
        else pdb.max_diameter
    )
    num_frames = args.num_frames if args.num_frames is not None else int(args.dose)

    # Sample all defocus values upfront — one per micrograph
    defocus_A = torch.rand(n) * (args.defocus_max - args.defocus_min) + args.defocus_min
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_A,
    }

    # Build once — __init__ generates the first specimen
    _section("Building specimen and image generator")
    model = MicrographGenerator(
        V,
        args.micrograph_size,
        args.pixel_size,
        ctf_params,
        args.energy,
        args.dose,
        ice_model=ice_model,
        ice_thickness=args.ice_thickness,
        scattering_model=args.scattering_model,
        aberration_model=args.aberration_model,
        noise_model=noise_model,
        klim=None,
        alpha=args.alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=args.crowd_max_distance_z,
        water_air_interface=args.water_air_interface,
        pad_fft=args.pad_fft,
        chunk_size=args.chunk_size,
        move_to_cpu=True,
        detector_model=detector_model,
        verbose=False,
        progressbars=False,
        coincidence_radius=args.coincidence_radius,
        num_frames=num_frames,
        save_clean_exitwaves=args.save_clean_exitwaves,
    ).to(args.device)

    # Loop: regenerate fresh specimen for each micrograph, use i-th CTF
    _section(f"Generating {n} micrograph(s) on {args.device}")
    images = []
    exitwaves = [] if args.save_exitwaves else None
    clean_exitwaves = [] if args.save_clean_exitwaves else None

    for i in track(range(n), description="Generating micrographs"):
        if i > 0:
            model.regenerate_specimen()
        with torch.no_grad():
            img = model(torch.tensor([i]))
        images.append(img.detach().cpu())
        if exitwaves is not None:
            exitwaves.append(model.exitwaves.detach().cpu())
        if clean_exitwaves is not None:
            clean_exitwaves.append(model.clean_exitwaves.detach().cpu())

    images = torch.concat(images, dim=0)
    if exitwaves is not None:
        exitwaves = torch.concat(exitwaves, dim=0)
    if clean_exitwaves is not None:
        clean_exitwaves = torch.concat(clean_exitwaves, dim=0)

    # --- Post-processing ---
    _section("Post-processing")
    if args.normalize_micrographs:
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    _section("Saving .mrcs")
    import mrcfile

    os.makedirs(args.output_dir, exist_ok=True)
    mrcs_path = os.path.join(args.output_dir, args.filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mrcs_path}")

    def _save_exitwave_pair(ew, suffix: str) -> None:
        if args.pad_fft:
            ew = _crop_center(ew, args.micrograph_size)
        mag_path = os.path.join(
            args.output_dir, f"{args.filename}_{suffix}_magnitude.mrcs"
        )
        phase_path = os.path.join(
            args.output_dir, f"{args.filename}_{suffix}_phase.mrcs"
        )
        with mrcfile.new(mag_path, overwrite=True) as mrc:
            mrc.set_data(ew.abs().numpy().astype("float32"))
        _console.print(f"  [green]✓[/green] {mag_path}")
        with mrcfile.new(phase_path, overwrite=True) as mrc:
            mrc.set_data(ew.angle().numpy().astype("float32"))
        _console.print(f"  [green]✓[/green] {phase_path}")

    if exitwaves is not None:
        _section("Saving exit waves")
        _save_exitwave_pair(exitwaves, "exitwave")

    if clean_exitwaves is not None:
        _section("Saving clean exit waves")
        _save_exitwave_pair(clean_exitwaves, "clean_exitwave")

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
