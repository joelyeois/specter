"""
Generate simulated cryo-EM micrographs and save as an .mrcs stack.

The particle volume, ice, and crowding are assembled once at initialisation;
each forward pass applies a different CTF (defocus drawn uniformly at random).

Usage:
    python demo-scripts/generate_micrograph.py --pdb_code 6bdf --output_dir /path/to/output

Parameters are loaded from --config (default: configs/micrograph/micrograph.toml);
any flag below overrides the corresponding field in that file.

Device options:
    --device cpu          Single CPU
    --device cuda         Single GPU (default)
    --device cuda:0       Specific GPU

Example (HPC):
    python demo-scripts/generate_micrograph.py \\
        --config configs/micrograph/micrograph.toml \\
        --pdb_code 6bdf \\
        --n_micrographs 10 \\
        --num_pixels 256 \\
        --pixel_size 1.056 \\
        --micrograph_size 4096 \\
        --energy 300 \\
        --dose_min 53 \\
        --defocus_min 5000 \\
        --defocus_max 15000 \\
        --cs 2.7 \\
        --alpha 0.07 \\
        --scattering_model multislice \\
        --aberration_model holography \\
        --noise_model poisson \\
        --coincidence_radius_min 2.1 \\
        --ice_model gd \\
        --ice_thickness 500 \\
        --chunk_size 8 \\
        --normalize_micrographs True \\
        --device cuda:0 \\
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
    from specter.config import REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Simulate cryo-EM micrographs and save as .mrcs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "micrograph" / "micrograph.toml"),
        help="Path to a TOML config file. Every other flag below overrides a field in it.",
    )

    # --- PDB / potential ---
    parser.add_argument(
        "--pdb_code",
        type=str,
        default=argparse.SUPPRESS,
        help="PDB accession code or path to a local .cif/.pdb file. Overrides --config.",
    )
    parser.add_argument(
        "--assembly",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Fetch biological assembly. Overrides --config.",
    )
    parser.add_argument(
        "--pdb_savefolder",
        type=str,
        default=argparse.SUPPRESS,
        help="Folder to cache downloaded PDB files. Overrides --config.",
    )
    parser.add_argument(
        "--num_pixels",
        type=int,
        default=argparse.SUPPRESS,
        help="Particle box size in pixels (used to build the scattering potential). Overrides --config.",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=argparse.SUPPRESS,
        help="Pixel size in Ångstrom. Overrides --config.",
    )
    parser.add_argument(
        "--micrograph_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Micrograph size in pixels (square). Overrides --config.",
    )

    # --- Microscope / physics ---
    parser.add_argument(
        "--energy",
        type=float,
        default=argparse.SUPPRESS,
        help="Electron beam energy in keV. Overrides --config.",
    )
    parser.add_argument(
        "--dose_min",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum dose in e⁻/Å². Used as fixed dose if --dose_max is not set. Overrides --config.",
    )
    parser.add_argument(
        "--dose_max",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum dose in e⁻/Å². If set, dose is sampled uniformly per micrograph. Overrides --config.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of frames. Defaults to int(dose) if not set. Overrides --config.",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=argparse.SUPPRESS,
        help="Spherical aberration in mm (1–3 mm typical). Overrides --config.",
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
        help="Beam convergence semi-angle in mrad, for the Cs (spatial coherence) envelope. Overrides --config.",
    )
    parser.add_argument(
        "--cc",
        type=float,
        default=argparse.SUPPRESS,
        help="Chromatic aberration coefficient in mm, for the Cc (temporal coherence) envelope. Overrides --config.",
    )
    parser.add_argument(
        "--energy_spread",
        type=float,
        default=argparse.SUPPRESS,
        help="FWHM of the beam energy spread in eV, used by the Cc envelope. Overrides --config.",
    )
    parser.add_argument(
        "--deltaV_V",
        type=float,
        default=argparse.SUPPRESS,
        help="Relative high-voltage instability, used by the Cc envelope. Overrides --config.",
    )
    parser.add_argument(
        "--deltaI_I",
        type=float,
        default=argparse.SUPPRESS,
        help="Relative objective-lens current instability, used by the Cc envelope. Overrides --config.",
    )
    parser.add_argument(
        "--dose_envelope",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Apply the Grant & Grigorieff (2015) cumulative-dose envelope. Overrides --config.",
    )

    # --- Defocus ---
    parser.add_argument(
        "--defocus_min",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum defocus in Ångstrom. Overrides --config.",
    )
    parser.add_argument(
        "--defocus_max",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum defocus in Ångstrom. Overrides --config.",
    )

    # --- Dataset size ---
    parser.add_argument(
        "--n_micrographs",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of micrographs to simulate. Overrides --config.",
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
        "--coincidence_radius_min",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum coincidence radius in pixels. Used as fixed value if --coincidence_radius_max is not set. Overrides --config.",
    )
    parser.add_argument(
        "--coincidence_radius_max",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum coincidence radius in pixels. If set, sampled uniformly per micrograph. Overrides --config.",
    )
    parser.add_argument(
        "--potential_scale_min",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum potential scale factor. Used as fixed value if --potential_scale_max is not set. Overrides --config.",
    )
    parser.add_argument(
        "--potential_scale_max",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum potential scale factor. If set, sampled uniformly per micrograph. Values < 1 approximate thicker ice. Overrides --config.",
    )
    parser.add_argument(
        "--ice_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["gd", "ap", "mcmc", "random", "none"],
        help="Ice model: 'gd' (gradient descent, default), 'ap' (atomic potential), 'mcmc', 'random', or 'none'. Overrides --config.",
    )
    parser.add_argument(
        "--ice_thickness",
        type=float,
        default=argparse.SUPPRESS,
        help="Ice thickness in Ångstrom. Overrides --config.",
    )
    parser.add_argument(
        "--num_unique_icecubes",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of unique ice cubes to pre-build into the IceBank. Overrides --config.",
    )
    parser.add_argument(
        "--ice_build_batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of unique ice cubes to generate at once while building the IceBank. Lower values reduce peak GPU memory. Overrides --config.",
    )
    parser.add_argument(
        "--icecube_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Side length (in voxels) of each cubic ice block generated by the IceBank. Smaller cubes build faster (especially for 'gd') and are tiled to fill the micrograph. Defaults to whichever is smaller: num_pixels, or the voxel count spanning 256 Å (256 / pixel_size). Overrides --config.",
    )
    parser.add_argument(
        "--crowd_min_distance",
        type=float,
        default=argparse.SUPPRESS,
        help="Min distance between crowded particles in Ångstrom. Defaults to pdb.max_diameter. Set to 0 to disable crowding. Overrides --config.",
    )
    parser.add_argument(
        "--crowd_max_distance_z",
        type=float,
        default=argparse.SUPPRESS,
        help="Max z-distance between crowded particles in Ångstrom. Default: None (uses ice thickness). Overrides --config.",
    )
    parser.add_argument(
        "--water_air_interface",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Simulate water-air interface. Overrides --config.",
    )
    parser.add_argument(
        "--pad_fft",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Pad volume for FFT to avoid edge artifacts. Overrides --config.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Slice chunk size for specimen generation. Set if GPU memory is limited (e.g. 8). Overrides --config.",
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
        "--normalize_micrographs",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Normalize micrographs to zero mean and unit std. Overrides --config.",
    )
    parser.add_argument(
        "--save_exitwaves",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Save icy exit wave magnitude and phase as separate .mrcs files. Overrides --config.",
    )
    parser.add_argument(
        "--save_clean_exitwaves",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Save clean (no-ice) exit wave magnitude and phase. Runs scattering twice per micrograph. Overrides --config.",
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

    import specter
    from specter.config import apply_overrides, load_config, MicrographConfig
    from specter.cryosparc import create_micrograph_starfile
    from specter.ice import IceBank
    from specter.imagegenerator import MicrographGenerator
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.progress import track

    args = parse_args()
    config = load_config(args.config, MicrographConfig)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    apply_overrides(config, overrides)
    specter.set_verbosity(logging.INFO)

    t_start = time.perf_counter()

    # --- Building 3D scattering potential ---
    _section("Building 3D scattering potential")
    pdb = PDB(
        config.pdb_code, assembly=config.assembly, savefolder=config.pdb_savefolder
    )

    cs_angstrom = config.cs * 1e7

    pb = PotentialBuilder(config.num_pixels, config.pixel_size, pdb.atomic_numbers).to(
        "cpu"
    )
    with torch.no_grad():
        V = pb(pdb.coordinates).clone()

    n = config.n_micrographs

    noise_model = None if config.noise_model == "none" else config.noise_model
    ice_model = None if config.ice_model == "none" else config.ice_model
    detector_model = None if config.detector_model == "none" else config.detector_model
    crowd_min_distance = (
        None
        if config.crowd_min_distance == 0
        else config.crowd_min_distance
        if config.crowd_min_distance is not None
        else pdb.max_diameter
    )

    # Sample all per-micrograph parameters upfront
    defocus_A = (
        torch.rand(n) * (config.defocus_max - config.defocus_min) + config.defocus_min
    )

    dose_max = config.dose_max if config.dose_max is not None else config.dose_min
    dose = torch.rand(n) * (dose_max - config.dose_min) + config.dose_min

    cr_max = (
        config.coincidence_radius_max
        if config.coincidence_radius_max is not None
        else config.coincidence_radius_min
    )
    coincidence_radius = (
        torch.rand(n) * (cr_max - config.coincidence_radius_min)
        + config.coincidence_radius_min
    )

    ps_max = (
        config.potential_scale_max
        if config.potential_scale_max is not None
        else config.potential_scale_min
    )
    potential_scale = (
        torch.rand(n) * (ps_max - config.potential_scale_min)
        + config.potential_scale_min
    )

    num_frames = (
        config.num_frames if config.num_frames is not None else int(dose.mean().item())
    )
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_A,
    }

    # --- Pre-building ice bank ---
    # Built once, upfront, on the target device — regenerate_specimen() draws
    # fresh (randomly tiled) ice from the same cached cubes on every call.
    icecube_size = (
        config.icecube_size
        if config.icecube_size is not None
        else min(config.num_pixels, int(256 / config.pixel_size))
    )
    icemaker = None
    if ice_model is not None:
        _section("Pre-building ice bank")
        icemaker = IceBank(
            n=icecube_size,
            dx=config.pixel_size,
            method=ice_model,
            num_unique=config.num_unique_icecubes,
            build_batch_size=config.ice_build_batch_size,
        ).to(config.device)
        icemaker.build()

    # Build once — __init__ generates the first specimen
    _section("Building specimen and image generator")
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None
    model = MicrographGenerator(
        V,
        config.micrograph_size,
        config.pixel_size,
        ctf_params,
        config.energy,
        dose,
        icemaker=icemaker,
        ice_thickness=config.ice_thickness,
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        klim=None,
        alpha=config.alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=config.crowd_max_distance_z,
        water_air_interface=config.water_air_interface,
        pad_fft=config.pad_fft,
        chunk_size=config.chunk_size,
        move_to_cpu=True,
        detector_model=detector_model,
        verbose=False,
        progressbars=False,
        coincidence_radius=coincidence_radius,
        num_frames=num_frames,
        potential_scale=potential_scale,
        save_clean_exitwaves=config.save_clean_exitwaves,
        convergence_angle=config.convergence_angle,
        cc=cc_angstrom,
        energy_spread=config.energy_spread,
        deltaV_V=config.deltaV_V,
        deltaI_I=config.deltaI_I,
        dose_envelope=config.dose_envelope,
    ).to(config.device)

    # Loop: regenerate fresh specimen for each micrograph, use i-th CTF
    _section(f"Generating {n} micrograph(s) on {config.device}")
    images = []
    exitwaves = [] if config.save_exitwaves else None
    clean_exitwaves = [] if config.save_clean_exitwaves else None

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
    if config.normalize_micrographs:
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / std.clamp(min=1e-8)

    # --- Saving ---
    _section("Saving .mrcs + .star")
    import mrcfile

    os.makedirs(config.output_dir, exist_ok=True)
    mrcs_path = os.path.join(config.output_dir, config.filename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy().astype("float32"))
    _console.print(f"  [green]✓[/green] {mrcs_path}")

    create_micrograph_starfile(
        n,
        energy=config.energy,
        pixel_size=config.pixel_size,
        alpha=config.alpha,
        ctf_params=ctf_params,
        folderpath=config.output_dir,
        filename=config.filename,
        dose_per_angstrom=dose,
        coincidence_radius=coincidence_radius,
        potential_scale=potential_scale,
    )

    def _save_exitwave_pair(ew, suffix: str) -> None:
        if config.pad_fft:
            ew = _crop_center(ew, config.micrograph_size)
        mag_path = os.path.join(
            config.output_dir, f"{config.filename}_{suffix}_magnitude.mrcs"
        )
        phase_path = os.path.join(
            config.output_dir, f"{config.filename}_{suffix}_phase.mrcs"
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
