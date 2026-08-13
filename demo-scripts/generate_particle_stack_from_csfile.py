"""
Generate a simulated cryo-EM particle stack from a CryoSPARC .cs file.

Poses, CTF parameters, pixel size, voltage, and amplitude contrast are all
read directly from the .cs file — no random sampling needed.

Usage:
    python generate_particle_stack_from_csfile.py --cs_path /path/to/file.cs --pdb_code 6bdf --dose 53 --output_dir /path/to/output

Device options:
    --device cpu          Single CPU
    --device cuda         Single GPU (default device)
    --device cuda:0       Specific single GPU
    --device 0,1,2,3      Multi-GPU via Lightning DDP (comma-separated GPU IDs)

Example (HPC, multi-GPU):
    python generate_particle_stack_from_csfile.py \
        --cs_path /scratch/loh/joel/J35/J35_passthrough_particles.cs \
        --pdb_code 6bdf \
        --n_particles 1000 \
        --num_pixels 256 \
        --dose 53 \
        --scattering_model multislice \
        --aberration_model holography \
        --noise_model poisson \
        --coincidence_radius 0.8378 \
        --ice_model gd \
        --normalize_particles True \
        --device 0,1,2,3 \
        --batchsize 5 \
        --output_dir /scratch/loh/joel/simulated_data/ \
        --filename 6bdf_from_cs
"""

import argparse
import glob
import os
import time

from rich.console import Console
from rich.rule import Rule
from specter.config import default_output_dir, default_pdb_cache_dir

_console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a cryo-EM particle stack from a .cs file and save as .mrcs + .star.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Inputs ---
    parser.add_argument(
        "--cs_path",
        type=str,
        required=True,
        help="Path to CryoSPARC .cs file containing poses and CTF parameters. (required)",
    )
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
        default=default_pdb_cache_dir(),
        help="Folder to cache downloaded PDB files.",
    )
    parser.add_argument(
        "--num_pixels",
        type=int,
        default=256,
        help="Number of pixels per axis for the 3-D potential box.",
    )

    # --- Dose (not in .cs file) ---
    parser.add_argument(
        "--dose",
        type=float,
        required=True,
        help="Electron dose in e⁻/Å². Check the EMDB Experiment tab for this value. (required)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames. Defaults to int(dose) if not set.",
    )

    # --- Dataset size ---
    parser.add_argument(
        "--n_particles",
        type=int,
        default=None,
        help="Number of particles to simulate. Defaults to all particles in the .cs file.",
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
        default=2.1,
        help="Coincidence loss radius in pixels. Set to 0 for default Poisson.",
    )
    parser.add_argument(
        "--ice_model",
        type=str,
        default="gd",
        choices=["gd", "random", "none"],
        help="Ice model: 'gd' (samples from the pre-generated IceBank cache, default), "
        "'random' (instant, cheap RandomIcemaker placement), or 'none'.",
    )
    parser.add_argument(
        "--ice_thickness",
        type=float,
        default=0.0,
        help="Ice thickness in Ångstrom. 0 = minimum (particle box size).",
    )
    parser.add_argument(
        "--ice_cache_dir",
        type=str,
        default=None,
        help="Directory of cached ice configs for ice_model='gd'. Defaults to the "
        "bundled ice-data/ice_cache.",
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
        help="Max z-distance between crowded particles in Ångstrom. Default: None.",
    )
    parser.add_argument(
        "--pad_fft",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="True|False",
        help="Pad volume for FFT to avoid edge artifacts.",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="none",
        choices=["none", "perfect", "k3_300kv", "k3_200kv"],
        help="Detector model.",
    )

    # --- Envelopes ---
    parser.add_argument(
        "--convergence_angle",
        type=float,
        default=None,
        help="Beam convergence semi-angle in mrad, for the Cs (spatial coherence) envelope. None disables it.",
    )
    parser.add_argument(
        "--cc",
        type=float,
        default=None,
        help="Chromatic aberration coefficient in mm, for the Cc (temporal coherence) envelope. None disables it.",
    )
    parser.add_argument(
        "--energy_spread",
        type=float,
        default=0.7,
        help="FWHM of the beam voltage spread in eV, used by the Cc envelope.",
    )
    parser.add_argument(
        "--deltaV_V",
        type=float,
        default=0.06e-6,
        help="Relative high-voltage instability, used by the Cc envelope.",
    )
    parser.add_argument(
        "--deltaI_I",
        type=float,
        default=0.01e-6,
        help="Relative objective-lens current instability, used by the Cc envelope.",
    )
    parser.add_argument(
        "--dose_envelope",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Apply the Grant & Grigorieff (2015) cumulative-dose envelope.",
    )

    # --- Post-processing ---
    parser.add_argument(
        "--normalize_particles",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="True|False",
        help="Normalize particles to zero mean and unit std.",
    )
    parser.add_argument(
        "--save_exitwaves",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Save exit wave magnitude and phase as separate .mrcs files.",
    )
    parser.add_argument(
        "--save_clean_exitwaves",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Save clean (particle-only, no ice) exit wave magnitude and phase. Runs scattering twice per batch.",
    )

    # --- Compute ---
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device to use. Options: cpu | cuda | cuda:0 | 0,1,2,3. "
            "Comma-separated integers trigger multi-GPU Lightning DDP."
        ),
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=5,
        help="Number of particles per forward pass.",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir("particles"),
        help="Directory to save .mrcs and .star files.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="particles",
        help="Base name for output files (no extension).",
    )

    return parser.parse_args()


def _crop_center(t, nxy: int):
    """Center-crop a (..., H, W) tensor to (..., nxy, nxy). Matches Detector.forward crop."""
    H, W = t.shape[-2], t.shape[-1]
    if H == nxy and W == nxy:
        return t
    cy, cx = H // 2, W // 2
    half = nxy // 2
    return t[..., cy - half : cy + half + (nxy % 2), cx - half : cx + half + (nxy % 2)]


def _section(msg: str) -> None:
    """Print a full-width titled rule as a section separator."""
    _console.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def _parse_device(device_str: str) -> tuple[str, str | list[int]]:
    """
    Parse --device string into a mode and device target.

    Returns
    -------
    mode : "single" or "multi"
    target : device string (single) or list of GPU IDs (multi)

    Examples
    --------
    "cpu"     → ("single", "cpu")
    "cuda"    → ("single", "cuda")
    "cuda:0"  → ("single", "cuda:0")
    "0"       → ("single", "cuda:0")
    "0,1,2"   → ("multi",  [0, 1, 2])
    """
    parts = device_str.split(",")
    if len(parts) > 1:
        try:
            return "multi", [int(p.strip()) for p in parts]
        except ValueError:
            pass
    if parts[0].strip().isdigit():
        return "single", f"cuda:{parts[0].strip()}"
    return "single", device_str


def _generate_single(
    model,
    n: int,
    batchsize: int,
    track,
    collect_exitwaves: bool = False,
    collect_clean_exitwaves: bool = False,
):
    """Run image generation on a single device."""
    import torch

    idx = torch.arange(n)
    images = []
    exitwaves = [] if collect_exitwaves else None
    clean_exitwaves = [] if collect_clean_exitwaves else None
    with torch.no_grad():
        for i in track(range(0, n, batchsize), description="Generating images"):
            batch = model(idx[i : i + batchsize])
            images.append(batch.detach().cpu())
            if collect_exitwaves:
                exitwaves.append(model.exitwaves.detach().cpu())
            if collect_clean_exitwaves:
                clean_exitwaves.append(model.clean_exitwaves.detach().cpu())
    images = torch.concat(images, dim=0)
    if collect_exitwaves:
        exitwaves = torch.concat(exitwaves, dim=0)
    if collect_clean_exitwaves:
        clean_exitwaves = torch.concat(clean_exitwaves, dim=0)
    return images, exitwaves, clean_exitwaves


def _generate_multi(
    model,
    n: int,
    batchsize: int,
    gpu_ids: list,
    output_dir: str,
    collect_exitwaves: bool = False,
    collect_clean_exitwaves: bool = False,
):
    """Run image generation across multiple GPUs using Lightning DDP.

    Returns (images, exitwaves, clean_exitwaves) on rank 0, (None, None, None) on worker ranks.
    exitwaves / clean_exitwaves are None if their collect flag is False.
    """
    import torch
    import lightning as L
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import BasePredictionWriter
    from typing import Any, Sequence

    class _Writer(BasePredictionWriter):
        def __init__(
            self, out_dir: str, save_exitwaves: bool, save_clean_exitwaves: bool
        ) -> None:
            super().__init__("epoch")
            self.out_dir = out_dir
            self.save_exitwaves = save_exitwaves
            self.save_clean_exitwaves = save_clean_exitwaves
            self._exitwaves: list = []
            self._clean_exitwaves: list = []

        def on_predict_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            if self.save_exitwaves and hasattr(pl_module, "exitwaves"):
                self._exitwaves.append(pl_module.exitwaves.cpu())
            if self.save_clean_exitwaves and hasattr(pl_module, "clean_exitwaves"):
                self._clean_exitwaves.append(pl_module.clean_exitwaves.cpu())

        def write_on_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            predictions: Sequence[Any],
            batch_indices: Sequence[Any],
        ) -> None:
            rank = trainer.global_rank
            images = torch.concat(predictions, dim=0)
            torch.save(images, os.path.join(self.out_dir, f"predictions_{rank}.pt"))
            idx = torch.squeeze(torch.tensor(batch_indices)).reshape(-1)
            torch.save(idx, os.path.join(self.out_dir, f"batch_indices_{rank}.pt"))
            if self.save_exitwaves and self._exitwaves:
                torch.save(
                    torch.cat(self._exitwaves, dim=0),
                    os.path.join(self.out_dir, f"exitwaves_{rank}.pt"),
                )
            if self.save_clean_exitwaves and self._clean_exitwaves:
                torch.save(
                    torch.cat(self._clean_exitwaves, dim=0),
                    os.path.join(self.out_dir, f"clean_exitwaves_{rank}.pt"),
                )

    os.makedirs(output_dir, exist_ok=True)
    dataloader = DataLoader(
        torch.arange(n),
        batch_size=batchsize,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=gpu_ids,
        strategy="ddp",
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
        callbacks=[_Writer(output_dir, collect_exitwaves, collect_clean_exitwaves)],
    )

    print(f"Running multi-GPU generation on GPUs: {gpu_ids}")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)

    if trainer.global_rank != 0:
        return None, None, None

    # Reassemble images in original order
    prediction_files = sorted(glob.glob(os.path.join(output_dir, "predictions_*.pt")))
    index_files = sorted(glob.glob(os.path.join(output_dir, "batch_indices_*.pt")))

    all_preds = torch.cat([torch.load(f) for f in prediction_files], dim=0)
    all_indices = torch.cat([torch.load(f) for f in index_files], dim=0)
    sort_order = torch.argsort(all_indices)
    images = all_preds[sort_order]

    for f in prediction_files + index_files:
        os.remove(f)

    # Reassemble exit waves if collected
    exitwaves = None
    if collect_exitwaves:
        exitwave_files = sorted(glob.glob(os.path.join(output_dir, "exitwaves_*.pt")))
        if exitwave_files:
            all_exitwaves = torch.cat([torch.load(f) for f in exitwave_files], dim=0)
            exitwaves = all_exitwaves[sort_order]
            for f in exitwave_files:
                os.remove(f)

    # Reassemble clean exit waves if collected
    clean_exitwaves = None
    if collect_clean_exitwaves:
        clean_files = sorted(
            glob.glob(os.path.join(output_dir, "clean_exitwaves_*.pt"))
        )
        if clean_files:
            all_clean = torch.cat([torch.load(f) for f in clean_files], dim=0)
            clean_exitwaves = all_clean[sort_order]
            for f in clean_files:
                os.remove(f)

    return images, exitwaves, clean_exitwaves


def main() -> None:
    import logging

    import torch

    import specter
    from specter.arrays import compute_nz
    from specter.io import create_particle_starfile, extract_parameters_from_csfile
    from specter.ice import resolve_icemaker
    from specter.image import normalize_particles
    from specter.imagegenerator import ImageGenerator
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.progress import track

    args = parse_args()
    specter.set_verbosity(logging.INFO)

    mode, device_target = _parse_device(args.device)
    t_start = time.perf_counter()

    is_main = "LOCAL_RANK" not in os.environ

    # ------------------------------------------------------------------ #
    # 1. Load parameters from .cs file                                    #
    # ------------------------------------------------------------------ #
    if is_main:
        _section("Loading parameters from .cs file")

    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        _halfset_labels,
    ) = extract_parameters_from_csfile(args.cs_path)

    n_total = len(rotations)
    n = args.n_particles if args.n_particles is not None else n_total

    if is_main:
        from rich.table import Table

        _tbl = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        _tbl.add_column("key", style="bold dim")
        _tbl.add_column("val")
        _tbl.add_row("Particles in file", str(n_total))
        _tbl.add_row("Simulating", f"[bold]{n}[/bold]")
        _tbl.add_row("Voltage", f"{voltage_kv:.1f} kV")
        _tbl.add_row("Pixel size", f"{pixel_size.item():.4f} Å")
        _tbl.add_row("Alpha", f"{alpha:.3f}")
        _console.print(_tbl)

    # Subset to first n particles
    rotations = rotations[:n]
    translations_A = translations_A[:n]
    ctf_params = {k: v[:n] for k, v in ctf_params.items()}

    # ------------------------------------------------------------------ #
    # 2. Build 3-D scattering potential                                   #
    # ------------------------------------------------------------------ #
    if is_main:
        _section("Building 3D scattering potential")

    pdb = PDB(args.pdb_code, assembly=args.assembly, savefolder=args.pdb_savefolder)

    pb = PotentialBuilder(args.num_pixels, pixel_size.item(), pdb.atomic_numbers).to(
        "cpu"
    )
    with torch.no_grad():
        V = pb(pdb.coordinates).clone()

    # ------------------------------------------------------------------ #
    # 3. Build ImageGenerator                                             #
    # ------------------------------------------------------------------ #
    if is_main:
        _section("Building image generator")

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

    cc_angstrom = args.cc * 1e7 if args.cc is not None else None

    # --- Ice ---
    # resolve_icemaker derives (n, nz) for a fresh RandomIcemaker itself, so
    # it always matches the particle volume V it gets blended into --
    # IceBank(cache_dir=...) just loads small pre-generated coordinate files
    # from disk, cheap enough that every DDP rank can construct it
    # independently (no rank-0-builds-then-broadcasts dance needed).
    ice_nz = compute_nz(args.num_pixels, args.ice_thickness, pixel_size.item())
    icemaker = resolve_icemaker(
        ice_model,
        pixel_size.item(),
        args.num_pixels,
        ice_nz,
        ice_cache_dir=args.ice_cache_dir,
    )

    if icemaker is not None:
        icemaker_device = (
            f"cuda:{device_target[0]}" if mode == "multi" else device_target
        )
        if icemaker_device != "cpu":
            icemaker = icemaker.to(icemaker_device)

    model = ImageGenerator(
        V,
        pixel_size.item(),
        rotations,
        translations_A,
        ctf_params,
        voltage_kv,
        args.dose,
        icemaker=icemaker,
        ice_thickness=args.ice_thickness,
        scattering_model=args.scattering_model,
        aberration_model=args.aberration_model,
        noise_model=noise_model,
        klim=None,
        ews_curvature_sign="positive",
        alpha=alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=args.crowd_max_distance_z,
        pad_fft=args.pad_fft,
        detector_model=detector_model,
        verbose=False,
        coincidence_radius=args.coincidence_radius,
        num_frames=num_frames,
        convergence_angle=args.convergence_angle,
        cc=cc_angstrom,
        energy_spread=args.energy_spread,
        deltaV_V=args.deltaV_V,
        deltaI_I=args.deltaI_I,
        dose_envelope=args.dose_envelope,
    )

    if args.save_clean_exitwaves:
        model.save_clean_exitwaves = True

    # ------------------------------------------------------------------ #
    # 4. Generate images                                                  #
    # ------------------------------------------------------------------ #
    if mode == "multi":
        if is_main:
            _section(f"Initializing multi-GPU on devices {device_target}")
        images, exitwaves, clean_exitwaves = _generate_multi(
            model,
            n,
            args.batchsize,
            device_target,
            args.output_dir,
            collect_exitwaves=args.save_exitwaves,
            collect_clean_exitwaves=args.save_clean_exitwaves,
        )
        if images is None:
            return
    else:
        if is_main:
            _section(f"Generating images on {device_target}")
        model = model.to(device_target)
        images, exitwaves, clean_exitwaves = _generate_single(
            model,
            n,
            args.batchsize,
            track,
            collect_exitwaves=args.save_exitwaves,
            collect_clean_exitwaves=args.save_clean_exitwaves,
        )

    # ------------------------------------------------------------------ #
    # 5. Optionally normalise and flip sign                               #
    # ------------------------------------------------------------------ #
    if is_main:
        _section("Post-processing")

    if args.normalize_particles:
        particles, _means, _stds = normalize_particles(images)
        particles = -particles
    else:
        particles = images

    # ------------------------------------------------------------------ #
    # 6. Save .mrcs + .star                                               #
    # ------------------------------------------------------------------ #
    if is_main:
        _section("Saving .mrcs + .star")

    create_particle_starfile(
        particles,
        rotations=rotations,
        translations=translations_A,
        ctf_params=ctf_params,
        dx=pixel_size.item(),
        voltage=voltage_kv,
        alpha=alpha,
        filename=args.filename,
        folderpath=args.output_dir,
        dose_per_angstrom=args.dose,
        coincidence_radius=args.coincidence_radius,
    )

    if is_main:
        import mrcfile

        def _save_exitwave_pair(ew, suffix: str) -> None:
            if args.pad_fft:
                ew = _crop_center(ew, args.num_pixels)
            os.makedirs(args.output_dir, exist_ok=True)
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
