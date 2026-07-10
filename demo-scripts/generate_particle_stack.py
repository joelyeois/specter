"""
Generate a simulated cryo-EM particle stack and save as .mrcs + .star files.

Usage:
    python generate_particle_stack.py --pdb_code 6bdf --n_particles 20 --output_dir /path/to/output

Device options:
    --device cpu          Single CPU
    --device cuda         Single GPU (default device)
    --device cuda:0       Specific single GPU
    --device 0,1,2,3      Multi-GPU via Lightning DDP (comma-separated GPU IDs)

Example (HPC, multi-GPU):
    python generate_particle_stack.py \
        --pdb_code 6bdf \
        --n_particles 3000 \
        --num_pixels 256 \
        --pixel_size 1.0 \
        --energy 300 \
        --dose_min 20.0 \
        --defocus_min 5000 \
        --defocus_max 15000 \
        --cs 2.0 \
        --alpha 0.1 \
        --shift 2.0 \
        --scattering_model multislice \
        --aberration_model holography \
        --noise_model poisson \
        --coincidence_radius_min 1.8 \
        --ice_model gd \
        --ice_thickness 0 \
        --normalize_particles True \
        --device 0,1,2,3 \
        --batchsize 5 \
        --output_dir /scratch/loh/joel/simulated_data/ \
        --filename my_particle_stack
"""

import argparse
import glob
import os
import time

from rich.console import Console
from rich.rule import Rule

_console = Console()


def parse_args() -> argparse.Namespace:
    from specter.config import REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Simulate a cryo-EM particle stack and save as .mrcs + .star.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "particle_stack" / "particle.toml"),
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
        help="Number of pixels per axis for the 3-D potential box. Overrides --config.",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=argparse.SUPPRESS,
        help="Pixel size in Ångstrom. Overrides --config.",
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
        help="Maximum dose in e⁻/Å². If set, dose is sampled uniformly per particle. Overrides --config.",
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
        help="Beam convergence semi-angle in mrad, for the Cs envelope. Overrides --config.",
    )
    parser.add_argument(
        "--cc",
        type=float,
        default=argparse.SUPPRESS,
        help="Chromatic aberration coefficient in mm, for the Cc envelope. Overrides --config.",
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
    parser.add_argument(
        "--bfactor",
        type=float,
        default=argparse.SUPPRESS,
        help="Isotropic B-factor envelope in Å². Overrides --config.",
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

    # --- Translations / shifts ---
    parser.add_argument(
        "--shift",
        type=float,
        default=argparse.SUPPRESS,
        help="Max in-plane shift in Ångstrom (uniform ±shift). Overrides --config.",
    )

    # --- Dataset size ---
    parser.add_argument(
        "--n_particles",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of particles to simulate. Overrides --config.",
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
        help="Minimum coincidence radius in pixels. Overrides --config.",
    )
    parser.add_argument(
        "--coincidence_radius_max",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum coincidence radius in pixels. Overrides --config.",
    )
    parser.add_argument(
        "--ice_model",
        type=str,
        default=argparse.SUPPRESS,
        choices=["gd", "ap", "mcmc", "random", "none"],
        help="Ice model. Overrides --config.",
    )
    parser.add_argument(
        "--ice_thickness",
        type=float,
        default=argparse.SUPPRESS,
        help="Ice thickness in Ångstrom. 0 = minimum (particle box size). Overrides --config.",
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
        help="Number of unique ice cubes to generate at once while building the IceBank. Overrides --config.",
    )
    parser.add_argument(
        "--icecube_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Side length (in voxels) of each cubic ice block. Overrides --config.",
    )
    parser.add_argument(
        "--crowd_min_distance",
        type=float,
        default=argparse.SUPPRESS,
        help="Min distance between crowded particles in Ångstrom. Overrides --config.",
    )
    parser.add_argument(
        "--crowd_max_distance_z",
        type=float,
        default=argparse.SUPPRESS,
        help="Max z-distance between crowded particles in Ångstrom. Overrides --config.",
    )
    parser.add_argument(
        "--potential_scale_min",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum potential scale factor. Overrides --config.",
    )
    parser.add_argument(
        "--potential_scale_max",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum potential scale factor. Overrides --config.",
    )
    parser.add_argument(
        "--pad_fft",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Pad volume for FFT to avoid edge artifacts. Overrides --config.",
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
        "--normalize_particles",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Normalize particles to zero mean and unit std. Overrides --config.",
    )
    parser.add_argument(
        "--save_exitwaves",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Save exit wave magnitude and phase as separate .mrcs files. Overrides --config.",
    )
    parser.add_argument(
        "--save_clean_exitwaves",
        type=lambda x: x.lower() == "true",
        default=argparse.SUPPRESS,
        metavar="True|False",
        help="Save clean (particle-only, no ice) exit wave magnitude and phase. Overrides --config.",
    )

    # --- Compute ---
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Device to use. Options: cpu | cuda | cuda:0 | 0,1,2,3. "
            "Comma-separated integers trigger multi-GPU Lightning DDP. Overrides --config."
        ),
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of particles per forward pass. Overrides --config.",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Directory to save .mrcs and .star files. Overrides --config.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=argparse.SUPPRESS,
        help="Base name for output files (no extension). Overrides --config.",
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
    # Single bare integer → cuda:N
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

    # Only rank 0 reassembles; worker ranks exit cleanly
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
    from specter import rotations
    from specter.cryosparc import create_particle_starfile
    from specter.ice import IceBank
    from specter.image import normalize_particles
    from specter.imagegenerator import ImageGenerator
    from specter.pdb import PDB
    from specter.potential import PotentialBuilder
    from specter.progress import track

    from specter.config import apply_overrides, load_config

    args = parse_args()
    config = load_config(args.config)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    apply_overrides(config, overrides)
    specter.set_verbosity(logging.INFO)

    mode, device_target = _parse_device(config.device)
    t_start = time.perf_counter()

    # LOCAL_RANK is absent in the original process and set (0, 1, ...) in DDP workers
    is_main = "LOCAL_RANK" not in os.environ

    # --- Building 3D scattering potential ---
    if is_main:
        _section("Building 3D scattering potential")
    pdb = PDB(
        config.pdb_code, assembly=config.assembly, savefolder=config.pdb_savefolder
    )

    # Convert cs from mm → Ångstrom (1 mm = 1e7 Å)
    cs_angstrom = config.cs * 1e7

    # Only the main process (always global rank 0 for this single-node launcher)
    # builds V for real. Other DDP ranks hold a zero placeholder of the same
    # shape: Lightning's trainer.predict() calls _sync_module_states() before
    # the predict loop starts, which broadcasts rank 0's real buffer values
    # (V is a registered buffer) to every rank — so building it per-rank would
    # just be wasted, redundant compute.
    if is_main:
        pb = PotentialBuilder(
            config.num_pixels, config.pixel_size, pdb.atomic_numbers
        ).to("cpu")
        with torch.no_grad():
            V = pb(pdb.coordinates).clone()
    else:
        V = torch.zeros(config.num_pixels, config.num_pixels, config.num_pixels)

    # --- Sampling poses, defocus, and translations ---
    if is_main:
        _section("Sampling poses, defocus, and translations")
    n = config.n_particles

    quats = rotations.random_quaternion(n)

    defocus_max = (
        config.defocus_max if config.defocus_max is not None else config.defocus_min
    )
    defocus_A = torch.rand(n) * (defocus_max - config.defocus_min) + config.defocus_min
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_A,
    }

    rlnOriginXAngst = 2 * (torch.rand(n) - 0.5) * config.shift
    rlnOriginYAngst = 2 * (torch.rand(n) - 0.5) * config.shift
    translations = torch.stack([rlnOriginXAngst, rlnOriginYAngst], dim=-1)

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
    num_frames = (
        config.num_frames if config.num_frames is not None else int(dose.mean().item())
    )
    cc_angstrom = config.cc * 1e7 if config.cc is not None else None

    # --- Pre-building ice bank ---
    # IceBank.build() runs an internal L-BFGS optimization (for the 'gd' method)
    # that requires autograd, so it can't happen lazily inside solvate() under
    # torch.no_grad()/Lightning's inference-mode predict loop — it must be built
    # eagerly here, before any forward pass. Only the main process (rank 0) does
    # this real build; other DDP ranks allocate a zero placeholder of the same
    # shape and rely on Lightning's automatic _sync_module_states() (see the V
    # comment above) to receive rank 0's real bank before predicting.
    icecube_size = (
        config.icecube_size
        if config.icecube_size is not None
        else min(config.num_pixels, int(256 / config.pixel_size))
    )
    icemaker = None
    if ice_model is not None:
        if is_main:
            _section("Pre-building ice bank")
        icemaker = IceBank(
            n=icecube_size,
            dx=config.pixel_size,
            method=ice_model,
            num_unique=config.num_unique_icecubes,
            build_batch_size=config.ice_build_batch_size,
        )
        if is_main:
            icemaker_device = (
                f"cuda:{device_target[0]}" if mode == "multi" else device_target
            )
            if icemaker_device != "cpu":
                icemaker = icemaker.to(icemaker_device)
            icemaker.build()
        else:
            icemaker.allocate_placeholder()

    model = ImageGenerator(
        V,
        config.pixel_size,
        quats,
        translations,
        ctf_params,
        config.energy,
        dose,
        icemaker=icemaker,
        ice_thickness=config.ice_thickness,
        scattering_model=config.scattering_model,
        aberration_model=config.aberration_model,
        noise_model=noise_model,
        klim=None,
        ews_curvature_sign="positive",
        alpha=config.alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=config.crowd_max_distance_z,
        pad_fft=config.pad_fft,
        detector_model=detector_model,
        verbose=False,
        coincidence_radius=coincidence_radius,
        num_frames=num_frames,
        potential_scale=potential_scale,
        convergence_angle=config.convergence_angle,
        cc=cc_angstrom,
        energy_spread=config.energy_spread,
        deltaV_V=config.deltaV_V,
        deltaI_I=config.deltaI_I,
        dose_envelope=config.dose_envelope,
        bfactor=config.bfactor,
    )

    if config.save_clean_exitwaves:
        model.save_clean_exitwaves = True

    # --- Generating images ---
    if mode == "multi":
        if is_main:
            _section(f"Initializing multi-GPU on devices {device_target}")
        images, exitwaves, clean_exitwaves = _generate_multi(
            model,
            n,
            config.batchsize,
            device_target,
            config.output_dir,
            collect_exitwaves=config.save_exitwaves,
            collect_clean_exitwaves=config.save_clean_exitwaves,
        )
        if images is None:
            return  # worker rank — rank 0 handles saving
    else:
        if is_main:
            _section(f"Generating images on {device_target}")
        model = model.to(device_target)
        images, exitwaves, clean_exitwaves = _generate_single(
            model,
            n,
            config.batchsize,
            track,
            collect_exitwaves=config.save_exitwaves,
            collect_clean_exitwaves=config.save_clean_exitwaves,
        )

    # --- Post-processing ---
    if is_main:
        _section("Post-processing")
    if config.normalize_particles:
        particles, _means, _stds = normalize_particles(images)
        particles = -particles
    else:
        particles = images

    if is_main:
        _section("Saving .mrcs + .star")
    create_particle_starfile(
        particles,
        rotations=quats,
        translations=translations,
        ctf_params=ctf_params,
        dx=config.pixel_size,
        energy=config.energy,
        alpha=config.alpha,
        filename=config.filename,
        folderpath=config.output_dir,
        dose_per_angstrom=dose,
        coincidence_radius=coincidence_radius,
        potential_scale=potential_scale,
    )

    if is_main:
        import mrcfile

        def _save_exitwave_pair(ew, suffix: str) -> None:
            if config.pad_fft:
                ew = _crop_center(ew, config.num_pixels)
            os.makedirs(config.output_dir, exist_ok=True)
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
