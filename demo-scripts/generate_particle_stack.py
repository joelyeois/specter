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
        --dose 20.0 \
        --defocus_min 5000 \
        --defocus_max 15000 \
        --cs 2.0 \
        --alpha 0.1 \
        --shift_angstroms 2.0 \
        --scattering_model multislice \
        --aberration_model holography \
        --noise_model poisson \
        --coincidence_radius 1.8 \
        --ice_model iterative \
        --ice_thickness 0 \
        --normalize_particles True \
        --device 0,1,2,3 \
        --batchsize 5 \
        --output_dir /scratch/loh/joel/simulated_data/ \
        --filename my_particle_stack
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a cryo-EM particle stack and save as .mrcs + .star.",
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
        help="Number of pixels per axis for the 3-D potential box.",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=1.0,
        help="Pixel size in Ångstrom.",
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
        help="Dose in e⁻/Å².",
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

    # --- Translations / shifts ---
    parser.add_argument(
        "--shift",
        type=float,
        default=2.0,
        help="Max in-plane shift in Ångstrom (uniform ±shift).",
    )

    # --- Dataset size ---
    parser.add_argument(
        "--n_particles",
        type=int,
        default=20,
        help="Number of particles to simulate.",
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
        default=0.0,
        help="Ice thickness in Ångstrom. 0 = minimum (particle box size).",
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

    # --- Post-processing ---
    parser.add_argument(
        "--normalize_particles",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="True|False",
        help="Normalize particles to zero mean and unit std.",
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
        default="./output/",
        help="Directory to save .mrcs and .star files.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="particles",
        help="Base name for output files (no extension).",
    )

    return parser.parse_args()


def _section(msg: str) -> None:
    """Print a bold yellow section separator."""
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"\n{YELLOW_BOLD}--- {msg} ---{RESET}")


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


def _generate_single(model, n: int, batchsize: int, track):
    """Run image generation on a single device."""
    import torch

    idx = torch.arange(n)
    images = []
    with torch.no_grad():
        for i in track(range(0, n, batchsize), description="Generating images"):
            batch = model(idx[i : i + batchsize])
            images.append(batch.detach().cpu())
    return torch.concat(images, dim=0)


def _generate_multi(model, n: int, batchsize: int, gpu_ids: list, output_dir: str):
    """Run image generation across multiple GPUs using Lightning DDP.

    Returns the assembled image tensor on rank 0, None on worker ranks.
    """
    import glob
    import os

    import torch
    import lightning as L
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import BasePredictionWriter
    from typing import Any, Sequence

    class _Writer(BasePredictionWriter):
        def __init__(self, out_dir: str) -> None:
            super().__init__("epoch")
            self.out_dir = out_dir

        def write_on_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            predictions: Sequence[Any],
            batch_indices: Sequence[Any],
        ) -> None:
            images = torch.concat(predictions, dim=0)
            torch.save(
                images,
                os.path.join(self.out_dir, f"predictions_{trainer.global_rank}.pt"),
            )
            idx = torch.squeeze(torch.tensor(batch_indices)).reshape(-1)
            torch.save(
                idx,
                os.path.join(self.out_dir, f"batch_indices_{trainer.global_rank}.pt"),
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
        callbacks=[_Writer(output_dir)],
    )

    print(f"Running multi-GPU generation on GPUs: {gpu_ids}")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)

    # Only rank 0 reassembles; worker ranks exit cleanly
    if trainer.global_rank != 0:
        return None

    # Reassemble in original order
    prediction_files = sorted(glob.glob(os.path.join(output_dir, "predictions_*.pt")))
    index_files = sorted(glob.glob(os.path.join(output_dir, "batch_indices_*.pt")))

    all_preds = torch.cat([torch.load(f) for f in prediction_files], dim=0)
    all_indices = torch.cat([torch.load(f) for f in index_files], dim=0)

    images = all_preds[torch.argsort(all_indices)]

    for f in prediction_files + index_files:
        os.remove(f)

    return images


def main() -> None:
    import logging

    import torch

    import cryosim
    from cryosim import rotations
    from cryosim.cryosparc_utils import create_particle_starfile
    from cryosim.image_tools import normalize_particles
    from cryosim.imagegenerator import ImageGenerator
    from cryosim.pdbtools import PDB
    from cryosim.potential import PotentialBuilder
    from cryosim.progress import track

    import time

    args = parse_args()
    cryosim.set_verbosity(logging.INFO)

    mode, device_target = _parse_device(args.device)
    t_start = time.perf_counter()

    import os as _os

    # LOCAL_RANK is absent in the original process and set (0, 1, ...) in DDP workers
    is_main = "LOCAL_RANK" not in _os.environ

    # --- Building 3D scattering potential ---
    if is_main:
        _section("Building 3D scattering potential")
    pdb = PDB(args.pdb_code, assembly=args.assembly, savefolder=args.pdb_savefolder)

    # Convert cs from mm → Ångstrom (1 mm = 1e7 Å)
    cs_angstrom = args.cs * 1e7

    pb = PotentialBuilder(args.num_pixels, args.pixel_size, pdb.atomic_numbers).to(
        "cpu"
    )
    with torch.no_grad():
        V = pb(pdb.coordinates).clone()

    # --- Sampling poses, defocus, and translations ---
    if is_main:
        _section("Sampling poses, defocus, and translations")
    n = args.n_particles

    quats = rotations.random_quaternion(n)

    defocus_A = torch.rand(n) * (args.defocus_max - args.defocus_min) + args.defocus_min
    ctf_params = {
        "cs": torch.tensor([cs_angstrom] * n),
        "dfu": defocus_A,
    }

    rlnOriginXAngst = 2 * (torch.rand(n) - 0.5) * args.shift
    rlnOriginYAngst = 2 * (torch.rand(n) - 0.5) * args.shift
    translations = torch.stack([rlnOriginXAngst, rlnOriginYAngst], dim=-1)

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

    model = ImageGenerator(
        V,
        args.pixel_size,
        quats,
        translations,
        ctf_params,
        args.energy,
        args.dose,
        ice_model=ice_model,
        ice_thickness=args.ice_thickness,
        scattering_model=args.scattering_model,
        aberration_model=args.aberration_model,
        noise_model=noise_model,
        klim=None,
        flip_curvature=False,
        alpha=args.alpha,
        crowd_min_distance=crowd_min_distance,
        crowd_max_distance_z=args.crowd_max_distance_z,
        pad_fft=args.pad_fft,
        detector_model=detector_model,
        verbose=False,
        coincidence_radius=args.coincidence_radius,
        num_frames=num_frames,
    )

    # --- Generating images ---
    if mode == "multi":
        if is_main:
            _section(f"Initializing multi-GPU on devices {device_target}")
        images = _generate_multi(
            model, n, args.batchsize, device_target, args.output_dir
        )
        if images is None:
            return  # worker rank — rank 0 handles saving
    else:
        if is_main:
            _section(f"Generating images on {device_target}")
        model = model.to(device_target)
        images = _generate_single(model, n, args.batchsize, track)

    # --- Post-processing ---
    if is_main:
        _section("Post-processing")
    if args.normalize_particles:
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
        dx=args.pixel_size,
        energy=args.energy,
        alpha=args.alpha,
        filename=args.filename,
        folderpath=args.output_dir,
    )

    elapsed = time.perf_counter() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        print(f"Total time: {h}h {m}m {s}s")
    elif m > 0:
        print(f"Total time: {m}m {s}s")
    else:
        print(f"Total time: {s}s")


if __name__ == "__main__":
    main()
