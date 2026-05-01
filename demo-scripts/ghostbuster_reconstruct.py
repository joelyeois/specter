"""
Ghostbuster 3D reconstruction from a CryoSPARC .cs file and a particle stack.

Poses, CTF parameters, pixel size, energy, and amplitude contrast are read from
the .cs file.  Particle images are supplied separately as a .pt or .mrcs file.

Usage (single run, full dataset):
    python ghostbuster_reconstruct.py \\
        --cs_path   /path/to/particles.cs \\
        --particles /path/to/particles.pt \\
        --dose      53 \\
        --run_dir   /path/to/output/run001

Usage (halfset A, on HPC node):
    python ghostbuster_reconstruct.py \\
        --cs_path   /path/to/particles.cs \\
        --particles /path/to/particles.pt \\
        --dose      53 \\
        --halfset   A \\
        --run_dir   /path/to/output/run001

Usage (halfset B, on a second HPC node, same --run_dir):
    python ghostbuster_reconstruct.py \\
        --cs_path   /path/to/particles.cs \\
        --particles /path/to/particles.pt \\
        --dose      53 \\
        --halfset   B \\
        --run_dir   /path/to/output/run001

Both halfset jobs write into the same run_dir:
    run001/
        params_A.json
        params_B.json
        epochs/
            001_A.mrc
            001_B.mrc
            ...
        vol_A.mrc
        vol_B.mrc

PBS example (submit one script per halfset, pointing at the same --run_dir):
    qsub -v HALFSET=A pbs_reconstruct.sh
    qsub -v HALFSET=B pbs_reconstruct.sh
"""

import argparse
import time
from pathlib import Path

import lightning as L
import mrcfile
import torch
import torch.utils.data
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from specter.cryosparc import extract_parameters_from_csfile
from specter.ghostbuster import Ghostbuster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ghostbuster 3D reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Inputs ---
    parser.add_argument(
        "--cs_path",
        type=str,
        required=True,
        help="Path to CryoSPARC .cs file containing poses and CTF parameters.",
    )
    parser.add_argument(
        "--particles",
        type=str,
        required=True,
        help=(
            "Path to particle images.  Accepts a .pt file (torch tensor, shape "
            "(N, H, W)) or a .mrcs file."
        ),
    )
    parser.add_argument(
        "--dose",
        type=float,
        required=True,
        help="Electron dose in e⁻/Å².  Check the EMDB Experiment tab for this value.",
    )
    parser.add_argument(
        "--init_volume",
        type=str,
        default=None,
        help=(
            "Path to a .mrc file used as the initial volume estimate.  "
            "If omitted, the volume is initialised to zeros and --volume_size is required."
        ),
    )
    parser.add_argument(
        "--volume_size",
        type=int,
        default=None,
        help=(
            "Side length of the cubic volume in voxels.  Required when "
            "--init_volume is not provided."
        ),
    )

    # --- Halfset ---
    parser.add_argument(
        "--halfset",
        type=str,
        default=None,
        choices=["A", "B"],
        help=(
            "Which halfset to reconstruct.  Reads the alignments3D/split field "
            "from the .cs file to assign particles.  Omit to use all particles."
        ),
    )

    # --- Reconstruction hyperparameters ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of particles per gradient step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the volume.  Set to 0 to freeze the volume.",
    )
    parser.add_argument(
        "--lr_r",
        type=float,
        default=None,
        help="Learning rate for rotations.  Omit to freeze rotations.",
    )
    parser.add_argument(
        "--lr_t",
        type=float,
        default=None,
        help="Learning rate for translations.  Omit to freeze translations.",
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=None,
        help="Learning rate for defocus offset.  Omit to freeze defocus.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="L1 sparsity weight on the volume.  Omit to disable.",
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        default=None,
        help="Point-group symmetry to enforce (e.g. C3, D2, I).  Omit for no symmetry.",
    )
    parser.add_argument(
        "--symmetry_batchsize",
        type=int,
        default=1,
        help=(
            "Number of symmetry operations applied per batch during symmetrisation.  "
            "Lower values use less RAM at the cost of speed.  Default 1 is safest for HPC."
        ),
    )
    parser.add_argument(
        "--scattering_model",
        type=str,
        default="multislice",
        choices=["multislice", "rytov", "firstborn", "projection"],
        help="Scattering model used in the forward pass.",
    )
    parser.add_argument(
        "--aberration_model",
        type=str,
        default="holography",
        choices=["holography", "ctf"],
        help="Aberration model used in the forward pass.",
    )
    parser.add_argument(
        "--use_ncc",
        action="store_true",
        help="Use normalised cross-correlation loss instead of MSE.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="untagged",
        help="Human-readable label stored in params.json.",
    )

    # --- Compute ---
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use.  Pass -1 for CPU.",
    )

    # --- Output ---
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help=(
            "Directory to write epoch volumes, final volume, and metadata into.  "
            "Created if it does not exist.  Both halfset jobs should share the same path."
        ),
    )

    return parser.parse_args()


def _load_particles(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix == ".pt":
        return torch.load(p, weights_only=True)
    if p.suffix in (".mrcs", ".mrc"):
        with mrcfile.open(p, mode="r", permissive=True) as mrc:
            data = mrc.data.copy()
        return torch.from_numpy(data).float()
    raise ValueError(
        f"Unsupported particle file format: {p.suffix}  (expected .pt or .mrcs)"
    )


def _load_init_volume(path: str) -> torch.Tensor:
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = mrc.data.copy()
    return torch.from_numpy(data).float()


_con = Console()


def _section(msg: str) -> None:
    _con.print(Rule(f"[bold yellow]{msg}[/bold yellow]", style="yellow"))


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    accelerator = "cpu" if args.gpu < 0 else "gpu"
    devices = 1 if args.gpu < 0 else [args.gpu]
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"

    # ------------------------------------------------------------------ #
    # 1. Load poses and CTF from .cs file                                  #
    # ------------------------------------------------------------------ #
    _section("Loading parameters from .cs file")

    (
        energy_kev,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        _scale,
        _anisomag,
        _indices,
        halfset_labels,
    ) = extract_parameters_from_csfile(args.cs_path)

    tbl = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    tbl.add_column("key", style="bold dim")
    tbl.add_column("val")
    tbl.add_row("Particles (total)", str(len(rotations)))
    tbl.add_row("Energy", f"{float(energy_kev):.1f} keV")
    tbl.add_row("Pixel size", f"{float(pixel_size):.4f} Å")
    _con.print(tbl)

    # ------------------------------------------------------------------ #
    # 2. Load particle images                                              #
    # ------------------------------------------------------------------ #
    _section("Loading particle images")

    images = _load_particles(args.particles)
    _con.print(
        f"Loaded {images.shape[0]} particles  {images.shape[1]}×{images.shape[2]} px"
    )

    if images.shape[0] != len(rotations):
        raise ValueError(
            f"Particle count mismatch: {images.shape[0]} images vs "
            f"{len(rotations)} poses in .cs file."
        )

    # ------------------------------------------------------------------ #
    # 3. Subset to requested halfset                                       #
    # ------------------------------------------------------------------ #
    halfset_label: str | None = None
    if args.halfset is not None:
        if halfset_labels is None:
            raise ValueError(
                "--halfset requested but the .cs file has no alignments3D/split field."
            )
        split_val = 0 if args.halfset == "A" else 1
        mask = halfset_labels == split_val
        idx = torch.where(mask)[0]
        images = images[idx]
        rotations = rotations[idx]
        translations_A = translations_A[idx]
        ctf_params = {k: v[idx] for k, v in ctf_params.items()}
        halfset_label = args.halfset
        _con.print(
            f"Halfset {args.halfset}: {len(idx)} / {len(mask)} particles selected"
        )

    # ------------------------------------------------------------------ #
    # 4. Initialise volume                                                 #
    # ------------------------------------------------------------------ #
    _section("Initialising volume")

    if args.init_volume is not None:
        V = _load_init_volume(args.init_volume)
        _con.print(
            f"Loaded initial volume from {args.init_volume}  shape={tuple(V.shape)}"
        )
    else:
        if args.volume_size is None:
            raise ValueError("Provide --volume_size N when --init_volume is not given.")
        n = args.volume_size
        V = torch.zeros(n, n, n)
        _con.print(f"Initialised volume to zeros  shape=({n}, {n}, {n})")

    # ------------------------------------------------------------------ #
    # 5. Build Ghostbuster                                                 #
    # ------------------------------------------------------------------ #
    _section("Building Ghostbuster")

    lr = args.lr if args.lr != 0.0 else None

    model = Ghostbuster(
        V=V,
        voxel_size=float(pixel_size),
        quaternions=rotations,
        translations=translations_A,
        ctf_params=ctf_params,
        energy=float(energy_kev),
        dose_per_angstrom=args.dose,
        lr=lr,
        lr_R=args.lr_r,
        lr_T=args.lr_t,
        lr_D=args.lr_d,
        sparsity=args.sparsity,
        symmetry=args.symmetry,
        symmetry_batchsize=args.symmetry_batchsize,
        scattering_model=args.scattering_model,
        aberration_model=args.aberration_model,
        use_ncc=args.use_ncc,
        tag=args.tag,
        run_dir=args.run_dir,
        halfset_label=halfset_label,
    )

    # ------------------------------------------------------------------ #
    # 6. Build dataloader                                                  #
    # ------------------------------------------------------------------ #
    n_particles = images.shape[0]
    dataset = torch.utils.data.TensorDataset(images, torch.arange(n_particles))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(accelerator == "gpu"),
    )

    # ------------------------------------------------------------------ #
    # 7. Train                                                             #
    # ------------------------------------------------------------------ #
    _section(
        f"Reconstructing on {device}  ({n_particles} particles, {args.epochs} epochs)"
    )

    torch.set_float32_matmul_precision("high")

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, dataloader)

    # ------------------------------------------------------------------ #
    # 8. Done                                                              #
    # ------------------------------------------------------------------ #
    elapsed = time.perf_counter() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"
    _con.print(f"\n[bold]Total time:[/bold] {time_str}")


if __name__ == "__main__":
    main()
