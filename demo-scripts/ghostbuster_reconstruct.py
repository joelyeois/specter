"""
Ghostbuster training script using the jobs API.

Converts and reconstructs 3D maps from experimental particles using the
physics-based forward model and parameter refinement.

Usage:
    python ghostbuster_reconstruct.py \\
        --project empiar-10202 \\
        --mrc_file /path/to/particles.mrc \\
        --cs_file /path/to/particles.cs \\
        --fsc_ref /path/to/reference.mrc \\
        --fsc_mask /path/to/mask.mrc \\
        --cryosparc_ref /path/to/cryosparc_vol.mrc \\
        --dose_per_angstrom 22.5 \\
        --symmetry I1 \\
        --return_class 1

Save dataset B to same J001 as dataset A:
    python ghostbuster_reconstruct.py \\
        --project empiar-10202 \\
        --job_id J001 \\
        --mrc_file /path/to/particles_B.mrc \\
        ... (same other params as A)

Submit to PBS:
    qsub -N ghostbuster -l select=1:ngpus=1 -l walltime=24:00:00 \\
         ghostbuster_reconstruct.py

This script uses the jobs API to automatically manage output directories
and metadata tracking.
"""

import argparse
import sys

from specter.ghostbuster import Ghostbuster
import specter.jobs as jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ghostbuster training using the jobs API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data (required) ---
    parser.add_argument("--mrc_file", type=str, required=True)
    parser.add_argument("--cs_file", type=str, required=True)
    parser.add_argument("--fsc_ref", type=str, required=True)
    parser.add_argument("--fsc_mask", type=str, required=True)
    parser.add_argument("--cryosparc_ref", type=str, required=True)
    parser.add_argument("--dose_per_angstrom", type=float, required=True)
    parser.add_argument("--symmetry", type=str, required=True)
    parser.add_argument("--return_class", type=str, required=True)

    # --- Training hyperparameters ---
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_r", type=float, default=None)
    parser.add_argument("--lr_t", type=float, default=None)
    parser.add_argument("--lr_d", type=float, default=None)
    parser.add_argument("--defocus_offset", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="LambdaLR")
    parser.add_argument(
        "--scattering_model",
        type=str,
        default="rytov",
        choices=["multislice", "rytov", "firstborn", "projection"],
    )
    parser.add_argument(
        "--aberration_model",
        type=str,
        default="holography",
        choices=["holography", "ctf"],
    )
    parser.add_argument("--symmetry_batchsize", type=int, default=3)
    parser.add_argument("--symmetry_mode", type=str, default="fourier")
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--rotate_mode", type=str, default="real")
    parser.add_argument(
        "--ews_curvature_sign",
        type=str,
        default="negative",
        choices=["negative", "positive"],
    )
    parser.add_argument("--klim", type=float, default=None)
    parser.add_argument("--nps_weight", type=float, default=None)
    parser.add_argument("--learn_noise_model", action="store_true")
    parser.add_argument("--use_ncc", action="store_true")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--num_particles", type=int, default=None)

    # --- Compute ---
    parser.add_argument("--device", type=int, default=1)

    # --- Job management ---
    parser.add_argument(
        "--job_base_dir", type=str, default="/scratch/loh/joel/specter/"
    )
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Save into existing job directory (e.g. 'J001'). Omit to create new job.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jobs.base_directory(args.job_base_dir)

    print(f"Running Ghostbuster training on device {args.device}")
    print(f"MRC file: {args.mrc_file}")
    print(f"CS file: {args.cs_file}")
    print(f"Epochs: {args.epochs}")

    try:
        with jobs.Job("ghostbuster", args.project, job_id=args.job_id) as job:
            print(f"Job folder: {job.dir}")

            gb = job.create(
                Ghostbuster,
                cs_file=args.cs_file,
                mrc_file=args.mrc_file,
                dose_per_angstrom=args.dose_per_angstrom,
                lr=args.lr,
                lr_R=args.lr_r,
                lr_T=args.lr_t,
                lr_D=args.lr_d,
                defocus_offset=args.defocus_offset,
                scheduler=args.scheduler,
                epochs=args.epochs,
                batch_size=args.batch_size,
                scattering_model=args.scattering_model,
                aberration_model=args.aberration_model,
                symmetry=args.symmetry,
                symmetry_batchsize=args.symmetry_batchsize,
                symmetry_mode=args.symmetry_mode,
                sparsity=args.sparsity,
                rotate_mode=args.rotate_mode,
                ews_curvature_sign=args.ews_curvature_sign,
                klim=args.klim,
                nps_weight=args.nps_weight,
                learn_noise_model=args.learn_noise_model,
                use_ncc=args.use_ncc,
                fsc_ref=args.fsc_ref,
                fsc_mask=args.fsc_mask,
                cryosparc_ref=args.cryosparc_ref,
                precision=args.precision,
                num_workers=args.num_workers,
                num_particles=args.num_particles,
                return_class=args.return_class,
            )

            _ = gb.run(device=args.device)

            print(f"Training complete. Results saved to {job.dir}")

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
