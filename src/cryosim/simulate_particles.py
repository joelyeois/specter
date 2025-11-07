#!/usr/bin/env python
import argparse
import torch
from torch.utils.data import DataLoader
import lightning as L
from cryosim.imagegenerator import ImageGenerator
from lightning.pytorch.callbacks import BasePredictionWriter
import os


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        images = torch.concat(predictions, dim=0)
        torch.save(
            images,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        idx = torch.squeeze(torch.tensor(batch_indices)).reshape(-1)
        # idx = torch.concat(idx, dim=0)
        torch.save(
            idx,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate particle images with multi-GPU Lightning."
    )
    parser.add_argument(
        "--V_path", type=str, required=True, help="Path to the saved V tensor (.pt)"
    )
    parser.add_argument("--dx", type=float, required=True, help="Pixel size in Å")
    parser.add_argument("--quaternions_path", type=str, required=True)
    parser.add_argument("--translations_path", type=str, required=True)
    parser.add_argument("--ctf_params_path", type=str, required=True)
    parser.add_argument("--energy", type=float, required=True)
    parser.add_argument("--dose_per_angstrom", type=float, required=True)
    parser.add_argument("--n_particles", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--scattering_model", type=str, default="multislice")
    parser.add_argument("--aberration_model", type=str, default="holography")
    parser.add_argument("--noise_model", type=str, default=None)
    parser.add_argument("--ice_model", type=str, default=None)
    parser.add_argument("--ice_thickness", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--crowd_min_distance", type=float, default=None)
    parser.add_argument("--pad_fft", type=str2bool, default=False)
    parser.add_argument(
        "--output_path",
        type=str,
        default="../cache/images.pt",
        help="Path to save the final image stack",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated GPU device IDs, e.g. '0,1'",
    )
    args = parser.parse_args()

    device_list = [int(d) for d in args.devices.split(",")]

    # Load tensors
    print("Loading tensors...")
    V = torch.load(args.V_path)  # [Z, X, Y]
    quaternions = torch.load(args.quaternions_path)  # [N, 4]
    translations = torch.load(args.translations_path)  # [N, 2]
    ctf_params = torch.load(args.ctf_params_path)  # [N, 9]

    # Initialize model
    model = ImageGenerator(
        V,
        args.dx,
        quaternions,
        translations,
        ctf_params,
        args.energy,
        args.dose_per_angstrom,
        ice_model=args.ice_model,
        ice_thickness=args.ice_thickness,
        scattering_model=args.scattering_model,
        aberration_model=args.aberration_model,
        noise_model=args.noise_model,
        klim=None,
        flip_curvature=False,
        alpha=args.alpha,
        crowd_min_distance=args.crowd_min_distance,
        pad_fft=args.pad_fft,
        progressbars=False,
    )
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)

    # Create DataLoader over particle indices
    idx = torch.arange(args.n_particles)
    dataloader = DataLoader(idx, batch_size=args.batch_size, shuffle=False)

    pred_writer = CustomWriter(output_dir=output_dir, write_interval="epoch")

    # Setup Lightning Trainer with multi-GPU
    trainer = L.Trainer(
        accelerator="gpu",
        devices=device_list,
        strategy="ddp",
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
        callbacks=[pred_writer],  # Add the callback here
    )

    # Predict images
    print("Running predictions...")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)


if __name__ == "__main__":
    main()
