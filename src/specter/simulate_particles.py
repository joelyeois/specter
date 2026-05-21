from __future__ import annotations

import argparse
import os
from typing import Any, Sequence

import lightning as L
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from specter.imagegenerator import ImageGenerator


class CustomWriter(BasePredictionWriter):
    """
    Custom prediction writer for saving particle simulation results.

    Saves predictions and batch indices to disk at the end of each epoch,
    with separate files for each process rank in distributed training.

    Parameters
    ----------
    output_dir : str
        Directory where prediction files will be saved.
    write_interval : str
        When to write predictions. Typically 'epoch' or 'batch'.

    Attributes
    ----------
    output_dir : str
        Directory for saving output files.
    """

    def __init__(self, output_dir: str, write_interval: str) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        """
        Write predictions and batch indices to disk at epoch end.

        Parameters
        ----------
        trainer : lightning.Trainer
            PyTorch Lightning trainer instance.
        pl_module : lightning.LightningModule
            The LightningModule being trained.
        predictions : list of torch.Tensor
            List of prediction batches from all processes.
        batch_indices : list
            List of batch indices corresponding to predictions.

        Notes
        -----
        Creates two files per process rank:
        - predictions_{rank}.pt: Concatenated predictions
        - batch_indices_{rank}.pt: Corresponding data indices
        """
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


def str2bool(v: str | bool) -> bool:
    """
    Convert string representation to boolean value.

    Accepts various string formats representing True or False values.

    Parameters
    ----------
    v : str or bool
        Value to convert. Can be boolean or string.

    Returns
    -------
    bool
        Boolean interpretation of input value.

    Raises
    ------
    argparse.ArgumentTypeError
        If string value is not a recognized boolean representation.

    Examples
    --------
    >>> str2bool('yes')
    True
    >>> str2bool('false')
    False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def float_or_none(value):
    if value.lower() == "none":
        return None
    return float(value)


def main():
    """
    Simulate particle images using multi-GPU Lightning framework.

    Command-line entry point for generating simulated cryo-EM particle images.
    Loads pre-computed potential, quaternions, translations, and CTF parameters,
    then uses PyTorch Lightning with distributed data parallel (DDP) to generate
    images across multiple GPUs.

    Command-line Arguments
    ----------------------
    Required:
        --V_path : str
            Path to saved 3D potential volume tensor (.pt file).
        --dx : float
            Pixel size in Å.
        --quaternions_path : str
            Path to rotation quaternions tensor (.pt file).
        --translations_path : str
            Path to 2D translations tensor (.pt file).
        --ctf_params_path : str
            Path to CTF parameters tensor (.pt file).
        --energy : float
            Electron beam energy in keV.
        --dose_per_angstrom : float
            Electron dose in electrons per Å².
        --n_particles : int
            Total number of particle images to generate.

    Optional:
        --batch_size : int
            Batch size for image generation. Default is 5.
        --scattering_model : str
            Scattering model ('multislice', 'projection', etc.). Default is 'multislice'.
        --aberration_model : str
            Aberration model ('holography' or 'ctf'). Default is 'holography'.
        --noise_model : str
            Noise model ('poisson' or None). Default is None.
        --ice_model : str
            Ice model to use. Default is None.
        --ice_thickness : float
            Ice thickness in Å. Default is 0.
        --alpha : float
            Amplitude contrast ratio. Default is 0.0.
        --crowd_min_distance : float
            Minimum distance between crowded particles. Default is None.
        --pad_fft : bool
            Whether to pad for FFT operations. Default is False.
        --output_path : str
            Path to save final image stack. Default is '../cache/images.pt'.
        --devices : str
            Comma-separated GPU device IDs (e.g., '0,1'). Default is '0'.

    Notes
    -----
    Output files are saved per GPU rank:
    - predictions_{rank}.pt: Generated images
    - batch_indices_{rank}.pt: Corresponding particle indices
    """
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
    parser.add_argument("--coincidence_radius", type=float, default=0.0)
    parser.add_argument("--ice_model", type=str, default=None)
    parser.add_argument("--ice_thickness", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--crowd_min_distance", type=float, default=None)
    parser.add_argument("--crowd_max_distance_z", type=float_or_none, default=None)
    parser.add_argument("--pad_fft", type=str2bool, default=False)
    parser.add_argument("--detector_model", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=1)
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
        coincidence_radius=args.coincidence_radius,
        klim=None,
        ews_curvature_sign="positive",
        alpha=args.alpha,
        crowd_min_distance=args.crowd_min_distance,
        crowd_max_distance_z=args.crowd_max_distance_z,
        pad_fft=args.pad_fft,
        detector_model=args.detector_model,
        progressbars=False,
        num_frames=args.num_frames,
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
