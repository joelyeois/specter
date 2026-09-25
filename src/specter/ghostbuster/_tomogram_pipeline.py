"""
`TomogramGhostbuster`: the end-to-end cryo-ET reconstruction pipeline, from
a tilt series and its geometry to a trained `TomogramReconstructor`.
"""

from __future__ import annotations

from ..progress import console

from pathlib import Path
from typing import Any, Sequence

import mrcfile
import roma
import torch
import torch.nn as nn
import torch.utils.data

from ._helpers import _images_to_counts
from ._pipeline_base import _GhostbusterBase
from ._tomogram_reconstructor import TomogramReconstructor
from ..settings import Optics, Propagation, TiltGeometry
from specter.options import ImageUnits, Scheduler, TiltAxis


class TomogramGhostbuster(_GhostbusterBase):
    """
    End-to-end tomogram reconstruction pipeline for cryo-ET tilt series.

    Loads tilt-series images, builds a :class:`TomogramReconstructor`, and
    drives it via a Lightning ``Trainer``.  The ``run`` / ``test_run`` API
    mirrors :class:`Ghostbuster`.

    The forward model is noiseless and predicts electron counts per pixel,
    ``dose_per_angstrom * voxel_size² * |CTF(exitwave)|²``, as the particle
    pipeline's does. ``image_units`` says what the tilt series holds: counts
    (the default, and what a motion-corrected tilt series and
    :class:`~specter.imagegenerator.TiltSeriesGenerator`'s detected images
    are) are used as they are, and a normalised series is mapped back to
    counts from the dose.

    Parameters
    ----------
    tilt_series : torch.Tensor or str or Path
        Observed tilt-series images, shape ``(N_tilts, H, W)``, or path to a
        ``.mrc`` file containing the tilt series.
    voxel_size : float
        Pixel size in Å.
    voltage : float
        Electron beam accelerating voltage in kV.
    ctf_params : dict[str, torch.Tensor]
        Per-tilt CTF parameters; each value must have leading dimension
        ``N_tilts``.
    dose_per_angstrom : float or torch.Tensor
        Electron dose (fluence) per tilt image in e⁻/Å². Scalar, or a 1-D
        tensor of length ``N_tilts`` for a dose that differs per tilt.
    angles : sequence of float or torch.Tensor, optional
        Tilt angles in degrees.  Mutually exclusive with ``quaternions``.
    quaternions : torch.Tensor, optional
        Per-tilt rotation quaternions ``(N_tilts, 4)``.  Mutually exclusive
        with ``angles``.
    translations : torch.Tensor, optional
        Per-tilt in-plane translations in Å ``(N_tilts, 2)``.  Defaults to
        zero.
    tilt : TiltGeometry, optional
        Tilt axis and edge tapers, see `TomogramReconstructor`. Default
        ``TiltGeometry()``.
    nz : int, optional
        Z depth of the reconstructed volume in voxels.  Defaults to the image
        width (square volume).
    V_init : torch.Tensor, optional
        Initial volume ``(Z, Y, X)``.  Defaults to all-zeros.
    flip_contrast : bool, optional
        Negate a normalised ``tilt_series`` before converting it to counts,
        for a series stored with inverted contrast. ``None`` (the default)
        flips normalised input and leaves counts alone; ``True`` with counts
        is an error, since counts have a physical sign.
    lr : float, optional
        Learning rate for V.  ``None`` disables optimisation.
    sparsity : float, optional
        L1 regularisation weight on V.
    epochs : int
        Training epochs.  Default 5.
    batchsize : int
        Tilt images per optimisation step.  Default 1 (one tilt per step).
    propagation : Propagation, optional
        How the forward model computes the exit wave (model, amplitude
        contrast, bandlimit). Default ``Propagation()``.
    optics : Optics, optional
        The aberration engine and phase plate. Default ``Optics()``.
    use_fov_mask : bool
        Mask MSE loss to the real-FOV region per tilt.  Default ``True``.
    scheduler : str
        LR scheduler.  Default ``"LambdaLR"``.
    slice_batchsize : int
        Z-slice chunk size for ``IterativeScattering``.  Default 1.
    num_workers : int
        DataLoader worker processes.  Default 0.
    precision : str
        Lightning ``Trainer`` precision.  Default ``"16-mixed"``.
    run_dir : str or Path, optional
        Output directory for volumes and metadata.
    image_units : {"counts", "normalized"}
        What ``tilt_series``' values are: electron counts per pixel, used as
        is, or a series normalised to zero mean and unit variance, mapped
        back to counts as ``sqrt(N) * x + N`` with ``N`` each tilt's dose per
        pixel. Default ``"counts"``.
    """

    def __init__(
        self,
        tilt_series: torch.Tensor | str | Path,
        voxel_size: float,
        voltage: float,
        ctf_params: dict[str, Any],
        dose_per_angstrom: float | torch.Tensor,
        angles: Sequence[float] | torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        tilt: TiltGeometry = TiltGeometry(),
        nz: int | None = None,
        V_init: torch.Tensor | None = None,
        flip_contrast: bool | None = None,
        lr: float | None = None,
        sparsity: float | None = None,
        epochs: int = 5,
        batchsize: int = 1,
        propagation: Propagation = Propagation(),
        optics: Optics = Optics(),
        use_fov_mask: bool = True,
        scheduler: Scheduler = "LambdaLR",
        slice_batchsize: int = 1,
        num_workers: int = 0,
        precision: str = "16-mixed",
        run_dir: str | Path | None = None,
        image_units: ImageUnits = "counts",
    ) -> None:
        images = self._load_tilt_series(tilt_series)
        n_tilts, H, W = images.shape
        dose = torch.as_tensor(dose_per_angstrom, dtype=torch.float32).flatten()
        if dose.numel() not in (1, n_tilts):
            raise ValueError(
                f"dose_per_angstrom has {dose.numel()} entries; expected 1 or "
                f"one per tilt ({n_tilts})"
            )
        if flip_contrast is None:
            flip_contrast = image_units == "normalized"
        images = _images_to_counts(
            images,
            image_units,
            (dose * voxel_size**2).reshape(-1, 1, 1),
            flip_contrast,
        )
        console.print(
            f"  {n_tilts} tilts  |  {H}×{W} px  |  {voxel_size:.3f} Å/px  |  "
            f"{voltage:.0f} kV"
        )

        quats = self._resolve_tilt_quaternions(angles, quaternions, tilt.tilt_axis)
        trans = (
            torch.zeros(n_tilts, 2, dtype=torch.float32)
            if translations is None
            else torch.as_tensor(translations, dtype=torch.float32)
        )
        volume_init = self._build_initial_volume(V_init, nz, nxy=W)

        # Store preprocessed data and settings
        self._images = images
        self._quaternions = quats
        self._translations = trans
        self._ctf_params = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in ctf_params.items()
        }
        self._volume_init = volume_init
        self._voxel_size = voxel_size
        self._voltage = voltage
        self._dose_per_angstrom = dose

        self.lr = lr
        self.sparsity = sparsity
        self.epochs = epochs
        self.batchsize = batchsize
        self.propagation = propagation
        self.optics = optics
        self.tilt = tilt
        self.use_fov_mask = use_fov_mask
        self.scheduler = scheduler
        self.slice_batchsize = slice_batchsize
        self.num_workers = num_workers
        self.precision = precision
        self.run_dir = Path(run_dir) if run_dir is not None else None

    @staticmethod
    def _load_tilt_series(tilt_series: torch.Tensor | str | Path) -> torch.Tensor:
        """Load a tilt series from a tensor or an .mrc file path."""
        if isinstance(tilt_series, (str, Path)):
            console.print(f"Loading tilt series from {Path(tilt_series).name} ...")
            with mrcfile.open(str(tilt_series)) as mrc:
                images = torch.as_tensor(mrc.data.copy()).float()
        else:
            images = torch.as_tensor(tilt_series).float()
        return images

    @staticmethod
    def _resolve_tilt_quaternions(
        angles: Sequence[float] | torch.Tensor | None,
        quaternions: torch.Tensor | None,
        tilt_axis: TiltAxis,
    ) -> torch.Tensor:
        """Resolve per-tilt rotation quaternions from either angles or explicit quaternions."""
        if angles is not None and quaternions is not None:
            raise ValueError("Provide either 'angles' or 'quaternions', not both.")
        if angles is None and quaternions is None:
            raise ValueError("Either 'angles' or 'quaternions' must be provided.")

        if angles is not None:
            angles_t = torch.as_tensor(angles, dtype=torch.float32)
            theta_rad = torch.deg2rad(angles_t)
            tilt_axis_lower = tilt_axis.lower()
            if tilt_axis_lower == "x":
                rotvecs = torch.stack(
                    [
                        theta_rad,
                        torch.zeros_like(theta_rad),
                        torch.zeros_like(theta_rad),
                    ],
                    dim=-1,
                )
            else:
                rotvecs = torch.stack(
                    [
                        torch.zeros_like(theta_rad),
                        theta_rad,
                        torch.zeros_like(theta_rad),
                    ],
                    dim=-1,
                )
            quaternions = roma.rotvec_to_unitquat(rotvecs)

        return torch.as_tensor(quaternions, dtype=torch.float32)

    @staticmethod
    def _build_initial_volume(
        V_init: torch.Tensor | None, nz: int | None, nxy: int
    ) -> torch.Tensor:
        """Build the initial reconstruction volume: V_init if given, else zeros."""
        if V_init is not None:
            return torch.as_tensor(V_init).float()
        depth = nz if nz is not None else nxy
        return torch.zeros(depth, nxy, nxy)

    def _build_reconstructor_and_loader(
        self,
        images: torch.Tensor,
        volume_init: torch.Tensor,
        voxel_size: float,
        batchsize: int,
    ) -> tuple["TomogramReconstructor", torch.utils.data.DataLoader]:
        n_tilts = images.shape[0]
        idx = torch.arange(n_tilts)
        dataset = torch.utils.data.TensorDataset(images, idx)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=self.num_workers,
        )
        model = TomogramReconstructor(
            volume_init,
            voxel_size,
            self._quaternions,
            self._translations,
            self._ctf_params,
            self._voltage,
            self._dose_per_angstrom,
            tilt=self.tilt,
            lr=self.lr,
            sparsity=self.sparsity,
            use_fov_mask=self.use_fov_mask,
            propagation=self.propagation,
            optics=self.optics,
            scheduler=self.scheduler,
            slice_batchsize=self.slice_batchsize,
            run_dir=self.run_dir,
        )
        return model, loader

    def run(
        self,
        device: int | Sequence[int] | str = 0,
        callbacks: list[Any] | None = None,
    ) -> "TomogramReconstructor":
        """
        Run the full reconstruction and return the trained
        :class:`TomogramReconstructor`.

        Parameters
        ----------
        device : int or sequence of int or {"cpu"}
            GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to
            train across multiple GPUs via Lightning DDP. A single tomogram's
            tilt series is usually small (tens of tilts), so the benefit of
            splitting it across GPUs is more limited than for particle-stack
            reconstruction. Pass ``"cpu"`` to force CPU
            training; any other value uses the GPU when CUDA is
            available and falls back to the CPU when it is not.
        callbacks : list, optional
            Additional Lightning callbacks.

        Returns
        -------
        TomogramReconstructor
            Trained model.  Access the volume via ``model.V.detach()``.
        """
        n_tilts = len(self._images)
        nz, nxy = self._volume_init.shape[0], self._volume_init.shape[-1]
        console.print(
            f"Starting reconstruction: {n_tilts} tilts  |  "
            f"volume {nz}×{nxy}×{nxy}  |  {self.propagation.scattering_model}  |  "
            f"{self.epochs} epochs  |  batch {self.batchsize}  |  "
            f"{self._device_label(device)}"
        )
        model, loader = self._build_reconstructor_and_loader(
            self._images, self._volume_init, self._voxel_size, self.batchsize
        )
        return self._fit(model, loader, device, self.epochs, self.precision, callbacks)

    def test_run(
        self,
        bin_factor: int = 4,
        device: int | Sequence[int] | str = 0,
        callbacks: list[Any] | None = None,
    ) -> "TomogramReconstructor":
        """
        Quick sanity check: 1 epoch on spatially binned tilt images.

        Bins images and volume init by ``bin_factor`` in each spatial
        dimension and runs a single epoch.  Use this to verify that data
        loading, CTF parameters, and the physics pipeline are wired up
        correctly before committing to a full run.

        Parameters
        ----------
        bin_factor : int
            Spatial downsampling factor.  Default 4.
        device : int or sequence of int or {"cpu"}
            GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to run
            across multiple GPUs via Lightning DDP, or ``"cpu"`` to force
            CPU training.
        callbacks : list, optional
            Additional Lightning callbacks.

        Returns
        -------
        TomogramReconstructor
            Trained model after one epoch.
        """
        console.print(
            f"Test run: {len(self._images)} tilts  |  {bin_factor}× binned  |  1 epoch"
        )
        images_binned, voxel_size_binned = self._bin_images(bin_factor)

        # Bin the initial volume in XY and Z as well. A voxel holds a
        # potential in volts, not a per-voxel integral, so binning averages:
        # the projected potential sum(V * dz) is then unchanged, as dz grows
        # by bin_factor while the slice count shrinks by it.
        V_b = (
            nn.functional.avg_pool3d(
                self._volume_init.unsqueeze(0).unsqueeze(0),
                kernel_size=bin_factor,
                stride=bin_factor,
            )
            .squeeze(0)
            .squeeze(0)
        )

        model, loader = self._build_reconstructor_and_loader(
            images_binned, V_b, voxel_size_binned, self.batchsize
        )
        model = self._fit(model, loader, device, 1, "32", callbacks)
        self._report_test_run(
            model,
            f"{bin_factor}× binned, {len(self._images)} tilts, "
            f"volume {tuple(model.V.shape)}",
        )
        return model
