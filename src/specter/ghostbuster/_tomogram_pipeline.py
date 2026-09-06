from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import mrcfile
import roma
import torch
import torch.nn as nn
import torch.utils.data

from ._run_helpers import build_trainer, resolve_device
from ._tomogram_reconstructor import TomogramReconstructor
from ..settings import Optics, Propagation
from specter.options import Scheduler, TiltAxis


class TomogramGhostbuster:
    """
    End-to-end tomogram reconstruction pipeline for cryo-ET tilt series.

    Loads tilt-series images, builds a :class:`TomogramReconstructor`, and
    drives it via a Lightning ``Trainer``.  The ``run`` / ``test_run`` API
    mirrors :class:`Ghostbuster`.

    The forward model is noiseless; the observed images are compared directly
    to ``|CTF(exitwave)|²``.  Images should be preprocessed so that their
    intensity scale matches this quantity.  For simulated data from
    :class:`~specter.imagegenerator.TiltSeriesGenerator`, pass
    ``clean_images`` directly.  For experimental data, dividing by the
    expected dose per pixel (``dose_per_angstrom * voxel_size²``) brings
    images into the correct scale.

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
    angles : sequence of float or torch.Tensor, optional
        Tilt angles in degrees.  Mutually exclusive with ``quaternions``.
    quaternions : torch.Tensor, optional
        Per-tilt rotation quaternions ``(N_tilts, 4)``.  Mutually exclusive
        with ``angles``.
    translations : torch.Tensor, optional
        Per-tilt in-plane translations in Å ``(N_tilts, 2)``.  Defaults to
        zero.
    tilt_axis : str
        Tilt rotation axis (``"x"`` or ``"y"``).  Default ``"x"``.
    nz : int, optional
        Z depth of the reconstructed volume in voxels.  Defaults to the image
        width (square volume).
    V_init : torch.Tensor, optional
        Initial volume ``(Z, Y, X)``.  Defaults to all-zeros.
    flip_contrast : bool
        Negate ``tilt_series`` on load (standard cryo-EM convention).
        Default ``True``.
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
    taper_width : int
        XY cosine taper on V before each forward pass.  Default 0.
    z_taper_width : int
        Z cosine taper on V.  Default 0.
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
    """

    def __init__(
        self,
        tilt_series: torch.Tensor | str | Path,
        voxel_size: float,
        voltage: float,
        ctf_params: dict[str, Any],
        angles: Sequence[float] | torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        tilt_axis: TiltAxis = "x",
        nz: int | None = None,
        V_init: torch.Tensor | None = None,
        flip_contrast: bool = True,
        lr: float | None = None,
        sparsity: float | None = None,
        epochs: int = 5,
        batchsize: int = 1,
        propagation: Propagation = Propagation(),
        optics: Optics = Optics(),
        taper_width: int = 0,
        z_taper_width: int = 0,
        use_fov_mask: bool = True,
        scheduler: Scheduler = "LambdaLR",
        slice_batchsize: int = 1,
        num_workers: int = 0,
        precision: str = "16-mixed",
        run_dir: str | Path | None = None,
    ) -> None:
        images = self._load_tilt_series(tilt_series, flip_contrast)
        n_tilts, H, W = images.shape
        print(
            f"  {n_tilts} tilts  |  {H}×{W} px  |  {voxel_size:.3f} Å/px  |  "
            f"{voltage:.0f} kV"
        )

        quats = self._resolve_tilt_quaternions(angles, quaternions, tilt_axis)
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

        self.lr = lr
        self.sparsity = sparsity
        self.epochs = epochs
        self.batchsize = batchsize
        self.propagation = propagation
        self.optics = optics
        self.taper_width = taper_width
        self.z_taper_width = z_taper_width
        self.use_fov_mask = use_fov_mask
        self.tilt_axis = tilt_axis
        self.scheduler = scheduler
        self.slice_batchsize = slice_batchsize
        self.num_workers = num_workers
        self.precision = precision
        self.run_dir = Path(run_dir) if run_dir is not None else None

    @staticmethod
    def _load_tilt_series(
        tilt_series: torch.Tensor | str | Path, flip_contrast: bool
    ) -> torch.Tensor:
        """Load a tilt series from a tensor or an .mrc file path, optionally flipping contrast."""
        if isinstance(tilt_series, (str, Path)):
            print(f"Loading tilt series from {Path(tilt_series).name} ...")
            with mrcfile.open(str(tilt_series)) as mrc:
                images = torch.as_tensor(mrc.data.copy()).float()
        else:
            images = torch.as_tensor(tilt_series).float()

        if flip_contrast:
            images = -images
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
            tilt_axis=self.tilt_axis,
            lr=self.lr,
            sparsity=self.sparsity,
            taper_width=self.taper_width,
            z_taper_width=self.z_taper_width,
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
        use_gpu, device = resolve_device(device)
        _device_str = f"GPU {device}" if use_gpu else "CPU"
        print(
            f"Starting reconstruction: {n_tilts} tilts  |  "
            f"volume {nz}×{nxy}×{nxy}  |  {self.propagation.scattering_model}  |  "
            f"{self.epochs} epochs  |  batch {self.batchsize}  |  {_device_str}"
        )
        model, loader = self._build_reconstructor_and_loader(
            self._images, self._volume_init, self._voxel_size, self.batchsize
        )
        trainer = build_trainer(use_gpu, device, self.epochs, self.precision, callbacks)
        trainer.fit(model, loader)
        return model

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
        print(
            f"Test run: {len(self._images)} tilts  |  {bin_factor}× binned  |  1 epoch"
        )
        pool = nn.AvgPool2d(bin_factor, stride=bin_factor)
        images_binned = pool(self._images.unsqueeze(1)).squeeze(1) * bin_factor**2
        voxel_size_binned = self._voxel_size * bin_factor

        # Bin the initial volume in XY and Z with adaptive pooling
        V_b = (
            nn.functional.avg_pool3d(
                self._volume_init.unsqueeze(0).unsqueeze(0),
                kernel_size=bin_factor,
                stride=bin_factor,
            )
            .squeeze(0)
            .squeeze(0)
            * bin_factor**3
        )

        model, loader = self._build_reconstructor_and_loader(
            images_binned, V_b, voxel_size_binned, self.batchsize
        )
        use_gpu, device = resolve_device(device)
        trainer = build_trainer(use_gpu, device, 1, "32", callbacks)
        trainer.fit(model, loader)
        v = model.V.detach()
        print(
            f"Test run passed — {bin_factor}× binned, {len(self._images)} tilts, "
            f"volume {tuple(v.shape)}  |  "
            f"V min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}"
        )
        return model
