"""
`TomogramGhostbuster`: the end-to-end cryo-ET reconstruction pipeline, from
a tilt series and its geometry to a trained `TomogramReconstructor`.
"""

from __future__ import annotations

from .._defaults import LR_DECAY, SCHEDULER
from ..io import read_mrc
from ..progress import console

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import roma
import torch
import torch.nn as nn
import torch.utils.data

from ._helpers import _images_to_counts
from ._pipeline_base import _GhostbusterBase
from ._run_helpers import torch_devices
from ._tomogram_reconstructor import TomogramReconstructor
from ._tomogram_solver import solve_tilt_series, solve_tilt_series_in_strips
from ..settings import Optics, Propagation, TiltGeometry
from specter.options import EwaldSphereSign, ImageUnits, Scheduler, TiltAxis


class TomogramGhostbuster(_GhostbusterBase):
    """
    End-to-end tomogram reconstruction pipeline for cryo-ET tilt series.

    Loads tilt-series images, builds a :class:`TomogramReconstructor`, and
    fits it: by full-batch conjugate gradients over the whole series
    (``solver="cg"``, the default) or by Adam on a tilt at a time
    (``solver="adam"``, ``specter.ghostbuster._training.fit``). The ``run`` /
    ``test_run`` API mirrors :class:`Ghostbuster`.

    By default the volume is made wide enough to hold every ray that
    reaches the detector (``extend``), the forward model is Rytov sampled on
    each tilt's Ewald surface, and the reconstructed potential is relative
    to the specimen's mean (``dose_per_angstrom=None``). A model that follows
    the defocus through the specimen's depth needs the hand of that depth,
    ``defocus_hand``, which differs between datasets. A tilt series does
    not determine its tomogram on its own; set ``tv_weight`` or
    ``smoothness`` (`TomogramReconstructor.regularisation`), choosing the
    weight by fitting one half of the dose and scoring the other.

    The forward model is noiseless and predicts electron counts per pixel,
    ``dose_per_angstrom * voxel_size² * |CTF(exitwave)|²``, as the particle
    pipeline's does. ``image_units`` says what the tilt series holds: counts
    (the default, and what a motion-corrected tilt series and a simulated
    one, detected by :meth:`~specter.imaging.Image.detect`, hold) are used
    as they are, and a normalised series is mapped back to
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
    dose_per_angstrom : float or torch.Tensor, optional
        Electron dose (fluence) per tilt image in e⁻/Å². Scalar, or a 1-D
        tensor of length ``N_tilts`` for a dose that differs per tilt.
        ``None`` (the default) takes each tilt's mean counts per unit area,
        the dose its specimen transmitted, which makes the reconstructed
        potential relative to the specimen's mean and needs no dose
        calibration; it requires ``image_units="counts"``.
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
        Learning rate for V under ``solver="adam"``, where ``None`` disables
        optimisation. The ``"cg"`` solver sets its own step and ignores it.
    epochs : int
        Training epochs of ``solver="adam"``.  Default 5.
    batchsize : int
        Tilt images per optimisation step of ``solver="adam"``, over every
        device.  Default 1 (one tilt per step).
    propagation : Propagation, optional
        How the forward model computes the exit wave (model, amplitude
        contrast, bandlimit). Default ``Propagation()``.
    optics : Optics, optional
        The aberration engine and phase plate. Default ``Optics()``.
    use_fov_mask : bool
        Restrict the loss to the pixels the forward model can predict: the
        real-FOV region per tilt, less the band at the field's edges that the
        transfer function fills from outside it
        (``TomogramReconstructor._edge_pads``). Default ``True``.
    scheduler : str
        LR scheduler.  Default ``"LambdaLR"``.
    slice_batchsize : int
        Z-slice chunk size for ``IterativeScattering``.  Default 1.
    num_workers : int
        DataLoader worker processes.  Default 0.
    precision : str
        Training precision: ``"16-mixed"`` (default), ``"bf16-mixed"`` or ``"32"``.
    run_dir : str or Path, optional
        Output directory for volumes and metadata.
    image_units : {"counts", "normalized"}
        What ``tilt_series``' values are: electron counts per pixel, used as
        is, or a series normalised to zero mean and unit variance, mapped
        back to counts as ``sqrt(N) * x + N`` with ``N`` each tilt's dose per
        pixel. Default ``"counts"``.
    checkpoint_chunks : int or None
        Split the multislice slice loop into chunks of this many slices and
        run each under gradient checkpointing, trading one extra forward pass
        per chunk for activation memory, see `TomogramReconstructor`.
        ``None`` (the default) disables checkpointing.
    lr_decay : float
        Decay coefficient for the reciprocal-sqrt LR schedule. Default ``0.1``.
    kmask : torch.Tensor, optional
        3-D Fourier-space mask applied to V after every gradient update. Its
        shape must be the reconstructed volume's. `test_run` bins the volume
        and therefore runs without it. Default ``None`` (no mask).
    extend : bool
        Reconstruct a volume wide enough to hold every ray that reaches the
        detector (`TomogramReconstructor.covering_shape`) rather than one as
        wide as the images with the loss masked to what it explains. Applies
        when the forward model is sampled on the surface and neither
        ``V_init`` nor ``kmask``, each of which fixes the volume's shape, is
        given. Default ``True``.
    solver : {"cg", "adam"}
        ``"cg"`` fits every tilt at once by conjugate gradients
        (`solve_tilt_series`), on one device, or in ``strips``, for ``iterations``;
        ``"adam"`` takes ``batchsize`` tilts per step for ``epochs``.
        Default ``"cg"``.
    iterations : int
        Conjugate-gradient iterations. Default 100.
    smoothness, tv_weight, tv_delta : float
        The priors, see `TomogramReconstructor`. Default none.
    tilt_batch : int
        Tilts predicted per forward pass of the ``"cg"`` solver. Default 8.
    defocus_hand : {"negative", "positive"}, optional
        Which way the defocus runs with depth along the beam: the forward
        model's ``Propagation.ews_curvature_sign``, which this replaces when
        given. A point of the specimen at height ``z`` in a tilt's beam
        frame (the third coordinate of ``R^T r``, for a pose ``R`` and a
        volume point ``r``) is imaged at defocus ``df + z`` under
        ``"negative"`` and ``df - z`` under ``"positive"``. It is a
        property of the dataset's geometry, and the wrong hand applies
        the depth correction backwards, which is worse than applying none.
        SPECTER's simulators use ``"negative"``; for a series aligned by
        AreTomo3, `specter.tilt.read_aretomo3_ctf` reads it from the CTF
        file. ``None`` (the default) keeps ``propagation``'s sign, with a
        warning when the model depends on it (every model but
        ``"projection"`` and ``"ctf"``).
    strips : int
        Solve the volume in this many overlapping strips along the tilt
        axis (`solve_tilt_series_in_strips`), each on one device, with the
        volume itself held on the CPU. Requires the tilt axis along volume
        X or Y (`specter.tilt.turn_to_tilt_axis`) and ``solver="cg"``.
        Default 1, the whole volume at once.
    max_voxels : int, optional
        Divide the volume into the fewest strips of at most this many
        voxels instead of naming their number. A 46 GB device holds about
        250 million.

    Notes
    -----
    The default forward model is Rytov rather than multislice. On a
    simulated 180 nm specimen imaged at 300 kV, Rytov images differed from
    multislice ones by at most 0.2% of the signal power at every spatial
    frequency to 5 A, while a tilt of the slice loop costs as much as the
    whole series on the surface.
    """

    def __init__(
        self,
        tilt_series: torch.Tensor | str | Path,
        voxel_size: float,
        voltage: float,
        ctf_params: dict[str, Any],
        dose_per_angstrom: float | torch.Tensor | None = None,
        angles: Sequence[float] | torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        tilt: TiltGeometry = TiltGeometry(),
        nz: int | None = None,
        V_init: torch.Tensor | None = None,
        flip_contrast: bool | None = None,
        lr: float | None = None,
        epochs: int = 5,
        batchsize: int = 1,
        propagation: Propagation = Propagation(scattering_model="rytov"),
        optics: Optics = Optics(),
        use_fov_mask: bool = True,
        scheduler: Scheduler = SCHEDULER,
        slice_batchsize: int = 1,
        num_workers: int = 0,
        precision: str = "16-mixed",
        run_dir: str | Path | None = None,
        image_units: ImageUnits = "counts",
        checkpoint_chunks: int | None = None,
        lr_decay: float = LR_DECAY,
        kmask: torch.Tensor | None = None,
        extend: bool = True,
        solver: Literal["cg", "adam"] = "cg",
        iterations: int = 100,
        smoothness: float = 0.0,
        tv_weight: float = 0.0,
        tv_delta: float = 0.02,
        tilt_batch: int = 8,
        defocus_hand: EwaldSphereSign | None = None,
        strips: int = 1,
        max_voxels: int | None = None,
    ) -> None:
        if solver not in ("cg", "adam"):
            raise ValueError(f"solver={solver!r}; expected 'cg' or 'adam'")
        if strips < 1 or (max_voxels is not None and strips != 1):
            raise ValueError(
                "strips must be at least 1, and is found from max_voxels when "
                "that is given: pass one or the other"
            )
        if (strips > 1 or max_voxels is not None) and solver != "cg":
            raise ValueError("a strip-wise solve needs solver='cg'")
        if defocus_hand is not None:
            if defocus_hand not in ("negative", "positive"):
                raise ValueError(
                    f"defocus_hand={defocus_hand!r}; expected 'negative' or 'positive'"
                )
            propagation = dataclasses.replace(
                propagation, ews_curvature_sign=defocus_hand
            )
        elif propagation.scattering_model not in ("projection", "ctf"):
            warnings.warn(
                f"defocus_hand is not set, and scattering_model="
                f"{propagation.scattering_model!r} follows the defocus through "
                "the specimen's depth: assuming ews_curvature_sign="
                f"{propagation.ews_curvature_sign!r}. The hand is a property of "
                "the dataset; the wrong one is worse than a flat model. Pass "
                "defocus_hand (specter.tilt.read_aretomo3_ctf reads it from an "
                "AreTomo3 CTF file).",
                stacklevel=2,
            )
        images = self._load_tilt_series(tilt_series)
        n_tilts, H, W = images.shape
        if dose_per_angstrom is None:
            if image_units != "counts":
                raise ValueError(
                    "dose_per_angstrom=None reads each tilt's dose off its mean "
                    "counts, so it needs image_units='counts'"
                )
            dose_per_angstrom = images.float().mean(dim=(-2, -1)) / voxel_size**2
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
        depth = nz if nz is not None else W
        shape = (depth, W, W)
        if (
            extend
            and V_init is None
            and kmask is None
            and propagation.rotation() == "surface"
        ):
            shape = TomogramReconstructor.covering_shape(W, depth, quats)
        volume_init = (
            torch.zeros(shape) if V_init is None else torch.as_tensor(V_init).float()
        )
        # Whether the volume is this pipeline's own empty one, which a test
        # run rebuilds for its binned geometry rather than pooling.
        self._covering = V_init is None and shape != (depth, W, W)
        if kmask is not None and tuple(kmask.shape) != tuple(volume_init.shape):
            raise ValueError(
                f"kmask has shape {tuple(kmask.shape)}; it must match the "
                f"reconstructed volume's {tuple(volume_init.shape)}"
            )

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
        self.checkpoint_chunks = checkpoint_chunks
        self.lr_decay = lr_decay
        self.kmask = kmask
        self.solver = solver
        self.iterations = iterations
        self.smoothness = smoothness
        self.tv_weight = tv_weight
        self.tv_delta = tv_delta
        self.tilt_batch = tilt_batch
        self.strips = strips
        self.max_voxels = max_voxels
        self._image_size = W

    @staticmethod
    def _load_tilt_series(tilt_series: torch.Tensor | str | Path) -> torch.Tensor:
        """Load a tilt series from a tensor or an .mrc file path."""
        if isinstance(tilt_series, (str, Path)):
            console.print(f"Loading tilt series from {Path(tilt_series).name} ...")
            images, _ = read_mrc(tilt_series)
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

    def _solve(
        self,
        model: "TomogramReconstructor",
        images: torch.Tensor,
        device: int | Sequence[int] | str,
        iterations: int,
    ) -> "TomogramReconstructor":
        """Fit ``model`` to ``images`` by conjugate gradients and save the result."""
        devices = torch_devices(device)
        log: list[Any]
        if self.strips > 1 or self.max_voxels is not None:
            # The volume stays on the CPU; each strip is solved on a device.
            log = solve_tilt_series_in_strips(
                model,
                images,
                iterations,
                strips=None if self.max_voxels is not None else self.strips,
                max_voxels=self.max_voxels,
                devices=devices,
                tilt_batch=self.tilt_batch,
            )
            for i, entries in enumerate(log):
                if entries:
                    console.print(
                        f"  strip {i + 1}/{len(log)}: chi²/pixel "
                        f"{entries[-1]['chi2_per_pixel']:.4f} after "
                        f"{len(entries)} iterations, {entries[-1]['seconds']:.0f} s"
                    )
        else:
            if len(devices) > 1:
                console.print(
                    "  [dim]solver='cg' solves the whole volume on one device; "
                    "using the first (strips divides it between several)[/dim]"
                )
            model = model.to(devices[0])
            log = solve_tilt_series(
                model, images, iterations, tilt_batch=self.tilt_batch
            )
            if log:
                console.print(
                    f"  chi²/pixel {log[-1]['chi2_per_pixel']:.4f} after "
                    f"{len(log)} iterations, {log[-1]['seconds']:.0f} s"
                )
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "params.json").write_text(
                json.dumps(dict(model.hparams), indent=2, default=str)
            )
            (self.run_dir / "metrics.json").write_text(json.dumps(log, indent=2))
            model._save_volume(
                model.field_volume().cpu().float(), self.run_dir / "volume.mrc"
            )
            console.print(f"  Saved volume → {self.run_dir / 'volume.mrc'}")
        return model

    def _build_reconstructor_and_loader(
        self,
        images: torch.Tensor,
        volume_init: torch.Tensor,
        voxel_size: float,
        batchsize: int,
        kmask: torch.Tensor | None,
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
            use_fov_mask=self.use_fov_mask,
            propagation=self.propagation,
            optics=self.optics,
            scheduler=self.scheduler,
            lr_decay=self.lr_decay,
            kmask=kmask,
            slice_batchsize=self.slice_batchsize,
            checkpoint_chunks=self.checkpoint_chunks,
            run_dir=self.run_dir,
            image_size=images.shape[-1],
            smoothness=self.smoothness,
            tv_weight=self.tv_weight,
            tv_delta=self.tv_delta,
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
            train data-parallel across several GPUs. A single tomogram's
            tilt series is usually small (tens of tilts), so the benefit of
            splitting it across GPUs is more limited than for particle-stack
            reconstruction. Pass ``"cpu"`` to force CPU
            training; any other value uses the GPU when CUDA is
            available and falls back to the CPU when it is not.
        callbacks : list, optional
            Training hooks, as for :meth:`Ghostbuster.run`.

        Returns
        -------
        TomogramReconstructor
            Trained model.  Access the volume via ``model.V.detach()``.
        """
        n_tilts = len(self._images)
        console.print(
            f"Starting reconstruction: {n_tilts} tilts  |  "
            f"volume {'×'.join(str(n) for n in self._volume_init.shape)}  |  "
            f"{self.propagation.scattering_model}  |  "
            + (
                f"{self.iterations} iterations"
                if self.solver == "cg"
                else f"{self.epochs} epochs  |  batch {self.batchsize}"
            )
            + f"  |  {self._device_label(device)}"
        )
        model, loader = self._build_reconstructor_and_loader(
            self._images,
            self._volume_init,
            self._voxel_size,
            self.batchsize,
            self.kmask,
        )
        if self.solver == "cg":
            return self._solve(model, self._images, device, self.iterations)
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
            data-parallel across several GPUs, or ``"cpu"`` to force CPU
            training.
        callbacks : list, optional
            Training hooks, as for :meth:`Ghostbuster.run`.

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

        if self._covering:
            # An empty covering volume is rebuilt for the binned images: its
            # pooled shape need not keep their parity.
            V_b = torch.zeros(
                TomogramReconstructor.covering_shape(
                    images_binned.shape[-1], V_b.shape[0], self._quaternions
                )
            )

        # A kmask is shaped for the full volume and does not survive binning.
        kmask = self.kmask if bin_factor == 1 else None
        if self.kmask is not None and kmask is None:
            console.print(
                "  [dim]kmask skipped: it does not match the binned volume[/dim]"
            )
        model, loader = self._build_reconstructor_and_loader(
            images_binned, V_b, voxel_size_binned, self.batchsize, kmask
        )
        if self.solver == "cg":
            run_dir, self.run_dir = self.run_dir, None  # a test run writes nothing
            try:
                model = self._solve(
                    model, images_binned, device, min(self.iterations, 10)
                )
            finally:
                self.run_dir = run_dir
        else:
            model = self._fit(model, loader, device, 1, "32", callbacks)
        self._report_test_run(
            model,
            f"{bin_factor}× binned, {len(self._images)} tilts, "
            f"volume {tuple(model.V.shape)}",
        )
        return model
