"""
`TomogramReconstructor`: reconstructs a volume from a tilt series with the
same forward model as `imaging.TiltSeries`, one tilt per training step.
"""

from __future__ import annotations

from specter import logger

import json
import math
from pathlib import Path
from typing import Any

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler

from .. import tilt as tilt_geometry
from ..aberrations._functions import defocus_midplane_shift
from ..arrays import center_crop
from ..constants import energy_to_wavelength
from ..imaging import CTF, ExitWave
from ..imaging._tilt import tilt_defocus_shift, tilt_theta
from ..scattering import IterativeScattering
from ._base_reconstructor import _BaseReconstructor
from ..settings import Optics, Propagation, TiltGeometry
from specter.options import Scheduler


class TomogramReconstructor(_BaseReconstructor):
    """
    Differentiable tomogram reconstruction from a cryo-ET tilt series.

    Reconstructs a 3D electrostatic potential volume from tilt-series images
    by minimising the discrepancy between simulated and observed images, using
    the same forward model as :class:`~specter.imaging.TiltSeries`.

    One tilt per training step (``batch_size=1`` in the dataloader) keeps GPU
    memory bounded regardless of tomogram size.  Larger batch sizes accumulate
    the loss over multiple tilts before each parameter update.

    Typically constructed and driven by :class:`TomogramGhostbuster`.

    Parameters
    ----------
    V : torch.Tensor
        Initial volume estimate, shape ``(Z, Y, X)``.
    voxel_size : float
        Voxel size in Å.
    quaternions : torch.Tensor
        Per-tilt rotation quaternions, shape ``(N_tilts, 4)``, xyzw convention.
    translations : torch.Tensor
        Per-tilt in-plane translations in Å, shape ``(N_tilts, 2)``.
    ctf_params : dict[str, torch.Tensor]
        Per-tilt CTF parameters; each value has leading dimension ``N_tilts``.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose (fluence) per tilt image in e⁻/Å². Scalar, or a 1-D
        tensor of length ``N_tilts`` giving a separate dose for each tilt, as
        `Exposure.tilt_series` holds it. The forward model predicts electron
        counts per pixel, ``dose_per_angstrom * voxel_size**2 * |CTF(psi)|²``,
        and the loss divides each tilt's squared residual by that tilt's
        expected counts per pixel, its Poisson variance.
    tilt : TiltGeometry, optional
        Tilt axis and edge tapers -- the same object the `TiltSeries` that
        produced the data was built with. The axis decides which image
        dimension shrinks at high tilt for FOV masking; the tapers are applied
        to V before each forward pass. Default ``TiltGeometry()``.
    lr : float, optional
        Learning rate for volume V.  ``None`` freezes V.
    use_fov_mask : bool
        If ``True``, the loss at each tilt is restricted to the pixels the
        forward model can predict: the central strip corresponding to real
        (non-padded) volume at high tilt, less a band at every edge as wide
        as the transfer function delocalises the image (:meth:`_edge_pads`).
        Within that band the data carry whatever surrounds the field, which
        the volume does not contain. Default ``True``.
    propagation : Propagation, optional
        How the forward model computes the exit wave (model, amplitude
        contrast, bandlimit). The same object the `TiltSeries` that produced
        the data was scattered with, when it was simulated. Default
        ``Propagation()``.
    optics : Optics, optional
        The aberration engine and phase plate. Default ``Optics()``. No
        `Envelopes` or `Camera`: the inverse does not model what the data
        cannot determine.
    scheduler : str
        LR scheduler for the volume optimiser.  Default ``"LambdaLR"``.
    lr_decay : float
        Decay coefficient for the reciprocal-sqrt LR schedule.  Default ``0.1``.
    kmask : torch.Tensor, optional
        3-D Fourier-space mask applied to V after every gradient update.
    slice_batchsize : int
        Z-slice chunk size passed to ``IterativeScattering``.  Reduce for large
        volumes to stay within GPU memory.  Default ``1``.
    checkpoint_chunks : int or None
        If set, the multislice slice loop is split into chunks of this size
        and run under gradient checkpointing.  Only the wave function at chunk
        boundaries is retained; intermediates are recomputed on the backward
        pass.  This reduces activation memory from ``O(nz_tilt)`` (tens of GB
        at high tilt) to ``O(checkpoint_chunks × nxy²)`` at the cost of one
        extra forward pass per chunk.  Suggested starting value: ``50``.
        ``None`` disables checkpointing.
    run_dir : str or Path, optional
        Directory for per-epoch MRC volumes and training metrics.  ``None``
        disables all file output.
    """

    def __init__(
        self,
        V: torch.Tensor,
        voxel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, torch.Tensor],
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        tilt: TiltGeometry = TiltGeometry(),
        lr: float | None = None,
        use_fov_mask: bool = True,
        propagation: Propagation = Propagation(),
        optics: Optics = Optics(),
        scheduler: Scheduler = "LambdaLR",
        lr_decay: float = 0.1,
        kmask: torch.Tensor | None = None,
        slice_batchsize: int = 1,
        checkpoint_chunks: int | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if propagation.absorption_model == "inelastic_mfp":
            raise ValueError(
                "TomogramReconstructor does not support absorption_model='inelastic_mfp': "
                "its iterative volume path requires explicit material/solvent "
                "information to construct the absorption field. Use 'alpha'."
            )
        scattering_model = propagation.scattering_model
        self.save_hyperparameters(
            ignore=["V", "quaternions", "translations", "ctf_params", "kmask"]
        )
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self.automatic_optimization = False

        # Geometry
        self.nxy = V.shape[-1]
        self.nz = V.shape[0]
        self.voxel_size = voxel_size
        self.voltage = voltage
        self.tilt = tilt
        self.tilt_axis = tilt.tilt_axis

        # Optimisation settings
        self.lr = lr
        self.lr_decay = lr_decay
        self.scheduler = scheduler
        self.taper_width = tilt.taper_width
        self.z_taper_width = tilt.z_taper_width
        self.use_fov_mask = use_fov_mask
        self.slice_batchsize = slice_batchsize
        self.checkpoint_chunks = checkpoint_chunks

        self._init_loss_logs()

        # Volume parameter
        if lr is None:
            self.register_buffer("V", V.float())
        else:
            self.V = nn.Parameter(V.float())

        # Expected electrons per pixel for each tilt, the scale of the
        # predicted counts and the Poisson variance the loss divides by.
        n_tilts = len(quaternions)
        dose = torch.as_tensor(dose_per_angstrom, dtype=torch.float32).flatten()
        if dose.numel() not in (1, n_tilts):
            raise ValueError(
                f"dose_per_angstrom has {dose.numel()} entries; expected 1 or "
                f"one per tilt ({n_tilts})"
            )
        if not bool((dose > 0).all()):
            raise ValueError("dose_per_angstrom must be positive")
        self.register_buffer(
            "dose_per_pixel", dose.expand(n_tilts).clone() * voxel_size**2
        )

        # Tilt geometry buffers (fixed — orientations are known in cryo-ET)
        self.register_buffer(
            "quaternions", torch.as_tensor(quaternions, dtype=torch.float32)
        )
        self.register_buffer(
            "translations", torch.as_tensor(translations, dtype=torch.float32)
        )

        # Per-tilt CTF params as named buffers
        self.ctf_params: dict[str, torch.Tensor] = {}
        for k, v in ctf_params.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
            self.ctf_params[k] = getattr(self, k)

        # Multislice returns the wave at the volume's exit face, while dfu/dfv
        # are the specimen midplane's defocus, the plane a CTF fit reports.
        # The simulators' chain applies the offset through
        # `ExitWave.defocus_shift`, which `_forward_tilt` builds the same way
        # `TiltSeries.scatter` does (`imaging._tilt.tilt_defocus_shift`), so
        # the two cannot drift apart. Recorded for the parity test.
        self._defocus_shift_angstrom = (
            0.0
            if scattering_model in ("projection", "ctf")
            else defocus_midplane_shift(self.nz, voxel_size)
        )

        # Fourier-space mask applied after each gradient step
        self.register_buffer("kmask", kmask)

        # Physics modules
        self.propagation = propagation
        self.optics = optics
        self.iterative_scattering = IterativeScattering(
            self.nxy,
            voxel_size,
            voltage,
            scattering_model=scattering_model,
            klim=propagation.klim,
            oversample=propagation.oversample,
            crop=False,
            alpha=propagation.alpha,
            # The same canvas `TiltSeries` propagates on, kept through the
            # transfer function as it keeps it, so the two agree.
            pad_fft=propagation.pads(True),
            keep_pad=True,
        )
        # The image grid: the transfer function is applied and the intensity
        # formed on the propagation grid, then sampled to the pixel grid, as
        # in every simulator.
        self._image_oversample = (
            propagation.oversample if scattering_model != "ctf" else 1
        )

    # ------------------------------------------------------------------ #
    # Forward helpers                                                      #
    # ------------------------------------------------------------------ #

    def _prepare_volume(self) -> torch.Tensor:
        """Apply cosine taper to V for the current tilt pass.

        No XY padding is applied. At the maximum tilt angle the projected
        volume extent (nxy·cos θ + nz·sin θ) is smaller than nxy, so all
        volume content fits within the existing grid. Pixels sampled outside
        the volume boundary by grid_sample return zero (vacuum), which is
        physically correct. The real-FOV loss mask (use_fov_mask) already
        discards gradient contributions from the zero-padded periphery.
        """
        V: torch.Tensor = self.V
        if self.taper_width > 0 or self.z_taper_width > 0:
            V = tilt_geometry.apply_volume_cosine_taper(
                V, taper_xy=self.taper_width, taper_z=self.z_taper_width
            )
        return V

    def _loss_mask(self, tilt_idx: int) -> torch.Tensor | None:
        """
        The pixels of tilt `tilt_idx` the loss compares, or ``None`` for all.

        The real-FOV strip (:meth:`_fov_mask`) less the delocalisation band
        at every edge (:meth:`_edge_pads`).
        """
        mask = self._fov_mask(tilt_idx)
        edge = self._edge_pads()[tilt_idx]
        if edge == 0:
            return mask
        if mask is None:
            mask = torch.ones(
                self.nxy, self.nxy, device=self.device, dtype=torch.float32
            )
        mask[:edge, :] = 0.0
        mask[self.nxy - edge :, :] = 0.0
        mask[:, :edge] = 0.0
        mask[:, self.nxy - edge :] = 0.0
        return mask

    def _edge_pads(self) -> list[int]:
        """
        Per-tilt width, in pixels, of the band at the field's edges that the
        transfer function fills from outside the field.

        The lens displaces the image of each spatial frequency ``k`` by
        ``|dchi/dk| / (2 pi) = lambda k |Df - Cs lambda^2 k^2|``, the
        delocalisation of an aberrated image. Out to the pixel Nyquist, and
        over every plane of the tilted column (the nominal defocus, the
        midplane's, plus or minus half the column's depth, and both
        astigmatic axes), the largest displacement bounds how far the
        specimen's surroundings reach into the field. The forward model
        cannot predict those pixels, since its volume ends at the field, so
        the loss leaves them out: fitting them anyway makes the edge voxels
        explain what lies outside the volume, and the interior degrades the
        longer the fit runs.

        A band wider than a quarter of the field per side leaves no pixel the
        surroundings cannot reach, so excluding it only discards data; such a
        tilt gets no band (0), and the log says so.

        Computed once per device and cached, as :meth:`_fov_pads` is.
        """
        key = self.quaternions.device
        cached = getattr(self, "_edge_pads_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        wavelength = energy_to_wavelength(self.voltage)
        k = torch.linspace(0.0, 0.5 / self.voxel_size, 1025, dtype=torch.float64)
        params = {name: getattr(self, name).detach().cpu() for name in self.ctf_params}
        n_tilts = len(self.quaternions)
        zero = torch.zeros(n_tilts)
        dfu = params.get("dfu", zero)
        dfv = params.get("dfv", dfu)
        cs = params.get("cs", zero)
        model = self.propagation.scattering_model
        cap = self.nxy // 4
        pads: list[int] = []
        too_wide = []
        for i in range(n_tilts):
            theta = tilt_theta(
                self.quaternions[i : i + 1].detach().cpu(),
                self.translations[i : i + 1].detach().cpu(),
                self.nxy,
                self.voxel_size,
            )
            half_depth = abs(
                tilt_defocus_shift(
                    (1, self.nz, self.nxy, self.nxy), theta, self.voxel_size, model
                )
            )
            low = float(min(dfu[i], dfv[i])) - half_depth
            high = float(max(dfu[i], dfv[i])) + half_depth
            spherical = float(cs[i]) * wavelength**2 * k**2
            reach = max(
                float((wavelength * k * (defocus - spherical).abs()).max())
                for defocus in (low, high)
            )
            pad = math.ceil(reach / self.voxel_size)
            if pad > cap:
                too_wide.append(i)
                pad = 0
            pads.append(pad)
        if too_wide:
            logger.warning(
                "the transfer function delocalises the image of tilts %s "
                "further than a quarter of the %d px field, so no part of it "
                "is free of the specimen's surroundings; those tilts are "
                "compared without an edge band",
                too_wide,
                self.nxy,
            )
        self._edge_pads_cache = (key, pads)
        return pads

    def _fov_mask(self, tilt_idx: int) -> torch.Tensor | None:
        """
        Binary mask for the real-FOV region at the given tilt.

        Returns ``None`` for near-zero tilts where the full image is real FOV.

        The shrinking dimension is X for ``tilt_axis="y"`` and Y for
        ``tilt_axis="x"``.
        """
        pad = self._fov_pads()[tilt_idx]
        if pad is None:
            return None

        mask = torch.ones(self.nxy, self.nxy, device=self.device, dtype=torch.float32)
        if self.tilt_axis == "x":
            # rotation around X → Y shrinks
            mask[:pad, :] = 0.0
            mask[self.nxy - pad :, :] = 0.0
        else:
            # rotation around Y → X shrinks
            mask[:, :pad] = 0.0
            mask[:, self.nxy - pad :] = 0.0
        return mask

    def _fov_pads(self) -> list[int | None]:
        """
        Per-tilt border width of the real-FOV mask, ``None`` where there is none.

        The tilt poses are fixed buffers, so this is computed once, on first
        use, rather than read back from the device with an ``.item()`` sync on
        every training step. Keyed on the poses' device, since the widths are
        evaluated there and the trigonometry may round differently on another.
        The mask itself is rebuilt per call from the cached width: it is an
        asynchronous fill, and caching one ``nxy**2`` mask per tilt would grow
        device memory with the tilt count.
        """
        key = self.quaternions.device
        cached = getattr(self, "_fov_pads_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        pads: list[int | None] = []
        for Q in self.quaternions:
            rotvec = roma.unitquat_to_rotvec(Q.unsqueeze(0))[0]
            theta = rotvec.norm()  # radians (scalar tensor)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            real_fov = int((self.nxy * cos_t - self.nz * sin_t).clamp(min=1).item())
            pads.append(None if real_fov >= self.nxy else (self.nxy - real_fov) // 2)
        self._fov_pads_cache = (key, pads)
        return pads

    def _forward_tilt(self, V_prepared: torch.Tensor, tilt_idx: int) -> torch.Tensor:
        """
        Simulate one noiseless tilt image.

        Parameters
        ----------
        V_prepared : torch.Tensor
            Tapered volume from ``_prepare_volume``, shape ``(Z, Y, X)``.
        tilt_idx : int
            Index into ``self.quaternions`` / ``self.translations``.

        Returns
        -------
        torch.Tensor
            Noiseless expected electron counts per pixel, shape ``(H, W)``:
            the intensity a `TiltSeries` forms for the tilt without noise
            (``ExitWave.intensity(cc=None).pixels()``), times the tilt's dose
            per pixel.
        """
        theta_matrix = tilt_theta(
            self.quaternions[tilt_idx : tilt_idx + 1],
            self.translations[tilt_idx : tilt_idx + 1],
            V_prepared.shape[-1],
            self.voxel_size,
        )
        V_batched = V_prepared.unsqueeze(0)  # (1, Z, Y', X')
        exitwave = self.iterative_scattering(
            V_batched,
            theta_matrix,
            slice_batchsize=self.slice_batchsize,
            checkpoint_chunks=self.checkpoint_chunks,
        )
        model = self.propagation.scattering_model
        waves = ExitWave(
            array=exitwave,
            pixel_size=self.voxel_size,
            voltage=self.voltage,
            nxy=self.nxy,
            pad_nxy=self.iterative_scattering.output_nxy,
            oversample=self._image_oversample,
            defocus_shift=tilt_defocus_shift(
                tuple(V_batched.shape), theta_matrix, self.voxel_size, model
            ),
            ids=torch.zeros(1, dtype=torch.long),
            tilt=tilt_idx,
            linear=model == "ctf",
            alpha=self.propagation.alpha,
        )
        # getattr, not the dict built at __init__: Lightning replaces buffer
        # tensors on .to(device), and the dict would keep the old ones.
        ctf = CTF(
            {k: getattr(self, k)[tilt_idx : tilt_idx + 1] for k in self.ctf_params}
        )
        image = waves.apply_ctf(
            ctf,
            backend=self.optics.aberration_backend,
            lpp_params=self.optics.lpp_params,
        ).intensity(cc=None)
        field = center_crop(image.pixels()[0], self.nxy, dim=(-2, -1))
        return self.dose_per_pixel[tilt_idx] * field

    def forward(self, tilt_idx: int) -> torch.Tensor:
        """Simulate tilt image ``tilt_idx`` using the current volume."""
        return self._forward_tilt(self._prepare_volume(), tilt_idx)

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
    # ------------------------------------------------------------------ #

    def _compute_loss(
        self,
        sim: torch.Tensor,
        obs: torch.Tensor,
        tilt_idx: int,
    ) -> torch.Tensor:
        """
        Squared residual per pixel over the tilt's Poisson variance, over the
        pixels the forward model can predict (:meth:`_loss_mask`) when
        ``use_fov_mask``.

        Dividing by the expected counts per pixel makes the loss a per-pixel
        chi-squared under Poisson statistics. Shot noise then contributes
        about one per pixel at any dose, while a real misfit weighs in
        proportion to the dose, as the evidence for it does. Without the
        division both would scale with the square of the dose.
        """
        mse = F.mse_loss(sim, obs, reduction="none") / self.dose_per_pixel[tilt_idx]
        if self.use_fov_mask:
            mask = self._loss_mask(tilt_idx)
            if mask is not None:
                # Normalise by the number of valid pixels so that all tilt
                # angles contribute equal gradient signal regardless of FOV
                # size (which shrinks by cos(θ) at high tilt).
                return (mse * mask).sum() / mask.sum()
        return mse.mean()

    # ------------------------------------------------------------------ #
    # Optimisation                                                         #
    # ------------------------------------------------------------------ #

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        """Build AdamW optimiser and LR scheduler for V."""
        return self._volume_optimizer()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Run one manual optimisation step over a batch of tilt images.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            ``(tilt_images, tilt_indices)`` from the dataloader, shapes
            ``(B, H, W)`` and ``(B,)``.
        batch_idx : int
            Batch index (unused, required by Lightning).

        Returns
        -------
        torch.Tensor
            Scalar loss for the current batch.
        """
        obs_images, tilt_indices = batch

        # Prepare V once per step (the taper is a deterministic transform)
        V_prepared = self._prepare_volume()

        total_norm_loss = torch.tensor(0.0, device=self.device)
        # One host transfer for the batch's tilt indices, not a sync per tilt.
        for i, idx in enumerate(tilt_indices.tolist()):
            sim = self._forward_tilt(V_prepared, idx)
            norm_loss = self._compute_loss(sim, obs_images[i], idx)
            total_norm_loss = total_norm_loss + norm_loss

        norm_loss = total_norm_loss / max(len(tilt_indices), 1)
        loss = norm_loss

        # Gathered/averaged across ranks under multi-GPU (DDP) for logging
        # only -- `loss` itself (fed to manual_backward below) stays the
        # local, per-rank value; DDP synchronises gradients on its own.
        # Kept on the device, as in `Reconstructor`: a `.cpu()` here is a
        # host sync per step. The lists are read once, when the metrics are
        # built.
        self.log_norm_loss.append(self._gather_for_logging(norm_loss).detach())
        self.log_total_loss.append(self._gather_for_logging(loss).detach())

        self._optimise(loss)
        return loss

    # ------------------------------------------------------------------ #
    # Epoch / fit callbacks                                                #
    # ------------------------------------------------------------------ #

    def on_fit_start(self) -> None:
        """Create run directory and write hyperparameter metadata.

        Rank-0-only under multi-GPU (DDP): every replica would otherwise
        race to create/write the same files.
        """
        if not self._make_run_dir():
            return
        assert self._run_dir is not None

        if self.kmask is not None:
            torch.save(self.kmask.detach().cpu(), self._run_dir / "kmask.pt")

        meta: dict[str, Any] = dict(self.hparams)
        (self._run_dir / "params.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        logger.info(f"Run directory: {self._run_dir}")

    def on_train_epoch_end(self) -> None:
        """Save per-epoch volume as MRC.

        Rank-0-only under multi-GPU (DDP): ``self.V`` is already identical
        across replicas at this point (kept in sync by DDP's gradient
        all-reduce every step), so every rank would otherwise race to write
        the same file.
        """
        if self._run_dir is None or not self.trainer.is_global_zero:
            return
        epoch = self.current_epoch + 1
        self._save_volume(
            self._volume_cpu(), self._run_dir / "epochs" / f"{epoch:03d}.mrc"
        )

    def on_fit_end(self) -> None:
        """Save final reconstructed volume and training metrics.

        Rank-0-only under multi-GPU (DDP), same reasoning as
        ``on_train_epoch_end``.
        """
        self._save_metrics()
        if self._run_dir is None or not self.trainer.is_global_zero:
            return
        volume_path = self._run_dir / "volume.mrc"
        self._save_volume(self._volume_cpu(), volume_path)
        logger.info(f"Saved final volume → {volume_path}")
