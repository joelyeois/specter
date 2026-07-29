from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import lightning as L
import mrcfile
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .. import rotations
from ..aberrations import Aberration
from ..imagegenerator._tiltseries import TiltSeriesGenerator
from ..scattering import IterativeScattering
from ._helpers import (
    _apply_kmask_inplace,
    _build_epoch_metrics,
    _build_lr_scheduler,
    _log_current_lr,
)


class TomogramReconstructor(L.LightningModule):
    """
    Differentiable tomogram reconstruction from a cryo-ET tilt series.

    Reconstructs a 3D electrostatic potential volume from tilt-series images
    by minimising the discrepancy between simulated and observed images, using
    the same forward model as :class:`~specter.imagegenerator.TiltSeriesGenerator`.

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
    energy : float
        Electron beam energy in keV.
    tilt_axis : str
        Axis around which the sample tilts (``"x"`` or ``"y"``).  Determines
        which image dimension shrinks at high tilt for FOV masking.  Default ``"x"``.
    lr : float, optional
        Learning rate for volume V.  ``None`` freezes V.
    sparsity : float, optional
        L1 regularisation weight on V.
    taper_width : int
        Cosine taper applied to V in XY before each forward pass, smoothing
        the volume-to-vacuum boundary to suppress edge diffraction.  Default 0.
    z_taper_width : int
        Cosine taper in Z.  Default 0.
    use_fov_mask : bool
        If ``True``, the MSE loss at each tilt is restricted to the central
        strip corresponding to real (non-padded) volume, eliminating gradient
        contributions from reflected boundary content.  Default ``True``.
    scattering_model : str
        Wave propagation model.  Default ``"multislice"``.
    aberration_model : str
        CTF aberration model.  Default ``"holography"``.
    klim : float, optional
        Hard reciprocal-space frequency cutoff.
    alpha : float
        Amplitude contrast ratio.  Default ``0.0``.
    scheduler : str
        LR scheduler for the volume optimiser.  Default ``"LambdaLR"``.
    lr_decay : float
        Decay coefficient for the reciprocal-sqrt LR schedule.  Default ``0.1``.
    kmask : torch.Tensor, optional
        3-D Fourier-space mask applied to V after every gradient update.
    slice_batch_size : int
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
        energy: float,
        tilt_axis: str = "x",
        lr: float | None = None,
        sparsity: float | None = None,
        taper_width: int = 0,
        z_taper_width: int = 0,
        use_fov_mask: bool = True,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        klim: float | None = None,
        alpha: float = 0.0,
        scheduler: Literal[
            "LambdaLR",
            "OneCycleLR",
            "CosineAnnealingWarmRestarts",
            "MultiplicativeLR",
        ] = "LambdaLR",
        lr_decay: float = 0.1,
        kmask: torch.Tensor | None = None,
        slice_batch_size: int = 1,
        checkpoint_chunks: int | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["V", "quaternions", "translations", "ctf_params", "kmask"]
        )
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self.automatic_optimization = False

        # Geometry
        self.nxy = V.shape[-1]
        self.nz = V.shape[0]
        self.voxel_size = voxel_size
        self.tilt_axis = tilt_axis.lower()

        # Optimisation settings
        self.lr = lr
        self.lr_decay = lr_decay
        self.scheduler = scheduler
        self.sparsity = sparsity
        self.taper_width = taper_width
        self.z_taper_width = z_taper_width
        self.use_fov_mask = use_fov_mask
        self.slice_batch_size = slice_batch_size
        self.checkpoint_chunks = checkpoint_chunks

        # Logging
        self.log_total_loss: list[torch.Tensor] = []
        self.log_norm_loss: list[torch.Tensor] = []
        self.log_sparsity_loss: list[torch.Tensor] = []
        self.log_lrs: list[float] = []

        # Volume parameter
        if lr is None:
            self.register_buffer("V", V.float())
        else:
            self.V = nn.Parameter(V.float())

        # Informational: minimum XY size a forward model would need for this tilt range.
        # TomogramReconstructor does NOT pre-pad to this size; it works at nxy throughout.
        max_tilt_deg = TiltSeriesGenerator._infer_max_tilt_from_inputs(
            angles=None, quaternions=quaternions
        )
        self.required_nxy = int(
            TiltSeriesGenerator._estimate_required_nxy(self.nxy, self.nz, max_tilt_deg)
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

        # Fourier-space mask applied after each gradient step
        self.register_buffer("kmask", kmask)

        # Physics modules
        self.iterative_scattering = IterativeScattering(
            self.nxy,
            voxel_size,
            energy,
            scattering_model=scattering_model,
            klim=klim,
            alpha=alpha,
        )
        self.aberration = Aberration(
            self.nxy,
            voxel_size,
            energy,
            aberration_model=aberration_model,
            alpha=alpha if aberration_model == "ctf" else None,
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
            V = TiltSeriesGenerator._apply_cosine_taper(
                V, taper_xy=self.taper_width, taper_z=self.z_taper_width
            )
        return V

    @staticmethod
    def _compute_nz_tilt(V_shape: tuple[int, ...], theta_matrix: torch.Tensor) -> int:
        """Z-slice count needed to propagate through the rotated volume."""
        _, Z, Y, X = V_shape
        R = theta_matrix[:, :3, :3]
        corners = torch.tensor(
            [
                [-X / 2, -Y / 2, -Z / 2],
                [X / 2, -Y / 2, -Z / 2],
                [-X / 2, Y / 2, -Z / 2],
                [X / 2, Y / 2, -Z / 2],
                [-X / 2, -Y / 2, Z / 2],
                [X / 2, -Y / 2, Z / 2],
                [-X / 2, Y / 2, Z / 2],
                [X / 2, Y / 2, Z / 2],
            ],
            device=theta_matrix.device,
            dtype=theta_matrix.dtype,
        ).t()  # (3, 8)
        rotated = torch.bmm(
            R.transpose(1, 2), corners.unsqueeze(0).expand(R.shape[0], -1, -1)
        )
        z_min = rotated[:, 2, :].min(dim=1).values
        z_max = rotated[:, 2, :].max(dim=1).values
        return max(1, int(torch.ceil((z_max - z_min).max()).item()))

    def _fov_mask(self, tilt_idx: int) -> torch.Tensor | None:
        """
        Binary mask for the real-FOV region at the given tilt.

        Returns ``None`` for near-zero tilts where the full image is real FOV.

        The shrinking dimension is X for ``tilt_axis="y"`` and Y for
        ``tilt_axis="x"``.
        """
        Q = self.quaternions[tilt_idx]
        rotvec = roma.unitquat_to_rotvec(Q.unsqueeze(0))[0]
        theta = rotvec.norm()  # radians (scalar tensor)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        real_fov = int((self.nxy * cos_t - self.nz * sin_t).clamp(min=1).item())
        if real_fov >= self.nxy:
            return None

        pad = (self.nxy - real_fov) // 2
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

    def _forward_tilt(self, V_prepared: torch.Tensor, tilt_idx: int) -> torch.Tensor:
        """
        Simulate one noiseless tilt image.

        Parameters
        ----------
        V_prepared : torch.Tensor
            Tapered and padded volume, shape ``(Z, Y', X')``.
        tilt_idx : int
            Index into ``self.quaternions`` / ``self.translations``.

        Returns
        -------
        torch.Tensor
            Clean intensity image, shape ``(H, W)``, same units as
            ``TiltSeriesGenerator.generate_tilt_series`` ``clean_images``.
        """
        Q = self.quaternions[tilt_idx : tilt_idx + 1]  # (1, 4)
        T = self.translations[tilt_idx : tilt_idx + 1]  # (1, 2)
        R_mat = roma.unitquat_to_rotmat(Q)
        if R_mat.ndim == 2:
            R_mat = R_mat.unsqueeze(0)
        T_torch = rotations.translations_angstrom_to_torch(
            T, V_prepared.shape[-1], self.voxel_size
        )
        theta_matrix = rotations.build_affine_matrix(R_mat, T_torch)

        V_batched = V_prepared.unsqueeze(0)  # (1, Z, Y', X')
        exitwave = self.iterative_scattering(
            V_batched,
            theta_matrix,
            slice_batch_size=self.slice_batch_size,
            checkpoint_chunks=self.checkpoint_chunks,
        )

        # Defocus correction: multislice propagates nz_new slices instead of
        # self.nz, shifting the effective specimen centre by z_offset.
        # Use getattr rather than self.ctf_params[k] so we always get the
        # live buffer (Lightning replaces buffer tensors on .to(device) but
        # dict entries stored at __init__ time would point to the old CPU tensor).
        ctf_batch = {
            k: getattr(self, k)[tilt_idx : tilt_idx + 1] for k in self.ctf_params
        }
        if self.iterative_scattering.scattering_model not in ("projection", "ctf"):
            nz_new = self._compute_nz_tilt(tuple(V_batched.shape), theta_matrix)
            z_offset = (nz_new - self.nz) * self.voxel_size / 2.0
            if "dfu" in ctf_batch:
                ctf_batch["dfu"] = ctf_batch["dfu"] - z_offset
            if "dfv" in ctf_batch:
                ctf_batch["dfv"] = ctf_batch["dfv"] - z_offset

        detector_waves = self.aberration(exitwave, ctf_batch)
        return torch.abs(detector_waves[0]) ** 2  # (H, W)

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
        """MSE loss with optional FOV masking and sparsity regularisation."""
        mse = F.mse_loss(sim, obs, reduction="none")
        if self.use_fov_mask:
            mask = self._fov_mask(tilt_idx)
            if mask is not None:
                # Normalise by the number of valid pixels so that all tilt
                # angles contribute equal gradient signal regardless of FOV
                # size (which shrinks by cos(θ) at high tilt).
                return (mse * mask).sum() / mask.sum()
        return mse.mean()

    # ------------------------------------------------------------------ #
    # Optimisation                                                         #
    # ------------------------------------------------------------------ #

    def reciprocal_lr_scheduler(self, *args: Any) -> float:
        """Reciprocal-square-root decay: ``1 / (1 + decay * step^0.5)``."""
        return 1 / (1 + self.lr_decay * self.global_step**0.5)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        """Build AdamW optimiser and LR scheduler for V."""
        if self.lr is None:
            return [], []

        optimizerV = AdamW([self.V], lr=self.lr, weight_decay=0.0)
        lr_scheduler = _build_lr_scheduler(
            self.scheduler,
            optimizerV,
            self.lr,
            self.reciprocal_lr_scheduler,
            self.num_training_steps_per_epoch,
            self.num_training_steps,
        )
        return [optimizerV], [lr_scheduler]

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

        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        for opt in opts:
            opt.zero_grad()

        # Prepare V once per step (taper + padding are deterministic transforms)
        V_prepared = self._prepare_volume()

        total_norm_loss = torch.tensor(0.0, device=self.device)
        for i in range(len(tilt_indices)):
            idx = int(tilt_indices[i].item())
            sim = self._forward_tilt(V_prepared, idx)
            norm_loss = self._compute_loss(sim, obs_images[i], idx)
            total_norm_loss = total_norm_loss + norm_loss

        norm_loss = total_norm_loss / max(len(tilt_indices), 1)
        loss = norm_loss

        if self.sparsity is not None:
            sparsity_loss = self.sparsity * torch.mean(torch.abs(self.V))
            loss = loss + sparsity_loss
            self.log_sparsity_loss.append(sparsity_loss.detach().cpu())

        self.log_norm_loss.append(norm_loss.detach().cpu())
        self.log_total_loss.append(loss.detach().cpu())

        self.manual_backward(loss)
        for opt in opts:
            opt.step()

        if not self.automatic_optimization:
            sch = self.lr_schedulers()
            if sch:
                if not isinstance(sch, (list, tuple)):
                    sch = [sch]
                for s in sch:
                    if isinstance(s, ReduceLROnPlateau):
                        raise TypeError(
                            "ReduceLROnPlateau is not supported by this manual "
                            "scheduler step loop; _build_lr_scheduler never "
                            "constructs one."
                        )
                    s.step()

        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Record the learning rate before each batch."""
        _log_current_lr(self.trainer, self.lr, self.log_lrs)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Apply the Fourier-space k-mask to V after each gradient update."""
        _apply_kmask_inplace(self.V, self.kmask)

    # ------------------------------------------------------------------ #
    # Epoch / fit callbacks                                                #
    # ------------------------------------------------------------------ #

    def on_fit_start(self) -> None:
        """Create run directory and write hyperparameter metadata."""
        if self._run_dir is None:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "epochs").mkdir(exist_ok=True)

        if self.kmask is not None:
            torch.save(self.kmask.detach().cpu(), self._run_dir / "kmask.pt")

        meta: dict[str, Any] = dict(self.hparams)
        (self._run_dir / "params.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        print(f"Run directory: {self._run_dir}")

    def on_train_epoch_end(self) -> None:
        """Save per-epoch volume as MRC."""
        if self._run_dir is None:
            return
        epoch = self.current_epoch + 1
        v = self.V.detach().cpu().float()
        mrc_path = self._run_dir / "epochs" / f"{epoch:03d}.mrc"
        with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
            mrc.set_data(v.numpy())
            mrc.voxel_size = self.voxel_size

    def on_fit_end(self) -> None:
        """Save final reconstructed volume and training metrics."""
        self._save_metrics()
        if self._run_dir is None:
            return
        v = self.V.detach().cpu().float()
        vol_path = self._run_dir / "vol.mrc"
        with mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(v.numpy())
            mrc.voxel_size = self.voxel_size
        print(f"Saved final volume → {vol_path}")

    def _save_metrics(self) -> None:
        """Save training metrics (loss, lr) to JSON."""
        if self._run_dir is None or not self.log_total_loss:
            return

        metrics_path = self._run_dir / "metrics.json"
        meta = _build_epoch_metrics(
            self.log_total_loss,
            self.log_norm_loss,
            self.log_sparsity_loss,
            self.log_lrs,
            self.current_epoch,
        )
        metrics_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metrics → {metrics_path}")

    # ------------------------------------------------------------------ #
    # Step-count helpers (identical to Reconstructor)                      #
    # ------------------------------------------------------------------ #

    def num_training_steps_per_epoch(self) -> int:
        """Number of optimiser steps per epoch."""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        self.trainer.fit_loop.setup_data()
        assert self.trainer.train_dataloader is not None
        dataset_size = len(self.trainer.train_dataloader)
        return dataset_size // self.trainer.accumulate_grad_batches

    def num_training_steps(self) -> int:
        """Total optimiser steps across the full run."""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs < 1:
            raise ValueError("OneCycleLR requires a positive trainer.max_epochs.")
        return self.num_training_steps_per_epoch() * max_epochs
