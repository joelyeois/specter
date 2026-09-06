from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler

from .. import rotations
from .. import tilt as tilt_geometry
from ..aberrations import (
    Aberration,
    aberration_model_for_scattering,
    defocus_midplane_shift,
)
from ..ctf import LegacyAberrationAdapter
from ..scattering import IterativeScattering
from ._base_reconstructor import _BaseReconstructor
from ._helpers import _build_lr_scheduler
from ._io import save_volume_mrc
from specter.options import ScatteringModel, Scheduler, TiltAxis


class TomogramReconstructor(_BaseReconstructor):
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
    voltage : float
        Electron beam accelerating voltage in kV.
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
    aberration_backend : {"legacy", "torch_ctf"}, optional
        Which engine computes the CTF/aberration transfer function.
        ``"legacy"`` (default) uses ``aberrations.Aberration``; ``"torch_ctf"``
        uses ``ctf.LegacyAberrationAdapter`` (verified parity, see
        ``ImageGenerator``'s docstring). Opt-in only; not yet the default.
    lpp_params : dict[str, float], optional
        Laser-phase-plate config, in ``ctf.CTFParameters``-native units.
        Requires ``aberration_backend="torch_ctf"``; raises at construction
        time otherwise, since ``aberrations.Aberration`` has no LPP model.
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
        tilt_axis: TiltAxis = "x",
        lr: float | None = None,
        sparsity: float | None = None,
        taper_width: int = 0,
        z_taper_width: int = 0,
        use_fov_mask: bool = True,
        scattering_model: ScatteringModel = "multislice",
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
        klim: float | None = None,
        alpha: float = 0.0,
        scheduler: Scheduler = "LambdaLR",
        lr_decay: float = 0.1,
        kmask: torch.Tensor | None = None,
        slice_batchsize: int = 1,
        checkpoint_chunks: int | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if lpp_params is not None and aberration_backend != "torch_ctf":
            raise ValueError(
                "lpp_params requires aberration_backend='torch_ctf' -- "
                "aberrations.Aberration has no laser-phase-plate model."
            )
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
        self.slice_batchsize = slice_batchsize
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
        max_tilt_deg = tilt_geometry.infer_max_tilt_from_inputs(
            angles=None, quaternions=quaternions
        )
        self.required_nxy = int(
            tilt_geometry.estimate_required_nxy(self.nxy, self.nz, max_tilt_deg)
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

        # Multislice evaluates the exit wave's phase at the volume's midplane,
        # but dfu/dfv follow the CryoSPARC/RELION convention of being measured
        # from the entry face. `TiltSeriesGenerator` gets this correction from
        # `BaseImager._apply_defocus_shift`; this class does not inherit
        # `BaseImager`, so without it the inverse images the specimen at a
        # different defocus than the simulator that produced the data --
        # nz * voxel_size / 2 Angstrom out, which is 750 A on a 300-slice
        # tomogram at 5 A/voxel. Skipped for the models with no Z extent to
        # offset from, matching `_apply_defocus_shift`'s own `shift_required`.
        self._defocus_shift_angstrom = (
            0.0
            if scattering_model in ("projection", "ctf")
            else defocus_midplane_shift(self.nz, voxel_size)
        )
        if self._defocus_shift_angstrom:
            for name in ("dfu", "dfv"):
                if hasattr(self, name):
                    setattr(
                        self, name, getattr(self, name) - self._defocus_shift_angstrom
                    )
                    self.ctf_params[name] = getattr(self, name)

        # Fourier-space mask applied after each gradient step
        self.register_buffer("kmask", kmask)

        # Physics modules
        self.iterative_scattering = IterativeScattering(
            self.nxy,
            voxel_size,
            voltage,
            scattering_model=scattering_model,
            klim=klim,
            alpha=alpha,
        )
        self.aberration_backend = aberration_backend
        # Derived from scattering_model, not user-configurable independently
        # -- see aberrations.aberration_model_for_scattering.
        aberration_model = aberration_model_for_scattering(scattering_model)
        self.aberration: Aberration | LegacyAberrationAdapter
        if aberration_backend == "torch_ctf":
            self.aberration = LegacyAberrationAdapter(
                self.nxy,
                voxel_size,
                voltage,
                aberration_model=aberration_model,
                lpp_params=lpp_params,
            )
        else:
            self.aberration = Aberration(
                self.nxy,
                voxel_size,
                voltage,
                aberration_model=aberration_model,
                alpha=alpha if aberration_model == "linear" else None,
                # Same derivation as `BaseImager._init_optics`. Left at the
                # default the amplitude-contrast term is applied on one side of
                # the forward/inverse pair and not the other.
                specimen_absorption=scattering_model != "ctf",
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
            slice_batchsize=self.slice_batchsize,
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
            ctf_batch = tilt_geometry.shift_ctf_defocus_for_tilt(
                ctf_batch,
                tuple(V_batched.shape),
                theta_matrix,
                self.nz,
                self.voxel_size,
            )

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
            self.n_training_steps_per_epoch,
            self.n_training_steps,
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

        opts = self._optimizers_list()
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

        # Gathered/averaged across ranks under multi-GPU (DDP) for logging
        # only -- `loss` itself (fed to manual_backward below) stays the
        # local, per-rank value; DDP synchronises gradients on its own.
        self.log_norm_loss.append(self._gather_for_logging(norm_loss).cpu())
        self.log_total_loss.append(self._gather_for_logging(loss).cpu())

        self.manual_backward(loss)
        for opt in opts:
            opt.step()

        self._step_schedulers()

        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return loss

    # ------------------------------------------------------------------ #
    # Epoch / fit callbacks                                                #
    # ------------------------------------------------------------------ #

    def on_fit_start(self) -> None:
        """Create run directory and write hyperparameter metadata.

        Rank-0-only under multi-GPU (DDP): every replica would otherwise
        race to create/write the same files.
        """
        if self._run_dir is None or not self.trainer.is_global_zero:
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
        """Save per-epoch volume as MRC.

        Rank-0-only under multi-GPU (DDP): ``self.V`` is already identical
        across replicas at this point (kept in sync by DDP's gradient
        all-reduce every step), so every rank would otherwise race to write
        the same file.
        """
        if self._run_dir is None or not self.trainer.is_global_zero:
            return
        epoch = self.current_epoch + 1
        v = self.V.detach().cpu().float()
        mrc_path = self._run_dir / "epochs" / f"{epoch:03d}.mrc"
        save_volume_mrc(mrc_path, v, self.voxel_size)

    def on_fit_end(self) -> None:
        """Save final reconstructed volume and training metrics.

        Rank-0-only under multi-GPU (DDP), same reasoning as
        ``on_train_epoch_end``.
        """
        self._save_metrics()
        if self._run_dir is None or not self.trainer.is_global_zero:
            return
        v = self.V.detach().cpu().float()
        volume_path = self._run_dir / "volume.mrc"
        save_volume_mrc(volume_path, v, self.voxel_size)
        print(f"Saved final volume → {volume_path}")
