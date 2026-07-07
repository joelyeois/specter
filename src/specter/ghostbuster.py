from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import lightning as L
import mrcfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LRScheduler,
    OneCycleLR,
)

from . import rotations
from .aberrations import Aberration
from .arrays import compute_nps_2d
from .fft import fft3, ifft3
from .imagegenerator import ImageGenerator
from .imagegenerator._tiltseries import TiltSeriesGenerator
from .rotations import Rotation
from .scattering import IterativeScattering
from .symmetries import apply_symmetry, get_rotation_matrices


class Reconstructor(L.LightningModule):
    """
    Differentiable 3D reconstruction module for cryo-EM/cryo-ET.

    Reconstructs a 3D electrostatic potential volume from 2D experimental images
    by minimising the discrepancy between simulated and observed images. Supports
    joint refinement of the volume, rotations, translations, and defocus.

    Typically constructed and driven by :class:`Ghostbuster`, but can be used
    directly when particle tensors are already available in memory.

    Parameters
    ----------
    V : torch.Tensor
        Initial volume estimate, shape (Z, Y, X).
    voxel_size : float
        Voxel size in Å.
    quaternions : torch.Tensor
        Per-particle rotation quaternions, shape (N, 4), in xyzw convention.
    translations : torch.Tensor
        Per-particle translations in Å, shape (N, 2).
    ctf_params : dict[str, torch.Tensor]
        Per-particle CTF parameters; each value has leading dimension N.
    energy : float
        Electron energy in keV.
    dose_per_angstrom : float
        Electron dose in e⁻/Å².
    lr : float, optional
        Learning rate for volume V. None disables V optimisation.
    lr_R : float, optional
        Learning rate for rotations. None disables rotation refinement.
    lr_T : float, optional
        Learning rate for translations. None disables translation refinement.
    lr_D : float, optional
        Learning rate for defocus offset. None disables defocus refinement.
    bfactor : float or torch.Tensor, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function, damping high-resolution signal. Scalar (applied to all
        particles) or per-particle tensor. If given, overrides any
        ``"bfactor"`` entry already present in ``ctf_params``. None or 0.0
        means no envelope. Default None.
    scheduler : {"LambdaLR", "OneCycleLR", "CosineAnnealingWarmRestarts", "MultiplicativeLR"}
        LR scheduler applied to the volume optimiser. Default is "LambdaLR".
        ``"OneCycleLR"`` treats ``lr`` as the peak learning rate and decays
        aggressively over the configured run, which is useful for short
        screening jobs.
    sparsity : float, optional
        L1 regularisation weight on V. None disables sparsity.
    symmetry : str, optional
        Symmetry group to enforce (e.g. "C3", "D2"). None disables symmetry.
    scattering_model : str
        Scattering model passed to ImageGenerator. Default is "multislice".
    aberration_model : str
        Aberration model passed to ImageGenerator. Default is "holography".
    run_dir : str or Path, optional
        Directory to write per-epoch volumes, final volume, and metadata into.
        ``None`` disables all file output.
    use_2d_mask : bool
        If True, rotate ``fsc_mask`` for each particle, project it to 2D, and
        use that projected mask to weight the image-domain MSE loss.
    halfset_label : str, optional
        Short label (e.g. ``"A"`` or ``"B"``) appended to saved filenames when
        two halfset runs share the same ``run_dir``.
    """

    def __init__(
        self,
        V: torch.Tensor,
        voxel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, torch.Tensor],
        energy: float,
        dose_per_angstrom: float,
        anisomag: torch.Tensor | None = None,
        alpha: float = 0.0,
        defocus_offset: torch.Tensor = torch.tensor(0.0),
        bfactor: float | torch.Tensor | None = None,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        klim: float | None = None,
        sparsity: float | None = None,
        lr: float | None = None,
        lr_R: float | None = None,
        lr_T: float | None = None,
        lr_D: float | None = None,
        lr_decay: float = 0.1,
        scheduler: Literal[
            "LambdaLR",
            "OneCycleLR",
            "CosineAnnealingWarmRestarts",
            "MultiplicativeLR",
        ] = "LambdaLR",
        kmask: torch.Tensor | None = None,
        nps_weight: torch.Tensor | None = None,
        learn_noise_model: bool = False,
        noise_ema_momentum: float = 0.9,
        use_ncc: bool = False,
        ews_curvature_sign: str = "negative",
        fsc_ref: torch.Tensor | str | Path | None = None,
        fsc_mask: torch.Tensor | float | str | Path | None = None,
        cryosparc_ref: torch.Tensor | str | Path | None = None,
        use_2d_mask: bool = False,
        rotate_mode: Literal["real", "fourier"] = "real",
        symmetry: str | None = None,
        symmetry_batchsize: int | None = None,
        symmetry_mode: Literal["real", "fourier"] = "fourier",
        use_cpu_for_symmetry: bool = False,
        tag: str = "untagged",
        run_dir: str | Path | None = None,
        halfset_label: str | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "V",
                "quaternions",
                "translations",
                "ctf_params",
                "anisomag",
                "bfactor",
                "kmask",
                "nps_weight",
                "fsc_ref",
                "cryosparc_ref",
                "fsc_mask",
                "run_dir",
                "halfset_label",
            ]
        )
        self._run_dir: Path | None = Path(run_dir) if run_dir is not None else None
        self._halfset_label: str | None = halfset_label

        # Always use manual optimization to handle masking and multiple optimizers consistently
        self.automatic_optimization = False
        if all(l_rate is None for l_rate in [lr, lr_R, lr_T, lr_D]):
            print("Non-optimization mode.")
        elif sum(l_rate is None for l_rate in [lr, lr_R, lr_T, lr_D]) == 3:
            print("Single parameter optimization mode.")
        else:
            print("Multi-parameter optimization mode.")

        # optimization parameters
        self.sparsity = sparsity
        self.lr = lr
        self.lr_R = lr_R
        self.lr_T = lr_T
        self.lr_D = lr_D
        self.lr_decay = lr_decay
        self.log_lrs: list[float] = []
        self.log_total_loss: list[torch.Tensor] = []
        self.log_sparsity_loss: list[torch.Tensor] = []
        self.log_norm_loss: list[torch.Tensor] = []
        self.scheduler = scheduler

        # symmetry parameters
        self.symmetry = symmetry
        self.symmetry_batchsize = symmetry_batchsize
        self.symmetry_mode = symmetry_mode
        if symmetry is not None:
            sym_rot_matrices = get_rotation_matrices(symmetry)
            self.register_buffer("sym_rot_matrices", sym_rot_matrices)
        self.use_cpu_for_symmetry = use_cpu_for_symmetry

        # masks
        self.register_buffer("kmask", kmask)
        self.register_buffer("nps_weight", nps_weight)
        self.use_2d_mask = use_2d_mask

        # learned noise model (RELION-style): sigma^2(k) estimated from residuals
        self.learn_noise_model = learn_noise_model
        self.noise_ema_momentum = noise_ema_momentum
        self.use_ncc = use_ncc
        n = V.shape[-1]
        self.sigma2_k: torch.Tensor
        self.register_buffer("sigma2_k", torch.ones(n, n // 2 + 1))

        # fsc — load from file if path provided
        if isinstance(fsc_ref, (str, Path)):
            fsc_ref = torch.as_tensor(mrcfile.read(str(fsc_ref)))
        if isinstance(fsc_mask, (str, Path)):
            fsc_mask = torch.as_tensor(mrcfile.read(str(fsc_mask)))
        if fsc_mask is None:
            fsc_mask = 1
        if isinstance(fsc_mask, torch.Tensor):
            self.register_buffer("fsc_mask", fsc_mask)
        else:
            self.fsc_mask = fsc_mask
        self.fsc_ref = fsc_ref

        # cryosparc_ref — load from file if path provided
        if isinstance(cryosparc_ref, (str, Path)):
            cryosparc_ref = torch.as_tensor(mrcfile.read(str(cryosparc_ref)))
        self.cryosparc_ref = cryosparc_ref

        # model parameters
        self.dose_per_angstrom = dose_per_angstrom
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.energy = energy
        self.rotate_mode = rotate_mode
        if lr is None:
            self.register_buffer("V", V)
        else:
            self.V = nn.Parameter(V)

        # ctf
        self.ctf_params = {}
        for k, v in ctf_params.items():
            v_adjusted = v + defocus_offset if k in ("dfu", "dfv") else v
            self.register_buffer(k, v_adjusted)
            self.ctf_params[k] = getattr(self, k)
        if lr_D is None:
            self.register_buffer("defocus_offset", defocus_offset)
        else:
            self.defocus_offset = nn.Parameter(defocus_offset)

        # rotations
        if lr_R is None:
            self.register_buffer("rotations", quaternions)
        else:
            self.rotations = nn.Parameter(quaternions)

        # translations
        if lr_T is None:
            self.register_buffer("translations", translations)
        else:
            self.translations = nn.Parameter(translations)

        # anisomag
        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)

        # imaging models
        self.ews_curvature_sign = ews_curvature_sign
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model

        # initialize imagegenerator
        self.imagegenerator = ImageGenerator(
            self.V,
            self.voxel_size,
            self.rotations,
            self.translations,
            self.ctf_params,
            self.energy,
            self.dose_per_angstrom,
            anisomag=self.anisomag,
            ice_model=None,
            scattering_model=self.scattering_model,
            aberration_model=self.aberration_model,
            noise_model=None,
            klim=klim,
            ews_curvature_sign=self.ews_curvature_sign,
            alpha=self.alpha,
            rotate_mode=self.rotate_mode,
            bfactor=bfactor,
        )

    def forward(self, idx: torch.Tensor | int | slice) -> torch.Tensor:
        """
        Simulate images for the given particle indices.

        Parameters
        ----------
        idx : torch.Tensor, int, or slice
            Indices into the stored rotations/translations/CTF parameters.

        Returns
        -------
        torch.Tensor
            Simulated images, shape (B, H, W).
        """
        return self.imagegenerator(idx)

    def symmetrize(self) -> None:
        """Apply the configured point-group symmetry to the current volume in-place."""
        self.V.data = apply_symmetry(
            self.V.data,
            self.sym_rot_matrices,
            batchsize=self.symmetry_batchsize,
            method=self.symmetry_mode,
        )

    def reciprocal_lr_scheduler(self, *args: Any) -> float:
        """
        Reciprocal-square-root decay schedule: ``1 / (1 + decay * step^0.5)``.

        Returns
        -------
        float
            LR multiplier at the current global step.
        """
        return 1 / (1 + self.lr_decay * self.global_step**0.5)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        """
        Build optimizers and LR schedulers for all active parameters.

        Returns
        -------
        tuple
            (optimizers, lr_schedulers) — only the volume optimiser gets a
            scheduler; rotation/translation/defocus optimisers use a fixed LR.
        """
        if self.lr is not None:
            optimizerV = AdamW([self.V], lr=self.lr, weight_decay=0.0)
            # optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9)
            # optimizer = NAdam(self.parameters(), lr=self.lr)
        if self.lr_R is not None:
            optimizerR = AdamW([self.rotations], lr=self.lr_R)
            # optimizerR = Adam([self.rotations], lr=self.lr_R)
            # optimizer = LBFGS([self.rotations], lr=self.lr_R)
        if self.lr_T is not None:
            optimizerT = AdamW([self.translations], lr=self.lr_T)
        if self.lr_D is not None:
            optimizerD = AdamW([self.defocus_offset], lr=self.lr_D)

        lr_schedulers = []
        if self.lr is not None:
            if self.scheduler == "CosineAnnealingWarmRestarts":
                lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizerV,
                    self.num_training_steps_per_epoch(),
                    eta_min=1e-6,
                    T_mult=2,
                )
            elif self.scheduler == "OneCycleLR":
                lr_scheduler = OneCycleLR(
                    optimizerV,
                    max_lr=self.lr,
                    total_steps=self.num_training_steps(),
                    pct_start=0.1,
                    anneal_strategy="cos",
                    div_factor=10.0,
                    final_div_factor=1000.0,
                )
            elif self.scheduler == "MultiplicativeLR":
                lr_scheduler = ExponentialLR(optimizerV, 0.999)
            elif self.scheduler == "LambdaLR":
                lr_scheduler = LambdaLR(optimizerV, self.reciprocal_lr_scheduler)
            else:
                raise ValueError(
                    f"Unknown scheduler '{self.scheduler}'. "
                    "Choose 'LambdaLR', 'OneCycleLR', "
                    "'CosineAnnealingWarmRestarts', or 'MultiplicativeLR'."
                )
            lr_schedulers.append(lr_scheduler)

        opts = []
        if self.lr is not None:
            opts.append(optimizerV)
        if self.lr_R is not None:
            opts.append(optimizerR)
        if self.lr_T is not None:
            opts.append(optimizerT)
        if self.lr_D is not None:
            opts.append(optimizerD)
        return opts, lr_schedulers

    @staticmethod
    def _ncc_loss(
        pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Normalized cross-correlation loss scaled to MSE-equivalent units.

        Computes ``var(target) * (1 - NCC)`` per image, then averages over the
        batch.  The ``var(target)`` factor makes the loss dimensionally
        equivalent to MSE: at the optimum (low-SNR regime) both quantities
        converge to the noise variance ``σ²``, so the same learning rate can be
        used without retuning.

        Parameters
        ----------
        pred : torch.Tensor
            Simulated images, shape ``(B, H, W)``.
        target : torch.Tensor
            Experimental images, shape ``(B, H, W)``.
        eps : float
            Small constant added to the denominator for numerical stability.

        Returns
        -------
        torch.Tensor
            Scalar loss.

        Notes
        -----
        NCC is invariant to multiplicative and additive intensity rescaling,
        which makes it robust to gain-reference errors and forward-model scale
        mismatches that inflate ordinary MSE.
        """
        p = pred.flatten(1)  # (B, N)
        t = target.flatten(1)
        p_c = p - p.mean(dim=1, keepdim=True)
        t_c = t - t.mean(dim=1, keepdim=True)
        ncc = (p_c * t_c).sum(dim=1) / (
            p_c.norm(dim=1) * t_c.norm(dim=1) + eps
        )  # (B,), range [-1, 1]
        # var(target) per image: scales (1 - NCC) into MSE-equivalent units
        var_t = (t_c**2).mean(dim=1)  # (B,)
        return (var_t * (1.0 - ncc)).mean()

    def _update_sigma2(self, residuals: torch.Tensor) -> None:
        """Update the per-shell noise variance sigma^2(k) from real-space residuals.

        Mirrors the RELION noise model: the unexplained residuals are
        radially-averaged in Fourier space (via compute_nps_2d) to estimate
        sigma^2(k) per shell, then an EMA smooths the estimate across batches.

        sigma^2(k) is normalised by its mean after each update so that the
        relative spectral weighting adapts while the loss magnitude stays
        stable (comparable to the nps_weight and plain-MSE modes).
        """
        with torch.no_grad():
            # Raw per-shell power spectrum of residuals, shape (H, W//2+1)
            new_sigma2 = compute_nps_2d(
                residuals.detach(), normalize=False, zero_dc=False
            ).clamp(min=1e-10)
            # EMA update
            self.sigma2_k = (
                self.noise_ema_momentum * self.sigma2_k
                + (1 - self.noise_ema_momentum) * new_sigma2
            )
            # Normalise by mean so that a flat sigma^2 gives uniform weights
            # (i.e. loss magnitude remains comparable to real-space MSE).
            self.sigma2_k = self.sigma2_k / self.sigma2_k.mean().clamp(min=1e-10)

    def _compute_loss(
        self, out: torch.Tensor, images: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the image-domain loss between simulated and experimental images.

        Selects the active loss model in priority order: NCC → learned noise →
        NPS-weighted → plain MSE. Sparsity regularisation on V is added on top.

        Parameters
        ----------
        out : torch.Tensor
            Simulated images, shape (B, H, W).
        images : torch.Tensor
            Experimental images, shape (B, H, W).
        idx : torch.Tensor
            Particle indices for this batch.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        if self.use_ncc:
            loss = self._ncc_loss(out, images)
        elif self.learn_noise_model:
            # RELION-style: estimate sigma^2(k) from residuals, weight by 1/sigma^2(k).
            # sigma2_k is updated with no_grad (EM E-step); gradient flows only
            # through the residuals (M-step).
            images_f = torch.fft.rfft2(images)
            out_f = torch.fft.rfft2(out)
            H, W = images.shape[-2:]
            self._update_sigma2(images - out)
            loss = torch.mean(
                (images_f - out_f).abs() ** 2 / self.sigma2_k.detach()
            ) / (H * W)
        elif self.nps_weight is not None:
            images_f = torch.fft.rfft2(images)
            out_f = torch.fft.rfft2(out)
            H, W = images.shape[-2:]
            # Divide by H*W so that a flat (normalised) NPS weight gives the
            # same loss magnitude as real-space MSE (Parseval equivalence).
            loss = torch.mean(self.nps_weight * (images_f - out_f).abs() ** 2) / (H * W)
        else:
            mse = F.mse_loss(images, out, reduction="none")
            if self.use_2d_mask:
                mse = mse * self._project_fsc_mask_2d(idx, images.shape)
            loss = mse.mean()
        self.log_norm_loss.append(loss.detach().cpu())

        if self.sparsity is not None:
            sparsity_loss = self.sparsity * torch.mean(torch.abs(self.V))
            loss = loss + sparsity_loss
            self.log_sparsity_loss.append(sparsity_loss.detach().cpu())

        self.log_total_loss.append(loss.detach().cpu())
        return loss

    def _project_fsc_mask_2d(
        self, idx: torch.Tensor, image_shape: torch.Size
    ) -> torch.Tensor:
        """Rotate ``fsc_mask`` with the image generator geometry and max-project it."""
        if not isinstance(self.fsc_mask, torch.Tensor):
            raise ValueError("use_2d_mask=True requires fsc_mask to be a 3D tensor.")
        if self.fsc_mask.ndim != 3:
            raise ValueError(
                f"use_2d_mask=True requires a 3D fsc_mask, got shape {tuple(self.fsc_mask.shape)}."
            )

        mask = self.fsc_mask
        if tuple(mask.shape) != tuple(self.V.shape[-3:]):
            raise ValueError(
                "use_2d_mask=True requires fsc_mask shape to match V shape; "
                f"got {tuple(mask.shape)} and {tuple(self.V.shape[-3:])}."
            )

        with torch.no_grad():
            Q = self.rotations[idx]
            T = self.translations[idx]
            if len(Q.shape) < 2:
                Q = Q.unsqueeze(0)
            if len(T.shape) < 2:
                T = T.unsqueeze(0)
            R = Rotation.from_quat(Q)
            T = rotations.translations_angstrom_to_torch(
                T, self.imagegenerator.nxy, self.imagegenerator.pixel_size
            )
            theta = rotations.build_affine_matrix(R.as_matrix(), T)
            projected = self.imagegenerator.rotator(mask, theta).max(dim=1).values

        if projected.shape != image_shape:
            raise ValueError(
                "Projected 2D mask shape does not match image batch shape; "
                f"got {tuple(projected.shape)} and {tuple(image_shape)}."
            )
        return projected

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Link parameters to simulator to ensure autograd graph connectivity
        if hasattr(self.imagegenerator, "V"):
            self.imagegenerator.V = self.V
        if hasattr(self.imagegenerator, "quaternions"):
            self.imagegenerator.quaternions = self.rotations

        images, idx = batch
        out = self.forward(idx)
        loss = self._compute_loss(out, images, idx)
        return loss, out, images

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Run one manual optimisation step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (images, indices) from the dataloader.
        batch_idx : int
            Batch index (unused, required by Lightning interface).

        Returns
        -------
        torch.Tensor
            Scalar loss for the current batch.
        """
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        loss, _, _ = self._common_step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        for opt in opts:
            opt.zero_grad()
        self.manual_backward(loss)
        for opt in opts:
            opt.step()

        # manual optimization
        if not self.automatic_optimization:
            sch = self.lr_schedulers()
            if sch:
                if not isinstance(sch, (list, tuple)):
                    sch = [sch]
                for s in sch:
                    s.step()
        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Record the current learning rate before each batch."""
        if self.lr is not None:
            self.log_lrs.append(
                self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[
                    0
                ]["lr"]
            )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Apply the Fourier-space k-mask to V after each gradient update."""
        if self.kmask is not None:
            self.V.data = torch.real(
                ifft3(fft3(self.V.data, shift=True) * self.kmask, shift=True)
            )

    def on_fit_start(self) -> None:
        """Create the run directory and write metadata."""
        if self._run_dir is None:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "epochs").mkdir(exist_ok=True)

        saved_arrays: list[str] = []
        for name in ("nps_weight", "kmask"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                torch.save(tensor.detach().cpu(), self._run_dir / f"{name}.pt")
                saved_arrays.append(name)

        meta: dict[str, Any] = {}
        meta.update(dict(self.hparams))
        if saved_arrays:
            meta["saved_arrays"] = saved_arrays
        suffix = f"_{self._halfset_label}" if self._halfset_label is not None else ""
        (self._run_dir / f"params{suffix}.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        print(f"Run directory: {self._run_dir}")

    def _save_metrics(self) -> None:
        """Save training metrics (loss, lr) to JSON per epoch."""
        if self._run_dir is None or not self.log_total_loss:
            return

        suffix = f"_{self._halfset_label}" if self._halfset_label is not None else ""
        metrics_path = self._run_dir / f"metrics{suffix}.json"

        # Convert per-batch lists to per-epoch statistics
        n_batches_per_epoch = len(self.log_total_loss) // max(self.current_epoch, 1)
        if n_batches_per_epoch == 0:
            n_batches_per_epoch = len(self.log_total_loss)

        epochs = {}
        for epoch in range(self.current_epoch + 1):
            start_idx = epoch * n_batches_per_epoch
            end_idx = (epoch + 1) * n_batches_per_epoch

            epoch_losses = [
                float(value) for value in self.log_total_loss[start_idx:end_idx]
            ]
            epoch_norm_losses = [
                float(value) for value in self.log_norm_loss[start_idx:end_idx]
            ]
            epoch_lrs = self.log_lrs[start_idx:end_idx]  # Already floats

            epochs[f"epoch_{epoch + 1:02d}"] = {
                "loss": {
                    "mean": sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses
                    else None,
                    "min": min(epoch_losses) if epoch_losses else None,
                    "max": max(epoch_losses) if epoch_losses else None,
                    "std": (
                        sum(
                            (x - (sum(epoch_losses) / len(epoch_losses))) ** 2
                            for x in epoch_losses
                        )
                        / len(epoch_losses)
                    )
                    ** 0.5
                    if len(epoch_losses) > 1
                    else 0.0,
                },
                "norm_loss": {
                    "mean": sum(epoch_norm_losses) / len(epoch_norm_losses)
                    if epoch_norm_losses
                    else None,
                },
                "lr": {
                    "start": epoch_lrs[0] if epoch_lrs else None,
                    "end": epoch_lrs[-1] if epoch_lrs else None,
                    "min": min(epoch_lrs) if epoch_lrs else None,
                },
            }
            if self.log_sparsity_loss:
                epoch_sparsity = [
                    float(value) for value in self.log_sparsity_loss[start_idx:end_idx]
                ]
                epochs[f"epoch_{epoch + 1:02d}"]["sparsity_loss"] = {
                    "mean": sum(epoch_sparsity) / len(epoch_sparsity)
                    if epoch_sparsity
                    else None,
                }

        meta = {
            "total_batches": len(self.log_total_loss),
            "batches_per_epoch": n_batches_per_epoch,
            "total_epochs": self.current_epoch + 1,
            "total_loss": [float(v) for v in self.log_total_loss],
            "epochs": epochs,
        }

        metrics_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metrics → {metrics_path}")

    def on_fit_end(self) -> None:
        """Save the final reconstructed volume, FSC figure, and training metrics."""
        # Save metrics first (before v is computed, so they capture all epochs)
        self._save_metrics()

        if self._run_dir is None:
            return
        suffix = f"_{self._halfset_label}" if self._halfset_label is not None else ""
        v = self.V.detach().cpu().float()
        vol_path = self._run_dir / f"vol{suffix}.mrc"
        with mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(v.numpy())
            mrc.voxel_size = self.voxel_size
        print(f"Saved final volume → {vol_path}")

        if self.fsc_ref is not None:
            self._save_fsc_figure(
                v, suffix, self._run_dir / f"fsc{suffix}.png", label=f"final{suffix}"
            )

    def on_train_epoch_end(self) -> None:
        """Enforce symmetry, save per-epoch volume, plot3d preview, and FSC."""
        if self.symmetry is not None:
            self.V.data = apply_symmetry(
                self.V.data,
                self.sym_rot_matrices,
                batchsize=self.symmetry_batchsize,
                method=self.symmetry_mode,
            )

        if self._run_dir is None:
            return

        epoch = self.current_epoch + 1
        suffix = f"_{self._halfset_label}" if self._halfset_label is not None else ""
        v = self.V.detach().cpu().float()

        mrc_path = self._run_dir / "epochs" / f"{epoch:03d}{suffix}.mrc"
        with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
            mrc.set_data(v.numpy())
            mrc.voxel_size = self.voxel_size

        self._save_plot3d(v, suffix=suffix, epoch=epoch)
        if self.fsc_ref is not None:
            self._save_fsc_figure(
                v,
                suffix,
                self._run_dir / "epochs" / f"fsc_{epoch:03d}{suffix}.png",
                label=f"epoch {epoch}{suffix}",
            )

    def _save_plot3d(self, v: torch.Tensor, suffix: str, epoch: int) -> None:
        """Save a plot3d preview of the current volume. Silently skips on failure."""
        if self._run_dir is None:
            return
        try:
            import matplotlib.pyplot as plt

            from .plots import plot3d

            fig = plot3d(v, title=f"Epoch {epoch}{suffix}", show=False)
            assert fig is not None
            fig.savefig(
                self._run_dir / "epochs" / f"vol_{epoch:03d}{suffix}.png",
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as exc:
            print(f"[Reconstructor] plot3d preview skipped: {exc}")

    def _save_fsc_figure(
        self,
        v: torch.Tensor,
        suffix: str,
        path: Path,
        label: str,
    ) -> None:
        """Compute and save an FSC figure with optional CryoSPARC reference. Silently skips on failure."""
        try:
            import matplotlib.pyplot as plt

            from .plots import plot_map_to_model_fsc

            # Move references to volume's device for GPU FSC computation.
            device = v.device
            fsc_ref = (
                self.fsc_ref.detach().to(device).float()
                if isinstance(self.fsc_ref, torch.Tensor)
                else self.fsc_ref
            )
            fsc_mask = (
                self.fsc_mask.detach().to(device)
                if isinstance(self.fsc_mask, torch.Tensor)
                else None
            )

            # Build list of volumes and labels
            vols = [v]
            labels = [label]

            # Add CryoSPARC reference if both fsc_ref and cryosparc_ref are set
            if self.cryosparc_ref is not None and self.fsc_ref is not None:
                cs_ref = (
                    self.cryosparc_ref.detach().to(device).float()
                    if isinstance(self.cryosparc_ref, torch.Tensor)
                    else self.cryosparc_ref
                )
                vols.append(cs_ref)
                labels.append("CryoSPARC")

            fig = plot_map_to_model_fsc(
                vols,
                fsc_ref,
                voxel_size=self.voxel_size,
                mask=fsc_mask,
                labels=labels,
                show=False,
            )
            assert fig is not None
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"[Reconstructor] FSC plot skipped: {exc}")

    def num_training_steps_per_epoch(self) -> int:
        """
        Return the number of optimizer steps per training epoch.

        Returns
        -------
        int
            Steps per epoch, accounting for gradient accumulation.
        """
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches

        return num_steps

    def num_training_steps(self) -> int:
        """
        Return the total number of optimizer steps in the configured run.

        Returns
        -------
        int
            Total scheduler steps across all epochs.
        """
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs < 1:
            raise ValueError("OneCycleLR requires a positive trainer.max_epochs.")

        return self.num_training_steps_per_epoch() * max_epochs


def compare_runs(
    base_dir: str | Path = "output/runs",
    show_params: list[str] | None = None,
) -> None:
    """
    Print a summary table of all saved Ghostbuster runs.

    Parameters
    ----------
    base_dir : str or Path
        Root directory containing run subdirectories.
    show_params : list of str, optional
        Parameter keys to display as columns. Defaults to
        ``["scattering_model", "lr", "rotate_mode", "use_nps", "niter"]``.
    """
    if show_params is None:
        show_params = ["scattering_model", "lr", "rotate_mode", "use_nps", "niter"]

    seen: set[Path] = set()
    runs = sorted(
        p
        for p in Path(base_dir).glob("*/params*.json")
        if not (p.parent in seen or seen.add(p.parent))  # type: ignore[func-returns-value]
    )
    if not runs:
        print(f"No runs found in {base_dir}.")
        return

    col_w = 20
    header = f"{'run':<45}" + "".join(f"  {k:<{col_w}}" for k in show_params)
    print(header)
    print("-" * len(header))
    for p in runs:
        d = json.loads(p.read_text())
        row = f"{p.parent.name:<45}" + "".join(
            f"  {str(d.get(k, '—')):<{col_w}}" for k in show_params
        )
        print(row)


class Ghostbuster:
    """
    End-to-end reconstruction pipeline for cryo-EM/cryo-ET.

    .. note::
        ``return_class`` and ``cryosparc_ref`` are excluded from the job
        parameter log (see :attr:`_job_log_exclude`) because they differ
        between halfsets. ``return_class`` encodes halfset identity in the
        output filenames (``vol_A.mrc`` / ``vol_B.mrc``), and ``cryosparc_ref``
        points to the corresponding halfset reference. Both halfset runs must
        use identical settings for every other parameter.

    Loads particle data from CryoSPARC output files, preprocesses images
    (sign flip, dose/scale normalisation), and drives a :class:`Reconstructor`
    via a Lightning ``Trainer``.

    All parameters — data paths, physics settings, and training hyperparameters
    — are passed at construction so that :meth:`~specter.jobs.Job.create` can
    capture the full run configuration in one call:

    .. code-block:: python

        with Job("ghostbuster", "my-project") as job:
            gb = job.create(
                Ghostbuster,
                cs_file="particles.cs",
                mrc_file="stack.mrcs",
                dose_per_angstrom=40.0,
                lr=0.1,
                symmetry="I1",
                epochs=5,
            )
            model = gb.run(device=0)

    Parameters
    ----------
    cs_file : str or Path
        Path to a CryoSPARC ``.cs`` file.
    mrc_file : str or Path
        Path to the particle stack ``.mrc``/``.mrcs`` file.
    dose_per_angstrom : float
        Electron dose in e⁻/Å².
    lr : float, optional
        Learning rate for the volume. ``None`` disables volume optimisation.
    lr_R : float, optional
        Learning rate for rotations.
    lr_T : float, optional
        Learning rate for translations.
    lr_D : float, optional
        Learning rate for defocus offset.
    defocus_offset : float, optional
        Initial defocus offset in Ångströms added to all particles' dfu and dfv.
    bfactor : float, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function, damping high-resolution signal. None or 0.0 means no
        envelope. Default None.
    scheduler : {"LambdaLR", "OneCycleLR", "CosineAnnealingWarmRestarts", "MultiplicativeLR"}
        LR scheduler for the volume optimiser.
    epochs : int
        Number of training epochs.
    batch_size : int
        Dataloader batch size.
    scattering_model : str
        Wave propagation model (``"multislice"``, ``"rytov"``, ``"firstborn"``,
        ``"projection"``).
    aberration_model : str
        CTF aberration model passed to ``ImageGenerator``.
    symmetry : str, optional
        Point-group symmetry to enforce (e.g. ``"C3"``, ``"I1"``).
    symmetry_batchsize : int, optional
        Batch size used when applying symmetry to the volume.
    symmetry_mode : {"real", "fourier"}
        Domain in which symmetry is applied.
    sparsity : float, optional
        L1 regularisation weight on V.
    rotate_mode : {"real", "fourier"}
        Domain in which rotations are applied.
    ews_curvature_sign : str
        Ewald sphere curvature sign matching CryoSPARC's convention.
        ``'negative'`` (default) or ``'positive'``.
    klim : float, optional
        Hard frequency cutoff passed to ``ImageGenerator``.
    nps_weight : torch.Tensor, optional
        Per-frequency noise power spectrum weight for the loss.
    learn_noise_model : bool
        Whether to estimate sigma²(k) from residuals (RELION-style).
    use_ncc : bool
        Whether to use normalised cross-correlation loss instead of MSE.
    fsc_ref : torch.Tensor, str, Path, or None
        Reference volume for map-to-model FSC logging. Can be a tensor or a
        path to a .mrc file to load.
    fsc_mask : torch.Tensor, float, str, Path, or None
        Mask applied before FSC computation. Can be a tensor, scalar, or a
        path to a .mrc file to load.
    cryosparc_ref : torch.Tensor, str, Path, or None
        CryoSPARC reference volume for FSC comparison. Can be a tensor or a
        path to a .mrc file to load. Only plotted alongside fsc_ref when
        both are provided. Default is None.
    use_2d_mask : bool
        If True, rotate ``fsc_mask`` for each particle, max-project it to 2D,
        and use that projected mask to weight the image-domain MSE loss.
    precision : str
        Lightning ``Trainer`` precision (e.g. ``"16-mixed"``, ``"32"``).
        Falls back to ``"32"`` automatically on CPU.
    num_workers : int
        Dataloader worker processes.
    num_particles : int, optional
        Use only the first ``num_particles`` particles. Defaults to all.
    return_class : {"0", "1", "all"}
        Particle class to extract from the ``.cs`` file.
    run_dir : str or Path, optional
        Directory for all job outputs. Injected automatically by
        :meth:`~specter.jobs.Job.create` when used inside a ``Job`` context.
    """

    _job_log_exclude: tuple[str, ...] = ("return_class", "cryosparc_ref")

    def __init__(
        self,
        cs_file: str | Path,
        mrc_file: str | Path,
        dose_per_angstrom: float,
        lr: float | None = None,
        lr_R: float | None = None,
        lr_T: float | None = None,
        lr_D: float | None = None,
        defocus_offset: float = 0.0,
        bfactor: float | None = None,
        scheduler: Literal[
            "LambdaLR",
            "OneCycleLR",
            "CosineAnnealingWarmRestarts",
            "MultiplicativeLR",
        ] = "LambdaLR",
        epochs: int = 5,
        batch_size: int = 3,
        scattering_model: str = "rytov",
        aberration_model: str = "holography",
        symmetry: str | None = None,
        symmetry_batchsize: int | None = None,
        symmetry_mode: Literal["real", "fourier"] = "fourier",
        sparsity: float | None = None,
        rotate_mode: Literal["real", "fourier"] = "real",
        ews_curvature_sign: str = "negative",
        klim: float | None = None,
        nps_weight: torch.Tensor | None = None,
        learn_noise_model: bool = False,
        use_ncc: bool = False,
        fsc_ref: torch.Tensor | str | Path | None = None,
        fsc_mask: torch.Tensor | float | str | Path | None = None,
        cryosparc_ref: torch.Tensor | str | Path | None = None,
        use_2d_mask: bool = False,
        precision: str = "16-mixed",
        num_workers: int = 0,
        num_particles: int | None = None,
        return_class: Literal["0", "1", "all"] = "all",
        run_dir: str | Path | None = None,
    ) -> None:
        from .cryosparc import extract_parameters_from_csfile

        _halfset_map: dict[str, str | None] = {"0": "A", "1": "B", "all": None}
        self.halfset_label: str | None = _halfset_map[return_class]

        print(f"Loading particle parameters from {Path(cs_file).name} ...")
        (
            energy,
            pixel_size,
            alpha,
            rotations,
            translations,
            ctf_params,
            scale,
            anisomag,
            indices,
            _split,
        ) = extract_parameters_from_csfile(
            str(cs_file), return_class=return_class, n_particles=num_particles
        )
        _n_loaded = len(rotations)
        _energy_val = float(energy.item() if hasattr(energy, "item") else energy)
        _pixel_size_val = float(
            pixel_size.item() if hasattr(pixel_size, "item") else pixel_size
        )
        print(
            f"  {_n_loaded} particles  |  {_energy_val:.0f} keV  |  {_pixel_size_val:.3f} Å/px"
        )

        print(f"Loading particle stack from {Path(mrc_file).name} ...")
        with mrcfile.mmap(str(mrc_file)) as mrc:
            images = torch.as_tensor((mrc.data[indices]).copy())
        _h, _w = images.shape[-2], images.shape[-1]
        print(f"  {len(images)} images  |  box {_h}×{_w}  |  dtype {images.dtype}")

        images = -images
        voxel_size = float(
            pixel_size.item() if hasattr(pixel_size, "item") else pixel_size
        )
        dose_per_area = dose_per_angstrom * voxel_size**2
        images = dose_per_area**0.5 * scale[..., None, None] * images + dose_per_area

        # preprocessed particle data (not hyperparams — not logged by job.create)
        self._images = images
        self._rotations = rotations
        self._translations = translations
        self._ctf_params = ctf_params
        self._anisomag = anisomag
        self._energy = float(energy.item() if hasattr(energy, "item") else energy)
        self._voxel_size = voxel_size
        self._alpha = float(alpha.item() if hasattr(alpha, "item") else alpha)

        # training hyperparameters (stored for run() and test_run())
        self.lr = lr
        self.lr_R = lr_R
        self.lr_T = lr_T
        self.lr_D = lr_D
        self.defocus_offset = defocus_offset
        self.bfactor = bfactor
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.symmetry = symmetry
        self.symmetry_batchsize = symmetry_batchsize
        self.symmetry_mode = symmetry_mode
        self.sparsity = sparsity
        self.rotate_mode = rotate_mode
        self.ews_curvature_sign = ews_curvature_sign
        self.klim = klim
        self.nps_weight = nps_weight
        self.learn_noise_model = learn_noise_model
        self.use_ncc = use_ncc
        self.fsc_ref = fsc_ref
        self.fsc_mask = fsc_mask
        self.cryosparc_ref = cryosparc_ref
        self.use_2d_mask = use_2d_mask
        self.precision = precision
        self.num_workers = num_workers
        self.num_particles = num_particles
        self.dose_per_angstrom = dose_per_angstrom
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def _build_reconstructor_and_loader(
        self,
        images: torch.Tensor,
        voxel_size: float,
        scattering_model: str,
        batch_size: int,
    ) -> tuple["Reconstructor", torch.utils.data.DataLoader]:
        from .arrays import ball3d

        n = images.shape[-1]
        kmask = ball3d(n, n)
        volume_init = torch.zeros(n, n, n)

        idx = torch.arange(len(images))
        dataset = torch.utils.data.TensorDataset(images, idx)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers
        )

        model = Reconstructor(
            volume_init,
            voxel_size,
            self._rotations,
            self._translations,
            self._ctf_params,
            self._energy,
            self.dose_per_angstrom,
            anisomag=self._anisomag,
            alpha=self._alpha,
            defocus_offset=torch.tensor(self.defocus_offset),
            bfactor=self.bfactor,
            scattering_model=scattering_model,
            aberration_model=self.aberration_model,
            lr=self.lr,
            lr_R=self.lr_R,
            lr_T=self.lr_T,
            lr_D=self.lr_D,
            scheduler=self.scheduler,
            kmask=kmask,
            klim=self.klim,
            nps_weight=self.nps_weight,
            learn_noise_model=self.learn_noise_model,
            use_ncc=self.use_ncc,
            sparsity=self.sparsity,
            rotate_mode=self.rotate_mode,
            ews_curvature_sign=self.ews_curvature_sign,
            symmetry=self.symmetry,
            symmetry_batchsize=self.symmetry_batchsize,
            symmetry_mode=self.symmetry_mode,
            fsc_ref=self.fsc_ref,
            fsc_mask=self.fsc_mask,
            cryosparc_ref=self.cryosparc_ref,
            use_2d_mask=self.use_2d_mask,
            run_dir=self.run_dir,
            halfset_label=self.halfset_label,
        )
        return model, loader

    def run(
        self,
        device: int = 0,
        callbacks: list[Any] | None = None,
    ) -> "Reconstructor":
        """
        Run the full reconstruction and return the trained :class:`Reconstructor`.

        Parameters
        ----------
        device : int
            GPU index. Ignored when CUDA is unavailable.
        callbacks : list, optional
            Additional Lightning callbacks passed to the ``Trainer``.

        Returns
        -------
        Reconstructor
            The trained model. Access the volume via ``model.V.detach()``.
        """
        _box = self._images.shape[-1]
        use_gpu = torch.cuda.is_available()
        _device_str = f"GPU {device}" if use_gpu else "CPU"
        print(
            f"Starting reconstruction: {len(self._images)} particles  |  box {_box}³  |  "
            f"{self.scattering_model}  |  {self.epochs} epochs  |  "
            f"batch {self.batch_size}  |  {_device_str}"
        )
        model, loader = self._build_reconstructor_and_loader(
            self._images,
            self._voxel_size,
            self.scattering_model,
            self.batch_size,
        )

        trainer = L.Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            devices=[device] if use_gpu else 1,
            max_epochs=self.epochs,
            precision=self.precision if use_gpu else "32",
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks or [],
        )
        trainer.fit(model, loader)
        return model

    def test_run(
        self,
        bin_factor: int = 8,
        device: int = 0,
        callbacks: list[Any] | None = None,
    ) -> "Reconstructor":
        """
        Quick sanity check: run 1 epoch on spatially binned images.

        Bins the loaded images by ``bin_factor`` in each spatial dimension and
        runs a single training epoch using the symmetry and other settings
        configured at construction.  Use this after constructing a
        :class:`Ghostbuster` to verify that files, parameters, and the physics
        pipeline are all wired up correctly before committing to a full run.

        Parameters
        ----------
        bin_factor : int
            Spatial downsampling factor applied to images and voxel size.
            Default 8.
        device : int
            GPU index. Ignored when CUDA is unavailable.
        callbacks : list, optional
            Additional Lightning callbacks passed to the ``Trainer``.

        Returns
        -------
        Reconstructor
            The trained model after one epoch.
        """
        n_particles = len(self._images)
        print(f"Test run: {n_particles} particles  |  {bin_factor}× binned  |  1 epoch")

        pool = torch.nn.AvgPool2d(bin_factor, stride=bin_factor)
        images_binned = pool(self._images.unsqueeze(1)).squeeze(1) * bin_factor**2
        voxel_size_binned = self._voxel_size * bin_factor

        model, loader = self._build_reconstructor_and_loader(
            images_binned,
            voxel_size_binned,
            self.scattering_model,
            self.batch_size,
        )

        use_gpu = torch.cuda.is_available()
        trainer = L.Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            devices=[device] if use_gpu else 1,
            max_epochs=1,
            precision="32",
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks or [],
        )
        trainer.fit(model, loader)

        v = model.V.detach()
        print(
            f"Test run passed — {bin_factor}× binned, {n_particles} particles, "
            f"box {images_binned.shape[-1]}³  |  "
            f"V min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}"
        )
        return model


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
        rotvec = Rotation.from_quat(Q.unsqueeze(0)).as_rotvec()[0]
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
        R_mat = Rotation.from_quat(Q).as_matrix()
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

        if self.scheduler == "CosineAnnealingWarmRestarts":
            lr_scheduler: LRScheduler = CosineAnnealingWarmRestarts(
                optimizerV,
                self.num_training_steps_per_epoch(),
                eta_min=1e-6,
                T_mult=2,
            )
        elif self.scheduler == "OneCycleLR":
            lr_scheduler = OneCycleLR(
                optimizerV,
                max_lr=self.lr,
                total_steps=self.num_training_steps(),
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=1000.0,
            )
        elif self.scheduler == "MultiplicativeLR":
            lr_scheduler = ExponentialLR(optimizerV, 0.999)
        elif self.scheduler == "LambdaLR":
            lr_scheduler = LambdaLR(optimizerV, self.reciprocal_lr_scheduler)
        else:
            raise ValueError(
                f"Unknown scheduler '{self.scheduler}'. "
                "Choose 'LambdaLR', 'OneCycleLR', "
                "'CosineAnnealingWarmRestarts', or 'MultiplicativeLR'."
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
        if self.lr is not None and self.trainer.lr_scheduler_configs:
            self.log_lrs.append(
                self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[
                    0
                ]["lr"]
            )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Apply the Fourier-space k-mask to V after each gradient update."""
        if self.kmask is not None:
            self.V.data = torch.real(
                ifft3(fft3(self.V.data, shift=True) * self.kmask, shift=True)
            )

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
        n_batches_per_epoch = len(self.log_total_loss) // max(self.current_epoch, 1)
        if n_batches_per_epoch == 0:
            n_batches_per_epoch = len(self.log_total_loss)

        epochs: dict[str, Any] = {}
        for epoch in range(self.current_epoch + 1):
            start = epoch * n_batches_per_epoch
            end = (epoch + 1) * n_batches_per_epoch
            ep_losses = [float(v) for v in self.log_total_loss[start:end]]
            ep_norm = [float(v) for v in self.log_norm_loss[start:end]]
            ep_lrs = self.log_lrs[start:end]
            key = f"epoch_{epoch + 1:02d}"
            epochs[key] = {
                "loss": {
                    "mean": sum(ep_losses) / len(ep_losses) if ep_losses else None,
                    "min": min(ep_losses) if ep_losses else None,
                    "max": max(ep_losses) if ep_losses else None,
                },
                "norm_loss": {
                    "mean": sum(ep_norm) / len(ep_norm) if ep_norm else None,
                },
                "lr": {
                    "start": ep_lrs[0] if ep_lrs else None,
                    "end": ep_lrs[-1] if ep_lrs else None,
                },
            }
            if self.log_sparsity_loss:
                ep_sp = [float(v) for v in self.log_sparsity_loss[start:end]]
                epochs[key]["sparsity_loss"] = {
                    "mean": sum(ep_sp) / len(ep_sp) if ep_sp else None,
                }

        metrics_path.write_text(
            json.dumps(
                {
                    "total_batches": len(self.log_total_loss),
                    "batches_per_epoch": n_batches_per_epoch,
                    "total_epochs": self.current_epoch + 1,
                    "total_loss": [float(v) for v in self.log_total_loss],
                    "epochs": epochs,
                },
                indent=2,
            )
        )
        print(f"Saved metrics → {metrics_path}")

    # ------------------------------------------------------------------ #
    # Step-count helpers (identical to Reconstructor)                      #
    # ------------------------------------------------------------------ #

    def num_training_steps_per_epoch(self) -> int:
        """Number of optimiser steps per epoch."""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        self.trainer.fit_loop.setup_data()
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
    energy : float
        Electron beam energy in keV.
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
    batch_size : int
        Tilt images per optimisation step.  Default 1 (one tilt per step).
    scattering_model : str
        Wave propagation model.  Default ``"multislice"``.
    aberration_model : str
        CTF aberration model.  Default ``"holography"``.
    klim : float, optional
        Hard frequency cutoff.
    alpha : float
        Amplitude contrast ratio.  Default ``0.0``.
    taper_width : int
        XY cosine taper on V before each forward pass.  Default 0.
    z_taper_width : int
        Z cosine taper on V.  Default 0.
    use_fov_mask : bool
        Mask MSE loss to the real-FOV region per tilt.  Default ``True``.
    scheduler : str
        LR scheduler.  Default ``"LambdaLR"``.
    slice_batch_size : int
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
        energy: float,
        ctf_params: dict[str, Any],
        angles: Sequence[float] | torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        tilt_axis: str = "x",
        nz: int | None = None,
        V_init: torch.Tensor | None = None,
        flip_contrast: bool = True,
        lr: float | None = None,
        sparsity: float | None = None,
        epochs: int = 5,
        batch_size: int = 1,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        klim: float | None = None,
        alpha: float = 0.0,
        taper_width: int = 0,
        z_taper_width: int = 0,
        use_fov_mask: bool = True,
        scheduler: Literal[
            "LambdaLR",
            "OneCycleLR",
            "CosineAnnealingWarmRestarts",
            "MultiplicativeLR",
        ] = "LambdaLR",
        slice_batch_size: int = 1,
        num_workers: int = 0,
        precision: str = "16-mixed",
        run_dir: str | Path | None = None,
    ) -> None:
        # Load tilt series
        if isinstance(tilt_series, (str, Path)):
            print(f"Loading tilt series from {Path(tilt_series).name} ...")
            with mrcfile.open(str(tilt_series)) as mrc:
                images = torch.as_tensor(mrc.data.copy()).float()
        else:
            images = torch.as_tensor(tilt_series).float()

        if flip_contrast:
            images = -images

        n_tilts, H, W = images.shape
        print(
            f"  {n_tilts} tilts  |  {H}×{W} px  |  {voxel_size:.3f} Å/px  |  "
            f"{energy:.0f} keV"
        )

        # Resolve rotation quaternions
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
            quaternions = Rotation.from_rotvec(rotvecs).as_quat()

        quats = torch.as_tensor(quaternions, dtype=torch.float32)
        trans = (
            torch.zeros(n_tilts, 2, dtype=torch.float32)
            if translations is None
            else torch.as_tensor(translations, dtype=torch.float32)
        )

        # Initial volume
        nxy = W
        _nz = nz if nz is not None else nxy
        if V_init is not None:
            volume_init = torch.as_tensor(V_init).float()
        else:
            volume_init = torch.zeros(_nz, nxy, nxy)

        # Store preprocessed data and settings
        self._images = images
        self._quaternions = quats
        self._translations = trans
        self._ctf_params = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in ctf_params.items()
        }
        self._volume_init = volume_init
        self._voxel_size = voxel_size
        self._energy = energy

        self.lr = lr
        self.sparsity = sparsity
        self.epochs = epochs
        self.batch_size = batch_size
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.klim = klim
        self.alpha = alpha
        self.taper_width = taper_width
        self.z_taper_width = z_taper_width
        self.use_fov_mask = use_fov_mask
        self.tilt_axis = tilt_axis
        self.scheduler = scheduler
        self.slice_batch_size = slice_batch_size
        self.num_workers = num_workers
        self.precision = precision
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def _build_reconstructor_and_loader(
        self,
        images: torch.Tensor,
        volume_init: torch.Tensor,
        voxel_size: float,
        scattering_model: str,
        batch_size: int,
    ) -> tuple["TomogramReconstructor", torch.utils.data.DataLoader]:
        n_tilts = images.shape[0]
        idx = torch.arange(n_tilts)
        dataset = torch.utils.data.TensorDataset(images, idx)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        model = TomogramReconstructor(
            volume_init,
            voxel_size,
            self._quaternions,
            self._translations,
            self._ctf_params,
            self._energy,
            tilt_axis=self.tilt_axis,
            lr=self.lr,
            sparsity=self.sparsity,
            taper_width=self.taper_width,
            z_taper_width=self.z_taper_width,
            use_fov_mask=self.use_fov_mask,
            scattering_model=scattering_model,
            aberration_model=self.aberration_model,
            klim=self.klim,
            alpha=self.alpha,
            scheduler=self.scheduler,
            slice_batch_size=self.slice_batch_size,
            run_dir=self.run_dir,
        )
        return model, loader

    def run(
        self,
        device: int = 0,
        callbacks: list[Any] | None = None,
    ) -> "TomogramReconstructor":
        """
        Run the full reconstruction and return the trained
        :class:`TomogramReconstructor`.

        Parameters
        ----------
        device : int
            GPU index.  Ignored when CUDA is unavailable.
        callbacks : list, optional
            Additional Lightning callbacks.

        Returns
        -------
        TomogramReconstructor
            Trained model.  Access the volume via ``model.V.detach()``.
        """
        n_tilts = len(self._images)
        nz, nxy = self._volume_init.shape[0], self._volume_init.shape[-1]
        use_gpu = torch.cuda.is_available()
        _device_str = f"GPU {device}" if use_gpu else "CPU"
        print(
            f"Starting reconstruction: {n_tilts} tilts  |  "
            f"volume {nz}×{nxy}×{nxy}  |  {self.scattering_model}  |  "
            f"{self.epochs} epochs  |  batch {self.batch_size}  |  {_device_str}"
        )
        model, loader = self._build_reconstructor_and_loader(
            self._images,
            self._volume_init,
            self._voxel_size,
            self.scattering_model,
            self.batch_size,
        )
        trainer = L.Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            devices=[device] if use_gpu else 1,
            max_epochs=self.epochs,
            precision=self.precision if use_gpu else "32",
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks or [],
        )
        trainer.fit(model, loader)
        return model

    def test_run(
        self,
        bin_factor: int = 4,
        device: int = 0,
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
        device : int
            GPU index.  Ignored when CUDA is unavailable.
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
            images_binned,
            V_b,
            voxel_size_binned,
            self.scattering_model,
            self.batch_size,
        )
        use_gpu = torch.cuda.is_available()
        trainer = L.Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            devices=[device] if use_gpu else 1,
            max_epochs=1,
            precision="32",
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks or [],
        )
        trainer.fit(model, loader)
        v = model.V.detach()
        print(
            f"Test run passed — {bin_factor}× binned, {len(self._images)} tilts, "
            f"volume {tuple(v.shape)}  |  "
            f"V min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}"
        )
        return model
