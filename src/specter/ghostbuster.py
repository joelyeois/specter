from __future__ import annotations

from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LRScheduler,
)

from .array_utils import compute_nps_2d
from .fft_tools import fft3, ifft3
from .imagegenerator import ImageGenerator
from .symmetries import apply_symmetry, get_rotation_matrices


class Ghostbuster(L.LightningModule):
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
            "LambdaLR", "CosineAnnealingWarmRestarts", "MultiplicativeLR"
        ] = "LambdaLR",
        kmask: torch.Tensor | None = None,
        nps_weight: torch.Tensor | None = None,
        learn_noise_model: bool = False,
        noise_ema_momentum: float = 0.9,
        use_ncc: bool = False,
        flipcurvature: bool = False,
        fsc_ref: torch.Tensor | None = None,
        fsc_mask: torch.Tensor | float | None = None,
        rotate_mode: Literal["real", "fourier"] = "real",
        symmetry: str | None = None,
        symmetry_batchsize: int | None = None,
        symmetry_mode: Literal["real", "fourier"] = "fourier",
        use_cpu_for_symmetry: bool = False,
    ) -> None:
        super().__init__()

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
        self.log_lrs = []
        self.log_total_loss = []
        self.log_sparsity_loss = []
        self.log_norm_loss = []
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

        # learned noise model (RELION-style): sigma^2(k) estimated from residuals
        self.learn_noise_model = learn_noise_model
        self.noise_ema_momentum = noise_ema_momentum
        self.use_ncc = use_ncc
        n = V.shape[-1]
        self.register_buffer("sigma2_k", torch.ones(n, n // 2 + 1))

        # fsc
        if fsc_mask is None:
            fsc_mask = 1
        self.fsc_mask = fsc_mask
        self.fsc_ref = fsc_ref

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
        self.flip_curvature = flipcurvature
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
            ice_model=None,
            scattering_model=self.scattering_model,
            aberration_model=self.aberration_model,
            noise_model=None,
            klim=klim,
            flip_curvature=self.flip_curvature,
            alpha=self.alpha,
        )

    def forward(self, idx: torch.Tensor | int | slice) -> torch.Tensor:
        image = self.imagegenerator(idx)
        return image

    def symmetrize(self) -> None:
        self.V.data = apply_symmetry(
            self.V.data,
            self.sym_rot_matrices,
            batchsize=self.symmetry_batchsize,
            method=self.symmetry_mode,
        )

    def reciprocal_lr_scheduler(self, *args: Any) -> float:
        return 1 / (1 + self.lr_decay * self.global_step**0.5)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        # new. Single parameter optimization only
        if self.lr is not None:
            optimizerV = AdamW([self.V], lr=self.lr)
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

        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.niter//2, eta_min=1e-6)
        lr_schedulers = []
        if self.lr is not None:
            if self.scheduler == "CosineAnnealingWarmRestarts":
                lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizerV,
                    self.num_training_steps_per_epoch(),
                    eta_min=1e-6,
                    T_mult=2,
                )
            elif self.scheduler == "MultiplicativeLR":
                lr_scheduler = ExponentialLR(optimizerV, 0.999)
            elif self.scheduler == "LambdaLR":
                lr_scheduler = LambdaLR(optimizerV, self.reciprocal_lr_scheduler)
            lr_schedulers.append(lr_scheduler)

        # new. multi-parameter optimization.
        opts = []
        if self.lr is not None:
            opts.append(optimizerV)
        if self.lr_R is not None:
            # opts.append({"optimizerR": optimizerR})
            opts.append(optimizerR)
        if self.lr_T is not None:
            # opts.append({"optimizerT": optimizerT})
            opts.append(optimizerT)
        if self.lr_D is not None:
            # opts.append({"optimizerD": optimizerD})
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

        # loss function
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
            loss = F.mse_loss(images, out)
        self.log_norm_loss.append(loss.detach().cpu())

        # sparsity loss
        if self.sparsity is not None:
            sparsity_loss = self.sparsity * torch.mean(torch.abs(self.V))
            loss = loss + sparsity_loss
            self.log_sparsity_loss.append(sparsity_loss.detach().cpu())

        # total loss
        self.log_total_loss.append(loss.detach().cpu())

        return loss, out, images

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        opts = self.optimizers()

        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        loss, out, y1 = self._common_step(batch, batch_idx)
        # self.log('train_loss', loss)
        self.log_dict(
            {"train_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Zero gradients
        for opt in opts:
            opt.zero_grad()

        # Backward
        self.manual_backward(loss)

        # Step optimizers
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
        # log lr
        if self.lr is not None:
            self.log_lrs.append(
                self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[
                    0
                ]["lr"]
            )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.kmask is not None:
            self.V.data = torch.real(
                ifft3(fft3(self.V.data, shift=True) * self.kmask, shift=True)
            )

    def on_train_epoch_end(self) -> None:
        # enforce symmetry
        if self.symmetry is not None:
            self.V.data = apply_symmetry(
                self.V.data,
                self.sym_rot_matrices,
                batchsize=self.symmetry_batchsize,
                method=self.symmetry_mode,
            )

    def num_training_steps_per_epoch(self) -> int:
        """Get number of training steps per epoch"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches

        return num_steps
