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
from ..arrays import compute_nps_2d
from ..imagegenerator import ImageGenerator
from ..symmetries import apply_symmetry, get_rotation_matrices
from ._helpers import (
    _apply_kmask_inplace,
    _build_epoch_metrics,
    _build_lr_scheduler,
    _log_current_lr,
)


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
    scale : torch.Tensor, optional
        Per-particle loss weight, shape (N,) (e.g. cryoSPARC
        ``alignments3D/alpha``). Multiplies each particle's contribution to
        the image-domain loss before batch averaging — particles with a
        small scale are down-weighted in the gradient update. ``None``
        (default) weights every particle equally.
    voltage : float
        Electron beam accelerating voltage in kV.
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
    aberration_backend : {"legacy", "torch_ctf"}, optional
        Which engine computes the CTF/aberration transfer function inside
        the underlying ``ImageGenerator``. ``"legacy"`` (default) uses
        ``aberrations.Aberration``; ``"torch_ctf"`` uses
        ``ctf.LegacyAberrationAdapter`` (verified parity, see
        ``ImageGenerator``'s own docstring). Opt-in only; not yet the
        default.
    lpp_params : dict[str, float], optional
        Laser-phase-plate config, in ``ctf.CTFParameters``-native units.
        Requires ``aberration_backend="torch_ctf"``; raises at
        construction time otherwise (via ``ImageGenerator``).
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
        voltage: float,
        dose_per_angstrom: float,
        anisomag: torch.Tensor | None = None,
        alpha: float = 0.0,
        defocus_offset: torch.Tensor = torch.tensor(0.0),
        bfactor: float | torch.Tensor | None = None,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
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
        scale: torch.Tensor | None = None,
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
                "scale",
                "fsc_ref",
                "cryosparc_ref",
                "fsc_mask",
                "run_dir",
                "halfset_label",
            ]
        )
        self._run_dir: Path | None = Path(run_dir) if run_dir is not None else None
        self._halfset_label: str | None = halfset_label

        self._setup_optimization_state(
            lr, lr_R, lr_T, lr_D, sparsity, lr_decay, scheduler
        )
        self._setup_symmetry(
            symmetry, symmetry_batchsize, symmetry_mode, use_cpu_for_symmetry
        )
        self._setup_masks(kmask, nps_weight, use_2d_mask)
        self._setup_noise_model(V, learn_noise_model, noise_ema_momentum, use_ncc)
        self._load_fsc_and_refs(fsc_ref, fsc_mask, cryosparc_ref)

        # model parameters
        self.dose_per_angstrom = dose_per_angstrom
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.voltage = voltage
        self.rotate_mode = rotate_mode
        self._register_volume(V, lr)
        self._register_ctf_params(ctf_params, defocus_offset, lr_D)
        self._register_pose_params(quaternions, translations, lr_R, lr_T)
        self._register_anisomag_and_scale(anisomag, scale, quaternions.shape[0])

        # imaging models
        self.ews_curvature_sign = ews_curvature_sign
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.aberration_backend = aberration_backend
        self.lpp_params = lpp_params
        self._build_imagegenerator(klim, bfactor)

    def _setup_optimization_state(
        self,
        lr: float | None,
        lr_R: float | None,
        lr_T: float | None,
        lr_D: float | None,
        sparsity: float | None,
        lr_decay: float,
        scheduler: str,
    ) -> None:
        """Configure manual-optimisation mode, per-parameter LRs, and loss logs."""
        # Always use manual optimization to handle masking and multiple optimizers consistently
        self.automatic_optimization = False
        if all(l_rate is None for l_rate in [lr, lr_R, lr_T, lr_D]):
            print("Non-optimization mode.")
        elif sum(l_rate is None for l_rate in [lr, lr_R, lr_T, lr_D]) == 3:
            print("Single parameter optimization mode.")
        else:
            print("Multi-parameter optimization mode.")

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

    def _setup_symmetry(
        self,
        symmetry: str | None,
        symmetry_batchsize: int | None,
        symmetry_mode: Literal["real", "fourier"],
        use_cpu_for_symmetry: bool,
    ) -> None:
        """Store symmetry settings and register the symmetry rotation matrices."""
        self.symmetry = symmetry
        self.symmetry_batchsize = symmetry_batchsize
        self.symmetry_mode = symmetry_mode
        if symmetry is not None:
            sym_rot_matrices = get_rotation_matrices(symmetry)
            self.register_buffer("sym_rot_matrices", sym_rot_matrices)
        self.use_cpu_for_symmetry = use_cpu_for_symmetry

    def _setup_masks(
        self,
        kmask: torch.Tensor | None,
        nps_weight: torch.Tensor | None,
        use_2d_mask: bool,
    ) -> None:
        """Register the Fourier k-mask and NPS-weight buffers."""
        self.register_buffer("kmask", kmask)
        self.register_buffer("nps_weight", nps_weight)
        self.use_2d_mask = use_2d_mask

    def _setup_noise_model(
        self,
        V: torch.Tensor,
        learn_noise_model: bool,
        noise_ema_momentum: float,
        use_ncc: bool,
    ) -> None:
        """Configure the learned noise model (RELION-style sigma^2(k) buffer)."""
        self.learn_noise_model = learn_noise_model
        self.noise_ema_momentum = noise_ema_momentum
        self.use_ncc = use_ncc
        n = V.shape[-1]
        self.sigma2_k: torch.Tensor
        self.register_buffer("sigma2_k", torch.ones(n, n // 2 + 1))

    def _load_fsc_and_refs(
        self,
        fsc_ref: torch.Tensor | str | Path | None,
        fsc_mask: torch.Tensor | float | str | Path | None,
        cryosparc_ref: torch.Tensor | str | Path | None,
    ) -> None:
        """Load fsc_ref/fsc_mask/cryosparc_ref from file paths, if given, and store them."""
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

        if isinstance(cryosparc_ref, (str, Path)):
            cryosparc_ref = torch.as_tensor(mrcfile.read(str(cryosparc_ref)))
        self.cryosparc_ref = cryosparc_ref

    def _register_volume(self, V: torch.Tensor, lr: float | None) -> None:
        """Register V as a trainable Parameter (lr set) or a fixed buffer."""
        if lr is None:
            self.register_buffer("V", V)
        else:
            self.V = nn.Parameter(V)

    def _register_ctf_params(
        self,
        ctf_params: dict[str, torch.Tensor],
        defocus_offset: torch.Tensor,
        lr_D: float | None,
    ) -> None:
        """Register per-particle CTF parameter buffers and the defocus offset."""
        self.ctf_params = {}
        for k, v in ctf_params.items():
            v_adjusted = v + defocus_offset if k in ("dfu", "dfv") else v
            self.register_buffer(k, v_adjusted)
            self.ctf_params[k] = getattr(self, k)
        if lr_D is None:
            self.register_buffer("defocus_offset", defocus_offset)
        else:
            self.defocus_offset = nn.Parameter(defocus_offset)

    def _register_pose_params(
        self,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        lr_R: float | None,
        lr_T: float | None,
    ) -> None:
        """Register rotations/translations as trainable Parameters or fixed buffers."""
        if lr_R is None:
            self.register_buffer("rotations", quaternions)
        else:
            self.rotations = nn.Parameter(quaternions)

        if lr_T is None:
            self.register_buffer("translations", translations)
        else:
            self.translations = nn.Parameter(translations)

    def _register_anisomag_and_scale(
        self,
        anisomag: torch.Tensor | None,
        scale: torch.Tensor | None,
        n_particles: int,
    ) -> None:
        """Register the anisotropic magnification and per-particle scale buffers."""
        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)

        if scale is None:
            scale = torch.ones(n_particles)
        self.register_buffer("scale", scale)

    def _build_imagegenerator(
        self, klim: float | None, bfactor: float | torch.Tensor | None
    ) -> None:
        """Construct the underlying ImageGenerator from the registered parameters."""
        self.imagegenerator = ImageGenerator(
            self.V,
            self.voxel_size,
            self.rotations,
            self.translations,
            self.ctf_params,
            self.voltage,
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
            aberration_backend=self.aberration_backend,
            lpp_params=self.lpp_params,
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
            lr_scheduler = _build_lr_scheduler(
                self.scheduler,
                optimizerV,
                self.lr,
                self.reciprocal_lr_scheduler,
                self.num_training_steps_per_epoch,
                self.num_training_steps,
            )
            lr_schedulers.append(lr_scheduler)

        opts: list[torch.optim.Optimizer] = []
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
        """Per-image normalized cross-correlation loss in MSE-equivalent units.

        Computes ``var(target) * (1 - NCC)`` per image.  The ``var(target)``
        factor makes the loss dimensionally equivalent to MSE: at the optimum
        (low-SNR regime) both quantities converge to the noise variance
        ``σ²``, so the same learning rate can be used without retuning.

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
            Per-image loss, shape ``(B,)``. Callers reduce over the batch
            (optionally weighting by a per-particle scale first).

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
        return var_t * (1.0 - ncc)

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

    def _noise_weighted_loss(
        self, out: torch.Tensor, images: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """RELION-style loss: weight residuals by 1/sigma^2(k), sigma^2(k)
        estimated from residuals via an EMA (E-step, no_grad); gradient flows
        only through the residuals (M-step)."""
        images_f = torch.fft.rfft2(images)
        out_f = torch.fft.rfft2(out)
        H, W = images.shape[-2:]
        self._update_sigma2(images - out)
        residual = (images_f - out_f).abs() ** 2 / self.sigma2_k.detach()
        return torch.mean(w[:, None, None] * residual) / (H * W)

    def _nps_weighted_loss(
        self, out: torch.Tensor, images: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """MSE weighted by the noise power spectrum in Fourier space."""
        images_f = torch.fft.rfft2(images)
        out_f = torch.fft.rfft2(out)
        H, W = images.shape[-2:]
        # Divide by H*W so that a flat (normalised) NPS weight gives the
        # same loss magnitude as real-space MSE (Parseval equivalence).
        residual = self.nps_weight * (images_f - out_f).abs() ** 2
        return torch.mean(w[:, None, None] * residual) / (H * W)

    def _mse_loss(
        self,
        out: torch.Tensor,
        images: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Plain real-space MSE, optionally weighted by a projected 2D FSC mask."""
        mse = F.mse_loss(images, out, reduction="none")
        if self.use_2d_mask:
            mse = mse * self._project_fsc_mask_2d(idx, images.shape)
        return (w[:, None, None] * mse).mean()

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

        Notes
        -----
        Each branch is weighted by the per-particle ``self.scale[idx]``
        before the final batch mean, so a small scale down-weights that
        particle's contribution to the gradient update. Weighting is applied
        as a plain multiply-then-mean (not normalised by the sum of weights
        in the batch), since scale values are meaningful relative to each
        other across the whole dataset, not just within one batch.
        """
        w = self.scale[idx]  # (B,)
        if self.use_ncc:
            loss = (w * self._ncc_loss(out, images)).mean()
        elif self.learn_noise_model:
            loss = self._noise_weighted_loss(out, images, w)
        elif self.nps_weight is not None:
            loss = self._nps_weighted_loss(out, images, w)
        else:
            loss = self._mse_loss(out, images, idx, w)
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
            R = roma.unitquat_to_rotmat(Q)
            T = rotations.translations_angstrom_to_torch(
                T, self.imagegenerator.nxy, self.imagegenerator.pixel_size
            )
            theta = rotations.build_affine_matrix(R, T)
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
                    if isinstance(s, ReduceLROnPlateau):
                        raise TypeError(
                            "ReduceLROnPlateau is not supported by this manual "
                            "scheduler step loop; _build_lr_scheduler never "
                            "constructs one."
                        )
                    s.step()
        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Record the current learning rate before each batch."""
        _log_current_lr(self.trainer, self.lr, self.log_lrs)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Apply the Fourier-space k-mask to V after each gradient update."""
        _apply_kmask_inplace(self.V, self.kmask)

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
        meta = _build_epoch_metrics(
            self.log_total_loss,
            self.log_norm_loss,
            self.log_sparsity_loss,
            self.log_lrs,
            self.current_epoch,
            include_loss_std=True,
            include_lr_min=True,
        )
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

            from ..plots import plot3d

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

            from ..plots import plot_map_to_model_fsc

            # Move references to volume's device for GPU FSC computation.
            device = v.device
            fsc_ref = (
                self.fsc_ref.detach().to(device).float()
                if isinstance(self.fsc_ref, torch.Tensor)
                else self.fsc_ref
            )
            assert (
                fsc_ref is not None
            ), "_save_fsc_figure requires self.fsc_ref to be set"
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
        assert self.trainer.train_dataloader is not None
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
