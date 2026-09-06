"""
`Reconstructor`: reconstructs a 3-D volume from single-particle images with
the same forward model as `ImageGenerator`, optionally refining poses,
shifts and defocus.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mrcfile
import roma
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler

from .. import rotations
from ..imagegenerator import ImageGenerator
from ..plots import (
    HALFMAP_FSC_THRESHOLD,
    MAP_TO_MODEL_FSC_THRESHOLD,
    resolution_between,
)
from ..symmetries import apply_symmetry, get_rotation_matrices
from ._base_reconstructor import _BaseReconstructor
from ._io import (
    save_fsc_figure,
    save_halfmap_fsc_figure,
    save_plot3d_preview,
)
from ._losses import (
    mse_loss,
    ncc_loss,
    noise_weighted_loss,
    nps_weighted_loss,
    update_sigma2,
)
from ..settings import Camera, Optics, Propagation
from specter.options import RotateMode, Scheduler


class Reconstructor(_BaseReconstructor):
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
        Total electron dose (fluence) per image in e⁻/Å².
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
    propagation : Propagation, optional
        How the forward model computes the exit wave, including the
        amplitude contrast ratio. The same object the simulator that
        produced the data was built with, when it was simulated. Default
        ``Propagation()``.
    optics : Optics, optional
        The aberration engine and phase plate. Default ``Optics()``.
        `Envelopes` and `Camera` are deliberately not accepted: neither can
        be determined from a dataset, so their high-frequency loss is
        absorbed into the reconstructed volume rather than assumed.
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
        defocus_offset: torch.Tensor = torch.tensor(0.0),
        bfactor: float | torch.Tensor | None = None,
        propagation: Propagation = Propagation(),
        optics: Optics = Optics(),
        sparsity: float | None = None,
        lr: float | None = None,
        lr_R: float | None = None,
        lr_T: float | None = None,
        lr_D: float | None = None,
        lr_decay: float = 0.1,
        scheduler: Scheduler = "LambdaLR",
        kmask: torch.Tensor | None = None,
        nps_weight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        learn_noise_model: bool = False,
        noise_ema_momentum: float = 0.9,
        use_ncc: bool = False,
        fsc_ref: torch.Tensor | str | Path | None = None,
        fsc_mask: torch.Tensor | float | str | Path | None = None,
        cryosparc_ref: torch.Tensor | str | Path | None = None,
        use_2d_mask: bool = False,
        symmetry: str | None = None,
        symmetry_batchsize: int | None = None,
        symmetry_mode: RotateMode = "fourier",
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
        #: Per-epoch FSC resolutions, accumulated in memory rather than written
        #: to their own file per epoch -- see `results_summary`.
        self._epoch_resolutions: list[dict[str, Any]] = []
        #: Hyperparameters and array-save metadata from `on_fit_start`,
        #: accumulated rather than written to their own `params<suffix>.json`.
        self._extra_params: dict[str, Any] = {}
        #: Loss/lr history from `on_fit_end`, accumulated rather than written
        #: to its own `metrics<suffix>.json`.
        self._final_metrics: dict[str, Any] | None = None

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
        self.voltage = voltage
        self._register_volume(V, lr)
        self._register_ctf_params(ctf_params, defocus_offset, lr_D)
        self._register_pose_params(quaternions, translations, lr_R, lr_T)
        self._register_anisomag_and_scale(anisomag, scale, quaternions.shape[0])

        # imaging models
        self.propagation = propagation
        self.optics = optics
        self.scattering_model = propagation.scattering_model
        self._build_imagegenerator(bfactor)
        self._bind_refined_parameters()

    def _setup_optimization_state(
        self,
        lr: float | None,
        lr_R: float | None,
        lr_T: float | None,
        lr_D: float | None,
        sparsity: float | None,
        lr_decay: float,
        scheduler: Scheduler,
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
        self.scheduler = scheduler
        self._init_loss_logs()

    def _setup_symmetry(
        self,
        symmetry: str | None,
        symmetry_batchsize: int | None,
        symmetry_mode: RotateMode,
        use_cpu_for_symmetry: bool,
    ) -> None:
        """Store symmetry settings and register the symmetry rotation matrices."""
        self.symmetry = symmetry
        self.symmetry_batchsize = symmetry_batchsize
        self.symmetry_mode = symmetry_mode
        if symmetry is not None:
            sym_rot_matrices = get_rotation_matrices(symmetry)
            self.register_buffer("sym_rot_matrices", sym_rot_matrices, persistent=False)
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
        self._defocus_names: list[str] = []
        for k, v in ctf_params.items():
            v_adjusted = v + defocus_offset if k in ("dfu", "dfv") else v
            self.register_buffer(k, v_adjusted)
            self.ctf_params[k] = getattr(self, k)
            if k in ("dfu", "dfv"):
                # The un-offset value, normalised the way BaseImager normalises
                # its own CTF buffers, so :meth:`_bind_refined_parameters` can
                # rebuild the effective defocus from the *current* offset rather
                # than the one this model happened to be constructed with.
                base = torch.as_tensor(v)
                if base.ndim == 0:
                    base = base.unsqueeze(0)
                self.register_buffer(f"_{k}_base", base.clone(), persistent=False)
                self._defocus_names.append(k)
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

    def _build_imagegenerator(self, bfactor: float | torch.Tensor | None) -> None:
        """Construct the underlying ImageGenerator from the registered parameters.

        The forward model is noiseless and envelope-free: the inverse takes
        no `Envelopes` and no `Camera`, so the generator gets their defaults
        with the noise switched off.
        """
        self.imagegenerator = ImageGenerator(
            self.V,
            self.voxel_size,
            self.rotations,
            self.translations,
            self.ctf_params,
            self.voltage,
            self.dose_per_angstrom,
            anisomag=self.anisomag,
            propagation=self.propagation,
            optics=self.optics,
            camera=Camera(noise_model=None),
            bfactor=bfactor,
        )

    def _bind_refined_parameters(self) -> None:
        """
        Publish this module's refined parameters to the image generator.

        `Reconstructor` owns the refined parameters; `ImageGenerator` renders
        from them. Binding them by shared tensor identity is not durable:
        `torch.nn.Module._apply` swaps a `Parameter`'s ``.data`` in place but
        replaces a buffer with a fresh ``.to()`` copy, so a device move leaves
        the generator holding a snapshot. Gradients still reach the parameter
        through that copy, which is why the breakage is silent -- the optimiser
        steps, and the forward model keeps rendering the values it was moved
        with. On the CPU ``.to()`` is a no-op and identity survives, so this
        only ever failed where training actually happens.

        ``dfu``/``dfv`` are rebuilt rather than aliased, since the quantity the
        generator needs is derived: the base defocus plus the current
        ``defocus_offset``, less whatever midplane shift
        :meth:`~specter.imagegenerator.BaseImager._apply_defocus_shift` applied.
        Recomputing it here is what puts ``defocus_offset`` in the autograd
        graph at all -- folded in once at construction, it is a constant.

        Idempotent, and cheap enough to call every step.
        """
        gen = getattr(self, "imagegenerator", None)
        if gen is None:
            return

        gen.V = self.V
        gen.quaternions = self.rotations
        gen.translations = self.translations

        shift = getattr(gen, "_defocus_shift_angstrom", 0.0) or 0.0
        for name in getattr(self, "_defocus_names", []):
            base = getattr(self, f"_{name}_base")
            setattr(gen, name, base + self.defocus_offset - shift)

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        """Re-bind refined parameters after any device or dtype transform."""
        module = super()._apply(*args, **kwargs)
        module._bind_refined_parameters()
        return module

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
        opts, lr_schedulers = self._volume_optimizer()
        if self.lr_R is not None:
            opts.append(AdamW([self.rotations], lr=self.lr_R))
        if self.lr_T is not None:
            opts.append(AdamW([self.translations], lr=self.lr_T))
        if self.lr_D is not None:
            opts.append(AdamW([self.defocus_offset], lr=self.lr_D))
        return opts, lr_schedulers

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

        Under multi-GPU (DDP), the *logged* norm/total loss is
        gathered/averaged across ranks (see ``_gather_for_logging``) so it
        reflects the true global-batch loss -- but the ``loss`` this method
        returns (which feeds ``manual_backward``) stays the local, per-rank
        value throughout; DDP synchronises gradients on its own.
        """
        w = self.scale[idx]  # (B,)
        if self.use_ncc:
            loss = (w * ncc_loss(out, images)).mean()
        elif self.learn_noise_model:
            self.sigma2_k = update_sigma2(
                self.sigma2_k, images - out, self.noise_ema_momentum
            )
            loss = noise_weighted_loss(out, images, self.sigma2_k, w)
        elif self.nps_weight is not None:
            loss = nps_weighted_loss(out, images, self.nps_weight, w)
        else:
            mask = (
                self._project_fsc_mask_2d(idx, images.shape)
                if self.use_2d_mask
                else None
            )
            loss = mse_loss(out, images, w, mask=mask)
        # Kept on the device: a `.cpu()` here is a sync per step, and with
        # three of them the step ran 270 ms where the same work runs 160 ms
        # asynchronously. The lists are read once, when the epoch's metrics
        # are built.
        self.log_norm_loss.append(self._gather_for_logging(loss).detach())

        if self.sparsity is not None:
            sparsity_loss = self.sparsity * torch.mean(torch.abs(self.V))
            loss = loss + sparsity_loss
            self.log_sparsity_loss.append(sparsity_loss.detach())

        self.log_total_loss.append(self._gather_for_logging(loss).detach())
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
        self._bind_refined_parameters()

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
        loss, _, _ = self._common_step(batch, batch_idx)
        self._optimise(loss)
        return loss

    def _metrics_path_suffix(self) -> str:
        """Filename suffix for saved metrics/volumes, from the halfset label."""
        return f"_{self._halfset_label}" if self._halfset_label is not None else ""

    def on_fit_start(self) -> None:
        """Create the run directory and record metadata in memory.

        Rank-0-only under multi-GPU (DDP): every replica would otherwise
        race to create/write the same files. The hyperparameters here are not
        written to disk on their own -- ``results_summary()`` folds them into
        the single ``job.json`` a caller writes once this worker has exited
        (see that method's docstring for why: two halfset workers cannot
        safely share one file while training).
        """
        if not self._make_run_dir():
            return
        assert self._run_dir is not None

        saved_arrays: list[str] = []
        for name in ("nps_weight", "kmask"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                torch.save(tensor.detach().cpu(), self._run_dir / f"{name}.pt")
                saved_arrays.append(name)

        self._extra_params.update(dict(self.hparams))
        if saved_arrays:
            self._extra_params["saved_arrays"] = saved_arrays
        print(f"Run directory: {self._run_dir}")

    def on_fit_end(self) -> None:
        """Save the final reconstructed volume, FSC figure, and training metrics.

        Rank-0-only under multi-GPU (DDP): ``self.V`` is already identical
        across replicas at this point (kept in sync by DDP's gradient
        all-reduce every step), so every rank would otherwise race to write
        the same output files.
        """
        # Build metrics first (before v is computed, so they capture all
        # epochs). write=False: kept in memory for results_summary() rather
        # than written to its own metrics<suffix>.json -- see that method.
        self._final_metrics = self._save_metrics(
            include_loss_std=True, include_lr_min=True, write=False
        )

        if self._run_dir is None or not self.trainer.is_global_zero:
            return
        suffix = self._metrics_path_suffix()
        v = self._volume_cpu()
        volume_path = self._run_dir / f"volume{suffix}.mrc"
        self._save_volume(v, volume_path)
        print(f"Saved final volume → {volume_path}")

        if self.fsc_ref is not None:
            self._save_fsc_figure(
                v, suffix, self._run_dir / f"fsc{suffix}.png", label=f"final{suffix}"
            )

    def on_train_epoch_end(self) -> None:
        """Enforce symmetry, save per-epoch volume, plot3d preview, and FSC.

        Symmetrisation runs on every rank under multi-GPU (DDP) -- it mutates
        each replica's local ``V.data`` in place, and skipping it on
        non-zero ranks would desync the replicas for the next epoch. Only
        the file-writing tail below is rank-0-only.
        """
        if self.symmetry is not None:
            self.V.data = apply_symmetry(
                self.V.data,
                self.sym_rot_matrices,
                batchsize=self.symmetry_batchsize,
                method=self.symmetry_mode,
            )

        if self._run_dir is None or not self.trainer.is_global_zero:
            return

        epoch = self.current_epoch + 1
        suffix = self._metrics_path_suffix()
        v = self._volume_cpu()
        self._save_volume(v, self._run_dir / "epochs" / f"{epoch:03d}{suffix}.mrc")

        self._save_plot3d(v, suffix=suffix, epoch=epoch)
        if self.fsc_ref is not None:
            self._save_fsc_figure(
                v,
                suffix,
                self._run_dir / "epochs" / f"fsc_{epoch:03d}{suffix}.png",
                label=f"epoch {epoch}{suffix}",
            )
        self._record_map_to_model_resolutions(v, epoch)
        self._record_halfmap_resolutions(epoch)

    def results_summary(self) -> dict[str, Any]:
        """
        Everything a caller should fold into ``job.json`` after this run.

        `on_fit_start`/`on_fit_end` accumulate hyperparameters, final metrics
        and per-epoch resolutions in memory instead of writing
        ``params<suffix>.json``/``metrics<suffix>.json`` per halfset. The
        reason is a run directory is typically on NFS, where ``flock`` does
        not serialise concurrent writers -- it hangs -- so two halfset worker
        processes cannot safely share one file while training. Rather than
        work around that with one small file per epoch (the previous
        approach), each worker holds its own results in memory and a caller
        collects them once training has finished, when there is a single
        writer again: `specter.pipelines._reconstruct._run_single_halfset`
        sends this back to the orchestrator via a multiprocessing `Queue` for
        a gold-standard run; `run_reconstruction` reads it directly off the
        returned model for a single-halfset run, since that path never leaves
        the calling process.

        Returns
        -------
        dict
            Hyperparameters and ``saved_arrays`` at the top level, plus
            ``"metrics"`` (loss/lr history) and ``"resolutions"`` (per-epoch
            FSC entries) where available.

            **Not JSON-serializable as-is**: a hyperparameter may be a tensor
            (``defocus_offset`` is a 0-dim one), so a caller must pass this
            through `specter.jobs._job._serialize_value` before writing it or
            sending it between processes. `specter.jobs.Job.log` does that
            itself; the gold-standard worker has to do it explicitly, because
            pickling a tensor onto a `multiprocessing.Queue` hands over a file
            descriptor the sending process must outlive -- see
            `specter.pipelines._reconstruct._run_single_halfset`.
        """
        summary: dict[str, Any] = dict(self._extra_params)
        if self._final_metrics is not None:
            summary["metrics"] = self._final_metrics
        if self._epoch_resolutions:
            summary["resolutions"] = self._epoch_resolutions
        return summary

    def _mask_for(self, volume: torch.Tensor) -> torch.Tensor | None:
        """
        ``fsc_mask`` as a tensor, when it is one and matches ``volume``'s shape.

        Two ways there is no usable mask, neither of them an error.
        `_load_fsc_and_refs` substitutes the scalar ``1`` for an omitted mask
        so the loss-weighting path can multiply unconditionally -- that is not
        a mask to report a masked resolution against. And a ``test_run`` bins
        the volume (``bin_factor``) without binning the mask, so the shapes
        genuinely disagree; a sanity-check run must not die computing a
        diagnostic, which is the whole point of running one first.

        Parameters
        ----------
        volume : torch.Tensor
            The volume the mask would be applied to.

        Returns
        -------
        torch.Tensor or None
            The mask, or None when there isn't a usable one.
        """
        mask = self.fsc_mask
        if not isinstance(mask, torch.Tensor):
            return None
        if tuple(mask.shape) != tuple(volume.shape):
            return None
        return mask

    def _record_map_to_model_resolutions(self, v: torch.Tensor, epoch: int) -> None:
        """
        Record this epoch's map-to-model resolution, masked and unmasked.

        Skipped entirely without ``fsc_ref`` -- map-to-model is a comparison
        against a known reference, so with no reference there is nothing to
        report. The masked entry is additionally skipped without ``fsc_mask``.
        Neither is an error: a run with no reference maps is a perfectly
        ordinary run, it just has no map-to-model resolution to log.

        Parameters
        ----------
        v : torch.Tensor
            This epoch's volume, shape ``(Z, Y, X)``.
        epoch : int
            One-based epoch number just completed.
        """
        if self._run_dir is None or self.fsc_ref is None:
            return
        if tuple(self.fsc_ref.shape) != tuple(v.shape):
            # Same binning mismatch _mask_for guards against: a test_run's
            # volume is smaller than the reference it was given.
            return

        mask = self._mask_for(v)
        record: dict[str, Any] = {"epoch": epoch, "halfset": self._halfset_label}
        try:
            record["resolution_map_to_model"] = resolution_between(
                v, self.fsc_ref, self.voxel_size, MAP_TO_MODEL_FSC_THRESHOLD
            )
            record["resolution_map_to_model_masked"] = (
                resolution_between(
                    v,
                    self.fsc_ref,
                    self.voxel_size,
                    MAP_TO_MODEL_FSC_THRESHOLD,
                    mask=mask,
                )
                if mask is not None
                else None
            )
        except Exception as exc:
            print(f"[ghostbuster] map-to-model resolution skipped: {exc}")
            return

        self._epoch_resolutions.append(record)

    def _record_halfmap_resolutions(self, epoch: int) -> None:
        """
        Save this epoch's gold-standard half-map FSC, once both halves exist.

        A ``halfset="gold"`` run reconstructs A and B in two *separate* worker
        processes (see `specter.pipelines._reconstruct`), so neither holds the
        other's volume in memory and neither can compute a half-map FSC on its
        own -- which is why, before this, the gold-standard resolution only
        appeared once, after both workers had exited. Both workers do write
        ``epochs/<NNN>_<A|B>.mrc`` into the one shared run directory, so
        whichever finishes epoch N *second* finds its sibling's volume already
        on disk and computes the pair. The one that finishes first finds
        nothing and returns immediately.

        The write is claimed with ``O_CREAT | O_EXCL``, so if both workers
        somehow observe each other's volume they cannot both compute it: the
        loser of the race sees the file exists and skips.

        Parameters
        ----------
        epoch : int
            One-based epoch number just completed.
        """
        if self._run_dir is None or self._halfset_label not in ("A", "B"):
            return

        sibling = "B" if self._halfset_label == "A" else "A"
        epochs_dir = self._run_dir / "epochs"
        sibling_path = epochs_dir / f"{epoch:03d}_{sibling}.mrc"
        if not sibling_path.is_file():
            return

        # Claim the epoch by creating the figure exclusively. Using the artifact
        # itself as the claim avoids a marker file; matplotlib overwrites it below.
        figure_path = epochs_dir / f"fsc_halfmap_{epoch:03d}.png"
        try:
            os.close(os.open(figure_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            return

        try:
            own_path = epochs_dir / f"{epoch:03d}_{self._halfset_label}.mrc"
            own = torch.as_tensor(mrcfile.read(str(own_path)).copy())
            sib = torch.as_tensor(mrcfile.read(str(sibling_path)).copy())
            # Keep A first regardless of which worker got here second. The FSC
            # is symmetric, so this is for the figure's labelling, not the number.
            volume_a, volume_b = (
                (own, sib) if self._halfset_label == "A" else (sib, own)
            )
            save_halfmap_fsc_figure(
                figure_path,
                volume_a,
                volume_b,
                self.voxel_size,
                fsc_mask=self.fsc_mask,
            )
            # Computed rather than taken from the figure: plot_halfmap_fsc
            # returns the unmasked resolution even when it draws a masked
            # curve, so the masked number has to come from its own call.
            mask = self._mask_for(volume_a)
            resolution = resolution_between(
                volume_a, volume_b, self.voxel_size, HALFMAP_FSC_THRESHOLD
            )
            masked = (
                resolution_between(
                    volume_a,
                    volume_b,
                    self.voxel_size,
                    HALFMAP_FSC_THRESHOLD,
                    mask=mask,
                )
                if mask is not None
                else None
            )
            self._epoch_resolutions.append(
                {
                    "epoch": epoch,
                    "resolution_gold_standard": resolution,
                    "resolution_gold_standard_masked": masked,
                    "computed_by_halfset": self._halfset_label,
                }
            )
            print(
                f"Epoch {epoch} gold-standard resolution: {resolution}"
                + (f"  (masked {masked})" if masked is not None else "")
            )
        except Exception as exc:
            # A diagnostic must never take the run down with it -- the same
            # rule save_fsc_figure follows. Drop the claim so the empty record
            # isn't mistaken for a computed one.
            print(f"[ghostbuster] per-epoch half-map FSC skipped: {exc}")
            figure_path.unlink(missing_ok=True)

    def _save_plot3d(self, v: torch.Tensor, suffix: str, epoch: int) -> None:
        """Save a plot3d preview of the current volume. Silently skips on failure."""
        if self._run_dir is None:
            return
        save_plot3d_preview(
            self._run_dir / "epochs" / f"volume_{epoch:03d}{suffix}.png",
            v,
            title=f"Epoch {epoch}{suffix}",
        )

    def _save_fsc_figure(
        self,
        v: torch.Tensor,
        suffix: str,
        path: Path,
        label: str,
    ) -> None:
        """Compute and save an FSC figure with optional CryoSPARC reference. Silently skips on failure."""
        assert self.fsc_ref is not None, (
            "_save_fsc_figure requires self.fsc_ref to be set"
        )
        cryosparc_ref = (
            self.cryosparc_ref
            if self.cryosparc_ref is not None and self.fsc_ref is not None
            else None
        )
        save_fsc_figure(
            path,
            v,
            self.fsc_ref,
            self.voxel_size,
            label,
            fsc_mask=self.fsc_mask,
            cryosparc_ref=cryosparc_ref,
        )
