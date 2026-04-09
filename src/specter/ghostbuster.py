from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Literal

import lightning as L
import mrcfile
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
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
        tag: str = "untagged",
        output_dir: str | Path | None = None,
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
                "kmask",
                "nps_weight",
                "fsc_ref",
                "fsc_mask",
                "output_dir",
                "halfset_label",
            ]
        )
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._run_dir: Path | None = None
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

    def on_fit_start(self) -> None:
        if self._output_dir is None:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self.hparams.get("tag", "untagged")
        self._run_dir = self._output_dir / f"{timestamp}__{tag}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "epochs").mkdir(exist_ok=True)

        saved_arrays: list[str] = []
        for name in ("nps_weight", "kmask"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                torch.save(tensor.detach().cpu(), self._run_dir / f"{name}.pt")
                saved_arrays.append(name)

        meta: dict[str, Any] = {"timestamp": timestamp}
        meta.update(dict(self.hparams))
        if saved_arrays:
            meta["saved_arrays"] = saved_arrays
        (self._run_dir / "params.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        print(f"Run directory: {self._run_dir}")

    def on_train_epoch_end(self) -> None:
        # enforce symmetry
        if self.symmetry is not None:
            self.V.data = apply_symmetry(
                self.V.data,
                self.sym_rot_matrices,
                batchsize=self.symmetry_batchsize,
                method=self.symmetry_mode,
            )

        # save per-epoch volume
        if self._run_dir is not None:
            epoch = self.current_epoch + 1
            suffix = (
                f"_{self._halfset_label}" if self._halfset_label is not None else ""
            )
            path = self._run_dir / "epochs" / f"{epoch:03d}{suffix}.mrc"
            v = self.V.detach().cpu().float().numpy()
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(v)

    def num_training_steps_per_epoch(self) -> int:
        """Get number of training steps per epoch"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches

        return num_steps


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

    runs = sorted(Path(base_dir).glob("*/params.json"))
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


def _split_halfsets(n: int) -> torch.Tensor:
    """Assign N particles to two halves by even/odd index.

    Parameters
    ----------
    n : int
        Total number of particles.

    Returns
    -------
    torch.Tensor
        1-D integer label tensor of length ``n`` with values 0 (half A) or
        1 (half B), alternating as ``[0, 1, 0, 1, ...]``.
    """
    labels = torch.zeros(n, dtype=torch.long)
    labels[1::2] = 1
    return labels


def _halfset_worker(
    rank: int,
    run_dir: Path,
    all_images: list[torch.Tensor],
    all_ghost_kwargs: list[dict[str, Any]],
    batch_size: int,
    max_epochs: int,
    gpu_ids: list[int | None],
) -> None:
    """Worker function executed in each halfset subprocess.

    Parameters
    ----------
    rank : int
        0 for half A, 1 for half B.
    run_dir : Path
        Shared output directory created by :func:`run_halfsets`.
    all_images : list of torch.Tensor
        ``[images_A, images_B]`` — experimental images for each half.
    all_ghost_kwargs : list of dict
        ``[kwargs_A, kwargs_B]`` — constructor kwargs for each half's
        :class:`Ghostbuster`, with per-particle tensors already sliced.
    batch_size : int
        Dataloader batch size.
    max_epochs : int
        Number of training epochs.
    gpu_ids : list of int or None
        ``[gpu_A, gpu_B]`` — GPU index for each half, or ``None`` for CPU.
    """
    label = "AB"[rank]
    images = all_images[rank]
    ghost_kwargs = all_ghost_kwargs[rank]
    gpu_id = gpu_ids[rank]

    # Restrict this process to its assigned GPU so Lightning and CUDA ops
    # don't accidentally spill onto the other device.
    use_gpu = gpu_id is not None and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(gpu_id)

    torch.set_float32_matmul_precision("high")

    model = Ghostbuster(**ghost_kwargs, halfset_label=label, output_dir=None)
    # Pre-assign run_dir so on_fit_start (which returns early when
    # output_dir is None) does not create a new timestamped directory.
    model._run_dir = run_dir

    n = images.shape[0]
    dataset = torch.utils.data.TensorDataset(images, torch.arange(n))
    # num_workers=0: this function runs inside mp.spawn subprocesses;
    # spawning DataLoader workers within a spawned process causes nested
    # multiprocessing overhead. The dataset is already in RAM so there
    # is no I/O to parallelise anyway.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    trainer = L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=[gpu_id] if use_gpu else 1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, dataloader)

    v = model.V.detach().cpu().float().numpy()
    out_path = run_dir / f"vol_{label}.mrc"
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(v)
    print(f"[half {label}] saved final volume → {out_path}")


def run_halfsets(
    images: torch.Tensor,
    V_init: torch.Tensor,
    voxel_size: float,
    quaternions: torch.Tensor,
    translations: torch.Tensor,
    ctf_params: dict[str, torch.Tensor],
    energy: float,
    dose_per_angstrom: float,
    halfset_labels: torch.Tensor | None = None,
    batch_size: int = 3,
    max_epochs: int = 5,
    output_dir: str | Path = "output/runs",
    gpus: list[int] | None = None,
    tag: str | None = None,
    **ghostbuster_kwargs: Any,
) -> Path:
    """Run independent halfset reconstructions and save into a single run folder.

    Splits ``images``, ``quaternions``, ``translations``, and ``ctf_params``
    into two halves and runs a full :class:`Ghostbuster` reconstruction on each
    independently.  When ``gpus`` contains two entries, both halves run in
    parallel via :func:`torch.multiprocessing.spawn` on those specific devices;
    when it contains one entry, both halves run sequentially on that device.

    Output layout::

        <output_dir>/<timestamp>__<tag>/
            params.json          ← combined run metadata
            epochs/
                001_A.mrc        ← half-A volume after epoch 1
                001_B.mrc
                002_A.mrc
                002_B.mrc
                ...
            vol_A.mrc            ← final reconstructed volume, half A
            vol_B.mrc            ← final reconstructed volume, half B

    Parameters
    ----------
    images : torch.Tensor
        Experimental images, shape ``(N, H, W)``.
    V_init : torch.Tensor
        Initial 3-D volume estimate, shape ``(Z, X, Y)``.  Copied for each half.
    voxel_size : float
        Voxel size in Angstroms.
    quaternions : torch.Tensor
        Per-particle rotation quaternions, shape ``(N, 4)``.
    translations : torch.Tensor
        Per-particle translations, shape ``(N, 2)`` or ``(N, 3)``.
    ctf_params : dict[str, torch.Tensor]
        Per-particle CTF parameters; each value has leading dimension ``N``.
    energy : float
        Electron energy in keV.
    dose_per_angstrom : float
        Electron dose in e⁻/Å².
    halfset_labels : torch.Tensor, optional
        1-D integer tensor of length ``N`` with values ``0`` (half A) or
        ``1`` (half B), e.g. ``[0, 0, 1, 0, 1, ...]``.  Matches the
        ``alignments3D/split`` field returned by
        :func:`~specter.cryosparc_utils.extract_parameters_from_csfile`.
        Defaults to alternating even/odd assignment.
    batch_size : int
        Dataloader batch size for each half.
    max_epochs : int
        Number of training epochs.
    output_dir : str or Path
        Root directory under which the run folder is created.
    gpus : list of int, optional
        GPU indices to use.  ``[0, 2]`` runs half A on ``cuda:0`` and half B on
        ``cuda:2`` in parallel.  ``[1]`` runs both halves sequentially on
        ``cuda:1``.  Defaults to ``[0]`` when CUDA is available, CPU otherwise.
    tag : str, optional
        Human-readable tag appended to the run folder name.  ``None`` omits the
        tag suffix.
    **ghostbuster_kwargs
        Additional keyword arguments forwarded to :class:`Ghostbuster` unchanged
        (e.g. ``lr``, ``scattering_model``, ``symmetry``).

    Returns
    -------
    Path
        Path to the run directory.
    """
    # resolve GPU assignment
    if gpus is None:
        gpus = [0] if torch.cuda.is_available() else []

    output_dir = Path(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}__{tag}" if tag is not None else timestamp
    run_dir = output_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "epochs").mkdir(exist_ok=True)

    # determine halfset splits
    n = quaternions.shape[0]
    labels = halfset_labels if halfset_labels is not None else _split_halfsets(n)
    idx_A = torch.where(labels == 0)[0]
    idx_B = torch.where(labels == 1)[0]

    # write combined params.json (exclude raw tensors)
    meta: dict[str, Any] = {
        "timestamp": timestamp,
        "tag": tag,
        "gpus": gpus,
        "n_particles_total": n,
        "n_particles_A": int(len(idx_A)),
        "n_particles_B": int(len(idx_B)),
        "voxel_size": voxel_size,
        "energy": energy,
        "dose_per_angstrom": dose_per_angstrom,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
    }
    meta.update(
        {k: v for k, v in ghostbuster_kwargs.items() if not isinstance(v, torch.Tensor)}
    )
    (run_dir / "params.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Run directory: {run_dir}")

    def _slice_kwargs(idx: torch.Tensor) -> dict[str, Any]:
        half_ctf = {k: v[idx].cpu() for k, v in ctf_params.items()}
        return dict(
            V=V_init.cpu().clone(),
            voxel_size=voxel_size,
            quaternions=quaternions[idx].cpu(),
            translations=translations[idx].cpu(),
            ctf_params=half_ctf,
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            tag=tag,
            **ghostbuster_kwargs,
        )

    all_ghost_kwargs = [_slice_kwargs(idx_A), _slice_kwargs(idx_B)]
    all_images = [images[idx_A].cpu(), images[idx_B].cpu()]

    if len(gpus) >= 2:
        gpu_ids: list[int | None] = [gpus[0], gpus[1]]
        mp.spawn(
            _halfset_worker,
            args=(
                run_dir,
                all_images,
                all_ghost_kwargs,
                batch_size,
                max_epochs,
                gpu_ids,
            ),
            nprocs=2,
            join=True,
        )
    else:
        gpu_id: int | None = gpus[0] if gpus else None
        gpu_ids_seq: list[int | None] = [gpu_id, gpu_id]
        for rank in range(2):
            _halfset_worker(
                rank,
                run_dir,
                all_images,
                all_ghost_kwargs,
                batch_size,
                max_epochs,
                gpu_ids_seq,
            )

    return run_dir
