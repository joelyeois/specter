from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import mrcfile
import torch
import torch.utils.data

from ._helpers import _preprocess_particle_images
from ._reconstructor import Reconstructor
from ._run_helpers import build_trainer


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
        Total electron dose (fluence) per image in e⁻/Å².
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
    lr_decay : float, optional
        Decay rate for the ``"LambdaLR"`` schedule: multiplier =
        ``1 / (1 + lr_decay * sqrt(global_step))``. Unused by other
        schedulers. Default 0.1.
    epochs : int
        Number of training epochs.
    batch_size : int
        Dataloader batch size.
    scattering_model : str
        Wave propagation model (``"multislice"``, ``"rytov"``, ``"firstborn"``,
        ``"projection"``).
    aberration_model : str
        CTF aberration model passed to ``ImageGenerator``.
    aberration_backend : {"legacy", "torch_ctf"}, optional
        Which engine computes the CTF/aberration transfer function.
        ``"legacy"`` (default) uses ``aberrations.Aberration``; ``"torch_ctf"``
        uses ``ctf.LegacyAberrationAdapter`` (verified parity, see
        ``ImageGenerator``'s docstring). Opt-in only; not yet the default.
    lpp_params : dict[str, float], optional
        Laser-phase-plate config, in ``ctf.CTFParameters``-native units.
        Requires ``aberration_backend="torch_ctf"``; raises at construction
        time otherwise.
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
        lr_decay: float = 0.1,
        epochs: int = 5,
        batch_size: int = 3,
        scattering_model: str = "rytov",
        aberration_model: str = "holography",
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
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
        _halfset_map: dict[str, str | None] = {"0": "A", "1": "B", "all": None}
        self.halfset_label: str | None = _halfset_map[return_class]

        (
            voltage,
            pixel_size,
            alpha,
            rotations,
            translations,
            ctf_params,
            scale,
            anisomag,
            indices,
        ) = self._load_particle_parameters(cs_file, return_class, num_particles)
        images = self._load_particle_images(mrc_file, indices)

        voxel_size = float(
            pixel_size.item() if hasattr(pixel_size, "item") else pixel_size
        )
        images = _preprocess_particle_images(images, dose_per_angstrom, voxel_size)

        # preprocessed particle data (not hyperparams — not logged by job.create)
        self._images = images
        self._rotations = rotations
        self._translations = translations
        self._ctf_params = ctf_params
        self._scale = scale
        self._anisomag = anisomag
        self._voltage = float(voltage.item() if hasattr(voltage, "item") else voltage)
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
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.aberration_backend = aberration_backend
        self.lpp_params = lpp_params
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

    @staticmethod
    def _load_particle_parameters(
        cs_file: str | Path,
        return_class: Literal["0", "1", "all"],
        num_particles: int | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Extract poses, CTF parameters, and particle indices from a .cs file."""
        from ..io import extract_parameters_from_csfile

        print(f"Loading particle parameters from {Path(cs_file).name} ...")
        (
            voltage,
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
        n_loaded = len(rotations)
        voltage_val = float(voltage.item() if hasattr(voltage, "item") else voltage)
        pixel_size_val = float(
            pixel_size.item() if hasattr(pixel_size, "item") else pixel_size
        )
        print(
            f"  {n_loaded} particles  |  {voltage_val:.0f} kV  |  {pixel_size_val:.3f} Å/px"
        )
        return (
            voltage,
            pixel_size,
            alpha,
            rotations,
            translations,
            ctf_params,
            scale,
            anisomag,
            indices,
        )

    @staticmethod
    def _load_particle_images(
        mrc_file: str | Path, indices: torch.Tensor
    ) -> torch.Tensor:
        """Load the particle stack .mrc/.mrcs file, indexed to the extracted particles."""
        print(f"Loading particle stack from {Path(mrc_file).name} ...")
        with mrcfile.mmap(str(mrc_file)) as mrc:
            images = torch.as_tensor((mrc.data[indices]).copy())
        h, w = images.shape[-2], images.shape[-1]
        print(f"  {len(images)} images  |  box {h}×{w}  |  dtype {images.dtype}")
        return images

    def _build_reconstructor_and_loader(
        self,
        images: torch.Tensor,
        voxel_size: float,
        scattering_model: str,
        batch_size: int,
    ) -> tuple["Reconstructor", torch.utils.data.DataLoader]:
        from ..arrays import ball3d

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
            self._voltage,
            self.dose_per_angstrom,
            anisomag=self._anisomag,
            alpha=self._alpha,
            scale=self._scale,
            defocus_offset=torch.tensor(self.defocus_offset),
            bfactor=self.bfactor,
            scattering_model=scattering_model,
            aberration_model=self.aberration_model,
            aberration_backend=self.aberration_backend,
            lpp_params=self.lpp_params,
            lr=self.lr,
            lr_R=self.lr_R,
            lr_T=self.lr_T,
            lr_D=self.lr_D,
            scheduler=self.scheduler,
            lr_decay=self.lr_decay,
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
        device: int | Sequence[int] = 0,
        callbacks: list[Any] | None = None,
    ) -> "Reconstructor":
        """
        Run the full reconstruction and return the trained :class:`Reconstructor`.

        Parameters
        ----------
        device : int or sequence of int
            GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to train
            across multiple GPUs via Lightning DDP (``strategy="ddp"``).
            Gradients (for the volume, and for rotations/translations/defocus
            when pose refinement is enabled) are synchronised every step via
            DDP's all-reduce, so the returned model is identical whichever
            rank produced it. Ignored when CUDA is unavailable.
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

        trainer = build_trainer(use_gpu, device, self.epochs, self.precision, callbacks)
        trainer.fit(model, loader)
        return model

    def test_run(
        self,
        bin_factor: int = 8,
        device: int | Sequence[int] = 0,
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
        device : int or sequence of int
            GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to run
            across multiple GPUs via Lightning DDP. Ignored when CUDA is
            unavailable.
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
        trainer = build_trainer(use_gpu, device, 1, "32", callbacks)
        trainer.fit(model, loader)

        v = model.V.detach()
        print(
            f"Test run passed — {bin_factor}× binned, {n_particles} particles, "
            f"box {images_binned.shape[-1]}³  |  "
            f"V min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}"
        )
        return model
