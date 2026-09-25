"""
`Ghostbuster`: the end-to-end single-particle reconstruction pipeline, from
a CryoSPARC particle set to a trained `Reconstructor`.
"""

from __future__ import annotations

from ..progress import console

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Sequence

import mrcfile
import torch
import torch.utils.data

from ..settings import Optics, Propagation
from ._helpers import _images_to_counts
from ._reconstructor import Reconstructor
from ._pipeline_base import _GhostbusterBase
from specter.options import ImageUnits, RotateMode, Scheduler


class Ghostbuster(_GhostbusterBase):
    """
    End-to-end reconstruction pipeline for cryo-EM/cryo-ET.

    .. note::
        ``halfset`` and ``cryosparc_ref`` are excluded from the job
        parameter log (see :attr:`_job_log_exclude`) because they differ
        between halfsets. ``halfset`` encodes halfset identity in the
        output filenames (``volume_A.mrc`` / ``volume_B.mrc``), and ``cryosparc_ref``
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
    alpha : float, optional
        Amplitude contrast ratio in ``[0, 1]``, overriding the ``.cs`` file's
        ``ctf/amp_contrast``. Default None uses the ``.cs`` file's value.
    lr : float, optional
        Learning rate for the volume. ``None`` disables volume optimisation.
    lr_R : float, optional
        Learning rate for rotations.
    lr_T : float, optional
        Learning rate for translations.
    lr_D : float, optional
        Learning rate for defocus offset.
    defocus_offset : float, optional
        Initial defocus offset in Å added to all particles' dfu and dfv.
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
    batchsize : int
        Dataloader batch size.
    propagation : Propagation, optional
        How the forward model computes the exit wave. Default
        ``Propagation(scattering_model="rytov")``. Its ``alpha`` is replaced
        by the amplitude contrast recorded in the ``.cs`` file, since that is
        data rather than a modelling choice, or by `alpha` when given.
    optics : Optics, optional
        The aberration engine and phase plate. Default ``Optics()``.
    symmetry : str, optional
        Point-group symmetry to enforce (e.g. ``"C3"``, ``"I1"``).
    symmetry_batchsize : int, optional
        Batch size used when applying symmetry to the volume.
    symmetry_mode : {"real", "fourier"}
        Domain in which symmetry is applied.
    sparsity : float, optional
        L1 regularisation weight on V.
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
    n_particles : int, optional
        Use only the first ``n_particles`` particles. Defaults to all.
    halfset : {"A", "B", "all"}
        Gold-standard half-set to extract from the ``.cs`` file.
    run_dir : str or Path, optional
        Directory for all job outputs. Injected automatically by
        :meth:`~specter.jobs.Job.create` when used inside a ``Job`` context.
    image_units : {"normalized", "counts"}, optional
        What the particle stack's values are. ``"normalized"`` (the default)
        is a CryoSPARC/RELION stack with zero mean, unit variance and
        inverted contrast; it is sign-flipped and mapped back to counts as
        ``sqrt(N) * x + N``, with ``N = dose_per_angstrom * pixel_size**2``.
        ``"counts"`` is electrons per pixel, used as is, such as a specter
        stack written with ``normalize_particles=False``.
    """

    _job_log_exclude: tuple[str, ...] = ("halfset", "cryosparc_ref")

    def __init__(
        self,
        cs_file: str | Path,
        mrc_file: str | Path,
        dose_per_angstrom: float,
        address_by_blob_idx: bool = False,
        alpha: float | None = None,
        lr: float | None = None,
        lr_R: float | None = None,
        lr_T: float | None = None,
        lr_D: float | None = None,
        defocus_offset: float = 0.0,
        bfactor: float | None = None,
        scheduler: Scheduler = "LambdaLR",
        lr_decay: float = 0.1,
        epochs: int = 5,
        batchsize: int = 3,
        propagation: Propagation = Propagation(scattering_model="rytov"),
        optics: Optics = Optics(),
        symmetry: str | None = None,
        symmetry_batchsize: int | None = None,
        symmetry_mode: RotateMode = "fourier",
        sparsity: float | None = None,
        nps_weight: torch.Tensor | None = None,
        learn_noise_model: bool = False,
        use_ncc: bool = False,
        fsc_ref: torch.Tensor | str | Path | None = None,
        fsc_mask: torch.Tensor | float | str | Path | None = None,
        cryosparc_ref: torch.Tensor | str | Path | None = None,
        use_2d_mask: bool = False,
        precision: str = "16-mixed",
        num_workers: int = 0,
        n_particles: int | None = None,
        halfset: Literal["A", "B", "all"] = "all",
        run_dir: str | Path | None = None,
        image_units: ImageUnits = "normalized",
    ) -> None:
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            # Checked before the particle stack is read, not after.
            raise ValueError(f"alpha={alpha} must be in [0, 1]")
        self.halfset_label: str | None = halfset if halfset != "all" else None

        (
            voltage,
            pixel_size,
            alpha_cs,
            rotations,
            translations,
            ctf_params,
            scale,
            anisomag,
            indices,
        ) = self._load_particle_parameters(cs_file, halfset, n_particles)
        images = self._load_particle_images(
            cs_file, indices, mrc_file, address_by_blob_idx
        )

        voxel_size = float(
            pixel_size.item() if hasattr(pixel_size, "item") else pixel_size
        )
        images = _images_to_counts(
            images,
            image_units,
            dose_per_angstrom * voxel_size**2,
            flip_contrast=image_units == "normalized",
        )

        # preprocessed particle data (not hyperparams — not logged by job.create)
        self._images = images
        self._rotations = rotations
        self._translations = translations
        self._ctf_params = ctf_params
        self._scale = scale
        self._anisomag = anisomag
        self._voltage = float(voltage.item() if hasattr(voltage, "item") else voltage)
        self._voxel_size = voxel_size
        self._alpha = float(alpha_cs.item() if hasattr(alpha_cs, "item") else alpha_cs)
        if alpha is not None:
            console.print(
                f"  amplitude contrast {alpha} (overrides the .cs file's {self._alpha:g})"
            )
            self._alpha = float(alpha)

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
        self.batchsize = batchsize
        # The amplitude contrast is data, read from the .cs file above unless
        # `alpha` overrides it.
        self.propagation = replace(propagation, alpha=self._alpha)
        self.optics = optics
        self.symmetry = symmetry
        self.symmetry_batchsize = symmetry_batchsize
        self.symmetry_mode = symmetry_mode
        self.sparsity = sparsity
        self.nps_weight = nps_weight
        self.learn_noise_model = learn_noise_model
        self.use_ncc = use_ncc
        self.fsc_ref = fsc_ref
        self.fsc_mask = fsc_mask
        self.cryosparc_ref = cryosparc_ref
        self.use_2d_mask = use_2d_mask
        self.precision = precision
        self.num_workers = num_workers
        self.n_particles = n_particles
        self.dose_per_angstrom = dose_per_angstrom
        self.run_dir = Path(run_dir) if run_dir is not None else None

    @staticmethod
    def _load_particle_parameters(
        cs_file: str | Path,
        halfset: Literal["A", "B", "all"],
        n_particles: int | None,
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

        console.print(f"Loading particle parameters from {Path(cs_file).name} ...")
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
            str(cs_file), halfset=halfset, n_particles=n_particles
        )
        n_loaded = len(rotations)
        voltage_val = float(voltage.item() if hasattr(voltage, "item") else voltage)
        pixel_size_val = float(
            pixel_size.item() if hasattr(pixel_size, "item") else pixel_size
        )
        console.print(
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
        cs_file: str | Path,
        indices: torch.Tensor,
        mrc_file: str | Path,
        address_by_blob_idx: bool,
    ) -> torch.Tensor:
        """
        Load each extracted particle's image from the stack.

        Row ``i`` of the ``.cs`` file pairs with slice ``i`` of the stack. That
        is the contract, because it is the one that survives the pair being
        copied off the machine that produced it, and it is checked rather than
        assumed: CryoSPARC's ``blob/idx`` says which slice each row's image
        really is, and a Restack Particles job does not write its stack in row
        order. Reading one as if it did pairs every pose with a different
        particle's image and says nothing about it, so a stack the ``.cs``
        contradicts is refused. ``address_by_blob_idx`` reads such a stack
        where the ``.cs`` says the images are, instead of refusing.
        """
        from ..io import particle_image_refs, read_particle_images, row_order_conflict

        rows = [int(i) for i in indices.flatten().tolist()]

        if address_by_blob_idx:
            refs = particle_image_refs(cs_file, stack_path=mrc_file)
            console.print(
                f"Loading particle stack from {Path(mrc_file).name} "
                "at the .cs file's blob/idx ..."
            )
            images = read_particle_images([refs[i] for i in rows])
        else:
            conflict = row_order_conflict(cs_file, rows)
            if conflict is not None:
                row, idx, stack = conflict
                raise ValueError(
                    f"{Path(cs_file).name} says its images are not in row order: "
                    f"row {row} is slice {idx} of {stack}, not slice {row}. Reading "
                    f"{Path(mrc_file).name} by row would pair every pose with "
                    "another particle's image. Point mrc_file at a stack written in "
                    "this file's row order (`specter export particles` writes one), "
                    "or pass address_by_blob_idx to read this one where the .cs "
                    "says the images are."
                )
            console.print(f"Loading particle stack from {Path(mrc_file).name} ...")
            with mrcfile.mmap(str(mrc_file)) as mrc:
                images = torch.as_tensor(mrc.data[rows].copy())

        h, w = images.shape[-2], images.shape[-1]
        console.print(
            f"  {len(images)} images  |  box {h}×{w}  |  dtype {images.dtype}"
        )
        return images

    def _build_reconstructor_and_loader(
        self,
        images: torch.Tensor,
        voxel_size: float,
        batchsize: int,
    ) -> tuple["Reconstructor", torch.utils.data.DataLoader]:
        from ..arrays import ball3d

        n = images.shape[-1]
        kmask = ball3d(n, n)
        volume_init = torch.zeros(n, n, n)

        idx = torch.arange(len(images))
        dataset = torch.utils.data.TensorDataset(images, idx)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, shuffle=True, num_workers=self.num_workers
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
            scale=self._scale,
            defocus_offset=torch.tensor(self.defocus_offset),
            bfactor=self.bfactor,
            propagation=self.propagation,
            optics=self.optics,
            lr=self.lr,
            lr_R=self.lr_R,
            lr_T=self.lr_T,
            lr_D=self.lr_D,
            scheduler=self.scheduler,
            lr_decay=self.lr_decay,
            kmask=kmask,
            nps_weight=self.nps_weight,
            learn_noise_model=self.learn_noise_model,
            use_ncc=self.use_ncc,
            sparsity=self.sparsity,
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
        device: int | Sequence[int] | str = 0,
        callbacks: list[Any] | None = None,
    ) -> "Reconstructor":
        """
        Run the full reconstruction and return the trained :class:`Reconstructor`.

        Parameters
        ----------
        device : int or sequence of int or {"cpu"}
            GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to train
            across multiple GPUs via Lightning DDP (``strategy="ddp"``).
            Gradients (for the volume, and for rotations/translations/defocus
            when pose refinement is enabled) are synchronised every step via
            DDP's all-reduce, so the returned model is identical whichever
            rank produced it. Pass ``"cpu"`` to force CPU
            training; any other value uses the GPU when CUDA is
            available and falls back to the CPU when it is not.
        callbacks : list, optional
            Additional Lightning callbacks passed to the ``Trainer``.

        Returns
        -------
        Reconstructor
            The trained model. Access the volume via ``model.V.detach()``.
        """
        console.print(
            f"Starting reconstruction: {len(self._images)} particles  |  "
            f"box {self._images.shape[-1]}³  |  "
            f"{self.propagation.scattering_model}  |  {self.epochs} epochs  |  "
            f"batch {self.batchsize}  |  {self._device_label(device)}"
        )
        model, loader = self._build_reconstructor_and_loader(
            self._images, self._voxel_size, self.batchsize
        )
        return self._fit(model, loader, device, self.epochs, self.precision, callbacks)

    def test_run(
        self,
        bin_factor: int = 8,
        device: int | Sequence[int] | str = 0,
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
        device : int or sequence of int or {"cpu"}
            GPU index, or a sequence of GPU indices (e.g. ``[0, 1]``) to run
            across multiple GPUs via Lightning DDP, or ``"cpu"`` to force
            CPU training.
        callbacks : list, optional
            Additional Lightning callbacks passed to the ``Trainer``.

        Returns
        -------
        Reconstructor
            The trained model after one epoch.
        """
        n_particles = len(self._images)
        console.print(
            f"Test run: {n_particles} particles  |  {bin_factor}× binned  |  1 epoch"
        )
        images_binned, voxel_size_binned = self._bin_images(bin_factor)
        model, loader = self._build_reconstructor_and_loader(
            images_binned, voxel_size_binned, self.batchsize
        )
        model = self._fit(model, loader, device, 1, "32", callbacks)
        self._report_test_run(
            model,
            f"{bin_factor}× binned, {n_particles} particles, "
            f"box {images_binned.shape[-1]}³",
        )
        return model
