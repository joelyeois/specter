from __future__ import annotations

from typing import Any, Literal

import torch

from specter import logger

from ._base import BaseImager, compute_nz, pad_volume
from ..ice import IceBank, RandomIcemaker, blend_ice_into_volume, resolve_icemaker
from ..progress import status
from ..scattering import IterativeScattering
from ..specimen import MicrographSpecimenGenerator


class MicrographGenerator(BaseImager):
    """
    Generates large micrographs by processing a full specimen volume.

    The volume is either supplied directly (``volume``) or assembled internally
    via ``MicrographSpecimenGenerator`` from a ``scattering_potential`` template with
    optional crowding and ice.  Scattering is performed slice-by-slice using
    ``IterativeScattering``.

    Parameters
    ----------
    scattering_potential : torch.Tensor or None
        Template potential (Z, Y, X) used by ``MicrographSpecimenGenerator`` to build
        the specimen volume.  Must be ``None`` when ``volume`` is provided.
    micrograph_size : int or tuple[int, int]
        Output image size in pixels (must be square).
    pixel_size : float
        Pixel size in Å.
    ctf_params : dict[str, torch.Tensor]
        CTF parameters; each value is a 1-D tensor of length n.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per micrograph in e⁻/Å². Scalar, or a
        1-D tensor of length n giving a separate dose for each micrograph.
    volume : torch.Tensor, optional
        Pre-assembled specimen volume of shape (1, Z, Y, X) -- e.g. the
        output of
        :func:`~specter.pipelines.build_tomogram_generator`/`specter build
        tomogram`.  When provided,
        ``scattering_potential`` and crowding parameters are ignored, but
        ``ice_model``/``icemaker`` are still honored: if either is set, ice
        is generated to match ``volume``'s own size and voxel size and blended
        in wherever ``volume`` has little existing scattering potential (same
        masking rule as ``ImageGenerator``'s ``solvate()``), once, at
        construction time. ``ice_thickness`` is ignored in this path since
        the volume's Z extent is fixed by ``volume`` itself.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices, shape (n, 2, 2).
    ice_model : str, optional
        Ice generation algorithm: ``'gd'`` (samples from the pre-generated
        :class:`~specter.ice.IceBank` cache) or ``'random'`` (instant, cheap
        :class:`~specter.ice.RandomIcemaker` placement). Used by
        ``MicrographSpecimenGenerator`` when ``scattering_potential`` is given, or
        blended directly into ``volume`` when ``volume`` is given (see above).
        Ignored when ``icemaker`` is provided.
    ice_thickness : float, optional
        Ice thickness in Å passed to ``MicrographSpecimenGenerator``. Ignored when
        ``volume`` is given.
    ice_cache_dir : str, optional
        Directory of cached ice configs for ``ice_model='gd'`` (see
        :func:`specter.ice.build_ice_cache`). Defaults to the bundled
        ``ice_data/ice_cache``. Ignored for other ``ice_model`` values or
        when ``icemaker`` is provided.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across multiple generator
        instances. When supplied, ``ice_model`` and ``ice_cache_dir`` are
        both ignored. Honored both when ``scattering_potential`` is given
        (forwarded to ``MicrographSpecimenGenerator``) and when ``volume`` is given
        (blended directly into ``volume``, see above).
    ice_relax_steps : int, optional
        Forwarded to :meth:`~specter.ice.IceBank.generate_big_ice` when
        ``ice_model='gd'`` (or an ``IceBank`` ``icemaker``): number of local
        MLBOP relaxation steps used to heal tile seams. Default 0 (no
        relaxation). Ignored for ``RandomIcemaker``.
    crowd_min_distance : float, optional
        Minimum inter-particle distance in Å for crowding.
    crowd_max_distance_z : float, optional
        Maximum Z range for crowding placement in Å.
    water_air_interface : bool, optional
        Simulate water-air interface in crowding and ice. Default True.
    scattering_model : str, optional
        Scattering model passed to ``IterativeScattering``. Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal-space frequency limit.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.0.
    pad_fft : bool, optional
        Whether to XY-pad the volume for FFT antialiasing. Default False.
    chunk_size : int, optional
        Chunk size for ``MicrographSpecimenGenerator`` parallel processing.
    move_to_cpu : bool, optional
        Move the assembled volume to CPU after generation to save GPU memory.
        Default True.
    detector_model : str, optional
        Detector MTF model ('k3_300kv', 'k3_200kv', 'perfect', None).
    slice_batchsize : int, optional
        Number of Z slices propagated together in ``IterativeScattering``.
        Default 1.
    progressbars : bool, optional
        Show progress bars. Default True.
    verbose : bool, optional
        Emit debug-level log messages. Default True.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Default 0.0.
    n_frames : int, optional
        Number of detector frames to simulate. Default None.
    potential_scale : float or torch.Tensor, optional
        Multiplier applied to the potential before scattering. Default 1.0.
    save_clean_exitwaves : bool, optional
        Save exit waves computed without ice (requires ``scattering_potential``
        path). Default False.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    convergence_angle : float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. Default None (envelope disabled).
    cc : float, optional
        Chromatic aberration coefficient in Å, used for the Cc
        (temporal coherence) envelope. Default None (envelope disabled).
    energy_spread : float, optional
        FWHM of the beam energy spread in eV, used by the Cc envelope.
        Default 0.7.
    deltaV_V : float, optional
        Relative high-voltage instability, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I : float, optional
        Relative objective-lens current instability, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope : bool, optional
        Whether to apply the Grant & Grigorieff (2015) cumulative-dose
        envelope, using ``dose_per_angstrom``. Default False.
    """

    def __init__(
        self,
        scattering_potential: torch.Tensor | None,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any],
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        volume: torch.Tensor | None = None,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        ice_cache_dir: str | None = None,
        icemaker: IceBank | RandomIcemaker | None = None,
        ice_relax_steps: int = 0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        water_air_interface: bool = True,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str | None = "poisson",
        klim: float | None = None,
        alpha: float = 0.0,
        pad_fft: bool = False,
        chunk_size: int | None = None,
        move_to_cpu: bool = True,
        detector_model: str | None = None,
        slice_batchsize: int = 1,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        n_frames: int | None = None,
        potential_scale: float | torch.Tensor = 1.0,
        save_clean_exitwaves: bool = False,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        if isinstance(micrograph_size, int):
            nxy = micrograph_size
        elif (
            isinstance(micrograph_size, (tuple, list))
            and micrograph_size[0] == micrograph_size[1]
        ):
            nxy = micrograph_size[0]
        else:
            raise ValueError("micrograph_size must have same dimensions in x and y.")

        self.pad_fft = pad_fft
        self.pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy

        if volume is not None:
            self.nz = volume.shape[1]
        elif scattering_potential is not None:
            self.nz = compute_nz(
                scattering_potential.shape[0], ice_thickness, pixel_size
            )
        else:
            raise ValueError(
                "Either 'volume' or 'scattering_potential' must be provided."
            )
        self.ice_thickness = self.nz * pixel_size

        super().__init__(
            pixel_size=pixel_size,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            nxy=nxy,
            nz=self.nz,
            pad_nxy=self.pad_nxy,
            aberration_model=aberration_model,
            noise_model=noise_model,
            alpha=alpha,
            detector_model=detector_model,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=progressbars,
            verbose=verbose,
            coincidence_radius=coincidence_radius,
            n_frames=n_frames,
            potential_scale=potential_scale,
            bfactor=bfactor,
            convergence_angle=convergence_angle,
            cc=cc,
            energy_spread=energy_spread,
            deltaV_V=deltaV_V,
            deltaI_I=deltaI_I,
            dose_envelope=dose_envelope,
            aberration_backend=aberration_backend,
            lpp_params=lpp_params,
        )

        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.water_air_interface = water_air_interface
        self.ice_model = ice_model
        self.scattering_model = scattering_model
        self.klim = klim
        self.alpha = alpha
        self.crowd_max_distance_z = (
            crowd_max_distance_z if crowd_max_distance_z is not None else self.nz
        )

        self._apply_defocus_shift(
            shift_required=scattering_model not in ["projection", "ctf"]
        )

        self._init_optics()
        self.slice_batchsize = slice_batchsize
        self.iterative_scattering = IterativeScattering(
            self.pad_nxy,
            self.pixel_size,
            self.voltage,
            scattering_model=self.scattering_model,
            klim=self.klim,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        self.save_clean_exitwaves = save_clean_exitwaves

        if volume is not None:
            volume_icemaker = resolve_icemaker(
                ice_model,
                pixel_size,
                nxy=volume.shape[-1],
                nz=volume.shape[-3],
                ice_cache_dir=ice_cache_dir,
                icemaker=icemaker,
            )
            if volume_icemaker is not None:
                if self.verbose:
                    logger.info(f"Adding ice to volume using {ice_model} model")
                with (
                    torch.no_grad(),
                    status("Tiling ice volume", disable=not self.progressbars),
                ):
                    volume = blend_ice_into_volume(
                        volume, volume_icemaker, pixel_size, relax_steps=ice_relax_steps
                    )
            self.register_buffer("volume", volume)
        else:
            self.specimen_gen = MicrographSpecimenGenerator(
                pixel_size=pixel_size,
                nz=self.nz,
                nxy=self.nxy,
                scattering_potential=scattering_potential,
                crowd_min_distance=crowd_min_distance,
                crowd_max_distance_z=crowd_max_distance_z,
                ice_model=ice_model,
                ice_thickness=ice_thickness,
                ice_cache_dir=ice_cache_dir,
                icemaker=icemaker,
                ice_relax_steps=ice_relax_steps,
                water_air_interface=water_air_interface,
                progressbars=progressbars,
                chunk_size=chunk_size,
                move_to_cpu=move_to_cpu,
                save_clean_exitwaves=save_clean_exitwaves,
            )

    def _generate_volume(self) -> None:
        if self.verbose:
            logger.info(
                "Generating specimen volume (this may take a while for large micrographs)"
            )
        self.volume = self.specimen_gen.generate()
        if self.move_to_cpu:
            self.volume = self.volume.cpu()

    def regenerate_specimen(self) -> None:
        """
        Regenerate the specimen volume with fresh ice and crowding placement.

        Allows multiple independent micrographs from a single model instance
        without reinstantiating.

        Raises
        ------
        RuntimeError
            If the model was constructed with a pre-built ``volume`` (no
            ``specimen_gen`` available).
        """
        if not hasattr(self, "specimen_gen"):
            raise RuntimeError(
                "regenerate_specimen() requires the model to have been constructed "
                "with a scattering_potential, not a pre-built volume."
            )
        self._generate_volume()

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate micrograph images for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated micrographs.
        """
        if not hasattr(self, "volume"):
            self._generate_volume()
        batchsize = len(idx) if isinstance(idx, torch.Tensor) else 1
        with status("Transferring volume to GPU", disable=not self.progressbars):
            V = self.volume.to(self.device).expand(batchsize, -1, -1, -1)
        V = pad_volume(V, self.nxy, self.nz, None, self.pad_fft, xy_pad_mode="reflect")
        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1).to(V.device)
        V = V * scale

        if (
            self.save_clean_exitwaves
            and hasattr(self, "specimen_gen")
            and hasattr(self.specimen_gen, "clean_V")
        ):
            V_clean = self.specimen_gen.clean_V.to(self.device).expand(
                batchsize, -1, -1, -1
            )
            V_clean = pad_volume(
                V_clean, self.nxy, self.nz, None, self.pad_fft, xy_pad_mode="reflect"
            )
            V_clean = V_clean * scale
            self.clean_exitwaves = self.iterative_scattering(
                V_clean, pose=0, slice_batchsize=self.slice_batchsize
            )

        self.exitwaves = self.iterative_scattering(
            V, pose=0, slice_batchsize=self.slice_batchsize
        )

        self.detector_waves = self.aberration(self.exitwaves, self._ctf_batch(idx))

        dose_batch = self.dose_per_angstrom[idx]
        cr_batch = self.coincidence_radius[idx]
        if self.anisomag is None:
            images = self.detector(
                self.detector_waves, dose_batch, cr_batch, nxy=self.nxy
            )
        else:
            images = self.detector(
                self.detector_waves,
                dose_batch,
                cr_batch,
                self.anisomag[idx],
                nxy=self.nxy,
            )
        return images
