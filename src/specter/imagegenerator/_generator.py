from __future__ import annotations

from typing import Any, Literal

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from specter import logger

from .. import rotations
from ._base import BaseImager, compute_nz, pad_volume
from ..crowding import CrowdWithDuplicates
from ..ice import IceBank, RandomIcemaker, ice_blend_mask
from ._micrograph import MicrographGenerator
from ..potential import PotentialBuilder
from ..rotations import VolumeRotator, translate_coordinates
from ..scattering import Scattering
from ._tiltseries import TiltSeriesGenerator


__all__ = [
    "ImageGenerator",
    "ImageGeneratorFromCoordinates",
    "MicrographGenerator",
    "TiltSeriesGenerator",
    "compute_nz",
    "pad_volume",
]


class ParticleGeneratorBase(BaseImager):
    """
    Base class for particle image generators.

    Extends ``BaseImager`` with the particle-specific imaging pipeline:
    crowding → potential scaling → ice (solvation) → scatter → aberrate →
    detect.  ``ImageGenerator`` and ``ImageGeneratorFromCoordinates`` both
    inherit from this class.

    The concrete subclass is responsible for building the rotated volume ``V``
    of shape (B, Z, Y, X) and passing it to ``process_volume``.
    """

    quaternions: torch.Tensor
    translations: torch.Tensor

    def solvate(self, V: torch.Tensor) -> torch.Tensor:
        """
        Embed the volume in amorphous ice.

        V must already be Z-padded (via ``pad_volume``) before calling this.
        Ice is XY-padded here to match the FFT-padded size when ``pad_fft``
        is True.

        Parameters
        ----------
        V : torch.Tensor
            Volume potential of shape (B, Z, Y, X).

        Returns
        -------
        V : torch.Tensor
            Volume with ice blended in via icemask.
        """
        if isinstance(self.icemaker, IceBank):
            ice = self.icemaker.generate_big_ice(
                n=self.nxy,
                dx=self.pixel_size,
                nz=self.nz,
                batchsize=len(V),
                relax_steps=self.ice_relax_steps,
            ).to(V.device)
        else:
            ice = self.icemaker.generate_ice(batchsize=len(V)).to(V.device)

        if self.pad_fft:
            ice = F.pad(
                ice,
                (self.nxy // 2, self.nxy // 2, self.nxy // 2, self.nxy // 2, 0, 0),
                mode="reflect",
            )

        mask = ice_blend_mask(V)
        ice.mul_(mask)
        V = V.add_(ice)
        if hasattr(self, "icemask"):
            self.icemask = mask.to(V.dtype).detach().cpu()
        return V

    def process_volume(self, V: torch.Tensor, idx: torch.Tensor | int) -> torch.Tensor:
        """
        Run the particle imaging pipeline on a prepared volume.

        Parameters
        ----------
        V : torch.Tensor
            Rotated and padded volume of shape (B, Z, Y, X).
        idx : torch.Tensor or int
            Batch indices used to select per-image parameters.

        Returns
        -------
        images : torch.Tensor
            Simulated detector images.
        """
        if hasattr(self, "crowd"):
            if self.verbose:
                logger.info("Adding crowding molecules to volume")
            with torch.no_grad():
                for i in range(len(V)):
                    vols = self.crowd()
                    if not isinstance(vols, float):
                        self.vols = vols.detach().cpu()
                    V[i] += vols

        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1)
        V = V * scale

        if getattr(self, "save_clean_exitwaves", False):
            self.clean_exitwaves = self.scattering(V)

        if hasattr(self, "icemaker"):
            if self.verbose:
                logger.info(f"Adding ice to volume using {self.ice_model} model")
            with torch.no_grad():
                V = self.solvate(V)

        if self.verbose:
            logger.info(f"Applying scattering using {self.scattering_model} model")
        self.exitwaves = self.scattering(V)

        if self.verbose:
            logger.info(f"Applying aberrations using {self.aberration_model} model")
        self.detector_waves = self.aberration(self.exitwaves, self._ctf_batch(idx))

        if self.verbose:
            logger.info(f"Applying detector and noise using {self.noise_model} model")
        dose_batch = self.dose_per_angstrom[idx]
        cr_batch = self.coincidence_radius[idx]
        if self.anisomag is None:
            images = self.detector(
                self.detector_waves, dose_batch, cr_batch, nxy=self.nxy
            )
        else:
            images = self.detector(
                self.detector_waves, dose_batch, cr_batch, self.anisomag[idx], self.nxy
            )
        return images


class ImageGeneratorFromCoordinates(ParticleGeneratorBase):
    """
    Generates images from atomic coordinates.

    Particles are defined by atomic coordinates, rotated, and then voxelized into potential volumes.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates. Shape (n_atoms, 3).
    atomic_numbers : torch.Tensor
        Atomic numbers for each atom. Shape (n_atoms,).
    nxy : int
        Image size in pixels.
    pixel_size : float
        Pixel size in Å.
    quaternions : torch.Tensor
        Rotation quaternions for each batch item. Shape (B, 4).
    translations : torch.Tensor
        Translations (x, y) in Å for each batch item. Shape (B, 2).
    ctf_params : dict
        CTF parameters.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per image in e⁻/Å². Scalar, or a 1-D
        tensor of length n giving a separate dose for each image.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice generation algorithm: ``'gd'`` (samples from the pre-generated
        :class:`~specter.ice.IceBank` cache) or ``'random'`` (instant, cheap
        :class:`~specter.ice.RandomIcemaker` placement -- no realism, useful
        as a fast baseline/smoke test). Ignored when ``icemaker`` is provided.
    ice_thickness : float, optional
        Thickness of ice in Å.
    ice_cache_dir : str, optional
        Directory of cached ice configs for ``ice_model='gd'`` (see
        :func:`specter.ice.build_ice_cache`). Defaults to the bundled
        ``ice_data/ice_cache``. Ignored for other ``ice_model`` values or
        when ``icemaker`` is provided.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across multiple
        ``ImageGenerator`` instances. When supplied, ``ice_model`` and
        ``ice_cache_dir`` are both ignored. ``ice_thickness`` is still
        respected for computing ``nz``.
    ice_relax_steps : int, optional
        Forwarded to :meth:`~specter.ice.IceBank.generate_big_ice` when
        ``ice_model='gd'`` (or an ``IceBank`` ``icemaker``): number of local
        MLBOP relaxation steps used to heal tile seams. Default 0 (no
        relaxation). Ignored for ``RandomIcemaker``.
    scattering_model : str, optional
        Scattering model ('multislice', 'projection', 'ctf'). Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal space limit.
    ews_curvature_sign : str, optional
        Ewald sphere curvature sign matching CryoSPARC's convention.
        ``'negative'`` (default) or ``'positive'``.
    alpha : float, optional
        Amplitude contrast ratio.
    crowd_min_distance : float, optional
        Crowding minimum distance.
    crowd_max_distance_z : float, optional
        Crowding maximum Z distance.
    crowd_chunk_size : int or None, optional
        Number of crowding volumes rotated per GPU batch. Default 1 (memory-safe).
        Set to ``None`` to rotate all at once (faster but O(N × volume) GPU RAM).
    pad_fft : bool, optional
        Whether to pad for FFT.
    conv_backend : str, optional
        Backend for convolution in potential building. Default 'fftconvolve'.
    detector_model : str, optional
        Detector model name.
    periodic_potential : bool, optional
        If True, use periodic boundary conditions when voxelizing coordinates
        into the potential. Required when coordinates come from a periodic ice
        generator (e.g. GradientSKIcemaker) to avoid density deficiency at
        the box boundary. Default False.
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
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        nxy: int,
        pixel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, Any],
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        ice_cache_dir: str | None = None,
        icemaker: IceBank | RandomIcemaker | None = None,
        ice_relax_steps: int = 0,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str | None = "poisson",
        klim: float | None = None,
        ews_curvature_sign: str = "negative",
        alpha: float = 0.0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        crowd_chunk_size: int | None = 1,
        pad_fft: bool = False,
        conv_backend: str = "fftconvolve",
        detector_model: str | None = None,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        n_frames: int | None = None,
        mean_squared_displacement_per_dose: float = 0.0,
        periodic_potential: bool = False,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
    ):
        self.pad_fft = pad_fft
        self.ice_thickness = ice_thickness
        self.ice_relax_steps = ice_relax_steps
        self.nxy = nxy

        self.pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy
        self.nz = compute_nz(self.nxy, ice_thickness, pixel_size)

        super().__init__(
            pixel_size=pixel_size,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            nxy=self.nxy,
            nz=self.nz,
            pad_nxy=self.pad_nxy,
            aberration_model=aberration_model,
            noise_model=noise_model,
            alpha=alpha,
            detector_model=detector_model,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=False,
            verbose=verbose,
            coincidence_radius=coincidence_radius,
            n_frames=n_frames,
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
        self.ice_model = ice_model
        self.crowd_max_distance_z = (
            crowd_max_distance_z
            if crowd_max_distance_z is not None
            else self.nz * pixel_size
        )
        self.scattering_model = scattering_model
        self.klim = klim
        self.ews_curvature_sign = ews_curvature_sign
        self.crowd_min_distance = crowd_min_distance

        self.coordinates = nn.Parameter(coordinates)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        self.atomic_numbers = atomic_numbers
        self.mean_squared_displacement_per_dose = mean_squared_displacement_per_dose
        if mean_squared_displacement_per_dose != 0:
            print(f"Perturbing coordinates by: {mean_squared_displacement_per_dose}.")

        self.potentialbuilder = PotentialBuilder(
            self.nxy,
            self.pixel_size,
            self.atomic_numbers,
            conv_backend=conv_backend,
            periodic=periodic_potential,
        )
        self.atomic_numbers = atomic_numbers

        self.V = self.potentialbuilder(self.coordinates.detach())

        self._init_optics()

        self.scattering = Scattering(
            self.pad_nxy,
            self.pixel_size,
            self.voltage,
            scattering_model=self.scattering_model,
            klim=self.klim,
            ews_curvature_sign=self.ews_curvature_sign,
            nz=self.nz,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"]
        )

        if self.crowd_min_distance is not None:
            self.crowd = CrowdWithDuplicates(
                self.V,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=self.pad_nxy if pad_fft else self.nxy,
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=None,
                method="3d",
                n_points=torch.inf,
                seed="origin",
                chunk_size=crowd_chunk_size,
            )

        # No assignment at all (rather than self.icemaker = None) when ice_model
        # is None/"none" -- process_volume() checks hasattr(self, "icemaker"),
        # not "is not None", to decide whether to call solvate() at all.
        self.icemaker: IceBank | RandomIcemaker | None
        if icemaker is not None:
            self.icemaker = icemaker
            self.ice_model = icemaker.method
        elif self.ice_model == "gd":
            self.icemaker = IceBank(cache_dir=ice_cache_dir)
        elif self.ice_model == "random":
            self.icemaker = RandomIcemaker(dx=pixel_size, n=self.nxy, nz=self.nz)
        elif self.ice_model not in (None, "none"):
            raise ValueError(
                f"Unknown ice_model '{self.ice_model}'. Choose 'gd', 'random', or 'none'."
            )

    def rotate(self, Q: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Rotate and translate atomic coordinates.

        Parameters
        ----------
        Q : torch.Tensor
            Rotation quaternions.
        T : torch.Tensor
            Translations (x, y) in Å.

        Returns
        -------
        r_coordinates : torch.Tensor
            Rotated and translated coordinates, in Å.

        Notes
        -----
        The translation is applied directly in Å, after the rotation, so it acts
        in the lab frame and is subtracted -- the same origin-offset semantics
        :class:`ImageGenerator`'s volume path gets from
        :func:`~specter.rotations.build_affine_matrix`. It is deliberately *not*
        routed through
        :func:`~specter.rotations.translations_angstrom_to_torch`, which
        normalizes to ``grid_sample``'s [-1, 1] convention and only makes sense
        for an affine grid, not for atom positions.
        """
        R = roma.unitquat_to_rotmat(Q).transpose(-2, -1)  # inverse rotation
        N = self.coordinates.shape[0]
        if R.ndim == 2:
            rotated = self.coordinates @ R.T
        else:
            B = R.shape[0]
            vectors_exp = self.coordinates.unsqueeze(0).expand(B, N, 3)
            rotated = torch.einsum("bij,bkj->bki", R, vectors_exp)
        r_coordinates = translate_coordinates(rotated, T, inverse=True)
        return r_coordinates

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate images for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        if self.mean_squared_displacement_per_dose != 0.0:
            msd = (
                self.mean_squared_displacement_per_dose * self.dose_per_angstrom.mean()
            )
            self.sigma_angstrom = (msd / 3) ** 0.5
            self.coordinates.add_(
                torch.randn_like(self.coordinates) * self.sigma_angstrom
            )

        coordinates = self.rotate(self.quaternions[idx], self.translations[idx])

        V = self.potentialbuilder(coordinates)
        if V.ndim == 3:
            V = V.unsqueeze(0)

        V = pad_volume(
            V,
            self.nxy,
            self.nz,
            self.ice_thickness,
            self.pad_fft,
            xy_pad_mode="reflect",
        )
        return self.process_volume(V, idx)


class ImageGenerator(ParticleGeneratorBase):
    """
    Generates images from a pre-computed scattering potential volume.

    The volume is rotated and translated for each simulation instance.

    Parameters
    ----------
    scattering_potential : torch.Tensor
        Scattering potential volume. Shape (Z, Y, X).
    pixel_size : float
        Pixel size in Å.
    quaternions : torch.Tensor
        Rotation quaternions. Shape (B, 4).
    translations : torch.Tensor
        Translations (x, y) in Å. Shape (B, 2).
    ctf_params : dict
        CTF parameters.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per image in e⁻/Å². Scalar, or a 1-D
        tensor of length n giving a separate dose for each image.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice generation algorithm: ``'gd'`` (samples from the pre-generated
        :class:`~specter.ice.IceBank` cache) or ``'random'`` (instant, cheap
        :class:`~specter.ice.RandomIcemaker` placement -- no realism, useful
        as a fast baseline/smoke test). Ignored when ``icemaker`` is provided.
    ice_thickness : float, optional
        Ice thickness in Å.
    ice_cache_dir : str, optional
        Directory of cached ice configs for ``ice_model='gd'`` (see
        :func:`specter.ice.build_ice_cache`). Defaults to the bundled
        ``ice_data/ice_cache``. Ignored for other ``ice_model`` values or
        when ``icemaker`` is provided.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across multiple
        ``ImageGeneratorFromCoordinates`` instances. When supplied,
        ``ice_model`` and ``ice_cache_dir`` are both ignored. ``ice_thickness``
        is still respected for computing ``nz``.
    ice_relax_steps : int, optional
        Forwarded to :meth:`~specter.ice.IceBank.generate_big_ice` when
        ``ice_model='gd'`` (or an ``IceBank`` ``icemaker``): number of local
        MLBOP relaxation steps used to heal tile seams. Default 0 (no
        relaxation). Ignored for ``RandomIcemaker``.
    scattering_model : str, optional
        Scattering model. Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal space limit.
    ews_curvature_sign : str, optional
        Ewald sphere curvature sign matching CryoSPARC's convention.
        ``'negative'`` (default) or ``'positive'``.
    alpha : float, optional
        Amplitude contrast ratio.
    crowd_min_distance : float, optional
        Crowding minimum distance.
    crowd_max_distance_z : float, optional
        Crowding maximum Z distance.
    crowd_max_distance_xy : float, optional
        Crowding maximum XY distance. Defaults to ``nxy_out * dx + min_distance``.
    crowd_chunk_size : int or None, optional
        Number of crowding volumes rotated per GPU batch. ``1`` (default) avoids
        the O(N × nz × ny × nx × 3) peak allocation that causes OOM for large
        volumes with many crowding particles. Set to ``None`` to rotate all at
        once (faster but requires N × volume_size RAM).
    crowd_method : {"2d", "3d"}, optional
        Poisson-disk sampling dimensionality for crowding placement. Default
        ``"3d"``.
    crowd_n_points : int, optional
        Cap on the number of crowding duplicates. ``None`` (default) fills the
        volume (no cap).
    crowd_seed : {"origin", "random"}, optional
        Crowding placement seed strategy. Default ``"origin"``.
    crowd_move_to_cpu : bool, optional
        Move crowding intermediates to CPU between steps, trading speed for
        lower GPU memory. Default False.
    water_air_interface : bool, optional
        Apply a bimodal density distribution along z when placing crowding
        duplicates, mimicking particle adsorption at the ice-water interface.
        Default False.
    pad_fft : bool, optional
        Whether to pad for FFT.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    ice_parameterization : str, optional
        Atomic potential parameterization used to build the ice kernel:
        ``'kirkland'``, ``'lobato'``, or ``'shtyrov'``. Default ``'shtyrov'``,
        matching :class:`~specter.potential.PotentialBuilder`'s own default.
    detector_model : str, optional
        Detector model name.
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
    rotate_mode : {"real", "fourier"}, optional
        Volume rotation method. ``"real"`` uses trilinear interpolation;
        ``"fourier"`` rotates in Fourier space (no boundary artifacts).
        Default ``"real"``.
    """

    def __init__(
        self,
        scattering_potential: torch.Tensor,
        pixel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, Any],
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        ice_cache_dir: str | None = None,
        icemaker: IceBank | RandomIcemaker | None = None,
        ice_relax_steps: int = 0,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str | None = "poisson",
        klim: float | None = None,
        ews_curvature_sign: str = "negative",
        alpha: float = 0.0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        crowd_max_distance_xy: float | None = None,
        crowd_chunk_size: int | None = 1,
        crowd_method: Literal["2d", "3d"] = "3d",
        crowd_n_points: int | None = None,
        crowd_seed: Literal["origin", "random"] = "origin",
        crowd_move_to_cpu: bool = False,
        water_air_interface: bool = False,
        pad_fft: bool = False,
        progressbars: bool = True,
        verbose: bool = True,
        ice_parameterization: str = "shtyrov",
        detector_model: str | None = None,
        coincidence_radius: float | torch.Tensor = 0.0,
        n_frames: int | None = None,
        potential_scale: float | torch.Tensor = 1.0,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        rotate_mode: Literal["real", "fourier"] = "real",
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
    ):
        nxy = scattering_potential.shape[-1]
        self.pad_fft = pad_fft
        self.ice_thickness = ice_thickness
        self.ice_relax_steps = ice_relax_steps
        self.pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy

        vol_nz = scattering_potential.shape[0]
        self.nz = compute_nz(vol_nz, ice_thickness, pixel_size)

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

        self.ice_parameterization = ice_parameterization
        self.ice_model = ice_model
        self.crowd_max_distance_z = (
            crowd_max_distance_z
            if crowd_max_distance_z is not None
            else self.nz * pixel_size
        )
        self.scattering_model = scattering_model
        self.klim = klim
        self.ews_curvature_sign = ews_curvature_sign
        self.crowd_min_distance = crowd_min_distance

        self.register_buffer("V", scattering_potential)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        if self.verbose:
            logger.info("Initializing ImageGenerator modules")

        self._init_optics()

        self.scattering = Scattering(
            self.pad_nxy,
            self.pixel_size,
            self.voltage,
            scattering_model=self.scattering_model,
            klim=self.klim,
            ews_curvature_sign=self.ews_curvature_sign,
            nz=self.nz,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        if self.crowd_min_distance is not None:
            self.crowd = CrowdWithDuplicates(
                self.V,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=self.pad_nxy if pad_fft else self.nxy,
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=crowd_max_distance_xy,
                method=crowd_method,
                n_points=crowd_n_points if crowd_n_points is not None else torch.inf,
                seed=crowd_seed,
                move_to_cpu=crowd_move_to_cpu,
                progressbars=self.progressbars,
                chunk_size=crowd_chunk_size,
                water_air_interface=water_air_interface,
            )

        # No assignment at all (rather than self.icemaker = None) when ice_model
        # is None/"none" -- process_volume() checks hasattr(self, "icemaker"),
        # not "is not None", to decide whether to call solvate() at all.
        self.icemaker: IceBank | RandomIcemaker | None
        if icemaker is not None:
            self.icemaker = icemaker
            self.ice_model = icemaker.method
        elif self.ice_model == "gd":
            self.icemaker = IceBank(
                cache_dir=ice_cache_dir,
                parameterization=self.ice_parameterization,
                progressbars=self.progressbars,
            )
        elif self.ice_model == "random":
            self.icemaker = RandomIcemaker(
                dx=pixel_size,
                n=self.nxy,
                nz=self.nz,
                parameterization=self.ice_parameterization,
                progressbars=self.progressbars,
            )
        elif self.ice_model not in (None, "none"):
            raise ValueError(
                f"Unknown ice_model '{self.ice_model}'. Choose 'gd', 'random', or 'none'."
            )

        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"]
        )

        nz, ny, nx = self.V.shape
        self.rotator = VolumeRotator(nz, ny, nx, origin="relion", mode=rotate_mode)

    def rotate(self, Q: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Rotate volume using affine transformation.

        Parameters
        ----------
        Q : torch.Tensor
            Rotation quaternions.
        T : torch.Tensor
            Translations.

        Returns
        -------
        V : torch.Tensor
            Rotated volume.
        """
        if len(Q.shape) < 2:
            Q = Q.unsqueeze(0)
        if len(T.shape) < 2:
            T = T.unsqueeze(0)
        R = roma.unitquat_to_rotmat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        theta = rotations.build_affine_matrix(R, T)
        V = self.rotator(self.V, theta)
        return V

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate images for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        if self.verbose:
            logger.info("Rotating volume")
        V = self.rotate(self.quaternions[idx], self.translations[idx])

        V = pad_volume(V, self.nxy, self.nz, self.ice_thickness, self.pad_fft)
        return self.process_volume(V, idx)
