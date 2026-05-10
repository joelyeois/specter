from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from specter import logger

from .. import rotations
from ._base import BaseImager, compute_nz, pad_volume
from ..crowding import CrowdWithDuplicates
from ..ice import IceBank
from ._micrograph import MicrographGenerator
from ..potential import PotentialBuilder
from ..rotations import Rotation, VolumeRotator
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
        ice = self.icemaker.generate_ice(batchsize=len(V)).to(V.device)

        if self.pad_fft:
            ice = F.pad(
                ice,
                (self.nxy // 2, self.nxy // 2, self.nxy // 2, self.nxy // 2, 0, 0),
                mode="reflect",
            )

        icemask = V.detach().clone()
        threshold = 0.05 * V.max()
        icemask[icemask < threshold] = 1
        icemask[icemask >= threshold] = 0
        V = V + ice * icemask
        if hasattr(self, "icemask"):
            self.icemask = icemask.detach().cpu()
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
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose per Å². Scalar or 1-D tensor of length n.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice generation algorithm: ``'ap'`` (alternating projections), ``'gd'``
        (gradient descent), ``'mcmc'`` (Metropolis Monte Carlo), or ``'random'``
        (naive random placement).
    ice_thickness : float, optional
        Thickness of ice in Å.
    num_unique_icecubes : int, optional
        Number of unique ice cubes pre-built into the ``IceBank``. Default 8.
    ice_build_batch_size : int, optional
        Number of unique ice cubes to generate at once while building the
        ``IceBank``. Lower values reduce peak GPU memory. Default 1.
    scattering_model : str, optional
        Scattering model ('multislice', 'projection', 'ctf'). Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal space limit.
    flip_curvature : bool, optional
        Whether to flip curvature.
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
    bfactor_envelope : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
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
        energy: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        num_unique_icecubes: int = 8,
        ice_build_batch_size: int = 1,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        flip_curvature: bool = False,
        alpha: float = 0.0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        crowd_chunk_size: int | None = 1,
        pad_fft: bool = False,
        conv_backend: str = "fftconvolve",
        detector_model: str | None = None,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        num_frames: int | None = None,
        mean_squared_displacement_per_dose: float = 0.0,
        periodic_potential: bool = False,
        bfactor_envelope: float | torch.Tensor | None = None,
    ):
        self.pad_fft = pad_fft
        self.ice_thickness = ice_thickness
        self.nxy = nxy

        self.pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy
        self.nz = compute_nz(self.nxy, ice_thickness, pixel_size)

        super().__init__(
            pixel_size=pixel_size,
            energy=energy,
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
            num_frames=num_frames,
            bfactor_envelope=bfactor_envelope,
        )
        self.ice_model = ice_model
        self.crowd_max_distance_z = (
            crowd_max_distance_z
            if crowd_max_distance_z is not None
            else self.nz * pixel_size
        )
        self.scattering_model = scattering_model
        self.klim = klim
        self.flip_curvature = flip_curvature
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
            self.energy,
            scattering_model=self.scattering_model,
            klim=self.klim,
            flip_curvature=self.flip_curvature,
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

        if self.ice_model is not None:
            if self.ice_model not in ("ap", "gd", "mcmc", "random"):
                raise ValueError(
                    f"Unknown ice_model '{self.ice_model}'. Choose 'ap', 'gd', 'mcmc', or 'random'."
                )
            self.icemaker = IceBank(
                n=self.nxy,
                dx=pixel_size,
                nz=self.nz,
                method=self.ice_model,
                num_unique=num_unique_icecubes,
                build_batch_size=ice_build_batch_size,
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
            Rotated and translated coordinates.
        """
        R = Rotation.from_quat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        r_coordinates = R.apply(self.coordinates, T=T)
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
            self.coordinates += torch.randn_like(self.coordinates) * self.sigma_angstrom

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
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose per Å². Scalar or 1-D tensor of length n.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice generation algorithm: ``'ap'`` (alternating projections), ``'gd'``
        (gradient descent), ``'mcmc'`` (Metropolis Monte Carlo), or ``'random'``
        (naive random placement).
    ice_thickness : float, optional
        Ice thickness in Å.
    num_unique_icecubes : int, optional
        Number of unique ice cubes pre-built into the ``IceBank``. Default 8.
    ice_build_batch_size : int, optional
        Number of unique ice cubes to generate at once while building the
        ``IceBank``. Lower values reduce peak GPU memory. Default 1.
    scattering_model : str, optional
        Scattering model. Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal space limit.
    flip_curvature : bool, optional
        Whether to flip curvature.
    alpha : float, optional
        Amplitude contrast ratio.
    crowd_min_distance : float, optional
        Crowding minimum distance.
    crowd_max_distance_z : float, optional
        Crowding maximum Z distance.
    crowd_chunk_size : int or None, optional
        Number of crowding volumes rotated per GPU batch. ``1`` (default) avoids
        the O(N × nz × ny × nx × 3) peak allocation that causes OOM for large
        volumes with many crowding particles. Set to ``None`` to rotate all at
        once (faster but requires N × volume_size RAM).
    pad_fft : bool, optional
        Whether to pad for FFT.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    parameterization : str, optional
        Parameterization for ice potential. Default 'kirkland'.
    detector_model : str, optional
        Detector model name.
    bfactor_envelope : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
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
        energy: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        num_unique_icecubes: int = 8,
        ice_build_batch_size: int = 1,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        flip_curvature: bool = False,
        alpha: float = 0.0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        crowd_chunk_size: int | None = 1,
        pad_fft: bool = False,
        progressbars: bool = True,
        verbose: bool = True,
        parameterization: str = "kirkland",
        detector_model: str | None = None,
        slice_batch_size: int = 1,
        coincidence_radius: float | torch.Tensor = 0.0,
        num_frames: int | None = None,
        potential_scale: float | torch.Tensor = 1.0,
        bfactor_envelope: float | torch.Tensor | None = None,
        rotate_mode: Literal["real", "fourier"] = "real",
    ):
        nxy = scattering_potential.shape[-1]
        self.pad_fft = pad_fft
        self.ice_thickness = ice_thickness
        self.pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy

        vol_nz = scattering_potential.shape[0]
        self.nz = compute_nz(vol_nz, ice_thickness, pixel_size)

        super().__init__(
            pixel_size=pixel_size,
            energy=energy,
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
            num_frames=num_frames,
            potential_scale=potential_scale,
            bfactor_envelope=bfactor_envelope,
        )

        self.parameterization = parameterization
        self.ice_model = ice_model
        self.crowd_max_distance_z = (
            crowd_max_distance_z
            if crowd_max_distance_z is not None
            else self.nz * pixel_size
        )
        self.scattering_model = scattering_model
        self.klim = klim
        self.flip_curvature = flip_curvature
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
            self.energy,
            scattering_model=self.scattering_model,
            klim=self.klim,
            flip_curvature=self.flip_curvature,
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
                max_distance_xy=None,
                method="3d",
                n_points=torch.inf,
                seed="origin",
                progressbars=self.progressbars,
                chunk_size=crowd_chunk_size,
            )

        if self.ice_model is not None:
            if self.ice_model not in ("ap", "gd", "mcmc", "random"):
                raise ValueError(
                    f"Unknown ice_model '{self.ice_model}'. Choose 'ap', 'gd', 'mcmc', or 'random'."
                )
            self.icemaker = IceBank(
                n=self.nxy,
                dx=pixel_size,
                nz=self.nz,
                method=self.ice_model,
                num_unique=num_unique_icecubes,
                build_batch_size=ice_build_batch_size,
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
        R = Rotation.from_quat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        theta = rotations.build_affine_matrix(R.as_matrix(), T)
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
