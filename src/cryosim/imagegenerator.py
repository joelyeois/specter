from __future__ import annotations

from typing import Any, Sequence

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from .progress import track

from cryosim import logger
from cryosim.detectors import k3_200kv, k3_300kv, perfect_detector

from . import rotations
from .crowding import CrowdWithDuplicates
from .icemaker import Icemaker, NaiveIcemaker
from .microscope import Aberration, Detector
from .potential import PotentialBuilder
from .rotations import Rotation, VolumeRotator
from .scattering import IterativeScattering, Scattering
from .specimen import TomogramGenerator


class BaseImageGenerator(L.LightningModule):
    """
    Base class for image generation with common functionality.

    Handles initialization of common parameters, detector MTF, CTF parameters,
    and references to processing modules.

    Parameters
    ----------
    pixel_size : float
        Pixel size in Å.
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float
        Electron dose per Å².
    nxy : int
        Image size in pixels.
    nz : int
        Number of slices in z dimension.
    pad_nxy : int, optional
        Padded image size. Default is nxy.
    aberration_model : str, optional
        Model for microscope aberrations ('holography', 'phase_plate'). Default 'holography'.
    noise_model : str, optional
        Noise model for detection ('poisson', 'gaussian', None). Default 'poisson'.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.0.
    detector_model : str, optional
        Detector model for MTF ('k3_300kv', 'k3_200kv', 'perfect', None).
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ctf_params : dict, optional
        Dictionary of CTF parameters.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    verbose : bool, optional
        Whether to enable verbose logging for this instance. Default True.
    coincidence_radius : float, optional
        Coincidence radius for detector. Default 0.0.
    """

    def __init__(
        self,
        pixel_size: float,
        energy: float,
        dose_per_angstrom: float,
        nxy: int,
        nz: int,
        pad_nxy: int | None = None,
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        alpha: float = 0.0,
        detector_model: str | None = None,
        anisomag: torch.Tensor | None = None,
        ctf_params: dict[str, Any] | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float = 0.0,
        num_frames: int | None = None,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.alpha = alpha
        self.progressbars = progressbars
        self.verbose = verbose
        self.nxy = nxy
        self.nz = nz
        self.pad_nxy = pad_nxy if pad_nxy is not None else nxy
        self.detector_model = detector_model
        self._init_detector_mtf()
        self.coincidence_radius = coincidence_radius
        self.num_frames = num_frames

        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", torch.as_tensor(anisomag))

        if ctf_params is not None:
            # Register CTF parameters as buffers.
            # Ensure they are at least 1D so they can be indexed by [idx]
            for k, v in ctf_params.items():
                v_tensor = torch.as_tensor(v)
                if v_tensor.ndim == 0:
                    v_tensor = v_tensor.unsqueeze(0)
                self.register_buffer(k, v_tensor)
            self._ctf_param_names = list(ctf_params.keys())
        else:
            self._ctf_param_names = []

    def _init_detector_mtf(self) -> None:
        """Initialize the detector MTF based on the model name."""
        if self.detector_model == "k3_300kv":
            self.register_buffer("detector_mtf", k3_300kv(self.nxy, self.pixel_size))
        elif self.detector_model == "k3_200kv":
            self.register_buffer("detector_mtf", k3_200kv(self.nxy, self.pixel_size))
        elif self.detector_model == "perfect":
            self.register_buffer(
                "detector_mtf", perfect_detector(self.nxy, self.pixel_size)
            )
        else:
            self.detector_mtf = None

    def _apply_defocus_shift(self, shift_required: bool = True) -> None:
        """Apply defocus shift to account for volume thickness."""
        if shift_required:
            shift = (self.nz * self.pixel_size) / 2
            if hasattr(self, "dfu"):
                self.dfu = self.dfu - shift
            if hasattr(self, "dfv"):
                self.dfv = self.dfv - shift

    def _init_optics(self) -> None:
        """Initialize Aberration and Detector modules."""
        self.aberration = Aberration(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            aberration_model=self.aberration_model,
            alpha=self.alpha,
        )

        self.detector = Detector(
            self.pixel_size,
            self.dose_per_angstrom,
            aberration_model=self.aberration_model,
            noise_model=self.noise_model,
            mtf=self.detector_mtf,
            coincidence_radius=self.coincidence_radius,
            num_frames=self.num_frames,
        )


class VolumeProcessingMixin:
    """
    Mixin for volume processing, handling solvation, crowding, and scattering.
    Expects the subclass to provide `icemaker`, `crowd`, `scattering`,
    `aberration`, `detector`, `anisomag`, `nxy`, `ice_thickness`, `nz`, `pad_fft`, `verbose`.
    """

    def solvate(self, V: torch.Tensor) -> torch.Tensor:
        """
        Embed the volume in ice.

        Parameters
        ----------
        V : torch.Tensor
            Input volume potential.

        Returns
        -------
        V_solvated : torch.Tensor
            Volume with ice added.
        """
        # generates ice with size (B x Z x Y x X)
        ice = self.icemaker.generate_ice(batchsize=len(V))

        if getattr(self, "pad_fft", False):
            ice = F.pad(
                ice,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="reflect",
            )

        # pad V in z-axis if ice_thickness is not None
        if getattr(self, "ice_thickness", None) is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(
                V,
                (
                    0,
                    0,  # x-axis
                    0,
                    0,  # y-axis
                    zpad_px // 2,
                    self.nz - zpad_px // 2 - V.shape[1],  # z-axis
                ),
            )
        icemask = V.detach().clone()
        icemask[icemask < 10] = 1
        icemask[icemask >= 10] = 0
        V = V + ice * icemask
        if hasattr(self, "icemask"):
            self.icemask = icemask.detach().cpu()
        return V

    def process_volume(self, V: torch.Tensor, idx: torch.Tensor | int) -> torch.Tensor:
        """
        Process the volume: add crowding, ice, scatter, aberrate, and detect.

        Parameters
        ----------
        V : torch.Tensor
            Input volume potential.
        idx : torch.Tensor or int
            Batch indices for parameter selection.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        # add crowding
        if hasattr(self, "crowd"):
            if self.verbose:
                logger.info("Adding crowding molecules to volume")
            with torch.no_grad():
                for i, v in enumerate(V):
                    vols = self.crowd()
                    if not isinstance(vols, float):
                        self.vols = vols.detach().cpu()
                    V[i] += vols

        # clean exit wave: scatter particle-only volume before ice is added
        if getattr(self, "save_clean_exitwaves", False):
            self.clean_exitwaves = self.scattering(V)

        # add ice
        if hasattr(self, "icemaker"):
            if self.verbose:
                logger.info(f"Adding ice to volume using {self.ice_model} model")
            with torch.no_grad():
                V = self.solvate(V)

        # scatter V
        if self.verbose:
            logger.info(f"Applying scattering using {self.scattering_model} model")
        self.exitwaves = self.scattering(V)

        # aberrate exitwaves
        if self.verbose:
            logger.info(f"Applying aberrations using {self.aberration_model} model")
        ctf_batch = {k: getattr(self, k)[idx] for k in self._ctf_param_names}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)

        # image/noise
        if self.verbose:
            logger.info(f"Applying detector and noise using {self.noise_model} model")
        if getattr(self, "anisomag", None) is None:
            images = self.detector(self.detector_waves, nxy=self.nxy)
        else:
            images = self.detector(self.detector_waves, self.anisomag[idx], self.nxy)
        return images

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Standard Lightning predict step."""
        return self(batch)

    def predict_epoch_end(self, outputs: list[torch.Tensor]) -> torch.Tensor | None:
        """
        Gather predictions from all GPUs at epoch end.

        Parameters
        ----------
        outputs : list
            List of outputs from `predict_step`.

        Returns
        -------
        preds : torch.Tensor or None
            Concatenated predictions from all ranks. Returns None on non-zero ranks.
        """
        # outputs is a list of batch predictions from THIS GPU
        preds = torch.cat(outputs, dim=0)

        # gather across all GPUs
        preds_all = self.trainer.strategy.all_gather(preds)

        # return only once on rank 0
        if self.trainer.is_global_zero:
            return preds_all.cpu()


class ImageGeneratorFromCoordinates(BaseImageGenerator, VolumeProcessingMixin):
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
    dose_per_angstrom : float
        Electron dose per Å².
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Model for ice generation.
    ice_thickness : float, optional
        Thickness of ice in Å.
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
    pad_fft : bool, optional
        Whether to pad for FFT.
    conv_backend : str, optional
        Backend for convolution in potential building. Default 'fftconvolve'.
    detector_model : str, optional
        Detector model name.
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
        dose_per_angstrom: float,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        flip_curvature: bool = False,
        alpha: float = 0.0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        pad_fft: bool = False,
        conv_backend: str = "fftconvolve",
        detector_model: str | None = None,
        verbose: bool = True,
        coincidence_radius: float = 0.0,
        num_frames: int | None = None,
        mean_squared_displacement_per_dose: float = 0.0,
    ):
        self.pad_fft = pad_fft
        self.ice_thickness = ice_thickness
        self.nxy = nxy

        if self.pad_fft:
            self.pad_nxy = self.nxy + (self.nxy // 2) * 2
        else:
            self.pad_nxy = self.nxy

        if ice_thickness is None:
            self.nz = self.nxy
        elif ice_thickness < self.nxy * pixel_size:
            self.nz = self.nxy
        else:
            self.nz = int(ice_thickness // pixel_size)

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
        )
        self.ice_model = ice_model

        if crowd_max_distance_z is None:
            self.crowd_max_distance_z = self.nz
        else:
            self.crowd_max_distance_z = crowd_max_distance_z

        self.scattering_model = scattering_model
        self.klim = klim
        self.flip_curvature = flip_curvature
        self.crowd_min_distance = crowd_min_distance

        # register buffers
        self.coordinates = nn.Parameter(coordinates)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        self.atomic_numbers = atomic_numbers
        self.mean_squared_displacement_per_dose = mean_squared_displacement_per_dose
        if mean_squared_displacement_per_dose != 0:
            print(f"Perturbing coordinates by: {mean_squared_displacement_per_dose}.")

        # initialize modules
        self.potentialbuilder = PotentialBuilder(
            self.nxy,
            self.pixel_size,
            self.atomic_numbers,
            trainable=True,
            conv_backend=conv_backend,
        )
        self.atomic_numbers = (
            atomic_numbers  # Needed for potentialbuilder init but not stored in super
        )

        # Pre-calculate V for crowd initialization if needed
        self.V = self.potentialbuilder(self.coordinates.detach())

        # Initialize Optics
        self._init_optics()

        # Initialize Scattering
        self.scattering = Scattering(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            self.dose_per_angstrom,
            scattering_model=self.scattering_model,
            klim=self.klim,
            flip_curvature=self.flip_curvature,
            nz=self.nz,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        # Apply defocus shift
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
            )

        if ice_model is not None:
            if ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(n=self.nxy, dx=pixel_size, nz=self.nz)
            elif ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
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

        Rotating coordinates, voxelizing, and processing volume.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        # adds perturbation to coordinates
        if self.mean_squared_displacement_per_dose != 0.0:
            msd = self.mean_squared_displacement_per_dose * self.dose_per_angstrom
            self.sigma_angstrom = (msd / 3) ** 0.5
            # coordinates are in Ångstroms, so perturb by sigma_angstrom directly
            self.coordinates += torch.randn_like(self.coordinates) * self.sigma_angstrom

        # rotate coordinates, returns (B x N x 3)
        coordinates = self.rotate(self.quaternions[idx], self.translations[idx])

        # sample coordinates to volume
        V = self.potentialbuilder(coordinates)
        if V.ndim == 3:
            V = V.unsqueeze(0)

        # pad z
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(
                V,
                (
                    0,
                    0,  # x-axis, last dim
                    0,
                    0,  # y-axis, second last dim
                    zpad_px // 2,
                    self.nz - zpad_px // 2 - V.shape[1],  # z-axis
                ),
                mode="constant",
            )
        # pad xy
        if self.pad_fft:
            V = F.pad(
                V,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                # mode="constant",
                mode="reflect",  # for ice coordinates
            )
        return self.process_volume(V, idx)


class ImageGenerator(BaseImageGenerator, VolumeProcessingMixin):
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
    dose_per_angstrom : float
        Electron dose per Å².
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice model.
    ice_thickness : float, optional
        Ice thickness in Å.
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
    pad_fft : bool, optional
        Whether to pad for FFT.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    parameterization : str, optional
        Parameterization for ice potential. Default 'kirkland'.
    detector_model : str, optional
        Detector model name.
    """

    def __init__(
        self,
        scattering_potential: torch.Tensor,
        pixel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, Any],
        energy: float,
        dose_per_angstrom: float,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        flip_curvature: bool = False,
        alpha: float = 0.0,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        pad_fft: bool = False,
        progressbars: bool = True,
        verbose: bool = True,
        parameterization: str = "kirkland",
        detector_model: str | None = None,
        slice_batch_size: int = 1,
        coincidence_radius: float = 0.0,
        num_frames: int | None = None,
    ):
        nxy = scattering_potential.shape[-1]
        self.pad_fft = pad_fft
        self.ice_thickness = ice_thickness
        if self.pad_fft:
            self.pad_nxy = nxy + (nxy // 2) * 2
        else:
            self.pad_nxy = nxy

        p_nz = scattering_potential.shape[0]
        if ice_thickness is None:
            self.nz = p_nz
        elif ice_thickness < p_nz * pixel_size:
            self.nz = p_nz
        else:
            self.nz = int(ice_thickness // pixel_size)

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
        )

        self.parameterization = parameterization
        self.ice_model = ice_model

        if crowd_max_distance_z is None:
            self.crowd_max_distance_z = self.nz
        else:
            self.crowd_max_distance_z = crowd_max_distance_z

        self.scattering_model = scattering_model
        self.klim = klim
        self.flip_curvature = flip_curvature
        self.crowd_min_distance = crowd_min_distance

        self.parameterization = parameterization

        # register buffers
        self.register_buffer("V", scattering_potential)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        if self.verbose:
            logger.info("Initializing ImageGenerator modules")

        # Initialize Optics
        self._init_optics()

        # Initialize Scattering
        self.scattering = Scattering(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            self.dose_per_angstrom,
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
            )

        if ice_model is not None:
            if ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
                    progressbars=self.progressbars,
                )
            elif ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
                    progressbars=self.progressbars,
                    parameterization=parameterization,
                )

        # Apply defocus shift
        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"]
        )

        # self.V has shape (Z, Y, X)
        nz, ny, nx = self.V.shape

        # Create VolumeRotator instance
        self.rotator = VolumeRotator(nz, ny, nx, origin="relion", mode="real")

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

        Rotating potential volume and processing volume.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        # rotate V, returns (B x Z x Y x X)
        if self.verbose:
            logger.info("Rotating volume")
        V = self.rotate(self.quaternions[idx], self.translations[idx])

        # pad z
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(
                V,
                (
                    0,
                    0,  # x-axis, last dim
                    0,
                    0,  # y-axis, second last dim
                    zpad_px // 2,
                    self.nz - zpad_px // 2 - V.shape[1],  # z-axis
                ),
                mode="constant",
            )
        # pad xy
        if self.pad_fft:
            V = F.pad(
                V,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="constant",
            )

        return self.process_volume(V, idx)


class MicrographGenerator(BaseImageGenerator):
    """
    Generates large micrographs by stitching or processing large volumes.

    Handles large detectors and potentially chunked processing.

    Parameters
    ----------
    scattering_potential : torch.Tensor
        Scattering potential volume.
    micrograph_size : int or tuple
        Size of the micrograph in pixels.
    pixel_size : float
        Pixel size in Å.
    ctf_params : dict
        CTF parameters.
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float
        Electron dose per Å².
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice model.
    ice_thickness : float, optional
        Ice thickness in Å.
    scattering_model : str, optional
        Scattering model. Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal space limit.
    alpha : float, optional
        Amplitude contrast ratio.
    crowd_min_distance : float, optional
        Crowding minimum distance.
    crowd_max_distance_z : float, optional
        Crowding maximum Z distance.
    pad_fft : bool, optional
        Whether to pad for FFT.
    chunk_size : int, optional
        Chunk size for processing.
    move_to_cpu : bool, optional
        Whether to move intermediate results to CPU to save GPU memory. Default True.
    water_air_interface : bool, optional
        Whether to simulate water-air interface. Default True.
    detector_model : str, optional
        Detector model name.
    """

    def __init__(
        self,
        scattering_potential: torch.Tensor | None,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any],
        energy: float,
        dose_per_angstrom: float,
        vol: torch.Tensor | None = None,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_thickness: float | None = None,
        crowd_min_distance: float | None = None,
        crowd_max_distance_z: float | None = None,
        water_air_interface: bool = True,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        alpha: float = 0.0,
        pad_fft: bool = False,
        chunk_size: int | None = None,
        move_to_cpu: bool = True,
        detector_model: str | None = None,
        slice_batch_size: int = 1,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float = 0.0,
        num_frames: int | None = None,
        save_clean_exitwaves: bool = False,
        **kwargs: Any,
    ):
        # Determine nxy
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
        if self.pad_fft:
            self.pad_nxy = nxy + (nxy // 2) * 2
        else:
            self.pad_nxy = nxy

        # Determine nz and ice_thickness
        if vol is not None:
            self.nz = vol.shape[0]
            self.ice_thickness = self.nz * pixel_size
        else:
            if scattering_potential is not None:
                p_nz = scattering_potential.shape[0]
                if ice_thickness is None or ice_thickness < p_nz * pixel_size:
                    self.nz = p_nz
                else:
                    self.nz = int(ice_thickness // pixel_size)
                self.ice_thickness = self.nz * pixel_size
            else:
                raise ValueError(
                    "Either 'vol' or 'scattering_potential' must be provided."
                )

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
        )

        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.water_air_interface = water_air_interface
        self.ice_model = ice_model

        self.scattering_model = scattering_model
        self.klim = klim
        self.alpha = alpha

        # Apply defocus shift
        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"]
        )

        # Re-init crowd_max_distance_z if it was None (it defaults to self.nz in base)
        if crowd_max_distance_z is None:
            self.crowd_max_distance_z = self.nz
        else:
            self.crowd_max_distance_z = crowd_max_distance_z

        # Re-init modules with correct nz
        self._init_optics()
        self.slice_batch_size = slice_batch_size
        self.iterative_scattering = IterativeScattering(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            self.dose_per_angstrom,
            scattering_model=self.scattering_model,
            klim=self.klim,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        self.save_clean_exitwaves = save_clean_exitwaves

        if vol is not None:
            self.vol = vol
        else:
            # Use TomogramGenerator for internal volume generation
            self.specimen_gen = TomogramGenerator(
                pixel_size=pixel_size,
                nz=self.nz,
                nxy=self.nxy,
                scattering_potential=scattering_potential,
                crowd_min_distance=crowd_min_distance,
                crowd_max_distance_z=crowd_max_distance_z,
                ice_model=ice_model,
                ice_thickness=ice_thickness,
                water_air_interface=water_air_interface,
                progressbars=progressbars,
                chunk_size=chunk_size,
                save_clean_exitwaves=save_clean_exitwaves,
            )
            if self.verbose:
                logger.info(
                    "Generating specimen volume (this may take a while for large micrographs)"
                )
            self.vol = self.specimen_gen.generate()

        if self.move_to_cpu:
            self.vol = self.vol.cpu()

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate micrograph images.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated micrographs.
        """
        # MicrographGenerator forward is different: starts with empty V
        # The volume is now pre-generated and stored in self.vol
        V = self.vol.to(self.device).expand(
            len(idx), -1, -1, -1
        )  # Expand to batch size

        # No need to add crowd or ice here, it's done during self.vol generation

        # pad xy
        if self.pad_fft:
            V = F.pad(
                V,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="reflect",
            )

        # clean exit wave: scatter particle-only volume (no ice)
        if (
            self.save_clean_exitwaves
            and hasattr(self, "specimen_gen")
            and hasattr(self.specimen_gen, "clean_V")
        ):
            V_clean = self.specimen_gen.clean_V.to(self.device).expand(
                len(idx), -1, -1, -1
            )
            if self.pad_fft:
                V_clean = F.pad(
                    V_clean,
                    (self.nxy // 2, self.nxy // 2, self.nxy // 2, self.nxy // 2, 0, 0),
                    mode="reflect",
                )
            self.clean_exitwaves = self.iterative_scattering(
                V_clean, pose=0, slice_batch_size=self.slice_batch_size
            )

        # scatter V
        self.exitwaves = self.iterative_scattering(
            V, pose=0, slice_batch_size=self.slice_batch_size
        )

        # aberrate exitwaves
        ctf_batch = {k: getattr(self, k)[idx] for k in self._ctf_param_names}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)

        # image/noise
        if self.anisomag is None:
            images = self.detector(
                self.detector_waves, nxy=self.nxy
            )  # nxy=None in original
        else:
            images = self.detector(
                self.detector_waves, self.anisomag[idx], nxy=self.nxy
            )
        return images


class TiltSeriesGenerator(MicrographGenerator):
    """
    Generates tilt series images by tilting the specimen.

    Parameters
    ----------
    scattering_potential : torch.Tensor
        Scattering potential volume.
    micrograph_size : int or tuple
        Micrograph size.
    pixel_size : float
        Pixel size in Angstroms.
    ctf_params : dict
        CTF parameters.
    energy : float
        Beam energy in kV.
    dose_per_angstrom : float
        Beam dose per square Angstrom.
    angles : list or torch.Tensor
        List of tilt angles in degrees.
    sample_size : int, optional
        Size of the sample volume. If None, calculated from angles.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    ice_model : str, optional
        Ice model.
    ice_thickness : float, optional
        Ice thickness in Angstroms.
    scattering_model : str, optional
        Scattering model. Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal space limit.
    alpha : float, optional
        Amplitude contrast ratio.
    crowd_min_distance : float, optional
        Crowding minimum distance.
    crowd_max_distance_z : float, optional
        Crowding maximum Z distance.
    pad_fft : bool, optional
        Whether to pad for FFT.
    chunk_size : int, optional
        Chunk size for processing.
    slice_batch_size : int, optional
        Number of Z slices sampled together during iterative scattering.
    max_tilt_angle_deg : float, optional
        Override the tilt angle used for XY-size validation. If None, inferred
        from `angles` or `quaternions`.
    move_to_cpu : bool, optional
        Move to CPU. Default True.
    water_air_interface : bool, optional
        Simulate water-air interface. Default True.
    detector_model : str, optional
        Detector model.
    vol : torch.Tensor, optional
        Pre-computed volume. If provided, `scattering_potential`, `crowd_min_distance`,
        `crowd_max_distance_z`, `ice_model`, `ice_thickness`, `water_air_interface`
        are ignored for volume generation.
    pad_volume : bool, optional
        If True (default), automatically pad the volume in XY using reflection
        when it is too small for the requested tilt coverage. Padding is applied
        symmetrically on both sides. If False, a warning is printed but the
        volume is used as-is (the old behaviour).
    taper_width : int, optional
        Number of pixels of additional reflect-padded apron to add on each XY
        side *beyond* the tilt-coverage padding, with a cosine taper applied
        over that apron (weight ramps from 1 at the inner edge to 0 at the
        outer edge). This eliminates the hard boundary that would otherwise
        appear when the tilted beam samples outside the volume's XY extent.
        Default is 0 (no apron).
    z_taper_width : int, optional
        Number of pixels of cosine taper to apply along the Z direction at
        the top and bottom of the volume. This smoothes the transition to
        zero (vacuum) and can reduce Fourier artifacts at high tilt angles.
        Default is 0.
    tilt_axis : str, optional
        The axis around which the sample is tilted ('x' or 'y'). Default is 'x'.
    """

    @staticmethod
    def _estimate_required_nxy(
        desired_nxy: int, nz: int, max_tilt_angle_deg: float
    ) -> int:
        """
        Approximate the minimum XY size required for the 3D volume so that,
        at max tilt, the projected span still covers desired_nxy pixels.
        """
        theta_rad = torch.deg2rad(max_tilt_angle_deg)
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)
        required_nxy = int(torch.ceil((desired_nxy + nz * sin_t) / cos_t))
        return required_nxy

    @staticmethod
    def _estimate_max_allowed_nxy(
        available_nxy: int, nz: int, max_tilt_angle_deg: float
    ) -> int:
        """
        Approximate the minimum XY size required for the 3D volume so that,
        at max tilt, the projected span still covers desired_nxy pixels.
        """
        theta_rad = torch.deg2rad(max_tilt_angle_deg)
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)
        allowed_nxy = int(torch.ceil(available_nxy * cos_t - nz * sin_t))
        return allowed_nxy

    @staticmethod
    def _estimate_max_allowed_tilt_deg(
        desired_nxy: int, nz: int, available_nxy: int
    ) -> float:
        """
        Approximate the largest tilt angle (in degrees) such that:
        available_nxy * cos(theta) - nz * sin(theta) >= desired_nxy.
        """
        thetas_deg = torch.linspace(0.0, 89.9, 4000)
        thetas_rad = torch.deg2rad(thetas_deg)
        spans = available_nxy * torch.cos(thetas_rad) - nz * torch.sin(thetas_rad)
        valid = spans >= desired_nxy
        if not bool(valid.any()):
            return 0.0
        return float(thetas_deg[valid][-1].item())

    @staticmethod
    def _infer_max_tilt_from_inputs(angles=None, quaternions=None) -> float:
        """Infer max tilt magnitude in degrees from provided poses."""
        if angles is not None:
            return angles.abs().max()
        if quaternions is not None:
            rotvecs = Rotation.from_quat(torch.as_tensor(quaternions)).as_rotvec()
            max_angle_rad = torch.linalg.norm(rotvecs, dim=-1).max()
            return max_angle_rad * (180.0 / torch.pi)
        return torch.tensor([0.0])

    @staticmethod
    def _pad_vol_xy_for_tilt(vol, required_nxy: int, available_nxy: int):
        """
        Pad `vol` symmetrically on both XY sides using reflect mode so that
        its XY extent reaches `required_nxy`.

        Parameters
        ----------
        vol : torch.Tensor
            Input volume of shape (..., Z, Y, X).
        required_nxy : int
            Target XY size after padding.
        available_nxy : int
            Current XY size of `vol`.

        Returns
        -------
        vol : torch.Tensor
            Padded volume.
        """
        pad_each_side = (required_nxy - available_nxy + 1) // 2
        return F.pad(
            vol,
            (
                pad_each_side,
                pad_each_side,  # x-axis (last dim)
                pad_each_side,
                pad_each_side,  # y-axis (second last dim)
                0,
                0,  # z-axis (no padding)
            ),
            mode="reflect",
        )

    @staticmethod
    def _get_cosine_window(n: int, taper_px: int, device, dtype):
        """Helper to create a 1D cosine window."""
        win = torch.ones(n, device=device, dtype=dtype)
        if taper_px <= 0:
            return win
        taper_px = min(taper_px, n // 2)
        if taper_px <= 0:
            return win
        ramp = 0.5 * (
            1
            - torch.cos(
                torch.pi * torch.linspace(0, 1, taper_px, device=device, dtype=dtype)
            )
        )
        win[:taper_px] = ramp
        win[-taper_px:] = ramp.flip(0)
        return win

    @staticmethod
    def _apply_cosine_taper(vol, taper_xy: int = 0, taper_z: int = 0):
        """
        Apply a cosine taper to the XY and/or Z edges of the volume. The taper
        ramps smoothly from 1 at `taper_px` from the edge to 0 at the outer edge.

        Parameters
        ----------
        vol : torch.Tensor
            Volume of shape (..., Z, Y, X).
        taper_xy : int
            Width of the taper in XY pixels. Pass 0 to skip.
        taper_z : int
            Width of the taper in Z pixels. Pass 0 to skip.

        Returns
        -------
        vol : torch.Tensor
            Volume with cosine taper applied.
        """
        if taper_xy <= 0 and taper_z <= 0:
            return vol

        nz, ny, nx = vol.shape[-3], vol.shape[-2], vol.shape[-1]
        device, dtype = vol.device, vol.dtype

        mask = torch.ones(1, device=device, dtype=dtype)

        if taper_xy > 0:
            win_y = TiltSeriesGenerator._get_cosine_window(ny, taper_xy, device, dtype)
            win_x = TiltSeriesGenerator._get_cosine_window(nx, taper_xy, device, dtype)
            mask = mask * win_y[:, None] * win_x[None, :]  # (Y, X)

        if taper_z > 0:
            win_z = TiltSeriesGenerator._get_cosine_window(nz, taper_z, device, dtype)
            if mask.ndim == 2:
                mask = win_z[:, None, None] * mask  # (Z, Y, X)
            else:
                mask = win_z[:, None, None]  # (Z, 1, 1)

        return vol * mask

    def __init__(
        self,
        vol: torch.Tensor,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any],
        energy: float,
        dose_per_angstrom: float,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        angles: torch.Tensor | Sequence[float] | None = None,
        anisomag: torch.Tensor | None = None,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        alpha: float = 0.0,
        pad_fft: bool = False,
        chunk_size: int | None = None,
        move_to_cpu: bool = False,
        detector_model: str | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        slice_batch_size: int = 1,
        pad_volume: bool = True,
        taper_width: int = 0,
        z_taper_width: int = 0,
        tilt_axis: str = "x",
        coincidence_radius: float = 0.0,
        num_frames: int | None = None,
        **kwargs: Any,
    ):
        if vol is None:
            raise ValueError("'vol' must be provided for TiltSeriesGenerator.")

        if isinstance(micrograph_size, int):
            desired_nxy = micrograph_size
        elif (
            isinstance(micrograph_size, (tuple, list))
            and len(micrograph_size) == 2
            and micrograph_size[0] == micrograph_size[1]
        ):
            desired_nxy = micrograph_size[0]
        else:
            raise ValueError("micrograph_size must have same dimensions in x and y.")

        self.tilt_axis = tilt_axis.lower()
        if self.tilt_axis not in ["x", "y"]:
            raise ValueError(f"Unsupported tilt_axis: {tilt_axis}. Use 'x' or 'y'.")

        max_tilt_angle_deg = self._infer_max_tilt_from_inputs(
            angles=angles, quaternions=quaternions
        )

        nz_input = int(vol.shape[-3])
        available_nxy = int(min(vol.shape[-2], vol.shape[-1]))
        required_nxy = self._estimate_required_nxy(
            desired_nxy=desired_nxy,
            nz=nz_input,
            max_tilt_angle_deg=max_tilt_angle_deg,
        )
        target_nxy = required_nxy + 2 * taper_width
        self.recommended_nxy_for_max_tilt = required_nxy
        self.max_tilt_angle_deg = float(max_tilt_angle_deg)
        self.max_allowed_tilt_deg_for_volume = self._estimate_max_allowed_tilt_deg(
            desired_nxy=desired_nxy, nz=nz_input, available_nxy=available_nxy
        )
        self.max_allowed_nxy = self._estimate_max_allowed_nxy(
            available_nxy=available_nxy,
            nz=nz_input,
            max_tilt_angle_deg=max_tilt_angle_deg,
        )

        if available_nxy < target_nxy:
            if pad_volume:
                vol = self._pad_vol_xy_for_tilt(vol, target_nxy, available_nxy)
                msg = (
                    "[TiltSeriesGenerator] Volume XY too small for requested tilt coverage"
                    + (" and taper" if taper_width > 0 else "")
                    + f"; padded (reflect) from {available_nxy} to {vol.shape[-1]} px in XY.\n"
                    f"  micrograph_size={desired_nxy}, requested_max_tilt={self.max_tilt_angle_deg:.2f} deg, "
                    f"required_volume_nxy>={required_nxy}"
                )
                if taper_width > 0:
                    msg += f", target_nxy (with taper)>={target_nxy}"
                print(msg + ".")
            else:
                print(
                    "[TiltSeriesGenerator] Input volume XY may be too small for requested tilt "
                    "coverage; proceeding anyway (pad_volume=False).\n"
                    f"  micrograph_size={desired_nxy}, volume_shape={tuple(vol.shape)}, "
                    f"requested_max_tilt={self.max_tilt_angle_deg:.2f} deg,\n"
                    f"  required_volume_nxy>={required_nxy}, current_volume_nxy={available_nxy}, \n"
                    f"  max_allowed_tilt_with_current_volume\u2248{self.max_allowed_tilt_deg_for_volume:.2f} deg,\n"
                    f"  max_allowed_nxy\u2248{self.max_allowed_nxy}."
                )

        if taper_width > 0 or z_taper_width > 0:
            vol = self._apply_cosine_taper(
                vol, taper_xy=int(taper_width), taper_z=int(z_taper_width)
            )
            if taper_width > 0:
                print(
                    f"[TiltSeriesGenerator] Applied cosine-taper over {taper_width} px "
                    f"at the XY edges."
                )
            if z_taper_width > 0:
                print(
                    f"[TiltSeriesGenerator] Applied cosine-taper over {z_taper_width} px "
                    f"at the Z edges (top/bottom)."
                )

        # Initialize parent (sets self.vol, self.nz, aberration, detector, etc.)
        # Pass move_to_cpu=False so we can try GPU first
        super().__init__(
            scattering_potential=None,
            micrograph_size=micrograph_size,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            vol=vol,
            anisomag=anisomag,
            scattering_model=scattering_model,
            aberration_model=aberration_model,
            noise_model=noise_model,
            klim=klim,
            alpha=alpha,
            pad_fft=pad_fft,
            chunk_size=chunk_size,
            move_to_cpu=move_to_cpu,  # we handle placement below
            detector_model=detector_model,
            progressbars=progressbars,
            verbose=verbose,
            slice_batch_size=slice_batch_size,
            coincidence_radius=coincidence_radius,
            num_frames=num_frames,
            **kwargs,
        )

        # Try GPU first for speed; fall back to CPU if VRAM insufficient
        if move_to_cpu:
            self.vol = self.vol.cpu()
            self._vol_device = "cpu"
            print("[TiltSeriesGenerator] Volume on CPU (move_to_cpu=True).", flush=True)
        elif torch.cuda.is_available():
            try:
                self.vol = self.vol.cuda()
                torch.cuda.synchronize()
                self._vol_device = "cuda"
                print("[TiltSeriesGenerator] Volume on GPU.", flush=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    self.vol = self.vol.cpu()
                    self._vol_device = "cpu"
                    print(
                        "[TiltSeriesGenerator] GPU VRAM insufficient; volume on CPU.",
                        flush=True,
                    )
                else:
                    raise
        else:
            self.vol = self.vol.cpu()
            self._vol_device = "cpu"
            print(
                "[TiltSeriesGenerator] CUDA not available; volume on CPU.", flush=True
            )

        self.slice_batch_size = slice_batch_size
        self.iterative_scattering = IterativeScattering(
            desired_nxy,
            pixel_size,
            energy,
            dose_per_angstrom,
            scattering_model=scattering_model,
            klim=klim,
            alpha=alpha,
            progressbars=progressbars,
        )

        if quaternions is not None:
            self.register_buffer("quaternions", torch.as_tensor(quaternions))
            if translations is not None:
                self.register_buffer("translations", torch.as_tensor(translations))
            else:
                self.register_buffer("translations", torch.zeros(len(quaternions), 2))
            self.angles = None
        elif angles is not None:
            self.angles = torch.as_tensor(angles)
            B = len(self.angles)
            theta_rad = torch.deg2rad(self.angles)

            if self.tilt_axis == "x":
                rotvecs = torch.stack(
                    [
                        theta_rad,
                        torch.zeros_like(theta_rad),
                        torch.zeros_like(theta_rad),
                    ],
                    dim=-1,
                )
            else:  # 'y'
                rotvecs = torch.stack(
                    [
                        torch.zeros_like(theta_rad),
                        theta_rad,
                        torch.zeros_like(theta_rad),
                    ],
                    dim=-1,
                )

            quats = Rotation.from_rotvec(rotvecs).as_quat()
            self.register_buffer("quaternions", quats)
            self.register_buffer("translations", torch.zeros(B, 2))
        else:
            raise ValueError("Either 'angles' or 'quaternions' must be provided.")

    def get_nz_tilt(self, V: torch.Tensor, theta_matrix: torch.Tensor) -> int:
        """
        Calculate the number of slices needed to cover a transformed volume.

        Parameters
        ----------
        V : torch.Tensor
            Input volume of shape (B, Z, Y, X).
        theta_matrix : torch.Tensor
            Affine transformation matrix.

        Returns
        -------
        nz_new : int
            Number of slices needed.
        """
        B, Z, Y, X = V.shape
        device = V.device
        theta_matrix = theta_matrix.to(V.device)
        R = theta_matrix[:, :3, :3]

        # Corners in pixel units
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
            device=device,
            dtype=V.dtype,
        ).t()  # (3, 8)

        # rotated_corners = R_inv @ corners
        rotated_corners = torch.bmm(
            R.transpose(1, 2), corners.unsqueeze(0).expand(B, -1, -1)
        )
        z_min = rotated_corners[:, 2, :].min(dim=1).values
        z_max = rotated_corners[:, 2, :].max(dim=1).values

        # Max extent across the batch
        nz_new = int(torch.ceil((z_max - z_min).max()).item())
        return max(1, nz_new)

    def generate_tilt_series(
        self, idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a tilt series for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices (to select CTF/Anisomag parameters).

        Returns
        -------
        tilt_series : torch.Tensor
            Tensor of images. Shape (B, N_angles, Y, X).
        exitwaves : torch.Tensor
            Tensor of exitwaves.
        clean_images : torch.Tensor
            Tensor of clean images.
        """
        if self._vol_device == "cuda" and self.vol.device != self.device:
            self.vol = self.vol.to(self.device)

        tilt_series = []
        exitwaves = []
        clean_images = []
        B = len(idx)
        n_frames = len(self.quaternions)

        for i in track(
            range(n_frames),
            description="Generating tilt series.",
            disable=not (self.progressbars),
        ):
            Q = self.quaternions[i].unsqueeze(0).expand(B, -1)
            T = self.translations[i].unsqueeze(0).expand(B, -1)

            # Construct theta_matrix
            R_mat = Rotation.from_quat(Q).as_matrix()
            if R_mat.ndim == 2:
                R_mat = R_mat.unsqueeze(0)
            # translations in angstrom to torch normalized
            T_torch = rotations.translations_angstrom_to_torch(
                T, self.vol.shape[-1], self.pixel_size
            )
            theta_matrix = rotations.build_affine_matrix(R_mat, T_torch)

            exitwave = self.iterative_scattering(
                self.vol, theta_matrix, slice_batch_size=self.slice_batch_size
            )

            # 3. Aberration
            ctf_batch = {
                k: getattr(self, k)[idx].clone() for k in self._ctf_param_names
            }
            # Adjust defocus for propagation in multislice/rytov/firstborn; skip for projection/ctf
            if self.scattering_model not in ["projection", "ctf"]:
                nz_new = self.get_nz_tilt(self.vol, theta_matrix)
                z_offset = (nz_new - self.nz) * self.pixel_size / 2.0
                if "dfu" in ctf_batch:
                    ctf_batch["dfu"] = ctf_batch["dfu"] - z_offset
                if "dfv" in ctf_batch:
                    ctf_batch["dfv"] = ctf_batch["dfv"] - z_offset

            detector_waves = self.aberration(exitwave, ctf_batch)

            # 4. Detection
            if self.anisomag is None:
                image = self.detector(detector_waves, nxy=None)
            else:
                image = self.detector(detector_waves, self.anisomag[idx], nxy=None)

            tilt_series.append(image.detach().cpu())
            exitwaves.append(exitwave.detach().cpu())
            clean_images.append(torch.abs(detector_waves.detach().cpu()) ** 2)

        return (
            torch.stack(tilt_series, dim=1),
            torch.stack(exitwaves, dim=1),
            torch.stack(clean_images, dim=1),
        )  # (B, N_angles, Y, X)
