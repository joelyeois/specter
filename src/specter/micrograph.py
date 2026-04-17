from __future__ import annotations

from typing import Any

import torch

from specter import logger

from .base_imager import BaseImager, compute_nz, pad_volume
from .scattering import IterativeScattering
from .specimen import TomogramGenerator


class MicrographGenerator(BaseImager):
    """
    Generates large micrographs by processing a full specimen volume.

    The volume is either supplied directly (``vol``) or assembled internally
    via ``TomogramGenerator`` from a ``scattering_potential`` template with
    optional crowding and ice.  Scattering is performed slice-by-slice using
    ``IterativeScattering``.

    Parameters
    ----------
    scattering_potential : torch.Tensor or None
        Template potential (Z, Y, X) used by ``TomogramGenerator`` to build
        the specimen volume.  Must be ``None`` when ``vol`` is provided.
    micrograph_size : int or tuple[int, int]
        Output image size in pixels (must be square).
    pixel_size : float
        Pixel size in Å.
    ctf_params : dict[str, torch.Tensor]
        CTF parameters; each value is a 1-D tensor of length n.
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose per Å². Scalar or 1-D tensor of length n.
    vol : torch.Tensor, optional
        Pre-assembled specimen volume of shape (1, Z, Y, X).  When provided,
        ``scattering_potential``, crowding, and ice parameters are ignored.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices, shape (n, 2, 2).
    ice_model : str, optional
        Ice model passed to ``TomogramGenerator`` ('randomchoice' or 'iterative').
    ice_thickness : float, optional
        Ice thickness in Å passed to ``TomogramGenerator``.
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
        Chunk size for ``TomogramGenerator`` parallel processing.
    move_to_cpu : bool, optional
        Move the assembled volume to CPU after generation to save GPU memory.
        Default True.
    detector_model : str, optional
        Detector MTF model ('k3_300kv', 'k3_200kv', 'perfect', None).
    slice_batch_size : int, optional
        Number of Z slices propagated together in ``IterativeScattering``.
        Default 1.
    progressbars : bool, optional
        Show progress bars. Default True.
    verbose : bool, optional
        Emit debug-level log messages. Default True.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Default 0.0.
    num_frames : int, optional
        Number of detector frames to simulate. Default None.
    potential_scale : float or torch.Tensor, optional
        Multiplier applied to the potential before scattering. Default 1.0.
    save_clean_exitwaves : bool, optional
        Save exit waves computed without ice (requires ``scattering_potential``
        path). Default False.
    """

    def __init__(
        self,
        scattering_potential: torch.Tensor | None,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any],
        energy: float,
        dose_per_angstrom: float | torch.Tensor,
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
        coincidence_radius: float | torch.Tensor = 0.0,
        num_frames: int | None = None,
        potential_scale: float | torch.Tensor = 1.0,
        save_clean_exitwaves: bool = False,
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

        if vol is not None:
            self.nz = vol.shape[1]
        elif scattering_potential is not None:
            self.nz = compute_nz(
                scattering_potential.shape[0], ice_thickness, pixel_size
            )
        else:
            raise ValueError("Either 'vol' or 'scattering_potential' must be provided.")
        self.ice_thickness = self.nz * pixel_size

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
        self.slice_batch_size = slice_batch_size
        self.iterative_scattering = IterativeScattering(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            scattering_model=self.scattering_model,
            klim=self.klim,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

        self.save_clean_exitwaves = save_clean_exitwaves

        if vol is not None:
            self.vol = vol
        else:
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

    def regenerate_specimen(self) -> None:
        """
        Regenerate the specimen volume with fresh ice and crowding placement.

        Allows multiple independent micrographs from a single model instance
        without reinstantiating.

        Raises
        ------
        RuntimeError
            If the model was constructed with a pre-built ``vol`` (no
            ``specimen_gen`` available).
        """
        if not hasattr(self, "specimen_gen"):
            raise RuntimeError(
                "regenerate_specimen() requires the model to have been constructed "
                "with a scattering_potential, not a pre-built vol."
            )
        self.vol = self.specimen_gen.generate()
        if self.move_to_cpu:
            self.vol = self.vol.cpu()

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
        V = self.vol.to(self.device).expand(len(idx), -1, -1, -1)
        V = pad_volume(V, self.nxy, self.nz, None, self.pad_fft, xy_pad_mode="reflect")

        if (
            self.save_clean_exitwaves
            and hasattr(self, "specimen_gen")
            and hasattr(self.specimen_gen, "clean_V")
        ):
            V_clean = self.specimen_gen.clean_V.to(self.device).expand(
                len(idx), -1, -1, -1
            )
            V_clean = pad_volume(
                V_clean, self.nxy, self.nz, None, self.pad_fft, xy_pad_mode="reflect"
            )
            self.clean_exitwaves = self.iterative_scattering(
                V_clean, pose=0, slice_batch_size=self.slice_batch_size
            )

        self.exitwaves = self.iterative_scattering(
            V, pose=0, slice_batch_size=self.slice_batch_size
        )

        ctf_batch = {k: getattr(self, k)[idx] for k in self._ctf_param_names}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)

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
