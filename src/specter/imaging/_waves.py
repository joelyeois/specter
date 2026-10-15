"""
`ExitWave` and `Image`: the wave leaving the specimen, and what is recorded.

Each carries its own grid (the pixel size, the FFT-padded box, how many
image-grid points a pixel holds) and the reference plane its defocus is
measured from, so the stage that consumes it never re-derives them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import mrcfile
import torch

from ..aberrations import FocalSpread
from ..aberrations._envelopes import cc_envelope
from ..constants import energy_to_wavelength
from ..detectors import detector_mtf, dqe0_for_detector
from ..microscope import Detector
from ..options import AberrationModel, NoiseModel, TemporalCoherence
from . import _stages as stages
from ._data import CTF, Exposure, per_image
from ._engines import aberration_engine

__all__ = ["ExitWave", "Image"]


def _seeds(
    seed: int | None, ids: torch.Tensor, stage: int, tilt: int | None = None
) -> list[int] | None:
    if seed is None:
        return None
    from ..random_seed import derive_seed

    keys = () if tilt is None else (tilt,)
    return [derive_seed(seed, int(i), stage, *keys) for i in ids.tolist()]


@dataclass(frozen=True)
class ExitWave:
    """
    The wave leaving the specimen, one per image.

    Attributes
    ----------
    array : torch.Tensor
        ``(n, Y, X)``: complex, on the propagation grid (``oversample``
        times finer than the pixels) or, for the models that never leave it,
        on the pixel grid; real for the linear ``"ctf"`` model, which
        propagates a projected potential rather than a wave.
    pixel_size : float
        Of the pixel grid, in Angstrom.
    voltage : float
        In kV.
    nxy, pad_nxy : int
        The image, and the FFT-padded pixel grid it was propagated on.
    oversample : int
        Image-grid points per pixel, per axis: the grid the image is formed
        on. Multislice, Rytov and projection waves are already on it; first
        Born and kinematic propagate on the pixel grid, and :meth:`apply_ctf`
        interpolates them onto it, so until then their ``array`` is
        ``oversample`` times coarser than this says. Read the array's own
        grid from its shape.
    defocus_shift : float
        Angstrom from the exit face, where the wave was computed, to the
        plane the nominal defocus refers to (the specimen's scattering
        centroid, normally the volume's midplane); subtracted from the
        defocus by :meth:`apply_ctf`.
    ids : torch.Tensor
        The images' indices in their stack; for a tilt, the series'.
    tilt : int, optional
        Which tilt of its series the wave is, when it is one; the noise is
        then drawn per series and per tilt.
    linear : bool
        Imaged by the linear (weak-phase) model.
    alpha : float
        Amplitude contrast, folded into the linear model's transfer function.
    aberrated : bool
        Whether the transfer function has been applied.
    damaged, solvent_filtered : bool
        Whether the dose envelope, or the solvent's exposure filter, has
        already acted; :meth:`apply_ctf` refuses to apply it a second time.
    """

    array: torch.Tensor
    pixel_size: float
    voltage: float
    nxy: int
    pad_nxy: int
    oversample: int
    defocus_shift: float
    ids: torch.Tensor
    tilt: int | None = None
    linear: bool = False
    alpha: float = 0.0
    aberrated: bool = False
    damaged: bool = False
    solvent_filtered: bool = False

    def pixels(self) -> torch.Tensor:
        """The wave on the (FFT-padded) pixel grid."""
        return stages.to_pixel_grid(self.array, self.pad_nxy, self.oversample)

    def apply_ctf(
        self,
        ctf: CTF,
        *,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        dose_envelope: Exposure | None = None,
        backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
    ) -> ExitWave:
        """
        Apply the objective lens's transfer function.

        Parameters
        ----------
        ctf : CTF
            One set of terms per image, the defocus that of the specimen's
            midplane.
        bfactor : float or torch.Tensor, optional
            Isotropic B-factor envelope, in A^2, one value or one per image.
        convergence_angle : float, optional
            Illumination semi-angle in mrad, for the spatial-coherence
            envelope.
        dose_envelope : Exposure, optional
            Fade the transfer function by each exposure's radiation damage.
            Refused when the specimen was already damaged or its solvent
            already decorrelated, which would count the exposure twice.
        backend : {"legacy", "torch_ctf"}, optional
            The transfer-function engine. Default "legacy".
        lpp_params : dict, optional
            Laser phase plate settings; ``"torch_ctf"`` only.

        Returns
        -------
        ExitWave
            At the detector, on the image grid.
        """
        if self.aberrated:
            raise ValueError("the transfer function has already been applied")
        if dose_envelope is not None and (self.damaged or self.solvent_filtered):
            raise ValueError(
                "the exposure already acted on the specimen or its solvent; a "
                "dose envelope on the transfer function would count it twice"
            )
        n, device = len(self.ids), self.array.device
        if len(ctf) != n:
            raise ValueError(f"{len(ctf)} CTF entries for {n} images")
        if ctf.ids is not None and not torch.equal(ctf.ids.cpu(), self.ids.cpu()):
            raise ValueError("the CTF belongs to different images than the wave")
        params = {k: v.to(device) for k, v in ctf.params.items()}
        if self.defocus_shift:
            for name in ("dfu", "dfv"):
                if name in params:
                    params[name] = params[name] - self.defocus_shift
        bfactors = None if bfactor is None else per_image(bfactor, n, "bfactor")
        if dose_envelope is not None:
            params["dose"] = dose_envelope.doses(n).to(device)
            if dose_envelope.pre_exposure is not None:
                params["pre_exposure"] = dose_envelope.pre_exposure.to(device)
        weighted = True if dose_envelope is None else dose_envelope.weighted
        s = self.oversample
        model: AberrationModel = "linear" if self.linear else "nonlinear"
        aberration: torch.nn.Module
        if backend == "torch_ctf" and bfactors is not None:
            # This engine takes the B-factor at construction, not per call.
            from ..ctf import LegacyAberrationAdapter

            aberration = LegacyAberrationAdapter(
                self.pad_nxy * s,
                self.pixel_size / s,
                self.voltage,
                aberration_model=model,
                specimen_absorption=not self.linear,
                bfactor=bfactors.to(device),
                convergence_angle=convergence_angle,
                dose_envelope=dose_envelope is not None,
                dose_weighted=weighted,
                lpp_params=lpp_params,
            ).to(device)
        else:
            if lpp_params is not None and backend != "torch_ctf":
                raise ValueError("lpp_params needs backend='torch_ctf'")
            if bfactors is not None:
                params["bfactor"] = bfactors.to(device)
            aberration = aberration_engine(
                self.pad_nxy * s,
                self.pixel_size / s,
                self.voltage,
                model,
                self.alpha if self.linear else None,
                convergence_angle,
                dose_envelope is not None,
                weighted,
                backend,
                None if lpp_params is None else tuple(sorted(lpp_params.items())),
                str(device),
            )
        wave = stages.aberrate(self.array, aberration, params, s, self.pad_nxy)
        return replace(
            self,
            array=wave,
            aberrated=True,
            damaged=self.damaged or dose_envelope is not None,
        )

    def intensity(
        self,
        *,
        cc: float | None = 2.7,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        temporal_coherence: TemporalCoherence = "integrated",
    ) -> Image:
        """
        The intensity the wave deposits, relative to the vacuum beam.

        Parameters
        ----------
        cc : float or None, optional
            Chromatic aberration in millimetres; the image is the incoherent
            sum over the focal spread it causes. None or 0 for a coherent
            beam. Default 2.7.
        energy_spread : float, optional
            Beam energy spread, FWHM in eV. Default 0.7.
        deltaV_V, deltaI_I : float, optional
            Fractional instabilities of the high tension and the objective
            lens current.
        temporal_coherence : {"integrated", "envelope"}, optional
            ``"integrated"`` sums the image over the focal spread;
            ``"envelope"`` damps the wave by its Gaussian average first, the
            approximation that is exact only for the image's linear term.
            The linear model is all linear term and always takes the
            envelope. Default "integrated".

        Returns
        -------
        Image
            On the wave's grid.

        References
        ----------
        .. [1] Ishizuka, K. (1980). Contrast transfer of crystal images in TEM.
           *Ultramicroscopy*, 5(1–3), 55–65.
           https://doi.org/10.1016/0304-3991(80)90011-X
        .. [2] Wade, R. H., & Frank, J. (1977). Electron microscope transfer
           functions for partially coherent axial illumination and chromatic
           defocus spread. *Optik*, 49, 81–92.
        """
        grid = self.array.shape[-1] // self.pad_nxy
        dx = self.pixel_size / grid
        wavelength = energy_to_wavelength(self.voltage)
        wave = self.array
        if cc and (self.linear or temporal_coherence == "envelope"):
            n = wave.shape[-1]
            k = torch.fft.fftfreq(n, dx, device=wave.device, dtype=wave.real.dtype)
            envelope = cc_envelope(
                k[:, None] ** 2 + k[None, :] ** 2,
                wavelength,
                cc * 1e7,
                self.voltage * 1e3,
                energy_spread,
                deltaV_V,
                deltaI_I,
            )
            wave = torch.fft.ifft2(torch.fft.fft2(wave) * envelope)
            if self.linear:
                wave = wave.real
        if self.linear:
            values = wave + 1
        elif cc and temporal_coherence == "integrated":
            focal = FocalSpread.from_instrument(
                cc * 1e7,
                self.voltage * 1e3,
                energy_spread,
                deltaV_V,
                deltaI_I,
                wavelength,
            )
            values = focal.intensity(wave, dx, grid)
        else:
            values = wave.abs().square()
        return Image(
            array=values,
            pixel_size=self.pixel_size,
            nxy=self.nxy,
            pad_nxy=self.pad_nxy,
            ids=self.ids,
            tilt=self.tilt,
        )

    @staticmethod
    def cat(waves: list[ExitWave]) -> ExitWave:
        """Stack batches of the same imaging run."""
        return replace(
            waves[0],
            array=torch.cat([w.array for w in waves]),
            ids=torch.cat([w.ids for w in waves]),
        )


@dataclass(frozen=True)
class Image:
    """
    An intensity per image: relative to the vacuum beam, then counts.

    Attributes
    ----------
    array : torch.Tensor
        Real ``(n, Y, X)``.
    pixel_size : float
        Of the pixel grid, in Angstrom.
    nxy, pad_nxy : int
        The image, and the FFT-padded pixel grid.
    ids : torch.Tensor
        The images' indices in their stack; for a tilt, the series'.
    tilt : int, optional
        Which tilt of its series the image is, when it is one.
    detected : bool
        Whether :meth:`detect` has turned the intensity into counts.
    """

    array: torch.Tensor
    pixel_size: float
    nxy: int
    pad_nxy: int
    ids: torch.Tensor
    tilt: int | None = None
    detected: bool = False

    def pixels(self) -> torch.Tensor:
        """The intensity on the (FFT-padded) pixel grid."""
        grid = self.array.shape[-1] // self.pad_nxy
        return stages.to_pixel_grid(self.array, self.pad_nxy, grid)

    def detect(
        self,
        exposure: Exposure,
        *,
        detector: str | None = None,
        noise: NoiseModel | None = "poisson",
        coincidence_radius: float | torch.Tensor = 0.0,
        anisomag: torch.Tensor | None = None,
        detector_pixel_size: float | None = None,
        seed: int | None = None,
    ) -> Image:
        """
        Record the intensity: dose, the detector's response, the noise.

        Parameters
        ----------
        exposure : Exposure
            Dose per image, and its frames.
        detector : str, optional
            A bundled detector model, for its MTF and DQE(0).
        noise : {"poisson"} or None, optional
            Default "poisson".
        coincidence_radius : float or torch.Tensor, optional
            In physical pixels, per image. Default 0.
        anisomag : torch.Tensor, optional
            ``(n, 2, 2)`` anisotropic magnification.
        detector_pixel_size : float, optional
            The camera's physical pixel, when the images were resampled.
        seed : int, optional
            Run seed; each image's noise is drawn from
            ``(seed, ids[i], detector stage)``.

        Returns
        -------
        Image
            Counts on the ``nxy`` pixel grid.
        """
        if self.detected:
            raise ValueError("the image has already been detected")
        if noise == "none":
            noise = None
        n, device = len(self.ids), self.array.device
        mtf = detector_mtf(
            detector, self.nxy, self.pixel_size, physical_pixel_size=detector_pixel_size
        )
        camera = Detector(
            self.pixel_size,
            noise_model=noise,
            mtf=mtf,
            dqe0=dqe0_for_detector(detector),
            n_frames=exposure.n_frames,
            dose_weights=exposure.weights,
            dose_weights_max_frequency=exposure.weights_max_frequency,
            progressbars=False,
            oversample=self.array.shape[-1] // self.pad_nxy,
        ).to(device)
        dose = exposure.doses(n).to(device)
        expected = (dose * self.pixel_size**2 * camera.dqe0)[:, None, None] * self.array
        counts = camera.detect_expected_counts(
            expected,
            dose,
            per_image(coincidence_radius, n, "coincidence_radius").to(device),
            None if anisomag is None else anisomag.to(device),
            nxy=self.nxy,
            seeds=_seeds(seed, self.ids, stages.SEED_STAGE_DETECTOR, self.tilt),
        )
        return replace(self, array=counts, pad_nxy=self.nxy, detected=True)

    def save(self, path: str) -> None:
        """Write the stack as an MRC file."""
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(self.array.detach().cpu().float().numpy())
            mrc.voxel_size = self.pixel_size

    @staticmethod
    def cat(images: list[Image]) -> Image:
        """Stack batches of the same imaging run."""
        return replace(
            images[0],
            array=torch.cat([i.array for i in images]),
            ids=torch.cat([i.ids for i in images]),
        )
