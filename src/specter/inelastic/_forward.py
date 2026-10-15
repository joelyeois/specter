"""Exposure-resolved zero-loss forward model using SPECTER's production optics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import torch
from torch import nn

from specter.scattering import IterativeScattering
from specter.scattering._kernels import fourier_crop_xy
from specter.microscope import Detector
from ._exposure import ExposureStep, exposure_steps
from ._specimen import PotentialState
from ._spectrum import PlasmonFilter


@dataclass
class FrozenPlasmonResult:
    """
    One exposure, frame by frame.

    Attributes
    ----------
    images : torch.Tensor
        Detected counts, ``(frames, B, Y, X)``, on the field's pixel grid.
    intensities : torch.Tensor
        The dose-averaged intensity each frame deposits before detection,
        relative to the vacuum beam, on the same pixel grid as `images`:
        sampled from the propagation grid exactly as the detector samples it
        (a Fourier crop, then the field's centre), before the anisotropic
        magnification, MTF, DQE and noise. With none of those, ``images`` is
        ``intensities`` times each frame's dose and the pixel area. Keeps
        gradients.
    frame_doses : torch.Tensor
        Each frame's dose, e-/A^2.
    metadata : dict
        Provenance: the spectrum, the source calibration, the sampling.
    """

    images: torch.Tensor
    intensities: torch.Tensor
    frame_doses: torch.Tensor
    metadata: dict

    @property
    def summed_image(self) -> torch.Tensor:
        return self.images.sum(0)


class FrozenPlasmonForward(nn.Module):
    """Integrate frozen states through paired multislice, optics and detection.

    Supply a state provider accepting an ExposureStep and returning paired
    fields. A provider with ``reset()`` is replayed for every forward call.
    Optics receives ``(exitwave, step)`` so damage/defocus may follow dose.
    Every frame is a physical detector readout: the detector must have
    ``n_frames=None`` or 1 and no external dose weights. Artificial numerical
    substeps never create additional detector readouts.

    With ``plasmon_filter=None``, sources are interpreted as direct imaginary
    potentials, providing the MFP comparison path through identical operators.

    References
    ----------
    .. [1] Himes, B., & Grigorieff, N. (2021). Cryo-TEM simulations of amorphous
       radiation-sensitive samples using multislice wave propagation. *IUCrJ*,
       8(6), 943–953. https://doi.org/10.1107/S2052252521008538
    """

    def __init__(
        self,
        scattering: IterativeScattering,
        detector: Detector,
        plasmon_filter: PlasmonFilter | None,
        *,
        optics: Callable[[torch.Tensor, ExposureStep], torch.Tensor] | None = None,
    ):
        super().__init__()
        if scattering.scattering_model != "multislice" or scattering.alpha != 0:
            raise ValueError("frozen plasmon requires multislice with alpha=0")
        if detector.aberration_model != "nonlinear":
            raise ValueError("frozen plasmon requires a nonlinear detector")
        if detector.pixel_size != scattering.pixel_size:
            raise ValueError("detector and propagator pixel sizes must agree")
        if detector.n_frames not in (None, 1) or detector.dose_weights is not None:
            raise ValueError(
                "supply a single-readout detector; temporal fractionation is explicit"
            )
        if plasmon_filter is not None and (
            plasmon_filter.pixel_size != scattering.pixel_size
            or plasmon_filter.voltage != scattering.voltage
        ):
            raise ValueError(
                "plasmon filter and propagator sampling/voltage must agree"
            )
        self.scattering = scattering
        self.detector = detector
        self.plasmon_filter = plasmon_filter
        self.optics = optics

    def _on_pixel_grid(self, intensity: torch.Tensor, nxy: int | None) -> torch.Tensor:
        """
        `intensity` on the grid the detector records it on: Fourier-cropped
        from the propagation grid, then the field's centre, the same two
        steps and arithmetic :meth:`Detector.detect_expected_counts` begins
        with, so it lines up pixel for pixel with the detected images.
        """
        if self.detector.oversample > 1:
            intensity = fourier_crop_xy(
                intensity, intensity.shape[-1] // self.detector.oversample
            )
        if nxy is None or nxy == intensity.shape[-1]:
            return intensity
        cy, cx = intensity.shape[-2] // 2, intensity.shape[-1] // 2
        half = nxy // 2
        return intensity[
            ...,
            cy - half : cy + half + (nxy % 2),
            cx - half : cx + half + (nxy % 2),
        ]

    def forward(
        self,
        specimen: Callable[[ExposureStep], PotentialState],
        frame_doses: Sequence[float],
        *,
        pose: float | torch.Tensor = 0.0,
        poses: Sequence[float | torch.Tensor] | None = None,
        substeps: int = 1,
        pre_exposure: float = 0.0,
        coincidence_radius: float = 0.0,
        anisomag: torch.Tensor | None = None,
        nxy: int | None = None,
        checkpoint_chunks: int | None = None,
        slice_batchsize: int = 1,
        detector_seed: int | None = None,
    ) -> FrozenPlasmonResult:
        """Render a movie, or a tilt sequence with one pose per physical frame.

        ``poses`` follows acquisition order. Exposure advances monotonically
        across tilts, so a moving specimen and damage share cumulative dose.
        Expected intensities retain gradients; stochastic images do not have
        useful reconstruction gradients. Use ``noise_model=None`` to fit data.
        """
        steps = exposure_steps(frame_doses, substeps, pre_exposure)
        if poses is not None and len(poses) != len(frame_doses):
            raise ValueError("poses must contain one pose per physical frame")
        if coincidence_radius < 0 or not torch.isfinite(
            torch.tensor(coincidence_radius)
        ):
            raise ValueError("coincidence radius must be finite and nonnegative")
        if coincidence_radius and self.detector.n_frames != 1:
            raise ValueError("coincidence requires detector.n_frames=1")
        reset = getattr(specimen, "reset", None)
        if reset is not None:
            reset()
        integrated = None
        intensities, images = [], []
        device = self.scattering.device
        # Scope the detector seed only around detector sampling below; do not
        # change specimen trajectories or the caller's random-number stream.
        for index, step in enumerate(steps):
            state = specimen(step)
            if state.elastic.ndim != 4:
                raise ValueError("state fields must have shape (B,Z,Y,X)")
            wave = self.scattering(
                state.elastic,
                pose if poses is None else poses[step.frame],
                slice_batchsize=slice_batchsize,
                checkpoint_chunks=checkpoint_chunks,
                absorption_source=state.absorption_source,
                absorption_filter=self.plasmon_filter,
            )
            if self.optics is not None:
                wave = self.optics(wave, step)
            contribution = self.detector.intensity(wave) * step.dose
            integrated = (
                contribution if integrated is None else integrated + contribution
            )
            if (index + 1) % substeps:
                continue
            intensity = integrated / frame_doses[step.frame]
            intensities.append(self._on_pixel_grid(intensity, nxy))
            dose = intensity.new_full((len(intensity),), frame_doses[step.frame])
            radius = torch.full_like(dose, coincidence_radius)

            def detect() -> torch.Tensor:
                return self.detector.from_intensity(
                    intensity, dose, radius, anisomag, nxy
                )

            if detector_seed is None:
                images.append(detect())
            else:
                devices = (
                    [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                    if device.type == "cuda"
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(
                        detector_seed + step.frame
                    )
                    if device.type == "cuda":
                        torch.cuda.default_generators[devices[0]].manual_seed(
                            detector_seed + step.frame
                        )
                    images.append(detect())
            integrated = None
        return FrozenPlasmonResult(
            torch.stack(images),
            torch.stack(intensities),
            torch.as_tensor(frame_doses, device=device),
            {
                "spectrum": None
                if self.plasmon_filter is None
                else self.plasmon_filter.provenance,
                "source_calibration": getattr(
                    specimen, "provenance", "user-supplied state provider"
                ),
                "substeps": substeps,
                "pre_exposure": pre_exposure,
                "detector_seed": detector_seed,
                "channel": "zero-loss",
                "motion_seed": getattr(specimen, "seed", None),
                "water_msd_per_dose": getattr(specimen, "water_msd_per_dose", None),
                "species_strengths_V_A3": getattr(specimen, "species_strengths", None),
                "water_strength_V_A3": getattr(specimen, "water_strength", None),
                "damage_model": type(getattr(specimen, "damage_model", None)).__name__,
                "pixel_size_A": self.scattering.pixel_size,
                "voltage_kV": self.scattering.voltage,
                "plasmon_halo_pixels": None
                if self.plasmon_filter is None
                else self.plasmon_filter.halo,
            },
        )
