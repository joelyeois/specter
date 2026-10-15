"""
An exposure simulated as explicit frozen states, with the chain's optics and
camera: the :class:`~specter.inelastic.FrozenPlasmonForward` engine, set up
from the same keywords :meth:`~specter.imaging.ExitWave.apply_ctf`,
:meth:`~specter.imaging.ExitWave.intensity` and
:meth:`~specter.imaging.Image.detect` take.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from ..aberrations import FocalSpread
from ..aberrations._functions import defocus_midplane_shift
from ..constants import energy_to_wavelength
from ..detectors import detector_mtf, dqe0_for_detector
from ..microscope import Detector
from ..options import EwaldSphereSign, NoiseModel
from ..scattering import IterativeScattering
from . import _stages as stages
from ._data import CTF, per_image
from ._engines import aberration_engine

if TYPE_CHECKING:
    from ..inelastic import (
        ExposureStep,
        FrozenPlasmonResult,
        PlasmonFilter,
        PotentialState,
    )

__all__ = ["simulate_frozen"]


class _Depth:
    """The specimen callback, recording the depth of the states it returns."""

    def __init__(self, specimen: Callable[[ExposureStep], PotentialState]) -> None:
        self.specimen = specimen
        self.nz: int | None = None

    def __call__(self, step: ExposureStep) -> PotentialState:
        state = self.specimen(step)
        self.nz = int(state.elastic.shape[1])
        return state

    def reset(self) -> None:
        reset = getattr(self.specimen, "reset", None)
        if reset is not None:
            reset()


def simulate_frozen(
    specimen: Callable[[ExposureStep], PotentialState],
    plasmon_filter: PlasmonFilter | None,
    frame_doses: Sequence[float],
    *,
    voxel_size: float,
    voltage: float,
    nxy: int,
    ctf: CTF | None = None,
    pose: float | torch.Tensor = 0.0,
    poses: Sequence[float | torch.Tensor] | None = None,
    substeps: int = 1,
    pre_exposure: float = 0.0,
    pad_fft: bool = False,
    oversample: int = 2,
    klim: float | None = None,
    ews_curvature_sign: EwaldSphereSign = "negative",
    convergence_angle: float | None = None,
    cc: float | None = 2.7,
    energy_spread: float = 0.7,
    deltaV_V: float = 0.06e-6,
    deltaI_I: float = 0.01e-6,
    detector: str | None = None,
    noise: NoiseModel | None = "poisson",
    coincidence_radius: float = 0.0,
    anisomag: torch.Tensor | None = None,
    seed: int | None = None,
    checkpoint_chunks: int | None = None,
    device: torch.device | str = "cpu",
) -> FrozenPlasmonResult:
    """
    Render an exposure frame by frame from explicit frozen specimen states.

    Parameters
    ----------
    specimen : callable
        Returns each step's :class:`~specter.inelastic.PotentialState`; it
        owns all solvent, crowding, scaling and damage.
    plasmon_filter : PlasmonFilter or None
        The transverse plasmon operator, or None for elastic scattering only.
    frame_doses : sequence of float
        The physical frames' doses, e-/A^2.
    voxel_size, voltage, nxy
        Sampling (Angstrom), beam (kV) and image size (pixels).
    ctf : CTF, optional
        One set of terms for the whole movie, the defocus that of the
        specimen's midplane. None images with no objective lens.
    pose, poses
        One pose for the movie, or one per physical frame for a tilt
        acquisition.
    substeps, pre_exposure
        Frozen states per frame, and the dose received before the movie.
    pad_fft, oversample, klim, ews_curvature_sign
        As `Propagation`.
    convergence_angle
        As :meth:`~specter.imaging.ExitWave.apply_ctf`.
    cc, energy_spread, deltaV_V, deltaI_I
        As :meth:`~specter.imaging.ExitWave.intensity`; the focal spread is
        summed over in every frame.
    detector, noise, coincidence_radius, anisomag
        As :meth:`~specter.imaging.Image.detect`; each frame is one readout.
    seed : int, optional
        The detector's seed; the specimen's own draws are its callback's.
    checkpoint_chunks : int, optional
        Recompute multislice chunks in the backward to bound memory.
    device : torch.device or str, optional
        Where to compute. Default "cpu".

    Returns
    -------
    FrozenPlasmonResult

    References
    ----------
    .. [1] Himes, B., & Grigorieff, N. (2021). Cryo-TEM simulations of amorphous
       radiation-sensitive samples using multislice wave propagation. *IUCrJ*,
       8(6), 943–953. https://doi.org/10.1107/S2052252521008538
    """
    from ..inelastic import FrozenPlasmonForward

    pad_nxy = nxy + (nxy // 2) * 2 if pad_fft else nxy
    scattering = IterativeScattering(
        pad_nxy,
        voxel_size,
        voltage,
        alpha=0,
        klim=klim,
        oversample=oversample,
        crop=False,
        ews_curvature_sign=ews_curvature_sign,
        progressbars=False,
    ).to(device)
    focal = (
        FocalSpread.from_instrument(
            cc * 1e7,
            voltage * 1e3,
            energy_spread,
            deltaV_V,
            deltaI_I,
            energy_to_wavelength(voltage),
        )
        if cc
        else None
    )
    camera = Detector(
        voxel_size,
        aberration_model="nonlinear",
        noise_model=None if noise == "none" else noise,
        mtf=detector_mtf(detector, nxy, voxel_size),
        dqe0=dqe0_for_detector(detector),
        oversample=oversample,
        focal_spread=focal,
        n_frames=1,
        progressbars=False,
    ).to(device)
    tracked = _Depth(specimen)

    def optics(wave: torch.Tensor, step: ExposureStep) -> torch.Tensor:
        if ctf is None:
            return stages.aberrate(wave, None, {}, oversample, pad_nxy)
        n = len(wave)
        shift = defocus_midplane_shift(tracked.nz or 0, voxel_size)
        params = {k: v[:1].expand(n).to(wave.device) for k, v in ctf.params.items()}
        for name in ("dfu", "dfv"):
            if name in params:
                params[name] = params[name] - shift
        params["dose"] = wave.real.new_full((n,), step.dose)
        params["pre_exposure"] = wave.real.new_full((n,), step.start)
        engine = aberration_engine(
            pad_nxy * oversample,
            voxel_size / oversample,
            voltage,
            "nonlinear",
            None,
            convergence_angle,
            False,
            True,
            "legacy",
            None,
            str(wave.device),
        )
        return stages.aberrate(wave, engine, params, oversample, pad_nxy)

    model = FrozenPlasmonForward(scattering, camera, plasmon_filter, optics=optics)
    return model(
        tracked,
        frame_doses,
        pose=pose,
        poses=poses,
        substeps=substeps,
        pre_exposure=pre_exposure,
        detector_seed=seed,
        checkpoint_chunks=checkpoint_chunks,
        coincidence_radius=float(per_image(coincidence_radius, 1, "radius")[0]),
        anisomag=anisomag,
        nxy=nxy,
    )
