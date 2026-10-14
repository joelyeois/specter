"""
The two engines worth keeping between calls: building the propagator and the
transfer function's frequency grids costs 2-6 ms each at a 256-512 box, which
a reconstruction would otherwise pay on every step. Each is cached on the
settings that determine it, including the device, and holds only buffers, so
sharing one between calls carries no state from one image to the next.
"""

from __future__ import annotations

import functools

import torch

from ..aberrations import Aberration
from ..options import AberrationModel, EwaldSphereSign, ScatteringModel
from ..scattering import Scattering

__all__ = ["aberration_engine", "scattering_engine"]

_CACHE_SIZE = 8


@functools.lru_cache(maxsize=_CACHE_SIZE)
def scattering_engine(
    pad_nxy: int,
    voxel_size: float,
    voltage: float,
    scattering_model: ScatteringModel,
    klim: float | None,
    oversample: int,
    ews_curvature_sign: EwaldSphereSign,
    nz: int,
    alpha: float,
    uniform_absorption: float,
    device: str,
) -> Scattering:
    """A `Scattering` returning the wave on its propagation grid, on `device`."""
    return Scattering(
        pad_nxy,
        voxel_size,
        voltage,
        scattering_model=scattering_model,
        klim=klim,
        oversample=oversample,
        crop=False,
        ews_curvature_sign=ews_curvature_sign,
        nz=nz,
        alpha=alpha,
        uniform_absorption=uniform_absorption,
        progressbars=False,
    ).to(device)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def aberration_engine(
    n: int,
    dx: float,
    voltage: float,
    model: AberrationModel,
    alpha: float | None,
    convergence_angle: float | None,
    dose_envelope: bool,
    dose_weighted: bool,
    backend: str,
    lpp_params: tuple[tuple[str, float], ...] | None,
    device: str,
) -> torch.nn.Module:
    """
    The transfer-function engine on an ``n``-point grid of spacing `dx`.

    `alpha` reaches the linear model only, the one whose wave has no
    absorptive part of its own. ``lpp_params`` is the phase plate's settings
    as sorted ``(key, value)`` pairs, so it can be part of the cache key.
    """
    linear = model == "linear"
    engine: torch.nn.Module
    if backend == "torch_ctf":
        from ..ctf import LegacyAberrationAdapter

        engine = LegacyAberrationAdapter(
            n,
            dx,
            voltage,
            aberration_model=model,
            specimen_absorption=not linear,
            convergence_angle=convergence_angle,
            dose_envelope=dose_envelope,
            dose_weighted=dose_weighted,
            lpp_params=None if lpp_params is None else dict(lpp_params),
        )
    else:
        engine = Aberration(
            n,
            dx,
            voltage,
            aberration_model=model,
            alpha=alpha if linear else None,
            specimen_absorption=not linear,
            convergence_angle=convergence_angle,
            dose_envelope=dose_envelope,
            dose_weighted=dose_weighted,
            progressbars=False,
        )
    return engine.to(device)
