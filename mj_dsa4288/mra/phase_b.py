"""
Phase B: put the cryo-EM physics back, one term at a time.

Here SPECTER's forward model is used end-to-end (fixed view, random in-plane
shifts, optional CTF / Poisson noise / detector / ice), so the data no longer
follow the paper's white-Gaussian model. The question is which of the paper's
predictions survive each added term. Noise is set by dose rather than by
``sigma``.

The returned stacks are ordinary ``[n, Y, X]`` tensors. Note two departures
from Phase A:

* shifts are real-space (not cyclic), so ``max_shift`` must keep the particle
  inside the box;
* with a CTF switched on, the estimand is the CTF-filtered projection, not the
  projection itself. Compare against a clean CTF-on reference image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PhysicsConfig:
    """
    Which physics terms to switch on. Defaults reproduce the Phase A limit.

    Attributes
    ----------
    voltage, cs, alpha : float
        Microscope constants (kV, mm, amplitude-contrast ratio).
    defocus_angstrom : float
        Defocus in Angstrom; 0 together with ``cs = 0`` gives no CTF.
    defocus_jitter_angstrom : float
        Half-width of a per-particle uniform defocus perturbation (Phase B.2).
    scattering_model : str
        ``"projection"`` (linear) or ``"multislice"`` (full).
    noise_model : str or None
        ``None`` (clean) or ``"poisson"`` (dose-limited counting noise).
    detector_model : str or None
        ``None``, ``"perfect"``, ``"k3_300kv"`` ...
    coincidence_radius : float
        Detector coincidence-loss radius in pixels (needs ``noise_model``).
    ice_model : str or None
        ``None``, ``"random"`` or ``"gd"``.
    ice_thickness : float
        Ice thickness in Angstrom.
    max_shift_angstrom : float
        Uniform in-plane shift range (real-space, not cyclic).
    bfactor : float or None
        B-factor envelope; deliberately off (Section 3.2 low-pass warning).
    """

    voltage: float = 300.0
    cs: float = 0.0
    alpha: float = 0.0
    defocus_angstrom: float = 0.0
    defocus_jitter_angstrom: float = 0.0
    scattering_model: str = "projection"
    aberration_model: str = "holography"
    noise_model: str | None = None
    detector_model: str | None = None
    coincidence_radius: float = 0.0
    ice_model: str | None = None
    ice_thickness: float = 0.0
    max_shift_angstrom: float = 10.0
    bfactor: float | None = None
    device: str = "cpu"


def build_generator(
    volume: torch.Tensor,
    pixel_size: float,
    n: int,
    dose: float,
    cfg: PhysicsConfig,
    quaternion: torch.Tensor | None = None,
    seed: int = 0,
) -> Any:
    """
    Construct a SPECTER ``ImageGenerator`` for a fixed view with in-plane shifts.

    Parameters
    ----------
    volume : torch.Tensor
        Scattering potential ``[Z, Y, X]`` (from ``PotentialBuilder``).
    pixel_size : float
        Angstrom per pixel.
    n : int
        Number of particles in the stack.
    dose : float
        Electron dose per square Angstrom (sets the SNR when noise is on).
    cfg : PhysicsConfig
        Which physics to include.
    quaternion : torch.Tensor, optional
        Fixed view; identity if omitted.
    seed : int, optional
        Seed for the shift / defocus sampling.

    Returns
    -------
    ImageGenerator
        Call ``model(idx)`` with an index tensor to obtain images ``[B, Y, X]``.
    """
    from specter.imagegenerator import ImageGenerator

    gen = torch.Generator().manual_seed(seed)
    if quaternion is None:
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    quats = quaternion[None].float().expand(n, 4).contiguous()
    shifts = (2 * torch.rand(n, 2, generator=gen) - 1) * cfg.max_shift_angstrom
    defocus = torch.full((n,), cfg.defocus_angstrom)
    if cfg.defocus_jitter_angstrom > 0:
        defocus = (
            defocus
            + (2 * torch.rand(n, generator=gen) - 1) * cfg.defocus_jitter_angstrom
        )
    ctf_params = {
        "dfu": defocus,
        "dfv": defocus.clone(),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), cfg.cs * 1e7),  # mm -> Angstrom
    }
    return ImageGenerator(
        volume,
        pixel_size,
        quaternions=quats,
        translations=shifts,
        ctf_params=ctf_params,
        voltage=cfg.voltage,
        dose_per_angstrom=dose,
        ice_model=cfg.ice_model,
        ice_thickness=cfg.ice_thickness if cfg.ice_model else None,
        scattering_model=cfg.scattering_model,
        aberration_model=cfg.aberration_model,
        noise_model=cfg.noise_model,
        detector_model=cfg.detector_model,
        coincidence_radius=cfg.coincidence_radius,
        alpha=cfg.alpha,
        bfactor=cfg.bfactor,
        verbose=False,
    ).to(cfg.device)


def generate_stack(model: Any, n: int, batch_size: int = 64) -> torch.Tensor:
    """Run ``model`` over ``range(n)`` in batches and return images ``[n, Y, X]`` on CPU."""
    out = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n), device=model.device)
            out.append(model(idx).detach().cpu().float())
    return torch.cat(out, 0)
