"""Effective solvent exposure under a Gaussian displacement model.

This is a second-moment approximation, exact for projected linear scattering.
It is not an atomistic dynamics integrator or an exact multislice exposure.
The underlying solvent structure factor still comes from the generated ice.
"""

from __future__ import annotations

import math

import torch


def solvent_exposure_power(
    k: torch.Tensor,
    dose: float,
    displacement_variance: float,
    n_frames: int,
    weights: torch.Tensor | None = None,
    weights_max_frequency: float | None = None,
) -> torch.Tensor:
    """Return a^T C a for equal-dose fractions, including within-frame motion.

    ``displacement_variance`` is per-axis Å² per (electron/Å²). Correlation
    between instantaneous Fourier amplitudes is exp(-2π² k² s² |D-D'|).
    Normalized amplitude weights a sum to one at each frequency. The O(N)
    recursion evaluates all off-diagonal pairs without a dense covariance.
    """
    if dose < 0 or displacement_variance < 0 or n_frames < 1:
        raise ValueError("dose/motion must be nonnegative and n_frames positive")
    if weights is not None:
        if weights.shape[0] != n_frames or weights_max_frequency is None:
            raise ValueError("weights require matching frames and a frequency axis")
        if weights_max_frequency <= 0 or torch.any(weights < 0):
            raise ValueError("invalid weight axis or negative weights")
        idx = (k / weights_max_frequency * (weights.shape[1] - 1)).round().long()
        idx = idx.clamp(0, weights.shape[1] - 1)
        a = weights.to(k)[:, idx]
        total = a.sum(0)
        if torch.any(total <= 0):
            raise ValueError("frame weights must have positive sum")
        a = a / total
    else:
        a = torch.ones((n_frames, *k.shape), device=k.device, dtype=k.dtype) / n_frames
    z = 2 * math.pi**2 * k.square() * displacement_variance * dose / n_frames
    safe = z.clamp_min(1e-5)
    g = torch.where(z < 1e-3, 1 - z / 2 + z.square() / 6, -torch.expm1(-safe) / safe)
    diagonal = torch.where(
        z < 1e-2,
        1 - z / 3 + z.square() / 12 - z.pow(3) / 60,
        2 * (safe + torch.expm1(-safe)) / safe.square(),
    )
    history = torch.zeros_like(k)
    pairs = torch.zeros_like(k)
    decay = torch.exp(-z)
    for j in range(1, n_frames):
        history = a[j - 1] + decay * history
        pairs += a[j] * history
    return (diagonal * a.square().sum(0) + 2 * g.square() * pairs).clamp(0, 1)


def apply_solvent_exposure(
    ice: torch.Tensor,
    pixel_size: float,
    dose: float,
    displacement_variance: float,
    n_frames: int,
    weights: torch.Tensor | None,
    weights_max_frequency: float | None,
) -> None:
    """Filter generated ice in place; DC/mean potential is preserved.

    A radial 3D extension supplies a rotationally invariant effective volume.
    Its projected power has the specified temporal average. Applying this
    before multislice is an approximation, explicitly reported by the matcher.
    """
    from ..potential._damage import _damage_item

    def envelope(k: torch.Tensor) -> torch.Tensor:
        return solvent_exposure_power(
            k, dose, displacement_variance, n_frames, weights, weights_max_frequency
        ).sqrt()

    for volume in ice:
        _damage_item(volume, pixel_size, envelope)
