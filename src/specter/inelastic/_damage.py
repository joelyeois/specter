"""Optional empirical structural damage, kept separate from absorption."""

from __future__ import annotations

import math
import torch
from torch import nn
from specter.aberrations import dose_envelope
from ._exposure import ExposureStep


class PotentialDoseDamage(nn.Module):
    """Apply the empirical dose envelope to the particle's 3D real potential.

    The instantaneous amplitude at the substep midpoint damps high spatial
    frequencies while preserving DC (total potential). This approximation
    represents loss of ordered particle signal, not atom-specific chemistry
    or mass loss. Solvent and imaginary source are left to their own models.
    Substep convergence is required. Do not also enable an optical dose
    envelope for the same exposure.
    """

    def __init__(self, pixel_size: float, voltage: float = 300.0):
        super().__init__()
        if (
            not math.isfinite(pixel_size)
            or pixel_size <= 0
            or not math.isfinite(voltage)
            or voltage <= 0
        ):
            raise ValueError("pixel size and voltage must be finite and positive")
        self.pixel_size = pixel_size
        self.voltage = voltage

    def forward(self, potential: torch.Tensor, step: ExposureStep) -> torch.Tensor:
        z, y, x = potential.shape[-3:]
        fz = torch.fft.fftfreq(z, self.pixel_size, device=potential.device)
        fy = torch.fft.fftfreq(y, self.pixel_size, device=potential.device)
        fx = torch.fft.fftfreq(x, self.pixel_size, device=potential.device)
        q = (
            fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2
        ).sqrt()
        envelope = dose_envelope(
            q,
            potential.new_tensor(0.0),
            pre_exposure=step.midpoint,
            weighted=False,
            voltage=self.voltage,
        )
        return torch.fft.ifftn(
            torch.fft.fftn(potential, dim=(-3, -2, -1)) * envelope, dim=(-3, -2, -1)
        ).real
