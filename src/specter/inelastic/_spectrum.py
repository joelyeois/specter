"""Himes-style low-loss amplitude kernels with explicit normalization.

The kernel sets spectral shape only. Source strengths, in V Å³ per centre,
set absorption; a DC-normalized kernel cannot predict a mean free path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from specter.constants import energy_to_wavelength


class PlasmonFilter(nn.Module):
    """Energy-integrated Lorentzian amplitude, applied transverse to the beam.

    ``energies`` (eV) and ``loss_density`` must describe a single-scattering
    loss spectrum without the zero-loss peak, covering 7.5–100 eV. The shape
    follows Himes & Grigorieff (2021), equations 8–9. ``provenance`` records
    the measurement or the explicitly selected approximation.

    Padding is vacuum (zero source). For bulk periodic specimens, set
    ``padding=0`` and supply a sufficiently large periodic transverse cell.
    Convergence in padding must be checked for each physical geometry.
    """

    def __init__(
        self,
        pixel_size: float,
        voltage: float,
        energies: torch.Tensor,
        loss_density: torch.Tensor,
        *,
        provenance: str,
        padding: int = 32,
    ):
        super().__init__()
        if (
            not np.isfinite(pixel_size)
            or pixel_size <= 0
            or not np.isfinite(voltage)
            or voltage <= 0
        ):
            raise ValueError("pixel size and voltage must be finite and positive")
        if not isinstance(padding, int) or padding < 0 or not provenance.strip():
            raise ValueError(
                "padding must be a nonnegative integer and provenance is required"
            )
        e = torch.as_tensor(energies, dtype=torch.float64).detach().cpu()
        p = torch.as_tensor(loss_density, dtype=torch.float64).detach().cpu()
        if e.ndim != 1 or e.shape != p.shape or len(e) < 2:
            raise ValueError(
                "spectrum must contain equally sized energy and density vectors"
            )
        if not torch.isfinite(e).all() or not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("spectrum must be finite and nonnegative")
        if not (torch.diff(e) > 0).all() or e[0] > 7.5 or e[-1] < 100:
            raise ValueError("energies must increase and cover 7.5–100 eV")
        # Include exact endpoints, even when the measured grid straddles them.
        energy = np.unique(
            np.r_[7.5, e.numpy()[(e.numpy() > 7.5) & (e.numpy() < 100)], 100.0]
        )
        density = np.interp(energy, e.numpy(), p.numpy())
        area = np.trapezoid(density, energy)
        if area <= 0:
            raise ValueError("spectrum must have positive area in 7.5–100 eV")
        self.register_buffer("energies", torch.from_numpy(energy))
        self.register_buffer("loss_density", torch.from_numpy(density / area))
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.padding = padding
        self.provenance = provenance
        self._cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        pixel_size: float,
        voltage: float,
        *,
        provenance: str,
        padding: int = 32,
    ) -> PlasmonFilter:
        """Read two columns with a header: energy_eV, loss_density."""
        table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if table.shape[1] != 2:
            raise ValueError("spectrum CSV must have exactly two columns")
        return cls(
            pixel_size,
            voltage,
            torch.from_numpy(table[:, 0]),
            torch.from_numpy(table[:, 1]),
            provenance=provenance,
            padding=padding,
        )

    @classmethod
    def approximate_drude(
        cls,
        pixel_size: float,
        voltage: float,
        *,
        damping: float = 8.0,
        padding: int = 32,
    ) -> PlasmonFilter:
        """Explicit development approximation; not the paper's measured EELS."""
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be finite and positive")
        energy = torch.linspace(7.5, 100, 512, dtype=torch.float64)
        density = (
            20.8**2
            * damping
            * energy
            / ((energy**2 - 20.8**2) ** 2 + (damping * energy) ** 2)
        )
        return cls(
            pixel_size,
            voltage,
            energy,
            density,
            provenance=f"Approximate Drude: peak 20.8 eV, damping {damping} eV; not measured EELS",
            padding=padding,
        )

    def kernel(self, shape: tuple[int, int], reference: torch.Tensor) -> torch.Tensor:
        """Return a cached DC-normalized amplitude on an rFFT grid."""
        key = (
            shape,
            reference.device,
            reference.dtype,
            self.energies._version,
            self.loss_density._version,
        )
        if key not in self._cache:
            e = self.energies.detach().cpu().numpy()
            density = self.loss_density.detach().cpu().numpy()
            qmax = np.sqrt(2) / (2 * self.pixel_size)
            q = np.r_[0.0, np.geomspace(min(1e-8, qmax / 1e6), qmax, 8191)]
            theta = energy_to_wavelength(self.voltage) * q
            dcs = np.trapezoid(
                density[None]
                / (theta[:, None] ** 2 + (e[None] / (2 * self.voltage * 1000)) ** 2),
                e,
                axis=1,
            )
            fy = np.fft.fftfreq(shape[0], self.pixel_size)
            fx = np.fft.rfftfreq(shape[1], self.pixel_size)
            radius = np.hypot(fy[:, None], fx[None, :])
            values = np.interp(radius, q, np.sqrt(dcs / dcs[0]))
            # Bound cached canvases across varying crops and devices.
            if len(self._cache) >= 8:
                self._cache.clear()
            self._cache[key] = torch.as_tensor(
                values, device=reference.device, dtype=reference.dtype
            )
        return self._cache[key]

    @property
    def halo(self) -> int:
        """Neighbouring source pixels to fetch before transverse convolution."""
        return self.padding

    def filter_sampled(self, source: torch.Tensor) -> torch.Tensor:
        """Filter a beam-frame slice whose halo was sampled from the specimen.

        Unlike padding a cropped slice with vacuum, this includes material
        outside the propagation ROI. The outer FFT boundary still requires
        convergence in halo width.
        """
        result = self._convolve(source)
        h = self.halo
        return result[..., h:-h, h:-h] if h else result

    def _convolve(self, source: torch.Tensor) -> torch.Tensor:
        shape = (source.shape[-2], source.shape[-1])
        result = torch.fft.irfft2(
            torch.fft.rfft2(source) * self.kernel(shape, source), s=shape
        )
        if result.requires_grad:
            result.register_hook(lambda grad: grad.contiguous())
        return result

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Filter a standalone source with vacuum outside its given extent."""
        if source.is_complex() or source.ndim < 2:
            raise ValueError("source must be real with at least two spatial dimensions")
        padded = F.pad(source, (self.padding,) * 4) if self.padding else source
        return self.filter_sampled(padded)
