"""Scattering-potential volume builders: free-function analytic/FFT builders and the two `PotentialBuilder` classes."""

from __future__ import annotations

from ._absorption import apply_amplitude_contrast
from ._builders import (
    build_atomic_potential_kernel,
    build_potential_volume_analytic_scatter,
    build_potential_volume_analytic_scatter_kirkland,
    build_potential_volume_analytic_scatter_lobato,
    build_potential_volume_fftconvolve_2d,
    build_potential_volume_fftconvolve_3d,
    compute_supersampling_parameters,
    potential_from_deltas,
    recommended_rcut,
)
from ._gemmi_builder import GemmiPotentialBuilder
from ._potential_builder import PotentialBuilder

__all__ = [
    "apply_amplitude_contrast",
    "GemmiPotentialBuilder",
    "PotentialBuilder",
    "build_atomic_potential_kernel",
    "build_potential_volume_analytic_scatter",
    "build_potential_volume_analytic_scatter_kirkland",
    "build_potential_volume_analytic_scatter_lobato",
    "build_potential_volume_fftconvolve_2d",
    "build_potential_volume_fftconvolve_3d",
    "compute_supersampling_parameters",
    "potential_from_deltas",
    "recommended_rcut",
]
