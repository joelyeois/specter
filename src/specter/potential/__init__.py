"""Scattering-potential volume builders: free-function analytic/FFT builders and the two `PotentialBuilder` classes."""

from __future__ import annotations

from ._absorption import (
    INELASTIC_MFP_ICE_A,
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    apply_amplitude_contrast,
    inelastic_absorption_potential,
)
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
from ._damage import apply_dose_damage, frame_damage_envelope
from ._gemmi_builder import GemmiPotentialBuilder
from ._occupancy import (
    FULL_OCCUPANCY_POTENTIAL_V,
    WATER_COARSE_GRAIN_SIGMA_ANGSTROM,
    occupancy_blur_halo_voxels,
    potential_occupancy,
)
from ._potential_builder import PotentialBuilder

__all__ = [
    "INELASTIC_MFP_ICE_A",
    "INELASTIC_MFP_PROTEIN_A",
    "absorption_potential",
    "apply_dose_damage",
    "frame_damage_envelope",
    "apply_amplitude_contrast",
    "inelastic_absorption_potential",
    "FULL_OCCUPANCY_POTENTIAL_V",
    "GemmiPotentialBuilder",
    "PotentialBuilder",
    "WATER_COARSE_GRAIN_SIGMA_ANGSTROM",
    "occupancy_blur_halo_voxels",
    "potential_occupancy",
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
