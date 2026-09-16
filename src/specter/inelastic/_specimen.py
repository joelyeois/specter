"""Coupled elastic and absorption sources from explicit scattering centres."""

from __future__ import annotations

from collections.abc import Mapping, Callable
from dataclasses import dataclass
import math
import torch
from torch import nn

from specter.arrays import soft_voxelize_coordinates
from specter.fft import fftconvolve
from specter.potential import PotentialBuilder, potential_occupancy
from specter.ice._kernels import build_water_kernel
from ._exposure import BrownianCoordinates, ExposureStep


@dataclass
class PotentialState:
    """Real fields (B,Z,Y,X), in volts, in specimen coordinates.

    ``absorption_source`` is filtered only after beam-frame sampling.
    ``absorption_filter=None`` instead interprets it as an imaginary potential.
    """

    elastic: torch.Tensor
    absorption_source: torch.Tensor


class AtomicPlasmonSpecimen(nn.Module):
    """Build both potentials from the same atoms and moving pseudo-waters.

    Species strengths are explicit integrated source coefficients in V Å³
    per atom; ``water_strength`` has the same units per pseudo-water. Their
    provenance/calibration is required. For bulk number density n (Å⁻³),
    MFP λ (Å), and interaction parameter σ, a calibrated coefficient is
    ``1 / (2 * σ * λ * n)``. This sets the mean, not an absolute cross-section
    prediction. There is no frame-by-frame renormalization.

    The default solvent exclusion is SPECTER's potential occupancy prior.
    Supply ``solvent_fraction`` for a measured hydration/exclusion field.
    ``evolve_atoms`` may supply dose-dependent coordinates (chemical or rigid
    motion model); it must return coordinates in Å on the same device.
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        shape: tuple[int, int, int],
        pixel_size: float,
        species_strengths: Mapping[int, float],
        *,
        provenance: str,
        water_coordinates: torch.Tensor | None = None,
        water_strength: float = 0.0,
        water_msd_per_dose: float = 0.0,
        seed: int = 0,
        solvent_fraction: Callable[[torch.Tensor], torch.Tensor] | None = None,
        evolve_atoms: Callable[[torch.Tensor, ExposureStep], torch.Tensor]
        | None = None,
        damage_model: Callable[[torch.Tensor, ExposureStep], torch.Tensor]
        | None = None,
    ):
        super().__init__()
        if len(shape) != 3 or min(shape) < 1 or shape[1] != shape[2]:
            raise ValueError("shape must be positive (Z,Y,X) with square XY")
        if not math.isfinite(pixel_size) or pixel_size <= 0 or not provenance.strip():
            raise ValueError(
                "positive pixel_size and calibration provenance are required"
            )
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or atomic_numbers.shape != coordinates.shape[:1]
        ):
            raise ValueError("provide (N,3) coordinates and (N,) atomic numbers")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        species = [int(z) for z in torch.unique(atomic_numbers).tolist()]
        if any(z not in species_strengths for z in species):
            raise ValueError(
                "every atomic species requires an explicit absorption strength"
            )
        if any(
            not math.isfinite(species_strengths[z]) or species_strengths[z] < 0
            for z in species
        ):
            raise ValueError("species strengths must be finite and nonnegative")
        if not math.isfinite(water_strength) or water_strength < 0:
            raise ValueError("water strength must be finite and nonnegative")
        if not math.isfinite(water_msd_per_dose) or water_msd_per_dose < 0:
            raise ValueError("water MSD must be finite and nonnegative")
        self.register_buffer("coordinates", coordinates.clone())
        self.register_buffer("atomic_numbers", atomic_numbers.to(coordinates.device))
        self.register_buffer(
            "water_coordinates",
            None
            if water_coordinates is None
            else water_coordinates.to(coordinates).clone(),
        )
        self.register_buffer(
            "water_kernel", build_water_kernel(pixel_size, "kirkland").to(coordinates)
        )
        self.builder = PotentialBuilder(
            (shape[2], shape[1], shape[0]),
            pixel_size,
            atomic_numbers.to(coordinates.device),
            parameterization="kirkland",
            progressbars=False,
        ).to(coordinates.device)
        self.shape = shape
        self.pixel_size = pixel_size
        self.species_strengths = dict(species_strengths)
        self.provenance = provenance
        self.water_strength = water_strength
        self.water_msd_per_dose = water_msd_per_dose
        self.seed = seed
        self.solvent_fraction = solvent_fraction
        self.evolve_atoms = evolve_atoms
        self.damage_model = damage_model
        self.applies_damage = damage_model is not None
        self._water_motion: BrownianCoordinates | None = None

    def reset(self) -> None:
        """Replay the same trajectory without affecting the global RNG."""
        self._water_motion = None
        if self.water_coordinates is not None:
            box = self.coordinates.new_tensor(self.shape[::-1]) * self.pixel_size
            self._water_motion = BrownianCoordinates(
                self.water_coordinates, self.water_msd_per_dose, self.seed, box
            )

    def forward(self, step: ExposureStep) -> PotentialState:
        coords = (
            self.coordinates
            if self.evolve_atoms is None
            else self.evolve_atoms(self.coordinates, step)
        )
        elastic = self.builder(coords, method="analytic")
        undamaged = elastic
        if self.damage_model is not None:
            elastic = self.damage_model(elastic, step)
        source = torch.zeros_like(elastic)
        for z, strength in self.species_strengths.items():
            source = source + strength / self.pixel_size**3 * soft_voxelize_coordinates(
                coords[self.atomic_numbers == z], self.shape, self.pixel_size
            )
        if self.water_coordinates is not None:
            if self._water_motion is None:
                self.reset()
            assert self._water_motion is not None
            waters = self._water_motion.advance_to(step.midpoint)
            deltas = soft_voxelize_coordinates(
                waters, self.shape, self.pixel_size, periodic=True
            )
            fraction = (
                1 - potential_occupancy(undamaged, self.pixel_size)
                if self.solvent_fraction is None
                else self.solvent_fraction(undamaged)
            )
            if (
                fraction.shape != elastic.shape
                or not torch.isfinite(fraction).all()
                or (fraction < 0).any()
                or (fraction > 1).any()
            ):
                raise ValueError(
                    "solvent fraction must match the volume and lie in [0,1]"
                )
            ice = fftconvolve(
                deltas[None], self.water_kernel[None], mode="same", axes=(-3, -2, -1)
            )[0]
            elastic = elastic + fraction * ice
            source = source + fraction * deltas * (
                self.water_strength / self.pixel_size**3
            )
        return PotentialState(elastic[None], source.to(elastic)[None])
