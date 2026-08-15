"""
Filament path geometry: a persistent random walk in 3-D.

Port of CryoTomoSim (CTS)'s ``WIP/filament_polym_linear_script.m`` "random
curve through step walk" block -- never merged into CTS's own mainline, only
ever exercised as a standalone exploratory script there. At each step, the
walk advances by a fixed distance (``step``, the monomer's axial rise) along
the current direction, then perturbs that direction by a random angle drawn
from ``[0, flex_deg]`` about a random axis roughly perpendicular to it --
bounded per-step curvature, unbounded cumulative curvature, the same
"worm-like chain" flavor CTS used for both its actin and microtubule presets
(``step=27.3, flex=12`` and ``step=85, flex=3`` respectively).

References
----------
Purnell, C., Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S., Grillo, M.,
Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T. (2023). Rapid
synthesis of cryo-ET data for training deep learning models. bioRxiv
2023.04.28.538636. https://doi.org/10.1101/2023.04.28.538636
CTS source: https://github.com/carsonpurnell/cryotomosim_CTS
"""

from __future__ import annotations

import math

import roma
import torch


def _random_unit_vector(generator: torch.Generator | None) -> torch.Tensor:
    v = torch.randn(3, generator=generator)
    return v / v.norm().clamp_min(1e-8)


def _perturb_direction(
    direction: torch.Tensor, flex_rad: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Rotate `direction` by a random angle in [0, flex_rad] about a random
    axis perpendicular to it (re-drawn if the sampled reference vector comes
    out too close to parallel, i.e. would give a degenerate cross product).
    """
    axis = torch.zeros(3)
    for _ in range(10):
        reference = _random_unit_vector(generator)
        axis = torch.linalg.cross(direction, reference)
        if float(axis.norm()) > 1e-6:
            break
    axis = axis / axis.norm().clamp_min(1e-8)

    angle = float(torch.rand(1, generator=generator).item()) * flex_rad
    rotation = roma.rotvec_to_rotmat((axis * angle).unsqueeze(0))[0]
    new_direction = rotation @ direction
    return new_direction / new_direction.norm().clamp_min(1e-8)


def generate_filament_path(
    n_monomers: int,
    step: float,
    flex_deg: float,
    origin_xyz: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    direction_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate one filament's monomer-center positions via a persistent random
    walk.

    Parameters
    ----------
    n_monomers : int
        Number of monomer positions to generate.
    step : float
        Axial rise per monomer, Angstrom (the distance advanced along the
        current direction at each step).
    flex_deg : float
        Maximum per-step direction change, degrees. Each step's turn angle
        is drawn uniformly from ``[0, flex_deg]`` about a random axis
        roughly perpendicular to the current direction, so curvature is
        bounded per-step but unbounded over the whole filament.
    origin_xyz : torch.Tensor, optional
        Starting position, shape ``(3,)``, physical ``(x, y, z)``. Default
        the origin.
    generator : torch.Generator, optional
        Random generator for the initial direction and all per-step turns.
        Default None (uses torch's global RNG).
    direction_xyz : torch.Tensor, optional
        Initial direction, shape ``(3,)`` (normalized internally). Default
        None: drawn uniformly at random. Used by microtubule placement,
        which picks the direction itself so it can reject orientations that
        would leave a thin slab (see ``_tube.place_microtubules``).

    Returns
    -------
    torch.Tensor
        Monomer-center positions, shape ``(n_monomers, 3)``, physical
        ``(x, y, z)``, in the same frame as ``origin_xyz``.
    """
    if n_monomers < 1:
        raise ValueError(f"n_monomers must be >= 1, got {n_monomers}")

    flex_rad = math.radians(flex_deg)
    pos = torch.zeros(3) if origin_xyz is None else origin_xyz.clone()
    if direction_xyz is None:
        direction = _random_unit_vector(generator)
    else:
        direction = direction_xyz / direction_xyz.norm().clamp_min(1e-8)

    positions = torch.empty((n_monomers, 3))
    for i in range(n_monomers):
        positions[i] = pos
        pos = pos + direction * step
        direction = _perturb_direction(direction, flex_rad, generator)
    return positions
