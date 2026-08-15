"""
Filament placement: random walks of monomer positions/orientations scattered
through a specimen volume, one call per set of like filaments (a "species").

Pure geometry only -- no PDB fetching, no potential rendering. A caller
(``TomogramSpecimenGenerator._stamp_filaments``) fetches each spec's
monomer structure, builds one potential template per species, then
rotates/inserts a copy at every ``FilamentInstance`` this module returns --
the same division of labor its protein stamping already uses (geometry
decided here, rendering done by the caller, which alone knows about voxel
sizes/atomic potentials).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._path import generate_filament_path
from ._placement import filament_orientations


@dataclass
class FilamentSpec:
    """One filament species to place.

    Parameters
    ----------
    code : str
        PDB ID (fetched from RCSB) or local file path for the monomer
        structure repeated along the filament.
    step : float
        Axial rise per monomer, Angstrom.
    flex_deg : float
        Maximum per-step path curvature, degrees (see
        ``generate_filament_path``).
    twist_deg : float, optional
        Helical twist per monomer, degrees, accumulated along the filament.
        Default 0 (no twist -- a single, untwisted strand, e.g. a
        microtubule protofilament).
    n_copies : int, optional
        Number of independent filament instances of this species to place.
        Default 1.
    n_monomers : int or tuple of int, optional
        Monomer count per filament instance. A fixed int, or a
        ``(min, max)`` range (inclusive) drawn uniformly per instance.
        Default ``(10, 30)``.
    """

    code: str
    step: float
    flex_deg: float
    twist_deg: float = 0.0
    n_copies: int = 1
    n_monomers: int | tuple[int, int] = (10, 30)


#: Tubulin alpha-beta dimer (PDB 1TUB, electron crystallography) repeated
#: along a single protofilament. step=85 A is the real tubulin-dimer axial
#: repeat; flex=3 deg is CTS's own tuned value for a protofilament's gentle
#: curvature. twist=0: a single protofilament doesn't itself twist (the
#: microtubule's ~13-protofilament tube geometry, and its small supertwist,
#: are out of scope here -- see the module docstring in
#: ``specimen/filament/``).
MICROTUBULE_SPEC = FilamentSpec(
    code="1TUB", step=85.0, flex_deg=3.0, twist_deg=0.0, n_monomers=(10, 30)
)

#: Uncomplexed G-actin monomer (PDB 1J6Z) repeated as F-actin. step=27.3 A
#: and twist=166.15 deg are the real, measured F-actin helical repeat
#: (Holmes/Egelman); flex=12 deg is CTS's own tuned value for actin's
#: greater flexibility relative to microtubules.
ACTIN_SPEC = FilamentSpec(
    code="1J6Z", step=27.3, flex_deg=12.0, twist_deg=166.15, n_monomers=(20, 60)
)


@dataclass
class FilamentInstance:
    """One placed monomer copy, for rendering and ground-truth bookkeeping."""

    code: str
    filament_id: int
    monomer_index: int
    position_xyz: torch.Tensor  # (3,), physical Angstrom
    rotation_matrix: torch.Tensor  # (3, 3)


def _monomer_count(
    n_monomers: int | tuple[int, int], generator: torch.Generator | None
) -> int:
    if isinstance(n_monomers, int):
        return n_monomers
    low, high = n_monomers
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def place_filaments(
    specs: list[FilamentSpec],
    target_shape: tuple[int, int, int],
    voxel_size: float,
    generator: torch.Generator | None = None,
) -> list[FilamentInstance]:
    """
    Place every filament instance for every spec, each starting at a
    uniformly random point within the volume's physical extent (filaments
    that wander outside the volume are simply truncated at render/insert
    time, the same edge behavior CTS/polnet-placed particles already get).

    Parameters
    ----------
    specs : list of FilamentSpec
    target_shape : tuple of int
        Specimen volume shape (Z, Y, X), voxels.
    voxel_size : float
        Voxel size, Angstrom.
    generator : torch.Generator, optional
        Random generator for start positions, initial directions, and all
        per-step turns. Default None (uses torch's global RNG).

    Returns
    -------
    list of FilamentInstance
    """
    extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * voxel_size

    instances: list[FilamentInstance] = []
    for spec in specs:
        for filament_id in range(spec.n_copies):
            n = _monomer_count(spec.n_monomers, generator)
            origin_xyz = torch.rand(3, generator=generator) * extent_xyz
            positions = generate_filament_path(
                n, spec.step, spec.flex_deg, origin_xyz, generator
            )
            rotations = filament_orientations(positions, spec.twist_deg)
            for i in range(n):
                instances.append(
                    FilamentInstance(
                        code=spec.code,
                        filament_id=filament_id,
                        monomer_index=i,
                        position_xyz=positions[i],
                        rotation_matrix=rotations[i],
                    )
                )
    return instances
