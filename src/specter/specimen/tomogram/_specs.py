"""Dataclasses for TomogramSpecimenGenerator's inputs and placement bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from ...config import ScalarOrRange, parse_scalar_or_range
from ..membrane import MembraneGenerator


@dataclass
class TomogramProteinSpec:
    """
    One cytosolic- or lumen-dwelling protein species to densely pack.

    Attributes
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF file path.
    location : {"cytosol", "lumen"}, optional
        Which region this species is restricted to -- "cytosol" (outside
        every membrane compartment) or "lumen" (inside an enclosed
        compartment, e.g. a vesicle's interior). Default "cytosol".
    ratio : float, optional
        Relative abundance weight among OTHER `ratio`-mode specs sharing
        the same `location` (species at different locations are packed
        independently, so ratios don't compare across locations). Ignored
        if `n_copies` is set. Default 1.0.
    n_copies : int, optional
        If set, place exactly this many instances of this species instead
        of ratio-weighted filling -- "target"/ground-truth semantics.
        Placed FIRST within this spec's `location`, before any
        `ratio`-mode species there (which then avoid the exact-count
        placements via an exclusion field, same mechanism used to avoid
        the membrane shell/filaments). Default None (ratio-weighted
        filler mode).
    """

    pdb_source: str
    location: Literal["cytosol", "lumen"] = "cytosol"
    ratio: float = 1.0
    n_copies: int | None = None

    def __post_init__(self) -> None:
        if self.n_copies is not None and self.n_copies <= 0:
            raise ValueError(
                f"TomogramProteinSpec({self.pdb_source!r}): n_copies must be "
                f"a positive int if set, got {self.n_copies}."
            )


@dataclass
class TomogramPlacement:
    """One placed cytosolic/lumen instance, for ground-truth bookkeeping.

    `role` is "target" for a `TomogramProteinSpec` placed via `n_copies`
    (exact count), "filler" for one placed via `ratio` -- `export_picks`
    excludes "filler" placements by default.
    """

    species_id: str
    location: str
    position_xyz: torch.Tensor
    rotation_matrix: torch.Tensor
    instance_id: int
    role: Literal["target", "filler"] = "target"


@dataclass
class TomogramBeadSpec:
    """
    One gold fiducial bead population to place -- built from real fcc gold
    atoms at bulk number density, so each bead carries lattice texture
    rather than being a uniform ball, and averages to gold's real mean
    inner potential (see `.._grid.BeadGenerator`). Every instance is an
    independent realisation, with its own crystal orientation. Scattered at
    an unrestricted ("any") location: fiducials sit in the ice itself, not
    a specific cytosol/lumen compartment, so there's no `location` field
    here the way `TomogramProteinSpec` has one. Placement still avoids the
    membrane shell and any already-placed filaments/other bead populations
    (see `TomogramSpecimenGenerator._stamp_beads`).

    Attributes
    ----------
    radius : float or [low, high]
        Bead radius, Å. A ``[low, high]`` pair draws a fresh radius
        uniformly per instance -- real colloidal gold is not monodisperse
        (a "10 nm" prep typically spans roughly 8.5-11.5 nm). Radii are
        drawn *before* packing, so the placement's collision test uses each
        bead's own size. Same scalar-or-range spelling as
        `config.ParticleStackConfig`'s `dose`/`defocus`.
    count : int, optional
        Number of instances to place. Default 1.
    """

    radius: ScalarOrRange
    count: int = 1

    def __post_init__(self) -> None:
        self.radius_range = parse_scalar_or_range(self.radius)
        if self.radius_range[0] <= 0:
            raise ValueError(f"TomogramBeadSpec: radius must be > 0, got {self.radius}")
        if self.radius_range[1] < self.radius_range[0]:
            raise ValueError(
                f"TomogramBeadSpec: radius range must be [low, high], got {self.radius}"
            )
        if self.count <= 0:
            raise ValueError(f"TomogramBeadSpec: count must be > 0, got {self.count}")


@dataclass
class BeadPlacement:
    """One placed gold fiducial bead instance, for ground-truth bookkeeping."""

    radius: float
    position_xyz: torch.Tensor
    instance_id: int


@dataclass
class MembraneInstance:
    """
    One membrane, already configured, to composite into a shared tomogram
    alongside others.

    Attributes
    ----------
    generator : MembraneGenerator
        Already-configured (not yet `.generate()`-called) membrane
        generator -- any `shape_backend`, independent per instance. Always
        centered at physical (0,0,0) (`MembraneGenerator`'s own
        convention), then shifted into place via `position_xyz` at
        composite time -- its own `target_shape` need NOT match the
        owning `TomogramSpecimenGenerator`'s (typically shouldn't: leave it
        `None` for a small, auto-sized local grid, see `MembraneGenerator`'s
        own docstring).
    position_xyz : tuple of float, optional
        Physical (x, y, z) offset from the shared tomogram's own center,
        Å. Not settable at construction time -- resolved by
        `generate()` via collision-rejecting random placement (see this
        module's own docstring) -- an instance that doesn't fit gets
        dropped, and `position_xyz` is set here (mutated in place) for
        whichever instances are accepted, so it's inspectable afterward.
    """

    generator: MembraneGenerator
    position_xyz: tuple[float, float, float] | None = field(default=None, init=False)
