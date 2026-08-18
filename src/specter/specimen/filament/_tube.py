"""
Microtubule placement: whole tubes scattered through a specimen volume.

A microtubule is expressed here as many rigid copies of one alpha-beta
tubulin dimer -- exactly the form ``_generator.place_filaments`` already
returns, and exactly what ``TomogramSpecimenGenerator._stamp_filaments``
already knows how to render. That is deliberate: tube geometry is decided
in this module, and rendering, carbon-film exclusion, instance labelling
and pick export all come for free from the existing filament path.

Per ring of the lattice, the axis path supplies a point and a
parallel-transport frame; the lattice (`_lattice.solve_tube_lattice`)
supplies each protofilament's azimuth and axial register. Note the frame
must be parallel-transported rather than built per-point: a frame that
rolls about the tangent shears the protofilaments apart along a bend (see
``_frames``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._frames import parallel_transport_frames
from ._generator import FilamentInstance
from ._lattice import MicrotubuleSpec, TubeLattice, thermal_flex_deg
from ._path import generate_filament_path
from ._tubulin import extract_mt_dimer


@dataclass
class MicrotubuleInstance:
    """One placed microtubule, for ground-truth bookkeeping.

    The per-dimer `FilamentInstance` copies carry the density; this carries
    the object. A pick file listing ~950 dimers per tube is rarely what a
    consumer wants -- the axis polyline is.

    Attributes
    ----------
    tube_id : int
        Index of this microtubule within its species.
    code : str
        Dimer structure the tube was built from.
    axis_xyz : torch.Tensor
        Axis polyline, shape ``(n_rings, 3)``, physical ``(x, y, z)``,
        corner-relative (the convention `export_picks` writes out).
    lattice : TubeLattice
        Surface lattice used, including radius and seam offset.
    """

    tube_id: int
    code: str
    axis_xyz: torch.Tensor
    lattice: TubeLattice


def _random_unit_vector(generator: torch.Generator | None) -> torch.Tensor:
    v = torch.randn(3, generator=generator)
    return v / v.norm().clamp_min(1e-8)


def _sample_direction(
    length: float,
    extent_xyz: torch.Tensor,
    confine_to_slab: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """
    Draw an initial direction, optionally rejecting ones that would leave
    the volume's thinnest dimension.

    A micrometre-long, 25 nm-wide tube cannot be steeply tilted inside
    100-300 nm of ice: the geometry forbids it. Rejecting on the actual box
    rather than capping a tilt angle keeps that a consequence of the
    specimen, not a tuned parameter.
    """
    if not confine_to_slab:
        return _random_unit_vector(generator)

    thin_axis = int(torch.argmin(extent_xyz))
    max_component = float(extent_xyz[thin_axis]) / max(length, 1e-8)
    if max_component >= 1.0:
        return _random_unit_vector(generator)

    for _ in range(64):
        direction = _random_unit_vector(generator)
        if abs(float(direction[thin_axis])) <= max_component:
            return direction

    # Fall back to a direction that satisfies the constraint by
    # construction rather than looping forever on a very thin slab.
    direction = _random_unit_vector(generator)
    direction[thin_axis] = direction[thin_axis].sign() * max_component
    return direction / direction.norm().clamp_min(1e-8)


def _arc_path(
    n_rings: int,
    step: float,
    bend_radius: float,
    origin_xyz: torch.Tensor,
    direction_xyz: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Constant-curvature arc through ``origin_xyz`` along ``direction_xyz``.

    The mechanically bent microtubules seen in cellular tomograms curve
    smoothly; a random walk with a larger flex angle wanders as far but
    does it by kinking, which is a different object.
    """
    tangent = direction_xyz / direction_xyz.norm().clamp_min(1e-8)
    # Bend within a randomly chosen plane containing the tangent.
    normal = _random_unit_vector(generator)
    normal = normal - (normal * tangent).sum() * tangent
    if float(normal.norm()) < 1e-6:
        normal = torch.tensor([0.0, 0.0, 1.0], dtype=tangent.dtype)
        normal = normal - (normal * tangent).sum() * tangent
    normal = normal / normal.norm().clamp_min(1e-8)

    theta = (torch.arange(n_rings, dtype=tangent.dtype) * step) / bend_radius
    theta = theta - theta.mean()  # centre the arc on the origin
    positions = (
        origin_xyz
        + bend_radius * torch.sin(theta).unsqueeze(1) * tangent
        + bend_radius * (1.0 - torch.cos(theta)).unsqueeze(1) * normal
    )
    return positions


def microtubule_axis_path(
    spec: MicrotubuleSpec,
    n_rings: int,
    origin_xyz: torch.Tensor,
    direction_xyz: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Axis polyline for one microtubule, sampled once per lattice ring.

    Parameters
    ----------
    spec : MicrotubuleSpec
    n_rings : int
        Number of rings (dimer repeats) along the tube.
    origin_xyz : torch.Tensor
        Point the tube is centred on, shape ``(3,)``.
    direction_xyz : torch.Tensor
        Overall direction, shape ``(3,)``.
    generator : torch.Generator, optional

    Returns
    -------
    torch.Tensor
        Axis points, shape ``(n_rings, 3)``.

    Notes
    -----
    The path is centred on ``origin_xyz`` rather than started there, so a
    microtubule crosses the field instead of trailing off from a random
    interior point.
    """
    lattice = spec.lattice()
    step = lattice.dimer_repeat

    if spec.bend_radius is not None:
        if spec.bend_radius <= 0:
            raise ValueError(f"bend_radius must be > 0, got {spec.bend_radius}")
        return _arc_path(
            n_rings, step, spec.bend_radius, origin_xyz, direction_xyz, generator
        )

    start = origin_xyz - direction_xyz * (0.5 * n_rings * step)
    return generate_filament_path(
        n_rings,
        step,
        thermal_flex_deg(step),
        origin_xyz=start,
        generator=generator,
        direction_xyz=direction_xyz,
    )


def build_tube_instances(
    code: str,
    tube_id: int,
    axis_xyz: torch.Tensor,
    lattice: TubeLattice,
) -> list[FilamentInstance]:
    """
    Lay one microtubule's dimers onto an axis polyline.

    Parameters
    ----------
    code : str
        Dimer structure, carried through to each instance.
    tube_id : int
        Identifies the tube; becomes each instance's ``filament_id``.
    axis_xyz : torch.Tensor
        Axis points, shape ``(n_rings, 3)``.
    lattice : TubeLattice

    Returns
    -------
    list of FilamentInstance
        ``n_rings * n_protofilaments`` copies. Each copy's rotation maps
        the template's ``+Z`` onto the local axis tangent and its ``+X``
        onto the outward radial direction at that protofilament's azimuth
        -- which is why the template has to be in the microtubule frame
        (see `_tubulin.extract_mt_dimer`).
    """
    frames = parallel_transport_frames(axis_xyz)
    n_pf = lattice.n_protofilaments

    phi = 2 * math.pi * torch.arange(n_pf, dtype=axis_xyz.dtype) / n_pf
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    z_offset = torch.arange(n_pf, dtype=axis_xyz.dtype) * lattice.stagger

    # Local offsets of every protofilament within one ring, in the ring's
    # own frame: out to the wall radius at its azimuth, then along the axis
    # by its register.
    local = torch.stack(
        [lattice.radius * cos_phi, lattice.radius * sin_phi, z_offset], dim=1
    )
    # Rotation taking the template's +X to the outward radial direction and
    # leaving +Z along the tube axis.
    spin = torch.zeros((n_pf, 3, 3), dtype=axis_xyz.dtype)
    spin[:, 0, 0] = cos_phi
    spin[:, 0, 1] = -sin_phi
    spin[:, 1, 0] = sin_phi
    spin[:, 1, 1] = cos_phi
    spin[:, 2, 2] = 1.0

    instances: list[FilamentInstance] = []
    for ring, (point, frame) in enumerate(zip(axis_xyz, frames)):
        positions = point + local @ frame.T
        rotations = frame @ spin
        for p in range(n_pf):
            instances.append(
                FilamentInstance(
                    code=code,
                    filament_id=tube_id,
                    monomer_index=ring,
                    position_xyz=positions[p],
                    rotation_matrix=rotations[p],
                    protofilament_index=p,
                )
            )
    return instances


def place_microtubules(
    specs: list[MicrotubuleSpec],
    target_shape: tuple[int, int, int],
    voxel_size: float,
    generator: torch.Generator | None = None,
    pdb_cache_dir: str | None = None,
) -> tuple[list[FilamentInstance], list[MicrotubuleInstance]]:
    """
    Place every microtubule for every spec.

    Parameters
    ----------
    specs : list of MicrotubuleSpec
    target_shape : tuple of int
        Specimen volume shape ``(Z, Y, X)``, voxels.
    voxel_size : float
        Voxel size, Å.
    generator : torch.Generator, optional
        Random generator for centres, directions and path turns.
    pdb_cache_dir : str, optional
        Where to cache the source structure and extracted dimer. Default
        None: specter's usual PDB cache.

    Returns
    -------
    instances : list of FilamentInstance
        Per-dimer copies, ready for the tomogram generator's filament
        stamping.
    tubes : list of MicrotubuleInstance
        Per-tube records (axis polyline, lattice).

    Notes
    -----
    Like `place_filaments`, this has no obstacle awareness: microtubules do
    not avoid one another, the membrane shell, or the carbon film (monomers
    landing in carbon are dropped downstream). Parts that wander outside the
    volume are truncated when rendered.
    """
    extent_xyz = torch.tensor(target_shape[::-1], dtype=torch.float32) * voxel_size
    diagonal = float(extent_xyz.norm())

    instances: list[FilamentInstance] = []
    tubes: list[MicrotubuleInstance] = []

    for spec in specs:
        lattice = spec.lattice()
        code = spec.code or extract_mt_dimer(savefolder=pdb_cache_dir)
        length = spec.length if spec.length is not None else diagonal
        n_rings = max(2, int(round(length / lattice.dimer_repeat)))

        for tube_id in range(spec.n_copies):
            origin = torch.rand(3, generator=generator) * extent_xyz
            direction = _sample_direction(
                length, extent_xyz, spec.confine_to_slab, generator
            )
            axis = microtubule_axis_path(
                spec, n_rings, origin, direction, generator=generator
            )
            instances += build_tube_instances(code, tube_id, axis, lattice)
            tubes.append(
                MicrotubuleInstance(
                    tube_id=tube_id, code=code, axis_xyz=axis, lattice=lattice
                )
            )

    return instances, tubes
