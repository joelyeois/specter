"""
Particle placement -- a from-scratch port of CryoTomoSim (CTS)'s
``helper_randomfill.m`` core placement loop, reimplemented in PyTorch. No
dependency on polnet or VTK.

Algorithm: reject-and-retry Monte Carlo. For each attempted placement, draw
a random rotation (via ``specter.rotations``, itself a thin wrapper around
``roma``), rotate the candidate particle's density tensor, pick a random
candidate voxel location (optionally restricted to a location mask, e.g.
"inside a membrane's vesicle interior"), test for voxel overlap against the
volume built so far (binarize-and-max test, the direct equivalent of CTS's
``helper_arrayinsert.m`` ``'overlaptest'`` mode), retry up to
``max_attempts`` times, and on success insert via a max-merge (not sum, to
avoid double-counting density where two placements graze the same boundary
voxel -- the same convention ``specimen/cryoet.py``'s ``_insert_clipped``
already uses).

Three placement modes, matching CTS's own set in ``helper_randomfill.m``:

- ``single`` (default): one independent random rotation + position per
  particle.
- ``cluster``: place one primary particle, then scatter several more
  copies of the same species at randomized nearby (Gaussian-radius)
  offsets around it -- port of CTS's ``radialfill`` used in "cluster" mode.
- ``bundle``: place one primary particle, then place several more radially
  around a shared random axis through it, sliding along the axis and
  growing the radius on repeated failures -- port of CTS's ``radialfill``
  used in "bundle" mode (a rough approximation of filament/bundle
  arrangements, matching CTS's own level of fidelity here, not a real
  filament physics model).

Location-flag gating (``"any"``/``"membrane"``/``"vesicle"``/``"cytosol"``)
restricts candidate voxels to a caller-supplied boolean mask per flag. When
placing a ``single``-mode particle flagged ``"membrane"`` and a matching
entry is present in ``normal_fields``, the particle's own local +Z axis
(physical (x, y, z), matching ``specter.rotations.rotate_volume``'s affine
convention) is aligned to the local surface normal at the chosen voxel, with
a random spin around that axis -- the same "spin then tilt to the normal"
approach CTS's own ``testmem`` subfunction uses, and the same assumption
CTS's membrane-protein inputs require: the supplied particle density must
already be oriented with its own membrane-insertion axis along local +Z.
Cluster/bundle satellites and non-"membrane" flags always use a fully
random rotation -- normal alignment is not applied there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import roma
import torch

from ..arrays import clip_insert_bounds
from ..rotations import build_affine_matrix, random_rotation_matrix, rotate_volume

LocationFlag = str  # "any" | "membrane" | "vesicle" | "cytosol"
PlacementMode = Literal["single", "cluster", "bundle"]


@dataclass
class ParticleSpec:
    """
    One placeable species.

    Attributes
    ----------
    species_id : str
        Identifier for this species (used in the returned placement
        records).
    density : torch.Tensor
        Unrotated scattering-potential (or bulk-material) density template
        for one copy of this species, shape (nz, ny, nx).
    max_count : int
        Maximum number of copies to attempt to place. For ``"cluster"``/
        ``"bundle"`` modes, this caps the total placed across all
        primary+satellite attempts combined (a cluster/bundle "run" stops
        early if it would exceed the remaining budget).
    location : {"any", "membrane", "vesicle", "cytosol"}, optional
        Restricts the PRIMARY placement's candidate voxels to a location
        mask supplied at call time (see ``ParticlePlacer.run``). Satellite
        members of a cluster/bundle are placed near the primary and are
        not independently re-tested against the location mask. "any"
        (default) considers every empty voxel in the volume.
    mode : {"single", "cluster", "bundle"}, optional
        See module docstring. Default "single".
    cluster_size : int, optional
        Number of satellite copies to scatter around the primary, for
        ``mode="cluster"``. Default 8.
    bundle_size : int, optional
        Number of satellite copies to place radially around the shared
        axis, for ``mode="bundle"``. Default 8.
    bundle_length : float, optional
        Axial slide range (Angstrom) for ``mode="bundle"``; satellites are
        slid by a uniform random offset in
        ``[-bundle_length/2, bundle_length/2]`` along the shared axis.
        Defaults to ``3 * max(density.shape)`` (voxel units) if not given.
    max_attempts_per_copy : int, optional
        Retry budget per copy before giving up on that one placement.
        Default 20.
    """

    species_id: str
    density: torch.Tensor
    max_count: int
    location: LocationFlag = "any"
    mode: PlacementMode = "single"
    cluster_size: int = 8
    bundle_size: int = 8
    bundle_length: float | None = None
    max_attempts_per_copy: int = 20


@dataclass
class PlacedInstance:
    """One successfully placed copy, for ground-truth bookkeeping."""

    species_id: str
    center_zyx: torch.Tensor  # (3,) voxel-index center in the specimen volume
    rotation_matrix: torch.Tensor  # (3, 3)


def _random_rotation_aligned_to_normal(
    normal_xyz: torch.Tensor, rng: torch.Generator | None
) -> torch.Tensor:
    """
    Build a rotation matrix that maps the physical +Z axis ([0, 0, 1], in
    the (x, y, z) component convention ``rotate_volume``'s affine matrices
    use) onto `normal_xyz`, composed with a random spin around that same
    axis applied first -- CTS's own "spin then tilt to the normal"
    approach (see ``helper_randomfill.m``'s ``testmem``).

    Parameters
    ----------
    normal_xyz : torch.Tensor
        Shape (3,), need not be unit length (renormalized internally).
    rng : torch.Generator

    Returns
    -------
    torch.Tensor
        Shape (3, 3).
    """
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    target = normal_xyz / (normal_xyz.norm() + 1e-12)

    spin_angle = torch.rand(1, generator=rng).item() * 2 * math.pi
    spin_rotvec = z_axis * spin_angle
    r_spin = roma.rotvec_to_rotmat(spin_rotvec.unsqueeze(0))[0]

    axis = torch.linalg.cross(z_axis, target)
    axis_norm = axis.norm()
    dot = torch.clamp(torch.dot(z_axis, target), -1.0, 1.0)
    if axis_norm < 1e-6:
        # target already (anti-)parallel to z_axis: no tilt needed, or a
        # 180-degree flip about any axis perpendicular to z.
        r_align = (
            torch.eye(3)
            if dot > 0
            else roma.rotvec_to_rotmat(
                (torch.tensor([1.0, 0.0, 0.0]) * math.pi).unsqueeze(0)
            )[0]
        )
    else:
        axis = axis / axis_norm
        angle = torch.acos(dot)
        r_align = roma.rotvec_to_rotmat((axis * angle).unsqueeze(0))[0]

    return r_align @ r_spin


@dataclass
class ParticlePlacer:
    """
    Places one or more particle species into a growing specimen volume via
    reject-and-retry Monte Carlo, CTS-style.

    Parameters
    ----------
    volume : torch.Tensor
        The specimen volume to place into and mutate in place, shape
        (nz, ny, nx).
    density_cutoff : float, optional
        Stop placing (across all remaining species) once the occupied
        voxel fraction (``(volume > 0).float().mean()``) reaches this
        value. Default 0.4, matching CTS's own default (``param_model.m``'s
        ``density`` parameter).
    rng : torch.Generator, optional
        Random generator for rotations/positions. Default: a fresh
        generator seeded from the OS.
    """

    volume: torch.Tensor
    density_cutoff: float = 0.4
    rng: torch.Generator | None = None
    placements: list[PlacedInstance] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = torch.Generator()

    def _occupied_fraction(self) -> float:
        return (self.volume > 0).float().mean().item()

    def _random_candidate_voxel(
        self, location_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Pick a uniformly random voxel index (z, y, x) satisfying
        `location_mask` (or any voxel not yet occupied, if no mask given).
        Returns None if no valid candidate voxels exist at all."""
        if location_mask is not None:
            candidates = torch.nonzero(location_mask, as_tuple=False)
        else:
            candidates = torch.nonzero(self.volume == 0, as_tuple=False)
        if candidates.shape[0] == 0:
            return None
        idx = int(
            torch.randint(0, candidates.shape[0], (1,), generator=self.rng).item()
        )
        return candidates[idx].float()

    def _test_and_insert(
        self,
        rotated_density: torch.Tensor,
        center_zyx: torch.Tensor,
        ignore_overlap_mask: torch.Tensor | None = None,
    ) -> bool:
        """Overlap-test `rotated_density` at `center_zyx`; if no overlap
        with already-occupied voxels, insert (max-merge) and return True.

        `ignore_overlap_mask`, if given, marks voxels that don't count as
        a blocking overlap even if already occupied -- used for
        membrane-flagged placements, where a transmembrane protein is
        expected to displace the local lipid at its own insertion site
        (CTS's own ``testmem`` subfunction does the equivalent by
        subtracting the local membrane density from the destination
        before testing overlap; masking it out of the overlap test here
        has the same effect while still catching overlap with any OTHER
        content at non-masked voxels within the footprint, and still
        composites the membrane density normally via the max-merge below).
        """
        bounds = clip_insert_bounds(
            center_zyx.tolist(), rotated_density.shape, self.volume.shape
        )
        if bounds is None:
            return False
        dst, src = bounds
        existing = self.volume[dst]
        candidate = rotated_density[src]
        existing_for_test = existing
        if ignore_overlap_mask is not None:
            existing_for_test = existing.clone()
            existing_for_test[ignore_overlap_mask[dst]] = 0
        overlap = (existing_for_test > 0) & (candidate > 0)
        if overlap.any():
            return False
        self.volume[dst] = torch.maximum(existing, candidate)
        return True

    def _pick_rotation(
        self,
        spec: ParticleSpec,
        center_zyx: torch.Tensor,
        normal_field: torch.Tensor | None,
    ) -> torch.Tensor:
        """Random rotation, unless `spec` is membrane-flagged single-mode
        and a `normal_field` is available -- then align to the local
        normal (see module docstring)."""
        if spec.location == "membrane" and normal_field is not None:
            zi, yi, xi = (int(v) for v in center_zyx.tolist())
            zi = min(max(zi, 0), normal_field.shape[1] - 1)
            yi = min(max(yi, 0), normal_field.shape[2] - 1)
            xi = min(max(xi, 0), normal_field.shape[3] - 1)
            normal_xyz = normal_field[:, zi, yi, xi]
            return _random_rotation_aligned_to_normal(normal_xyz, self.rng)
        return random_rotation_matrix(batchsize=1)

    def _attempt_single(
        self,
        spec: ParticleSpec,
        mask: torch.Tensor | None,
        normal_field: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Try up to `spec.max_attempts_per_copy` times to place one copy
        of `spec` at a random (mask-restricted) location. Returns
        (center_zyx, rotation_matrix) on success, None on exhausted
        retries or no valid candidate voxels left."""
        # A membrane-flagged particle is expected to displace the local
        # lipid at its own insertion site (see _test_and_insert's
        # docstring) -- only relevant/passed for the "membrane" flag, not
        # "vesicle"/"cytosol"/"any", where any occupied voxel is a real
        # blocking overlap.
        ignore_overlap_mask = mask if spec.location == "membrane" else None
        for _ in range(spec.max_attempts_per_copy):
            center = self._random_candidate_voxel(mask)
            if center is None:
                return None
            R = self._pick_rotation(spec, center, normal_field)
            theta = build_affine_matrix(R.unsqueeze(0))
            rotated = rotate_volume(spec.density, theta, padding_mode="zeros")[0]
            if self._test_and_insert(rotated, center, ignore_overlap_mask):
                self.placements.append(
                    PlacedInstance(
                        species_id=spec.species_id, center_zyx=center, rotation_matrix=R
                    )
                )
                return center, R
        return None

    def _place_single_many(
        self,
        spec: ParticleSpec,
        mask: torch.Tensor | None,
        normal_field: torch.Tensor | None,
    ) -> int:
        placed = 0
        for _ in range(spec.max_count):
            if self._occupied_fraction() >= self.density_cutoff:
                break
            if self._attempt_single(spec, mask, normal_field) is None:
                continue
            placed += 1
        return placed

    def _place_cluster(
        self,
        spec: ParticleSpec,
        mask: torch.Tensor | None,
        normal_field: torch.Tensor | None,
    ) -> int:
        """Port of CTS's ``radialfill`` cluster mode: place one primary,
        then scatter satellites at Gaussian-radius offsets around it."""
        primary = self._attempt_single(spec, mask, normal_field)
        if primary is None:
            return 0
        placed = 1
        primary_center, _ = primary
        particle_size = max(spec.density.shape)
        n_satellites = min(spec.cluster_size, spec.max_count - 1)

        for _ in range(n_satellites):
            if self._occupied_fraction() >= self.density_cutoff:
                break
            for _attempt in range(spec.max_attempts_per_copy):
                offset = torch.randn(3, generator=self.rng) * (particle_size / 2.0)
                candidate_center = primary_center + offset
                R = random_rotation_matrix(batchsize=1)
                theta = build_affine_matrix(R.unsqueeze(0))
                rotated = rotate_volume(spec.density, theta, padding_mode="zeros")[0]
                if self._test_and_insert(rotated, candidate_center):
                    self.placements.append(
                        PlacedInstance(
                            species_id=spec.species_id,
                            center_zyx=candidate_center,
                            rotation_matrix=R,
                        )
                    )
                    placed += 1
                    break
        return placed

    def _place_bundle(
        self,
        spec: ParticleSpec,
        mask: torch.Tensor | None,
        normal_field: torch.Tensor | None,
    ) -> int:
        """Port of CTS's ``radialfill`` bundle mode: place one primary,
        then place satellites radially around a shared random axis through
        it, sliding along the axis and growing the radius on repeated
        failures."""
        primary = self._attempt_single(spec, mask, normal_field)
        if primary is None:
            return 0
        placed = 1
        primary_center, _ = primary

        axis = torch.randn(3, generator=self.rng)
        axis /= axis.norm() + 1e-12
        reference = (
            torch.tensor([0.0, 0.0, 1.0])
            if abs(axis[2].item()) < 0.9
            else torch.tensor([1.0, 0.0, 0.0])
        )
        u = torch.linalg.cross(axis, reference)
        u /= u.norm() + 1e-12
        v = torch.linalg.cross(axis, u)

        particle_size = max(spec.density.shape)
        radius = particle_size / 2.0
        bundle_length = spec.bundle_length or 3 * particle_size
        n_satellites = min(spec.bundle_size, spec.max_count - 1)

        r_accum = radius
        failures = 0
        for _ in range(n_satellites):
            if self._occupied_fraction() >= self.density_cutoff:
                break
            placed_this_one = False
            for _attempt in range(spec.max_attempts_per_copy):
                theta_ang = torch.rand(1, generator=self.rng).item() * 2 * math.pi
                radial = r_accum * (u * math.cos(theta_ang) + v * math.sin(theta_ang))
                slide = (
                    (torch.rand(1, generator=self.rng).item() - 0.5)
                    * bundle_length
                    * axis
                )
                candidate_center = primary_center + radial + slide

                spin_angle = torch.rand(1, generator=self.rng).item() * 2 * math.pi
                R = roma.rotvec_to_rotmat((axis * spin_angle).unsqueeze(0))[0]
                theta = build_affine_matrix(R.unsqueeze(0))
                rotated = rotate_volume(spec.density, theta, padding_mode="zeros")[0]
                if self._test_and_insert(rotated, candidate_center):
                    self.placements.append(
                        PlacedInstance(
                            species_id=spec.species_id,
                            center_zyx=candidate_center,
                            rotation_matrix=R,
                        )
                    )
                    placed += 1
                    placed_this_one = True
                    failures = 0
                    break
            if not placed_this_one:
                # CTS's own radialfill grows the radius after repeated
                # failures, to avoid getting permanently stuck at a
                # too-crowded radius.
                failures += 1
                r_accum += radius * failures / 10.0
        return placed

    def place_species(
        self,
        spec: ParticleSpec,
        location_masks: dict[str, torch.Tensor] | None = None,
        normal_fields: dict[str, torch.Tensor] | None = None,
    ) -> int:
        """
        Attempt to place up to `spec.max_count` copies of one species,
        dispatching on `spec.mode` (see module docstring).

        Parameters
        ----------
        spec : ParticleSpec
        location_masks : dict, optional
            Maps location-flag name ("membrane"/"vesicle"/"cytosol") to a
            boolean mask over `self.volume`'s voxels. Required if
            `spec.location != "any"`.
        normal_fields : dict, optional
            Maps location-flag name to a physical-(x,y,z)-component normal
            vector field, shape (3, nz, ny, nx), matching `self.volume`'s
            shape. Only consulted for `spec.location == "membrane"` and
            `spec.mode == "single"` (see module docstring).

        Returns
        -------
        int
            Number of copies actually placed.
        """
        mask = None
        if spec.location != "any":
            if location_masks is None or spec.location not in location_masks:
                raise ValueError(
                    f"species {spec.species_id!r} requests location "
                    f"{spec.location!r} but no matching mask was supplied"
                )
            mask = location_masks[spec.location]

        normal_field = None
        if normal_fields is not None and spec.location in normal_fields:
            normal_field = normal_fields[spec.location]

        if spec.mode == "single":
            return self._place_single_many(spec, mask, normal_field)
        elif spec.mode == "cluster":
            return self._place_cluster(spec, mask, normal_field)
        elif spec.mode == "bundle":
            return self._place_bundle(spec, mask, normal_field)
        raise ValueError(f"unknown placement mode {spec.mode!r}")

    def run(
        self,
        specs: list[ParticleSpec],
        location_masks: dict[str, torch.Tensor] | None = None,
        normal_fields: dict[str, torch.Tensor] | None = None,
    ) -> list[PlacedInstance]:
        """
        Place every species in `specs`, in order, respecting the shared
        `density_cutoff` across all of them (matches CTS's own
        layer-by-layer loop in ``helper_randomfill.m``).

        Returns
        -------
        list of PlacedInstance
            All successful placements from this call (also accumulated in
            `self.placements`).
        """
        before = len(self.placements)
        for spec in specs:
            if self._occupied_fraction() >= self.density_cutoff:
                break
            self.place_species(spec, location_masks, normal_fields)
        return self.placements[before:]
