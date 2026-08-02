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

Two placement STRATEGIES share the exact same per-voxel overlap test
(``_local_overlap_test_and_insert``, always exact -- no bounding-sphere/
ellipsoid approximation anywhere in this module) but differ in how
candidate positions are found for the unmasked ("any"-location) case:

- ``ParticlePlacer`` (the original): candidates for "any" are drawn via
  direct uniform per-axis index draws with O(1) occupancy rejection
  (falling back to an exhaustive ``nonzero`` scan only near saturation).
  Good for small-to-moderate volumes/occupancies.
- ``HierarchicalParticlePlacer``: adds a coarse/fine spatial hierarchy on
  top of the same exact fine-grained test -- a downsampled coarse
  occupancy grid tracks which coarse cells are already fully occupied, so
  candidate sampling for "any" can cheaply skip whole already-saturated
  regions (a ``nonzero`` scan over the much-smaller coarse grid, not the
  fine volume) instead of repeatedly drawing candidates that fall in
  already-packed space as global occupancy climbs. Only ``mode="single"``
  is supported for this first pass -- ``"cluster"``/``"bundle"`` raise
  ``NotImplementedError`` (deferred, not silently ignored). Masked
  location flags (membrane/vesicle/cytosol) use the same cached-nonzero-
  over-the-mask approach as ``ParticlePlacer`` in both strategies, since a
  location mask is typically already a bounded region, not a full-volume-
  scan problem the way "any" is.

Both classes share the same ``ParticleSpec``/``PlacedInstance`` types and
``place_species``/``run`` interface, so either is a drop-in choice for
``CryoTomoSimSpecimenGenerator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import roma
import torch

from ..arrays import clip_insert_bounds, coarse_occupancy_mask
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


def _local_overlap_test_and_insert(
    volume: torch.Tensor,
    rotated_density: torch.Tensor,
    center_zyx: torch.Tensor,
    ignore_overlap_mask: torch.Tensor | None = None,
) -> tuple[bool, int, tuple[slice, ...] | None]:
    """
    Exact per-voxel overlap test + max-merge insert, local to the
    candidate's own bounding box via ``clip_insert_bounds`` -- shared by
    both ``ParticlePlacer`` and ``HierarchicalParticlePlacer``, since the
    fine-grained check itself is identical between the two strategies
    (they only differ in how a candidate position is FOUND, not in how
    it's verified/rendered).

    `ignore_overlap_mask`, if given, marks voxels that don't count as a
    blocking overlap even if already occupied -- used for membrane-flagged
    placements, where a transmembrane protein is expected to displace the
    local lipid at its own insertion site (CTS's own ``testmem``
    subfunction does the equivalent by subtracting the local membrane
    density from the destination before testing overlap; masking it out
    of the overlap test here has the same effect while still catching
    overlap with any OTHER content at non-masked voxels within the
    footprint, and still composites the membrane density normally via the
    max-merge below).

    Returns
    -------
    success : bool
    newly_occupied : int
        Number of voxels that transitioned from unoccupied to occupied
        (0 if `success` is False), for the caller's own running occupancy
        count.
    dst : tuple of slice, or None
        The destination region actually touched in `volume` (None if
        `success` is False or the candidate fell entirely outside
        `volume`'s bounds) -- `HierarchicalParticlePlacer` uses this to
        know which coarse cells to refresh.
    """
    bounds = clip_insert_bounds(
        center_zyx.tolist(), rotated_density.shape, volume.shape
    )
    if bounds is None:
        return False, 0, None
    dst, src = bounds
    existing = volume[dst]
    candidate = rotated_density[src]
    existing_for_test = existing
    if ignore_overlap_mask is not None:
        existing_for_test = existing.clone()
        existing_for_test[ignore_overlap_mask[dst]] = 0
    overlap = (existing_for_test > 0) & (candidate > 0)
    if overlap.any():
        return False, 0, None
    # Track the occupancy delta from the ORIGINAL existing values (not
    # existing_for_test, which may have been zeroed out under
    # ignore_overlap_mask) so the caller's occupied count stays exact
    # regardless of membrane-overlap-ignore placements.
    newly_occupied = int(((existing == 0) & (candidate > 0)).sum().item())
    volume[dst] = torch.maximum(existing, candidate)
    return True, newly_occupied, dst


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
    _mask_candidate_cache: dict[int, torch.Tensor] = field(
        default_factory=dict, repr=False
    )
    #: Running count of occupied (> 0) voxels, lazily initialized (one
    #: full-volume scan) on first use and then updated incrementally by
    #: `_test_and_insert` -- see `_occupied_fraction`'s docstring for why
    #: this can't just be computed once in `__post_init__` instead.
    _occupied_count: int | None = field(default=None, repr=False)

    #: Direct uniform-index draws to try (see _random_candidate_voxel's
    #: unmasked path) before falling back to an exhaustive nonzero scan.
    #: At density_cutoff <= ~0.6 the free-voxel fraction at the point
    #: placement stops is always >= ~0.4, so the expected number of draws
    #: to hit a free voxel is ~2.5 even in the worst case at the cutoff
    #: boundary -- 50 is a large safety margin for locally denser regions
    #: (e.g. near a just-placed cluster), not a tight bound.
    _DIRECT_SAMPLE_TRIES = 50

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = torch.Generator()

    def _occupied_fraction(self) -> float:
        """Fraction of occupied (> 0) voxels in `self.volume`.

        Called before every single placement attempt (not just
        successes) to check the density cutoff, so at large volumes
        (tens of millions of voxels) recomputing this from scratch every
        time would itself dominate runtime -- `_occupied_count` is
        computed fresh exactly once, lazily, the first time this is
        called (capturing whatever pre-existing content, e.g. a carbon
        film, the caller already added to `self.volume` before placement
        started -- this placer has no way to know about such prior
        mutations except by scanning once), and updated incrementally by
        `_test_and_insert` from then on.
        """
        if self._occupied_count is None:
            self._occupied_count = int((self.volume > 0).sum().item())
        return self._occupied_count / self.volume.numel()

    def _random_candidate_voxel(
        self, location_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Pick a uniformly random voxel index (z, y, x) satisfying
        `location_mask` (or any voxel not yet occupied, if no mask given).
        Returns None if no valid candidate voxels exist at all.

        Unmasked ("any") case: samples via direct uniform per-axis index
        draws with cheap O(1) occupancy rejection, instead of
        materializing the full free-voxel index list via
        ``torch.nonzero`` -- at large volumes (tens of millions of
        voxels) and many placement attempts, a full-volume ``nonzero``
        scan on every single attempt (not just successes) would dominate
        runtime. Falls back to the exhaustive scan only if direct
        sampling doesn't find a free voxel within
        `_DIRECT_SAMPLE_TRIES` (i.e. the volume is likely nearly
        saturated) -- a correctness backstop, not the common path.

        Masked case: a location mask (membrane/vesicle/cytosol) is built
        once per specimen and doesn't change across placement attempts
        within a run (unlike `self.volume`'s occupancy, which changes on
        every successful placement) -- so its candidate-index list is
        cached (keyed by the mask tensor's identity) rather than
        recomputed via ``nonzero`` on every attempt.
        """
        if location_mask is None:
            shape = self.volume.shape
            shape_t = torch.tensor(shape, dtype=torch.float32)
            for _ in range(self._DIRECT_SAMPLE_TRIES):
                idx = (torch.rand(3, generator=self.rng) * shape_t).long()
                if self.volume[idx[0], idx[1], idx[2]] == 0:
                    return idx.float()
            # Fallback: volume is likely nearly saturated for uniform
            # sampling to find a free voxel quickly.
            candidates = torch.nonzero(self.volume == 0, as_tuple=False)
        else:
            key = id(location_mask)
            cached = self._mask_candidate_cache.get(key)
            if cached is None:
                cached = torch.nonzero(location_mask, as_tuple=False)
                self._mask_candidate_cache[key] = cached
            candidates = cached

        if candidates.shape[0] == 0:
            return None
        chosen_idx = int(
            torch.randint(0, candidates.shape[0], (1,), generator=self.rng).item()
        )
        return candidates[chosen_idx].float()

    def _test_and_insert(
        self,
        rotated_density: torch.Tensor,
        center_zyx: torch.Tensor,
        ignore_overlap_mask: torch.Tensor | None = None,
    ) -> bool:
        """Overlap-test `rotated_density` at `center_zyx`; if no overlap
        with already-occupied voxels, insert (max-merge) and return True.
        Thin wrapper around the module-level `_local_overlap_test_and_insert`
        (shared with `HierarchicalParticlePlacer`) that also maintains this
        placer's own incremental `_occupied_count`. See that function's
        docstring for `ignore_overlap_mask`'s semantics.
        """
        success, newly_occupied, _dst = _local_overlap_test_and_insert(
            self.volume, rotated_density, center_zyx, ignore_overlap_mask
        )
        if success and self._occupied_count is not None:
            self._occupied_count += newly_occupied
        return success

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
        ignore_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Try up to `spec.max_attempts_per_copy` times to place one copy
        of `spec` at a random (mask-restricted) location. Returns
        (center_zyx, rotation_matrix) on success, None on exhausted
        retries or no valid candidate voxels left.

        `mask` (candidate SAMPLING region) and `ignore_mask` (voxels that
        don't count as a blocking overlap) are deliberately separate
        parameters, not the same mask reused for both purposes: for a
        membrane-flagged spec, `mask` is expected to be the membrane's
        thin mid-thickness SKELETON (so insertion depth is consistent,
        not uniformly random across the full bilayer thickness), while
        `ignore_mask` is expected to be the FULL bilayer thickness
        footprint (so the overlap test doesn't wrongly block a candidate
        whose rotated density genuinely spans the whole membrane
        thickness once centered on the skeleton). See
        `_test_and_insert`'s docstring for why the ignore mechanism
        itself exists.
        """
        for _ in range(spec.max_attempts_per_copy):
            center = self._random_candidate_voxel(mask)
            if center is None:
                return None
            R = self._pick_rotation(spec, center, normal_field)
            theta = build_affine_matrix(R.unsqueeze(0))
            rotated = rotate_volume(spec.density, theta, padding_mode="zeros")[0]
            if self._test_and_insert(rotated, center, ignore_mask):
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
        ignore_mask: torch.Tensor | None = None,
    ) -> int:
        placed = 0
        for _ in range(spec.max_count):
            if self._occupied_fraction() >= self.density_cutoff:
                break
            if self._attempt_single(spec, mask, normal_field, ignore_mask) is None:
                continue
            placed += 1
        return placed

    def _place_cluster(
        self,
        spec: ParticleSpec,
        mask: torch.Tensor | None,
        normal_field: torch.Tensor | None,
        ignore_mask: torch.Tensor | None = None,
    ) -> int:
        """Port of CTS's ``radialfill`` cluster mode: place one primary,
        then scatter satellites at Gaussian-radius offsets around it."""
        primary = self._attempt_single(spec, mask, normal_field, ignore_mask)
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
        ignore_mask: torch.Tensor | None = None,
    ) -> int:
        """Port of CTS's ``radialfill`` bundle mode: place one primary,
        then place satellites radially around a shared random axis through
        it, sliding along the axis and growing the radius on repeated
        failures."""
        primary = self._attempt_single(spec, mask, normal_field, ignore_mask)
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
        ignore_masks: dict[str, torch.Tensor] | None = None,
    ) -> int:
        """
        Attempt to place up to `spec.max_count` copies of one species,
        dispatching on `spec.mode` (see module docstring).

        Parameters
        ----------
        spec : ParticleSpec
        location_masks : dict, optional
            Maps location-flag name ("membrane"/"vesicle"/"cytosol") to a
            boolean mask over `self.volume`'s voxels -- the region
            candidate positions are SAMPLED from. Required if
            `spec.location != "any"`. For `"membrane"`, this is expected
            to be the membrane's thin mid-thickness skeleton (see
            `_attempt_single`'s docstring), not the full bilayer
            thickness footprint.
        normal_fields : dict, optional
            Maps location-flag name to a physical-(x,y,z)-component normal
            vector field, shape (3, nz, ny, nx), matching `self.volume`'s
            shape. Only consulted for `spec.location == "membrane"` and
            `spec.mode == "single"` (see module docstring).
        ignore_masks : dict, optional
            Maps location-flag name to a boolean mask of voxels that don't
            count as a blocking overlap even if already occupied. Only
            consulted for `spec.location == "membrane"` -- expected to be
            the membrane's FULL bilayer thickness footprint (deliberately
            NOT the same mask as `location_masks["membrane"]`; see
            `_attempt_single`'s docstring for why these two must stay
            separate). If omitted for a `"membrane"`-flagged spec, no
            overlap is ignored (every occupied voxel blocks placement,
            which in practice makes membrane-embedded placement fail
            almost entirely, since the candidate's own footprint
            genuinely overlaps the membrane material it's meant to sit
            in).

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

        ignore_mask = None
        if (
            spec.location == "membrane"
            and ignore_masks is not None
            and spec.location in ignore_masks
        ):
            ignore_mask = ignore_masks[spec.location]

        if spec.mode == "single":
            return self._place_single_many(spec, mask, normal_field, ignore_mask)
        elif spec.mode == "cluster":
            return self._place_cluster(spec, mask, normal_field, ignore_mask)
        elif spec.mode == "bundle":
            return self._place_bundle(spec, mask, normal_field, ignore_mask)
        raise ValueError(f"unknown placement mode {spec.mode!r}")

    def run(
        self,
        specs: list[ParticleSpec],
        location_masks: dict[str, torch.Tensor] | None = None,
        normal_fields: dict[str, torch.Tensor] | None = None,
        ignore_masks: dict[str, torch.Tensor] | None = None,
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
            self.place_species(spec, location_masks, normal_fields, ignore_masks)
        return self.placements[before:]


@dataclass
class HierarchicalParticlePlacer:
    """
    Exact-collision particle placer -- same per-voxel correctness guarantee
    as `ParticlePlacer` (no bounding-sphere/ellipsoid approximation
    anywhere) -- with a two-level coarse/fine spatial hierarchy for fast
    candidate discovery in the unmasked ("any"-location) case. See the
    module docstring's "Two placement STRATEGIES" section for the
    conceptual comparison against `ParticlePlacer`.

    Limitation: only `mode="single"` is supported in this first pass;
    `"cluster"`/`"bundle"` raise `NotImplementedError` (deferred, not
    silently ignored -- `ParticlePlacer` still supports those directly).

    Parameters
    ----------
    volume : torch.Tensor
        The specimen volume to place into and mutate in place, shape
        (nz, ny, nx).
    density_cutoff : float, optional
        Same semantics as `ParticlePlacer.density_cutoff`. Default 0.4.
    coarse_factor : int, optional
        Coarse-grid downsample factor per axis. Default 8 -- a reasonable
        middle ground for typical protein-sized particles (tens of voxels
        across) at typical cryo-ET voxel sizes: large enough that a
        meaningful number of fine voxels share one coarse cell (so "fully
        occupied" cells actually start appearing as density climbs,
        making the skip-fully-occupied-cells optimization pay off), small
        enough that a coarse cell isn't so big it stays "not full" long
        after most of its own volume is already packed (which would waste
        attempts sampling into a nearly-saturated cell just as a full-
        volume approach would). Pass a larger factor for much bigger
        particles relative to the volume (e.g. large membranes), smaller
        for very fine/small ones.
    full_occupancy_threshold : float, optional
        Occupied-voxel-fraction (within one coarse cell) at or above which
        the cell is treated as "full" and skipped for new candidates.
        Default 0.7 -- deliberately NOT 1.0 (literal "every voxel
        nonzero"): for spherical/irregular particles, a handful of stray
        never-touched voxels (e.g. the gaps between packed spheres) can
        persist in a cell indefinitely, so requiring exact 100% occupancy
        means cells rarely if ever get marked full, and the coarse grid
        stops paying for itself right when it matters most (high global
        occupancy, where most candidates would otherwise be wasted
        attempts in already-crowded space). See
        `coarse_occupancy_mask`'s docstring for the full reasoning --
        this only trades candidate-search completeness for speed, never
        correctness (the fine-grained exact test is unaffected).
    rng : torch.Generator, optional
        Random generator for rotations/positions. Default: a fresh
        generator seeded from the OS.
    """

    volume: torch.Tensor
    density_cutoff: float = 0.4
    coarse_factor: int = 8
    full_occupancy_threshold: float = 0.7
    rng: torch.Generator | None = None
    placements: list[PlacedInstance] = field(default_factory=list)
    _mask_candidate_cache: dict[int, torch.Tensor] = field(
        default_factory=dict, repr=False
    )
    _occupied_count: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = torch.Generator()
        cf = self.coarse_factor
        nz, ny, nx = self.volume.shape
        self._coarse_shape = (
            (nz + cf - 1) // cf,
            (ny + cf - 1) // cf,
            (nx + cf - 1) // cf,
        )
        # One-time vectorized build -- captures whatever pre-existing
        # content (e.g. a carbon film) the caller already added to
        # `self.volume` before this placer was constructed, same reason
        # `_occupied_count` is lazily scanned rather than assumed zero.
        self._coarse_full_mask = coarse_occupancy_mask(
            self.volume, cf, self.full_occupancy_threshold
        )

    def _occupied_fraction(self) -> float:
        if self._occupied_count is None:
            self._occupied_count = int((self.volume > 0).sum().item())
        return self._occupied_count / self.volume.numel()

    def _coarse_cell_fine_slice(
        self, cz: int, cy: int, cx: int
    ) -> tuple[slice, slice, slice]:
        cf = self.coarse_factor
        nz, ny, nx = self.volume.shape
        return (
            slice(cz * cf, min((cz + 1) * cf, nz)),
            slice(cy * cf, min((cy + 1) * cf, ny)),
            slice(cx * cf, min((cx + 1) * cf, nx)),
        )

    def _refresh_coarse_cell(self, cz: int, cy: int, cx: int) -> None:
        sl = self._coarse_cell_fine_slice(cz, cy, cx)
        block = self.volume[sl]
        occupied_fraction = (block > 0).float().mean()
        self._coarse_full_mask[cz, cy, cx] = (
            occupied_fraction >= self.full_occupancy_threshold
        )

    def _refresh_coarse_cells_touched(self, dst: tuple[slice, ...]) -> None:
        """Refresh only the (few) coarse cells overlapping `dst` -- bounded
        by one placed particle's own footprint in coarse-cell units, never
        a full coarse-grid rebuild."""
        cf = self.coarse_factor
        cz0, cz1 = dst[0].start // cf, (dst[0].stop - 1) // cf + 1
        cy0, cy1 = dst[1].start // cf, (dst[1].stop - 1) // cf + 1
        cx0, cx1 = dst[2].start // cf, (dst[2].stop - 1) // cf + 1
        for cz in range(cz0, cz1):
            for cy in range(cy0, cy1):
                for cx in range(cx0, cx1):
                    self._refresh_coarse_cell(cz, cy, cx)

    def _random_candidate_voxel(
        self, location_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Masked case: identical cached-nonzero-over-the-mask approach as
        `ParticlePlacer`. Unmasked case: draw a coarse cell uniformly among
        NOT-fully-occupied cells (cheap -- a `nonzero` scan over the small
        coarse grid, not the fine volume), then a random fine voxel within
        it."""
        if location_mask is not None:
            key = id(location_mask)
            cached = self._mask_candidate_cache.get(key)
            if cached is None:
                cached = torch.nonzero(location_mask, as_tuple=False)
                self._mask_candidate_cache[key] = cached
            candidates = cached
            if candidates.shape[0] == 0:
                return None
            chosen_idx = int(
                torch.randint(0, candidates.shape[0], (1,), generator=self.rng).item()
            )
            return candidates[chosen_idx].float()

        free_cells = torch.nonzero(~self._coarse_full_mask, as_tuple=False)
        if free_cells.shape[0] == 0:
            return None
        chosen_idx = int(
            torch.randint(0, free_cells.shape[0], (1,), generator=self.rng).item()
        )
        cell = free_cells[chosen_idx]
        sl = self._coarse_cell_fine_slice(int(cell[0]), int(cell[1]), int(cell[2]))
        lo = torch.tensor([sl[0].start, sl[1].start, sl[2].start], dtype=torch.float32)
        span = torch.tensor(
            [
                sl[0].stop - sl[0].start,
                sl[1].stop - sl[1].start,
                sl[2].stop - sl[2].start,
            ],
            dtype=torch.float32,
        )
        offset = (torch.rand(3, generator=self.rng) * span).floor()
        return lo + offset

    def _pick_rotation(
        self,
        spec: ParticleSpec,
        center_zyx: torch.Tensor,
        normal_field: torch.Tensor | None,
    ) -> torch.Tensor:
        """Identical logic to `ParticlePlacer._pick_rotation`."""
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
        ignore_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """See `ParticlePlacer._attempt_single`'s docstring for why `mask`
        (candidate sampling) and `ignore_mask` (overlap-ignore) are kept
        separate parameters rather than one shared mask."""
        if spec.mode != "single":
            raise NotImplementedError(
                f"HierarchicalParticlePlacer only supports mode='single' "
                f"in this first pass (got spec.mode={spec.mode!r} for "
                f"species {spec.species_id!r}); use ParticlePlacer for "
                f"'cluster'/'bundle'."
            )
        for _ in range(spec.max_attempts_per_copy):
            center = self._random_candidate_voxel(mask)
            if center is None:
                return None
            R = self._pick_rotation(spec, center, normal_field)
            theta = build_affine_matrix(R.unsqueeze(0))
            rotated = rotate_volume(spec.density, theta, padding_mode="zeros")[0]
            success, newly_occupied, dst = _local_overlap_test_and_insert(
                self.volume, rotated, center, ignore_mask
            )
            if success:
                if self._occupied_count is not None:
                    self._occupied_count += newly_occupied
                if dst is not None:
                    self._refresh_coarse_cells_touched(dst)
                self.placements.append(
                    PlacedInstance(
                        species_id=spec.species_id,
                        center_zyx=center,
                        rotation_matrix=R,
                    )
                )
                return center, R
        return None

    def place_species(
        self,
        spec: ParticleSpec,
        location_masks: dict[str, torch.Tensor] | None = None,
        normal_fields: dict[str, torch.Tensor] | None = None,
        ignore_masks: dict[str, torch.Tensor] | None = None,
    ) -> int:
        """
        Attempt to place up to `spec.max_count` copies of one species.
        Only `spec.mode == "single"` is supported (see class docstring).

        Parameters
        ----------
        spec : ParticleSpec
        location_masks : dict, optional
            Maps location-flag name ("membrane"/"vesicle"/"cytosol") to a
            boolean mask over `self.volume`'s voxels -- the candidate
            SAMPLING region. Required if `spec.location != "any"`. For
            `"membrane"`, expected to be the thin mid-thickness skeleton,
            not the full bilayer footprint (see
            `ParticlePlacer._attempt_single`'s docstring).
        normal_fields : dict, optional
            Maps location-flag name to a physical-(x,y,z)-component normal
            vector field, shape (3, nz, ny, nx). Only consulted for
            `spec.location == "membrane"`.
        ignore_masks : dict, optional
            Maps location-flag name to a boolean mask of voxels that
            don't count as a blocking overlap. Only consulted for
            `spec.location == "membrane"` -- expected to be the FULL
            bilayer thickness footprint (deliberately different from
            `location_masks["membrane"]`).

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

        ignore_mask = None
        if (
            spec.location == "membrane"
            and ignore_masks is not None
            and spec.location in ignore_masks
        ):
            ignore_mask = ignore_masks[spec.location]

        placed = 0
        for _ in range(spec.max_count):
            if self._occupied_fraction() >= self.density_cutoff:
                break
            if self._attempt_single(spec, mask, normal_field, ignore_mask) is None:
                continue
            placed += 1
        return placed

    def run(
        self,
        specs: list[ParticleSpec],
        location_masks: dict[str, torch.Tensor] | None = None,
        normal_fields: dict[str, torch.Tensor] | None = None,
        ignore_masks: dict[str, torch.Tensor] | None = None,
    ) -> list[PlacedInstance]:
        """
        Place every species in `specs`, in order, respecting the shared
        `density_cutoff` across all of them.

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
            self.place_species(spec, location_masks, normal_fields, ignore_masks)
        return self.placements[before:]
