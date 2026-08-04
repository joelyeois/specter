"""
Multi-species hard-sphere packing algorithms for specimen generation.

Two backends, sharing the "draw a candidate pool, then place it" pattern:

- :func:`pack_hard_spheres_3d` -- Random Sequential Addition (RSA), fast
  (sub-second to a few seconds at a few thousand instances), capped by RSA's
  known jamming limit (~28-41% depending on species-size diversity, well
  below random-close-packing).
- :func:`pack_hard_spheres_3d_dense` -- force-biased relaxation in a padded,
  periodic box, cropped back down to the requested (walled) box. Slower
  (a few minutes at a few thousand instances) and typically denser than
  `pack_hard_spheres_3d` at the same request, though by how much is
  parameter-dependent -- see that function's own docstring for verified
  numbers and an important caveat about `occupancy_fraction` requests near
  the physical jamming ceiling.

Both were benchmarked in ``dev/packing_algorithms.py`` against several other
candidates (naive/voxel-grid RSA, Lubachevsky-Stillinger, a CellPACK-style
incremental distance field, an SDF-scored variant, and Tetris-style
contact-correlation packing) -- see that file for the full comparison; only
the two winners (fast-but-looser vs. slow-but-denser) are promoted here.
`pack_hard_spheres_3d` itself began as ``pack_rsa_batched`` there.

The force-biased core is a torch re-implementation of the update rule in
Vasili Baranov's PackingGeneration (C++, MIT license):
https://github.com/VasiliBaranov/packing-generation -- specifically
``BezrukovJodreyToryStep.cpp`` (Bezrukov, Bargiel & Stoyan 2002). It is
periodic-only (no walled-box variant): the reference implementation only
ever packs a periodic unit cell, and its reported densities (comfortably
beating RSA, approaching random-close-packing) are periodic-cell numbers --
relaxing directly against hard walls was tried first and plateaus around
15-20% regardless of iteration budget (particles driven toward a wall have
nowhere to go and jam there), well *below* RSA. :func:`pack_hard_spheres_3d_dense`
recovers walled-box behavior by relaxing in a padded periodic box and
cropping back to the requested box afterward -- see its own docstring.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import torch
import torch.nn.functional as F

# Matches FORCE_SCALING_FACTOR in BezrukovJodreyToryStep.cpp:
# https://github.com/VasiliBaranov/packing-generation/blob/master/PackingGeneration/Generation/PackingGenerators/Source/BezrukovJodreyToryStep.cpp
_FORCE_SCALING_FACTOR = 0.5


def draw_species_pool(
    species_radii: torch.Tensor,
    species_ratios: torch.Tensor,
    occupancy_fraction: float,
    box_volume: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a candidate instance pool (one row per sphere to attempt placing)
    whose combined bare-sphere volume reaches ``occupancy_fraction *
    box_volume``, species drawn with probability proportional to
    `species_ratios` (matching how
    `specter.specimen.cytosolic_filler.PEI2016_CROWDING_TABLE`'s
    `occurrence_freq` is meant to be used as a relative-abundance weight,
    not an absolute one).

    Parameters
    ----------
    species_radii : torch.Tensor, shape (S,)
        Real physical radius of each species, Angstrom.
    species_ratios : torch.Tensor, shape (S,)
        Relative abundance weight per species. Only ratios between entries
        matter, not the absolute values.
    occupancy_fraction : float
        Target combined bare-sphere volume, as a fraction of `box_volume`.
    box_volume : float
        Angstrom^3.
    seed : int, optional
        Random seed.

    Returns
    -------
    radii : torch.Tensor, shape (N,)
        Sorted largest-radius-first (matches `pack_hard_spheres_3d`'s own
        largest-first staging, though it re-groups by radius internally
        regardless of input order).
    species_idx : torch.Tensor, shape (N,)
        Index into `species_radii`/`species_ratios` for each drawn instance.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    target_volume = occupancy_fraction * box_volume
    probs = species_ratios / species_ratios.sum()

    radii_list: list[float] = []
    species_list: list[int] = []
    accumulated = 0.0
    # Defensive cap: a candidate pool this large is already absurd for this
    # use case (occupancy_fraction close to 1 with tiny radii, etc.).
    for _ in range(1_000_000):
        if accumulated >= target_volume:
            break
        k = int(torch.multinomial(probs, 1, generator=gen))
        r = float(species_radii[k])
        radii_list.append(r)
        species_list.append(k)
        accumulated += (4.0 / 3.0) * torch.pi * r**3

    radii = torch.tensor(radii_list)
    species_idx = torch.tensor(species_list, dtype=torch.long)
    if radii.numel() > 0:
        perm = torch.argsort(radii, descending=True)
        radii, species_idx = radii[perm], species_idx[perm]
    return radii, species_idx


def pack_hard_spheres_3d(
    radii: torch.Tensor,
    box: tuple[float, float, float],
    gap: float = 0.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    max_passes: int = 200,
    stall_patience: int = 15,
    clip_axes: tuple[bool, bool, bool] = (False, False, False),
    exclusion_distance_field: torch.Tensor | None = None,
    field_v_size: float | None = None,
    sampling_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack hard spheres of given (possibly different) radii into a box via
    Random Sequential Addition (RSA), fully vectorized across candidates.

    Places spheres largest-radius-first, in STAGES (one stage per unique
    radius in `radii`, carrying already-accepted spheres forward across
    stages) -- biggest species get first crack at the still-mostly-empty
    box, matching how a naive one-at-a-time RSA would behave, rather than
    mixing all sizes into one pool (which measurably starves large spheres:
    they have more potential conflict partners than small ones at any given
    density, so they lose more often once smaller spheres are already
    competing for the same space). Within a stage, every remaining
    candidate gets ONE trial position per "pass": positions are generated
    and conflict-checked simultaneously via
    `vesin_torch.NeighborList <https://github.com/Luthaf/vesin>`_ (already
    a project dependency -- see :mod:`specter.ice._energy`'s ML-BOP energy
    for the same pattern: a neighbor list that stays entirely in ``torch``,
    so GPU-resident positions never round-trip to the host), with
    conflicting candidates from the same pass resolved via a one-shot
    "local minimum priority wins" parallel independent-set selection
    (`torch.scatter_reduce_(reduce="amin")`) rather than a Python loop over
    pairs. This is what actually matters for speed: swapping the collision
    check's underlying data structure alone (dict/grid vs. a rasterized
    voxel-occupancy array, both tried) gave a much smaller win than
    resolving many candidates' accept/reject decisions per pass instead of
    one at a time -- roughly 90x faster than a naive one-candidate-at-a-time
    RSA loop at a few thousand spheres, with ~99% as many accepted (a small,
    tunable completeness/speed tradeoff via `max_passes`/`stall_patience`).

    Non-periodic: by default, candidates are rejected outright if they
    would extend past any box wall -- appropriate for a real specimen
    volume (bounded, not a periodic unit cell). `clip_axes` relaxes this
    per axis (see that parameter).

    Parameters
    ----------
    radii : torch.Tensor, shape (N,)
        Requested sphere radius per candidate instance, Angstrom. Order is
        irrelevant (spheres are internally grouped and processed
        largest-first regardless of input order).
    box : tuple of float
        (D, H, W) box extents in Angstrom (z, y, x), centered at the
        origin -- same convention as
        :func:`specter.crowding.poisson_disk_neighbors_3d`.
    gap : float, optional
        Extra clearance between sphere surfaces, Angstrom, beyond simple
        touching. Default 0.0.
    seed : int, optional
        Random seed.
    device : str or torch.device, optional
        Device to run the (fully vectorized, GPU-capable) packing on.
        Default "cpu" -- in practice CPU has outperformed GPU here at every
        scale tested so far (GPU kernel-launch/neighbor-list-construction
        overhead dominates at realistic particle counts); only pass "cuda"
        after confirming it actually helps for your problem size.
    max_passes : int, optional
        Maximum passes per radius stage. Default 200.
    stall_patience : int, optional
        Give up on a stage (move to the next-smaller radius) after this
        many consecutive passes place zero new spheres -- the box is
        saturated for that radius. Default 15.
    clip_axes : tuple of bool, optional
        (z, y, x), matching `box`'s own axis order. True on an axis means a
        candidate's CENTER just needs to stay within the box on that axis --
        the sphere itself may extend past that wall (it gets truncated
        naturally when later rendered via `insert_particles_into_micrograph`,
        which already clips at volume boundaries), rather than being
        rejected outright. False (default, all axes) requires the full
        sphere to fit within the box on that axis, the original behavior.
        Useful e.g. for a tomogram whose xy field of view is understood to
        be a crop of a larger cellular region (so edge particles there are
        fine to be truncated) but whose z extent is a real specimen-
        thickness boundary particles should not cross.
    exclusion_distance_field : torch.Tensor, optional
        Physical distance (Angstrom) to the nearest FORBIDDEN voxel, shape
        ``(Z, Y, X)``, on a box-centered grid at `field_v_size` spacing
        (same centering convention `.membrane._raster.rasterize_membrane_
        density` uses by default -- e.g. build this via ``scipy.ndimage.
        distance_transform_edt`` on the complement of a boolean "forbidden"
        mask). A candidate is rejected if the field, trilinearly sampled at
        its center, is less than `radius + gap` -- i.e. its full sphere
        (plus gap) must clear the forbidden region. This single mechanism
        covers two distinct use cases with one field: hard obstacle
        avoidance (forbidden = e.g. a membrane's own occupied shell) and
        region restriction (forbidden = the complement of wherever this
        species is allowed, e.g. everywhere outside a vesicle's lumen for a
        lumen-only species) -- a caller wanting both combines them into one
        mask (union) before taking the distance transform. Default None
        (no exclusion). Points outside the field's own extent sample the
        nearest boundary value (clamped, not wrapped).
    field_v_size : float, optional
        Voxel size of `exclusion_distance_field`/`sampling_mask`, Angstrom
        (the two share one grid). Required if either is given. Trilinear
        interpolation of a coarse `exclusion_distance_field` lets a
        candidate's sampled distance run a little past the true
        (exact-voxel) value near a boundary -- confirmed a couple of
        Angstrom of bleed at `field_v_size=5` for `gap=2`, vanishing by
        `field_v_size=2`; keep this fine relative to `gap` (or pad the
        forbidden mask by a voxel or two before taking its distance
        transform) if a hard guarantee matters more than exact `gap`
        fidelity at the boundary.
    sampling_mask : torch.Tensor, optional
        Boolean mask, shape ``(Z, Y, X)``, same grid as
        `exclusion_distance_field`. When given, candidates are drawn from a
        uniformly random True voxel of this mask (plus sub-voxel jitter)
        instead of uniformly across the whole `box`. This matters --
        distinct from what `exclusion_distance_field` alone already gets
        you -- whenever the geometrically valid region for a candidate's
        CENTER is a small fraction of `box`'s own volume: uniform box-wide
        sampling then has to blindly get lucky before rejection filtering
        can even engage, e.g. confirmed a real case (packing into a small
        vesicle lumen, valid region 0.008% of the box) placing 0 of 1
        candidates within `max_passes`, purely from having astronomically
        low odds of a hit at all, despite an exactly-computed valid region
        genuinely existing. Restricting sampling to the allowed region (its
        looser, unfiltered form -- e.g. the same region a corresponding
        `exclusion_distance_field` was built to keep candidates fully
        inside) fixes this by construction, the same technique
        `specimen._cts_placement.ParticlePlacer` already uses via
        `torch.nonzero(location_mask)`. Default None (uniform box-wide
        sampling, the original behavior).

    Returns
    -------
    coords : torch.Tensor, shape (M, 3)
        Accepted sphere centers (x, y, z), Angstrom, box-centered. M <= N.
    accepted_idx : torch.Tensor, shape (M,)
        Indices into `radii` for the spheres that were successfully
        placed, in the order accepted (not input order) -- use this to
        look up per-instance metadata (e.g. species) from arrays aligned
        with `radii`.
    """
    import vesin_torch

    has_field_consumer = (
        exclusion_distance_field is not None or sampling_mask is not None
    )
    if has_field_consumer != (field_v_size is not None):
        raise ValueError(
            "field_v_size is required together with (and only with) "
            "exclusion_distance_field and/or sampling_mask"
        )
    if (
        exclusion_distance_field is not None
        and sampling_mask is not None
        and exclusion_distance_field.shape != sampling_mask.shape
    ):
        raise ValueError(
            "exclusion_distance_field and sampling_mask must have the same shape"
        )

    mask_voxel_positions_xyz: torch.Tensor | None = None
    if sampling_mask is not None:
        nz, ny, nx = sampling_mask.shape
        voxel_idx_zyx = sampling_mask.nonzero(as_tuple=False)
        if voxel_idx_zyx.shape[0] == 0:
            raise ValueError("sampling_mask has no True voxels")
        extent = (
            torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
            * field_v_size
        )
        origin = -0.5 * extent
        idx_xyz = voxel_idx_zyx[:, [2, 1, 0]].to(device=device, dtype=torch.float32)
        mask_voxel_positions_xyz = origin + (idx_xyz + 0.5) * field_v_size

    def _half_extents_xyz(box: tuple[float, float, float]) -> torch.Tensor:
        D, H, W = box
        return torch.tensor([W / 2, H / 2, D / 2], device=device)

    def _box_dims_xyz(box: tuple[float, float, float]) -> torch.Tensor:
        D, H, W = box
        return torch.tensor([W, H, D], device=device)

    def _sample_exclusion_distance(points_xyz: torch.Tensor) -> torch.Tensor:
        """Trilinearly sample `exclusion_distance_field` at physical (x, y,
        z) points on its own box-centered grid -- same normalized-grid
        convention as `MembraneField._normalized_grid`, kept independent
        (not imported) so `specimen/packing` stays uncoupled from
        `specimen/membrane`."""
        assert exclusion_distance_field is not None and field_v_size is not None
        nz, ny, nx = exclusion_distance_field.shape
        extent = (
            torch.tensor([nx, ny, nz], dtype=points_xyz.dtype, device=device)
            * field_v_size
        )
        origin = -0.5 * extent
        idx = (points_xyz - origin) / field_v_size
        norm = (
            2.0
            * (idx + 0.5)
            / torch.tensor([nx, ny, nz], dtype=points_xyz.dtype, device=device)
            - 1.0
        )
        grid = norm.view(1, -1, 1, 1, 3)
        volume = exclusion_distance_field[None, None].to(
            device=device, dtype=points_xyz.dtype
        )
        sampled = F.grid_sample(
            volume, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(points_xyz.shape[0])

    def _run_stage(
        row_idx: torch.Tensor,
        accepted_pos: torch.Tensor,
        accepted_radii: torch.Tensor,
        accepted_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        remaining = row_idx
        stall = 0
        for _ in range(max_passes):
            if remaining.numel() == 0:
                break
            cand_radii = radii[remaining]
            if mask_voxel_positions_xyz is not None:
                assert field_v_size is not None
                chosen = torch.randint(
                    0,
                    mask_voxel_positions_xyz.shape[0],
                    (remaining.numel(),),
                    generator=gen,
                    device=device,
                )
                jitter = (
                    torch.rand(remaining.numel(), 3, generator=gen, device=device) - 0.5
                ) * field_v_size
                cand = mask_voxel_positions_xyz[chosen] + jitter
            else:
                cand = (
                    (
                        torch.rand(remaining.numel(), 3, generator=gen, device=device)
                        - 0.5
                    )
                    * 2
                    * half
                )
            # full sphere must fit on non-clippable axes; only the center
            # needs to stay in-bounds on clippable ones (see `clip_axes`).
            required_margin = torch.where(
                clip_allowed_xyz.unsqueeze(0), 0.0, cand_radii.unsqueeze(1)
            )
            blocked = ((cand.abs() + required_margin) > half.unsqueeze(0)).any(dim=1)

            if exclusion_distance_field is not None:
                dist = _sample_exclusion_distance(cand)
                blocked = blocked | (dist < cand_radii + gap)

            # candidates vs. already-accepted spheres (all prior stages
            # plus anything already accepted this stage)
            if accepted_pos.numel() > 0:
                free = (~blocked).nonzero(as_tuple=True)[0]
                if free.numel() > 0:
                    combined = torch.cat([cand[free], accepted_pos], dim=0)
                    nl = vesin_torch.NeighborList(cutoff=cutoff, full_list=False)
                    i_idx, j_idx, d = nl.compute(
                        combined, box_t, periodic=False, quantities="ijd"
                    )
                    # candidates are [0, free.numel()); accepted spheres
                    # come after. The neighbor list also returns plain
                    # candidate-vs-candidate pairs (both indices <
                    # free.numel()) -- excluded here, that's the next
                    # block's job.
                    cross = (i_idx < free.numel()) & (j_idx >= free.numel())
                    ci, aj, cd = i_idx[cross], j_idx[cross] - free.numel(), d[cross]
                    contact = cand_radii[free][ci] + accepted_radii[aj] + gap
                    conflict = cd < contact
                    if conflict.any():
                        blocked[free[ci[conflict].unique()]] = True

            # candidates vs. each other, one-shot parallel independent set
            # (within a stage all candidates share the same radius, so a
            # plain random tie-break is fair)
            active = (~blocked).nonzero(as_tuple=True)[0]
            if active.numel() > 1:
                sub_pos = cand[active]
                sub_radii = cand_radii[active]
                nl2 = vesin_torch.NeighborList(cutoff=cutoff, full_list=False)
                i2, j2, d2 = nl2.compute(
                    sub_pos, box_t, periodic=False, quantities="ijd"
                )
                contact2 = sub_radii[i2] + sub_radii[j2] + gap
                conflict2 = d2 < contact2
                i2c, j2c = i2[conflict2], j2[conflict2]
                if i2c.numel() > 0:
                    priority = torch.rand(active.numel(), generator=gen, device=device)
                    min_neighbor = torch.full(
                        (active.numel(),), float("inf"), device=device
                    )
                    min_neighbor.scatter_reduce_(0, i2c, priority[j2c], reduce="amin")
                    min_neighbor.scatter_reduce_(0, j2c, priority[i2c], reduce="amin")
                    survive = priority < min_neighbor
                else:
                    survive = torch.ones(
                        active.numel(), dtype=torch.bool, device=device
                    )
                newly_accepted = active[survive]
            else:
                newly_accepted = active

            if newly_accepted.numel() > 0:
                accepted_pos = torch.cat([accepted_pos, cand[newly_accepted]], dim=0)
                accepted_radii = torch.cat(
                    [accepted_radii, cand_radii[newly_accepted]], dim=0
                )
                accepted_row = torch.cat(
                    [accepted_row, remaining[newly_accepted]], dim=0
                )
                keep = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
                keep[newly_accepted] = False
                remaining = remaining[keep]
                stall = 0
            else:
                stall += 1
                if stall >= stall_patience:
                    break

        return accepted_pos, accepted_radii, accepted_row

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    half = _half_extents_xyz(box)
    box_t = torch.diag(_box_dims_xyz(box))
    N = radii.shape[0]
    radii = radii.to(device)
    # clip_axes is (z, y, x) matching box; reorder to (x, y, z) to match
    # `half`/candidate position columns.
    clip_allowed_xyz = torch.tensor(
        [clip_axes[2], clip_axes[1], clip_axes[0]], dtype=torch.bool, device=device
    )

    accepted_pos = torch.empty((0, 3), device=device)
    accepted_radii = torch.empty((0,), device=device)
    accepted_row = torch.empty((0,), dtype=torch.long, device=device)

    if N == 0:
        return accepted_pos.cpu(), accepted_row.cpu()

    cutoff = 2 * float(radii.max()) + gap
    for r_val in radii.unique(sorted=True).flip(0):  # largest first
        row_idx = (radii == r_val).nonzero(as_tuple=True)[0]
        accepted_pos, accepted_radii, accepted_row = _run_stage(
            row_idx, accepted_pos, accepted_radii, accepted_row
        )

    return accepted_pos.cpu(), accepted_row.cpu()


# ---------------------------------------------------------------------------
# Force-biased periodic relaxation (dense backend) -- private helpers
# ---------------------------------------------------------------------------
#
# Periodic-only: the only caller (`pack_hard_spheres_3d_dense`) always
# relaxes inside a padded periodic box, never against real walls, so the
# walled-box branches present in the dev/ comparison version are dropped
# here rather than carried along unused.


def _fb_half_extents_xyz(
    box: tuple[float, float, float], device: str | torch.device = "cpu"
) -> torch.Tensor:
    D, H, W = box
    return torch.tensor([W / 2, H / 2, D / 2], device=device)


def _fb_box_dims_xyz(
    box: tuple[float, float, float], device: str | torch.device = "cpu"
) -> torch.Tensor:
    D, H, W = box
    return torch.tensor([W, H, D], device=device)


def _fb_wrap_positions(
    positions: torch.Tensor, box: tuple[float, float, float]
) -> torch.Tensor:
    """Wrap positions into the periodic box, coordinates in [-half, half)."""
    dims = _fb_box_dims_xyz(box, device=positions.device)
    return (positions + dims / 2) % dims - dims / 2


def _fb_periodic_diff(
    diff: torch.Tensor, box: tuple[float, float, float]
) -> torch.Tensor:
    """Minimum-image displacement (broadcasts over any leading shape, last dim=3)."""
    dims = _fb_box_dims_xyz(box, device=diff.device)
    return diff - dims * torch.round(diff / dims)


def _fb_grid_shape(
    box: tuple[float, float, float], cell_size: float
) -> tuple[int, int, int]:
    D, H, W = box
    return (
        max(1, int(W / cell_size)),
        max(1, int(H / cell_size)),
        max(1, int(D / cell_size)),
    )


def _fb_cell_index(
    pos: torch.Tensor,
    cell_size: float,
    half: torch.Tensor,
    grid_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    idx = torch.floor((pos + half) / cell_size).long()
    gx, gy, gz = grid_shape
    return (int(idx[0]) % gx, int(idx[1]) % gy, int(idx[2]) % gz)


def _fb_neighbor_cells(cell: tuple[int, int, int], grid_shape: tuple[int, int, int]):
    cx, cy, cz = cell
    gx, gy, gz = grid_shape
    seen: set[tuple[int, int, int]] = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nb = ((cx + dx) % gx, (cy + dy) % gy, (cz + dz) % gz)
                if nb not in seen:
                    seen.add(nb)
                    yield nb


def _fb_find_overlapping_pairs(
    positions: torch.Tensor,
    radii: torch.Tensor,
    gap: float,
    cell_size: float,
    half: torch.Tensor,
    box: tuple[float, float, float],
    overlap_tol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grid-accelerated overlap query (minimum-image, periodic). Candidate
    pairs are found via a spatial hash (python dict, cell size = touch-
    distance of the two largest radii so any overlap is confined to the
    3x3x3 wrap-around neighborhood); distance/overlap for the found
    candidates is then computed vectorized. Returns (i_idx, j_idx, overlap,
    diffs, dist) for pairs with overlap > `overlap_tol`, i_idx < j_idx (each
    unordered pair reported exactly once). `overlap_tol` matters a lot in
    practice: `_fb_relax_overlaps`'s pairwise displacement converges
    asymptotically (like gradient descent near an optimum), so a strict
    `overlap_tol=0.0` (the default, appropriate for driving the relaxation
    itself) will keep reporting hundreds of physically negligible
    sub-Angstrom-fraction "overlaps" as real ones long after they stop
    mattering -- callers deciding whether a packing is "done"
    (`_fb_drop_worst_overlaps`, `_fb_verify_no_overlap`) should pass a small
    positive tolerance instead, the same way polnet's own placement config
    has PMER_OVER_TOL/MB_OVER_TOL rather than demanding exact non-overlap.
    """
    n = positions.shape[0]
    if n == 0:
        z = torch.empty(0, dtype=torch.long)
        return z, z, torch.empty(0), torch.empty((0, 3)), torch.empty(0)

    grid_shape = _fb_grid_shape(box, cell_size)
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        grid[_fb_cell_index(positions[i], cell_size, half, grid_shape)].append(i)

    pairs_i: list[int] = []
    pairs_j: list[int] = []
    for cell, idxs in grid.items():
        neighbor_idxs: list[int] = []
        for nb in _fb_neighbor_cells(cell, grid_shape):
            neighbor_idxs.extend(grid.get(nb, []))
        for i in idxs:
            for j in neighbor_idxs:
                if j > i:
                    pairs_i.append(i)
                    pairs_j.append(j)

    if not pairs_i:
        z = torch.empty(0, dtype=torch.long)
        return z, z, torch.empty(0), torch.empty((0, 3)), torch.empty(0)

    i_idx = torch.tensor(pairs_i, dtype=torch.long)
    j_idx = torch.tensor(pairs_j, dtype=torch.long)
    diffs = positions[i_idx] - positions[j_idx]
    diffs = _fb_periodic_diff(diffs, box)
    dist = diffs.norm(dim=1)
    contact = radii[i_idx] + radii[j_idx] + gap
    overlap = contact - dist
    mask = overlap > overlap_tol
    return i_idx[mask], j_idx[mask], overlap[mask], diffs[mask], dist[mask]


def _fb_verify_no_overlap(
    positions: torch.Tensor,
    radii: torch.Tensor,
    gap: float,
    box: tuple[float, float, float] | None = None,
    overlap_tol: float | None = None,
) -> bool:
    """Brute-force O(n^2) verification (only meant to be run once, after
    packing) that no two accepted spheres overlap by more than
    `overlap_tol` Angstrom -- a small positive default, not exact zero; see
    `_fb_find_overlapping_pairs`'s docstring for why (`_fb_relax_overlaps`
    converges asymptotically, so "exactly zero" is an unreasonable bar).
    `overlap_tol=None` (default) uses `1e-3 * radii.mean()`, matching the
    packing algorithm's own `tol_frac` convention -- scaling with radius
    rather than a fixed Angstrom value keeps the bar equally strict (0.1%
    of radius) whether species are 15 A or 60 A. `box` is only used if
    `periodic` minimum-image distance is needed by the caller -- pass None
    for a plain (non-periodic) cropped set, matching how
    `pack_hard_spheres_3d_dense` verifies its final, walled-box result."""
    n = positions.shape[0]
    if n < 2:
        return True
    if overlap_tol is None:
        overlap_tol = 1e-3 * float(radii.mean())
    diffs = positions.unsqueeze(0) - positions.unsqueeze(1)
    if box is not None:
        diffs = _fb_periodic_diff(diffs, box)
    dist = diffs.norm(dim=-1)
    contact = radii.unsqueeze(0) + radii.unsqueeze(1) + gap
    overlap = contact - dist
    overlap.fill_diagonal_(-float("inf"))
    return bool((overlap <= overlap_tol).all())


def _fb_relax_overlaps(
    positions: torch.Tensor,
    radii: torch.Tensor,
    gap: float,
    half: torch.Tensor,
    box: tuple[float, float, float],
    max_iterations: int,
    force_scale: float = _FORCE_SCALING_FACTOR,
    tol_frac: float = 1e-3,
) -> tuple[torch.Tensor, int, bool]:
    """Force-biased overlap relaxation in a periodic box -- the core
    mechanism of `pack_hard_spheres_3d_dense`. Returns (positions,
    n_iterations_run, converged)."""
    if positions.shape[0] == 0:
        return positions, 0, True
    cell_size = 2 * float(radii.max()) + gap
    it = 0
    for it in range(max_iterations):
        i_idx, j_idx, overlap, diffs, dist = _fb_find_overlapping_pairs(
            positions, radii, gap, cell_size, half, box
        )
        if i_idx.numel() == 0 or overlap.max() < tol_frac * radii.mean():
            return positions, it, True
        dist_safe = dist.clamp_min(1e-6)
        unit = diffs / dist_safe.unsqueeze(1)
        mass_i = radii[i_idx] ** 3
        mass_j = radii[j_idx] ** 3
        frac_i = (mass_j / (mass_i + mass_j)).unsqueeze(1)
        frac_j = (mass_i / (mass_i + mass_j)).unsqueeze(1)
        disp = torch.zeros_like(positions)
        disp.index_add_(0, i_idx, unit * overlap.unsqueeze(1) * force_scale * frac_i)
        disp.index_add_(0, j_idx, -unit * overlap.unsqueeze(1) * force_scale * frac_j)
        positions = _fb_wrap_positions(positions + disp, box)
    i_idx, *_ = _fb_find_overlapping_pairs(positions, radii, gap, cell_size, half, box)
    return positions, it + 1, i_idx.numel() == 0


def _fb_drop_worst_overlaps(
    positions: torch.Tensor,
    radii: torch.Tensor,
    species_idx: torch.Tensor,
    gap: float,
    half: torch.Tensor,
    box: tuple[float, float, float],
    max_drops: int = 10_000,
    overlap_tol: float = 1e-3,
    relax_between_drops: int = 20,
    force_scale: float = _FORCE_SCALING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cleanup pass: repeatedly drop the more-overlapping particle of the
    worst remaining overlapping pair (by more than `overlap_tol` -- NOT
    exact zero, see `_fb_find_overlapping_pairs`) until none remain,
    re-running a short `_fb_relax_overlaps` burst (`relax_between_drops`
    iterations) every few drops so the freed-up space actually gets used by
    the survivors instead of them sitting frozen in their pre-drop
    positions. Without that, `_fb_relax_overlaps`'s own convergence check (a
    single global max-overlap threshold) routinely leaves hundreds of pairs
    with a physically negligible but nonzero residual overlap each, and a
    zero-tolerance drop loop with no further relaxation would delete most
    of them for essentially no reason.
    """
    keep = torch.ones(positions.shape[0], dtype=torch.bool)
    n_since_relax = 0
    for _ in range(max_drops):
        active = keep.nonzero(as_tuple=True)[0]
        if active.numel() < 2:
            break
        pos = positions[active]
        rad = radii[active]
        cell_size = 2 * float(rad.max()) + gap
        i_idx, j_idx, overlap, *_ = _fb_find_overlapping_pairs(
            pos, rad, gap, cell_size, half, box, overlap_tol=overlap_tol
        )
        if i_idx.numel() == 0:
            break
        worst = int(overlap.argmax())
        keep[active[int(j_idx[worst])]] = False
        n_since_relax += 1
        if n_since_relax >= relax_between_drops:
            active = keep.nonzero(as_tuple=True)[0]
            pos, _, _ = _fb_relax_overlaps(
                positions[active],
                radii[active],
                gap,
                half,
                box,
                relax_between_drops,
                force_scale,
            )
            positions = positions.clone()
            positions[active] = pos
            n_since_relax = 0
    return positions[keep], radii[keep], species_idx[keep]


def pack_hard_spheres_3d_dense(
    species_radii: torch.Tensor,
    species_ratios: torch.Tensor,
    occupancy_fraction: float,
    box: tuple[float, float, float],
    gap: float = 0.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    pad_fraction: float = 0.5,
    n_stages: int = 20,
    iterations_per_stage: int = 100,
    initial_scale: float = 0.3,
    force_scale: float = _FORCE_SCALING_FACTOR,
    tol_frac: float = 1e-3,
    clip_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack hard spheres of given (possibly different) species densely into a
    box via force-biased relaxation, using the "pad, relax periodically,
    crop" technique to recover walled-box behavior from an algorithm that
    only works well periodically (see the module docstring).

    A candidate pool is drawn (species-ratio-weighted, largest-first, via
    `draw_species_pool`) sized to reach `occupancy_fraction` of a PADDED
    box (`box` grown by `pad_fraction` on each side), started at uniformly
    random positions and relaxed with periodic boundaries via a staged
    homotopy: sphere radii are scaled from `initial_scale` up to 1.0 (real
    target) over `n_stages` stages, relaxing `iterations_per_stage` times
    at each (temporarily shrunk) stage before growing further -- this
    mirrors the reference's own approach of relaxing at an artificially
    large nominal diameter and shrinking it down, letting particles
    rearrange with breathing room before the real constraint tightens.
    Residual overlaps after all stages are cleaned up by repeatedly
    dropping the more-overlapping particle of the worst remaining pair, so
    the result is always overlap-free (fewer instances may be *dropped*
    than requested, unlike `pack_hard_spheres_3d` which only ever *fails to
    attempt* excess candidates).

    Only spheres landing fully inside the ORIGINAL (unpadded) box are kept
    on non-clippable axes (see `clip_axes`) -- this is what makes the
    periodic result usable as a real, walled specimen volume: a sphere
    whose center is near the padded box's own periodic wrap boundary can
    straddle it (needing its periodic image to look whole), which would
    render as an abrupt cut if used directly, but `pad_fraction`'s margin
    keeps the crop boundary well inside the relaxed, homogeneous interior,
    away from that wrap boundary. A non-overlapping periodic set's cropped
    subset is trivially also non-overlapping, so no re-verification is
    needed after cropping.

    Always overlap-free (verified via `_fb_verify_no_overlap`'s brute-force
    check whenever this is tested), and typically denser than
    `pack_hard_spheres_3d` at the same request -- but the SIZE of that
    advantage is sensitive to how close `occupancy_fraction` is pushed
    toward the physical jamming ceiling for the given species mix (random-
    close-packing for spheres, ~64% bare-sphere fraction). Near that
    ceiling, the staged homotopy may not fully converge within
    `iterations_per_stage` at every stage -- residual overlaps then get
    resolved by `_fb_drop_worst_overlaps`'s cleanup, which can end up
    dropping a substantial and not precisely predictable fraction of the
    padded pool, so achieved occupancy in that regime is "best effort," not
    a guarantee, and can vary between otherwise-identical environments
    (results ARE bit-reproducible for a fixed seed on one machine/build,
    confirmed by rerunning the relaxation trajectory and comparing
    checksums -- but summation order in the underlying `index_add_` calls
    is not guaranteed identical across different thread counts/BLAS
    builds, so don't expect the exact achieved occupancy to transfer
    across machines at this parameter regime). Comfortably below the
    ceiling (e.g. `occupancy_fraction` a few tenths, well clear of ~0.6+),
    convergence is reliable and this backend's density advantage over
    `pack_hard_spheres_3d` is consistent and easy to reproduce -- that is
    the regime this is intended for; treat requests near the ceiling as
    exploratory, not something to build a pipeline around without
    rechecking `achieved_occupancy` (compute it from the returned
    `radii_out`) on your own actual run. This backend costs on the order of
    a few minutes at a few thousand instances vs. `pack_hard_spheres_3d`'s
    sub-second-to-seconds, so it's the "I can afford to wait for real
    density" option, not the default.

    `occupancy_fraction` is honored accurately only when `box` is large,
    on its non-clippable axes, relative to the largest species diameter
    (true of any realistic specimen volume, and true of the regime this
    was benchmarked in above: a 600 A box against species up to 32 A
    radius). The crop step keeps only spheres landing fully inside the
    ORIGINAL box on those axes -- a sphere's own center must stay `radius`
    away from every non-clippable wall -- so the fraction of the
    (homogeneously relaxed) padded pool that survives cropping shrinks per
    axis as that species' radius approaches the box's own extent on that
    axis, compounding across all non-clippable axes. For a box whose
    smallest non-clippable extent is only a few sphere diameters across,
    this can make the actual achieved occupancy silently undershoot the
    request by several-fold (unlike `pack_hard_spheres_3d`, where a
    too-small box just triggers ordinary RSA rejections against
    `n_candidates`, visible and interpretable) -- a warning is raised in
    that regime; consider `pack_hard_spheres_3d` instead there, a much
    larger `pad_fraction`, or allowing clipping on the tight axis via
    `clip_axes`.

    Parameters
    ----------
    species_radii : torch.Tensor, shape (S,)
        Real physical radius of each species, Angstrom.
    species_ratios : torch.Tensor, shape (S,)
        Relative abundance weight per species (see `draw_species_pool`).
    occupancy_fraction : float
        Target bare-sphere volume fraction, relative to `box` (not the
        internally padded box).
    box : tuple of float
        (D, H, W) box extents in Angstrom (z, y, x), centered at the
        origin.
    gap : float, optional
        Extra clearance between sphere surfaces, Angstrom. Default 0.0.
    seed : int, optional
        Random seed.
    device : str or torch.device, optional
        Device for the relaxation. Default "cpu".
    pad_fraction : float, optional
        Each box dimension is grown by this fraction (e.g. 0.5 means a
        600 A box is relaxed inside a 900 A periodic box) before cropping
        back down. Larger values give a more homogeneous, edge-effect-free
        crop at the cost of relaxing more candidate spheres. Default 0.5.
    n_stages, iterations_per_stage, initial_scale, force_scale, tol_frac
        Force-biased relaxation homotopy parameters -- see the function
        body / module docstring. Defaults match the values benchmarked in
        `dev/packing_algorithms.py`.
    clip_axes : tuple of bool, optional
        (z, y, x), matching `box`'s own axis order. True on an axis means a
        sphere's CENTER just needs to stay within the original box on that
        axis to survive the crop -- the sphere itself may extend past that
        wall (truncated naturally at render time, matching
        `pack_hard_spheres_3d`'s own `clip_axes`). False (default, all
        axes) requires the full sphere to fit, the original behavior. See
        the note above on `occupancy_fraction` accuracy -- allowing
        clipping on a tight axis avoids that undershoot entirely for that
        axis, since spheres are no longer discarded for crossing it.

    Returns
    -------
    coords : torch.Tensor, shape (M, 3)
        Accepted sphere centers (x, y, z), Angstrom, box-centered.
    radii_out : torch.Tensor, shape (M,)
        Radius of each accepted sphere, Angstrom.
    species_idx_out : torch.Tensor, shape (M,)
        Index into `species_radii`/`species_ratios` for each accepted
        sphere.
    """
    non_clippable_extents = [box[i] for i in range(3) if not clip_axes[i]]
    if (
        non_clippable_extents
        and species_radii.numel() > 0
        and float(species_radii.max()) > 0.15 * min(non_clippable_extents)
    ):
        warnings.warn(
            "pack_hard_spheres_3d_dense: the largest species radius "
            f"({float(species_radii.max()):.1f} A) is large relative to the "
            f"non-clippable box extent ({min(non_clippable_extents):.1f} A) -- "
            "the crop step will disproportionately discard large-species "
            "instances, and occupancy_fraction may be substantially "
            "undershot. See this function's docstring; pack_hard_spheres_3d "
            "may be a better fit for this box/species-size regime, or allow "
            "clipping on the tight axis via clip_axes.",
            stacklevel=2,
        )

    half_target = _fb_half_extents_xyz(box, device=device)
    padded_box = (
        box[0] * (1.0 + pad_fraction),
        box[1] * (1.0 + pad_fraction),
        box[2] * (1.0 + pad_fraction),
    )
    padded_box_volume = padded_box[0] * padded_box[1] * padded_box[2]

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    radii, species_idx = draw_species_pool(
        species_radii, species_ratios, occupancy_fraction, padded_box_volume, seed=seed
    )
    n = radii.shape[0]
    empty = torch.empty((0, 3), device=device)
    if n == 0:
        return (
            empty,
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.long),
        )

    radii = radii.to(device)
    species_idx = species_idx.to(device)
    half_padded = _fb_half_extents_xyz(padded_box, device=device)

    positions = (torch.rand(n, 3, generator=gen, device=device) - 0.5) * 2 * half_padded

    scales = torch.linspace(initial_scale, 1.0, n_stages)
    for scale in scales:
        eff_radii = radii * float(scale)
        positions, _, _ = _fb_relax_overlaps(
            positions,
            eff_radii,
            gap,
            half_padded,
            padded_box,
            iterations_per_stage,
            force_scale,
            tol_frac,
        )

    overlap_tol = tol_frac * float(radii.mean())
    positions, radii, species_idx = _fb_drop_worst_overlaps(
        positions,
        radii,
        species_idx,
        gap,
        half_padded,
        padded_box,
        overlap_tol=overlap_tol,
        force_scale=force_scale,
    )

    # full sphere must fit on non-clippable axes; only the center needs to
    # stay in-bounds on clippable ones (see `clip_axes`).
    clip_allowed_xyz = torch.tensor(
        [clip_axes[2], clip_axes[1], clip_axes[0]], dtype=torch.bool, device=device
    )
    required_margin = torch.where(
        clip_allowed_xyz.unsqueeze(0), 0.0, radii.unsqueeze(1)
    )
    inside = ((positions.abs() + required_margin) <= half_target).all(dim=1)
    return positions[inside].cpu(), radii[inside].cpu(), species_idx[inside].cpu()
