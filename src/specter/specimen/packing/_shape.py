"""
Exact-shape ("voxel occupancy") packing.

This is what CryoTomoSim does. Its `helper_arrayinsert.m` `'overlaptest'`
binarizes the destination patch and the rotated source patch and rejects the
placement if any voxel carries both; no sphere approximation appears anywhere
in its placement path. Reproducing that here is what closes the packing-
density gap against it: on a matched 10-species benchmark, colliding real
rotated shapes instead of bounding spheres moves the achievable macromolecule
volume fraction from 0.043 to ~0.15 (204 mg/mL, physiological protein
concentration), with `..algorithms.pack_hard_spheres_3d`'s own geometry
unchanged in every other respect.

The cost is not where it looks. Profiled per candidate at 6.8 A, rotating a
template costs 3.27 ms and testing it against the occupancy grid costs
0.08 ms -- 98% rotation, 2% collision. Rotating once per *attempt* is
therefore what makes a naive implementation slow, not the voxel test. This
module rotates each species' mask into a fixed cache of `n_orientations`
orientations once per stage and indexes that cache per attempt, so the
per-attempt cost is the 0.08 ms test alone. The cache is built and discarded
one species at a time (largest first, matching `pack_hard_spheres_3d`'s own
staging), which keeps it to tens of MB rather than the several GB that
caching every species at once would need.

Obstacles and region restriction both enter through the same `occupancy`
grid: seed it True wherever an instance may not go. That subsumes what
`pack_hard_spheres_3d` needs an `exclusion_distance_field` for -- membranes,
filaments and carbon are already rendered as volumes, so they are stamped
directly rather than distance-transformed.
"""

from __future__ import annotations

import numpy as np
import torch

import torch.nn.functional as F

from ...rotations._random import random_rotation_matrix
from ...rotations._volume import build_affine_matrix, rotate_volume

# Above this allowed-fraction, draw centers uniformly from the whole grid
# and let the occupancy test reject the rest, instead of enumerating the
# allowed voxels into a multi-GB index array. See its use below.
_DENSE_REGION_FRACTION = 0.30

# Footprint voxels sampled per orientation for the loop's cheap pre-test.
# See `_rotation_cache`'s `probes`.
_N_PROBE = 32

# Second-stage probe, applied only to the attempts the first probe lets
# through: a few hundred footprint voxels gathered in one vectorised pass
# cost far less than the full footprint AND (~25 us) those attempts would
# otherwise all reach. Like the first probe it is exact: an occupied probe
# voxel is a real clash, and a miss falls through to the full test.
_N_PROBE2 = 384

# Draws per batch. Random draws are batched rather than taken one scalar at a
# time, and the batch is also the unit over which every occupancy-independent
# test, and the probes against the grid as it stands, are evaluated at once.
_BATCH = 8192

# Rotation-cache chunking budget, in template voxels per batch. Caps the
# transient (batch, D, H, W) float32 `rotate_volume` allocates; see its use.
_ROTATE_CHUNK_VOXELS = 64_000_000


def build_species_mask(
    coordinates: torch.Tensor,
    voxel_size: float,
    gap: float = 0.0,
    atom_radius: float = 1.9,
) -> torch.Tensor:
    """
    Rasterize a molecule's atoms into a binary occupancy mask.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates, shape (N, 3), in the molecule's own frame.
        `specter.pdb.PDB.coordinates` is already centered on the atom
        centroid -- the same origin `PotentialBuilder` builds its template
        about -- so pass it unmodified.
    voxel_size : float
        A per voxel. Use the same value the volume will be rendered at.
    gap : float, optional
        Extra clearance baked into the mask, A, so two placed instances end
        up at least this far apart. Default 0.0.

        **Quantized to `voxel_size`**, and there is no way around that: the
        mask lives on the voxel grid, so it can only grow in whole voxels.
        The effective clearance is ``round_up((atom_radius + gap) /
        voxel_size)`` voxels. At 6.8 A that makes gap=0 and gap=2
        *identical* (both under one voxel), while gap=5 costs a full 6.8 A
        shell in every direction -- measured on the 121-species CRYOETSIM
        filler set, gap=0 and gap=2 both reach volume fraction 0.197 and
        gap=5 drops to 0.138. Check what your gap actually rounds to before
        attributing a density difference to it.

        CryoTomoSim itself uses no gap at all: its overlap test is pure
        contact.

        Must be >= 0. Collision here is strictly hard: two placed instances
        never share a voxel. CryoTomoSim's own Otsu overlap test is softer
        (only each particle's core blocks a placement, so sparse outer
        shells interpenetrate), which is deliberately not reproduced --
        matching it was measured to need only ~2% overlap tolerance, and
        buying that density costs a single owner per voxel in the instance
        labels, which are picker ground truth.
    atom_radius : float, optional
        Van der Waals radius assigned to each atom, A. Default 1.9,
        matching CryoTomoSim's own average-organic-atom radius.

    Returns
    -------
    mask : torch.Tensor
        Boolean mask, shape (Z, Y, X), centered on the molecule origin.
        Odd-sized on every axis so the origin sits at the center voxel.
    """
    from scipy import ndimage

    coords = coordinates.detach().cpu().numpy().astype(np.float64)
    pad = atom_radius + gap
    # Size the box for the atoms plus the outward growth, never less: a
    # nonsensical negative pad must still not clip atoms near the edge. The
    # +voxel_size keeps a background voxel of margin on every side, which
    # the distance transform below needs something to measure against.
    half_extent = np.abs(coords).max(axis=0) + max(pad, 0.0) + voxel_size
    n_half = np.ceil(half_extent / voxel_size).astype(int)
    shape = 2 * n_half + 1  # odd -> origin is the center voxel

    idx = np.round(coords[:, [2, 1, 0]] / voxel_size).astype(int) + n_half[::-1]
    inside = np.all((idx >= 0) & (idx < shape[::-1]), axis=1)
    idx = idx[inside]

    mask = np.zeros(tuple(shape[::-1]), dtype=bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Grow by the atom's own radius plus the requested gap, thresholding a
    # distance transform at the true radius in ANGSTROM rather than dilating
    # by ceil(pad / voxel_size) whole voxels. That does NOT make the result
    # sub-voxel accurate -- the mask is on the voxel grid, so it can only
    # grow in whole voxels either way (see `gap`'s own docstring). What it
    # buys is growing by the RIGHT number of voxels: ceil() rounds 1.9 A up
    # to a full 6.8 A voxel and 6.9 A up to two, a 2.7x volume
    # over-dilation for a 70 A protein that destroys achievable density.
    if pad > 0:
        dist = ndimage.distance_transform_edt(~mask) * voxel_size
        mask = dist <= pad

    return torch.from_numpy(np.ascontiguousarray(mask))


def coarsen_mask(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Max-pool a footprint mask down by an integer factor.

    A coarse voxel is set if ANY fine voxel inside it is set, so the fine
    shape is contained in the coarse one. That containment is what makes
    packing on a coarse grid safe for a fine render: two instances whose
    coarse masks are disjoint occupy disjoint coarse voxels, hence disjoint
    fine voxels, so a hard-collision guarantee established coarsely still
    holds at full resolution.

    Rasterizing atoms directly at the coarse voxel size does NOT give that.
    A coarse mask built that way is only "voxels containing an atom
    center", and an atom near a voxel boundary has its van der Waals radius
    spilling into an unmarked neighbour -- measured at 0.02% of footprint
    voxels overlapping when packing at 5 A and rendering at 1 A (0.07% at
    2.5 A). Recovering the guarantee by dilating the coarse mask instead
    costs a full coarse voxel and 42% of the placements; this costs only
    partial-voxel rounding.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask at the FINE voxel size.
    factor : int
        Integer ratio ``packing_voxel_size / voxel_size``.

    Returns
    -------
    torch.Tensor
        Boolean mask coarsened by `factor`, odd-sized on every axis so the
        molecule origin stays at the center voxel.
    """
    import torch.nn.functional as F

    if factor <= 1:
        return mask

    # Pad so the molecule origin lands in the CENTER coarse voxel, and the
    # coarse result is odd-sized. Padding only at the end (the obvious
    # thing) shifts the origin off-center by up to half a coarse voxel,
    # which would offset every coarse-packed position from where the fine
    # render draws it.
    pads: list[tuple[int, int]] = []
    shape_out: list[int] = []
    for s_ax in mask.shape:
        origin = (s_ax - 1) // 2
        c = -(-s_ax // factor)
        if c % 2 == 0:
            c += 1
        while True:
            origin_padded = (c - 1) // 2 * factor + (factor - 1) // 2
            before = origin_padded - origin
            after = c * factor - s_ax - before
            if before >= 0 and after >= 0:
                break
            c += 2
        pads.append((before, after))
        shape_out.append(c)

    # F.pad takes the last axis first: (x_before, x_after, y_..., z_...)
    flat: list[int] = []
    for before, after in reversed(pads):
        flat.extend((before, after))
    padded = F.pad(mask[None, None].to(torch.float32), flat)

    pooled = F.max_pool3d(padded, kernel_size=factor, stride=factor)[0, 0] > 0
    assert tuple(pooled.shape) == tuple(shape_out), (pooled.shape, shape_out)
    return pooled


def _rotation_cache(
    mask: torch.Tensor,
    n_orientations: int,
    device: torch.device | str,
    seed: int | None,
    pool_factor: int = 1,
) -> tuple[list[np.ndarray], list[tuple], list[bool], list[np.ndarray], torch.Tensor]:
    """
    Rotate `mask` into `n_orientations` orientations, once.

    With `pool_factor` > 1 the mask is rotated at ITS OWN (fine) resolution
    and each orientation is then max-pooled down to the packing grid. The
    order matters and is the whole point: pooling first and rotating the
    coarse result does not contain the rotated fine shape, because rotation
    interpolates, so a hard-collision guarantee established on the coarse
    grid would not survive being rendered at fine resolution. Rotating
    first keeps containment per orientation, which is what makes coarse
    packing exact rather than merely close.

    Also returns each orientation's shape and half-shape as plain Python
    ints, and whether its own center voxel is solid. Both exist purely to
    keep the RSA loop off numpy: at a 0.2% hit rate the loop runs ~300k
    times per stage and is dominated by per-attempt Python/numpy overhead
    (profiled at 60% of packing time, against 31% for the collision test
    itself), so anything hoistable out of it is worth hoisting.

    Returns
    -------
    cache : list of np.ndarray
        Trimmed boolean footprint per orientation.
    geom : list of tuple
        ``(nz, ny, nx, hz, hy, hx)`` per orientation, Python ints.
    probes : list of np.ndarray
        Up to `_N_PROBE` (i, j, k) indices of set voxels per orientation,
        sampled once. A placement whose footprint covers an occupied voxel
        is a clash, so finding one at a probe is an EXACT rejection, not a
        heuristic -- and at realistic density it lands one almost every
        time, which is what makes it worth checking a few dozen voxels
        before the full footprint (measured 7.3 us against 46.6 us, and
        catching 100% of rejections at 25% occupancy).
    centre_solid : list of bool
        Whether the footprint's own center voxel is set. Only when it is
        does "center voxel already occupied" imply a guaranteed collision,
        which is what licenses the loop's cheap pre-test -- a molecule with
        a hollow center (a shell, a channel) can legitimately be placed
        over an occupied center voxel.
    R : torch.Tensor
        The rotation matrices, shape (n_orientations, 3, 3).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    R = random_rotation_matrix(n_orientations)
    if R.dim() == 2:
        R = R.unsqueeze(0)
    theta = build_affine_matrix(R.to(device))
    vol = mask.to(device=device, dtype=torch.float32)
    # Rotate in chunks. `rotate_volume` materializes (batch, D, H, W) as
    # float32, and D grows as (molecule diameter / voxel_size) * sqrt(3):
    # for a 243 A species that is 0.7 GB at 5 A but 5.4 GB at 2.5 A and
    # 81 GB at 1 A, so a single-shot call OOMs at fine voxel sizes for no
    # reason. Peak scales with `chunk` instead of `n_orientations`.
    chunk = max(
        1, min(n_orientations, int(_ROTATE_CHUNK_VOXELS // max(vol.numel(), 1)))
    )
    # Written straight into the retained cache, NOT collected into a list and
    # concatenated. `torch.cat` has to hold its inputs and its output at the
    # same time, so it doubles the peak at exactly the moment the cache is
    # largest: profiled on a 219k-atom structure at 1 A voxels, the chunks
    # held 7.53 GiB and the concatenated copy another 7.53, which was 95% of
    # the whole run's 15.84 GiB peak. Filling a preallocated buffer keeps the
    # peak at one copy.
    rot: torch.Tensor | None = None
    filled = 0
    for i in range(0, n_orientations, chunk):
        piece = rotate_volume(vol, theta[i : i + chunk], padding_mode="zeros") > 0.5
        if pool_factor > 1:
            # Pool inside the chunk loop so the fine rotation is transient
            # and only the coarse result is retained.
            piece = (
                F.max_pool3d(
                    piece[:, None].to(torch.float32),
                    kernel_size=pool_factor,
                    stride=pool_factor,
                    ceil_mode=True,
                )[:, 0]
                > 0
            )
        if rot is None and piece.shape[0] == n_orientations:
            # One chunk covers every orientation: keep it as-is rather than
            # allocating a second buffer to copy it into.
            rot = piece
            filled = n_orientations
            continue
        if rot is None:
            rot = torch.empty(
                (n_orientations, *piece.shape[1:]),
                dtype=piece.dtype,
                device=piece.device,
            )
        rot[filled : filled + piece.shape[0]] = piece
        filled += piece.shape[0]
        del piece
    assert rot is not None and filled == n_orientations

    # Trim bounds from any-reductions rather than `np.nonzero`, which
    # materializes an index array per True voxel. Worth ~2x on the trim step
    # (32 ms -> 17 ms per 256 orientations), though the trim is not what
    # dominates this function -- `rotate_volume` above is, at 138 ms on CPU
    # against 65 ms on CUDA for the same batch, which is why `device` matters
    # here far more than any of this arithmetic does.
    n = rot.shape[0]
    zy = rot.any(dim=3)  # (N, Z, Y) -- shared by the Z and Y spans
    az = zy.any(dim=2)  # (N, Z)
    ay = zy.any(dim=1)  # (N, Y)
    ax = rot.any(dim=2).any(dim=1)  # (N, X)

    def _span(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """First and last True index per row; (0, 0) for an all-False row."""
        idx = torch.arange(a.shape[1], device=a.device).expand_as(a)
        big = torch.iinfo(torch.int64).max
        lo = torch.where(a, idx, torch.full_like(idx, big)).min(dim=1).values
        hi = torch.where(a, idx, torch.full_like(idx, -1)).max(dim=1).values
        empty = ~a.any(dim=1)
        return lo.masked_fill(empty, 0), hi.masked_fill(empty, -1)

    z0, z1 = _span(az)
    y0, y1 = _span(ay)
    x0, x1 = _span(ax)
    bounds = torch.stack([z0, z1, y0, y1, x0, x1], dim=1).cpu().numpy()
    rot_np = rot.cpu().numpy()

    rng = np.random.default_rng(0 if seed is None else seed)
    cache: list[np.ndarray] = []
    geom: list[tuple] = []
    centre_solid: list[bool] = []
    probes: list[np.ndarray] = []
    for i in range(n):
        bz0, bz1, by0, by1, bx0, bx1 = (int(v) for v in bounds[i])
        if bz1 < bz0:
            fp = rot_np[i]
        else:
            fp = np.ascontiguousarray(
                rot_np[i, bz0 : bz1 + 1, by0 : by1 + 1, bx0 : bx1 + 1]
            )
        cache.append(fp)
        sz, sy, sx = fp.shape
        geom.append((sz, sy, sx, sz // 2, sy // 2, sx // 2))
        centre_solid.append(bool(fp[sz // 2, sy // 2, sx // 2]) if fp.size else False)
        idx = np.array(np.nonzero(fp)).T
        if idx.shape[0] > _N_PROBE:
            idx = idx[rng.choice(idx.shape[0], _N_PROBE, replace=False)]
        probes.append(np.ascontiguousarray(idx))
    return cache, geom, centre_solid, probes, R


def pack_shapes_3d(
    species_masks: list[torch.Tensor],
    species_idx: torch.Tensor,
    grid_shape: tuple[int, int, int],
    voxel_size: float,
    occupancy: torch.Tensor | None = None,
    region_mask: torch.Tensor | None = None,
    n_orientations: int = 256,
    max_retries: int = 1500,
    stall_patience: int = 5_000,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    clip_axes: tuple[bool, bool, bool] = (False, False, False),
    pool_factor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack rigid molecular shapes into a voxel grid via Random Sequential
    Addition, colliding real rotated footprints against a running occupancy
    grid.

    Parameters
    ----------
    species_masks : list of torch.Tensor
        Per species, a boolean footprint mask from `build_species_mask`,
        already carrying whatever `gap` is wanted.
    species_idx : torch.Tensor
        Species index per candidate instance, shape (N,). Instances are
        attempted largest-footprint-species first.
    grid_shape : tuple of int
        (Z, Y, X) of the volume being packed into.
    voxel_size : float
        A per voxel, shared by `species_masks`, `occupancy` and `region_mask`.
    occupancy : torch.Tensor, optional
        Boolean (Z, Y, X) grid, True where an instance may NOT go. Seed it
        with membranes/filaments/carbon already rendered. Modified in place
        as instances are placed. Default None (empty).
    region_mask : torch.Tensor, optional
        Boolean (Z, Y, X); instance centers are drawn uniformly from its True
        voxels. Default None (anywhere in the grid).
    n_orientations : int, optional
        Size of the per-species rotation cache. Default 256.
    max_retries : int, optional
        Trial positions per instance before giving up on it. This is the
        knob that sets achieved density, and it matters far more than any
        other parameter here -- measured on a 10-species 6.8 A benchmark
        against CryoTomoSim's own 0.240 macromolecule volume fraction:

        =============  ================  ==========
        max_retries    volume fraction   wall time
        =============  ================  ==========
        60             0.208             8 s
        200            0.229             25 s
        600            0.241             72 s
        1500           0.251             176 s
        =============  ================  ==========

        Default 1500, paired with `stall_patience`'s 5000. The two are
        complementary, not alternatives: `max_retries` raises a stage's
        attempt CEILING (a stage holds a finite candidate pool, so its total
        budget is roughly n_candidates * max_retries), while
        `stall_patience` cuts that budget short once the species has
        saturated. Raising one and lowering the other beats tuning either:

        ===========  =======  =====  =========
        max_retries  stall    vf     wall time
        ===========  =======  =====  =========
        600          off      0.198  30.0 s
        1500         off      0.207  73.4 s
        600          5000     0.196  13.6 s
        1500         5000     0.199  14.2 s
        ===========  =======  =====  =========

        The last row is both faster and slightly denser than the first, so
        there is no speed/density trade to make between them.
    stall_patience : int, optional
        Abandon a species stage after this many consecutive failed
        ATTEMPTS. Default 5000.

        ==============  ========  =========  =========
        stall_patience  vf        wall time  vs unbounded
        ==============  ========  =========  =========
        unbounded       0.197     30.2 s     --
        20000           0.195     27.5 s     1.1x
        5000            0.195     15.0 s     2.0x
        2000            0.190      7.9 s     3.8x
        ==============  ========  =========  =========

        Note the density loss is a shift in COMPOSITION, not simply fewer
        particles: cutting a large species' stage short frees space that
        later, smaller species take, so 2000 places more instances than
        unbounded while reaching lower volume fraction.

        A FIXED count beats scaling it with the stage's candidate count,
        which was tried and rejected: ``20 * candidates`` resolves to ~48000
        at production scale, and paired with `max_retries` that raises the
        total attempt budget ABOVE no-limit-at-600, so the whole build ran
        8% slower (898 s) than the 832 s it was meant to improve on.

        Counted in attempts, not failed instances, which bounds the wasted
        work directly: a failed *instance* costs `max_retries` attempts, so
        an instance-counted patience of P allows P * max_retries doomed
        collision tests per species. A realistic run supplies far more
        candidates than fit -- `occupancy_fraction=1.0` on a 288M-voxel box
        drew ~63,000 candidates of which ~15,000 fit -- and with no early
        exit the other ~48,000 each burned all 600 retries, ~29 million
        doomed tests and the dominant cost of the whole build.

        A stage is a single species, so every candidate in it is the same
        size and consecutive misses genuinely mean that species has
        saturated. (An earlier version of this docstring argued the
        opposite, that a failure says nothing about the next candidate;
        that is true across species but not within a stage, which is the
        only thing this counter spans.)
    seed : int, optional
    device : str or torch.device, optional
        Device for the rotation cache only; the RSA loop is numpy on the
        host, where the collision test is already 0.08 ms.
    clip_axes : tuple of bool, optional
        (z, y, x). True lets a footprint extend past that wall (it is
        truncated); False requires it to fit entirely.
    pool_factor : int, optional
        Collide on a grid `pool_factor` times coarser than `species_masks`
        are rasterized at. `species_masks` are then at
        ``voxel_size / pool_factor``, and each orientation is rotated at
        that resolution before being pooled onto the packing grid -- see
        `_rotation_cache`, where the order of those two operations is what
        makes the guarantee hold at render resolution. Default 1 (masks are
        already at the packing grid's own voxel size).

    Returns
    -------
    coords : torch.Tensor
        Accepted instance centers (x, y, z), A, box-centered, shape (M, 3).
    rotations : torch.Tensor
        Rotation matrix each instance was accepted at, shape (M, 3, 3).
    accepted_idx : torch.Tensor
        Indices into `species_idx` for accepted instances, shape (M,).
    occupancy : torch.Tensor
        The updated occupancy grid.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = grid_shape
    grid = np.array([nz, ny, nx])

    occ = (
        np.zeros(grid_shape, dtype=bool)
        if occupancy is None
        else occupancy.detach().cpu().numpy().astype(bool).copy()
    )
    if region_mask is None:
        allowed = None
    else:
        rm = region_mask.detach().cpu().numpy()
        n_allowed = int(rm.sum())
        if n_allowed == 0:
            raise ValueError("region_mask has no True voxels")
        if n_allowed / rm.size >= _DENSE_REGION_FRACTION:
            # Enumerating the allowed voxels costs an (N, 3) int64 array --
            # 6.3 GB and ~8 s for the cytosol of a 288M-voxel box, per call.
            # When the region is most of the box that buys nothing: a
            # uniform draw lands inside it almost every time, and anything
            # outside is already marked occupied (callers seed `occupancy`
            # with the region's complement), so the collision test rejects
            # it for free.
            allowed = None
        else:
            allowed = np.stack(np.nonzero(rm), axis=1)

    species_idx_np = species_idx.detach().cpu().numpy()
    footprints = np.array([int(m.sum()) for m in species_masks])
    order = np.argsort(-footprints)

    out_pos: list[tuple[int, int, int]] = []
    out_rot: list[np.ndarray] = []
    out_row: list[int] = []

    # RSA at a realistic density rejects ~99.8% of attempts, so essentially
    # all of this loop's cost is the reject path. Each batch of `_BATCH`
    # draws is therefore screened at once:
    #
    # * the tests that need no grid (footprint bounds, wall clipping) are
    #   vectorised outright;
    # * the tests that read the grid (centre voxel, first and second probes)
    #   are vectorised against the grid AS IT STANDS when the batch is drawn.
    #   Occupancy only ever grows, so an attempt they reject would be
    #   rejected at its own moment too -- exact. An attempt they pass may
    #   still be doomed by an acceptance earlier in the batch, so survivors
    #   are re-tested one at a time against the live grid, exactly as before.
    #
    # Per-attempt bookkeeping (which candidate row an attempt belongs to,
    # the stall counter) is done arithmetically over runs of rejections. The
    # RNG is consumed in the same order as a one-attempt-at-a-time loop, so
    # placements are bitwise identical to that form (pinned in
    # tests/test_pack_shapes_reference.py).
    clip_z, clip_y, clip_x = clip_axes
    occ_flat = occ.reshape(-1)
    plane = ny * nx

    for s in order:
        rows = np.flatnonzero(species_idx_np == s)
        if rows.size == 0:
            continue
        cache, geom, centre_solid, probes, R = _rotation_cache(
            species_masks[int(s)], n_orientations, device, seed, pool_factor
        )
        R_np = R.cpu().numpy()
        n_cache = len(cache)
        g = np.asarray(geom, dtype=np.int64)  # (n_cache, 6): fz fy fx hz hy hx
        fz_a, fy_a, fx_a, hz_a, hy_a, hx_a = (g[:, i] for i in range(6))
        cs_a = np.asarray(centre_solid, dtype=bool)

        # Probe voxels padded to (n_cache, P) with a validity mask, as flat
        # offsets (for footprints wholly inside the grid) and as (i, j, k)
        # (for overhanging ones, where only the in-bounds voxels are probed).
        off_pad = np.zeros((n_cache, _N_PROBE), dtype=np.int64)
        off_ok = np.zeros((n_cache, _N_PROBE), dtype=bool)
        p1 = np.zeros((3, n_cache, _N_PROBE), dtype=np.int64)
        for oi, pr in enumerate(probes):
            m = pr.shape[0] if pr.size else 0
            if m:
                off_pad[oi, :m] = pr[:, 0] * plane + pr[:, 1] * nx + pr[:, 2]
                off_ok[oi, :m] = True
                p1[:, oi, :m] = pr.T
        rng2 = np.random.default_rng(0 if seed is None else seed)
        p2 = np.zeros((3, n_cache, _N_PROBE2), dtype=np.int64)
        p2_ok = np.zeros((n_cache, _N_PROBE2), dtype=bool)
        for oi, fp in enumerate(cache):
            idx = np.array(np.nonzero(fp)).T
            if idx.shape[0] > _N_PROBE2:
                idx = idx[rng2.choice(idx.shape[0], _N_PROBE2, replace=False)]
            m = idx.shape[0]
            if m:
                p2[:, oi, :m] = idx.T
                p2_ok[oi, :m] = True

        def _probe_hits(
            sel: np.ndarray,
            o: np.ndarray,
            lz: np.ndarray,
            ly: np.ndarray,
            lx: np.ndarray,
            pijk: np.ndarray,
            pok: np.ndarray,
        ) -> np.ndarray:
            """Which of the attempts `sel` have an occupied, in-bounds probe voxel."""
            oc = o[sel]
            pz = lz[sel][:, None] + pijk[0][oc]
            py = ly[sel][:, None] + pijk[1][oc]
            px = lx[sel][:, None] + pijk[2][oc]
            valid = (
                pok[oc]
                & (pz >= 0)
                & (pz < nz)
                & (py >= 0)
                & (py < ny)
                & (px >= 0)
                & (px < nx)
            )
            flat = np.where(valid, pz * plane + py * nx + px, 0)
            return (occ_flat[flat] & valid).any(axis=1)

        n_rows = rows.size
        row_i = 0  # candidate row the next attempt belongs to
        used = 0  # attempts spent on that row so far
        stall = 0
        done = False

        while not done:
            # ---- draw a batch: the same calls, in the same order, as a
            # one-attempt-at-a-time loop refilling an exhausted buffer ----
            oris = rng.integers(0, n_cache, _BATCH)
            if allowed is None:
                zs = rng.integers(0, nz, _BATCH)
                ys = rng.integers(0, ny, _BATCH)
                xs = rng.integers(0, nx, _BATCH)
            else:
                picks = allowed[rng.integers(0, allowed.shape[0], _BATCH)]
                zs, ys, xs = picks[:, 0], picks[:, 1], picks[:, 2]

            # ---- vectorised exact rejections ----
            rej = cs_a[oris] & occ[zs, ys, xs]
            lz = zs - hz_a[oris]
            ly = ys - hy_a[oris]
            lx = xs - hx_a[oris]
            ez = lz + fz_a[oris]
            ey = ly + fy_a[oris]
            ex = lx + fx_a[oris]
            loz = np.maximum(lz, 0)
            loy = np.maximum(ly, 0)
            lox = np.maximum(lx, 0)
            hiz = np.minimum(ez, nz)
            hiy = np.minimum(ey, ny)
            hix = np.minimum(ex, nx)
            rej |= (hiz <= loz) | (hiy <= loy) | (hix <= lox)
            if not clip_z:
                rej |= (loz != lz) | (hiz != ez)
            if not clip_y:
                rej |= (loy != ly) | (hiy != ey)
            if not clip_x:
                rej |= (lox != lx) | (hix != ex)
            inside = (
                (loz == lz)
                & (loy == ly)
                & (lox == lx)
                & (hiz == ez)
                & (hiy == ey)
                & (hix == ex)
            )
            sel = np.flatnonzero(inside & ~rej)
            if sel.size:
                base = lz[sel] * plane + ly[sel] * nx + lx[sel]
                hit = (
                    occ_flat[base[:, None] + off_pad[oris[sel]]] & off_ok[oris[sel]]
                ).any(axis=1)
                rej[sel[hit]] = True
            sel = np.flatnonzero(~inside & ~rej)
            if sel.size:
                rej[sel[_probe_hits(sel, oris, lz, ly, lx, p1, off_ok)]] = True
            sel = np.flatnonzero(~rej)
            if sel.size:
                rej[sel[_probe_hits(sel, oris, lz, ly, lx, p2, p2_ok)]] = True
            survivors = np.flatnonzero(~rej)

            # ---- walk the batch in draw order: runs of rejections are
            # absorbed arithmetically, each survivor is attempted exactly ----
            pos = 0
            si = 0
            while pos < _BATCH:
                nxt = int(survivors[si]) if si < survivors.size else _BATCH
                k = nxt - pos
                if k > 0:
                    # k consecutive rejections: each is an attempt that
                    # bumps the stall counter and consumes one retry of the
                    # current row, rows advancing as their retries run out.
                    n_exec = min(k, stall_patience - stall)
                    stall += n_exec
                    while n_exec > 0:
                        take = min(n_exec, max_retries - used)
                        used += take
                        n_exec -= take
                        if used == max_retries:
                            row_i += 1
                            used = 0
                            if row_i >= n_rows:
                                break
                    if stall >= stall_patience or row_i >= n_rows:
                        done = True
                        break
                    pos = nxt
                if nxt >= _BATCH:
                    break

                # the survivor at buffer index `nxt`, against the live grid
                stall += 1
                oi = int(oris[nxt])
                cz, cy, cx = int(zs[nxt]), int(ys[nxt]), int(xs[nxt])
                accepted = False
                if not (cs_a[oi] and occ[cz, cy, cx]):
                    fz, fy, fx, hz, hy, hx = geom[oi]
                    lz1, ly1, lx1 = cz - hz, cy - hy, cx - hx
                    ez1, ey1, ex1 = lz1 + fz, ly1 + fy, lx1 + fx
                    loz1 = 0 if lz1 < 0 else lz1
                    loy1 = 0 if ly1 < 0 else ly1
                    lox1 = 0 if lx1 < 0 else lx1
                    hiz1 = nz if ez1 > nz else ez1
                    hiy1 = ny if ey1 > ny else ey1
                    hix1 = nx if ex1 > nx else ex1
                    fp = cache[oi]
                    sub = fp[
                        loz1 - lz1 : hiz1 - lz1,
                        loy1 - ly1 : hiy1 - ly1,
                        lox1 - lx1 : hix1 - lx1,
                    ]
                    dst = (slice(loz1, hiz1), slice(loy1, hiy1), slice(lox1, hix1))
                    if not np.any(occ[dst] & sub):
                        occ[dst] |= sub
                        out_pos.append((cz, cy, cx))
                        out_rot.append(R_np[oi])
                        out_row.append(int(rows[row_i]))
                        accepted = True
                if accepted:
                    stall = 0
                    row_i += 1
                    used = 0
                else:
                    used += 1
                    if used == max_retries:
                        row_i += 1
                        used = 0
                pos = nxt + 1
                si += 1
                if row_i >= n_rows or stall >= stall_patience:
                    done = True
                    break

    if not out_pos:
        return (
            torch.empty((0, 3)),
            torch.empty((0, 3, 3)),
            torch.empty((0,), dtype=torch.long),
            torch.from_numpy(occ),
        )

    idx_zyx = np.asarray(out_pos)
    extent = grid[::-1] * voxel_size  # (x, y, z)
    origin = -0.5 * extent
    coords_xyz = origin + (idx_zyx[:, ::-1] + 0.5) * voxel_size

    return (
        torch.tensor(coords_xyz, dtype=torch.float32),
        torch.tensor(np.asarray(out_rot), dtype=torch.float32),
        torch.tensor(out_row, dtype=torch.long),
        torch.from_numpy(occ),
    )
