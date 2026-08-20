"""
Multi-species hard-sphere packing for specimen generation.

:func:`pack_hard_spheres_3d` -- Random Sequential Addition (RSA), fully
vectorized across candidates, fast (sub-second to a few seconds at a few
thousand instances), capped by RSA's known jamming limit (~28-41%
depending on species-size diversity, well below random-close-packing).
This is the backend behind `specter build tomogram`
(`specimen.tomogram.TomogramSpecimenGenerator`).

Benchmarked in ``dev/packing_algorithms.py`` against several other
candidates (naive/voxel-grid RSA, Lubachevsky-Stillinger, a CellPACK-style
incremental distance field, an SDF-scored variant, a periodic force-biased
relaxation reaching substantially higher density but with no
obstacle-avoidance mechanism, and Tetris-style contact-correlation
packing) -- see that file for the full comparison; this RSA
implementation is the one winner promoted here (began as
``pack_rsa_batched`` there). The force-biased and Tetris-style approaches
were themselves promoted into this package at one point
(``pack_hard_spheres_3d_dense``, `specimen.packing.tetris`) but never
gained a real caller beyond their own generators, which were themselves
superseded by `TomogramSpecimenGenerator`; both were removed rather than
carried along unused -- see git history if either is ever needed as a
reference again.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# specter.potential.compute_supersampling_parameters's own default atomic
# kernel width (potential.py: width_atom=5.0) -- each atom's own potential
# is evaluated over a +/-2.5 A box around it, so a molecule's bounding box
# needs at least this much margin beyond its outermost atom or that atom's
# kernel gets truncated by convolution. Same value as
# specimen/membrane/_profile.py's ATOM_KERNEL_HALF_WIDTH_A -- kept as an
# independent local constant (not imported) to keep this module's
# dependencies limited, matching this codebase's established
# per-generator zero-cross-coupling convention.
ATOM_KERNEL_HALF_WIDTH_A = 2.5


def estimate_protein_box_size(max_diameter: float, voxel_size: float) -> int:
    """
    Grid size (voxels, per axis) for a molecule with the given max diameter
    (Å, from ``PDB.max_diameter``) at voxel size ``voxel_size``.

    Parameters
    ----------
    max_diameter : float
    voxel_size : float

    Returns
    -------
    int
        Even grid size in voxels.
    """
    margin_a = 2 * ATOM_KERNEL_HALF_WIDTH_A
    n = int(np.ceil((max_diameter + 2 * margin_a) / voxel_size))
    n += n % 2
    return n


def draw_species_pool(
    species_radii: torch.Tensor,
    species_ratios: torch.Tensor,
    occupancy_fraction: float,
    box_volume: float,
    seed: int | None = None,
    species_volumes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a candidate instance pool (one row per sphere to attempt placing)
    whose combined volume reaches ``occupancy_fraction * box_volume``,
    species drawn with probability proportional to `species_ratios`
    (matching how
    `specter.specimen.cytosolic_filler.PEI2016_CROWDING_TABLE`'s
    `occurrence_freq` is meant to be used as a relative-abundance weight,
    not an absolute one).

    Parameters
    ----------
    species_radii : torch.Tensor, shape (S,)
        Real physical radius of each species, Å.
    species_ratios : torch.Tensor, shape (S,)
        Relative abundance weight per species. Only ratios between entries
        matter, not the absolute values.
    occupancy_fraction : float
        Target combined volume, as a fraction of `box_volume`. Measured in
        bare-sphere volume by default, or in `species_volumes` if given.
    box_volume : float
        Å³.
    seed : int, optional
        Random seed.
    species_volumes : torch.Tensor, shape (S,), optional
        Real occupied volume per species, Å³, used instead of the bare
        sphere ``4/3 pi r^3`` when accumulating toward the target.

        Required for `pack_shapes_3d`, whose candidates occupy their actual
        molecular footprint rather than a bounding sphere. A molecule's
        envelope is only ~0.178 of that sphere, so sizing a shape packer's
        pool the default way under-supplies it by roughly 5.6x -- the packer
        then exhausts its candidates well before the box is full, and
        reports a density that reflects the pool rather than its own
        geometry. Default None (bare-sphere volume, correct for
        `pack_hard_spheres_3d`).

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
        if species_volumes is None:
            accumulated += (4.0 / 3.0) * torch.pi * r**3
        else:
            accumulated += float(species_volumes[k])

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
    field_voxel_size: float | None = None,
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
        Requested sphere radius per candidate instance, Å. Order is
        irrelevant (spheres are internally grouped and processed
        largest-first regardless of input order).
    box : tuple of float
        (D, H, W) box extents in Å (z, y, x), centered at the
        origin -- same convention as
        :func:`specter.coords.poisson_disk_neighbors_3d`.
    gap : float, optional
        Extra clearance between sphere surfaces, Å, beyond simple
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
        Physical distance (Å) to the nearest FORBIDDEN voxel, shape
        ``(Z, Y, X)``, on a box-centered grid at `field_voxel_size` spacing
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
    field_voxel_size : float, optional
        Voxel size of `exclusion_distance_field`/`sampling_mask`, Å
        (the two share one grid). Required if either is given. Trilinear
        interpolation of a coarse `exclusion_distance_field` lets a
        candidate's sampled distance run a little past the true
        (exact-voxel) value near a boundary -- confirmed a couple of
        Å of bleed at `field_voxel_size=5` for `gap=2`, vanishing by
        `field_voxel_size=2`; keep this fine relative to `gap` (or pad the
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
        inside) fixes this by construction. Default None (uniform box-wide
        sampling, the original behavior).

    Returns
    -------
    coords : torch.Tensor, shape (M, 3)
        Accepted sphere centers (x, y, z), Å, box-centered. M <= N.
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
    if has_field_consumer != (field_voxel_size is not None):
        raise ValueError(
            "field_voxel_size is required together with (and only with) "
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
            * field_voxel_size
        )
        origin = -0.5 * extent
        idx_xyz = voxel_idx_zyx[:, [2, 1, 0]].to(device=device, dtype=torch.float32)
        mask_voxel_positions_xyz = origin + (idx_xyz + 0.5) * field_voxel_size

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
        assert exclusion_distance_field is not None and field_voxel_size is not None
        nz, ny, nx = exclusion_distance_field.shape
        extent = (
            torch.tensor([nx, ny, nz], dtype=points_xyz.dtype, device=device)
            * field_voxel_size
        )
        origin = -0.5 * extent
        idx = (points_xyz - origin) / field_voxel_size
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
                assert field_voxel_size is not None
                chosen = torch.randint(
                    0,
                    mask_voxel_positions_xyz.shape[0],
                    (remaining.numel(),),
                    generator=gen,
                    device=device,
                )
                jitter = (
                    torch.rand(remaining.numel(), 3, generator=gen, device=device) - 0.5
                ) * field_voxel_size
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
