"""
`pack_shapes_3d` screens each batch of draws vectorised and probes overhanging
footprints; placements must be bitwise identical to the one-attempt-at-a-time
loop it replaced (2026-09-02). That loop is kept here verbatim as the
reference and both are run from the same RNG state on a synthetic species set,
over the regimes that exercise every branch: uniform draws and allowed-list
draws, all walls clippable and none, a coarse packing grid, tight patience.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specter.specimen.packing._shape import (
    _DENSE_REGION_FRACTION,
    _rotation_cache,
    build_species_mask,
    pack_shapes_3d,
)


def pack_shapes_3d_reference(
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
    """The one-attempt-at-a-time loop this replaced, kept verbatim (docstring removed)."""
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
    # all of this loop's cost is the reject path, and that cost is dominated
    # by per-attempt Python/numpy overhead rather than the collision test
    # (profiled 60% vs 31%). Hence: random draws are batched rather than
    # taken one scalar at a time, every bound is a plain Python int rather
    # than a numpy array, a single-voxel pre-test skips the slice+AND for
    # the ~31% of attempts whose center voxel is already occupied, and a
    # sparse probe of `_N_PROBE` footprint voxels rejects the rest before
    # the full AND runs. Together these took a 600-retry benchmark pack
    # from 72 s to 5 s with volume fraction unchanged.
    clip_z, clip_y, clip_x = clip_axes
    all_clip = clip_z and clip_y and clip_x
    _BATCH = 8192

    for s in order:
        rows = np.flatnonzero(species_idx_np == s)
        if rows.size == 0:
            continue
        cache, geom, centre_solid, probes, R = _rotation_cache(
            species_masks[int(s)], n_orientations, device, seed, pool_factor
        )
        # Probe indices -> flat offsets, which needs the grid's own strides
        # and so cannot be precomputed inside the cache.
        occ_flat = occ.reshape(-1)
        probe_off = [
            (pr[:, 0] * ny * nx + pr[:, 1] * nx + pr[:, 2]) if pr.size else pr
            for pr in probes
        ]
        R_np = R.cpu().numpy()
        n_cache = len(cache)

        # Batched draws, refilled as consumed.
        buf_i = _BATCH
        oris = zs = ys = xs = picks = None
        stall = 0
        for row in rows:
            if stall >= stall_patience:
                break
            for _ in range(max_retries):
                if stall >= stall_patience:
                    break
                stall += 1
                if buf_i >= _BATCH:
                    oris = rng.integers(0, n_cache, _BATCH)
                    if allowed is None:
                        zs = rng.integers(0, nz, _BATCH)
                        ys = rng.integers(0, ny, _BATCH)
                        xs = rng.integers(0, nx, _BATCH)
                    else:
                        picks = allowed[rng.integers(0, allowed.shape[0], _BATCH)]
                    buf_i = 0
                # buf_i starts at _BATCH, so the refill above always runs
                # before this first indexes oris/zs/ys/xs/picks -- these are
                # never None here.
                assert oris is not None
                oi = int(oris[buf_i])
                if allowed is None:
                    assert zs is not None and ys is not None and xs is not None
                    cz = int(zs[buf_i])
                    cy = int(ys[buf_i])
                    cx = int(xs[buf_i])
                else:
                    assert picks is not None
                    p = picks[buf_i]
                    cz, cy, cx = int(p[0]), int(p[1]), int(p[2])
                buf_i += 1

                # Exact early reject: only valid when this orientation's own
                # center voxel is solid (see _rotation_cache's centre_solid)
                if centre_solid[oi] and occ[cz, cy, cx]:
                    continue

                fz, fy, fx, hz, hy, hx = geom[oi]
                lz, ly, lx = cz - hz, cy - hy, cx - hx
                ez, ey, ex = lz + fz, ly + fy, lx + fx
                loz = 0 if lz < 0 else lz
                loy = 0 if ly < 0 else ly
                lox = 0 if lx < 0 else lx
                hiz = nz if ez > nz else ez
                hiy = ny if ey > ny else ey
                hix = nx if ex > nx else ex
                if hiz <= loz or hiy <= loy or hix <= lox:
                    continue
                if not all_clip:
                    if not clip_z and (loz != lz or hiz != ez):
                        continue
                    if not clip_y and (loy != ly or hiy != ey):
                        continue
                    if not clip_x and (lox != lx or hix != ex):
                        continue

                # Sparse probe first, valid only when the footprint sits
                # wholly inside the grid (a clipped one's flat offsets no
                # longer line up). An occupied probe voxel is a real clash,
                # so this rejects exactly, and at realistic density it fires
                # on nearly every doomed attempt.
                if (
                    loz == lz
                    and loy == ly
                    and lox == lx
                    and hiz == ez
                    and hiy == ey
                    and hix == ex
                ):
                    off = probe_off[oi]
                    if off.size and occ_flat[lz * ny * nx + ly * nx + lx + off].any():
                        continue

                fp = cache[oi]
                sub = fp[loz - lz : hiz - lz, loy - ly : hiy - ly, lox - lx : hix - lx]
                dst = (slice(loz, hiz), slice(loy, hiy), slice(lox, hix))
                if np.any(occ[dst] & sub):
                    continue
                occ[dst] |= sub
                out_pos.append((cz, cy, cx))
                out_rot.append(R_np[oi])
                out_row.append(int(row))
                stall = 0
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


def _species(seed: int, n_atoms: int, radius: float, voxel: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n_atoms, 3, generator=g)
    pts = (
        pts
        / pts.norm(dim=1, keepdim=True)
        * radius
        * torch.rand(n_atoms, 1, generator=g) ** (1 / 3)
    )
    return build_species_mask(pts, voxel)


@pytest.mark.parametrize(
    "clip_axes,pool_factor,sparse_region,max_retries,stall_patience",
    [
        ((True, True, True), 1, False, 60, 900),
        ((False, False, False), 1, False, 60, 900),
        ((True, False, True), 2, False, 40, 600),
        ((True, True, True), 1, True, 30, 400),
        ((True, True, True), 1, False, 5, 50),
    ],
)
def test_pack_shapes_3d_matches_one_attempt_at_a_time_reference(
    clip_axes, pool_factor, sparse_region, max_retries, stall_patience
):
    voxel = 6.0
    fine = voxel / pool_factor
    masks = [
        _species(1, 400, 40.0, fine),
        _species(2, 250, 28.0, fine),
        _species(3, 120, 18.0, fine),
    ]
    grid = (24, 40, 40)
    species_idx = torch.tensor([0] * 40 + [1] * 120 + [2] * 300)
    g = torch.Generator().manual_seed(7)
    occupancy = torch.rand(grid, generator=g) < 0.02  # a few obstacles
    if sparse_region:
        region = torch.zeros(grid, dtype=torch.bool)
        region[6:18, 8:32, 8:32] = True
        region &= ~occupancy
        assert region.float().mean() < _DENSE_REGION_FRACTION
    else:
        region = ~occupancy

    def run(fn):
        torch.manual_seed(0)
        return fn(
            masks,
            species_idx,
            grid,
            voxel,
            occupancy=occupancy.clone(),
            region_mask=region.clone(),
            n_orientations=24,
            max_retries=max_retries,
            stall_patience=stall_patience,
            seed=3,
            clip_axes=clip_axes,
            pool_factor=pool_factor,
        )

    want = run(pack_shapes_3d_reference)
    got = run(pack_shapes_3d)
    assert want[0].shape[0] > 0
    for name, a, b in zip(("coords", "rotations", "rows", "occupancy"), want, got):
        assert torch.equal(a, b), name


def test_pack_shapes_3d_empty_region_still_raises():
    masks = [_species(1, 100, 20.0, 5.0)]
    with pytest.raises(ValueError, match="no True voxels"):
        pack_shapes_3d(
            masks,
            torch.zeros(5, dtype=torch.long),
            (16, 16, 16),
            5.0,
            region_mask=torch.zeros(16, 16, 16, dtype=torch.bool),
        )
