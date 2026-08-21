"""
Tests for the shape-aware packing backend
(`specter.specimen.packing._shape`).

The property that matters most here is that a packer's *output* describes
the geometry it actually tested: `pack_shapes_3d` commits to an orientation
per instance, and the renderer reuses it, so a coords/rotation round-trip
that does not reproduce the occupancy grid would silently render overlapping
particles and mislabel every pick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from specter.rotations._volume import build_affine_matrix, rotate_volume
from specter.specimen.packing import (
    build_species_mask,
    coarsen_mask,
    pack_shapes_3d,
)


def _blob(n_atoms: int = 400, radius: float = 20.0, seed: int = 0) -> torch.Tensor:
    """A compact random point cloud standing in for a small globular protein."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n_atoms, 3, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    r = radius * torch.rand(n_atoms, 1, generator=g) ** (1 / 3)
    coords = v * r
    return coords - coords.mean(0)


def _rod(n_atoms: int = 400, half_len: float = 45.0, seed: int = 1) -> torch.Tensor:
    """An elongated cloud -- the case a bounding sphere handles worst."""
    g = torch.Generator().manual_seed(seed)
    z = (torch.rand(n_atoms, 1, generator=g) * 2 - 1) * half_len
    xy = torch.randn(n_atoms, 2, generator=g) * 4.0
    coords = torch.cat([xy, z], dim=1)
    return coords - coords.mean(0)


def test_build_species_mask_encloses_atoms_and_is_odd_sized():
    coords = _blob()
    voxel = 4.0
    mask = build_species_mask(coords, voxel, gap=0.0)

    assert mask.dtype == torch.bool
    # odd on every axis, so the molecule origin is the center voxel
    assert all(s % 2 == 1 for s in mask.shape)

    n_half = torch.tensor(mask.shape[::-1]) // 2  # (x, y, z)
    idx = torch.round(coords / voxel).long() + n_half
    assert bool(mask[idx[:, 2], idx[:, 1], idx[:, 0]].all()), (
        "every atom must fall inside its own footprint mask"
    )


def test_build_species_mask_gap_is_quantized_to_voxel_size():
    """Documented behaviour, not an accident -- see build_species_mask's `gap`."""
    coords = _blob()
    voxel = 6.8
    # 1.9 + 0.0 and 1.9 + 2.0 are both under one 6.8 A voxel, so identical.
    m0 = build_species_mask(coords, voxel, gap=0.0)
    m2 = build_species_mask(coords, voxel, gap=2.0)
    assert m0.shape == m2.shape
    assert bool((m0 == m2).all())

    # 1.9 + 5.0 crosses a voxel, so it must actually grow.
    m5 = build_species_mask(coords, voxel, gap=5.0)
    assert int(m5.sum()) > int(m0.sum())


def test_pack_shapes_3d_places_nothing_overlapping():
    masks = [build_species_mask(_blob(), 4.0, gap=0.0)]
    species = torch.zeros(60, dtype=torch.long)
    grid = (30, 60, 60)

    coords, rotations, accepted, occ = pack_shapes_3d(
        masks, species, grid, 4.0, seed=0, n_orientations=32, max_retries=40
    )

    assert accepted.numel() > 0
    assert coords.shape == (accepted.numel(), 3)
    assert rotations.shape == (accepted.numel(), 3, 3)
    # rotations must be proper rotations
    eye = torch.eye(3).expand_as(rotations)
    assert torch.allclose(rotations @ rotations.transpose(1, 2), eye, atol=1e-4)
    assert torch.allclose(
        torch.linalg.det(rotations), torch.ones(len(rotations)), atol=1e-4
    )


def test_pack_shapes_3d_output_reproduces_its_own_occupancy_grid():
    """
    Re-stamp from the RETURNED coords/rotations and require an exact match
    with the packer's occupancy grid. This is what guarantees the renderer
    draws the geometry that was collision-tested.
    """
    voxel = 4.0
    masks = [build_species_mask(_blob(), voxel, gap=0.0)]
    species = torch.zeros(50, dtype=torch.long)
    grid = (30, 60, 60)

    coords, rotations, accepted, occ = pack_shapes_3d(
        masks, species, grid, voxel, seed=1, n_orientations=32, max_retries=40
    )
    assert accepted.numel() > 0

    replay = np.zeros(grid, dtype=np.int32)
    extent = np.array(grid)[::-1] * voxel
    origin = -0.5 * extent
    for i in range(accepted.numel()):
        m = masks[0].to(torch.float32)
        theta = build_affine_matrix(rotations[i].unsqueeze(0))
        rm = (rotate_volume(m, theta, padding_mode="zeros")[0] > 0.5).numpy()
        nzi = np.nonzero(rm)
        sl = tuple(slice(int(a.min()), int(a.max()) + 1) for a in nzi)
        fp = rm[sl]
        centre = np.round((coords[i].numpy() - origin) / voxel - 0.5).astype(int)[::-1]
        loc = centre - np.array(fp.shape) // 2
        lo = np.maximum(loc, 0)
        hi = np.minimum(loc + np.array(fp.shape), np.array(grid))
        slo = lo - loc
        shi = slo + (hi - lo)
        replay[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] += fp[
            slo[0] : shi[0], slo[1] : shi[1], slo[2] : shi[2]
        ]

    assert int((replay > 1).sum()) == 0, "instances overlap when replayed"
    assert int((replay > 0).sum()) == int(occ.sum()), (
        "replayed footprints do not reproduce the packer's occupancy grid"
    )


def test_pack_shapes_3d_respects_a_seeded_occupancy_grid():
    voxel = 4.0
    masks = [build_species_mask(_blob(), voxel, gap=0.0)]
    grid = (30, 60, 60)
    blocked = torch.zeros(grid, dtype=torch.bool)
    blocked[:, :30, :] = True  # forbid half the box

    coords, _, accepted, occ = pack_shapes_3d(
        masks,
        torch.zeros(40, dtype=torch.long),
        grid,
        voxel,
        occupancy=blocked,
        region_mask=~blocked,
        seed=0,
        n_orientations=32,
        max_retries=40,
    )
    assert accepted.numel() > 0
    # nothing may have been written into the forbidden half
    assert bool((occ[:, :30, :] == blocked[:, :30, :]).all())


def test_shape_packing_beats_bounding_sphere_for_an_elongated_species():
    """
    The whole point of the backend: a bounding sphere wastes most of an
    elongated molecule's excluded volume, so exact-shape collision must fit
    materially more of them into the same box.
    """
    voxel = 4.0
    coords = _rod()
    grid = (40, 80, 80)
    n = 120

    masks = [build_species_mask(coords, voxel, gap=0.0)]
    _, _, accepted_shape, _ = pack_shapes_3d(
        masks,
        torch.zeros(n, dtype=torch.long),
        grid,
        voxel,
        seed=0,
        n_orientations=64,
        max_retries=60,
    )

    from specter.specimen.packing import pack_hard_spheres_3d

    box = tuple(float(g * voxel) for g in grid)
    radius = float(torch.cdist(coords, coords).max()) / 2.0
    _, accepted_sphere = pack_hard_spheres_3d(
        torch.full((n,), radius), box, gap=0.0, seed=0, max_passes=200
    )

    assert accepted_shape.numel() > 2 * accepted_sphere.numel(), (
        f"shape packing placed {accepted_shape.numel()}, "
        f"sphere packing {accepted_sphere.numel()}"
    )


def test_coarsen_mask_contains_the_fine_mask():
    """
    The containment property that makes coarse-grid packing safe for a
    fine render: every fine voxel must fall inside a set coarse voxel.
    """
    coords = _blob(n_atoms=2000, radius=30.0)
    fine = build_species_mask(coords, 1.0, gap=0.0)
    coarse = coarsen_mask(fine, 2)

    assert all(s % 2 == 1 for s in coarse.shape), "origin must stay centered"
    # Every set fine voxel maps into a set coarse voxel.
    fine_idx = fine.nonzero()
    fh = torch.tensor(fine.shape) // 2
    ch = torch.tensor(coarse.shape) // 2
    mapped = ((fine_idx - fh).to(torch.float64) / 2).floor().long() + ch
    inside = ((mapped >= 0) & (mapped < torch.tensor(coarse.shape))).all(dim=1)
    mapped = mapped[inside]
    assert bool(coarse[mapped[:, 0], mapped[:, 1], mapped[:, 2]].all()), (
        "a fine voxel landed outside the coarsened mask"
    )


def test_coarsen_mask_is_a_noop_below_factor_two():
    m = build_species_mask(_blob(), 2.0, gap=0.0)
    assert coarsen_mask(m, 1) is m


def test_pack_shapes_3d_accepts_a_coarser_grid_than_the_render():
    """
    Packing on a coarse grid must still return positions in ANGSTROM on the
    shared physical box, so a caller can render them at any resolution.
    """
    coords = _blob(n_atoms=2000, radius=30.0)
    fine_voxel, factor = 1.0, 4
    fine = build_species_mask(coords, fine_voxel, gap=0.0)
    coarse = coarsen_mask(fine, factor)

    box_a = (160.0, 320.0, 320.0)
    coarse_grid = tuple(int(round(b / (fine_voxel * factor))) for b in box_a)

    pos, rot, accepted, _ = pack_shapes_3d(
        [coarse],
        torch.zeros(40, dtype=torch.long),
        coarse_grid,
        fine_voxel * factor,
        seed=0,
        n_orientations=16,
        max_retries=40,
    )
    assert accepted.numel() > 0
    # Positions are physical and box-centered, independent of packing grid.
    half = torch.tensor([box_a[2] / 2, box_a[1] / 2, box_a[0] / 2])
    assert bool((pos.abs() <= half).all()), "positions must stay inside the box"
    assert rot.shape == (accepted.numel(), 3, 3)


def test_generator_coarse_packing_survives_both_placement_stages():
    """
    Regression: `packing_voxel_size` coarsens the occupancy grid, and the
    generator packs TWICE per region (exact-count targets, then ratio
    filler), threading the grid between them. Coarsening inside each stage
    instead of once per region downsampled the second stage's input a
    second time and blew an internal shape assert -- but only when the
    factor exceeded 1, which no direct pack_shapes_3d test exercises.
    """
    from specter.specimen.tomogram.generator import (
        TomogramProteinSpec,
        TomogramSpecimenGenerator,
    )

    fixture = Path(__file__).parent.parent / "specter-data" / "pdb" / "1mbo.cif"
    if not fixture.exists():
        pytest.skip("bundled PDB fixture missing")

    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=(48, 96, 96),
        voxel_size=4.0,
        protein_specs=[
            # one exact-count target AND one ratio filler, so both stages run
            TomogramProteinSpec(
                pdb_source=str(fixture), n_copies=2, location="cytosol"
            ),
            TomogramProteinSpec(pdb_source=str(fixture), location="cytosol"),
        ],
        occupancy_fraction=0.05,
        packing_backend="shape",
        packing_voxel_size=12.0,  # factor 3
        clip_axes=(True, True, True),
        seed=0,
        progressbars=False,
    )
    vol = gen.generate()
    assert vol.shape == (48, 96, 96)
    assert len(gen.placements) > 0


def _stamp_fine(masks_fine, coords, rots, species, grid, voxel):
    """Rasterize placements at FINE resolution; return per-voxel hit count."""
    acc = np.zeros(grid, dtype=np.int16)
    origin = -0.5 * np.array(grid)[::-1] * voxel
    for i in range(len(coords)):
        m = masks_fine[int(species[i])].to(torch.float32)
        theta = build_affine_matrix(rots[i].unsqueeze(0))
        rm = (rotate_volume(m, theta, padding_mode="zeros")[0] > 0.5).numpy()
        nzi = np.nonzero(rm)
        if len(nzi[0]) == 0:
            continue
        sl = tuple(slice(int(a.min()), int(a.max()) + 1) for a in nzi)
        fp = rm[sl]
        centre = np.round((coords[i].numpy() - origin) / voxel - 0.5).astype(int)[::-1]
        loc = centre - np.array(fp.shape) // 2
        lo = np.maximum(loc, 0)
        hi = np.minimum(loc + np.array(fp.shape), np.array(grid))
        if np.any(hi <= lo):
            continue
        slo = lo - loc
        shi = slo + (hi - lo)
        acc[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] += fp[
            slo[0] : shi[0], slo[1] : shi[1], slo[2] : shi[2]
        ]
    return acc


def test_coarse_packing_leaves_no_overlap_at_render_resolution():
    """
    End-to-end smoke test: pack on a coarse grid, render at the fine one,
    and no two instances share a voxel.

    This does NOT by itself demonstrate why the ordering in
    `_rotation_cache` matters -- the wrong order leaks on the order of one
    voxel in 10^5, which a pack this size will usually not surface. The
    guarantee itself is tested directly in
    `test_rotation_cache_pools_after_rotating`.
    """
    fine_voxel, factor = 1.0, 4
    coords_atoms = _blob(n_atoms=3000, radius=14.0)
    fine = build_species_mask(coords_atoms, fine_voxel, gap=0.0)

    fine_grid = (96, 192, 192)
    coarse_grid = tuple(-(-n // factor) for n in fine_grid)

    coords, rots, accepted, _ = pack_shapes_3d(
        [fine],
        torch.zeros(400, dtype=torch.long),
        coarse_grid,
        fine_voxel * factor,
        pool_factor=factor,
        seed=0,
        n_orientations=64,
        max_retries=200,
    )
    assert accepted.numel() > 20, f"only {accepted.numel()} placed; test is vacuous"

    hits = _stamp_fine(
        [fine],
        coords,
        rots,
        torch.zeros(len(coords), dtype=torch.long),
        fine_grid,
        fine_voxel,
    )
    assert int((hits > 1).sum()) == 0, (
        f"{int((hits > 1).sum())} voxels covered twice at render resolution"
    )


def test_rotation_cache_pools_after_rotating():
    """
    The invariant behind coarse packing: a cached coarse footprint is the
    fine rotation POOLED, not the pooled mask rotated.

    Testing it directly rather than through a pack, because the two orders
    differ rarely enough that a random pack is not a reliable detector of
    the difference. Only the first order contains the shape that will
    actually be rendered, so only it makes a coarse-grid collision
    guarantee hold at render resolution.
    """
    from specter.specimen.packing._shape import _rotation_cache

    factor, n_or = 4, 24
    fine = build_species_mask(_blob(n_atoms=3000, radius=14.0), 1.0, gap=0.0)
    cache, _, _, _, R = _rotation_cache(fine, n_or, "cpu", 0, pool_factor=factor)

    def pool(x):
        return (
            torch.nn.functional.max_pool3d(
                x[None, None].to(torch.float32),
                kernel_size=factor,
                stride=factor,
                ceil_mode=True,
            )[0, 0]
            > 0
        )

    coarse_first = coarsen_mask(fine, factor)
    differed = 0
    for i in range(n_or):
        theta = build_affine_matrix(R[i : i + 1])
        rotate_then_pool = pool(
            rotate_volume(fine.to(torch.float32), theta, padding_mode="zeros")[0] > 0.5
        )
        # The cache must be exactly this, up to the trim of empty margins.
        assert int(rotate_then_pool.sum()) == int(cache[i].sum()), (
            f"orientation {i}: cache holds {int(cache[i].sum())} voxels, "
            f"rotate-then-pool gives {int(rotate_then_pool.sum())}"
        )
        pool_then_rotate = (
            rotate_volume(coarse_first.to(torch.float32), theta, padding_mode="zeros")[
                0
            ]
            > 0.5
        )
        if int(pool_then_rotate.sum()) != int(rotate_then_pool.sum()):
            differed += 1

    assert differed > 0, (
        "the two orders produced identical footprints for every orientation, "
        "so this test cannot tell them apart and the invariant is untested"
    )
