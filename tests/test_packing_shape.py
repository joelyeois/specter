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

import pytest
import torch

from specter.rotations._volume import build_affine_matrix, rotate_volume
from specter.specimen.packing import (
    build_species_mask,
    coarsen_mask,
    pack_shapes_3d,
)
from conftest import blob


def _rod(n_atoms: int = 400, half_len: float = 45.0, seed: int = 1) -> torch.Tensor:
    """An elongated cloud -- the case a bounding sphere handles worst."""
    g = torch.Generator().manual_seed(seed)
    z = (torch.rand(n_atoms, 1, generator=g) * 2 - 1) * half_len
    xy = torch.randn(n_atoms, 2, generator=g) * 4.0
    coords = torch.cat([xy, z], dim=1)
    return coords - coords.mean(0)


def test_build_species_mask_encloses_atoms_and_is_odd_sized():
    coords = blob()
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
    coords = blob()
    voxel = 6.8
    # 1.9 + 0.0 and 1.9 + 2.0 are both under one 6.8 A voxel, so identical.
    m0 = build_species_mask(coords, voxel, gap=0.0)
    m2 = build_species_mask(coords, voxel, gap=2.0)
    assert m0.shape == m2.shape
    assert bool((m0 == m2).all())

    # 1.9 + 5.0 crosses a voxel, so it must actually grow.
    m5 = build_species_mask(coords, voxel, gap=5.0)
    assert int(m5.sum()) > int(m0.sum())


def _lopsided(seed: int = 0) -> torch.Tensor:
    """
    An asymmetric cloud: an elongated body with a lobe off to one side.

    Its footprint's bounding box is centred well away from its origin, so a
    placement recorded at the footprint's middle rather than at the origin
    shows up as whole voxels of offset, not just rounding.
    """
    g = torch.Generator().manual_seed(seed)
    body = torch.randn(300, 3, generator=g) * torch.tensor([3.0, 4.0, 14.0])
    lobe = torch.randn(120, 3, generator=g) * 3.0 + torch.tensor([9.0, 0.0, 12.0])
    coords = torch.cat([body, lobe])
    return coords - coords.mean(0)


def _render_masks(masks, coords, rots, species, canvas_shape, voxel):
    """
    Draw each placement's footprint mask the way the renderer draws a copy.

    `fold_subvoxel_offsets` + `rotate_volume` + `insert_particles_into_micrograph`
    is `_insert_rotated_copies`' own sequence, with the mask standing in for
    the potential template, so the result is the footprint the canvas
    actually receives at the recorded position and rotation. Thresholded at
    0.5 like `_rotation_cache`, one copy at a time so neighbouring copies'
    interpolation tails cannot add up past it. Returns the per-voxel count.
    """
    from specter.crowding import fold_subvoxel_offsets, insert_particles_into_micrograph

    hits = torch.zeros(canvas_shape, dtype=torch.int16)
    for i in range(len(coords)):
        mask = masks[int(species[i])].to(torch.float32)
        theta = build_affine_matrix(rots[i].unsqueeze(0))
        theta, whole = fold_subvoxel_offsets(
            theta, coords[i : i + 1], voxel, mask.shape
        )
        rotated = rotate_volume(mask, theta, padding_mode="zeros")
        copy = insert_particles_into_micrograph(
            rotated, whole, pixel_size=voxel, micro_shape=canvas_shape
        )
        hits += (copy > 0.5).to(torch.int16)
    return hits


@pytest.mark.parametrize(
    "canvas_shape,factor",
    [
        ((32, 40, 40), 1),  # even
        ((33, 39, 41), 1),  # odd
        ((32, 40, 40), 2),  # coarse packing grid, even
        ((33, 39, 39), 3),  # coarse, odd
        ((33, 39, 41), 2),  # coarse grid overhanging the canvas
    ],
)
def test_rendered_copies_reproduce_the_packers_occupancy_grid(canvas_shape, factor):
    """
    Render every placement through the renderer's own functions, at the
    returned position and rotation, and require the result to be EXACTLY
    the occupancy the packer marked (after pooling onto the packing grid).

    This is the property that makes packing mean anything: the copy is drawn
    where its footprint was collision-tested. Until 2026-10-16 positions were
    the packing voxel's centre, ``-extent / 2 + (idx + 0.5) * voxel``, on a
    canvas that puts voxel ``k`` at ``(k - n // 2) * voxel``, and the voxel
    was the middle of the trimmed footprint rather than the molecule origin
    -- half a voxel on every even axis, plus up to several voxels for an
    asymmetric molecule.
    """
    voxel = 2.0
    grid = tuple(-(-n // factor) for n in canvas_shape)
    mask = build_species_mask(_lopsided(), voxel, gap=0.0)
    torch.manual_seed(0)  # the rotation cache draws from the global stream
    coords, rots, accepted, occ = pack_shapes_3d(
        [mask],
        torch.zeros(30, dtype=torch.long),
        grid,
        voxel * factor,
        pool_factor=factor,
        canvas_shape=canvas_shape,
        seed=1,
        n_orientations=16,
        max_retries=60,
    )
    assert accepted.numel() >= 3, "too few placements for the test to bite"
    # every position is a whole render voxel of the canvas
    assert torch.equal(coords, torch.round(coords / voxel) * voxel)

    hits = _render_masks([mask], coords, rots, accepted * 0, canvas_shape, voxel)
    assert int((hits > 1).sum()) == 0, "rendered copies overlap"
    rendered = (
        torch.nn.functional.max_pool3d(
            (hits > 0)[None, None].to(torch.float32),
            kernel_size=factor,
            stride=factor,
            ceil_mode=True,
        )[0, 0]
        > 0
    )
    assert int((rendered & ~occ).sum()) == 0, "a copy covers voxels never tested"
    missing = occ & ~rendered
    # The only marked voxels a copy may fail to cover are those whose render
    # voxels lie past the canvas (a coarse grid that overhangs it), where
    # insertion clips.
    overhang = torch.zeros(grid, dtype=torch.bool)
    for axis, n in enumerate(canvas_shape):
        if n % factor:
            sl = [slice(None)] * 3
            sl[axis] = slice(grid[axis] - 1, grid[axis])
            overhang[tuple(sl)] = True
    assert int((missing & ~overhang).sum()) == 0, (
        f"{int((missing & ~overhang).sum())} tested voxels are not where the "
        "copy was rendered"
    )


def test_pack_shapes_3d_respects_a_seeded_occupancy_grid():
    voxel = 4.0
    masks = [build_species_mask(blob(), voxel, gap=0.0)]
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
    coords = blob(n_atoms=2000, radius=30.0)
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


def test_generator_coarse_packing_survives_both_placement_stages():
    """
    Regression: `packing_voxel_size` coarsens the occupancy grid, and the
    generator packs TWICE per region (exact-count targets, then ratio
    filler), threading the grid between them. Coarsening inside each stage
    instead of once per region downsampled the second stage's input a
    second time and blew an internal shape assert -- but only when the
    factor exceeded 1, which no direct pack_shapes_3d test exercises.
    """
    from specter.specimen.tomogram import (
        TomogramProteinSpec,
        TomogramSpecimenGenerator,
    )

    fixture = Path(__file__).parent / "test_data" / "1mbo.cif"
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
        packing_voxel_size=12.0,  # factor 3
        clip_axes=(True, True, True),
        seed=0,
        progressbars=False,
    )
    vol = gen.generate()
    assert vol.shape == (48, 96, 96)
    assert len(gen.placements) > 0


@pytest.mark.parametrize(
    "target_shape,packing_voxel_size",
    [
        ((40, 48, 48), None),  # even, packed at render resolution
        ((40, 48, 48), 8.0),  # even, packed on a 2x coarser grid
        ((41, 49, 49), 8.0),  # odd, and a coarse grid overhanging the canvas
    ],
)
def test_generator_renders_a_copy_on_the_footprint_it_tested(
    target_shape, packing_voxel_size, monkeypatch
):
    """
    One copy of an asymmetric structure (myoglobin), packed and rendered by
    `TomogramSpecimenGenerator` itself: the voxels the packer marked as
    taken must line up with the voxels the rendered copy occupies.

    The two are different shapes -- the packer's footprint is the atoms
    grown by their van der Waals radius, the rendered label is the potential
    above 1% of its peak -- so they are compared by alignment: of every
    integer shift of the rendered footprint, the best overlap must be at
    zero. Before 2026-10-16 it never was in an even box (best shifts of one
    to two voxels, on the order of 10 A at 4 A/voxel), and a copy then
    rendered partly outside the space it had been granted.
    """
    import specter.specimen.tomogram._generator as generator_module
    from specter.specimen.tomogram import (
        TomogramProteinSpec,
        TomogramSpecimenGenerator,
    )

    fixture = Path(__file__).parent / "test_data" / "1mbo.cif"
    if not fixture.exists():
        pytest.skip("bundled PDB fixture missing")

    marked: list[torch.Tensor] = []
    real_pack = generator_module.pack_shapes_3d

    def recording_pack(*args, **kwargs):
        before = kwargs["occupancy"].clone()
        out = real_pack(*args, **kwargs)
        marked.append(out[3] & ~before)
        return out

    monkeypatch.setattr(generator_module, "pack_shapes_3d", recording_pack)

    voxel = 4.0
    gen = TomogramSpecimenGenerator(
        membrane_instances=[],
        target_shape=target_shape,
        voxel_size=voxel,
        protein_specs=[
            TomogramProteinSpec(pdb_source=str(fixture), n_copies=1, location="cytosol")
        ],
        packing_voxel_size=packing_voxel_size,
        parameterization="kirkland",
        seed=0,
        progressbars=False,
    )
    gen.generate()
    assert len(gen.placements) == 1 and len(marked) == 1
    tested = marked[0]

    factor = 1 if packing_voxel_size is None else round(packing_voxel_size / voxel)
    rendered = gen.instance_labels > 0
    rendered = (
        torch.nn.functional.max_pool3d(
            rendered[None, None].to(torch.float32),
            kernel_size=factor,
            stride=factor,
            ceil_mode=True,
        )[0, 0]
        > 0
    )
    assert rendered.shape == tested.shape

    def overlap(shift: tuple[int, int, int]) -> float:
        moved = torch.roll(rendered, shift, dims=(0, 1, 2))
        return float((moved & tested).sum()) / float((moved | tested).sum())

    shifts = [
        (dz, dy, dx)
        for dz in (-3, -2, -1, 0, 1, 2, 3)
        for dy in (-3, -2, -1, 0, 1, 2, 3)
        for dx in (-3, -2, -1, 0, 1, 2, 3)
    ]
    best = max(shifts, key=overlap)
    assert best == (0, 0, 0), (
        f"the rendered copy lines up best with the tested footprint shifted "
        f"by {best} packing voxels (overlap {overlap(best):.3f}, against "
        f"{overlap((0, 0, 0)):.3f} unshifted)"
    )
    position = gen.placements[0].position_xyz
    assert torch.equal(position, torch.round(position / voxel) * voxel), (
        "a packed position must be a whole render voxel"
    )


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
    coords_atoms = blob(n_atoms=3000, radius=14.0)
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

    hits = _render_masks(
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
    fine = build_species_mask(blob(n_atoms=3000, radius=14.0), 1.0, gap=0.0)
    cache, _, _, _, _, _, R = _rotation_cache(fine, n_or, "cpu", 0, pool_factor=factor)

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
