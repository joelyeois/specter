"""
A region can survive generate()'s own empty-region check at render resolution
and still have no free voxel on the coarser packing grid (coarsening marks a
packing voxel occupied if ANY fine voxel in it is, which swallows a thin
compartment). That used to reach pack_shapes_3d and raise, ending a 2 A build
of the shipped config in its last stage; `_pack_shapes` now warns and places
nothing, as the fine-grid path does.
"""

from __future__ import annotations

import warnings

import torch

from specter.specimen.tomogram import TomogramSpecimenGenerator


class _FakePDB:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.coordinates = torch.randn(80, 3, generator=g) * 6.0


def _bare_generator() -> TomogramSpecimenGenerator:
    gen = object.__new__(TomogramSpecimenGenerator)
    gen._mask_cache = {}
    gen.gap = 0.0
    gen.packing_max_retries = 20
    gen.n_orientations = 8
    gen.seed = 0
    gen.device = "cpu"
    gen.clip_axes = (True, True, True)
    return gen


def test_pack_shapes_places_nothing_when_the_packing_grid_is_full():
    gen = _bare_generator()
    pack_shape = (8, 8, 8)
    occupancy = torch.ones(pack_shape, dtype=torch.bool)  # every packing voxel taken
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coords, rots, accepted, occ = gen._pack_shapes(
            [_FakePDB()],
            torch.zeros(5, dtype=torch.long),
            pack_shape,
            8.0,
            2,
            occupancy,
        )
    assert coords.shape == (0, 3) and rots.shape == (0, 3, 3) and accepted.numel() == 0
    assert occ is occupancy
    assert any("no free voxel" in str(w.message) for w in caught)


def test_pack_shapes_still_packs_a_region_with_room():
    gen = _bare_generator()
    pack_shape = (12, 12, 12)
    occupancy = torch.zeros(pack_shape, dtype=torch.bool)
    coords, rots, accepted, occ = gen._pack_shapes(
        [_FakePDB()], torch.zeros(6, dtype=torch.long), pack_shape, 8.0, 2, occupancy
    )
    assert coords.shape[0] > 0 and occ.any()
