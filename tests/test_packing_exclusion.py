"""
Tests for pack_hard_spheres_3d's exclusion_distance_field parameter -- the
mechanism used to keep placed spheres away from a pre-existing membrane
and/or confined to a specific region (e.g. a vesicle's lumen), by rejecting
any candidate whose sphere would come within `radius + gap` of a forbidden
voxel (see specter.specimen.tomogram for how region masks feed into this).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from specter.specimen.packing import pack_hard_spheres_3d


def _half_space_exclusion_field(
    box: tuple[float, float, float], n: int
) -> tuple[torch.Tensor, float]:
    """Forbid the negative-x half of a cubic `box`, return (distance_field,
    field_voxel_size). box is (D, H, W) = (z, y, x); field shape is (z, y, x)."""
    assert box[0] == box[1] == box[2], "cubic box assumed for this helper"
    field_voxel_size = box[2] / n
    forbidden = np.zeros((n, n, n), dtype=bool)
    forbidden[:, :, : n // 2] = True  # x < 0
    dist = ndimage.distance_transform_edt(~forbidden) * field_voxel_size
    return torch.from_numpy(dist).float(), field_voxel_size


def test_exclusion_distance_field_none_is_unchanged_behavior():
    box = (200.0, 200.0, 200.0)
    radii = torch.full((60,), 10.0)
    coords_a, idx_a = pack_hard_spheres_3d(radii, box, gap=2.0, seed=0, device="cpu")
    coords_b, idx_b = pack_hard_spheres_3d(
        radii,
        box,
        gap=2.0,
        seed=0,
        device="cpu",
        exclusion_distance_field=None,
        field_voxel_size=None,
    )
    assert torch.equal(coords_a, coords_b)
    assert torch.equal(idx_a, idx_b)


def test_exclusion_distance_field_confines_placements_to_allowed_half():
    box = (200.0, 200.0, 200.0)
    radii = torch.full((60,), 10.0)
    gap = 2.0
    # fine enough field resolution to keep interpolation bleed well under gap
    field, field_voxel_size = _half_space_exclusion_field(box, n=100)

    coords, idx = pack_hard_spheres_3d(
        radii,
        box,
        gap=gap,
        seed=0,
        device="cpu",
        exclusion_distance_field=field,
        field_voxel_size=field_voxel_size,
    )
    assert coords.shape[0] > 0
    r = radii[idx]
    # every placed sphere must clear x=0 (the forbidden/allowed boundary)
    assert bool((coords[:, 0] - r >= -1e-2).all())


def test_exclusion_distance_field_requires_both_or_neither_param():
    box = (100.0, 100.0, 100.0)
    radii = torch.full((5,), 10.0)
    field = torch.zeros(10, 10, 10)
    try:
        pack_hard_spheres_3d(radii, box, exclusion_distance_field=field)
        raised = False
    except ValueError:
        raised = True
    assert raised

    try:
        pack_hard_spheres_3d(radii, box, field_voxel_size=10.0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_exclusion_distance_field_fully_forbidden_places_nothing():
    box = (100.0, 100.0, 100.0)
    radii = torch.full((20,), 10.0)
    n = 20
    field_voxel_size = box[0] / n
    forbidden = np.ones((n, n, n), dtype=bool)
    dist = ndimage.distance_transform_edt(~forbidden) * field_voxel_size  # all zero
    field = torch.from_numpy(dist).float()

    coords, idx = pack_hard_spheres_3d(
        radii,
        box,
        gap=2.0,
        seed=0,
        device="cpu",
        exclusion_distance_field=field,
        field_voxel_size=field_voxel_size,
    )
    assert coords.shape[0] == 0


def test_sampling_mask_finds_a_needle_in_haystack_region_that_uniform_sampling_misses():
    """Regression test for the real failure this parameter fixes (found
    while building specimen.tomogram.TomogramSpecimenGenerator: packing
    into a small vesicle lumen placed 0/1 candidates because the valid
    region was ~0.008% of the box). Here: a 100A cube pocket, centered well
    inside a much larger box, with a small enough margin that only its
    innermost core (~6% of the pocket, ~0.0008% of the whole box) is a
    geometrically valid center for the sphere -- plain uniform box-wide
    sampling essentially never hits that within a normal max_passes budget
    (verified across several seeds), but mask-restricted sampling finds it
    reliably by only ever drawing candidates from inside the pocket."""
    box = (2000.0, 2000.0, 2000.0)
    radii = torch.tensor([30.0])
    gap = 2.0
    field_voxel_size = 10.0
    n = 200
    pocket = torch.zeros(n, n, n, dtype=torch.bool)
    pocket[95:105, 95:105, 95:105] = True  # 100A cube near the box center
    dist = (
        torch.from_numpy(ndimage.distance_transform_edt(pocket.numpy())).float()
        * field_voxel_size
    )

    coords_uniform, _ = pack_hard_spheres_3d(
        radii,
        box,
        gap=gap,
        seed=0,
        device="cpu",
        exclusion_distance_field=dist,
        field_voxel_size=field_voxel_size,
        max_passes=100,
        stall_patience=100,
    )
    coords_masked, _ = pack_hard_spheres_3d(
        radii,
        box,
        gap=gap,
        seed=0,
        device="cpu",
        exclusion_distance_field=dist,
        field_voxel_size=field_voxel_size,
        sampling_mask=pocket,
        max_passes=100,
        stall_patience=100,
    )
    assert coords_uniform.shape[0] == 0
    assert coords_masked.shape[0] == 1


def test_sampling_mask_raises_on_all_false_mask():
    box = (100.0, 100.0, 100.0)
    radii = torch.full((5,), 10.0)
    mask = torch.zeros(10, 10, 10, dtype=torch.bool)
    try:
        pack_hard_spheres_3d(radii, box, field_voxel_size=10.0, sampling_mask=mask)
        raised = False
    except ValueError:
        raised = True
    assert raised
