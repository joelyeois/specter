"""
Tests for specter.coords.
"""

from __future__ import annotations

import torch

from specter.coords import (
    poisson_disk_neighbors,
    poisson_disk_neighbors_3d,
    radial_distribution_function,
)


def test_poisson_disk_neighbors_respects_min_distance():
    torch.manual_seed(0)
    min_distance = 10.0
    pts = poisson_disk_neighbors(min_distance, box=(128, 128))

    assert pts.shape[0] > 1
    assert pts.shape[1] == 2
    dists = torch.cdist(pts, pts)
    dists.fill_diagonal_(torch.inf)
    assert bool((dists.min(dim=1).values >= min_distance - 1e-4).all())


def test_poisson_disk_neighbors_stays_in_box():
    torch.manual_seed(1)
    box = (64, 96)
    pts = poisson_disk_neighbors(8.0, box=box, seed="random")
    y, x = pts[:, 0], pts[:, 1]
    assert bool((y >= -box[0] // 2).all() and (y < box[0] // 2).all())
    assert bool((x >= -box[1] // 2).all() and (x < box[1] // 2).all())


def test_poisson_disk_neighbors_3d_respects_min_distance():
    torch.manual_seed(2)
    min_distance = 15.0
    pts = poisson_disk_neighbors_3d(min_distance, box=(80.0, 80.0, 80.0))

    assert pts.shape[0] > 1
    assert pts.shape[1] == 3
    dists = torch.cdist(pts, pts)
    dists.fill_diagonal_(torch.inf)
    assert bool((dists.min(dim=1).values >= min_distance - 1e-4).all())


def test_crowding_reexports_match_coords():
    from specter.crowding import (
        poisson_disk_neighbors as crowding_poisson_disk_neighbors,
    )
    from specter.crowding import (
        poisson_disk_neighbors_3d as crowding_poisson_disk_neighbors_3d,
    )

    assert crowding_poisson_disk_neighbors is poisson_disk_neighbors
    assert crowding_poisson_disk_neighbors_3d is poisson_disk_neighbors_3d


def test_radial_distribution_function_smoke():
    torch.manual_seed(0)
    coords = (torch.rand(200, 3) - 0.5) * 40.0
    r, g_r = radial_distribution_function(coords, volume=40.0**3, dr=1.0)
    assert r.shape == g_r.shape
    assert torch.isfinite(g_r).all()
