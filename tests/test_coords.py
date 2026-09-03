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


def test_poisson_disk_neighbors_3d_respects_min_distance():
    """The spacing guarantee, asserted for the 3D sampler only.

    The 2D sampler tests each candidate against ``torch.stack(pts)``, rebuilt
    from the accepted list every iteration, so asserting the spacing of what
    it returns just re-runs its own acceptance rule and cannot fail. The 3D
    sampler instead tests against ``pts_t``, an incrementally grown *copy* of
    that list (a rewrite for speed -- the per-candidate grid scan it replaced
    cost 0.18 s for eight points). Appending to the list and concatenating to
    the copy are two separate statements, and if they ever fall out of step
    the distance test runs against a stale set while the returned points
    silently overlap. That is what this covers.
    """
    torch.manual_seed(2)
    min_distance = 15.0
    pts = poisson_disk_neighbors_3d(min_distance, box=(80.0, 80.0, 80.0))

    assert pts.shape[0] > 1
    assert pts.shape[1] == 3
    dists = torch.cdist(pts, pts)
    dists.fill_diagonal_(torch.inf)
    assert bool((dists.min(dim=1).values >= min_distance - 1e-4).all())


def test_poisson_disk_samplers_respect_a_non_cubic_box():
    """Box extents are given in the reverse order of the columns returned:
    the 2D sampler takes (H, W) and returns (y, x), the 3D one takes
    (D, H, W) and returns (x, y, z). An equal-sided box cannot tell a
    transposed extent from a correct one, so give every axis its own length.
    """
    torch.manual_seed(1)
    H, W = 64, 96
    pts = poisson_disk_neighbors(8.0, box=(H, W), seed="random")
    y, x = pts[:, 0], pts[:, 1]
    assert bool((y >= -H // 2).all() and (y < H // 2).all())
    assert bool((x >= -W // 2).all() and (x < W // 2).all())

    torch.manual_seed(3)
    depth, height, width = 40.0, 120.0, 80.0
    pts3 = poisson_disk_neighbors_3d(12.0, box=(depth, height, width))
    assert pts3.shape[0] > 1
    for column, extent in enumerate((width, height, depth)):
        assert bool((pts3[:, column].abs() <= extent / 2).all())
        # Each axis is filled out to its own bound, so the bound above is not
        # satisfied merely by points huddling near the origin.
        assert float(pts3[:, column].abs().max()) > 0.4 * extent


def test_radial_distribution_function_modes_agree():
    """The chunked and sampled paths against the plain ``torch.pdist`` one.

    Both are reachable from ``plots.plot_rdf`` and ``MDSimDump``, and neither
    was covered. Chunking only changes how the pair list is blocked, so it
    has to reproduce the unchunked histogram exactly -- including for a block
    size that does not divide N, and one larger than N, where the triangular
    within-block and rectangular tail terms degenerate. The sampled path
    draws random pairs instead and rescales by the total pair count, so it
    agrees only to its own 1/sqrt(n_samples) noise.
    """
    torch.manual_seed(0)
    coords = (torch.rand(257, 3) - 0.5) * 40.0
    volume = 40.0**3
    r, exact = radial_distribution_function(coords, volume=volume, dr=1.0)
    assert r.shape == exact.shape
    assert torch.isfinite(exact).all()
    assert float(exact.max()) > 0.0

    for chunk_size in (1, 100, 256, 257, 1000):
        _, chunked = radial_distribution_function(
            coords, volume=volume, dr=1.0, chunk_size=chunk_size
        )
        assert torch.equal(chunked, exact)

    torch.manual_seed(7)
    _, sampled = radial_distribution_function(
        coords, volume=volume, dr=1.0, approximate=True, n_samples=2_000_000
    )
    near = r < 12.0
    assert torch.allclose(sampled[near], exact[near], atol=0.1)
