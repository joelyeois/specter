"""
Tests for specter.coords.
"""

from __future__ import annotations

import math

import pytest
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


def test_radial_distribution_function_normalisation_is_parameter_free():
    """A particle sees N-1 others, whatever the density or bin width.

    Integrating rho*g(r)*4*pi*r^2 dr over every separation present counts the
    neighbours of one particle, so it must come back as N-1 with nothing
    fitted. r_max has to reach the box diagonal or the far pairs fall outside
    the last bin and the count comes up short. This is what pins the factor
    of 2 between unordered pair counting and the ordered-pair definition:
    without it the integral returns (N-1)/2 and an ideal gas plateaus at 0.5.
    """
    torch.manual_seed(0)
    n_points, side = 4000, 100.0
    coords = (torch.rand(n_points, 3) - 0.5) * side
    volume, dr = side**3, 0.5
    r, g_r = radial_distribution_function(
        coords, volume=volume, dr=dr, r_max=side * math.sqrt(3.0)
    )
    number_density = n_points / volume
    neighbours = float((g_r * number_density * 4 * math.pi * r**2 * dr).sum())
    assert neighbours == pytest.approx(n_points - 1, rel=1e-3)

    # And the ideal-gas plateau itself, at separations short enough that the
    # box surface has not yet eaten much of the shell.
    _, g_short = radial_distribution_function(coords, volume=volume, dr=1.0)
    assert torch.allclose(g_short[1:4], torch.ones(3), atol=0.05)


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


# ---------------------------------------------------------------------------
# 3-D Poisson disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["origin", "random"])
def test_poisson_disk_3d_respects_min_distance_and_is_deterministic(seed):
    torch.manual_seed(5)
    pts = poisson_disk_neighbors_3d(10.0, box=(60.0, 80.0, 70.0), seed=seed)
    assert len(pts) > 20
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))
    assert d.min() >= 10.0
    assert (pts.abs() <= torch.tensor([35.0, 40.0, 30.0])).all()
    torch.manual_seed(5)
    again = poisson_disk_neighbors_3d(10.0, box=(60.0, 80.0, 70.0), seed=seed)
    assert torch.equal(pts, again)
