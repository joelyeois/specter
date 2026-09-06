"""
Tests for the occupancy estimator and its use in ice blending.

Occupancy is read off the potential, but only after coarse-graining to the
water probe's own length scale. Unblurred, clamping ``1 - V/7`` finds the
boundary of every ATOM instead of the boundary of the MOLECULE and admits
bulk ice into every interatomic gap -- 0.597 of full strength inside 1A6M
at 1 A/voxel -- and how wrong it goes is set by the render grid, running
0.36 to 0.87 of the molecular volume over 0.75 to 4 A. These tests pin the
properties that failure has and this one must not.
"""

from __future__ import annotations

import pytest
import torch


class TestPotentialOccupancy:
    """
    The blurred estimator: the fallback wherever a specimen's geometry is
    not known. It has to be grid-agnostic, since that is the whole reason
    the coarse-graining length is given in Angstrom rather than voxels.
    """

    @staticmethod
    def _protein_like(n, dx, seed=0):
        """A blob of atoms dense enough to read as solid protein."""
        g = torch.Generator().manual_seed(seed)
        coords = torch.randn(4000, 3, generator=g) * (n * dx / 10.0)
        return coords, torch.full((4000,), 6)

    def test_blur_is_skipped_but_harmless_at_coarse_voxels(self):
        """
        At 10 A a 2 A sigma is a fifth of a voxel and the convolution is
        skipped. That is correct, not a degradation: the renderer's own
        voxel average has already coarse-grained past 2 A.
        """
        from specter.potential import occupancy_blur_halo_voxels, potential_occupancy

        V = torch.rand(8, 12, 12) * 14.0
        assert occupancy_blur_halo_voxels(10.0) == 0
        skipped = potential_occupancy(V, 10.0)
        raw = (V / 7.0).clamp(0, 1)
        torch.testing.assert_close(skipped, raw)

    def test_blur_runs_at_fine_voxels(self):
        from specter.potential import occupancy_blur_halo_voxels, potential_occupancy

        V = torch.zeros(16, 16, 16)
        V[8, 8, 8] = 100.0
        assert occupancy_blur_halo_voxels(1.0) > 0
        out = potential_occupancy(V, 1.0)
        # A lone spike spreads; a clamped raw read would not.
        assert float(out[8, 8, 9]) > 0.0
        assert float(out[8, 8, 8]) < 1.0

    def test_sigma_is_physical_not_per_voxel(self):
        """
        The same specimen rendered on two grids must give the same
        excluded volume. This is what the raw rule fails: it ran 0.36 to
        0.87 of the molecular volume over 0.75-4 A.
        """
        from specter.potential import potential_occupancy

        volumes = []
        for dx in (1.0, 2.0, 4.0):
            n = int(64 / dx)
            coords, Z = self._protein_like(n, dx)
            V = _render(coords, Z, n, dx)
            volumes.append(float(potential_occupancy(V, dx).sum()) * dx**3)
        spread = (max(volumes) - min(volumes)) / min(volumes)
        assert spread < 0.15, f"volumes {volumes}, spread {spread:.3f}"

    def test_rejects_a_nonpositive_pixel_size(self):
        from specter.potential import potential_occupancy

        with pytest.raises(ValueError, match="voxel_size"):
            potential_occupancy(torch.zeros(4, 4, 4), 0.0)

    def test_bounded_in_unit_interval(self):
        from specter.potential import potential_occupancy

        V = torch.randn(8, 8, 8) * 50.0
        out = potential_occupancy(V, 1.0)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def _render(coords, Z, n, dx):
    """Small helper: potential from coordinates on an (n,n,n) grid."""
    from specter.potential import PotentialBuilder

    pb = PotentialBuilder(
        n_xyz=(n, n, n),
        dx=dx,
        atomic_numbers=Z,
        parameterization="kirkland",
        progressbars=False,
    )
    with torch.no_grad():
        return pb(coords, method="analytic").squeeze()


def test_blend_slab_halo_matches_an_unchunked_blur():
    """
    `blend_ice_into_volume` evaluates the blurred field a z-slab at a
    time and must widen each slab by the blur's reach. Without the halo
    every chunk boundary prints into the ice as a seam.
    """
    from specter.ice import blend_ice_into_volume
    from specter.potential import potential_occupancy

    torch.manual_seed(0)
    n, nz = 16, 48
    V = torch.rand(1, nz, n, n) * 6.0
    maker = _RandomIce(n, nz)

    whole = (1.0 - potential_occupancy(V, 2.0)).clamp(0, 1)
    out = blend_ice_into_volume(V.clone(), maker, 2.0)
    added = out - V
    # Recover the weight the blend actually used, where ice is nonzero.
    ice = added / whole.clamp(min=1e-6)
    assert torch.isfinite(ice).all()
    # A seam would show as a z-profile discontinuity in the applied weight.
    prof = added.mean(dim=(0, 2, 3))
    jumps = (prof[1:] - prof[:-1]).abs()
    assert float(jumps.max()) < 6 * float(jumps.median() + 1e-6)


def _RandomIce(n, nz):
    from specter.ice import RandomIcemaker

    return RandomIcemaker(dx=2.0, n=n, nz=nz, progressbars=False)
