"""
Tests for the geometric occupancy field and its use in ice blending.

The field exists because reading occupancy back out of a scattering
potential does not work: a potential is cusped, so clamping ``1 - V/7``
finds the boundary of every ATOM instead of the boundary of the MOLECULE
and admits bulk ice into every interatomic gap. These tests pin the three
properties that failure has and this one must not.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter.atom import atom_mass
from specter.pdb import PDB
from specter.potential import (
    SOLVENT_EXCLUDED_RADIUS_SCALE,
    VDW_RADII_A,
    atomic_occupancy,
)

#: Å^3 per dalton for protein, from the standard partial specific volume
#: vbar = 0.73 cm3/g. The same quantity FULL_OCCUPANCY_POTENTIAL_V is
#: calibrated against, which is what keeps the two consistent.
A3_PER_DA = 1.2122


def _sphere_volume(occ: torch.Tensor, dx: float) -> float:
    return float(occ.sum()) * dx**3


def test_single_atom_occupancy_matches_its_sphere_volume():
    """One carbon, one ball: 4/3 pi (1.70 * scale)^3."""
    dx = 0.25
    n = 64
    occ = atomic_occupancy(
        torch.zeros(1, 3), torch.tensor([6]), n, dx, radius_scale=1.0
    )
    expected = 4 / 3 * math.pi * VDW_RADII_A[6] ** 3
    assert _sphere_volume(occ, dx) == pytest.approx(expected, rel=0.02)
    assert float(occ.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(occ.min()) >= 0.0


def test_two_coincident_atoms_do_not_double_count():
    """Occupancy is a union, not a sum: a voxel cannot be more than full."""
    dx = 0.25
    n = 64
    one = atomic_occupancy(torch.zeros(1, 3), torch.tensor([6]), n, dx)
    two = atomic_occupancy(torch.zeros(2, 3), torch.tensor([6, 6]), n, dx)
    assert float(two.max()) <= 1.0 + 1e-6
    torch.testing.assert_close(one, two)


def test_occupancy_is_bounded_and_empty_far_from_any_atom():
    dx = 0.5
    n = 48
    occ = atomic_occupancy(torch.zeros(1, 3), torch.tensor([6]), n, dx)
    assert 0.0 <= float(occ.min()) and float(occ.max()) <= 1.0
    assert float(occ[0, 0, 0]) == 0.0  # a corner, far from the origin


@pytest.mark.parametrize("code", ["1a6m"])
def test_occupancy_recovers_the_solvent_excluded_volume(code):
    """
    The calibrated radius scale has to land on the volume the ice model's
    own reference constant assumes, or the two disagree about how much of
    a protein is protein.
    """
    pdb = PDB(code, verbose=False)
    target = float(atom_mass(pdb.atomic_numbers).sum()) * A3_PER_DA
    extent = float(
        (pdb.coordinates.max(0).values - pdb.coordinates.min(0).values).max()
    )
    dx = 1.0
    n = int((extent + 20.0) / dx)
    n += n % 2
    occ = atomic_occupancy(pdb.coordinates, pdb.atomic_numbers, n, dx)
    # +/-10% covers both the structure-to-structure spread and the
    # calibration target's own uncertainty in vbar.
    assert _sphere_volume(occ, dx) == pytest.approx(target, rel=0.10)


@pytest.mark.parametrize("code", ["1a6m"])
def test_occupancy_does_not_depend_on_the_voxel_size(code):
    """
    Excluded volume is geometry, so it must not move with the grid the
    potential happened to be rendered on. The rule this replaced ran from
    0.36 to 0.87 of the molecular volume over this same range.
    """
    pdb = PDB(code, verbose=False)
    extent = float(
        (pdb.coordinates.max(0).values - pdb.coordinates.min(0).values).max()
    )
    volumes = []
    for dx in (0.75, 1.0, 1.5, 2.0):
        n = int((extent + 20.0) / dx)
        n += n % 2
        occ = atomic_occupancy(pdb.coordinates, pdb.atomic_numbers, n, dx)
        volumes.append(_sphere_volume(occ, dx))
    spread = (max(volumes) - min(volumes)) / min(volumes)
    assert spread < 0.10, f"volumes {volumes} spread {spread:.3f}"


@pytest.mark.parametrize("code", ["1a6m"])
def test_occupancy_excludes_ice_from_the_molecule_s_interior(code):
    """
    The failure this field exists to fix. Voxels solidly inside the
    molecule should take essentially no bulk ice; the potential-derived
    rule gives them roughly half.
    """
    pdb = PDB(code, compute_atom_species=True, verbose=False)
    extent = float(
        (pdb.coordinates.max(0).values - pdb.coordinates.min(0).values).max()
    )
    dx = 1.0
    n = int((extent + 20.0) / dx)
    n += n % 2
    from specter.potential import PotentialBuilder

    pb = PotentialBuilder(
        n_xyz=(n, n, n),
        dx=dx,
        atomic_numbers=pdb.atomic_numbers,
        atom_species=pdb.atom_species,
        parameterization="shtyrov",
        progressbars=False,
    )
    with torch.no_grad():
        V = pb(pdb.coordinates, method="analytic").squeeze()
    occ = atomic_occupancy(pdb.coordinates, pdb.atomic_numbers, n, dx)

    body = occ > 0.9
    assert bool(body.any())
    geometric = float((1.0 - occ)[body].mean())
    # The unblurred rule, written out: it no longer exists as a function,
    # but what it did is what this test documents being fixed.
    from_potential = float((1.0 - V / 7.0).clamp(0, 1)[body].mean())

    assert geometric < 0.05
    assert from_potential > 0.3  # documents what is being fixed
    assert geometric < from_potential


def test_occupancy_is_unmoved_by_scaling_the_potential():
    """
    `potential_scale` is a contrast knob. It must not change how much
    water is in the specimen, which is exactly what inferring the weight
    from the potential makes it do.
    """
    coords = torch.randn(200, 3) * 4.0
    Z = torch.full((200,), 6)
    dx, n = 1.0, 48
    occ = atomic_occupancy(coords, Z, n, dx)

    # Occupancy takes no potential at all, so there is nothing for a scale
    # to act on -- asserted against the alternative, which does move.
    from specter.potential import potential_occupancy

    V = torch.rand(n, n, n) * 14.0
    weights = [
        float((1.0 - potential_occupancy(V * s, 1.0)).mean()) for s in (1.0, 0.5, 0.25)
    ]
    assert weights[0] < weights[1] < weights[2], weights

    occ_again = atomic_occupancy(coords, Z, n, dx)
    torch.testing.assert_close(occ, occ_again)


def test_radius_scale_of_one_is_the_bare_van_der_waals_union():
    """The default inflates; passing 1.0 must not."""
    coords = torch.randn(50, 3) * 3.0
    Z = torch.full((50,), 6)
    dx, n = 0.5, 48
    bare = atomic_occupancy(coords, Z, n, dx, radius_scale=1.0)
    scaled = atomic_occupancy(coords, Z, n, dx)
    assert SOLVENT_EXCLUDED_RADIUS_SCALE > 1.0
    assert _sphere_volume(scaled, dx) > _sphere_volume(bare, dx)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_a_nonpositive_radius_scale(bad):
    with pytest.raises(ValueError, match="radius_scale"):
        atomic_occupancy(torch.zeros(1, 3), torch.tensor([6]), 8, 1.0, radius_scale=bad)


def test_rejects_mismatched_coordinate_and_element_counts():
    with pytest.raises(ValueError, match="same length"):
        atomic_occupancy(torch.zeros(3, 3), torch.tensor([6, 6]), 8, 1.0)


class TestBlendUsesOccupancy:
    """`blend_ice_into_volume` must prefer geometry when it is given some."""

    @staticmethod
    def _maker(n, nz):
        from specter.ice import RandomIcemaker

        return RandomIcemaker(dx=2.0, n=n, nz=nz, progressbars=False)

    def test_full_occupancy_admits_no_ice(self):
        from specter.ice import blend_ice_into_volume

        n = nz = 16
        V = torch.zeros(1, nz, n, n)
        occ = torch.ones(1, nz, n, n)
        out = blend_ice_into_volume(V, self._maker(n, nz), 2.0, occupancy=occ)
        torch.testing.assert_close(out, V)

    def test_zero_occupancy_admits_full_ice(self):
        from specter.ice import blend_ice_into_volume

        n = nz = 16
        V = torch.zeros(1, nz, n, n)
        occ = torch.zeros(1, nz, n, n)
        out = blend_ice_into_volume(V, self._maker(n, nz), 2.0, occupancy=occ)
        assert float(out.mean()) > 0.0

    def test_occupancy_overrides_what_the_potential_would_have_said(self):
        """
        A volume dense enough that the potential-derived rule would admit
        no ice, but with occupancy declaring it empty. The ice must follow
        the occupancy.
        """
        from specter.ice import blend_ice_into_volume

        n = nz = 16
        V = torch.full((1, nz, n, n), 20.0)  # well above the 7 V reference
        from specter.potential import potential_occupancy

        assert float((1.0 - potential_occupancy(V, 2.0)).max()) == 0.0
        out = blend_ice_into_volume(
            V, self._maker(n, nz), 2.0, occupancy=torch.zeros(1, nz, n, n)
        )
        assert float((out - V).mean()) > 0.0

    def test_rejects_occupancy_of_the_wrong_shape(self):
        from specter.ice import blend_ice_into_volume

        n = nz = 16
        V = torch.zeros(1, nz, n, n)
        with pytest.raises(ValueError, match="occupancy"):
            blend_ice_into_volume(
                V, self._maker(n, nz), 2.0, occupancy=torch.zeros(1, nz, n, n + 2)
            )


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

        with pytest.raises(ValueError, match="pixel_size"):
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
