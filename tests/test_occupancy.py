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


class TestTemplateOccupancyReference:
    """
    The reference a template's own occupancy is read against. How much ice a
    particle displaces is its volume, which scattering factors do not change;
    under a fixed 7.0 V it followed them, 0.84x the molecule for a Shtyrov
    render with hydrogens and 1.09x for a Kirkland one.
    """

    @staticmethod
    def _template(n=48, dx=1.0):
        from specter.potential import PotentialBuilder

        g = torch.Generator().manual_seed(1)
        coords = torch.randn(1500, 3, generator=g) * 5.0
        z = torch.full((1500,), 6)
        z[::2] = 1
        v = PotentialBuilder(n, dx, z, parameterization="kirkland")(coords)
        return v.detach(), z

    def _displaced(self, v, dx, ref):
        from specter.potential import potential_occupancy

        return float(potential_occupancy(v, dx, full_potential=ref).sum()) * dx**3

    @pytest.mark.parametrize("strength", [0.8, 1.0, 1.25])
    def test_displaces_exactly_the_molecular_volume_at_any_strength(self, strength):
        """
        Rescaling the potential stands in for a different scattering table:
        the reference moves with it and the displaced volume does not.
        """
        from specter.potential import full_occupancy_potential

        v, _ = self._template()
        v = v * strength
        volume = 4000.0
        ref = full_occupancy_potential(v, 1.0, volume)
        assert self._displaced(v, 1.0, ref) == pytest.approx(volume, rel=1e-3)

    def test_reads_below_the_mean_when_the_clamp_bites(self):
        """
        The clamp discards what dense voxels hold above the reference, so the
        solution sits below the mean inner potential, which alone would
        displace too little.
        """
        from specter.potential import full_occupancy_potential

        v, _ = self._template()
        volume = 4000.0
        mean = float(v.sum()) / volume
        ref = full_occupancy_potential(v, 1.0, volume)
        assert ref < mean
        assert self._displaced(v, 1.0, mean) < volume

    def test_empty_space_around_the_molecule_changes_nothing(self):
        from specter.potential import full_occupancy_potential

        v, _ = self._template(n=48)
        padded = torch.zeros(96, 96, 96)
        padded[24:72, 24:72, 24:72] = v
        a = full_occupancy_potential(v, 1.0, 4000.0)
        b = full_occupancy_potential(padded, 1.0, 4000.0)
        assert b == pytest.approx(a, rel=2e-4)

    def test_hydrogen_free_models_are_charged_their_hydrogens(self):
        from specter.atom import atom_mass
        from specter.potential import molecular_mass_from_atoms

        with_h = torch.tensor([6, 6, 1, 1, 8])
        heavy = torch.tensor([6, 6, 8])
        assert molecular_mass_from_atoms(with_h) == pytest.approx(
            float(atom_mass(with_h).sum())
        )
        assert molecular_mass_from_atoms(heavy) == pytest.approx(
            float(atom_mass(heavy).sum()) / 0.932
        )

    def test_generator_solves_its_reference_only_when_it_knows_the_mass(self):
        from specter.imagegenerator import ImageGenerator
        from specter.potential import (
            FULL_OCCUPANCY_POTENTIAL_V,
            PROTEIN_VOLUME_PER_DALTON_A3,
            full_occupancy_potential,
        )
        from specter.settings import Camera

        v, _ = self._template()

        def build(mass):
            return ImageGenerator(
                v,
                1.0,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.zeros(1, 2),
                None,
                300.0,
                torch.tensor([40.0]),
                optics=None,
                camera=Camera(noise_model="none"),
                progressbars=False,
                verbose=False,
                molecular_mass=mass,
            )

        assert build(None)._occupancy_reference() == FULL_OCCUPANCY_POTENTIAL_V
        mass = 4000.0 / PROTEIN_VOLUME_PER_DALTON_A3
        assert build(mass)._occupancy_reference() == pytest.approx(
            full_occupancy_potential(v, 1.0, 4000.0)
        )

    def test_micrograph_specimen_blends_against_its_own_reference(self, monkeypatch):
        """
        A micrograph's copies are all one template, so one solved reference
        serves them; it must be what the ice blend actually receives.
        """
        import specter.specimen._single_particle as sp
        from specter.potential import (
            FULL_OCCUPANCY_POTENTIAL_V,
            PROTEIN_VOLUME_PER_DALTON_A3,
            full_occupancy_potential,
        )
        from specter.settings import Ice

        v, _ = self._template()
        seen = []

        def spy(V, icemaker, pixel_size, full_potential=None, **kwargs):
            seen.append(full_potential)
            return V

        monkeypatch.setattr(sp, "blend_ice_into_volume", spy)
        mass = 4000.0 / PROTEIN_VOLUME_PER_DALTON_A3
        for m in (None, mass):
            gen = sp.MicrographSpecimenGenerator(
                v,
                1.0,
                48,
                ice=Ice(model="random"),
                progressbars=False,
                molecular_mass=m,
            )
            gen.generate()
        assert seen[0] == FULL_OCCUPANCY_POTENTIAL_V
        assert seen[1] == pytest.approx(full_occupancy_potential(v, 1.0, 4000.0))
