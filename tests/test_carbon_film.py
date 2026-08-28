"""
Unit tests for specter.specimen._carbon (CarbonFilmGenerator), the
alpha-shape-geometry/MIP-calibrated-splat replacement for the earlier
analytic-boundary carbon film generator. Broader integration coverage
(via TomogramSpecimenGenerator's carbon_film_spec) lives in
test_tomogram_generator.py; these focus on the physics this module adds:
real per-atom density calibration reproducing carbon's literature mean
inner potential (MIP), and reproducibility.
"""

from __future__ import annotations

import torch

from specter.specimen._carbon import CarbonFilmGenerator

# Literature mean inner potential range for amorphous carbon, V -- this
# module's placed density (_PLACED_DENSITY_FRACTION) was calibrated
# against real per-atom scattering physics to reproduce it, see
# _carbon.py's module docstring.
_MIP_LO, _MIP_HI = 7.0, 10.0

_TARGET_SHAPE_ZYX = (120, 120, 120)
_V_SIZE = 10.0

# The alpha-shape rim is genuinely rough in z as well as laterally (unlike
# the old analytic-boundary generator's flat top/bottom faces -- see module
# docstring), so a thin film's volume-averaged potential reads measurably
# below the bulk MIP: a real, physically-expected finite-thickness effect
# (surface porosity is a band of roughly fixed width, so it's a bigger
# fraction of a thin film), not a calibration bug -- verified by sweeping
# thickness: 150 A reads 0.82x the bulk target, 600 A+ converges to 1.000x.
# MIP-focused tests below use a thick film specifically to get a clean bulk
# measurement, isolated from that effect.
_BULK_THICKNESS = 1000.0


def _bulk_film(seed: int = 0, thickness: float = 150.0) -> torch.Tensor:
    """A carbon film with the hole placed far outside the frame -- pure
    bulk slab, no rim, so its occupied region is a clean density/MIP
    measurement (still has real alpha-shape surface roughness top/bottom,
    unlike the old generator)."""
    gen = CarbonFilmGenerator(
        voxel_size=_V_SIZE, parameterization="kirkland", seed=seed
    )
    film = gen.generate(
        _TARGET_SHAPE_ZYX,
        thickness=thickness,
        hole_radius=1.0,
        hole_center=(1.0e6, 1.0e6),
    )
    return film.density


def test_carbon_film_generator_bulk_mip_in_literature_range():
    """A thick, bulk (hole-free) film's mean occupied potential should
    land in amorphous carbon's real literature MIP range, not an
    arbitrary constant -- the real per-atom scattering physics
    _PLACED_DENSITY_FRACTION was calibrated against."""
    density = _bulk_film(thickness=_BULK_THICKNESS)
    occupied = density > 0

    assert torch.isfinite(density).all()
    assert occupied.float().mean().item() > 0.5  # hole-free: most of the frame
    assert (density[~occupied] == 0).all()

    mip = float(density[occupied].mean())
    assert _MIP_LO < mip < _MIP_HI


def test_carbon_film_generator_matches_its_own_calibration_target():
    """The measured bulk MIP should track CarbonFilmGenerator's own
    `mean_inner_potential` attribute (computed from the same placed
    density/atom potential integral), not just the literature range --
    catches a miscalibrated weight/density even if it happened to still
    land in [_MIP_LO, _MIP_HI]."""
    gen = CarbonFilmGenerator(voxel_size=_V_SIZE, parameterization="kirkland", seed=1)
    film = gen.generate(
        _TARGET_SHAPE_ZYX,
        thickness=_BULK_THICKNESS,
        hole_radius=1.0,
        hole_center=(1.0e6, 1.0e6),
    )
    occupied = film.density > 0
    measured = float(film.density[occupied].mean())
    # A few % of seed-to-seed statistical noise remains even at this
    # thickness (finite film volume -> finite atom count); a real
    # calibration bug (wrong density fraction, wrong per-atom weight)
    # would be off by tens of percent, not a few.
    assert abs(measured - gen.mean_inner_potential) < 0.08 * gen.mean_inner_potential


def test_carbon_film_generator_reproducible_with_same_seed():
    """Same seed -> the same output, up to CUDA's float32 `index_add_`
    accumulation-order nondeterminism (measured ~1e-6 max absolute
    difference -- orders of magnitude below any real regression, e.g. a
    different atom count/placement, which would differ by a large
    fraction of `mean_inner_potential`, not the last few float32 bits)."""
    a = _bulk_film(seed=42)
    b = _bulk_film(seed=42)
    assert torch.allclose(a, b, atol=1e-3)


def test_carbon_film_generator_different_seeds_differ():
    a = _bulk_film(seed=1)
    b = _bulk_film(seed=2)
    assert not torch.allclose(a, b, atol=1e-3)


def test_carbon_film_spec_normalises_a_list_edge_fraction_range():
    """TOML has no tuple type, so a `[[carbon_film]]` table's
    `edge_fraction = [0.02, 0.05]` arrives as a list -- and every consumer
    branches on `isinstance(..., tuple)` to tell a range from a fixed
    value. Un-normalised, the list fell through that check and reached
    `edge_hole_center` as-is, raising `TypeError: unsupported operand
    type(s) for -: 'int' and 'list'`. Latent until the canonical config
    turned the carbon film on."""
    import pytest

    from specter.specimen._carbon import CarbonFilmSpec

    assert CarbonFilmSpec(edge_fraction=[0.02, 0.05]).edge_fraction == (0.02, 0.05)
    assert isinstance(CarbonFilmSpec(edge_fraction=[0.02, 0.05]).edge_fraction, tuple)
    # A fixed value stays a plain float, so consumers still see "not a range".
    assert CarbonFilmSpec(edge_fraction=0.03).edge_fraction == 0.03

    with pytest.raises(ValueError, match=r"edge_fraction must be \[low, high\]"):
        CarbonFilmSpec(edge_fraction=[0.05, 0.02])
    with pytest.raises(ValueError, match=r"edge_fraction range must be \[low, high\]"):
        CarbonFilmSpec(edge_fraction=[0.1])
