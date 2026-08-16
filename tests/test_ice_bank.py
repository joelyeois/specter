"""
Tests for IceBank.
"""

import pytest
import torch

from specter.arrays import soft_voxelize_coordinates
from specter.ice import GradientSKIcemaker, IceBank, RandomIcemaker
from specter.ice._bank import (
    blend_ice_into_volume,
    build_ice_cache,
    random_rotation_matrix,
)
from specter.ice._helpers import ndensity_of_amorphous_ice


def _make_cache_config(tmp_path, name, n, dx, seed=0, n_steps=5):
    """A small, quickly-generated (not fully converged -- just real,
    physically-derived) config to stand in for a cache entry."""
    torch.manual_seed(seed)
    gd = GradientSKIcemaker(n=n, dx=dx, progressbars=False)
    gd.init_random()
    gd.optimize(n_steps=n_steps, record_every=n_steps, tol=None)
    path = tmp_path / name
    torch.save(
        {"positions": gd.positions.half(), "box_L": n * dx, "n": n, "dx": dx}, path
    )
    return path


def test_random_rotation_matrix_is_proper_rotation():
    torch.manual_seed(0)
    R = random_rotation_matrix()
    assert R.shape == (3, 3)
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    assert torch.det(R).item() == pytest.approx(1.0, abs=1e-5)


def test_icebank_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        IceBank(str(tmp_path))


def test_icebank_basic_extraction_shape_and_finite(tmp_path):
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)
    assert len(cache) == 1

    torch.manual_seed(1)
    ice = cache.generate_ice(n=16, dx=1.0, batchsize=3)
    assert ice.shape == (3, 16, 16, 16)
    assert torch.isfinite(ice).all()


def test_icebank_parameterization_changes_kernel(tmp_path):
    """Regression guard: IceBank._get_kernel should honor `parameterization`,
    not silently fall back to a hardcoded 'kirkland'."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    kirkland = IceBank(str(tmp_path), progressbars=False)
    lobato = IceBank(str(tmp_path), progressbars=False, parameterization="lobato")

    assert not torch.allclose(kirkland._get_kernel(1.0), lobato._get_kernel(1.0))


def test_icebank_generate_ice_deltas_shape(tmp_path):
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    deltas = cache.generate_ice_deltas(n=16, dx=1.0, batchsize=2)
    assert deltas.shape == (2, 16, 16, 16)


def test_icebank_noncubic_request(tmp_path):
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    ice = cache.generate_ice(n=16, nz=8, dx=1.0, batchsize=1)
    assert ice.shape == (1, 8, 16, 16)


def test_icebank_raises_for_oversized_request(tmp_path):
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    with pytest.raises(ValueError, match="exceeds"):
        cache.generate_ice(n=64, dx=1.0, batchsize=1)


def test_icebank_mlbop_energy_matches_source_quality(tmp_path):
    """A crop's local structure should be inherited from its source, not
    degraded by the extraction itself -- same story validated at production
    scale in dev/ice/seam_relax_256_*.py (crop vs source E_per_atom differ
    only by a gentle edge-truncation effect)."""
    torch.manual_seed(42)
    gd = GradientSKIcemaker(n=32, dx=1.0, progressbars=False)
    gd.init_random()
    gd.optimize(
        n_steps=15,
        record_every=15,
        rep_strength=0.0,
        mlbop_strength=0.05,
        mlbop_target=-0.413,
        tol=None,
    )
    source_energy = gd.mlbop_energy()
    path = tmp_path / "config_000.pt"
    torch.save(
        {"positions": gd.positions.half(), "box_L": 32.0, "n": 32, "dx": 1.0}, path
    )

    cache = IceBank(str(tmp_path), progressbars=False)
    torch.manual_seed(7)
    cache.generate_ice_deltas(n=16, dx=1.0, batchsize=1)
    crop_energy = cache.mlbop_energy()

    assert abs(crop_energy["E_per_atom"] - source_energy["E_per_atom"]) < 0.3
    assert abs(crop_energy["rij_var"] - source_energy["rij_var"]) < 0.1


def test_icebank_full_size_crop_atom_count_matches_density(tmp_path):
    """Regression test for a periodic-image bug: a crop whose extent equals
    the source's own box size needs reach > box_L/2, where the naive
    single-nearest-image wraparound silently drops atoms. Requesting a crop
    at == the source's own box size should still recover close to the
    expected atom count."""
    torch.manual_seed(3)
    gd = GradientSKIcemaker(n=24, dx=1.0, progressbars=False)
    gd.init_random()
    path = tmp_path / "config_000.pt"
    torch.save(
        {"positions": gd.positions.clone().half(), "box_L": 24.0, "n": 24, "dx": 1.0},
        path,
    )

    cache = IceBank(str(tmp_path), progressbars=False)
    torch.manual_seed(0)
    cache.generate_ice_deltas(n=24, dx=1.0, batchsize=1)  # crop == full source box size

    expected = ndensity_of_amorphous_ice * 24.0**3
    n_atoms = cache.positions.shape[0]
    assert abs(n_atoms - expected) / expected < 0.05


def test_icebank_small_crop_atom_count_matches_density_on_average(tmp_path):
    """Regression test: a crop much smaller than the source's own box (here
    1/4 of it, reach/box_L well under 0.5) must recover the expected atom
    count on average, regardless of where the random center happens to
    land.

    This is the regime the periodic-image count formula got wrong: m =
    ceil(reach/box_L - 0.5) gives m=0 (no wraparound at all, not even one
    neighboring image) whenever reach <= box_L/2, silently dropping every
    atom that should have wrapped in from the opposite side whenever the
    random center lands near a box edge -- systematically biasing the
    count low (was ~35-50% of expected in this regime), not just adding
    noise. Checks the *mean* over many draws against the expected density
    (not a single draw, or even the worst of several) since at this crop
    size (~130 expected atoms) plain Poisson counting noise alone spans a
    wide range (~75-125% of expected per draw) that would otherwise mask a
    real systematic bias of this size, or trigger false failures unrelated
    to it."""
    torch.manual_seed(4)
    gd = GradientSKIcemaker(n=64, dx=1.0, progressbars=False)
    gd.init_random()
    path = tmp_path / "config_000.pt"
    torch.save(
        {"positions": gd.positions.clone().half(), "box_L": 64.0, "n": 64, "dx": 1.0},
        path,
    )

    cache = IceBank(str(tmp_path), progressbars=False)
    expected = ndensity_of_amorphous_ice * 16.0**3
    counts = []
    for seed in range(50):
        torch.manual_seed(seed)
        cache.generate_ice_deltas(n=16, dx=1.0, batchsize=1)  # reach/box_L ~ 0.22
        counts.append(cache.positions.shape[0])

    mean_ratio = (sum(counts) / len(counts)) / expected
    assert abs(mean_ratio - 1.0) < 0.1


def test_icebank_deterministic_filtering_with_mixed_config_sizes(tmp_path):
    """Regression test: requesting an extent that only the larger of two
    differently-sized cached configs can satisfy must always succeed,
    regardless of which config a naive random draw might otherwise pick."""
    _make_cache_config(tmp_path, "config_small.pt", n=16, dx=1.0, seed=1)
    _make_cache_config(tmp_path, "config_big.pt", n=32, dx=1.0, seed=2)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(0)
    for _ in range(20):
        ice = cache.generate_ice(n=24, dx=1.0, batchsize=1)
        assert ice.shape == (1, 24, 24, 24)


def test_icebank_generate_big_ice_shape_and_finite(tmp_path):
    """A request larger than the single cached config (32^3) must tile
    (here 2x2x2 of 32A tiles -> 64^3) rather than raise."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(1)
    ice = cache.generate_big_ice(n=64, dx=1.0, batchsize=1, relax_steps=20)
    assert ice.shape == (1, 64, 64, 64)
    assert torch.isfinite(ice).all()


def test_icebank_generate_big_ice_deltas_noncubic_and_batched(tmp_path):
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(2)
    deltas = cache.generate_big_ice_deltas(
        n=64, nz=48, dx=1.0, batchsize=2, relax_steps=15
    )
    assert deltas.shape == (2, 48, 64, 64)


def test_icebank_generate_big_ice_deltas_streamed_splat_matches_monolithic(tmp_path):
    """When relax_steps=0, generate_big_ice_deltas splats each tile into
    the output volume as soon as it's drawn (bounding peak memory to a
    single tile's atom count) instead of gathering every tile's atoms into
    one tensor before voxelizing. Splatting is linear/accumulating, so the
    streamed result must exactly match voxelizing the same (post-`keep`)
    positions in one monolithic call."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(9)
    streamed = cache.generate_big_ice_deltas(n=64, dx=1.0, batchsize=1, relax_steps=0)
    reference = soft_voxelize_coordinates(
        cache.positions, grid_shape=(64, 64, 64), voxel_size=1.0, periodic=True
    )
    assert torch.allclose(streamed[0], reference, atol=1e-5)


def test_icebank_generate_big_ice_streamed_splat_matches_trimmed_monolithic_with_overflow(
    tmp_path,
):
    """Same invariant as the test above, but at a request size (48, with a
    32 A tile) that is NOT an exact multiple of the tile size -- unlike
    n=64 above, every edge tile's own footprint here genuinely overhangs
    the true requested box, exercising the per-tile overflow-discard fix
    in _place_tiles (a prior version of this splat call used an
    unconditional periodic wrap that pulled far-outside overflow atoms
    back in via modulo instead of dropping them, silently inflating
    density near the overhanging edges -- this is the regression guard for
    that, alongside the density-based checks below)."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(11)
    streamed = cache.generate_big_ice_deltas(n=48, dx=1.0, batchsize=1, relax_steps=0)
    reference = soft_voxelize_coordinates(
        cache.positions, grid_shape=(48, 48, 48), voxel_size=1.0, periodic=True
    )
    assert torch.allclose(streamed[0], reference, atol=1e-5)


def test_icebank_generate_big_ice_deltas_request_smaller_than_tile_matches_density(
    tmp_path,
):
    """Regression test for a real bug: a request smaller than a single
    tile (here tile_extent defaults to the cached config's own 64 A box,
    for a 24 A request) used to have its single tile's full-box atom set
    wrapped almost entirely into the small destination grid via an
    unconditional periodic index wrap in _place_tiles, inflating density
    by roughly (tile / request)^3 -- ~(64/24)^3 ~= 19x was observed before
    the fix, for a smaller-scale repro of the ~5.6x measured at production
    scale in dev/ice/analytic_tile_insertion_benchmark.py. Splatted mass
    (deltas.sum()) approximates atom count for a trilinear splat, so this
    checks it against the expected bulk-water atom count directly."""
    torch.manual_seed(5)
    gd = GradientSKIcemaker(n=64, dx=1.0, progressbars=False)
    gd.init_random()
    path = tmp_path / "config_000.pt"
    torch.save(
        {"positions": gd.positions.clone().half(), "box_L": 64.0, "n": 64, "dx": 1.0},
        path,
    )
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(6)
    deltas = cache.generate_big_ice_deltas(n=24, dx=1.0, batchsize=1, relax_steps=0)

    expected = ndensity_of_amorphous_ice * 24.0**3
    assert abs(deltas.sum().item() - expected) / expected < 0.3


def test_icebank_generate_big_ice_relaxation_improves_energy(tmp_path):
    """The core value proposition: naive tile concatenation (relax_steps=0)
    should be measurably worse than the same tiling with seam relaxation
    -- same story as dev/ice/seam_relax_256_assemble.py's naive-vs-relaxed
    comparison, just at a much smaller/faster scale for a unit test."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0, n_steps=10)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(5)
    cache.generate_big_ice_deltas(n=64, dx=1.0, batchsize=1, relax_steps=0)
    e_naive = cache.mlbop_energy(pbc=False)

    torch.manual_seed(5)
    cache.generate_big_ice_deltas(n=64, dx=1.0, batchsize=1, relax_steps=50)
    e_relaxed = cache.mlbop_energy(pbc=False)

    assert e_relaxed["E_per_atom"] < e_naive["E_per_atom"]
    assert abs(e_relaxed["E_per_atom"] - (-0.413)) < abs(
        e_naive["E_per_atom"] - (-0.413)
    )


def test_icebank_generate_big_ice_request_fitting_single_tile(tmp_path):
    """Degenerate case: a 'big' request that only needs a 1x1x1 tile grid
    (no seams at all) should still work without error."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(0)
    ice = cache.generate_big_ice(n=32, dx=1.0, batchsize=1, relax_steps=10)
    assert ice.shape == (1, 32, 32, 32)
    assert torch.isfinite(ice).all()


def test_build_ice_cache_writes_loadable_configs(tmp_path):
    """Smoke test: build_ice_cache() is a slower, single-process alternative
    to dev/ice/generate_ice_cache_worker.py (see its own docstring) for
    generating cache entries -- check its output round-trips through
    IceBank at a tiny, fast scale, not that it converges well."""
    torch.manual_seed(0)
    build_ice_cache(
        str(tmp_path),
        num_configs=2,
        n=8,
        dx=1.0,
        n_steps=3,
        device="cpu",
        seed_start=0,
        progressbars=False,
    )

    written = sorted(tmp_path.glob("config_*.pt"))
    assert len(written) == 2
    assert written[0].name == "config_000.pt"
    assert written[1].name == "config_001.pt"

    cache = IceBank(str(tmp_path), progressbars=False)
    assert len(cache) == 2
    ice = cache.generate_ice(n=8, dx=1.0, batchsize=1)
    assert ice.shape == (1, 8, 8, 8)
    assert torch.isfinite(ice).all()


def test_blend_ice_into_volume_random_icemaker_noncubic_nxy_nz():
    """
    blend_ice_into_volume's docstring states that a RandomIcemaker's own
    fixed (n, dx, nz) "must already match V" -- it has no tiling support,
    unlike IceBank. Regression test for a real bug found in the micrograph
    pipeline (then demo-scripts/generate_micrograph.py, now
    specter.pipelines.run_micrograph) and the matching notebook: both
    constructed RandomIcemaker with n=config.n_pixels (the separate,
    usually much smaller, particle-potential resolution) instead of
    n=config.micrograph_size, and relied on RandomIcemaker's nz defaulting
    to n (a cube) instead of the actual, generally much smaller nz derived
    from ice_thickness -- causing a broadcasting RuntimeError (and, once
    n was fixed but nz still defaulted to a cube, a large spurious CUDA
    OOM from allocating a needlessly cubic ice volume). This test builds
    V and the RandomIcemaker with genuinely different, non-cube (nz, nxy)
    to guard against a similar mismatch being reintroduced.
    """
    nz, nxy = 6, 16
    V = torch.zeros(1, nz, nxy, nxy)
    icemaker = RandomIcemaker(dx=1.0, n=nxy, nz=nz, progressbars=False)

    result = blend_ice_into_volume(V, icemaker, pixel_size=1.0)
    assert result.shape == (1, nz, nxy, nxy)
    assert torch.isfinite(result).all()


def test_insert_analytic_ice_leaves_occupied_region_untouched(tmp_path):
    """The whole point of insert_analytic_ice: pre-existing (occupied)
    voxels must be exactly preserved, both because they fail the
    per-voxel mask test and because a coarse-fully-occupied tile is
    skipped before any ice is even drawn for it."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    volume = torch.zeros(20, 24, 24)
    volume[4:16, 4:20, 4:20] = 5.0
    occupied = volume.clone()

    torch.manual_seed(0)
    out = cache.insert_analytic_ice(volume, dx=1.0, tile_extent=12.0, rcut=3.0)

    assert out is volume  # modified (and returned) in place
    assert torch.equal(out[4:16, 4:20, 4:20], occupied[4:16, 4:20, 4:20])


def test_insert_analytic_ice_fills_free_region_at_expected_density(tmp_path):
    """Free voxels should pick up a nonzero, finite potential whose overall
    density is in the right ballpark for bulk amorphous ice -- same
    expected-density check style as the crop-atom-count regression tests
    above, just carried through to the final analytically-evaluated
    potential rather than a raw atom count."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    volume = torch.zeros(20, 24, 24)
    # blend_ice_into_volume's masking convention is `V < threshold * V.max()`
    # -- an all-zero V has V.max() == 0, so nothing is ever "below threshold"
    # and no ice would ever be added, matching the existing convolve path's
    # own behaviour on an all-zero V. A single high "occupied" corner voxel
    # establishes a realistic V.max() (as any real specimen volume already
    # has, from particle/crowding density) without meaningfully shrinking
    # the free region under test.
    volume[0, 0, 0] = 10.0
    free_mask = volume == 0

    torch.manual_seed(1)
    out = cache.insert_analytic_ice(volume, dx=1.0, tile_extent=12.0, rcut=3.0)

    assert torch.isfinite(out).all()
    assert (out[free_mask] > 0).all()
    # A lone oxygen atom's own peak potential is a fixed, dx-independent
    # physical quantity (~11-12 V at dx=1 Å per the Kirkland kernel/analytic
    # cross-check used to validate this feature) -- bulk water's per-voxel
    # mean should sit well under that (heavily overlapping, partially
    # cancelling neighbor tails), and well above zero.
    assert 0.5 < out[free_mask].mean().item() < 11.0


def test_insert_analytic_ice_skips_fully_occupied_volume(tmp_path):
    """A volume that's already fully occupied (coarse mask all-full) must
    come back completely unmodified -- every tile gets skipped before any
    crop is drawn."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    volume = torch.full((20, 24, 24), 5.0)
    original = volume.clone()

    torch.manual_seed(2)
    out = cache.insert_analytic_ice(volume, dx=1.0, tile_extent=12.0, rcut=3.0)
    assert torch.equal(out, original)


@pytest.mark.parametrize("parameterization", ["kirkland", "lobato", "shtyrov"])
def test_insert_analytic_ice_every_parameterization(tmp_path, parameterization):
    """Smoke test: every parameterization IceBank supports for the
    splat/convolve kernel must also work through the analytic dispatch in
    _ice_analytic_potential."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(
        str(tmp_path), parameterization=parameterization, progressbars=False
    )

    volume = torch.zeros(16, 16, 16)
    volume[0, 0, 0] = 10.0  # establish a realistic V.max() > 0 -- see note above
    torch.manual_seed(3)
    out = cache.insert_analytic_ice(volume, dx=1.0, tile_extent=10.0, rcut=3.0)
    assert torch.isfinite(out).all()
    assert (out > 0).any()


def test_blend_ice_into_volume_method_analytic_matches_direct_call(tmp_path):
    """blend_ice_into_volume(method='analytic') should be a thin wrapper
    around IceBank.insert_analytic_ice on V[0] -- same result, batch dim
    added back."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    volume = torch.zeros(1, 16, 16, 16)
    torch.manual_seed(4)
    result = blend_ice_into_volume(
        volume.clone(), cache, pixel_size=1.0, method="analytic"
    )

    cache2 = IceBank(str(tmp_path), progressbars=False)
    torch.manual_seed(4)
    expected = cache2.insert_analytic_ice(volume[0].clone(), dx=1.0)

    assert result.shape == (1, 16, 16, 16)
    assert torch.equal(result[0], expected)


def test_blend_ice_into_volume_method_analytic_rejects_batch(tmp_path):
    """method='analytic' only supports a single-sample V -- documented,
    explicit ValueError rather than silently only blending the first
    sample."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0)
    cache = IceBank(str(tmp_path), progressbars=False)

    volume = torch.zeros(2, 16, 16, 16)
    with pytest.raises(ValueError):
        blend_ice_into_volume(volume, cache, pixel_size=1.0, method="analytic")
