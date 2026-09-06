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
    default_ice_cache_dir,
    build_one_ice_config,
    decode_positions,
    encode_positions,
)
from specter.ice import ice_config_filename
from specter.rotations import random_rotation_matrix_from_generator
from specter.ice._helpers import ndensity_of_amorphous_ice
import math
from specter.potential import potential_occupancy


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


def test_random_rotation_matrix_from_generator_is_proper_and_reproducible():
    g = torch.Generator().manual_seed(0)
    R = random_rotation_matrix_from_generator(g)
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(R, random_rotation_matrix_from_generator(g2))
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
    degraded by the extraction itself (crop vs source E_per_atom differ
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
    by roughly (tile / request)^3 -- ~(64/24)^3 ~= 19x here, ~5.6x at
    production scale. Splatted mass
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
    should be measurably worse than the same tiling with seam relaxation,
    at a small, fast scale."""
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
    to `specter build ice` (see its own docstring) for generating cache
    entries -- check its output round-trips through IceBank at a tiny, fast
    scale, not that it converges well."""
    torch.manual_seed(0)
    build_ice_cache(
        str(tmp_path),
        n_configs=2,
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


def test_build_ice_cache_names_configs_by_seed(tmp_path):
    """Cache entries are named after the seed that generated them, not their
    position in the batch, so a second run at a higher seed_start EXTENDS a
    library instead of overwriting the first run's configs."""
    build_ice_cache(
        str(tmp_path), n_configs=2, n=8, n_steps=2, device="cpu", progressbars=False
    )
    build_ice_cache(
        str(tmp_path),
        n_configs=2,
        n=8,
        n_steps=2,
        device="cpu",
        seed_start=2,
        progressbars=False,
    )

    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        ice_config_filename(seed) for seed in range(4)
    ]
    assert len(IceBank(str(tmp_path), progressbars=False)) == 4


def test_fixed_point_encoding_round_trips_within_half_a_grid_step():
    """encode/decode_positions must be accurate to half the grid spacing, and
    that spacing must be uniform across the box -- the whole point of
    fixed-point over float16, whose precision degrades with distance from the
    origin (0.0625 A across the outer octave of a 256 A cell)."""
    box_L = 256.0
    torch.manual_seed(0)
    pos = (torch.rand(200_000, 3) - 0.5) * box_L

    decoded = decode_positions(encode_positions(pos, box_L), box_L)
    spacing = box_L / 2 / 32767
    err = (decoded - pos).abs()
    assert err.max() <= spacing / 2 * 1.001  # allow float32 decode rounding
    assert decoded.dtype == torch.float32

    # Uniform, not magnitude-dependent: the worst error near the box face is
    # no worse than near the centre. float16 fails this by ~8x.
    r = pos.abs().max(dim=1).values
    inner = err[r < box_L / 8].max()
    outer = err[r > box_L / 2 * 0.9].max()
    assert outer <= inner * 1.5, f"inner {inner:.2e} vs outer {outer:.2e}"

    # And it beats raw float16 storage, the format it replaced.
    f16_err = (pos.half().float() - pos).abs().max()
    assert err.max() < f16_err / 8


def test_icebank_reads_both_coordinate_encodings(tmp_path):
    """A cache directory may hold configs written before the fixed-point
    encoding existed (the bundled ice-data/ice_cache) alongside newer ones.
    IceBank must serve both, keyed on `coord_encoding` rather than dtype."""
    n, dx = 8, 1.0
    torch.manual_seed(0)
    gd = GradientSKIcemaker(n=n, dx=dx, progressbars=False)
    gd.init_random()
    gd.optimize(n_steps=2, record_every=2, tol=None)
    pos, box_L = gd.positions, n * dx

    # Old format: raw float16, no coord_encoding key.
    torch.save(
        {"positions": pos.half(), "box_L": box_L, "n": n, "dx": dx},
        tmp_path / "config_000.pt",
    )
    # New format: fixed-point indices, tagged.
    torch.save(
        {
            "positions": encode_positions(pos, box_L),
            "box_L": box_L,
            "n": n,
            "dx": dx,
            "coord_encoding": "int16_fixed",
        },
        tmp_path / "config_001.pt",
    )

    cache = IceBank(str(tmp_path), progressbars=False)
    assert len(cache) == 2
    for config in cache._configs:
        assert config["positions"].dtype == torch.float32
        assert config["positions"].abs().max() <= box_L / 2 * 1.001
    # The fixed-point entry decodes closer to the original than the float16 one.
    err_f16 = (cache._configs[0]["positions"] - pos).abs().max()
    err_fixed = (cache._configs[1]["positions"] - pos).abs().max()
    assert err_fixed < err_f16

    assert cache.generate_ice(n=n, dx=dx, batchsize=1).shape == (1, n, n, n)


def test_build_one_ice_config_writes_fixed_point_coordinates(tmp_path):
    """Newly generated configs use the fixed-point encoding and say so, so
    _load_config doesn't have to guess."""
    path = tmp_path / "config_000.pt"
    build_one_ice_config(
        str(path), n=8, dx=1.0, n_steps=2, device="cpu", progressbars=False
    )

    raw = torch.load(path, weights_only=False)
    assert raw["positions"].dtype == torch.int16
    assert raw["coord_encoding"] == "int16_fixed"
    # Timing metadata is recorded for cost estimation, under the same keys the
    # bundled library uses.
    assert raw["wall_time"] > 0
    assert raw["n_steps_actual"] == 2


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

    result = blend_ice_into_volume(V, icemaker, voxel_size=1.0)
    assert result.shape == (1, nz, nxy, nxy)
    assert torch.isfinite(result).all()


def test_bundled_ice_data_resolves_inside_the_package():
    """The bundled cache and MD reference data must be found through the
    import system, not by walking up from __file__ to a repository root.

    Regression guard for a real packaging bug: the data used to live at
    `<repo>/ice-data/` and be located as dirname(dirname(dirname(__file__))),
    which resolves to the repo root only for an editable install from a
    checkout. Installed as a wheel it pointed inside the virtualenv's lib/,
    so `ice_model="gd"` AND GradientSKIcemaker's default S(k) target were
    both unreachable."""
    from pathlib import Path

    import specter
    from specter.ice_data import bundled_ice_data

    package_root = Path(specter.__file__).parent

    cache = Path(default_ice_cache_dir())
    assert cache.is_dir()
    assert package_root in cache.parents, (
        f"{cache} is outside the installed package at {package_root} -- it will "
        "not be found in a wheel install"
    )
    assert sorted(cache.glob("*.pt")), "bundled cache has no configs"

    # Every file GradientSKIcemaker needs by default, likewise inside the package.
    for name in (
        "lda_80k_frame799_full_coords.pt",
        "mdsim_f_radial_avg_400x400x400_0.25A.pt",
    ):
        path = bundled_ice_data(name)
        assert path.is_file(), f"{name} missing from specter.ice_data"
        assert package_root in path.parents


def test_bundled_data_is_declared_as_package_data():
    """Being inside the package is necessary but not sufficient -- setuptools
    only copies data files matched by [tool.setuptools.package-data], so a
    missing glob ships a wheel whose data silently isn't there."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject.is_file():  # installed without the source tree
        pytest.skip("pyproject.toml not available")
    globs = tomllib.loads(pyproject.read_text())["tool"]["setuptools"]["package-data"][
        "specter"
    ]

    for required in (
        "ice_data/*.pt",
        "ice_data/ice_cache/*.pt",
        "atom_data/*.txt",
        "atom_data/*.json",  # params_cat.json backs the default shtyrov path
    ):
        assert required in globs, f"{required} missing from package-data"


def _blend_reference(V, icemaker, pixel_size):
    """
    The whole-volume blend expression, verbatim.

    `V + ice * (1 - potential_occupancy(V))` holds five tensors the size of
    the whole canvas at once -- V, ice, the weight, the product and the sum.
    At `micrograph_size` that is 5 x 33.6 GB, and it was most of why a
    4096-pixel micrograph peaked at 237 GB of RSS. The shipped version
    weights and multiplies one z-slab at a time and accumulates in place;
    this asserts the two agree.
    """
    from specter.potential import potential_occupancy

    ice = icemaker.generate_ice(batchsize=V.shape[0]).to(V.device)
    return V + ice * (1.0 - potential_occupancy(V, pixel_size)).clamp(0.0, 1.0)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("nz,n", [(16, 24), (17, 24), (33, 16)])
def test_blend_ice_slabwise_matches_whole_volume_expression(nz, n, inplace):
    """
    Slab-wise in-place blending reproduces the whole-volume expression.

    This is now also the test of the blur's HALO. The occupancy estimator
    coarse-grains over 2 A, so it reads past whatever slab it is handed;
    the shipped loop widens each slab by the kernel's reach and discards
    the margin. Drop that and every chunk boundary becomes an edge the
    blur sees, which prints into the ice as a seam -- and shows up here as
    a mismatch against the unchunked expression.

    Non-dividing `nz` is covered because the slab loop's last chunk is
    short, and because the first and last slabs are the ones whose halo is
    truncated by the volume itself.
    """
    from specter.ice import RandomIcemaker
    from specter.ice._bank import blend_ice_into_volume

    torch.manual_seed(0)
    maker = RandomIcemaker(n=n, dx=2.0, nz=nz)
    torch.manual_seed(1)
    V0 = torch.rand(1, nz, n, n) * 3.0

    torch.manual_seed(7)
    want = _blend_reference(V0.clone(), maker, 2.0)

    torch.manual_seed(7)
    V = V0.clone()
    got = blend_ice_into_volume(V, maker, 2.0, inplace=inplace)

    # Not bit-exact: a separable conv over a slab-plus-halo and over the
    # whole volume are the same function but not the same kernel launch.
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    if inplace:
        assert got.data_ptr() == V.data_ptr(), "inplace must write through to V"
    else:
        assert torch.equal(V, V0), "inplace=False must leave the input untouched"


# ---------------------------------------------------------------------------
# device placement and slab-wise blending
# ---------------------------------------------------------------------------


def test_ice_bank_crop_stays_on_bank_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = IceBank().to(device)
    torch.manual_seed(0)
    ice = bank.generate_ice(n=24, dx=1.0, nz=24)
    assert ice.device.type == device
    assert bank.positions is not None and bank.positions.device.type == device
    assert math.isfinite(bank.mlbop_energy()["E_per_atom"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a device for the host-canvas path"
)
def test_host_volume_blend_matches_whole_canvas_blend():
    """
    A host-resident volume blended slab by slab on the device equals the
    whole-canvas blend of the same draws, to float noise, including at the
    z ends where the slab path wraps its context.
    """
    dev = torch.device("cuda")
    n, nz, dx = 96, 70, 1.0
    bank = IceBank(device=dev, progressbars=False)
    V0 = torch.zeros(1, nz, n, n)
    g = torch.Generator().manual_seed(1)
    for _ in range(8):
        c = (torch.rand(3, generator=g) * torch.tensor([nz, n, n])).long()
        zs, ys, xs = [slice(max(0, int(ci) - 8), int(ci) + 8) for ci in c]
        V0[0, zs, ys, xs] += 10.0

    # Reference: the whole-canvas path, forced by putting V on the device.
    gen = torch.Generator().manual_seed(7)
    ice = bank.generate_big_ice(
        n=n, dx=dx, nz=nz, batchsize=1, device=dev, generator=gen
    )
    ref = V0.to(dev) + ice * (1 - potential_occupancy(V0.to(dev), dx)).clamp(0, 1)

    # Slab path: V on the host, bank on the device, same draws, and a slab
    # small enough that the volume spans several.
    from specter.ice import _bank as bank_mod

    torch.manual_seed(0)
    new = V0.clone()
    gen2 = torch.Generator().manual_seed(7)
    real = bank.big_ice_positions

    def seeded_positions(*a, **kw):
        kw["generator"] = gen2
        return real(*a, **kw)

    bank.big_ice_positions = seeded_positions  # type: ignore[method-assign]
    bank_mod._blend_ice_slabwise(new, bank, dx, slab_voxels=n * n * 16)
    assert new.device.type == "cpu"
    diff = new.to(dev) - ref
    assert float(diff.abs().max()) < 1e-3 * float(ice.abs().max())
    # Plane means agree at both z ends (the wrapped context).
    pm = (new.to(dev) - ref).mean(dim=(0, 2, 3))
    assert float(pm.abs().max()) < 1e-4 * float(ice.mean())

    # And the public entry point routes a host V to this path.
    bank.big_ice_positions = real  # type: ignore[method-assign]
    out = blend_ice_into_volume(V0.clone(), bank, dx, inplace=True)
    assert out.device.type == "cpu"
    assert float((out - V0).mean()) == pytest.approx(float(ice.mean()), rel=0.05)


@pytest.mark.parametrize("chunk", [3, 7, 64])
def test_ice_slab_blender_matches_whole_volume_blend(chunk):
    """Slab by slab, in any slab size, equals the one-shot rule
    ``V + ice * clamp(1 - occupancy(V), 0, 1)``."""
    torch.manual_seed(0)
    nz, n = 40, 24
    V = torch.zeros(1, nz, n, n)
    V[0, 10:20, 5:15, 5:15] = 9.0
    ice = torch.rand(1, nz, n, n) * 4
    whole = V + ice * (1 - potential_occupancy(V, 1.0)).clamp(0, 1)

    from specter.ice._blend import IceSlabBlender

    out = V.clone()
    blender = IceSlabBlender(1.0)
    for start in range(0, nz, chunk):
        end = min(start + chunk, nz)
        blender.add(out, ice[:, start:end].clone(), start, end)
    assert torch.allclose(out, whole, rtol=1e-5, atol=1e-6)
