"""
Tests for IceBank.
"""

import pytest
import torch

from specter.ice import GradientSKIcemaker, IceBank
from specter.ice._bank import random_rotation_matrix
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
    degraded by the extraction itself -- same story validated in
    dev/ice/seam_relax_256_*.py this session (crop vs source E_per_atom
    differ only by a gentle edge-truncation effect)."""
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
    source_energy = gd.mlbop_energy(progressbar=False)
    path = tmp_path / "config_000.pt"
    torch.save(
        {"positions": gd.positions.half(), "box_L": 32.0, "n": 32, "dx": 1.0}, path
    )

    cache = IceBank(str(tmp_path), progressbars=False)
    torch.manual_seed(7)
    cache.generate_ice_deltas(n=16, dx=1.0, batchsize=1)
    crop_energy = cache.mlbop_energy(progressbar=False)

    assert abs(crop_energy["E_per_atom"] - source_energy["E_per_atom"]) < 0.3
    assert abs(crop_energy["rij_var"] - source_energy["rij_var"]) < 0.1


def test_icebank_full_size_crop_atom_count_matches_density(tmp_path):
    """Regression test for the periodic-image bug found this session: a
    crop whose extent equals the source's own box size needs reach >
    box_L/2, where the naive single-nearest-image wraparound silently
    drops atoms. Requesting a crop at == the source's own box size should
    still recover close to the expected atom count."""
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


def test_icebank_generate_big_ice_relaxation_improves_energy(tmp_path):
    """The core value proposition: naive tile concatenation (relax_steps=0)
    should be measurably worse than the same tiling with seam relaxation
    -- same story as dev/ice/seam_relax_256_assemble.py's naive-vs-relaxed
    comparison, just at a much smaller/faster scale for a unit test."""
    _make_cache_config(tmp_path, "config_000.pt", n=32, dx=1.0, n_steps=10)
    cache = IceBank(str(tmp_path), progressbars=False)

    torch.manual_seed(5)
    cache.generate_big_ice_deltas(n=64, dx=1.0, batchsize=1, relax_steps=0)
    e_naive = cache.mlbop_energy(pbc=False, progressbar=False)

    torch.manual_seed(5)
    cache.generate_big_ice_deltas(n=64, dx=1.0, batchsize=1, relax_steps=50)
    e_relaxed = cache.mlbop_energy(pbc=False, progressbar=False)

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
