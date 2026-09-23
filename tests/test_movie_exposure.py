"""Physical limits and independent checks for the movie-informed prototype."""

import numpy as np
import pytest
import torch
from specter.ice._exposure import solvent_exposure_power, apply_solvent_exposure
from specter.match._movie_calibration import (
    ice_covariance,
    estimate_solvent_motion,
    bases,
)
from specter.match._detector_calibration import load_detector_calibration


def test_weighted_exposure_matches_dense_interval_covariance():
    rng = np.random.default_rng(9)
    k = torch.tensor([0.0, 0.05, 0.27, 0.6], dtype=torch.float64)
    w = torch.tensor(rng.uniform(0.1, 1, (40, 4)), dtype=torch.float64)
    # Grid bins map exactly to these four physical frequencies below.
    kk = torch.linspace(0, 0.6, 4, dtype=torch.float64)
    actual = solvent_exposure_power(kk, 50.0, 0.38, 40, w, 0.6).numpy()
    a = w.numpy() / w.numpy().sum(0)
    expected = []
    for b, f in enumerate(kk.numpy()):
        c = (
            np.ones((40, 40))
            if f == 0
            else ice_covariance(np.linspace(0, 50, 41), f, 0.38)
        )
        expected.append(a[:, b] @ c @ a[:, b])
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    torch.testing.assert_close(solvent_exposure_power(k, 50, 0, 40), torch.ones_like(k))


def test_uniform_exposure_independent_of_readout_fractionation():
    k = torch.tensor([0.01, 0.27, 0.6], dtype=torch.float64)
    a = solvent_exposure_power(k, 50, 0.38, 1)
    for n in [7, 40, 200]:
        torch.testing.assert_close(
            solvent_exposure_power(k, 50, 0.38, n), a, rtol=1e-10, atol=1e-10
        )


def test_3d_effective_filter_preserves_mean_and_projected_power():
    torch.manual_seed(4)
    x = torch.rand(1, 16, 16, 16) + 2
    original = x.clone()
    apply_solvent_exposure(x, 1.0, 50.0, 0.38, 40, None, None)
    torch.testing.assert_close(x.mean(), original.mean(), rtol=1e-6, atol=1e-6)
    k = torch.hypot(torch.fft.fftfreq(16)[:, None], torch.fft.rfftfreq(16)[None, :])
    expected = (
        torch.fft.rfft2(original.sum(1))
        * solvent_exposure_power(k, 50, 0.38, 40).sqrt()
    )
    torch.testing.assert_close(
        torch.fft.rfft2(x.sum(1)), expected, atol=0.004, rtol=0.001
    )


def test_temporal_estimator_recovers_motion_not_spectral_amplitude(tmp_path):
    k = np.linspace(0.22, 0.33, 5)
    edges = np.linspace(0, 40, 41)
    s2 = 0.32
    cov = np.array(
        [
            np.einsum("c,cij->ij", [3 + b, 0.2, 1.0], bases(edges, f, s2))
            for b, f in enumerate(k)
        ]
    )
    p = tmp_path / "movie.npz"
    np.savez(
        p,
        k=k,
        dose_edges=edges,
        cov_outer=cov,
        n_particles=20,
        split="calibration",
        counts=np.ones(5),
    )
    result = estimate_solvent_motion([p])
    assert result["motion_variance"] == pytest.approx(s2, rel=1e-4)
    assert result["spectral_amplitudes_exported"] is False


def test_detector_calibration_keeps_signal_and_noise_distinct(tmp_path):
    p = tmp_path / "detector.npz"
    k = np.linspace(0, 1.0, 101)
    mtf = np.exp(-k * k)
    noise = np.exp(-2 * k * k)
    np.savez(p, k=k, effective_mtf=mtf, noise_transfer=noise, dqe0=0.8)
    m, n, d = load_detector_calibration(str(p), 32, 1.0)
    assert d == 0.8
    assert m[0, 0] == n[0, 0] == 1
    assert n[8, 0] < m[8, 0] < 1
    np.savez(p, k=k, effective_mtf=mtf * 2, noise_transfer=noise, dqe0=0.8)
    with pytest.raises(ValueError):
        load_detector_calibration(str(p), 32, 1.0)


def test_weighted_counter_without_coincidence_preserves_mean_and_variance():
    from specter.microscope import Detector

    torch.manual_seed(22)
    # Weights normalize to sum=N inside the detector. For [1,3], weighted
    # summed variance is 1.25 times a plain exposure; mean is unchanged.
    w = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    detector = Detector(
        1.0,
        noise_model="poisson",
        n_frames=2,
        dose_weights=w,
        dose_weights_max_frequency=1.0,
        progressbars=False,
    )
    result = detector.apply_coincidence(torch.full((256, 256), 100.0), 100.0, 0.0)
    assert result.mean().item() == pytest.approx(100.0, abs=0.3)
    assert result.var().item() == pytest.approx(125.0, rel=0.025)


def test_match_uses_calibration_movies_and_exports_physical_parameter(tmp_path):
    from specter.config import MatchConfig
    from specter.pipelines._match import _base_settings

    k = np.linspace(0.22, 0.33, 5)
    edges = np.linspace(0, 40, 41)
    for split, s2 in [("calibration", 0.32), ("validation", 2.0)]:
        cov = np.array(
            [
                np.einsum("c,cij->ij", [3 + b, 0.2, 1.0], bases(edges, f, s2))
                for b, f in enumerate(k)
            ]
        )
        np.savez(
            tmp_path / f"{split}.npz",
            k=k,
            dose_edges=edges,
            cov_outer=cov,
            n_particles=20,
            split=split,
            counts=np.ones(5),
        )
    config = MatchConfig(
        metadata_path="unused.cs",
        pdb_source="1abc",
        dose=40.0,
        movie_covariance_pattern=str(tmp_path / "*.npz"),
        ice_cache_dir=str(tmp_path / "independent_water"),
    )
    values = _base_settings(config, {"cs_path": "unused.cs"}, 64, 50.0)
    assert values["ice_motion_variance"] == pytest.approx(0.32, rel=1e-4)
    assert values["ice_decorrelation_dose"] is None
    assert values["bfactor"] == 0.0
    assert values["ice_cache_dir"] == str(tmp_path / "independent_water")
