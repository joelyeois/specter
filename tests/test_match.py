"""`specter match particles`: the physics conversions, the matched-pose metrics, and the pipeline."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from cryosparc.dataset import Dataset

from specter.config import MatchConfig, ParticleStackConfig
from specter.detectors import (
    EXCLUSION_RADIUS_PX,
    HARDWARE_FRAME_RATE_HZ,
    coincidence_occupancy,
    coincidence_radius_for_simulation,
)
from specter.match import (
    dumps_toml,
    edge_band_means,
    matched_index_correlation,
    matched_pose_snr,
    twin_test,
)

# ---------------------------------------------------------------------------
# Detector physics
# ---------------------------------------------------------------------------


def test_coincidence_conversion_matches_worked_examples() -> None:
    """The three datasets worked by hand on 2026-09-03, with the 2.0 px Falcon radius."""
    # EMPIAR-11377: Falcon 4, 4.1 e/px/s, 1293 fractions in 5.2 s; simulated at
    # 1.462 A with 40 frames at 40 e/A^2.
    occ = coincidence_occupancy(EXCLUSION_RADIUS_PX["falcon4i_300kv"], 4.1, 1293 / 5.2)
    assert occ == pytest.approx(0.207, abs=0.003)
    assert coincidence_radius_for_simulation(occ, 40.0, 1.462, 40) == pytest.approx(
        0.176, abs=0.003
    )
    # EMPIAR-10025: K2 at ~12 e/physical px/s and 400 fps; 1.1301 A, 38 frames, 53 e/A^2.
    occ = coincidence_occupancy(
        EXCLUSION_RADIUS_PX["k2_300kv"], 12.0, HARDWARE_FRAME_RATE_HZ["k2_300kv"]
    )
    assert occ == pytest.approx(0.608, abs=0.005)
    assert coincidence_radius_for_simulation(occ, 53.0, 1.1301, 38) == pytest.approx(
        0.329, abs=0.003
    )
    # Holding occupancy fixed is the invariant: pixel size and frame count cancel.
    r_a = coincidence_radius_for_simulation(0.3, 40.0, 1.0, 40)
    r_b = coincidence_radius_for_simulation(0.3, 40.0, 2.0, 160)
    assert r_a == pytest.approx(r_b)
    assert coincidence_radius_for_simulation(0.0, 40.0, 1.0, 40) == 0.0


def test_every_calibrated_detector_has_a_frame_rate() -> None:
    for name in EXCLUSION_RADIUS_PX:
        assert name in HARDWARE_FRAME_RATE_HZ


# ---------------------------------------------------------------------------
# Matched-pose metrics on synthetic stacks
# ---------------------------------------------------------------------------


def _smooth_signal(n: int, box: int, seed: int, cutoff: float = 0.15) -> torch.Tensor:
    """Per-particle band-limited random 'structure', unit variance."""
    g = torch.Generator().manual_seed(seed)
    white = torch.randn(n, box, box, generator=g)
    k = torch.fft.fftfreq(box)
    ky, kx = torch.meshgrid(k, k, indexing="ij")
    mask = (torch.sqrt(kx**2 + ky**2) < cutoff).float()
    s = torch.fft.ifft2(torch.fft.fft2(white) * mask).real
    return s / s.std(dim=(-2, -1), keepdim=True)


def test_matched_index_correlation_separates_aligned_from_misaligned() -> None:
    n, box, px = 64, 64, 2.0
    signal = _smooth_signal(n, box, seed=1)

    def noise(seed: int) -> torch.Tensor:
        return torch.randn(n, box, box, generator=torch.Generator().manual_seed(seed))

    exp = signal + 1.5 * noise(2)
    aligned = matched_index_correlation(signal + 1.5 * noise(3), exp, px)
    assert aligned.passed, vars(aligned)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(9))
    misaligned = matched_index_correlation(signal[perm] + 1.5 * noise(4), exp, px)
    assert not misaligned.passed, vars(misaligned)


def test_matched_pose_snr_recovers_a_known_noise_ratio() -> None:
    """exp has 2x the noise amplitude of sim -> SNR ratio sim/exp = 4 in every band."""
    n, box, px = 128, 64, 2.0
    signal = torch.randn(
        n, box, box, generator=torch.Generator().manual_seed(11)
    )  # white 'structure'

    def noise(seed: int) -> torch.Tensor:
        return torch.randn(n, box, box, generator=torch.Generator().manual_seed(seed))

    sim1, sim2 = signal + noise(12), signal + noise(13)
    exp = signal + 2.0 * noise(14)
    res = matched_pose_snr(sim1, sim2, exp, px)
    for r in res.ratio:
        if not math.isnan(r):
            assert r == pytest.approx(4.0, rel=0.35)
    assert abs(res.residual_bfactor) < 60.0  # no envelope was applied
    twin = twin_test(sim1, sim2, exp)
    assert twin.cohen_d > 1.0  # the experiment is measurably noisier than a second seed


def test_edge_band_means_flag_a_shared_border_pattern() -> None:
    n, box = 32, 64
    plain = torch.randn(n, box, box, generator=torch.Generator().manual_seed(5))
    assert max(abs(v) for v in edge_band_means(plain)) < 0.05
    ringed = plain.clone()
    ringed[:, :2, :] -= 1.0
    ringed[:, -2:, :] -= 1.0
    ringed[:, :, :2] -= 1.0
    ringed[:, :, -2:] -= 1.0
    assert edge_band_means(ringed)[0] < -0.5


def test_toml_writer_round_trips_through_tomllib() -> None:
    text = dumps_toml(
        {
            "specimen": {"pdb_source": "8b0x", "n_pixels": 256, "ice_thickness": 700.0},
            "microscope": {"dose": 40.0, "dose_envelope": True, "bfactor": None},
            "advanced": {"ice_candidates": [0.0, 400.0]},
        },
        header="written by a test\nsecond line",
    )
    data = tomllib.loads(text)
    assert data["specimen"] == {
        "pdb_source": "8b0x",
        "n_pixels": 256,
        "ice_thickness": 700.0,
    }
    assert data["microscope"] == {
        "dose": 40.0,
        "dose_envelope": True,
    }  # None is commented out
    assert data["advanced"]["ice_candidates"] == [0.0, 400.0]
    assert text.startswith("# written by a test\n# second line\n")


# ---------------------------------------------------------------------------
# End to end on a synthetic experiment specter made itself
# ---------------------------------------------------------------------------


def _write_csfile_with_random_poses(
    path: Path, n: int, pixel_size: float, box: int | None = None, shift: float = 0.0
) -> None:
    g = np.random.default_rng(0)
    rotvec = g.normal(size=(n, 3)).astype(np.float32)
    rotvec *= (np.pi * g.uniform(0.2, 1.0, size=(n, 1))).astype(
        np.float32
    ) / np.linalg.norm(rotvec, axis=1, keepdims=True)
    blob = (
        {
            "blob/shape": np.full((n, 2), box, dtype=np.uint32),
            "blob/psize_A": np.full(n, pixel_size, dtype=np.float32),
        }
        if box is not None
        else {}
    )
    Dataset(
        {
            **blob,
            "uid": np.arange(n, dtype=np.uint64),
            "alignments3D/shift": np.full((n, 2), shift, dtype=np.float32),
            "alignments3D/psize_A": np.full(n, pixel_size, dtype=np.float32),
            "alignments3D/pose": rotvec,
            "alignments3D/split": np.zeros(n, dtype=np.uint32),
            "alignments3D/alpha": np.ones(n, dtype=np.float32),
            "ctf/cs_mm": np.full(n, 2.7, dtype=np.float32),
            "ctf/df_angle_rad": np.zeros(n, dtype=np.float32),
            "ctf/df1_A": g.uniform(8000, 15000, size=n).astype(np.float32),
            "ctf/df2_A": g.uniform(8000, 15000, size=n).astype(np.float32),
            "ctf/amp_contrast": np.full(n, 0.1, dtype=np.float32),
            "ctf/accel_kv": np.full(n, 300.0, dtype=np.float32),
            "ctf/tilt_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/phase_shift_rad": np.zeros(n, dtype=np.float32),
            "ctf/shift_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/trefoil_A": np.zeros((n, 2), dtype=np.float32),
            "ctf/tetra_A": np.zeros((n, 4), dtype=np.float32),
            "ctf/anisomag": np.zeros((n, 4), dtype=np.float32),
        }
    ).save(str(path))


def test_run_match_on_a_synthetic_experiment(tmp_path: Path) -> None:
    """A stack specter simulated itself is matched by the recipe: the pose check
    passes, the neighbour-free experiment picks the neighbour-free candidate,
    and matched.toml loads back as a ParticleStackConfig."""
    from specter.config import load_config
    from specter.pipelines import run_match, run_particle_stack

    n, box, px = 16, 48, 2.5
    cs_path = tmp_path / "poses.cs"
    _write_csfile_with_random_poses(cs_path, n, px)
    # The 'experiment': the same forward model, no neighbours, box-minimum ice.
    run_particle_stack(
        ParticleStackConfig(
            pdb_source="6bdf",
            n_pixels=box,
            cs_path=str(cs_path),
            n_particles=n,
            dose=30.0,
            ice_model="gd",
            crowd_min_distance=0,
            detector_model="none",
            dose_envelope=True,
            device="cpu",
            batchsize=4,
            seed=123,
            output_dir=str(tmp_path),
            filename="experiment",
        )
    )
    with mrcfile.open(tmp_path / "experiment.mrcs") as m:
        assert m.data.shape == (n, box, box)

    cfg = MatchConfig(
        metadata_path=str(cs_path),
        pdb_source="6bdf",
        dose=30.0,
        images_path=str(tmp_path / "experiment.mrcs"),
        detector_model="unknown",
        n_probe=n,
        n_battery=n,
        ice_candidates=[0.0],
        crowd_candidates=[0.0, 1.0],
        device="cpu",
        seed=7,
        output_dir=str(tmp_path / "out"),
    )
    report = run_match(cfg)
    assert report.pose.passed, vars(report.pose)
    derived = {d.name: d.value for d in report.derived}
    assert derived["crowd_min_distance"] == 0
    assert derived["detector_model"] == "none"
    assert any("unknown" in w for w in report.warnings)
    out = tmp_path / "out"
    assert (out / "match_report.md").exists() and (out / "match_report.png").exists()
    matched = load_config(str(out / "matched.toml"), ParticleStackConfig)
    assert matched.n_pixels == box and matched.cs_path == str(cs_path)
    assert matched.dose_envelope is True and matched.coincidence_radius == 0.0


def test_rescale_metadata_follows_a_fourier_cropped_stack(tmp_path: Path) -> None:
    """A .cs extracted at 360 px / 0.5695 A describes 200 px images at 1.0251 A;
    pixel-unit shifts scale the other way so shifts in Angstrom are unchanged."""
    from specter.match import recorded_box, rescale_metadata

    src = tmp_path / "orig.cs"
    n = 4
    _write_csfile_with_random_poses(src, n, 0.5695, box=360, shift=18.0)
    assert recorded_box(str(src)) == 360
    out = tmp_path / "rescaled.cs"
    new_px = rescale_metadata(str(src), 200, str(out))
    assert new_px == pytest.approx(0.5695 * 360 / 200, rel=1e-5)
    r = Dataset.load(str(out))
    assert recorded_box(str(out)) == 200
    assert float(r["alignments3D/psize_A"][0]) == pytest.approx(new_px, rel=1e-5)
    assert float(r["alignments3D/shift"][0][0]) == pytest.approx(10.0, rel=1e-5)
    # Angstrom shift is invariant: 18 px * 0.5695 == 10 px * 1.0251
    assert 18.0 * 0.5695 == pytest.approx(10.0 * new_px, rel=1e-5)
