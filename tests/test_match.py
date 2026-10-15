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
    MatchedPoseSNR,
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


def test_k3_cds_frame_rate_raises_the_simulation_radius_by_sqrt2() -> None:
    """
    A K3 in correlated double sampling mode counts at 750 Hz, half its
    standard 1500 Hz: at the same dose rate the occupancy doubles and the
    simulation radius grows by sqrt(2), as `HARDWARE_FRAME_RATE_HZ` documents.
    """
    standard_hz = HARDWARE_FRAME_RATE_HZ["k3_300kv"]
    assert standard_hz == 1500.0
    # No K3 exclusion radius is calibrated; the ratios do not depend on it.
    standard = coincidence_occupancy(1.0, 15.0, standard_hz)
    cds = coincidence_occupancy(1.0, 15.0, 750.0)
    assert cds == pytest.approx(2 * standard)
    assert coincidence_radius_for_simulation(cds, 40.0, 1.0, 40) == pytest.approx(
        math.sqrt(2) * coincidence_radius_for_simulation(standard, 40.0, 1.0, 40)
    )


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


def test_experimental_images_load_in_the_recorded_polarity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    The same particles stored inverted (blob/sign -1) and as recorded (+1)
    load identically, in the recorded polarity; a stack whose metadata records
    no sign is taken to be inverted, as RELION and CryoSPARC write them.
    """
    from specter.io import _cryosparc
    from specter.match import load_experimental_images

    rng = np.random.default_rng(0)
    stored = rng.normal(size=(3, 8, 8)).astype(np.float32)

    def stack(name: str, data: np.ndarray) -> str:
        path = tmp_path / name
        with mrcfile.new(str(path), overwrite=True) as mrc:
            mrc.set_data(data)
        return str(path)

    def dataset(sign: float | None) -> type:
        class _Stub(dict):
            @classmethod
            def load(cls, csfile_path: str) -> "_Stub":
                table = {"uid": np.arange(3, dtype=np.uint64)}
                if sign is not None:
                    table["blob/sign"] = np.full(3, sign, dtype=np.float32)
                return cls(table)

        return _Stub

    cs = tmp_path / "particles.cs"
    cs.touch()
    loaded = {}
    for sign, data in ((-1.0, stored), (1.0, -stored), (None, stored)):
        monkeypatch.setattr(_cryosparc, "Dataset", dataset(sign))
        loaded[sign] = load_experimental_images(
            str(cs), images_path=stack(f"{sign}.mrcs", data)
        )
    recorded = torch.as_tensor(-stored)
    assert torch.equal(loaded[-1.0], recorded)
    assert torch.equal(loaded[1.0], recorded)
    assert torch.equal(loaded[None], recorded)


def test_pose_check_null_holds_at_every_shuffle_seed() -> None:
    """
    The z-score divides by the spread of the mean over many random pairings,
    so it does not hang on one shuffle. Against a single shuffled pairing
    taken as the exact null, unaligned 64-particle stacks passed at some
    seeds and failed at others; here a stack's z barely moves with the seed,
    and stacks with no pose information sit near zero, far below the pass
    threshold.
    """
    n, box, px = 64, 64, 2.0
    signal = _smooth_signal(n, box, seed=1)

    def noise(seed: int) -> torch.Tensor:
        return torch.randn(n, box, box, generator=torch.Generator().manual_seed(seed))

    exp = signal + 1.5 * noise(2)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(9))
    unaligned = signal[perm] + 1.5 * noise(4)
    z = [
        matched_index_correlation(unaligned, exp, px, seed=seed).z_score
        for seed in range(6)
    ]
    assert max(z) - min(z) < 0.5, z
    assert max(z) < 5.0, z

    nulls = [
        matched_index_correlation(_smooth_signal(n, box, seed=10 + k), exp, px)
        for k in range(8)
    ]
    assert all(abs(r.z_score) < 4.0 for r in nulls), [r.z_score for r in nulls]
    assert all(abs(r.fraction_above - 0.5) < 0.1 for r in nulls)


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
    # No envelope was applied, so the estimator must not claim one. Since it
    # gained a significance test it answers that by REFUSING (nan) rather than
    # by returning a small number, which is the stronger statement of the two;
    # the standard error is what carries the bound in that case.
    assert math.isnan(res.residual_bfactor) or abs(res.residual_bfactor) < 60.0
    assert res.residual_bfactor_stderr < 60.0
    twin = twin_test(sim1, sim2, exp)
    assert twin.cohen_d > 1.0  # the experiment is measurably noisier than a second seed


def test_matched_pose_snr_survives_a_ctf_zero_shared_by_the_stack() -> None:
    """A transfer zero inside a band, common to every particle, must not decide it.

    The signal goes through sin(pi (k/k0)^2), which is zero at k0 and
    sqrt(2) k0, both inside the 33-12 Å band, so the simulated cross-power
    is ~0 in those shells. Dividing per shell made such a band undefined, or
    gave an unbounded value, on EMPIAR-11752. The band sums must still recover
    the true noise ratio, and an exact-pose control must read 1.
    """
    n, box, px = 128, 64, 2.0
    k0 = 0.055
    k = torch.sqrt(
        torch.fft.fftfreq(box, d=px)[:, None] ** 2
        + torch.fft.fftfreq(box, d=px)[None, :] ** 2
    )
    transfer = torch.sin(math.pi * (k / k0) ** 2)
    white = torch.randn(n, box, box, generator=torch.Generator().manual_seed(21))
    signal = torch.fft.ifft2(torch.fft.fft2(white) * transfer).real

    def noise(seed: int) -> torch.Tensor:
        return 0.5 * torch.randn(
            n, box, box, generator=torch.Generator().manual_seed(seed)
        )

    sim1, sim2 = signal + noise(22), signal + noise(23)
    exp = signal + 2.0 * noise(24)
    res = matched_pose_snr(sim1, sim2, exp, px)
    control = matched_pose_snr(sim1, sim2, signal + noise(25), px)
    for band in range(4):  # the 4-3 Å band is beyond Nyquist at 2 Å
        assert res.ratio[band] == pytest.approx(4.0, rel=0.35)
        assert control.ratio[band] == pytest.approx(1.0, rel=0.25)
        assert res.ratio_lo[band] <= res.ratio[band] <= res.ratio_hi[band]
    # Same signal in both, measured on z-scored images: the amplitude ratio is
    # the ratio of the two stacks' standard deviations, not 1.
    assert res.signal_plateau == pytest.approx(float(sim1.std() / exp.std()), rel=0.1)


def test_verdict_counts_a_band_only_when_its_interval_clears_the_threshold() -> None:
    from specter.match import MatchReport, PoseAlignmentResult

    def report(ratio: float, lo: float, hi: float) -> MatchReport:
        snr = MatchedPoseSNR(
            snr_sim=[1.0] * 5,
            snr_exp=[1.0] * 5,
            ratio=[1.0, ratio, 1.0, 1.0, 1.0],
            ratio_lo=[0.9, lo, 0.9, 0.9, 0.9],
            ratio_hi=[1.1, hi, 1.1, 1.1, 1.1],
            signal_plateau=1.0,
            residual_bfactor=float("nan"),
            residual_bfactor_stderr=float("nan"),
        )
        return MatchReport(pose=PoseAlignmentResult(0.8, 0.1, 1.0, 10.0), snr=snr)

    # A point estimate of 5x whose interval reaches down to 2x is not a residual.
    assert report(5.0, 2.0, 9.0).verdict.startswith("match")
    resolved = report(5.0, 4.0, 6.5).verdict
    assert resolved.startswith("residual") and "4x to 6x" in resolved
    unbounded = report(float("inf"), 8.0, float("inf")).verdict
    assert "no measurable experimental signal" in unbounded


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
# Probe binning and the per-process potential cache
# ---------------------------------------------------------------------------


def test_probe_bin_is_capped_by_pixel_and_box() -> None:
    from specter.pipelines._match import _probe_bin

    assert _probe_bin(2, 360, 0.5695) == 2
    assert _probe_bin(2, 200, 1.025) == 2
    assert _probe_bin(2, 48, 2.5) == 1  # 24 px would be under the 32 px floor
    assert _probe_bin(4, 512, 1.5) == 3  # 6 A pixel is past the 5 A ceiling
    assert _probe_bin(1, 512, 1.0) == 1
    assert _probe_bin(0, 512, 1.0) == 1


def test_bin_images_crops_and_keeps_each_image_std() -> None:
    from specter.pipelines._match import _bin_images

    images = torch.randn(5, 64, 64, generator=torch.Generator().manual_seed(3))
    images[2] *= 3.0
    cropped, px = _bin_images(images, 1.5, 32)
    assert cropped.shape == (5, 32, 32)
    assert px == pytest.approx(3.0)
    torch.testing.assert_close(
        cropped.std(dim=(-2, -1)), images.std(dim=(-2, -1)), rtol=1e-4, atol=1e-5
    )
    same, px = _bin_images(images, 1.5, 64)
    assert same is images and px == 1.5


def test_particle_pipeline_reuses_the_potential_within_a_process() -> None:
    """Ten probes of one structure at one pixel size parse and render it once."""
    from specter.pipelines import _particles

    cfg = ParticleStackConfig(pdb_source="6bdf", n_pixels=32, pixel_size=3.0)
    _particles._POTENTIAL_CACHE.clear()
    pdb_a, v_a = _particles._structure_and_potential(cfg, 3.0, build=True)
    pdb_b, v_b = _particles._structure_and_potential(cfg, 3.0, build=True)
    assert pdb_b is pdb_a
    assert v_b is not v_a  # a copy, so a caller padding it cannot touch the cache
    torch.testing.assert_close(v_b, v_a)
    assert len(_particles._POTENTIAL_CACHE) == 1
    # A different pixel size is a different potential.
    _particles._structure_and_potential(cfg, 2.5, build=True)
    assert len(_particles._POTENTIAL_CACHE) == 2
    # A non-main DDP rank never builds and never fills the cache.
    _, zeros = _particles._structure_and_potential(cfg, 4.0, build=False)
    assert zeros.abs().sum() == 0 and len(_particles._POTENTIAL_CACHE) == 2
    _particles._POTENTIAL_CACHE.clear()


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
    passes, the matched config is crowded at a candidate spacing,
    and matched.toml loads back as a ParticleStackConfig."""
    from specter.config import load_config
    from specter.pipelines import run_match, run_particle_stack

    n, box, px = 16, 48, 2.5
    cs_path = tmp_path / "poses.cs"
    _write_csfile_with_random_poses(cs_path, n, px)
    # The 'experiment': the same forward model, neighbours one diameter apart
    # (the simulator's default), box-minimum ice.
    run_particle_stack(
        ParticleStackConfig(
            pdb_source="6bdf",
            n_pixels=box,
            cs_path=str(cs_path),
            n_particles=n,
            dose=30.0,
            ice_model="gd",
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
        crowd_candidates=[1.0],
        device="cpu",
        probe_workers=2,  # the spawned-worker path: warm-up, per-job logs, results by file
        write_stack=4,  # and the in-process path for the final stack
        seed=7,
        output_dir=str(tmp_path / "out"),
    )
    report = run_match(cfg)
    assert report.pose.passed, vars(report.pose)
    out = tmp_path / "out"
    assert (out / "probes" / "pose_check.log").exists()
    with mrcfile.open(out / "matched_particles.mrcs") as m:
        assert m.data.shape == (4, box, box)
    derived = {d.name: d.value for d in report.derived}
    # Always crowded: the one candidate, one diameter apart.
    assert derived["crowd_min_distance"] > 0
    assert derived["detector_model"] == "none"
    assert any("unknown" in w for w in report.warnings)
    assert (out / "match_report.md").exists() and (out / "match_report.png").exists()
    matched = load_config(str(out / "matched.toml"), ParticleStackConfig)
    assert matched.n_pixels == box and matched.cs_path == str(cs_path)
    assert matched.dose_envelope is True and matched.coincidence_radius == 0.0
    # Damage on the specimen, the solvent decorrelated by the fixed motion
    # variance, and absorption from measured mean free paths.
    assert matched.dose_envelope_target == "specimen"
    assert matched.ice_motion_variance == pytest.approx(0.38)
    assert matched.absorption_model == "inelastic_mfp"
    # A residual envelope is reported, never written as a B-factor.
    assert matched.bfactor is None


def test_report_preserves_angstrom_symbols_in_an_ascii_locale(tmp_path: Path) -> None:
    """Report files use UTF-8 even when a worker's locale uses ASCII."""
    import locale

    from specter.match import MatchReport, PoseAlignmentResult, render_report

    report = MatchReport(pose=PoseAlignmentResult(0.8, 0.1, 1.0, 10.0), pixel_size=2.5)
    images = torch.randn(5, 16, 16, generator=torch.Generator().manual_seed(0))
    previous_locale = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
        markdown_path, figure_path = render_report(
            report, images, images, str(tmp_path)
        )
    finally:
        locale.setlocale(locale.LC_CTYPE, previous_locale)

    assert "3.7 Å water-ring excess" in Path(markdown_path).read_text(encoding="utf-8")
    assert Path(figure_path).is_file()


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


def test_recorded_pixel_size_is_read_from_the_metadata(tmp_path: Path) -> None:
    """The movies' pixel comes from ``location/micrograph_psize_A`` when present."""
    from specter.pipelines._match import _recorded_pixel_size

    with_field = np.zeros(
        3, dtype=[("blob/psize_A", "f4"), ("location/micrograph_psize_A", "f4")]
    )
    with_field["blob/psize_A"], with_field["location/micrograph_psize_A"] = (
        0.7027,
        0.514,
    )
    without = np.zeros(3, dtype=[("blob/psize_A", "f4")])
    for name, arr in (("with.cs", with_field), ("without.cs", without)):
        with open(tmp_path / name, "wb") as fh:
            np.save(fh, arr)
    assert _recorded_pixel_size(str(tmp_path / "with.cs")) == pytest.approx(
        0.514, abs=1e-6
    )
    assert _recorded_pixel_size(str(tmp_path / "without.cs")) is None


@pytest.mark.parametrize("candidates", [[0.0, 1.0, 1.3], [-1.0], []])
def test_match_rejects_crowd_candidates_without_neighbours(
    candidates, tmp_path: Path
) -> None:
    """A matched config always has neighbours: a no-neighbour (or empty)
    candidate grid is refused at load, before any simulation runs."""
    from specter.config import validate_config

    metadata = tmp_path / "particles.cs"
    metadata.touch()
    cfg = MatchConfig(
        metadata_path=str(metadata),
        pdb_source="6bdf",
        dose=30.0,
        crowd_candidates=candidates,
        device="cpu",
    )
    with pytest.raises(ValueError, match="crowd_candidates"):
        validate_config(cfg)


def test_match_default_candidates_are_all_crowded() -> None:
    assert MatchConfig(
        metadata_path="x.cs", pdb_source="6bdf", dose=30.0
    ).crowd_candidates
    assert all(
        m > 0
        for m in MatchConfig(
            metadata_path="x.cs", pdb_source="6bdf", dose=30.0
        ).crowd_candidates
    )


def _background_stack(n: int, box: int, amplitude: float, seed: int) -> torch.Tensor:
    """Images whose structure outside a central disc scales with `amplitude`."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.arange(box) - box / 2, torch.arange(box) - box / 2, indexing="ij"
    )
    outside = (yy**2 + xx**2).sqrt() > box / 6
    smooth = torch.nn.functional.avg_pool2d(
        torch.randn(n, 1, box, box, generator=g), 9, stride=1, padding=4
    )[:, 0]
    disc = 3.0 * (~outside).float()
    return (
        disc
        + amplitude * smooth * outside
        + 0.05 * torch.randn(n, box, box, generator=g)
    )


def test_choose_probe_keeps_the_fallback_when_candidates_tie():
    """Two candidates drawn from the same distribution as each other: the data
    cannot separate them, so the first one stands and is marked unresolved
    unless the best happens to be it."""
    from specter.match import choose_probe

    exp = _background_stack(48, 64, 1.0, seed=0)
    a = _background_stack(48, 64, 1.5, seed=1)
    b = _background_stack(48, 64, 1.5, seed=2)
    choice = choose_probe([1.0, 1.3], [a, b], exp, first_bin=0, seed=3)
    assert choice.value == 1.0
    assert choice.best == 1.0 or not choice.distinguished
    assert all(e > 0 for e in choice.errors)


def test_choose_probe_takes_the_best_when_the_data_decide():
    """A candidate whose background matches the experiment beats a clearly
    wrong first candidate in essentially every replicate, and is taken."""
    from specter.match import choose_probe

    exp = _background_stack(48, 64, 1.0, seed=0)
    wrong = _background_stack(48, 64, 4.0, seed=1)
    right = _background_stack(48, 64, 1.0, seed=2)
    choice = choose_probe([0.0, 400.0], [wrong, right], exp, first_bin=0, seed=3)
    assert choice.value == 400.0 and choice.best == 400.0
    assert choice.distinguished
    assert choice.p_fallback_not_worse < 0.05
    assert choice.scores[1] < choice.scores[0]
