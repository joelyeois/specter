"""
Tests for the end-to-end pipeline classes in ghostbuster.py: Ghostbuster
(CryoSPARC .cs/.mrc loading + preprocessing), TomogramGhostbuster (tilt-series
loading), and compare_runs. None of these had any test coverage prior to this
file — only Reconstructor's forward/loss behaviour was tested.

CryoSPARC .cs loading is exercised via a fake `Dataset.load` (same pattern as
tests/test_cryosparc.py) since it is a file-format boundary, not physics; the
scattering/CTF math downstream is real.
"""

from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch

from specter import cryosparc
from specter.ghostbuster import (
    Ghostbuster,
    Reconstructor,
    TomogramGhostbuster,
    TomogramReconstructor,
    compare_runs,
)

# ---------------------------------------------------------------------------
# Fake CryoSPARC dataset (mirrors tests/test_cryosparc.py's _FakeDataset)
# ---------------------------------------------------------------------------

N_PARTICLES = 4
BOX = 8
PIXEL_SIZE = 1.5


class _FakeDataset(dict):
    @classmethod
    def load(cls, csfile_path: str) -> "_FakeDataset":
        n = N_PARTICLES
        dtype = np.float32
        rng = np.random.default_rng(0)
        return cls(
            {
                "alignments3D/shift": rng.normal(size=(n, 2)).astype(dtype),
                "alignments3D/psize_A": np.full(n, PIXEL_SIZE, dtype=dtype),
                "ctf/cs_mm": np.full(n, 2.7, dtype=dtype),
                "ctf/df_angle_rad": rng.normal(size=n).astype(dtype),
                "ctf/df1_A": (rng.normal(size=n) + 10000).astype(dtype),
                "ctf/df2_A": (rng.normal(size=n) + 10000).astype(dtype),
                "ctf/amp_contrast": np.full(n, 0.1, dtype=dtype),
                "ctf/accel_kv": np.full(n, 300.0, dtype=dtype),
                "alignments3D/pose": rng.normal(size=(n, 3)).astype(dtype),
                "alignments3D/split": np.array([0, 1, 0, 1][:n]),
                "ctf/tilt_A": np.zeros((n, 2), dtype=dtype),
                "ctf/phase_shift_rad": np.zeros(n, dtype=dtype),
                "ctf/shift_A": np.zeros((n, 2), dtype=dtype),
                "ctf/trefoil_A": np.zeros((n, 2), dtype=dtype),
                "alignments3D/alpha": np.ones(n, dtype=dtype),
                "ctf/anisomag": np.zeros((n, 4), dtype=dtype),
            }
        )


@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cryosparc, "Dataset", _FakeDataset)


@pytest.fixture
def mrc_file(tmp_path: Path) -> Path:
    """A synthetic particle stack matching _FakeDataset's particle count/box."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(N_PARTICLES, BOX, BOX)).astype(np.float32)
    path = tmp_path / "particles.mrcs"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
    return path


# ---------------------------------------------------------------------------
# Ghostbuster: loading and preprocessing
# ---------------------------------------------------------------------------


def test_ghostbuster_loads_and_preprocesses_images(mrc_file: Path) -> None:
    """Ghostbuster extracts .cs parameters, loads the .mrc stack, and applies
    the sign-flip + dose-scaling preprocessing formula."""
    dose_per_angstrom = 2.0
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=dose_per_angstrom,
        scattering_model="projection",
    )

    assert gb._images.shape == (N_PARTICLES, BOX, BOX)
    assert gb._rotations.shape == (N_PARTICLES, 4)
    assert gb._translations.shape == (N_PARTICLES, 2)
    assert gb._voxel_size == pytest.approx(PIXEL_SIZE)

    with mrcfile.mmap(str(mrc_file)) as mrc:
        raw = torch.as_tensor(mrc.data.copy())
    dose_per_area = dose_per_angstrom * PIXEL_SIZE**2
    expected = dose_per_area**0.5 * (-raw) + dose_per_area
    assert torch.allclose(gb._images, expected)


@pytest.mark.parametrize(
    "return_class,expected_label",
    [("0", "A"), ("1", "B"), ("all", None)],
)
def test_ghostbuster_halfset_label_mapping(
    mrc_file: Path, return_class: str, expected_label: str | None
) -> None:
    """return_class maps to halfset_label as documented: '0'->A, '1'->B, 'all'->None."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        scattering_model="projection",
        return_class=return_class,
    )
    assert gb.halfset_label == expected_label


def test_ghostbuster_test_run_executes(mrc_file: Path) -> None:
    """test_run() binning + one epoch completes and returns a trained Reconstructor."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        batch_size=2,
    )
    model = gb.test_run(bin_factor=2)
    assert isinstance(model, Reconstructor)
    assert model.V.shape == (BOX // 2, BOX // 2, BOX // 2)


def test_ghostbuster_run_executes(mrc_file: Path) -> None:
    """run() completes one full (unbinned) epoch and updates the volume."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        scattering_model="projection",
        lr=0.1,
        epochs=1,
        batch_size=2,
    )
    model = gb.run()
    assert isinstance(model, Reconstructor)
    assert model.V.shape == (BOX, BOX, BOX)
    assert not torch.equal(model.V.data, torch.zeros(BOX, BOX, BOX))


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


def test_compare_runs_prints_summary_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """compare_runs prints one row per run directory with the requested columns."""
    for name, params in [
        ("run1", {"scattering_model": "projection", "lr": 0.1}),
        ("run2", {"scattering_model": "multislice", "lr": 0.05}),
    ]:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "params.json").write_text(json.dumps(params))

    compare_runs(base_dir=tmp_path, show_params=["scattering_model", "lr"])
    out = capsys.readouterr().out
    assert "run1" in out and "run2" in out
    assert "projection" in out and "multislice" in out


def test_compare_runs_reports_no_runs_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty base_dir prints a 'no runs found' message instead of a table."""
    compare_runs(base_dir=tmp_path)
    out = capsys.readouterr().out
    assert "No runs found" in out


# ---------------------------------------------------------------------------
# TomogramGhostbuster: angle/quaternion resolution and run/test_run
# ---------------------------------------------------------------------------


@pytest.fixture
def tilt_series() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(3, BOX, BOX)


@pytest.fixture
def tomo_ctf_params() -> dict[str, torch.Tensor]:
    n = 3
    return {"dfu": torch.full((n,), 5000.0), "cs": torch.full((n,), 2.7)}


def test_tomogram_ghostbuster_angles_path(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    """Providing `angles` resolves to per-tilt quaternions and runs end-to-end."""
    tgb = TomogramGhostbuster(
        tilt_series=tilt_series,
        voxel_size=2.0,
        energy=300.0,
        ctf_params=tomo_ctf_params,
        angles=[-20.0, 0.0, 20.0],
        scattering_model="projection",
        lr=0.1,
        epochs=1,
        batch_size=3,
    )
    assert tgb._quaternions.shape == (3, 4)
    model = tgb.run()
    assert isinstance(model, TomogramReconstructor)


def test_tomogram_ghostbuster_quaternions_path(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    """Providing `quaternions` directly is accepted as an alternative to `angles`."""
    quats = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    tgb = TomogramGhostbuster(
        tilt_series=tilt_series,
        voxel_size=2.0,
        energy=300.0,
        ctf_params=tomo_ctf_params,
        quaternions=quats,
        scattering_model="projection",
        lr=0.1,
    )
    model = tgb.test_run(bin_factor=2)
    assert isinstance(model, TomogramReconstructor)


def test_tomogram_ghostbuster_rejects_both_angles_and_quaternions(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    with pytest.raises(ValueError, match="not both"):
        TomogramGhostbuster(
            tilt_series=tilt_series,
            voxel_size=2.0,
            energy=300.0,
            ctf_params=tomo_ctf_params,
            angles=[0.0, 10.0, 20.0],
            quaternions=torch.zeros(3, 4),
        )


def test_tomogram_ghostbuster_requires_angles_or_quaternions(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    with pytest.raises(ValueError, match="must be provided"):
        TomogramGhostbuster(
            tilt_series=tilt_series,
            voxel_size=2.0,
            energy=300.0,
            ctf_params=tomo_ctf_params,
        )
