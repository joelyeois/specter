"""
Tests for the end-to-end pipeline classes in ghostbuster.py: Ghostbuster
(CryoSPARC .cs/.mrc loading + preprocessing) and TomogramGhostbuster
(tilt-series loading). None of these had any test coverage prior to this
file — only Reconstructor's forward/loss behaviour was tested.

CryoSPARC .cs loading is exercised via a fake `Dataset.load` (same pattern as
tests/test_cryosparc.py) since it is a file-format boundary, not physics; the
scattering/CTF math downstream is real.
"""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch

from specter.ghostbuster import (
    Ghostbuster,
    Reconstructor,
    TomogramGhostbuster,
    TomogramReconstructor,
)
from specter.io import _cryosparc
from specter.settings import Propagation
from conftest import fake_cryosparc_dataset

N_PARTICLES = 4
BOX = 8
PIXEL_SIZE = 1.5


_FakeDataset = fake_cryosparc_dataset(N_PARTICLES, PIXEL_SIZE)


@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cryosparc, "Dataset", _FakeDataset)


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
        propagation=Propagation(scattering_model="projection"),
    )

    assert gb._images.shape == (N_PARTICLES, BOX, BOX)
    assert gb._rotations.shape == (N_PARTICLES, 4)
    assert gb._translations.shape == (N_PARTICLES, 2)
    assert gb._voxel_size == pytest.approx(PIXEL_SIZE)

    with mrcfile.mmap(str(mrc_file)) as mrc:
        raw = torch.as_tensor(mrc.data.copy())
    dose_per_area = dose_per_angstrom * PIXEL_SIZE**2
    expected = dose_per_area**0.5 * (-raw) + dose_per_area
    # Exact, not approximate: the default must reproduce earlier runs.
    assert torch.equal(gb._images, expected)


def test_ghostbuster_counts_stack_is_used_as_is(mrc_file: Path) -> None:
    """image_units='counts' skips the sign flip and the dose mapping."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        image_units="counts",
        propagation=Propagation(scattering_model="projection"),
    )
    with mrcfile.mmap(str(mrc_file)) as mrc:
        raw = torch.as_tensor(mrc.data.copy())
    assert torch.equal(gb._images, raw)


@pytest.mark.parametrize(
    "halfset,expected_label",
    [("A", "A"), ("B", "B"), ("all", None)],
)
def test_ghostbuster_halfset_label_mapping(
    mrc_file: Path, halfset: str, expected_label: str | None
) -> None:
    """halfset maps to halfset_label as documented: 'A'->A, 'B'->B, 'all'->None."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        halfset=halfset,
        propagation=Propagation(scattering_model="projection"),
    )
    assert gb.halfset_label == expected_label


@pytest.mark.parametrize("alpha,expected", [(None, 0.1), (0.0, 0.0), (0.5, 0.5)])
def test_ghostbuster_alpha_overrides_cs(
    mrc_file: Path, alpha: float | None, expected: float
) -> None:
    """alpha, when given, replaces the .cs file's amp_contrast (0.1 here) in
    the forward model; unset, the .cs value is used. 0.0 is a real override,
    not a falsy "unset"."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        alpha=alpha,
        propagation=Propagation(scattering_model="projection"),
    )
    assert gb.propagation.alpha == pytest.approx(expected)


def test_ghostbuster_alpha_out_of_range_rejected(mrc_file: Path) -> None:
    """An alpha outside [0, 1] is refused before anything is loaded."""
    with pytest.raises(ValueError, match="alpha"):
        Ghostbuster(
            cs_file="fake.cs",
            mrc_file=str(mrc_file),
            dose_per_angstrom=2.0,
            alpha=1.5,
        )


def test_ghostbuster_test_run_executes(mrc_file: Path) -> None:
    """test_run() binning + one epoch completes and returns a trained Reconstructor."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        lr=0.1,
        batchsize=2,
        propagation=Propagation(scattering_model="projection"),
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
        lr=0.1,
        epochs=1,
        batchsize=2,
        propagation=Propagation(scattering_model="projection"),
    )
    model = gb.run()
    assert isinstance(model, Reconstructor)
    assert model.V.shape == (BOX, BOX, BOX)
    assert not torch.equal(model.V.data, torch.zeros(BOX, BOX, BOX))


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
        voltage=300.0,
        ctf_params=tomo_ctf_params,
        dose_per_angstrom=1.0,
        angles=[-20.0, 0.0, 20.0],
        lr=0.1,
        epochs=1,
        batchsize=3,
        propagation=Propagation(scattering_model="projection"),
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
        voltage=300.0,
        ctf_params=tomo_ctf_params,
        dose_per_angstrom=1.0,
        quaternions=quats,
        lr=0.1,
        propagation=Propagation(scattering_model="projection"),
    )
    model = tgb.test_run(bin_factor=2)
    assert isinstance(model, TomogramReconstructor)


def test_tomogram_ghostbuster_maps_a_normalised_series_with_per_tilt_dose(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    """A normalised series is flipped and mapped to counts tilt by tilt."""
    dose = torch.tensor([1.0, 2.0, 4.0])
    tgb = TomogramGhostbuster(
        tilt_series=tilt_series,
        voxel_size=2.0,
        voltage=300.0,
        ctf_params=tomo_ctf_params,
        dose_per_angstrom=dose,
        angles=[-20.0, 0.0, 20.0],
        image_units="normalized",
        propagation=Propagation(scattering_model="projection"),
    )
    n = (dose * 2.0**2).reshape(-1, 1, 1)
    assert torch.allclose(tgb._images, n**0.5 * (-tilt_series) + n)


def test_tomogram_ghostbuster_refuses_to_flip_counts(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    with pytest.raises(ValueError, match="flip_contrast"):
        TomogramGhostbuster(
            tilt_series=tilt_series,
            voxel_size=2.0,
            voltage=300.0,
            ctf_params=tomo_ctf_params,
            dose_per_angstrom=1.0,
            angles=[-20.0, 0.0, 20.0],
            flip_contrast=True,
        )


def test_tomogram_test_run_bins_counts_by_sum_and_potential_by_mean(
    tomo_ctf_params: dict[str, torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Binning keeps the electrons a region received and the potential's value.

    Counts add across the pooled pixels; a potential in volts is averaged, so
    the projected potential sum(V * dz) survives the coarser slicing.
    """
    counts = torch.full((3, BOX, BOX), 7.0)
    V_init = torch.full((BOX, BOX, BOX), 0.5)
    tgb = TomogramGhostbuster(
        tilt_series=counts,
        voxel_size=2.0,
        voltage=300.0,
        ctf_params=tomo_ctf_params,
        dose_per_angstrom=1.0,
        angles=[-20.0, 0.0, 20.0],
        V_init=V_init,
        propagation=Propagation(scattering_model="projection"),
    )
    binned, _ = tgb._bin_images(2)
    assert torch.allclose(binned, torch.full((3, BOX // 2, BOX // 2), 28.0))
    # Skip the fit: what is under test is the volume test_run starts from.
    monkeypatch.setattr(tgb, "_fit", lambda model, *args, **kwargs: model)
    model = tgb.test_run(bin_factor=2, device="cpu")
    assert torch.allclose(model.V, torch.full((BOX // 2,) * 3, 0.5))


def test_tomogram_ghostbuster_rejects_both_angles_and_quaternions(
    tilt_series: torch.Tensor, tomo_ctf_params: dict[str, torch.Tensor]
) -> None:
    with pytest.raises(ValueError, match="not both"):
        TomogramGhostbuster(
            tilt_series=tilt_series,
            voxel_size=2.0,
            voltage=300.0,
            ctf_params=tomo_ctf_params,
            dose_per_angstrom=1.0,
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
            voltage=300.0,
            ctf_params=tomo_ctf_params,
            dose_per_angstrom=1.0,
        )


# ---------------------------------------------------------------------------
# Row order is the contract, and blob/idx is what checks it
# ---------------------------------------------------------------------------


def _dataset_with_blobs(
    blob_idx: list[int], stack_name: str = "particles.mrcs"
) -> type:
    """_FakeDataset whose stack sits at ``blob_idx`` rather than in row order."""

    class _WithBlobs(_FakeDataset):  # type: ignore[valid-type, misc]
        @classmethod
        def load(cls, csfile_path: str) -> "_WithBlobs":
            ds = super().load(csfile_path)
            ds["blob/path"] = np.array([f"J1/restack/{stack_name}"] * N_PARTICLES)
            ds["blob/idx"] = np.asarray(blob_idx, dtype=np.int64)
            return ds

    return _WithBlobs


def _preprocessed(raw: torch.Tensor) -> torch.Tensor:
    dose_per_area = 2.0 * PIXEL_SIZE**2
    return dose_per_area**0.5 * -raw + dose_per_area


def test_ghostbuster_pairs_row_i_with_slice_i(mrc_file: Path) -> None:
    """The contract: row i of the .cs is slice i of the stack. The default
    fixture is well-formed, so nothing is reordered."""
    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        propagation=Propagation(scattering_model="projection"),
    )

    with mrcfile.open(str(mrc_file)) as mrc:
        raw = torch.as_tensor(mrc.data.copy())
    assert torch.allclose(gb._images, _preprocessed(raw))


def test_ghostbuster_refuses_a_stack_the_cs_file_says_is_not_in_row_order(
    monkeypatch: pytest.MonkeyPatch, mrc_file: Path
) -> None:
    """A Restack Particles job writes its stack in the order it reads its
    inputs, so row i is some other slice. Reading it by row pairs every pose
    with another particle's image, and says nothing: the images are real
    particles, the loss falls, the map is mush. blob/idx is what makes that
    detectable, so it is checked before the stack is trusted."""
    monkeypatch.setattr(_cryosparc, "Dataset", _dataset_with_blobs([2, 3, 0, 1]))

    with pytest.raises(ValueError, match="not in row order"):
        Ghostbuster(
            cs_file="fake.cs",
            mrc_file=str(mrc_file),
            dose_per_angstrom=2.0,
            propagation=Propagation(scattering_model="projection"),
        )


def test_ghostbuster_reads_such_a_stack_in_place_when_asked(
    monkeypatch: pytest.MonkeyPatch, mrc_file: Path
) -> None:
    """address_by_blob_idx points straight at a restack without exporting a
    row-ordered copy of it first."""
    rotated = [2, 3, 0, 1]
    monkeypatch.setattr(_cryosparc, "Dataset", _dataset_with_blobs(rotated))

    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        address_by_blob_idx=True,
        propagation=Propagation(scattering_model="projection"),
    )

    with mrcfile.open(str(mrc_file)) as mrc:
        raw = torch.as_tensor(mrc.data.copy())
    assert torch.allclose(gb._images, _preprocessed(raw[rotated]))


def test_ghostbuster_trusts_row_order_when_there_is_nothing_to_check(
    monkeypatch: pytest.MonkeyPatch, mrc_file: Path
) -> None:
    """A passthrough file separated from its siblings carries no blob columns,
    which is the ordinary state of a dataset copied off the machine that made
    it. Row order is still the contract; it just cannot be verified."""

    class _NoBlobs(_FakeDataset):  # type: ignore[valid-type, misc]
        @classmethod
        def load(cls, csfile_path: str) -> "_NoBlobs":
            ds = super().load(csfile_path)
            for column in ("blob/path", "blob/idx"):
                ds.pop(column)
            return ds

    monkeypatch.setattr(_cryosparc, "Dataset", _NoBlobs)

    gb = Ghostbuster(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        propagation=Propagation(scattering_model="projection"),
    )

    with mrcfile.open(str(mrc_file)) as mrc:
        raw = torch.as_tensor(mrc.data.copy())
    assert torch.allclose(gb._images, _preprocessed(raw))


def test_ghostbuster_refuses_particles_spread_over_several_stack_files(
    monkeypatch: pytest.MonkeyPatch, mrc_file: Path
) -> None:
    """Particles drawn from several stacks cannot be in one stack's row order,
    and one mrc_file cannot address them at blob/idx either."""

    class _TwoStacks(_FakeDataset):  # type: ignore[valid-type, misc]
        @classmethod
        def load(cls, csfile_path: str) -> "_TwoStacks":
            ds = super().load(csfile_path)
            ds["blob/path"] = np.array(["J1/a.mrc", "J1/a.mrc", "J1/b.mrc", "J1/b.mrc"])
            ds["blob/idx"] = np.array([0, 1, 0, 1], dtype=np.int64)
            return ds

    monkeypatch.setattr(_cryosparc, "Dataset", _TwoStacks)
    common = dict(
        cs_file="fake.cs",
        mrc_file=str(mrc_file),
        dose_per_angstrom=2.0,
        propagation=Propagation(scattering_model="projection"),
    )

    with pytest.raises(ValueError, match="not in row order"):
        Ghostbuster(**common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2 stack files"):
        Ghostbuster(**common, address_by_blob_idx=True)  # type: ignore[arg-type]
