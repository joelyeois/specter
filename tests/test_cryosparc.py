import numpy as np
import pytest
import torch

from specter import cryosparc


class _FakeDataset(dict):
    @classmethod
    def load(cls, csfile_path: str) -> "_FakeDataset":
        n = 6
        dtype = np.float32
        rng = np.random.default_rng(0)
        return cls(
            {
                "alignments3D/shift": rng.normal(size=(n, 2)).astype(dtype),
                "alignments3D/psize_A": np.full(n, 1.5, dtype=dtype),
                "ctf/cs_mm": np.full(n, 2.7, dtype=dtype),
                "ctf/df_angle_rad": rng.normal(size=n).astype(dtype),
                "ctf/df1_A": (rng.normal(size=n) + 10000).astype(dtype),
                "ctf/df2_A": (rng.normal(size=n) + 10000).astype(dtype),
                "ctf/amp_contrast": np.full(n, 0.1, dtype=dtype),
                "ctf/accel_kv": np.full(n, 300.0, dtype=dtype),
                "alignments3D/pose": rng.normal(size=(n, 3)).astype(dtype),
                "alignments3D/split": np.array([0, 1, 0, 1, 0, 1]),
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


def test_extract_parameters_all_particles() -> None:
    (
        energy_kev,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        halfset_labels,
    ) = cryosparc.extract_parameters_from_csfile("fake.cs", return_class="all")

    assert rotations.shape == (6, 4)
    assert translations_A.shape == (6, 2)
    assert ctf_params["cs"].shape == (6,)
    assert torch.equal(indices, torch.arange(6))
    assert torch.equal(halfset_labels, torch.tensor([0, 1, 0, 1, 0, 1]))


def test_extract_parameters_by_class() -> None:
    (_, _, _, rotations, _, ctf_params, _, _, indices, halfset_labels) = (
        cryosparc.extract_parameters_from_csfile("fake.cs", return_class="1")
    )

    assert rotations.shape == (3, 4)
    assert ctf_params["cs"].shape == (3,)
    assert torch.equal(indices, torch.tensor([1, 3, 5]))
    assert halfset_labels is None


def test_extract_parameters_n_particles_truncates_after_class_filter() -> None:
    (_, _, _, rotations, _, ctf_params, _, _, indices, _) = (
        cryosparc.extract_parameters_from_csfile(
            "fake.cs", return_class="1", n_particles=2
        )
    )

    assert rotations.shape == (2, 4)
    assert ctf_params["cs"].shape == (2,)
    assert torch.equal(indices, torch.tensor([1, 3]))


def test_extract_parameters_n_particles_all() -> None:
    (_, _, _, rotations, _, _, _, _, indices, halfset_labels) = (
        cryosparc.extract_parameters_from_csfile(
            "fake.cs", return_class="all", n_particles=4
        )
    )

    assert rotations.shape == (4, 4)
    assert torch.equal(indices, torch.arange(4))
    assert torch.equal(halfset_labels, torch.tensor([0, 1, 0, 1]))
