import numpy as np
import pytest
import starfile
import torch

from specter import cryosparc
from specter.imagegenerator import ImageGenerator


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


def test_create_particle_starfile_writes_bfactor_column(tmp_path) -> None:
    n = 3
    ctf_params = {
        "dfu": torch.full((n,), 5000.0),
        "dfv": torch.full((n,), 5000.0),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), 2.7),
        "phaseshift": torch.zeros(n),
    }
    cryosparc.create_particle_starfile(
        torch.randn(n, 8, 8),
        rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n),
        translations=torch.zeros(n, 2),
        alpha=0.1,
        folderpath=str(tmp_path),
        energy=300.0,
        dx=1.5,
        filename="particles",
        ctf_params=ctf_params,
        dose_per_angstrom=2.0,
        coincidence_radius=1.8,
        potential_scale=0.75,
        bfactor=42.0,
    )

    df = starfile.read(tmp_path / "particles.star")
    assert (df["specterBfactor"] == 42.0).all()
    assert (df["specterPotentialScale"] == 0.75).all()
    assert (df["specterCoincidenceRadius"] == 1.8).all()
    assert (df["specterDosePerAngstrom"] == 2.0).all()


def test_create_particle_starfile_from_model_matches_model_params(tmp_path) -> None:
    volume = torch.zeros(16, 16, 16)
    volume[6:10, 6:10, 6:10] = 50.0
    ctf_params = {
        "dfu": torch.tensor([5000.0]),
        "dfv": torch.tensor([5000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.0]),
    }

    torch.manual_seed(0)
    model = ImageGenerator(
        scattering_potential=volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[0.7071, 0.0, 0.7071, 0.0]]),
        translations=torch.tensor([[1.0, -1.0]]),
        ctf_params=ctf_params,
        energy=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="multislice",
        alpha=0.1,
        coincidence_radius=1.8,
        potential_scale=0.75,
        bfactor=42.0,
        verbose=False,
        progressbars=False,
    )
    particles = model(torch.tensor([0]))

    cryosparc.create_particle_starfile_from_model(
        particles, model, folderpath=str(tmp_path), filename="particles"
    )

    df = starfile.read(tmp_path / "particles.star")
    assert (df["specterBfactor"] == 42.0).all()
    assert (df["specterPotentialScale"] == 0.75).all()
    assert (df["specterCoincidenceRadius"] == 1.8).all()
    assert (df["specterDosePerAngstrom"] == 2.0).all()
    # Multislice internally shifts dfu/dfv to the volume's midplane (see
    # _apply_defocus_shift); the saved STAR file must have the original,
    # externally-meaningful defocus, not the internally-shifted one.
    assert (df["rlnDefocusU"] == 5000.0).all()
    assert (df["rlnImagePixelSize"] == 2.0).all()
