import numpy as np
import pytest
import torch

from specter.constants import energy_to_wavelength
from specter.io import _cryosparc, extract_parameters_from_csfile


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
                "ctf/tetra_A": np.zeros((n, 4), dtype=dtype),
                "alignments3D/alpha": np.ones(n, dtype=dtype),
                "ctf/anisomag": np.zeros((n, 4), dtype=dtype),
            }
        )


@pytest.fixture(autouse=True)
def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cryosparc, "Dataset", _FakeDataset)


def test_extract_parameters_all_particles() -> None:
    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        halfset_labels,
    ) = extract_parameters_from_csfile("fake.cs", return_class="all")

    assert rotations.shape == (6, 4)
    assert translations_A.shape == (6, 2)
    assert ctf_params["cs"].shape == (6,)
    assert torch.equal(indices, torch.arange(6))
    assert torch.equal(halfset_labels, torch.tensor([0, 1, 0, 1, 0, 1]))


def test_extract_parameters_by_class() -> None:
    (_, _, _, rotations, _, ctf_params, _, _, indices, halfset_labels) = (
        extract_parameters_from_csfile("fake.cs", return_class="1")
    )

    assert rotations.shape == (3, 4)
    assert ctf_params["cs"].shape == (3,)
    assert torch.equal(indices, torch.tensor([1, 3, 5]))
    assert halfset_labels is None


def test_extract_parameters_n_particles_truncates_after_class_filter() -> None:
    (_, _, _, rotations, _, ctf_params, _, _, indices, _) = (
        extract_parameters_from_csfile("fake.cs", return_class="1", n_particles=2)
    )

    assert rotations.shape == (2, 4)
    assert ctf_params["cs"].shape == (2,)
    assert torch.equal(indices, torch.tensor([1, 3]))


def test_extract_parameters_n_particles_all() -> None:
    (_, _, _, rotations, _, _, _, _, indices, halfset_labels) = (
        extract_parameters_from_csfile("fake.cs", return_class="all", n_particles=4)
    )

    assert rotations.shape == (4, 4)
    assert torch.equal(indices, torch.arange(4))
    assert torch.equal(halfset_labels, torch.tensor([0, 1, 0, 1]))


class _TrefoilTetrafoilDataset(_FakeDataset):
    """Same fixture as _FakeDataset, but with non-zero trefoil_A/tetra_A so
    the CryoSPARC -> specter unit conversion can be checked numerically."""

    @classmethod
    def load(cls, csfile_path: str) -> "_FakeDataset":
        ds = super().load(csfile_path)
        n = 6
        dtype = np.float32
        ds["ctf/trefoil_A"] = np.tile(np.array([500.0, -300.0], dtype=dtype), (n, 1))
        ds["ctf/tetra_A"] = np.tile(
            np.array([100.0, -200.0, 50.0, -75.0], dtype=dtype), (n, 1)
        )
        return ds


def test_trefoil_scaling_matches_cryosparc_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chi_trefoil = (2*pi/3) * wavelength^2 * trefoil_A -- derived from
    CryoSPARC's own newctf.py (params_to_coeffs_odd/gen_basis_odd), not the
    old hardcoded /1000 scale factor."""
    monkeypatch.setattr(_cryosparc, "Dataset", _TrefoilTetrafoilDataset)
    (_, _, _, _, _, ctf_params, _, _, _, _) = extract_parameters_from_csfile(
        "fake.cs", return_class="all"
    )

    wavelength = energy_to_wavelength(torch.tensor(300.0))
    expected1 = (2 * torch.pi / 3) * wavelength**2 * 500.0
    expected2 = (2 * torch.pi / 3) * wavelength**2 * -300.0

    assert torch.allclose(
        ctf_params["trefoil1"], torch.full((6,), expected1.item()), atol=1e-6
    )
    assert torch.allclose(
        ctf_params["trefoil2"], torch.full((6,), expected2.item()), atol=1e-6
    )


def test_tetrafoil_extraction_matches_cryosparc_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chi_tetrafoil coefficients derived from CryoSPARC's own newctf.py
    (params_to_coeffs_even/gen_basis_even): tetra_A[0]/[1] are the n=4 m=+-2
    secondary-astigmatism terms (prefactor 2*pi*wavelength^3, sign-flipped
    on index 0), tetra_A[2]/[3] are the true n=4 m=+-4 tetrafoil terms
    (prefactor pi/2*wavelength^3, sign-flipped on index 3)."""
    monkeypatch.setattr(_cryosparc, "Dataset", _TrefoilTetrafoilDataset)
    (_, _, _, _, _, ctf_params, _, _, _, _) = extract_parameters_from_csfile(
        "fake.cs", return_class="all"
    )

    wavelength = energy_to_wavelength(torch.tensor(300.0))
    tetra_A = torch.tensor([100.0, -200.0, 50.0, -75.0])
    expected1 = -2 * torch.pi * wavelength**3 * tetra_A[0]
    expected2 = 2 * torch.pi * wavelength**3 * tetra_A[1]
    expected3 = (torch.pi / 2) * wavelength**3 * tetra_A[2]
    expected4 = -(torch.pi / 2) * wavelength**3 * tetra_A[3]

    assert torch.allclose(
        ctf_params["tetrafoil1"], torch.full((6,), expected1.item()), atol=1e-6
    )
    assert torch.allclose(
        ctf_params["tetrafoil2"], torch.full((6,), expected2.item()), atol=1e-6
    )
    assert torch.allclose(
        ctf_params["tetrafoil3"], torch.full((6,), expected3.item()), atol=1e-6
    )
    assert torch.allclose(
        ctf_params["tetrafoil4"], torch.full((6,), expected4.item()), atol=1e-6
    )
