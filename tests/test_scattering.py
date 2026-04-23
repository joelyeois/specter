import numpy as np
import pytest
import torch

from specter.rotations import Rotation, build_affine_matrix, rotate_volume
from specter.scattering import (
    IterativeScattering,
    Scattering,
    complex_potential,
    energy_to_wavelength,
    interaction_parameter,
)


def test_energy_to_wavelength():
    # Reference value for 300 keV is 0.019687 Å
    wl = energy_to_wavelength(300.0)
    assert np.isclose(wl, 0.019687, atol=1e-5)


def test_interaction_parameter():
    # 300 kV -> sigma ~ 0.00065 rad/(V*Å)
    sigma = interaction_parameter(300.0)
    assert np.isclose(sigma, 0.00065, atol=1e-4)


def test_complex_potential():
    v = torch.ones(4, 4)
    alpha = 0.1
    cv = complex_potential(v, alpha=alpha)

    expected_real = np.sqrt(0.99)
    assert torch.allclose(
        cv.real, torch.tensor(expected_real, dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(cv.imag, torch.tensor(0.1, dtype=torch.float32), atol=1e-6)


def test_iterative_scattering_batch_size(dummy_volume):
    scat_iter = IterativeScattering(
        nxy=64,
        pixel_size=1.0,
        energy=300.0,
        scattering_model="multislice",
        progressbars=False,
    )

    R = (
        Rotation.from_rotvec(torch.tensor([[0.0, 0.1, 0.0]]))
        .as_matrix()
        .to(dummy_volume.device)
    )
    theta_matrix = build_affine_matrix(R)

    psi1 = scat_iter.forward(dummy_volume, theta_matrix, slice_batch_size=1)
    psi4 = scat_iter.forward(dummy_volume, theta_matrix, slice_batch_size=4)

    assert torch.allclose(psi1, psi4, atol=1e-5)

    scat_iter_rytov = IterativeScattering(
        nxy=64,
        pixel_size=1.0,
        energy=300.0,
        scattering_model="rytov",
        progressbars=False,
    )

    psi1_rytov = scat_iter_rytov.rytov(dummy_volume, theta_matrix, slice_batch_size=1)
    psi4_rytov = scat_iter_rytov.rytov(dummy_volume, theta_matrix, slice_batch_size=4)

    assert torch.allclose(psi1_rytov, psi4_rytov, atol=1e-5)


@pytest.fixture
def dummy_volume() -> torch.Tensor:
    vol = torch.zeros(1, 32, 64, 64)
    vol[0, 12:20, 24:40, 24:40] = 1.0
    return vol


@pytest.fixture
def padded_volume():
    """Single 3D volume (Z, Y, X) with non-zero content only in the central 16^3 block.

    The generous zero padding ensures rotated coordinates outside the volume
    boundary contribute nothing, making Scattering and IterativeScattering
    numerically comparable at arbitrary tilt angles.
    """
    vol = torch.zeros(64, 64, 64)
    vol[24:40, 24:40, 24:40] = 1.0
    return vol


@pytest.mark.parametrize(
    "angle_deg, atol",
    [
        (0.0, 1e-4),
        # 45° tolerance reflects interpolation noise between 3-D volume resample
        # (rotate_volume) and 2-D slice-by-slice sampling (IterativeScattering).
        (45.0, 5e-3),
    ],
)
def test_scattering_vs_iterative_consistency(padded_volume, angle_deg, atol):
    """Scattering with a pre-rotated volume matches IterativeScattering for the same pose."""
    nxy = 64

    theta_rad = torch.deg2rad(torch.tensor(angle_deg))
    c, s = torch.cos(theta_rad), torch.sin(theta_rad)
    R = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32
    ).unsqueeze(0)
    theta_matrix = build_affine_matrix(R)

    kwargs = dict(
        nxy=nxy,
        pixel_size=1.0,
        energy=300.0,
        scattering_model="projection",
        alpha=0.1,
        progressbars=False,
    )

    scat = Scattering(**kwargs)
    scat_iter = IterativeScattering(**kwargs)

    vol_rotated = rotate_volume(padded_volume, theta_matrix, padding_mode="zeros")
    psi_scat = scat.forward(vol_rotated)

    psi_iter = scat_iter.forward(padded_volume.unsqueeze(0), theta_matrix)

    assert psi_scat.shape == psi_iter.shape
    assert torch.allclose(
        psi_scat, psi_iter, atol=atol
    ), f"Max diff at {angle_deg}°: {(psi_scat - psi_iter).abs().max().item():.2e}"
