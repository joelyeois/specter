import numpy as np
import torch

from specter.rotations import Rotation, build_affine_matrix
from specter.scattering import (
    IterativeScattering,
    Scattering,
    complex_potential,
    energy_to_wavelength,
    interaction_parameter,
)


def test_energy_to_wavelength():
    # 300 kV -> ~0.0197 Å
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

    # Real part = sqrt(1 - 0.1^2) = sqrt(0.99) ≈ 0.994987
    # Imag part = 0.1
    expected_real = np.sqrt(0.99)
    assert torch.allclose(
        cv.real, torch.tensor(expected_real, dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(cv.imag, torch.tensor(0.1, dtype=torch.float32), atol=1e-6)


def test_scattering_initialization():
    nxy = 64
    pixel_size = 1.0
    energy = 300.0
    dose = 1.0

    scat = Scattering(
        nxy=nxy,
        pixel_size=pixel_size,
        energy=energy,
        dose_per_angstrom=dose,
        scattering_model="multislice",
    )

    assert scat.nxy == nxy
    assert scat.wavelength > 0
    assert hasattr(scat, "F_real")
    assert hasattr(scat, "F_imag")


def test_scattering_projection_simple(dummy_volume):
    # Use the dummy_volume fixture from conftest.py
    scat = Scattering(
        nxy=64,
        pixel_size=1.0,
        energy=300.0,
        dose_per_angstrom=1.0,
        scattering_model="projection",
        alpha=0.1,
    )

    exitwave = scat.forward(dummy_volume)
    assert exitwave.shape == (1, 64, 64)
    assert exitwave.dtype == torch.complex64
    # Intensity should be close to 1.0 where there is no potential
    intensity = torch.abs(exitwave) ** 2
    assert torch.allclose(intensity[0, 0, 0], torch.tensor(1.0), atol=1e-5)


def test_iterative_scattering_batch_size(dummy_volume):
    nxy = 64
    pixel_size = 1.0
    energy = 300.0
    dose = 100.0

    # Use different slice_batch_sizes
    s_iter = IterativeScattering(
        nxy=nxy,
        pixel_size=pixel_size,
        energy=energy,
        dose_per_angstrom=dose,
        scattering_model="multislice",
    )

    # Non-identity matrix to trigger batch sampling
    R = (
        Rotation.from_rotvec(torch.tensor([[0.0, 0.1, 0.0]]))
        .as_matrix()
        .to(dummy_volume.device)
    )
    theta_matrix = build_affine_matrix(R)

    # Compare batch size 1 vs batch size 4
    psi1 = s_iter.forward(dummy_volume, theta_matrix, slice_batch_size=1)
    psi4 = s_iter.forward(dummy_volume, theta_matrix, slice_batch_size=4)

    assert torch.allclose(psi1, psi4, atol=1e-5)

    # Check for another model like Rytov
    s_iter_rytov = IterativeScattering(
        nxy=nxy,
        pixel_size=pixel_size,
        energy=energy,
        dose_per_angstrom=dose,
        scattering_model="rytov",
    )

    psi1_rytov = s_iter_rytov.rytov(dummy_volume, theta_matrix, slice_batch_size=1)
    psi4_rytov = s_iter_rytov.rytov(dummy_volume, theta_matrix, slice_batch_size=4)

    assert torch.allclose(psi1_rytov, psi4_rytov, atol=1e-5)
