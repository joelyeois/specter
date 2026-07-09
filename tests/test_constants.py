import numpy as np

from specter.constants import energy_to_wavelength, interaction_parameter


def test_energy_to_wavelength():
    # Reference value for 300 keV is 0.019687 Å
    wl = energy_to_wavelength(300.0)
    assert np.isclose(wl, 0.019687, atol=1e-5)


def test_interaction_parameter():
    # 300 kV -> sigma ~ 0.00065 rad/(V*Å)
    sigma = interaction_parameter(300.0)
    assert np.isclose(sigma, 0.00065, atol=1e-4)
