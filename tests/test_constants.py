import numpy as np
import pytest

from specter.constants import energy_to_wavelength, interaction_parameter


def test_energy_to_wavelength():
    # Reference value for 300 keV is 0.019687 Å
    wl = energy_to_wavelength(300.0)
    assert np.isclose(wl, 0.019687, atol=1e-5)


def test_interaction_parameter():
    # 300 kV -> sigma ~ 0.00065 rad/(V*Å)
    sigma = interaction_parameter(300.0)
    assert np.isclose(sigma, 0.00065, atol=1e-4)


def test_mott_bethe_prefactor_uses_codata_constants():
    """2 pi a0 e is Kirkland's 47.878 V*A^2, from CODATA a0 and e."""
    import math

    from specter.constants import bohr_radius, electron_charge_volt_angstrom

    assert bohr_radius() == pytest.approx(0.529177, abs=1e-6)
    assert electron_charge_volt_angstrom() == pytest.approx(14.39964, abs=1e-5)
    c1 = 2.0 * math.pi * bohr_radius() * electron_charge_volt_angstrom()
    assert c1 == pytest.approx(47.878, abs=1e-3)
