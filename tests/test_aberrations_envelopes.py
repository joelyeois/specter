from __future__ import annotations

import torch

from specter.aberrations import Aberration
from specter.aberrations._envelopes import (
    b_envelope,
    cc_envelope,
    cs_envelope,
    dose_envelope,
)


def test_b_envelope_dc_is_one():
    k2 = torch.tensor([0.0])
    bfactor = torch.tensor([50.0])
    result = b_envelope(k2, bfactor)
    assert torch.allclose(result, torch.ones_like(result))


def test_b_envelope_decreases_with_frequency():
    k2 = torch.tensor([0.0, 0.01, 0.04, 0.09])
    bfactor = torch.tensor(50.0)
    result = b_envelope(k2, bfactor)
    assert torch.all(result[:-1] >= result[1:])
    assert result[-1] < result[0]


def test_cs_envelope_dc_is_one():
    k = torch.tensor([0.0])
    result = cs_envelope(
        k,
        wavelength=0.0197,
        cs=torch.tensor(2.7e7),
        defocus=torch.tensor(10000.0),
        convergence_angle=0.02,
    )
    assert torch.allclose(result, torch.ones_like(result))


def test_cs_envelope_decreases_with_frequency():
    k = torch.tensor([0.0, 0.1, 0.2, 0.3])
    result = cs_envelope(
        k,
        wavelength=0.0197,
        cs=torch.tensor(2.7e7),
        defocus=torch.tensor(10000.0),
        convergence_angle=0.02,
    )
    assert torch.all(result[:-1] >= result[1:])
    assert result[-1] < result[0]


def test_cc_envelope_dc_is_one():
    k2 = torch.tensor([0.0])
    result = cc_envelope(
        k2,
        wavelength=0.0197,
        cc=2.7e7,
        voltage=3.0e5,
        energy_spread=0.7,
        deltaV_V=0.06e-6,
        deltaI_I=0.01e-6,
    )
    assert torch.allclose(result, torch.ones_like(result))


def test_cc_envelope_decreases_with_frequency():
    k2 = torch.tensor([0.0, 0.01, 0.04, 0.09])
    result = cc_envelope(
        k2,
        wavelength=0.0197,
        cc=2.7e7,
        voltage=3.0e5,
        energy_spread=0.7,
        deltaV_V=0.06e-6,
        deltaI_I=0.01e-6,
    )
    assert torch.all(result[:-1] >= result[1:])
    assert result[-1] < result[0]


def test_dose_envelope_dc_is_one():
    k = torch.tensor([0.0])
    result = dose_envelope(k, dose=torch.tensor(50.0))
    assert torch.allclose(result, torch.ones_like(result))


def test_dose_envelope_below_threshold_is_one():
    k = torch.tensor([0.1, 0.2, 0.3])
    result = dose_envelope(k, dose=torch.tensor(1.0))
    assert torch.allclose(result, torch.ones_like(result))


def test_dose_envelope_decreases_with_frequency_above_threshold():
    k = torch.tensor([0.05, 0.1, 0.2, 0.3])
    result = dose_envelope(k, dose=torch.tensor(50.0))
    assert torch.all(result[:-1] >= result[1:])
    assert result[-1] < result[0]


def test_aberration_bfactor_envelope_matches_inline_formula():
    ab = Aberration(8, pixel_size=1.0, energy=300.0, aberration_model="holography")
    ctf_params = {
        "dfu": torch.tensor([5000.0]),
        "bfactor_envelope": torch.tensor([50.0]),
    }
    transfer = ab.transfer_function(ctf_params)
    expected_envelope = torch.exp(-50.0 * ab.k2 / 4)
    assert torch.allclose(torch.abs(transfer), expected_envelope, atol=1e-6)


def test_aberration_convergence_angle_none_is_unchanged():
    ab_off = Aberration(8, pixel_size=1.0, energy=300.0, aberration_model="holography")
    ab_on = Aberration(
        8,
        pixel_size=1.0,
        energy=300.0,
        aberration_model="holography",
        convergence_angle=None,
    )
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7e7])}
    assert torch.allclose(
        ab_off.transfer_function(ctf_params), ab_on.transfer_function(ctf_params)
    )


def test_aberration_convergence_angle_attenuates_high_frequency():
    ab = Aberration(
        64,
        pixel_size=1.0,
        energy=300.0,
        aberration_model="holography",
        convergence_angle=0.02,
    )
    ctf_params = {"dfu": torch.tensor([10000.0]), "cs": torch.tensor([2.7e7])}
    transfer = ab.transfer_function(ctf_params)
    magnitude = torch.abs(transfer).squeeze(0)
    # k increases away from the DC corner (index 0); compare a low- and a
    # high-frequency pixel along the kx axis.
    assert magnitude[0, 1] > magnitude[0, 20]
