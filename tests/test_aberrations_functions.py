from __future__ import annotations

import math

import torch

from specter.aberrations._functions import (
    beamtilt,
    cs,
    defocus,
    phaseshift,
    tetrafoil,
    trefoil,
)

# Representative 300 kV imaging parameters, in specter's Angstrom-based units.
WAVELENGTH = 0.0197
CS = 2.7e7  # 2.7 mm spherical aberration, in Angstrom


def test_cs_zero_at_zero_frequency():
    k = torch.tensor([0.0])
    result = cs(k, WAVELENGTH, torch.tensor(CS))
    assert torch.allclose(result, torch.zeros_like(result))


def test_cs_matches_formula():
    k = torch.tensor([0.1])
    result = cs(k, WAVELENGTH, torch.tensor(CS))
    expected = math.pi / 2 * WAVELENGTH**3 * 0.1**4 * CS
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_cs_increases_with_frequency():
    k = torch.tensor([0.0, 0.05, 0.1, 0.2])
    result = cs(k, WAVELENGTH, torch.tensor(CS))
    assert torch.all(result[1:] > result[:-1])


def test_defocus_zero_at_zero_frequency():
    k2 = torch.tensor([0.0])
    radian = torch.tensor([0.0])
    result = defocus(
        k2,
        radian,
        WAVELENGTH,
        torch.tensor(5000.0),
        torch.tensor(5000.0),
        torch.tensor(0.0),
    )
    assert torch.allclose(result, torch.zeros_like(result))


def test_defocus_matches_formula_no_astigmatism():
    k2 = torch.tensor([0.01])
    radian = torch.tensor([0.0])
    dfu = dfv = torch.tensor(5000.0)
    result = defocus(k2, radian, WAVELENGTH, dfu, dfv, torch.tensor(0.0))
    expected = -math.pi * WAVELENGTH * 0.01 * 5000.0
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_defocus_astigmatism_swaps_axes_with_radian():
    """At dfang=0, the cos(2*(radian+dfang)) term makes ``dfv`` apply along
    radian=0 and ``dfu`` apply along radian=pi/2."""
    k2 = torch.tensor([0.01, 0.01])
    radian = torch.tensor([0.0, math.pi / 2])
    dfu, dfv = torch.tensor(4000.0), torch.tensor(6000.0)
    result = defocus(k2, radian, WAVELENGTH, dfu, dfv, torch.tensor(0.0))
    expected = torch.tensor(
        [
            -math.pi * WAVELENGTH * 0.01 * 6000.0,
            -math.pi * WAVELENGTH * 0.01 * 4000.0,
        ]
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_beamtilt_zero_when_no_tilt():
    kxx, kyy = torch.tensor(0.05), torch.tensor(0.02)
    k2 = kxx**2 + kyy**2
    result = beamtilt(
        k2, kxx, kyy, WAVELENGTH, torch.tensor(CS), torch.tensor(0.0), torch.tensor(0.0)
    )
    assert torch.allclose(result, torch.zeros_like(result))


def test_beamtilt_matches_formula():
    kxx, kyy = torch.tensor(0.05), torch.tensor(0.02)
    k2 = kxx**2 + kyy**2
    tiltx, tilty = torch.tensor(0.001), torch.tensor(0.002)
    result = beamtilt(k2, kxx, kyy, WAVELENGTH, torch.tensor(CS), tiltx, tilty)
    tilts = math.sin(0.002) * 0.05 + math.sin(0.001) * 0.02
    expected = -2 * math.pi * WAVELENGTH**2 * CS * float(k2) * tilts
    assert torch.allclose(result, torch.tensor(expected), atol=1e-6)


def test_trefoil_zero_at_zero_frequency():
    k = torch.tensor([0.0])
    radian = torch.tensor([0.3])
    result = trefoil(k, radian, torch.tensor(10.0), torch.tensor(5.0))
    assert torch.allclose(result, torch.zeros_like(result))


def test_trefoil_matches_formula():
    k = torch.tensor([0.1])
    radian = torch.tensor([0.3])
    result = trefoil(k, radian, torch.tensor(10.0), torch.tensor(5.0))
    expected = 10.0 * 0.1**3 * math.sin(3 * 0.3) + 5.0 * 0.1**3 * math.cos(3 * 0.3)
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_trefoil_has_three_fold_symmetry():
    k = torch.tensor([0.1, 0.1])
    radian = torch.tensor([0.3, 0.3 + 2 * math.pi / 3])
    result = trefoil(k, radian, torch.tensor(10.0), torch.tensor(5.0))
    assert torch.allclose(result[0], result[1], atol=1e-5)


def test_phaseshift_ctf_model_returns_negative_input_unchanged():
    phaseshift_val = torch.tensor([0.5])
    k = torch.zeros((1, 4, 4))
    result = phaseshift(phaseshift_val, k, n_pixels=4, aberration_model="ctf")
    assert torch.allclose(result, -phaseshift_val)


def test_phaseshift_holography_model_broadcasts_to_grid():
    phaseshift_val = torch.tensor([0.5])
    k = torch.zeros((1, 4, 4))
    result = phaseshift(phaseshift_val, k, n_pixels=4, aberration_model="holography")
    assert result.shape == k.shape
    nonzero = result[result != 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, -0.5))


def test_phaseshift_holography_zeroes_dc():
    """DC (k=0, index [0, 0] under torch.fft.fftfreq's unshifted ordering)
    must be zeroed for Fourier optics validity; other pixels keep -phaseshift."""
    n_pixels = 8
    kx = torch.fft.fftfreq(n_pixels, 1.0)
    kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
    k = torch.sqrt(kxx**2 + kyy**2).unsqueeze(0)
    phaseshift_val = torch.tensor([0.5])
    result = phaseshift(phaseshift_val, k, n_pixels, aberration_model="holography")
    assert result[0, 0, 0] == 0
    # Nyquist pixel (index n_pixels // 2) is unaffected.
    assert result[0, n_pixels // 2, n_pixels // 2] == -0.5


# ---------------------------------------------------------------------------
# Documented limitations
# ---------------------------------------------------------------------------


def test_tetrafoil_not_implemented():
    """tetrafoil() is a stub that returns None; see also
    test_ghostbuster.test_tetrafoil_not_implemented for the TypeError this
    causes downstream in Aberration.transfer_function."""
    result = tetrafoil(
        torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    )
    assert result is None
