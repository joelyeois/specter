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
from specter.settings import Camera, Propagation

# Representative 300 kV imaging parameters, in specter's Angstrom-based units.
WAVELENGTH = 0.0197
CS = 2.7e7  # 2.7 mm spherical aberration, in Angstrom


def test_cs_matches_formula():
    k = torch.tensor([0.1])
    result = cs(k, WAVELENGTH, torch.tensor(CS))
    expected = math.pi / 2 * WAVELENGTH**3 * 0.1**4 * CS
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_defocus_matches_formula_no_astigmatism():
    k2 = torch.tensor([0.01])
    radian = torch.tensor([0.0])
    dfu = dfv = torch.tensor(5000.0)
    result = defocus(k2, radian, WAVELENGTH, dfu, dfv, torch.tensor(0.0))
    expected = -math.pi * WAVELENGTH * 0.01 * 5000.0
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_defocus_astigmatism_axes_and_angle_in_degrees():
    """Which axis carries dfu, and that ``dfang`` arrives in degrees.

    The cos(2*(radian + dfang)) term puts ``dfv`` along radian=0 and ``dfu``
    along radian=pi/2 when dfang=0; 90 degrees exchanges them, and 45 degrees
    puts the mean on both. The unit matters and is checked here because
    nothing else pins it: ``dfang`` is the one term of the legacy ctf_params
    dict carried in degrees rather than radians (io/_cryosparc.py converts on
    the way in, this function converts back), so an implementation treating
    it as radians would put 4401.5 A on the radian=0 axis of the 90-degree
    case instead of 4000.
    """
    k2 = torch.tensor([0.01, 0.01])
    radian = torch.tensor([0.0, math.pi / 2])
    dfu, dfv = torch.tensor(4000.0), torch.tensor(6000.0)

    def chi(defocus_a: float, defocus_b: float) -> torch.Tensor:
        return torch.tensor(
            [
                -math.pi * WAVELENGTH * 0.01 * defocus_a,
                -math.pi * WAVELENGTH * 0.01 * defocus_b,
            ]
        )

    for dfang, expected in (
        (0.0, chi(6000.0, 4000.0)),
        (90.0, chi(4000.0, 6000.0)),
        (45.0, chi(5000.0, 5000.0)),
    ):
        result = defocus(k2, radian, WAVELENGTH, dfu, dfv, torch.tensor(dfang))
        assert torch.allclose(result, expected, atol=1e-6)


def test_beamtilt_matches_formula():
    """Note the cross-pairing, which the argument order invites getting
    backwards: the grids are positional in the order (KY, KX), and it is
    ``tilty`` that multiplies KY while ``tiltx`` multiplies KX. All four
    values differ here, so swapping either pair fails.
    """
    KY, KX = torch.tensor(0.05), torch.tensor(0.02)
    k2 = KY**2 + KX**2
    tiltx, tilty = torch.tensor(0.001), torch.tensor(0.002)
    result = beamtilt(k2, KY, KX, WAVELENGTH, torch.tensor(CS), tiltx, tilty)
    tilts = math.sin(0.002) * 0.05 + math.sin(0.001) * 0.02
    expected = -2 * math.pi * WAVELENGTH**2 * CS * float(k2) * tilts
    assert torch.allclose(result, torch.tensor(expected), atol=1e-6)


def test_trefoil_matches_formula():
    k = torch.tensor([0.1])
    radian = torch.tensor([0.3])
    result = trefoil(k, radian, torch.tensor(10.0), torch.tensor(5.0))
    expected = 10.0 * 0.1**3 * math.sin(3 * 0.3) + 5.0 * 0.1**3 * math.cos(3 * 0.3)
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_tetrafoil_matches_formula():
    k = torch.tensor([0.1])
    radian = torch.tensor([0.3])
    t1, t2, t3, t4 = 10.0, 5.0, 3.0, 2.0
    result = tetrafoil(
        k,
        radian,
        torch.tensor(t1),
        torch.tensor(t2),
        torch.tensor(t3),
        torch.tensor(t4),
    )
    expected = (
        t1 * 0.1**4 * math.cos(2 * 0.3)
        + t2 * 0.1**4 * math.sin(2 * 0.3)
        + t3 * 0.1**4 * math.cos(4 * 0.3)
        + t4 * 0.1**4 * math.sin(4 * 0.3)
    )
    assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)


def test_phaseshift_linear_model_returns_negative_input_unchanged():
    """No amplitude-contrast offset unless one is asked for -- this is the
    ``specimen_absorption=True`` / non-"ctf"-scattering case, where the
    contrast is already applied upstream via
    ``potential.apply_amplitude_contrast`` and must not be counted twice."""
    phaseshift_val = torch.tensor([0.5])
    k = torch.zeros((1, 4, 4))
    result = phaseshift(phaseshift_val, k, n_pixels=4, aberration_model="linear")
    assert torch.allclose(result, -phaseshift_val)
    explicit_none = phaseshift(
        phaseshift_val, k, n_pixels=4, aberration_model="linear", alpha=None
    )
    assert torch.allclose(explicit_none, -phaseshift_val)


def test_phaseshift_linear_model_with_alpha_adds_amp_contrast_offset():
    """CryoSPARC convention: chi_c0 = phase_shift - acos(amp_contrast), so
    chi_phaseshift = -phase_shift + acos(amp_contrast) -- matches the
    (2*pi/3)*wavelength^2 trefoil derivation's -1 global sign convention."""
    phaseshift_val = torch.tensor([0.5])
    alpha = torch.tensor(0.1)
    k = torch.zeros((1, 4, 4))
    result = phaseshift(
        phaseshift_val, k, n_pixels=4, aberration_model="linear", alpha=alpha
    )
    expected = -phaseshift_val + math.acos(0.1)
    assert torch.allclose(result, expected)


def test_phaseshift_linear_model_alpha_zero_still_adds_quarter_turn():
    """acos(0) = pi/2 exactly -- amplitude contrast defaulting to zero is
    still a real, nonzero phase-quadrature offset in CryoSPARC's
    convention, not a no-op."""
    phaseshift_val = torch.tensor([0.0])
    alpha = torch.tensor(0.0)
    k = torch.zeros((1, 4, 4))
    result = phaseshift(
        phaseshift_val, k, n_pixels=4, aberration_model="linear", alpha=alpha
    )
    assert torch.allclose(result, torch.tensor([math.pi / 2]))


def test_phaseshift_nonlinear_model_ignores_alpha():
    """alpha is only meaningful for the "linear" model -- the nonlinear
    model must ignore it even if passed, since amplitude contrast there is
    represented in the exit wave's complex/absorptive component instead."""
    phaseshift_val = torch.tensor([0.5])
    k = torch.zeros((1, 4, 4))
    with_alpha = phaseshift(
        phaseshift_val,
        k,
        n_pixels=4,
        aberration_model="nonlinear",
        alpha=torch.tensor(0.1),
    )
    without_alpha = phaseshift(
        phaseshift_val, k, n_pixels=4, aberration_model="nonlinear"
    )
    assert torch.allclose(with_alpha, without_alpha)


def test_phaseshift_nonlinear_broadcasts_to_grid_and_zeroes_dc():
    """A scalar shift becomes a k-shaped grid, with DC (index [0, 0] under
    torch.fft.fftfreq's unshifted ordering) zeroed for Fourier optics
    validity and every other pixel holding -phaseshift."""
    n_pixels = 8
    kx = torch.fft.fftfreq(n_pixels, 1.0)
    kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
    k = torch.sqrt(kxx**2 + kyy**2).unsqueeze(0)
    phaseshift_val = torch.tensor([0.5])
    result = phaseshift(phaseshift_val, k, n_pixels, aberration_model="nonlinear")
    assert result.shape == k.shape
    assert result[0, 0, 0] == 0
    # Nyquist pixel (index n_pixels // 2) is unaffected, and so is every
    # other non-DC pixel: exactly one of the 64 is zeroed.
    assert result[0, n_pixels // 2, n_pixels // 2] == -0.5
    assert int((result == 0).sum()) == 1
    assert torch.allclose(result[result != 0], torch.full((n_pixels**2 - 1,), -0.5))


def test_defocus_increases_with_z():
    """
    Pins the sign of the depth-to-defocus relation, which nothing in an image
    reveals but a per-particle defocus export depends on.

    The same scatterer at two depths differs by exactly the propagation
    distance between them, so a blob 96 A below the midplane must match the
    same blob at the midplane imaged at ``dfu - 96``. That fixes
    ``df_i = df_ref + z_i``: defocus increases with z, and the entry face is
    the high-defocus end -- which is why ``defocus_midplane_shift`` is
    subtracted rather than added.

    Heavier than the rest of this file, and kept here regardless: it pins
    the sign of ``defocus_midplane_shift``, which is defined alongside these
    functions and whose docstring cites this test by name.
    """
    from specter.imagegenerator import MicrographGenerator

    nxy, nz, px = 64, 128, 2.0
    dfu, centre = 10000.0, nz // 2
    k_low = centre - 48
    delta = (centre - k_low) * px  # 96 A below the midplane

    def image(k: int, defocus: float) -> torch.Tensor:
        V = torch.zeros(1, nz, nxy, nxy)
        c = nxy // 2
        V[0, k - 2 : k + 3, c - 4 : c + 5, c - 4 : c + 5] = 50.0
        model = MicrographGenerator(
            None,
            nxy,
            px,
            {"cs": torch.tensor([2.7e7]), "dfu": torch.tensor([defocus])},
            300.0,
            torch.tensor([100.0]),
            volume=V,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="multislice", alpha=0.1),
            camera=Camera(noise_model=None),
        )
        with torch.no_grad():
            return model(torch.tensor([0]))[0].detach()

    def rms(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(((a - b) ** 2).mean().sqrt())

    ref = image(k_low, dfu)
    matched = rms(ref, image(centre, dfu - delta))
    wrong_sign = rms(ref, image(centre, dfu + delta))
    unshifted = rms(ref, image(centre, dfu))

    assert matched < 0.05 * unshifted
    assert matched < 0.05 * wrong_sign
