"""
Partial temporal coherence as an incoherent sum over defocus (FocalSpread).

The analytic test is the textbook discriminator between the transmission
cross-coefficient and the envelope approximation: in a three-beam image the
second-harmonic fringe comes only from the +k0/-k0 pair, whose phase
difference under defocus is pi lambda df (k0^2 - k0^2) = 0, so NO focal
spread can damp it. The exact sum must leave it at its coherent amplitude;
the envelope on the wave damps it by E(k0)^2.
"""

from __future__ import annotations

import math

import pytest
import torch

from specter import rotations
from specter.aberrations import FocalSpread, cc_envelope, focal_spread_sigma
from specter.constants import energy_to_wavelength
from specter.imaging import CTF, ExitWave, Exposure, Poses, Specimen

WAVELENGTH = energy_to_wavelength(300.0)


def _three_beam(n: int, dx: float, h: int, a0: float = 1.0, a1: float = 0.2):
    """A wave holding only the beams 0, +k0 and -k0 (k0 = h / (n dx)) along x."""
    spectrum = torch.zeros(n, n, dtype=torch.complex128)
    spectrum[0, 0] = a0 * n * n
    spectrum[0, h] = a1 * n * n
    spectrum[0, -h] = a1 * n * n
    return torch.fft.ifft2(spectrum)


def _fourier_amplitude(image: torch.Tensor, h: int) -> float:
    n = image.shape[-1]
    return float(torch.fft.fft2(image)[0, h].abs()) / (n * n)


def test_second_harmonic_is_undamped_by_any_focal_spread():
    # k0 = 0.23 1/A keeps the 2 k0 fringe inside the grid; a 300 A focal
    # spread puts E(k0) near 0.6, so damped and undamped are far apart.
    n, dx, h = 64, 1.0, 15
    k0 = h / (n * dx)
    sigma_f = 300.0
    wave = _three_beam(n, dx, h)
    # Defocus the three beams, then image them.
    defocus = 8000.0
    kx = torch.fft.fftfreq(n, dx, dtype=torch.float64)
    k2 = kx[None, :] ** 2 + kx[:, None] ** 2
    ctf = torch.exp(1j * math.pi * WAVELENGTH * defocus * k2)
    focused = torch.fft.ifft2(torch.fft.fft2(wave) * ctf)

    coherent = focused.abs().square()
    exact = FocalSpread(sigma_f, WAVELENGTH).intensity(focused, dx)
    E = math.exp(-0.5 * (math.pi * WAVELENGTH * sigma_f * k0**2) ** 2)
    assert 0.3 < E < 0.9
    envelope_wave = torch.fft.ifft2(
        torch.fft.fft2(focused)
        * torch.exp(-0.5 * (math.pi * WAVELENGTH * sigma_f * k2) ** 2)
    )
    envelope = envelope_wave.abs().square()

    # 2 k0: the +k0/-k0 pair only. Undamped exactly; envelope damps it by E^2.
    assert _fourier_amplitude(exact, 2 * h) == pytest.approx(
        _fourier_amplitude(coherent, 2 * h), rel=1e-9
    )
    assert _fourier_amplitude(envelope, 2 * h) == pytest.approx(
        E**2 * _fourier_amplitude(coherent, 2 * h), rel=1e-9
    )
    # k0: the linear term, where envelope and exact sum are the same thing.
    assert _fourier_amplitude(exact, h) == pytest.approx(
        E * _fourier_amplitude(coherent, h), rel=1e-6
    )
    assert _fourier_amplitude(envelope, h) == pytest.approx(
        E * _fourier_amplitude(coherent, h), rel=1e-6
    )
    # Electrons are conserved by the sum (every node is a pure phase), and
    # lost by the envelope.
    assert float(exact.mean()) == pytest.approx(float(coherent.mean()), rel=1e-12)
    assert float(envelope.mean()) < float(coherent.mean())


@pytest.mark.parametrize(
    "voltage, dx, oversample", [(300.0, 0.5, 2), (120.0, 0.5, 2), (300.0, 1.0, 1)]
)
def test_quadrature_is_converged_on_its_own_grid(voltage, dx, oversample):
    """The automatically chosen node count matches a 150-node reference to
    the requested accuracy on a broadband random wave."""
    wavelength = energy_to_wavelength(voltage)
    sigma_f = focal_spread_sigma(2.7e7, voltage * 1e3, 0.7, 0.06e-6, 0.01e-6)
    torch.manual_seed(0)
    n = 96
    wave = 1 + 0.3 * torch.randn(1, n, n, dtype=torch.complex128)
    auto = FocalSpread(sigma_f, wavelength)
    got = auto.intensity(wave, dx, oversample)

    import numpy as np

    x, w = np.polynomial.hermite.hermgauss(150)
    ref = torch.zeros_like(got)
    kx = torch.fft.fftfreq(n, dx, dtype=torch.float64)
    k2 = kx[None, :] ** 2 + kx[:, None] ** 2
    spectrum = torch.fft.fft2(wave)
    for xi, wi in zip(x, w):
        d = math.sqrt(2) * sigma_f * xi
        phase = torch.exp(1j * math.pi * wavelength * d * k2)
        ref += (
            wi / math.sqrt(math.pi) * torch.fft.ifft2(spectrum * phase).abs().square()
        )

    # Compare where the detector will look: the pixel-grid band.
    def band(img):
        s = torch.fft.fft2(img)
        m = n // oversample
        keep = torch.zeros(n, dtype=torch.bool)
        keep[: m // 2] = True
        keep[-(m // 2) :] = True
        return s[..., keep, :][..., keep]

    err = (band(got) - band(ref)).abs().max() / band(ref).abs().max()
    assert float(err) < 1e-5
    assert len(auto.nodes(dx, oversample)[0]) < 128


def test_zero_focal_spread_is_the_coherent_image():
    torch.manual_seed(1)
    wave = torch.randn(2, 16, 16, dtype=torch.complex64)
    assert torch.equal(
        FocalSpread(0.0, WAVELENGTH).intensity(wave, 1.0), wave.abs().square()
    )


def _aberrated(scattering_model: str = "multislice") -> ExitWave:
    n = 24
    torch.manual_seed(0)
    z, y, x = torch.meshgrid(*[torch.arange(n) - n / 2] * 3, indexing="ij")
    V = 30.0 * torch.exp(-(x**2 + y**2 + z**2) / 10.0)
    ctf = CTF(
        {
            "dfu": torch.tensor([8000.0]),
            "dfv": torch.tensor([8000.0]),
            "cs": torch.tensor([2.7e7]),
        }
    )
    poses = Poses(
        rotations.random_quaternion(1).reshape(1, 4), torch.zeros(1, 2), torch.arange(1)
    )
    with torch.no_grad():
        waves = Specimen(V, 1.0).scatter(
            poses, voltage=300.0, scattering_model=scattering_model, oversample=2
        )
        return waves.apply_ctf(ctf)


def _image(waves: ExitWave, cc: float | None, mode: str) -> torch.Tensor:
    with torch.no_grad():
        image = waves.intensity(cc=cc, energy_spread=3.0, temporal_coherence=mode)
        return image.detect(Exposure(40.0), noise="none").array


def test_imager_integrates_instead_of_applying_the_envelope():
    """The default image is the focal-spread sum over the aberrated wave, which
    carries no Cc envelope of its own (`apply_ctf` takes none)."""
    waves = _aberrated()
    a = _image(waves, 2.7, "integrated")
    b = _image(waves, 2.7, "envelope")
    c = _image(waves, None, "integrated")
    focal = FocalSpread.from_instrument(2.7e7, 3e5, 3.0, 0.06e-6, 0.01e-6, WAVELENGTH)
    with torch.no_grad():
        summed = focal.intensity(waves.array, 0.5, 2)
    assert torch.equal(waves.intensity(cc=2.7, energy_spread=3.0).array, summed)
    assert not torch.equal(a, b)
    # The sum conserves the electrons the coherent image has; the envelope
    # removes some.
    assert float(a.mean()) == pytest.approx(float(c.mean()), rel=1e-5)
    assert float(b.mean()) < float(c.mean())


def test_linear_model_keeps_the_envelope():
    """scattering_model='ctf' images linearly, where the envelope is exact, so
    it takes the envelope whichever route is asked for."""
    waves = _aberrated("ctf")
    integrated = _image(waves, 2.7, "integrated")
    assert torch.equal(integrated, _image(waves, 2.7, "envelope"))
    assert not torch.equal(integrated, _image(waves, None, "integrated"))


def test_envelope_and_focal_spread_share_one_sigma():
    sigma = focal_spread_sigma(2.7e7, 3e5, 0.7, 0.06e-6, 0.01e-6)
    k2 = torch.tensor([0.1, 0.25])
    expected = torch.exp(-0.5 * (math.pi * WAVELENGTH * sigma * k2) ** 2)
    got = cc_envelope(k2, WAVELENGTH, 2.7e7, 3e5, 0.7, 0.06e-6, 0.01e-6)
    assert torch.allclose(got, expected)
    assert FocalSpread.from_instrument(
        2.7e7, 3e5, 0.7, 0.06e-6, 0.01e-6, WAVELENGTH
    ).sigma_f == pytest.approx(sigma)
