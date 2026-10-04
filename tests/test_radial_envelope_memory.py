"""Bound envelope interpolation without changing its table or arithmetic."""

import pytest
import torch

import specter.fft as fft


def reference_multiply(spectrum, shape, dx, envelope):
    """The previous allocating interpolation, kept as a numerical oracle."""
    nz, ny, nx = shape
    kz = torch.fft.fftfreq(nz, d=dx, device=spectrum.device)
    ky = torch.fft.fftfreq(ny, d=dx, device=spectrum.device)
    kx = torch.fft.rfftfreq(nx, d=dx, device=spectrum.device)
    kyx = ky[:, None] ** 2 + kx[None, :] ** 2
    k_max = float(
        torch.sqrt(kz.abs().max() ** 2 + ky.abs().max() ** 2 + kx.abs().max() ** 2)
    )
    table = envelope(torch.linspace(0.0, k_max, 4096, device=spectrum.device))
    # A whole small test grid also checks that changing slab boundaries does
    # not change any of the frequency coordinates or endpoint indexing.
    k = torch.sqrt(kz[:, None, None] ** 2 + kyx[None])
    pos = (k / k_max * 4095).clamp_(0, 4095)
    lo = pos.floor()
    frac = pos - lo
    i0 = lo.long()
    i1 = (i0 + 1).clamp_(max=4095)
    return spectrum * torch.lerp(table[i0], table[i1], frac)


DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(23, 15, 16), (22, 16, 17)])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("cap", [1, 2**10, 2**24])
@pytest.mark.parametrize("alias", [False, True])
def test_bounded_interpolation_is_bitwise_equal(
    monkeypatch, device, shape, dtype, cap, alias
):
    torch.manual_seed(71)
    source = torch.randn(*shape[:-1], shape[-1] // 2 + 1, dtype=dtype, device=device)
    envelope = lambda k: torch.exp(-3.0 * k**2)  # noqa: E731
    expected = reference_multiply(source, shape, 0.731, envelope)
    spectrum = source.clone()
    out = spectrum if alias else torch.empty_like(spectrum)
    monkeypatch.setattr(fft, "_RADIAL_ENVELOPE_MAX_ELEMENTS", cap)
    fft._multiply_radial_envelope(spectrum, out, shape, 0.731, envelope)
    assert torch.equal(out, expected)
    if not alias:
        assert torch.equal(spectrum, source)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interpolation_preserves_envelope_gradient(monkeypatch, device, dtype):
    torch.manual_seed(83)
    shape = (23, 15, 16)
    spectrum = torch.randn(23, 15, 9, dtype=torch.complex64, device=device)
    strength = torch.tensor(3.0, dtype=dtype, device=device, requires_grad=True)
    envelope = lambda k: torch.exp(-strength * k**2)  # noqa: E731
    expected = reference_multiply(spectrum, shape, 1.5, envelope)
    expected_gradient = torch.autograd.grad(expected.abs().square().sum(), strength)[0]
    # The cached-spectrum out= path intentionally does not support autograd;
    # the in-place path can differentiate a captured envelope parameter.
    out = spectrum.clone()
    monkeypatch.setattr(fft, "_RADIAL_ENVELOPE_MAX_ELEMENTS", 2**10)
    fft._multiply_radial_envelope(out, out, shape, 1.5, envelope)
    assert torch.equal(out, expected)
    actual_gradient = torch.autograd.grad(out.abs().square().sum(), strength)[0]
    # Different slab grouping changes gradient-reduction association, not
    # the voxel arithmetic. Compare to ordinary floating-point roundoff.
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_full_filter_and_cached_spectrum_match_previous_interpolation(
    monkeypatch, device
):
    torch.manual_seed(91)
    v = torch.rand(23, 15, 16, device=device)
    envelope = lambda k: torch.exp(-3.0 * k**2)  # noqa: E731
    spectrum = fft._radial_rfft3(v)
    expected = torch.empty_like(v)
    fft._radial_irfft3_into(
        reference_multiply(spectrum, v.shape, 1.5, envelope), expected
    )
    monkeypatch.setattr(fft, "_RADIAL_ENVELOPE_MAX_ELEMENTS", 2**10)
    actual = v.clone()
    fft.apply_radial_envelope_(actual, 1.5, envelope)
    assert torch.equal(actual, expected)
    cached = fft.RadialSpectrum(v, 1.5)
    for _ in range(2):
        cached.filter_into(actual, envelope)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cap", [2**10, 2**22])
def test_smaller_fft_batches_preserve_separable_transform(
    monkeypatch, device, dtype, cap
):
    torch.manual_seed(97)
    # Non-contiguous, rectangular, odd-z input, with partial chunks on both
    # transformed axes. The existing complex64 spectrum is unchanged.
    volume = torch.rand(23, 15, 32, device=device, dtype=dtype)[..., ::2]
    monkeypatch.setattr(fft, "_RADIAL_FFT_SINGLE_BATCH_ELEMENTS", 0)
    monkeypatch.setattr(fft, "_RADIAL_FFT_MAX_ELEMENTS", 2**26)
    expected_spectrum = fft._radial_rfft3(volume)
    expected_volume = torch.empty_like(volume)
    fft._radial_irfft3_into(expected_spectrum.clone(), expected_volume)
    monkeypatch.setattr(fft, "_RADIAL_FFT_MAX_ELEMENTS", cap)
    actual_spectrum = fft._radial_rfft3(volume)
    actual_volume = torch.empty_like(volume)
    fft._radial_irfft3_into(actual_spectrum.clone(), actual_volume)
    assert torch.equal(actual_spectrum, expected_spectrum)
    assert torch.equal(actual_volume, expected_volume)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nz", [128, 136])
def test_small_probe_retains_previous_single_batch_fft(monkeypatch, device, nz):
    # These production probe shapes straddle the new chunk budget. Keeping
    # their old FFT grouping avoids last-bit cuFFT differences that can
    # change Poisson counts and the coincidence detector's later RNG draws.
    torch.manual_seed(101)
    volume = torch.rand(nz, 256, 256, device=device)
    spectrum = torch.empty(nz, 256, 129, dtype=torch.complex64, device=device)
    spectrum.copy_(torch.fft.rfft2(volume))
    spectrum.copy_(torch.fft.fft(spectrum, dim=0))
    expected = torch.fft.irfft2(torch.fft.ifft(spectrum, dim=0), s=(256, 256))
    monkeypatch.setattr(fft, "_RADIAL_FFT_MAX_ELEMENTS", 2**22)
    actual_spectrum = fft._radial_rfft3(volume)
    actual = torch.empty_like(volume)
    fft._radial_irfft3_into(actual_spectrum.clone(), actual)
    assert torch.equal(actual_spectrum, spectrum)
    assert torch.equal(actual, expected)
