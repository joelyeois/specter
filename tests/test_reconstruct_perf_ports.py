"""
Pin the 2026-09 single-scatter rewrite to the code it replaced.

`Scattering.rytov`, `firstborn` and `kinematic` now accumulate the slice
sum a z-chunk at a time with the amplitude-contrast factor, the scalar and
the Ewald-sign flip folded into a cached propagator, instead of
materialising a complex copy of the volume for each. The results equal the
old spellings, which are kept here verbatim as references, to float
rounding (the z sum is reassociated), in value and in gradient.
"""

from __future__ import annotations

import pytest
import torch

from specter.fft import fft2, ifft2
from specter.potential import apply_amplitude_contrast
from specter.scattering import Scattering


def _reference(model: Scattering, name: str, V: torch.Tensor) -> torch.Tensor:
    """The pre-rewrite implementations, applied to a real V as `forward` did."""
    F = model.F_real + 1j * model.F_imag
    V = apply_amplitude_contrast(V, alpha=model.alpha)
    if model.ews_curvature_sign == "negative":
        V = torch.flip(V, dims=(1,))
    c = model.sigma * model.pixel_size
    if name == "rytov":
        scattered_k = (fft2(1j * c * V) * F[None]).sum(dim=1)
        return torch.exp(ifft2(scattered_k))
    if name == "firstborn":
        return 1 + 1j * ifft2(torch.sum(c * fft2(V) * F[None], 1))
    t = torch.exp(1j * c * V) - 1
    return 1 + ifft2(torch.sum(fft2(t) * F[None], 1))


@pytest.mark.parametrize("name", ["rytov", "firstborn", "kinematic"])
@pytest.mark.parametrize("sign", ["negative", "positive"])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_single_scatter_models_match_their_old_spelling(name, sign, alpha):
    from specter import scattering as sc

    torch.manual_seed(0)
    n, nz, B = 16, 11, 2
    model = Scattering(
        n,
        1.0,
        300.0,
        scattering_model=name,
        nz=nz,
        alpha=alpha,
        ews_curvature_sign=sign,
        progressbars=False,
    ).double()
    V = (torch.rand(B, nz, n, n, dtype=torch.float64) * 5).requires_grad_(True)

    ref = _reference(model, name, V)
    (g_ref,) = torch.autograd.grad(ref.abs().sum(), V)

    # Force several chunks so the accumulation path is exercised.
    old = sc._SLICE_SUM_CHUNK_ELEMENTS
    sc._SLICE_SUM_CHUNK_ELEMENTS = B * n * n * 3
    try:
        new = model(V)
        (g_new,) = torch.autograd.grad(new.abs().sum(), V)
    finally:
        sc._SLICE_SUM_CHUNK_ELEMENTS = old

    assert torch.allclose(new, ref, rtol=1e-10, atol=1e-12)
    assert torch.allclose(g_new, g_ref, rtol=1e-8, atol=1e-12)


def test_single_scatter_models_accept_a_complex_volume():
    """A caller that has already applied the absorption factor gets the same
    answer as one that hands over the real volume."""
    torch.manual_seed(1)
    n, nz = 12, 7
    model = Scattering(
        n, 1.0, 300.0, scattering_model="rytov", nz=nz, alpha=0.1, progressbars=False
    ).double()
    V = torch.rand(1, nz, n, n, dtype=torch.float64)
    from_real = model.rytov(V)
    from_complex = model.rytov(apply_amplitude_contrast(V, alpha=0.1))
    assert torch.allclose(from_real, from_complex, rtol=1e-10, atol=1e-12)


def test_propagator_is_cached_and_flipped_once():
    model = Scattering(
        8,
        1.0,
        300.0,
        scattering_model="rytov",
        nz=5,
        ews_curvature_sign="negative",
        progressbars=False,
    )
    V = torch.rand(1, 5, 8, 8)
    F1 = model._propagator(V)
    F2 = model._propagator(V)
    assert F1 is F2
    expected = (model.F_real + 1j * model.F_imag).flip(0)
    assert torch.equal(F1, expected)


def test_kmask_half_spectrum_matches_the_shifted_round_trip():
    """The per-step band limit equals the old centred complex round trip."""
    from specter.arrays import ball3d
    from specter.fft import fft3, ifft3
    from specter.ghostbuster._helpers import _apply_kmask_inplace, _kmask_half_spectrum

    torch.manual_seed(0)
    n = 24
    kmask = ball3d(n, n // 3)
    V = torch.randn(n, n, n)
    old = torch.real(ifft3(fft3(V, shift=True) * kmask, shift=True))
    new = V.clone()
    _apply_kmask_inplace(new, _kmask_half_spectrum(kmask))
    assert torch.allclose(new, old, rtol=1e-5, atol=1e-6)
    # And it is a band limit: the masked-out shell is empty afterwards.
    spectrum = fft3(new, shift=True)
    assert torch.allclose(
        spectrum[kmask == 0],
        torch.zeros(int((kmask == 0).sum()), dtype=spectrum.dtype),
        atol=1e-4,
    )


def test_rotate_volume_under_grad_matches_forward_only_and_batched_gradient():
    """
    With the volume requiring grad, `rotate_volume` samples one image at a
    time with a recomputed grid; the values equal the batched forward-only
    call exactly, and the gradient equals the batched kernel's to float
    accumulation order.
    """
    import torch.nn.functional as F

    from specter import rotations
    from specter.rotations._volume import _relion_rotation_grid

    torch.manual_seed(0)
    n = 20
    V = torch.randn(n, n, n)
    R = rotations.random_rotation_matrix(3)
    theta = rotations.build_affine_matrix(R)
    with torch.no_grad():
        plain = rotations.rotate_volume(V, theta)
    Vg = V.clone().requires_grad_(True)
    out = rotations.rotate_volume(Vg, theta)
    assert torch.equal(out, plain)
    (g_new,) = torch.autograd.grad((out**2).sum(), Vg)

    Vb = V.clone().requires_grad_(True)
    grid = _relion_rotation_grid(theta, n, n, n, False)
    ref = F.grid_sample(
        Vb[None, None].expand(3, 1, n, n, n),
        grid,
        align_corners=False,
        padding_mode="border",
    )[:, 0]
    (g_ref,) = torch.autograd.grad((ref**2).sum(), Vb)
    assert torch.allclose(g_new, g_ref, rtol=1e-5, atol=1e-6)


def test_volume_rotator_under_grad_matches_forward_only():
    from specter import rotations
    from specter.rotations import VolumeRotator

    torch.manual_seed(0)
    n = 20
    rot = VolumeRotator(n, n, n, origin="relion", mode="real")
    V = torch.randn(n, n, n)
    theta = rotations.build_affine_matrix(rotations.random_rotation_matrix(3))
    with torch.no_grad():
        plain = rot(V, theta)
    Vg = V.clone().requires_grad_(True)
    out = rot(Vg, theta)
    assert torch.equal(out, plain)
    (g,) = torch.autograd.grad((out**2).sum(), Vg)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
