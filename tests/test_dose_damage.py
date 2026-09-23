"""
The dose envelope damages the specimen's potential, not the solvent.

Raw EER movies show the 3.7 A water ring does not fade under exposure, so
the particle generators apply the envelope to the protein potential before
solvation (`potential.apply_dose_damage`) rather than to the transfer
function, which would filter the ice as hard as the protein.
"""

from __future__ import annotations

import torch

import specter
from specter.aberrations import dose_envelope
from specter.imagegenerator import ImageGenerator
from specter.potential import (
    apply_dose_damage,
    frame_damage_envelope,
    potential_occupancy,
)
from specter.settings import Camera, Envelopes, Ice, Propagation


def _ctf_params() -> dict[str, torch.Tensor]:
    return {
        "dfu": torch.tensor([8000.0]),
        "dfv": torch.tensor([8000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.0]),
    }


def test_apply_dose_damage_preserves_mass_and_attenuates_high_k():
    torch.manual_seed(0)
    V = torch.rand(2, 16, 16, 16)
    total = V.sum(dim=(-3, -2, -1))
    hi_before = torch.fft.fftn(V, dim=(-3, -2, -1)).abs()[:, 8, 8, 8]
    out = apply_dose_damage(V.clone(), 1.0, torch.tensor([0.0, 40.0]))
    assert torch.allclose(out.sum(dim=(-3, -2, -1)), total, rtol=1e-5)
    # Zero dose is the identity; 40 e/A^2 attenuates Nyquist hard.
    assert torch.allclose(out[0], V[0], atol=1e-6)
    hi_after = torch.fft.fftn(out, dim=(-3, -2, -1)).abs()[:, 8, 8, 8]
    assert hi_after[1] < 0.5 * hi_before[1]


def _generator(volume: torch.Tensor, dose_envelope: bool, ice: Ice) -> ImageGenerator:
    return ImageGenerator(
        scattering_potential=volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=_ctf_params(),
        voltage=300.0,
        dose_per_angstrom=40.0,
        verbose=False,
        progressbars=False,
        propagation=Propagation(scattering_model="ctf", alpha=0.0),
        envelopes=Envelopes(dose_envelope=dose_envelope),
        camera=Camera(noise_model=None),
        ice=ice,
    )


def test_potential_damage_equals_transfer_function_envelope_for_the_specimen(
    small_volume,
):
    """
    Without solvent the two placements are the same filter, by the
    projection-slice theorem: the 3D radial envelope at kz = 0 is the 2D
    envelope on the projection.
    """
    on = _generator(small_volume, True, Ice(model=None))
    off = _generator(small_volume, False, Ice(model=None))
    on(torch.tensor([0]))
    off(torch.tensor([0]))
    proj_on, proj_off = on.exitwaves, off.exitwaves
    n = proj_off.shape[-1]
    k = torch.fft.fftfreq(n, d=2.0)
    kk = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    env = dose_envelope(kk, torch.tensor(40.0), weighted=True, voltage=300.0)
    expected = torch.fft.ifft2(torch.fft.fft2(proj_off) * env).real
    assert torch.allclose(proj_on, expected, atol=1e-4 * proj_off.abs().max())
    assert not torch.allclose(proj_on, proj_off, atol=1e-3 * proj_off.abs().max())


def test_frame_sum_with_uniform_weights_is_the_plain_closed_form():
    """
    A plain frame sum is the midpoint rule for the unweighted integral, so
    with no exposure filter the explicit sum must reproduce it.
    """
    k = torch.linspace(0.0, 0.4, 41)
    got = frame_damage_envelope(k, 40.0, 40, voltage=300.0)
    want = dose_envelope(k, torch.tensor(40.0), weighted=False, voltage=300.0)
    assert torch.allclose(got, want, atol=2e-4)
    assert torch.isclose(got[0], torch.tensor(1.0))
    assert (got.diff() <= 1e-6).all(), "envelope must fall with frequency"


def test_front_loaded_weights_preserve_more_signal_than_a_plain_sum():
    """
    An exposure filter that up-weights early frames keeps more high-frequency
    signal, because those frames saw a less damaged specimen.
    """
    n_frames, n_bins = 20, 64
    k = torch.tensor([1 / 4.0])
    front = torch.zeros(n_frames, n_bins)
    front[:5] = 1.0
    back = torch.zeros(n_frames, n_bins)
    back[-5:] = 1.0
    kw = dict(weights_max_frequency=0.5, voltage=300.0)
    e_front = frame_damage_envelope(k, 40.0, n_frames, weights=front, **kw)
    e_plain = frame_damage_envelope(k, 40.0, n_frames, voltage=300.0)
    e_back = frame_damage_envelope(k, 40.0, n_frames, weights=back, **kw)
    assert e_back < e_plain < e_front


def test_occupancy_is_not_invariant_under_damage():
    """
    Why the order matters. The envelope conserves the integral of the
    potential but spreads it, so a molecule read after damage displaces
    water differently from the same molecule read before. Occupancy is a
    question about physical volume, so it must be read first.
    """
    V = torch.zeros(1, 48, 48, 48)
    V[:, 21:27, 21:27, 21:27] = 9.0  # a small feature, near the 7 V reference
    occ_before = potential_occupancy(V, 1.0)
    occ_after = potential_occupancy(apply_dose_damage(V.clone(), 1.0, 50.0), 1.0)
    assert occ_after.max() < occ_before.max()
    assert not torch.allclose(occ_before, occ_after, atol=1e-3)


def test_solvate_out_returns_the_weighted_ice_and_leaves_the_volume_alone(
    small_volume,
):
    """
    The `out` mode is what lets occupancy be read from the undamaged
    specimen and one solvent realisation serve every damage state: it must
    return the weighted ice and not touch the volume it read.
    """
    specter.seed(5)
    gen = _generator(small_volume, False, Ice(model="random"))
    V = gen.pad_volume(gen.V.clone()) if hasattr(gen, "pad_volume") else gen.V.clone()
    V = V if V.ndim == 4 else V.unsqueeze(0)
    before = V.clone()
    field = gen.solvate(V, out=torch.zeros_like(V))
    assert torch.equal(V, before), "solvate(out=...) must not write into V"
    assert field.shape == V.shape
    assert float(field.max()) > 0.0
    # Ice is excluded where the specimen already sits.
    solid = before[0] > before[0].max() * 0.5
    assert field[0][solid].mean() < field[0][~solid].mean()


def test_solvent_is_not_damaged():
    """An empty specimen in ice images identically with the envelope on or off."""
    empty = torch.zeros(32, 32, 32)
    waves = []
    for flag in (True, False):
        specter.seed(3)
        gen = _generator(empty, flag, Ice(model="random"))
        gen(torch.tensor([0]))
        waves.append(gen.exitwaves.clone())
    assert waves[0].abs().max() > 0
    assert torch.allclose(waves[0], waves[1])
