"""
Where radiation damage acts, and what happens to the solvent under exposure.

Raw EER movies show the 3.7 A water ring does not fade under exposure, so a
particle generator can apply the dose envelope to the specimen's potential
before solvation (``Envelopes(dose_envelope_target="specimen")``,
`potential.apply_dose_damage`) instead of to the transfer function, which
filters the ice as hard as the protein. The solvent then loses coherence
between frames instead (``Ice(motion_variance=...)``). Both are opt-in; the
default keeps the envelope on the transfer function.
"""

from __future__ import annotations

import pytest
import torch

import specter
from specter.aberrations import dose_envelope
from specter.config import ParticleStackConfig
from specter.config._validation import validate_config
from specter.imagegenerator import ImageGenerator, MicrographGenerator
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


def _generator(
    volume: torch.Tensor,
    dose_envelope: bool,
    ice: Ice,
    target: str = "specimen",
) -> ImageGenerator:
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
        envelopes=Envelopes(dose_envelope=dose_envelope, dose_envelope_target=target),  # type: ignore[arg-type]
        camera=Camera(noise_model=None),
        ice=ice,
    )


# --- the damage filter itself ----------------------------------------------


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


# --- where the envelope acts -----------------------------------------------


def test_default_keeps_the_envelope_on_the_transfer_function(small_volume):
    gen = _generator(small_volume, True, Ice(model=None), target="transfer_function")
    assert gen._damages_potential is False
    assert gen.aberration.dose_envelope is True
    assert Envelopes().dose_envelope_target == "transfer_function"


def test_specimen_target_moves_the_envelope_off_the_transfer_function(small_volume):
    gen = _generator(small_volume, True, Ice(model=None), target="specimen")
    assert gen._damages_potential is True
    assert gen.aberration.dose_envelope is False


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


def test_solvent_is_not_damaged_on_the_specimen_path():
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


def test_transfer_function_envelope_does_damage_the_solvent():
    """The default placement filters the ice with everything else."""
    empty = torch.zeros(32, 32, 32)
    images = []
    for flag in (True, False):
        specter.seed(3)
        gen = _generator(empty, flag, Ice(model="random"), target="transfer_function")
        images.append(gen(torch.tensor([0])).clone())
    assert not torch.allclose(images[0], images[1])


def test_free_fraction_path_matches_reading_occupancy_off_the_volume(small_volume):
    """
    With nothing damaged in between, weighting the ice by a precomputed free
    fraction must reproduce the in-line occupancy read to float16 precision.
    """
    gen = _generator(small_volume, False, Ice(model="random"))
    V = gen.V.clone().unsqueeze(0)
    free = gen.free_fraction_field(V)
    assert free.dtype == torch.float16
    assert float(free.min()) >= 0.0 and float(free.max()) <= 1.0
    specter.seed(5)
    inline = gen.solvate(V.clone())
    specter.seed(5)
    precomputed = gen.solvate(V.clone(), free=free)
    assert torch.allclose(precomputed, inline, atol=2e-3 * float(inline.abs().max()))


def test_micrograph_generator_refuses_the_specimen_target(small_volume_4d):
    with pytest.raises(ValueError, match="cannot apply the dose envelope"):
        MicrographGenerator(
            small_volume_4d,
            micrograph_size=32,
            pixel_size=2.0,
            ctf_params=_ctf_params(),
            voltage=300.0,
            dose_per_angstrom=40.0,
            envelopes=Envelopes(dose_envelope=True, dose_envelope_target="specimen"),
            verbose=False,
            progressbars=False,
        )


# --- solvent exposure --------------------------------------------------------


def test_solvent_motion_refuses_an_envelope_on_the_transfer_function(small_volume):
    with pytest.raises(ValueError, match="dose_envelope_target='specimen'"):
        _generator(
            small_volume,
            True,
            Ice(model="random", motion_variance=0.38),
            target="transfer_function",
        )
    # Fine without an envelope, and with it on the specimen.
    _generator(
        small_volume,
        False,
        Ice(model="random", motion_variance=0.38),
        "transfer_function",
    )
    _generator(
        small_volume, True, Ice(model="random", motion_variance=0.38), "specimen"
    )


def test_config_rejects_solvent_motion_with_a_transfer_function_envelope():
    with pytest.raises(ValueError, match="ice_motion_variance"):
        validate_config(
            ParticleStackConfig(
                pdb_source="1a6m", ice_motion_variance=0.38, dose_envelope=True
            )
        )
    validate_config(
        ParticleStackConfig(
            pdb_source="1a6m",
            ice_motion_variance=0.38,
            dose_envelope=True,
            dose_envelope_target="specimen",
        )
    )


def test_solvent_motion_filters_the_fluctuation_and_keeps_the_mean():
    """
    Ice alone (an empty specimen, so it all goes in): the exposure keeps the
    solvent's mean potential and removes most of its speckle.
    """
    empty = torch.zeros(32, 32, 32)
    fields = []
    for mv in (None, 0.38):
        gen = _generator(
            empty, False, Ice(model="gd", motion_variance=mv), "transfer_function"
        )
        V = torch.zeros(1, gen.nz, 32, 32)
        specter.seed(11)
        fields.append(gen.solvate(V))
    frozen, exposed = fields
    assert float(frozen.std()) > 0
    assert float(exposed.mean()) == pytest.approx(float(frozen.mean()), rel=1e-4)
    assert float(exposed.std()) < 0.5 * float(frozen.std())
