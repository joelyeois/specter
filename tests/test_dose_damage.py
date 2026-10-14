"""
Where radiation damage acts, and what happens to the solvent under exposure.

Raw EER movies show the 3.7 A water ring does not fade under exposure, so a
specimen can take the dose envelope on its potential before solvation
(``Specimen.damage()``, `potential.apply_dose_damage`) instead of on the
transfer function (``ExitWave.apply_ctf(dose_envelope=...)``), which filters
the ice as hard as the protein. The solvent then loses coherence between
frames instead (``Ice(motion_variance=...)``). Both are opt-in.
"""

from __future__ import annotations

import pytest
import torch

from specter.aberrations import dose_envelope
from specter.config import ParticleStackConfig
from specter.config._validation import validate_config
from specter.ice import RandomIcemaker
from specter.imaging import CTF, Exposure, Poses, Sample, Specimen
from specter.imaging import _stages as stages
from specter.potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    apply_dose_damage,
    frame_damage_envelope,
    potential_occupancy,
)


DOSE = 40.0
LINEAR = dict(scattering_model="ctf", alpha=0.0)


def _ctf() -> CTF:
    return CTF(
        {
            "dfu": torch.tensor([8000.0]),
            "dfv": torch.tensor([8000.0]),
            "dfang": torch.tensor([0.0]),
            "cs": torch.tensor([2.7]),
            "phaseshift": torch.tensor([0.0]),
        }
    )


def _pose() -> Poses:
    return Poses(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 2), torch.arange(1)
    )


def _waves(specimen: Specimen, seed: int | None = None):
    """The specimen's exit wave at one exposure, linear model, no absorption."""
    with torch.no_grad():
        return specimen.scatter(
            _pose(), voltage=300.0, exposure=Exposure(DOSE), seed=seed, **LINEAR
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


def test_potential_damage_equals_transfer_function_envelope_for_the_specimen(
    small_volume,
):
    """
    Without solvent the two placements are the same filter, by the
    projection-slice theorem: the 3D radial envelope at kz = 0 is the 2D
    envelope on the projection.
    """
    proj_on = _waves(Specimen(small_volume, 2.0).damage()).pixels()
    proj_off = _waves(Specimen(small_volume, 2.0)).pixels()
    n = proj_off.shape[-1]
    k = torch.fft.fftfreq(n, d=2.0)
    kk = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    env = dose_envelope(kk, torch.tensor(40.0), weighted=True, voltage=300.0)
    expected = torch.fft.ifft2(torch.fft.fft2(proj_off) * env).real
    assert torch.allclose(proj_on, expected, atol=1e-4 * proj_off.abs().max())
    assert not torch.allclose(proj_on, proj_off, atol=1e-3 * proj_off.abs().max())


def test_solvent_is_not_damaged_on_the_specimen_path():
    """An empty specimen in ice images identically with the envelope on or off."""
    solvated = Specimen(torch.zeros(32, 32, 32), 2.0).solvate(model="random")
    on = _waves(solvated.damage(), seed=3).pixels()
    off = _waves(solvated, seed=3).pixels()
    assert on.abs().max() > 0
    assert torch.allclose(on, off)


def test_transfer_function_envelope_does_damage_the_solvent():
    """The default placement filters the ice with everything else."""
    waves = _waves(Specimen(torch.zeros(32, 32, 32), 2.0).solvate(model="random"), 3)
    images = [
        waves.apply_ctf(_ctf(), dose_envelope=envelope)
        .intensity()
        .detect(Exposure(DOSE), noise=None)
        .array
        for envelope in (Exposure(DOSE), None)
    ]
    assert not torch.allclose(images[0], images[1])


def test_free_fraction_path_matches_reading_occupancy_off_the_volume(small_volume):
    """
    With nothing damaged in between, weighting the ice by a precomputed free
    fraction must reproduce the in-line occupancy read to float16 precision.
    """
    V = small_volume.clone().unsqueeze(0)
    full = torch.full((1, 1, 1, 1), FULL_OCCUPANCY_POTENTIAL_V)
    free = stages.free_fraction(V, 2.0, full)
    assert free.dtype == torch.float16
    assert float(free.min()) >= 0.0 and float(free.max()) <= 1.0
    icemaker = RandomIcemaker(2.0, 32, 32)
    kwargs = dict(pixel_size=2.0, nxy=32, pad=0, full=full, seeds=[5])
    inline = stages.solvate(V.clone(), icemaker, **kwargs)
    precomputed = stages.solvate(V.clone(), icemaker, free=free, **kwargs)
    assert torch.allclose(precomputed, inline, atol=2e-3 * float(inline.abs().max()))


def test_a_prepared_volume_takes_the_dose_on_the_specimen(small_volume_4d):
    """
    A pre-assembled volume prepared with ``damaged=True`` damages itself; the
    transfer function then refuses a dose envelope. See
    test_exposure_micrograph_tilt.
    """
    sample = Sample.from_volume(small_volume_4d, 2.0).prepare(
        voltage=300.0, damaged=True
    )
    assert sample.series is not None
    with torch.no_grad():
        waves = sample.scatter(voltage=300.0, exposure=Exposure(DOSE))
    assert waves.damaged
    with pytest.raises(ValueError, match="twice"):
        waves.apply_ctf(_ctf(), dose_envelope=Exposure(DOSE))


# --- solvent exposure --------------------------------------------------------


def test_solvent_motion_refuses_an_envelope_on_the_transfer_function(small_volume):
    moving = Specimen(small_volume, 2.0).solvate(model="random", motion_variance=0.38)
    waves = _waves(moving)
    assert waves.solvent_filtered
    with pytest.raises(ValueError, match="twice"):
        waves.apply_ctf(_ctf(), dose_envelope=Exposure(DOSE))
    # Fine without an envelope, and with it on the specimen.
    waves.apply_ctf(_ctf())
    _waves(moving.damage()).apply_ctf(_ctf())


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
        specimen = Specimen(empty, 2.0).solvate(model="gd", motion_variance=mv)
        with torch.no_grad():
            fields.append(
                specimen.materialize(
                    _pose(), voltage=300.0, exposure=Exposure(DOSE), seed=11
                )
            )
    frozen, exposed = fields
    assert float(frozen.std()) > 0
    assert float(exposed.mean()) == pytest.approx(float(frozen.mean()), rel=1e-4)
    assert float(exposed.std()) < 0.5 * float(frozen.std())
