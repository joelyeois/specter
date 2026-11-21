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
from specter.imaging import CTF, Exposure, Field, Poses, Sample, Specimen
from specter.imaging import _stages as stages
from specter.settings import Crowding, Ice, Propagation
from specter.potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    apply_dose_damage,
    frame_damage_envelope,
    potential_occupancy,
)


DOSE = 40.0
#: Posed exactly whether or not the specimen is damaged: a damaged specimen
#: is dressed and cannot take the surface its bare copy would pair with, and
#: the comparisons here are between the two.
LINEAR = dict(scattering_model="ctf", alpha=0.0, rotate_mode="exact")


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


def test_a_damaged_specimen_displaces_water_through_its_envelope(small_volume):
    """
    The occupancy takes the envelope the specimen took, so the envelope acts
    on the contrast against the solvent; the volume displaced is unchanged.
    """
    V = small_volume.clone().unsqueeze(0)
    full = torch.full((1, 1, 1, 1), FULL_OCCUPANCY_POTENTIAL_V)
    free = stages.free_fraction(V, 2.0, full)
    expected = apply_dose_damage(1.0 - free.float(), 2.0, DOSE, n_frames=40)
    got = stages.damaged_free_fraction(
        free.clone(), 2.0, torch.tensor([DOSE]), n_frames=40
    )
    assert got.dtype == torch.float16
    assert torch.allclose(1.0 - got.float(), expected.clamp(0.0, 1.0), atol=1e-3)
    assert not torch.allclose(got.float(), free.float(), atol=1e-2)
    assert float(got.float().mean()) == pytest.approx(
        float(free.float().mean()), abs=2e-3
    )


def test_dose_envelope_on_a_solvated_specimen_fades_its_contrast():
    """
    In ice the low-resolution image of a damaged specimen is the envelope
    times that of the undamaged one. A ball at full occupancy keeps 0.31 of
    its vacuum contrast in ice, so a solvent hole left at full depth would
    cost it 0.87 where the envelope is 0.96.
    """
    n = 48
    x = torch.arange(n) - n / 2
    r = 2.0 * torch.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x**2)
    ball = FULL_OCCUPANCY_POTENTIAL_V * torch.sigmoid((20.0 - r) / 2.0)
    solvated = Specimen(ball, 2.0).solvate(model="random")
    exposure = Exposure(DOSE, n_frames=40)

    def image(specimen: Specimen) -> torch.Tensor:
        with torch.no_grad():
            waves = specimen.scatter(
                _pose(), voltage=300.0, exposure=exposure, seed=3, **LINEAR
            )
        return waves.pixels()[0]

    bare = image(Specimen(ball, 2.0))
    on, off = image(solvated.damage()), image(solvated)
    k = torch.fft.fftfreq(n, d=2.0)
    radius = torch.hypot(k[:, None], k[None, :])
    band = (radius > 0) & (radius < 1 / 10.0)
    reference = torch.fft.fft2(bare - bare.mean())

    def kept(x: torch.Tensor) -> float:
        return float((torch.fft.fft2(x - x.mean()) * reference.conj()).real[band].sum())

    envelope = frame_damage_envelope(radius, DOSE, 40)
    weight = reference.abs() ** 2
    expected = float((envelope * weight)[band].sum() / weight[band].sum())
    assert kept(on) / kept(off) == pytest.approx(expected, abs=0.02)


def _ball(n: int = 48) -> torch.Tensor:
    x = torch.arange(n) - n / 2
    r = 2.0 * torch.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x**2)
    return FULL_OCCUPANCY_POTENTIAL_V * torch.sigmoid((20.0 - r) / 2.0)


def _contrast_kept(
    on: torch.Tensor, off: torch.Tensor, bare: torch.Tensor, exposure: Exposure
) -> tuple[float, float]:
    """
    The share of the low-resolution image of `bare` that `on` keeps of
    `off`, and the dose envelope's own, weighted by the same spectrum.
    """
    k = torch.fft.fftfreq(on.shape[-1], d=2.0)
    radius = torch.hypot(k[:, None], k[None, :])
    band = (radius > 0) & (radius < 1 / 10.0)
    reference = torch.fft.fft2(bare - bare.mean())

    def kept(x: torch.Tensor) -> float:
        return float((torch.fft.fft2(x - x.mean()) * reference.conj()).real[band].sum())

    dose = float(torch.as_tensor(exposure.dose).reshape(-1)[-1])
    pre = exposure.pre_exposure
    before = 0.0 if pre is None else float(torch.as_tensor(pre).reshape(-1)[-1])
    envelope = exposure.damage_envelope(dose, before, 300.0)(radius)
    weight = reference.abs() ** 2
    expected = float((envelope * weight)[band].sum() / weight[band].sum())
    return kept(on) / kept(off), expected


MODEL = dict(scattering_model="ctf", alpha=0.0)


@pytest.mark.parametrize("motion_variance", [None, 0.3])
def test_a_damaged_prepared_sample_fades_its_contrast(motion_variance):
    """
    A pre-assembled volume in ice keeps the envelope's share of its
    low-resolution contrast, 0.96, where a solvent hole left at full depth
    kept 0.87.
    """
    ball, exposure = _ball(), Exposure(DOSE, n_frames=40)
    ice = Ice(model="random", motion_variance=motion_variance)

    def image(damaged: bool) -> torch.Tensor:
        torch.manual_seed(5)
        sample = Sample.from_volume(ball[None].clone(), 2.0).prepare(
            ice,
            voltage=300.0,
            icemaker=RandomIcemaker(2.0, 48, 48),
            damaged=damaged,
            exposure=exposure,
            **MODEL,
        )
        with torch.no_grad():
            waves = sample.scatter(voltage=300.0, exposure=exposure, **MODEL)
        return waves.pixels()[0]

    with torch.no_grad():
        bare = Sample.from_volume(ball[None], 2.0).scatter(voltage=300.0, **MODEL)
    got, expected = _contrast_kept(
        image(True), image(False), bare.pixels()[0], exposure
    )
    assert got == pytest.approx(expected, abs=0.02)


def test_a_damaged_field_fades_its_particles_contrast():
    """The same for a micrograph specimen drawn from a template."""
    exposure = Exposure(DOSE, n_frames=40)
    template = _ball()[8:40, 8:40, 8:40].clone()

    def draw(damaged: bool) -> Sample:
        field = Field(
            template, 2.0, 96, nz=40, crowding=Crowding(min_distance=70.0)
        ).solvate(Ice(model="random"), icemaker=RandomIcemaker(2.0, 96, 40))
        if damaged:
            field = field.damage()
        return field.draw(
            voltage=300.0, exposure=exposure, seed=7, index=0, keep_dry=True, **MODEL
        )

    def image(sample: Sample) -> torch.Tensor:
        with torch.no_grad():
            return sample.scatter(voltage=300.0, **MODEL).pixels()[0].cpu()

    on, off = draw(True), draw(False)
    got, expected = _contrast_kept(image(on), image(off), image(off.clean()), exposure)
    assert got == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize(
    "dose, motion_variance",
    [(10.0, None), (10.0, 0.3), (torch.tensor([8.0, 10.0, 12.0]), 0.3)],
    ids=["fixed ice", "one filtered ice", "ice per tilt"],
)
def test_a_damaged_tilt_fades_its_contrast(dose, motion_variance):
    """
    The same for the last tilt of a series, through each of the three ways
    its ice is held. Untilted views, so that the bare specimen's image is
    the reference for every one.
    """
    ball = _ball()
    exposure = Exposure.tilt_series(dose, 3)

    def image(damaged: bool, ice: Ice) -> torch.Tensor:
        torch.manual_seed(5)
        series = Sample.from_volume(ball[None].clone(), 2.0).tilt_series(
            Poses.tilt_series([0.0, 0.0, 0.0]),
            exposure,
            voltage=300.0,
            micrograph_size=48,
            ice=ice,
            icemaker=RandomIcemaker(2.0, 48, 48) if ice.model else None,
            damaged=damaged,
            seed=7,
            propagation=Propagation(**MODEL),
            log=False,
        )
        with torch.no_grad():
            return series.scatter(2).pixels()[0].cpu()

    ice = Ice(model="random", motion_variance=motion_variance)
    bare = image(False, Ice())
    got, expected = _contrast_kept(
        image(True, ice), image(False, ice), bare, exposure[2:3]
    )
    assert got == pytest.approx(expected, abs=0.02)
