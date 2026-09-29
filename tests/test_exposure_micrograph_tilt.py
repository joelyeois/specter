"""Specimen-only dose damage and solvent motion in micrographs and tilt series."""

import pytest
import torch

import specter
from specter.aberrations import dose_envelope
from specter.cpu_threads import limited_cpu_threads
from specter.fft import RadialSpectrum, apply_radial_envelope_
from specter.ice import apply_solvent_exposure
from specter.imagegenerator import MicrographGenerator, TiltSeriesGenerator
from specter.potential import apply_dose_damage
from specter.settings import Camera, Crowding, Envelopes, Ice, TiltGeometry
from specter.specimen import MicrographSpecimenGenerator

DX = 2.0
DOSE = 40.0
SPECIMEN = Envelopes(dose_envelope=True, dose_envelope_target="specimen")
CAMERA = Camera(noise_model=None, detector_model=None)


def template():
    torch.manual_seed(0)
    v = torch.zeros(12, 12, 12)
    v[3:9, 3:9, 3:9] = 8.0 + torch.rand(6, 6, 6)
    return v


def specimen(ice=Ice(model="random"), crowd=True):
    return MicrographSpecimenGenerator(
        template() if crowd else None,
        DX,
        32,
        nz=16,
        crowding=Crowding(min_distance=20.0, n_points=3) if crowd else Crowding(),
        ice=ice,
        save_clean_exitwaves=crowd and ice.model is not None,
        progressbars=False,
    )


def micrograph(spec, envelopes=Envelopes(), dose=DOSE):
    return MicrographGenerator(
        spec,
        32,
        DX,
        None,
        300.0,
        dose,
        optics=None,
        envelopes=envelopes,
        camera=CAMERA,
        verbose=False,
        progressbars=False,
    )


def run(model, i=0, seed=0):
    specter.seed(seed)
    with torch.no_grad(), limited_cpu_threads(1):
        return model(torch.tensor([i]))


# --- micrographs ------------------------------------------------------------


def test_micrograph_specimen_damage_leaves_the_ice_untouched():
    """
    The damaged specimen is the dry specimen under the envelope for the
    micrograph's dose plus exactly the ice the undamaged one would get.
    """
    plain = micrograph(specimen(ice=Ice(model="random")))
    damaged = micrograph(specimen(ice=Ice(model="random")), SPECIMEN)
    run(plain)
    run(damaged)
    dry = plain.specimen_gen.clean_V
    torch.testing.assert_close(damaged.specimen_gen.clean_V, dry, rtol=0, atol=0)
    ice = plain.volume - dry
    expected = (
        apply_dose_damage(dry.clone(), DX, DOSE, weighted=True, voltage=300.0) + ice
    )
    torch.testing.assert_close(damaged.volume, expected, rtol=1e-5, atol=1e-5)
    # Damage conserves the mean and removes high-frequency power.
    assert float(damaged.volume.mean()) == pytest.approx(
        float(plain.volume.mean()), rel=1e-5
    )


def test_micrograph_specimen_is_rebuilt_for_each_dose():
    """A regenerated specimen is damaged for the dose of the micrograph imaging it."""
    model = micrograph(specimen(), SPECIMEN, dose=torch.tensor([10.0, 60.0]))
    run(model, 0)
    assert model._volume_dose == 10.0
    with pytest.raises(ValueError, match="regenerate_specimen"):
        run(model, 1)
    model.regenerate_specimen()
    assert not hasattr(model, "volume")
    run(model, 1)
    assert model._volume_dose == 60.0


def test_micrograph_solvent_motion_filters_the_ice():
    """A pure-ice micrograph's ice is the unfiltered ice under the exposure filter."""
    motion = Ice(model="random", motion_variance=0.38)
    plain = micrograph(specimen(Ice(model="random"), crowd=False))
    moving = micrograph(specimen(motion, crowd=False))
    run(plain)
    run(moving)
    expected = plain.volume.clone()
    apply_solvent_exposure(expected, DX, DOSE, 0.38, 1, None, None)
    torch.testing.assert_close(moving.volume, expected, rtol=1e-5, atol=1e-5)
    assert float(moving.volume.std()) < float(plain.volume.std())


def test_micrograph_motion_refuses_the_transfer_function_envelope():
    with pytest.raises(ValueError, match="dose_envelope_target='specimen'"):
        micrograph(
            specimen(Ice(model="random", motion_variance=0.38)),
            Envelopes(dose_envelope=True),
        )


def test_micrograph_volume_specimen_is_damaged_per_exposure():
    """A pre-assembled volume is damaged at forward time, for each dose."""
    volume = template().reshape(1, 12, 12, 12)
    volume = torch.nn.functional.pad(volume, (10, 10, 10, 10))
    original = volume.clone()
    model = MicrographGenerator(
        volume,
        32,
        DX,
        None,
        300.0,
        torch.tensor([10.0, 60.0]),
        optics=None,
        envelopes=SPECIMEN,
        camera=CAMERA,
        verbose=False,
        progressbars=False,
    )
    for i, dose in enumerate((10.0, 60.0)):
        run(model, i)
        expected = apply_dose_damage(original.clone(), DX, dose, voltage=300.0)
        torch.testing.assert_close(model.volume, expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(volume, original), "the caller's tensor was written"


# --- tilt series ------------------------------------------------------------


def tilt_volume():
    torch.manual_seed(1)
    v = torch.zeros(1, 16, 48, 48)
    v[:, 5:11, 18:30, 16:32] = 7.0 + torch.rand(6, 12, 16)
    return v


def tilt_series(volume, ice=Ice(), envelopes=SPECIMEN, dose=5.0, **kwargs):
    specter.seed(3)
    with limited_cpu_threads(1):
        return TiltSeriesGenerator(
            volume.clone(),
            32,
            DX,
            None,
            300.0,
            dose,
            angles=[-40.0, 0.0, 40.0],
            optics=None,
            envelopes=envelopes,
            camera=CAMERA,
            ice=ice,
            tilt=TiltGeometry(taper_width=2),
            verbose=False,
            progressbars=False,
            **kwargs,
        )


def test_radial_spectrum_matches_apply_radial_envelope():
    torch.manual_seed(2)
    v = torch.rand(10, 14, 12)
    env = lambda k: torch.exp(-3.0 * k**2)  # noqa: E731
    expected = v.clone()
    apply_radial_envelope_(expected, 1.5, env)
    out = torch.empty_like(v)
    RadialSpectrum(v, 1.5).filter_into(out, env)
    assert torch.equal(out, expected)


def test_each_tilt_is_damaged_at_its_own_pre_exposure():
    """
    Tilt i's specimen is the dry specimen under the envelope of its own dose
    after the pre-exposure of the tilts before it, and the ice is untouched.
    """
    volume = tilt_volume()
    plain = tilt_series(volume, Ice(model="random"), Envelopes())
    model = tilt_series(volume, Ice(model="random"))
    dry_model = tilt_series(volume, Ice(), Envelopes())
    dry = dry_model.volume  # the same padding and taper, without ice
    ice = plain.volume - dry
    seen = []
    render = model._render_exposure

    def spy(dose, pre):
        seen.append((dose, pre))
        render(dose, pre)
        expected = apply_dose_damage(
            dry.clone(), DX, dose, pre_exposure=pre, weighted=False, voltage=300.0
        )
        torch.testing.assert_close(model.volume, expected + ice, rtol=1e-5, atol=1e-5)

    model._render_exposure = spy
    with torch.no_grad():
        model.generate_tilt_series(torch.tensor([0]))
    assert seen == [(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)]


def test_tilt_damage_attenuates_high_k_by_the_tilts_envelope():
    """The specimen's spectrum falls by dose_envelope(k, dose, pre) at each tilt."""
    volume = tilt_volume()
    model = tilt_series(volume)
    dry = tilt_series(volume, envelopes=Envelopes()).volume[0]
    nz, ny, nx = dry.shape
    kz = torch.fft.fftfreq(nz, d=DX)
    ky = torch.fft.fftfreq(ny, d=DX)
    kx = torch.fft.rfftfreq(nx, d=DX)
    k = torch.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx**2)
    shell = (k > 0.18) & (k < 0.2)
    ref = torch.fft.rfftn(dry)[shell]
    for pre in (0.0, 5.0, 10.0):
        model._render_exposure(5.0, pre)
        ratio = (torch.fft.rfftn(model.volume[0])[shell] / ref).real
        expected = dose_envelope(k[shell], torch.tensor(5.0), pre, weighted=False)
        torch.testing.assert_close(ratio, expected, rtol=2e-3, atol=2e-4)
    assert float(expected.mean()) < 0.9


def test_tilt_solvent_motion_filters_the_ice_once():
    motion = Ice(model="random", motion_variance=0.38)
    volume = torch.zeros(1, 16, 48, 48)
    plain = tilt_series(volume, Ice(model="random"), Envelopes())
    moving = tilt_series(volume, motion, Envelopes())
    specter.seed(3)
    with limited_cpu_threads(1):
        from specter.ice import RandomIcemaker, blend_ice_into_volume

        ice = blend_ice_into_volume(
            volume.clone(), RandomIcemaker(dx=DX, n=48, nz=16), DX
        )
    apply_solvent_exposure(ice, DX, 5.0, 0.38, 1, None, None)
    expected = moving._fit_volume_to_tilt(
        ice, 32, [-40.0, 0.0, 40.0], None, 8, 2, 0, True
    )
    torch.testing.assert_close(moving.volume, expected, rtol=1e-5, atol=1e-5)
    assert float(moving.volume.std()) < float(plain.volume.std())


ANGLES = [-40.0, 0.0, 40.0]


def ice_for_dose(model, volume, dose):
    """The padded ice-filled volume a series of one dose would image."""
    from specter.ice import RandomIcemaker, blend_ice_into_volume

    specter.seed(3)
    with limited_cpu_threads(1):
        blended = blend_ice_into_volume(
            volume.clone(),
            RandomIcemaker(dx=DX, n=48, nz=16),
            DX,
            ice_filter=lambda c: apply_solvent_exposure(
                c, DX, dose, 0.38, 1, None, None
            ),
        )
    return model._fit_volume_to_tilt(blended, 32, ANGLES, None, 8, 2, 0, True)


def spy_on_tilts(model, check):
    """Run the series, calling ``check(i)`` once each tilt's volume is in place."""
    seen = []

    def hook(module, args):
        check(len(seen))
        seen.append(len(seen))

    handle = model.iterative_scattering.register_forward_pre_hook(hook)
    with torch.no_grad():
        model.generate_tilt_series(torch.tensor([0]))
    handle.remove()
    return seen


def test_tilt_solvent_motion_follows_each_tilts_dose():
    """
    With unequal doses each tilt's ice is apply_solvent_exposure at that
    tilt's own dose, blended around the specimen and padded, as a series of
    that one dose would have it.
    """
    volume = tilt_volume()
    doses = [5.0, 8.0, 2.0]
    model = tilt_series(
        volume,
        Ice(model="random", motion_variance=0.38),
        Envelopes(),
        dose=torch.tensor(doses),
    )
    expected = [ice_for_dose(model, volume, d) for d in doses]

    def check(i):
        torch.testing.assert_close(model.volume, expected[i], rtol=1e-5, atol=1e-5)

    assert spy_on_tilts(model, check) == [0, 1, 2]
    assert float(expected[1].std()) < float(expected[2].std())


def test_tilt_solvent_motion_at_one_dose_matches_the_single_filter():
    """A tilt rendered at dose d equals the ice of a series with d on every tilt."""
    volume = tilt_volume()
    motion = Ice(model="random", motion_variance=0.38)
    uniform = tilt_series(volume, motion, Envelopes())
    varying = tilt_series(
        volume, motion, Envelopes(), dose=torch.tensor([5.0, 8.0, 5.0])
    )
    varying._ensure_volume_placed()
    varying._render_tilt_solvent(5.0)
    torch.testing.assert_close(varying.volume, uniform.volume, rtol=1e-6, atol=1e-6)


def test_tilt_solvent_motion_per_tilt_with_specimen_damage():
    """Each tilt is its damaged dry specimen plus the ice of its own dose."""
    volume = tilt_volume()
    doses = [5.0, 8.0, 2.0]
    model = tilt_series(
        volume,
        Ice(model="random", motion_variance=0.38),
        dose=torch.tensor(doses),
    )
    dry = tilt_series(volume, Ice(), Envelopes()).volume
    ice = [ice_for_dose(model, volume, d) - dry for d in doses]
    pre = [0.0, 5.0, 13.0]

    def check(i):
        damaged = apply_dose_damage(
            dry.clone(),
            DX,
            doses[i],
            pre_exposure=pre[i],
            weighted=False,
            voltage=300.0,
        )
        torch.testing.assert_close(model.volume, damaged + ice[i], rtol=1e-5, atol=1e-5)

    assert spy_on_tilts(model, check) == [0, 1, 2]
