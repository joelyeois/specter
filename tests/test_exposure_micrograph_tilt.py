"""Specimen-only dose damage and solvent motion in micrographs and tilt series."""

import pytest
import torch

import specter
from specter.aberrations import dose_envelope
from specter.cpu_threads import limited_cpu_threads
from specter.fft import RadialSpectrum, apply_radial_envelope_
from specter.ice import apply_solvent_exposure
from specter.imaging import CTF, Exposure, Field, Poses, Sample
from specter.imaging._tilt import fit_volume_to_tilt
from specter.potential import apply_dose_damage
from specter.settings import Ice, TiltGeometry

DX = 2.0
DOSE = 40.0


def template():
    torch.manual_seed(0)
    v = torch.zeros(12, 12, 12)
    v[3:9, 3:9, 3:9] = 8.0 + torch.rand(6, 6, 6)
    return v


def field(ice=Ice(model="random"), crowd=True):
    f = Field(template() if crowd else None, DX, 32, nz=16)
    if crowd:
        f = f.pack(min_distance=20.0, n_points=3)
    return f.solvate(ice) if ice.model is not None else f


def draw(f, dose=None, seed=0, **kwargs):
    exposure = None if dose is None else Exposure(dose)
    with torch.no_grad(), limited_cpu_threads(1):
        return f.draw(voltage=300.0, exposure=exposure, seed=seed, **kwargs)


# --- micrographs ------------------------------------------------------------


def test_micrograph_specimen_damage_leaves_the_ice_untouched():
    """
    The damaged specimen is the dry specimen under the envelope for the
    micrograph's dose plus exactly the ice the undamaged one would get.
    """
    plain = draw(field(), keep_dry=True)
    damaged = draw(field().damage(), DOSE, keep_dry=True)
    dry = plain.dry
    torch.testing.assert_close(damaged.dry, dry, rtol=0, atol=0)
    ice = plain.volume - dry
    expected = (
        apply_dose_damage(dry.clone(), DX, DOSE, weighted=True, voltage=300.0) + ice
    )
    torch.testing.assert_close(damaged.volume, expected, rtol=1e-5, atol=1e-5)
    # Damage conserves the mean and removes high-frequency power.
    assert float(damaged.volume.mean()) == pytest.approx(
        float(plain.volume.mean()), rel=1e-5
    )


def test_micrograph_specimen_is_drawn_for_each_dose():
    """A damaged sample is imaged only at the dose it was drawn for."""
    sample = draw(field().damage(), 10.0)
    assert sample.dose == 10.0
    with torch.no_grad():
        sample.scatter(voltage=300.0, exposure=Exposure(10.0))
        with pytest.raises(ValueError, match="Draw another"):
            sample.scatter(voltage=300.0, exposure=Exposure(60.0))
    assert draw(field().damage(), 60.0).dose == 60.0
    with pytest.raises(ValueError, match="exposure"):
        draw(field().damage())


def test_micrograph_solvent_motion_filters_the_ice():
    """A pure-ice micrograph's ice is the unfiltered ice under the exposure filter."""
    plain = draw(field(Ice(model="random"), crowd=False))
    moving = draw(field(Ice(model="random", motion_variance=0.38), crowd=False), DOSE)
    expected = plain.volume.clone()
    apply_solvent_exposure(expected, DX, DOSE, 0.38, 1, None, None)
    torch.testing.assert_close(moving.volume, expected, rtol=1e-5, atol=1e-5)
    assert float(moving.volume.std()) < float(plain.volume.std())


def test_micrograph_motion_refuses_the_transfer_function_envelope():
    moving = draw(field(Ice(model="random", motion_variance=0.38)), DOSE)
    with torch.no_grad():
        waves = moving.scatter(voltage=300.0, exposure=Exposure(DOSE))
    with pytest.raises(ValueError, match="twice"):
        waves.apply_ctf(CTF.from_values(1.0e4), dose_envelope=Exposure(DOSE))


def test_micrograph_volume_specimen_is_damaged_per_exposure():
    """A pre-assembled volume is damaged at scatter time, for each dose."""
    volume = template().reshape(1, 12, 12, 12)
    volume = torch.nn.functional.pad(volume, (10, 10, 10, 10))
    original = volume.clone()
    sample = Sample.from_volume(volume, DX).prepare(voltage=300.0, damaged=True)
    for dose in (10.0, 60.0):
        with torch.no_grad():
            sample.scatter(voltage=300.0, exposure=Exposure(dose))
        expected = apply_dose_damage(original.clone(), DX, dose, voltage=300.0)
        torch.testing.assert_close(sample.volume, expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(volume, original), "the caller's tensor was written"
    with pytest.raises(ValueError, match="exposure"):
        sample.scatter(voltage=300.0)


# --- tilt series ------------------------------------------------------------


def tilt_volume():
    torch.manual_seed(1)
    v = torch.zeros(1, 16, 48, 48)
    v[:, 5:11, 18:30, 16:32] = 7.0 + torch.rand(6, 12, 16)
    return v


ANGLES = [-40.0, 0.0, 40.0]


def tilt_series(volume, ice=Ice(), damaged=True, dose=5.0):
    specter.seed(3)
    with limited_cpu_threads(1):
        return Sample.from_volume(volume.clone(), DX).tilt_series(
            Poses.tilt_series(ANGLES),
            Exposure.tilt_series(dose, len(ANGLES)),
            voltage=300.0,
            micrograph_size=32,
            ice=ice,
            damaged=damaged,
            tilt=TiltGeometry(taper_width=2),
            log=False,
        )


def fit(volume):
    """`volume` padded and tapered as a series of these tilts does it."""
    poses = Poses.tilt_series(ANGLES)
    fitted, _ = fit_volume_to_tilt(
        volume, 32, poses.quaternions, 8, 2, 0, True, log=False
    )
    return fitted


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
    plain = tilt_series(volume, Ice(model="random"), damaged=False)
    model = tilt_series(volume, Ice(model="random"))
    dry_model = tilt_series(volume, Ice(), damaged=False)
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
        for i in range(len(model)):
            model.scatter(i)
    assert seen == [(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)]


def test_tilt_damage_attenuates_high_k_by_the_tilts_envelope():
    """The specimen's spectrum falls by dose_envelope(k, dose, pre) at each tilt."""
    volume = tilt_volume()
    model = tilt_series(volume)
    dry = tilt_series(volume, damaged=False).volume[0]
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
    plain = tilt_series(volume, Ice(model="random"), damaged=False)
    moving = tilt_series(volume, motion, damaged=False)
    specter.seed(3)
    with limited_cpu_threads(1):
        from specter.ice import RandomIcemaker, blend_ice_into_volume

        ice = blend_ice_into_volume(
            volume.clone(), RandomIcemaker(dx=DX, n=48, nz=16), DX
        )
    apply_solvent_exposure(ice, DX, 5.0, 0.38, 1, None, None)
    expected = fit(ice)
    torch.testing.assert_close(moving.volume, expected, rtol=1e-5, atol=1e-5)
    assert float(moving.volume.std()) < float(plain.volume.std())


def ice_for_dose(volume, dose):
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
    return fit(blended)


def spy_on_tilts(model, check):
    """Run the series, calling ``check(i)`` once each tilt's volume is in place."""
    seen = []

    def hook(module, args):
        check(len(seen))
        seen.append(len(seen))

    handle = model.scattering.register_forward_pre_hook(hook)
    with torch.no_grad():
        for i in range(len(model)):
            model.scatter(i)
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
        damaged=False,
        dose=torch.tensor(doses),
    )
    expected = [ice_for_dose(volume, d) for d in doses]

    def check(i):
        torch.testing.assert_close(model.volume, expected[i], rtol=1e-5, atol=1e-5)

    assert spy_on_tilts(model, check) == [0, 1, 2]
    assert float(expected[1].std()) < float(expected[2].std())


def test_tilt_solvent_motion_at_one_dose_matches_the_single_filter():
    """A tilt rendered at dose d equals the ice of a series with d on every tilt."""
    volume = tilt_volume()
    motion = Ice(model="random", motion_variance=0.38)
    uniform = tilt_series(volume, motion, damaged=False)
    varying = tilt_series(
        volume, motion, damaged=False, dose=torch.tensor([8.0, 5.0, 8.0])
    )
    varying._render_solvent(5.0)
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
    dry = tilt_series(volume, Ice(), damaged=False).volume
    ice = [ice_for_dose(volume, d) - dry for d in doses]
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
