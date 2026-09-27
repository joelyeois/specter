"""Physical and geometry checks for explicit tilt-series absorption fields."""

import math

import pytest
import torch

import specter
from specter.cpu_threads import limited_cpu_threads
from specter.imagegenerator import ImageGenerator, TiltSeriesGenerator
from specter.potential import (
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    aperture_lowpass_isotropic,
    ice_inelastic_mfp,
    inelastic_absorption_potential,
)
from specter.settings import Camera, Crowding, Ice, Optics, Propagation, TiltGeometry

MFP = Propagation(absorption_model="inelastic_mfp")
ANGLES = (-60, -30, 0, 30, 60)


def blob(nz=16, nxy=64):
    """A small protein-like blob in a dry specimen volume."""
    volume = torch.zeros(1, nz, nxy, nxy)
    c = nxy // 2
    volume[:, nz // 2 - 3 : nz // 2 + 3, c - 4 : c + 4, c - 4 : c + 4] = 9.0
    return volume


def seeded(volume, *, seed=0, **kwargs):
    """
    A generator built under a fixed seed, so its ice is reproducible. One
    thread, since a multithreaded ice splat reorders its sums (~4e-6 V).
    """
    specter.seed(seed)
    with limited_cpu_threads(1):
        return make(volume.clone(), **kwargs)


def images(model):
    with torch.no_grad(), limited_cpu_threads(1):
        return model.generate_tilt_series(torch.tensor([0]))


def make(volume, field=None, angles=(-45, 0, 45), pixel_size=5.0, **kwargs):
    return TiltSeriesGenerator(
        volume,
        24,
        pixel_size,
        kwargs.pop("ctf_params", None),
        300.0,
        1.0,
        angles=list(angles),
        optics=kwargs.pop("optics", None),
        camera=Camera(noise_model=None, detector_model=None),
        absorption_potential=field,
        progressbars=False,
        verbose=False,
        **kwargs,
    )


@pytest.mark.parametrize("pad_fft", [False, True])
def test_uniform_slab_obeys_tilted_beer_lambert(pad_fft):
    angles = [-60, -45, 0, 45, 60]
    volume = torch.zeros(1, 16, 64, 64)
    field = torch.full_like(volume, absorption_potential(1000, 300))
    model = make(
        volume,
        field,
        angles=angles,
        propagation=Propagation(absorption_model="inelastic_mfp", pad_fft=pad_fft),
    )
    with torch.no_grad():
        _, wave, clean = model.generate_tilt_series(torch.tensor([0]))
    for i, angle in enumerate(angles):
        path = 80 / math.cos(math.radians(angle))
        observed = clean[0, i, 8:16, 8:16].mean().item()
        # Measured: 0.02% in transmission, 0.17% in inferred path length at
        # +/-45 deg, where the tilted slab faces cut the voxel grid worst.
        assert observed == pytest.approx(math.exp(-path / 1000), rel=5e-3)
        assert -1000 * math.log(observed) == pytest.approx(path, rel=5e-3)
    torch.testing.assert_close(clean, wave.abs().square())


def test_zero_absorption_preserves_legacy_exitwave():
    torch.manual_seed(42)
    volume = torch.rand(1, 8, 48, 48)
    legacy = make(volume.clone())
    explicit = make(volume.clone(), torch.zeros_like(volume))
    with torch.no_grad():
        a = legacy.generate_tilt_series(torch.tensor([0]))
        b = explicit.generate_tilt_series(torch.tensor([0]))
    for expected, observed in zip(a, b, strict=True):
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_padding_and_taper_keep_material_fields_aligned():
    volume = torch.ones(1, 12, 24, 24)
    model = make(volume, 2 * volume, tilt=TiltGeometry(taper_width=3, z_taper_width=2))
    assert model.volume.shape[-1] > 24
    torch.testing.assert_close(model.absorption_potential, 2 * model.volume)


@pytest.mark.parametrize("bad", ["negative", "nan", "shape", "alpha"])
def test_invalid_absorption_is_rejected(bad):
    volume = torch.zeros(1, 8, 32, 32)
    field = torch.zeros_like(volume)
    propagation = Propagation()
    if bad == "negative":
        field[0, 0, 0, 0] = -1
    elif bad == "nan":
        field[0, 0, 0, 0] = float("nan")
    elif bad == "shape":
        field = field[:, :-1]
    else:
        propagation = Propagation(alpha=0.07)
    with pytest.raises(ValueError, match="absorption|alpha"):
        make(volume, field, propagation=propagation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paired_field_oom_falls_back_without_changing_wave(monkeypatch):
    volume = torch.zeros(1, 8, 48, 48)
    volume[:, 2:6, 20:28, 20:28] = 10
    field = torch.full_like(volume, absorption_potential(1000, 300))
    resident = make(volume.clone(), field.clone()).cuda()
    streamed = make(volume.clone(), field.clone()).cuda()
    with torch.no_grad():
        reference = resident.generate_tilt_series(torch.tensor([0]))[1]
    source = streamed.absorption_potential
    original_to = torch.Tensor.to

    def fail_field(tensor, *args, **kwargs):
        if tensor is source:
            raise torch.cuda.OutOfMemoryError("simulated paired-field upload OOM")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_field)
    with pytest.warns(UserWarning, match="streaming both"), torch.no_grad():
        actual = streamed.generate_tilt_series(torch.tensor([0]))[1]
    assert streamed.volume.device.type == "cpu"
    assert streamed.absorption_potential.device.type == "cpu"
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)


def test_blended_ice_absorbs_along_the_tilted_path():
    """
    With no explicit field, ice now absorbs: a pure-ice slab transmits
    exp(-t / (cos(theta) * Lambda_ice)) relative to the same ice unabsorbed.
    """
    volume = torch.zeros(1, 32, 64, 64)
    ice = Ice(model="random")
    absorbed = seeded(volume, angles=ANGLES, propagation=MFP, ice=ice)
    reference = seeded(volume, angles=ANGLES, ice=ice)
    assert torch.equal(absorbed.volume, reference.volume)
    clean = images(absorbed)[2]
    unabsorbed = images(reference)[2]
    for i, angle in enumerate(ANGLES):
        path = 160 / math.cos(math.radians(angle))
        ratio = clean[0, i, 8:16, 8:16].mean() / unabsorbed[0, i, 8:16, 8:16].mean()
        # Measured: 2e-5 relative up to +/-60 deg.
        assert ratio.item() == pytest.approx(
            math.exp(-path / ice_inelastic_mfp(300.0)), rel=5e-4
        )


def test_specimen_field_is_read_off_the_dry_specimen_in_the_volume_frame():
    """
    With a specimen mean free path, the field is inelastic_absorption_potential
    of the DRY specimen, and it sits where the elastic potential does.
    """
    dry = blob(nxy=96)  # large enough that no tilt padding is needed
    propagation = Propagation(
        absorption_model="inelastic_mfp",
        inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A,
    )
    model = seeded(
        dry, angles=(-30, 0, 30), propagation=propagation, ice=Ice(model="random")
    )
    assert model.volume.shape == dry.shape
    expected = inelastic_absorption_potential(
        dry,
        5.0,
        300.0,
        mfp_solvent_A=ice_inelastic_mfp(300.0),
        mfp_specimen_A=INELASTIC_MFP_PROTEIN_A,
    )
    assert torch.equal(model.absorption_potential, expected)
    # Aligned: the most absorbing voxel is inside the specimen.
    specimen = dry > 0
    assert specimen.flatten()[model.absorption_potential.argmax()]
    solvent = absorption_potential(ice_inelastic_mfp(300.0), 300.0)
    assert model.absorption_potential[~specimen].min() > 0.9 * solvent


@pytest.mark.parametrize("aperture", [None, 12.0])
@pytest.mark.parametrize("specimen_mfp", [None, INELASTIC_MFP_PROTEIN_A])
def test_built_field_matches_the_explicit_override(specimen_mfp, aperture):
    """The field built from the settings is what a user would pass by hand."""
    propagation = Propagation(
        absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
    )
    optics = None if aperture is None else Optics(objective_aperture=aperture)
    kwargs = dict(
        angles=ANGLES,
        propagation=propagation,
        ice=Ice(model="random"),
        tilt=TiltGeometry(taper_width=3, z_taper_width=2),
    )
    if optics is not None:
        n = len(ANGLES)
        kwargs["optics"] = optics
        kwargs["ctf_params"] = {
            "dfu": torch.full((n,), 1.0e4),
            "dfv": torch.full((n,), 1.0e4),
            "dfang": torch.zeros(n),
            "cs": torch.full((n,), 2.7e7),
        }
    built = seeded(blob(), **kwargs)
    mfps = built._removal_mfps
    field = inelastic_absorption_potential(
        blob(),
        5.0,
        300.0,
        mfp_solvent_A=mfps.removal("solvent"),
        mfp_specimen_A=None if specimen_mfp is None else mfps.removal("specimen"),
    )
    explicit = seeded(blob(), field=field, **kwargs)
    assert torch.equal(built.absorption_potential, explicit.absorption_potential)
    for a, b in zip(images(built), images(explicit), strict=True):
        assert torch.equal(a, b)


def _particle_generator(aperture, specimen_mfp):
    return ImageGenerator(
        torch.zeros(16, 16, 16),
        5.0,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.zeros(1, 2),
        {
            "dfu": torch.tensor([1.0e4]),
            "dfv": torch.tensor([1.0e4]),
            "dfang": torch.tensor([0.0]),
            "cs": torch.tensor([2.7e7]),
        },
        300.0,
        dose_per_angstrom=40.0,
        propagation=Propagation(
            absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
        ),
        optics=Optics(objective_aperture=aperture),
        ice=Ice(model="gd", thickness=100.0),
        crowding=Crowding(n_points=0),
        camera=Camera(noise_model="none", detector_model="none"),
        progressbars=False,
    )


def test_aperture_removal_matches_the_particle_path():
    """An objective aperture charges the tilt series the particle path's loss."""
    n = len(ANGLES)
    ctf = {
        "dfu": torch.full((n,), 1.0e4),
        "dfv": torch.full((n,), 1.0e4),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), 2.7e7),
    }
    tilt = make(
        blob(),
        angles=ANGLES,
        ctf_params=ctf,
        optics=Optics(objective_aperture=12.0),
        propagation=Propagation(
            absorption_model="inelastic_mfp",
            inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A,
        ),
    )
    particle = _particle_generator(12.0, INELASTIC_MFP_PROTEIN_A)
    for material in ("solvent", "specimen"):
        assert tilt._removal_mfp(material) == particle._removal_mfp(material)
    # Both loss channels are counted: shorter than the inelastic path alone.
    assert tilt._removal_mfp("solvent") < ice_inelastic_mfp(300.0)
    assert tilt.objective_aperture == 12.0
    assert tilt.optics.objective_aperture == 12.0


def test_aperture_requires_the_mfp_model():
    with pytest.raises(ValueError, match="objective_aperture"):
        make(
            blob(),
            ctf_params={
                k: torch.full((3,), v)
                for k, v in {"dfu": 1e4, "dfv": 1e4, "dfang": 0.0, "cs": 2.7e7}.items()
            },
            optics=Optics(objective_aperture=12.0),
        )


def test_aperture_lowpasses_the_elastic_volume_isotropically():
    """
    Below 0.82 A/px a 12 mrad aperture lies inside the grid: the elastic
    volume keeps nothing beyond it in any direction, so no tilt sees any.
    """
    torch.manual_seed(0)
    volume = torch.rand(1, 16, 64, 64)
    n = 3
    model = make(
        volume,
        pixel_size=0.5,
        ctf_params={
            k: torch.full((n,), v)
            for k, v in {"dfu": 1e4, "dfv": 1e4, "dfang": 0.0, "cs": 2.7e7}.items()
        },
        optics=Optics(objective_aperture=12.0),
        propagation=MFP,
    )
    spectrum = torch.fft.fftn(model.volume, dim=(-3, -2, -1)).abs()
    nz, ny, nx = model.volume.shape[-3:]
    kz = torch.fft.fftfreq(nz, d=0.5)[:, None, None]
    ky = torch.fft.fftfreq(ny, d=0.5)[None, :, None]
    kx = torch.fft.fftfreq(nx, d=0.5)[None, None, :]
    k_ap = 0.012 / 0.019687  # rad / wavelength at 300 kV, 1/A
    outside = (kz**2 + ky**2 + kx**2).sqrt() > 1.01 * k_ap
    assert spectrum[0][outside].max() < 1e-4 * spectrum.max()
    # A no-op on a grid coarser than the aperture.
    assert aperture_lowpass_isotropic(volume, 1.0, 12.0, 300.0) is volume


def test_isotropic_aperture_lowpass_commutes_with_rotation():
    """A 90-degree rotation of the grid commutes with the spherical filter."""
    torch.manual_seed(1)
    v = torch.rand(1, 32, 32, 32)
    a = aperture_lowpass_isotropic(v, 0.5, 12.0, 300.0).transpose(-3, -1)
    b = aperture_lowpass_isotropic(v.transpose(-3, -1), 0.5, 12.0, 300.0)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


@pytest.mark.parametrize("specimen_mfp", [None, INELASTIC_MFP_PROTEIN_A])
def test_vacuum_does_not_absorb_without_ice(specimen_mfp):
    """
    With no ice there is no solvent term: vacuum transmits fully. Without a
    specimen mean free path nothing absorbs at all and no field is built.
    """
    propagation = Propagation(
        absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
    )
    model = make(blob(), angles=ANGLES, propagation=propagation)
    if specimen_mfp is None:
        assert model.absorption_potential is None
        return
    assert model.absorption_potential.min() == 0
    assert model.absorption_potential.max() > 0
    empty = make(torch.zeros(1, 16, 64, 64), angles=ANGLES, propagation=propagation)
    assert torch.count_nonzero(empty.absorption_potential) == 0
    torch.testing.assert_close(
        images(empty)[2], torch.ones_like(images(empty)[2]), rtol=0, atol=1e-6
    )
