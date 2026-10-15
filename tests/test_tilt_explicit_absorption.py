"""Physical and geometry checks for explicit tilt-series absorption fields."""

import math

import pytest
import torch

from specter.arrays import center_crop

from specter.cpu_threads import limited_cpu_threads
from specter.imaging import Exposure, Poses, Sample, Specimen
from specter.imaging import _stages as stages
from specter.potential import (
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    aperture_lowpass_isotropic,
    ice_inelastic_mfp,
    inelastic_absorption_potential,
)
from specter.settings import Ice, Optics, Propagation, TiltGeometry

MFP = Propagation(absorption_model="inelastic_mfp")
ANGLES = (-60, -30, 0, 30, 60)


def blob(nz=16, nxy=64):
    """A small protein-like blob in a dry specimen volume."""
    volume = torch.zeros(1, nz, nxy, nxy)
    c = nxy // 2
    volume[:, nz // 2 - 3 : nz // 2 + 3, c - 4 : c + 4, c - 4 : c + 4] = 9.0
    return volume


def seeded(volume, **kwargs):
    """
    A series whose ice is drawn from a fixed seed. One thread, since a
    multithreaded ice splat reorders its sums (~4e-6 V).
    """
    with limited_cpu_threads(1):
        return make(volume.clone(), seed=0, **kwargs)


def waves(series, device=None):
    """Every tilt's exit wave on the field's pixel grid, ``(n_tilts, Y, X)``."""
    with torch.no_grad(), limited_cpu_threads(1):
        return torch.cat(
            [
                center_crop(
                    series.scatter(i, device=device).pixels(), series.nxy, dim=(-2, -1)
                )
                for i in range(len(series))
            ]
        )


def make(
    volume,
    field=None,
    angles=(-45, 0, 45),
    pixel_size=5.0,
    propagation=Propagation(),
    **kwargs,
):
    return Sample.from_volume(volume, pixel_size).tilt_series(
        Poses.tilt_series(list(angles)),
        Exposure.tilt_series(1.0, len(angles)),
        voltage=300.0,
        micrograph_size=24,
        absorption=field,
        propagation=propagation,
        log=False,
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
    clean = waves(model).abs().square()
    for i, angle in enumerate(angles):
        path = 80 / math.cos(math.radians(angle))
        observed = clean[i, 8:16, 8:16].mean().item()
        # Measured: 0.02% in transmission, 0.17% in inferred path length at
        # +/-45 deg, where the tilted slab faces cut the voxel grid worst.
        assert observed == pytest.approx(math.exp(-path / 1000), rel=5e-3)
        assert -1000 * math.log(observed) == pytest.approx(path, rel=5e-3)


def test_zero_absorption_preserves_legacy_exitwave():
    torch.manual_seed(42)
    volume = torch.rand(1, 8, 48, 48)
    legacy = make(volume.clone())
    explicit = make(volume.clone(), torch.zeros_like(volume))
    torch.testing.assert_close(waves(explicit), waves(legacy), rtol=1e-5, atol=1e-5)


def test_padding_and_taper_keep_material_fields_aligned():
    volume = torch.ones(1, 12, 24, 24)
    model = make(volume, 2 * volume, tilt=TiltGeometry(taper_width=3, z_taper_width=2))
    assert model.volume.shape[-1] > 24
    torch.testing.assert_close(model.absorption, 2 * model.volume)


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
    resident = make(volume.clone(), field.clone())
    streamed = make(volume.clone(), field.clone())
    reference = waves(resident, device="cuda")
    source = streamed.absorption
    original_to = torch.Tensor.to

    def fail_field(tensor, *args, **kwargs):
        if tensor is source:
            raise torch.cuda.OutOfMemoryError("simulated paired-field upload OOM")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_field)
    with pytest.warns(UserWarning, match="streaming them from the host"):
        actual = waves(streamed, device="cuda")
    assert streamed.volume.device.type == "cpu"
    assert streamed.absorption.device.type == "cpu"
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
    clean = waves(absorbed).abs().square()
    unabsorbed = waves(reference).abs().square()
    for i, angle in enumerate(ANGLES):
        path = 160 / math.cos(math.radians(angle))
        ratio = clean[i, 8:16, 8:16].mean() / unabsorbed[i, 8:16, 8:16].mean()
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
    assert torch.equal(model.absorption, expected)
    # Aligned: the most absorbing voxel is inside the specimen.
    specimen = dry > 0
    assert specimen.flatten()[model.absorption.argmax()]
    solvent = absorption_potential(ice_inelastic_mfp(300.0), 300.0)
    assert model.absorption[~specimen].min() > 0.9 * solvent


@pytest.mark.parametrize("aperture", [None, 12.0])
@pytest.mark.parametrize("specimen_mfp", [None, INELASTIC_MFP_PROTEIN_A])
def test_built_field_matches_the_explicit_override(specimen_mfp, aperture):
    """The field built from the settings is what a user would pass by hand."""
    propagation = Propagation(
        absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
    )
    kwargs = dict(
        angles=ANGLES,
        propagation=propagation,
        ice=Ice(model="random"),
        tilt=TiltGeometry(taper_width=3, z_taper_width=2),
        objective_aperture=aperture,
    )
    built = seeded(blob(), **kwargs)
    mfps = stages.resolve_removal_mfps(
        propagation, Optics(objective_aperture=aperture), 300.0
    )
    field = inelastic_absorption_potential(
        blob(),
        5.0,
        300.0,
        mfp_solvent_A=mfps.removal("solvent"),
        mfp_specimen_A=None if specimen_mfp is None else mfps.removal("specimen"),
    )
    explicit = seeded(blob(), field=field, **kwargs)
    assert torch.equal(built.absorption, explicit.absorption)
    assert torch.equal(waves(built), waves(explicit))


def test_aperture_removal_matches_the_particle_path(monkeypatch):
    """An objective aperture charges the tilt series the particle path's loss."""
    resolved = []
    resolve = stages.resolve_removal_mfps
    monkeypatch.setattr(
        stages,
        "resolve_removal_mfps",
        lambda *a: resolved.append(resolve(*a)) or resolved[-1],
    )
    propagation = dict(
        absorption_model="inelastic_mfp", inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A
    )
    make(
        blob(),
        angles=ANGLES,
        objective_aperture=12.0,
        propagation=Propagation(**propagation),
    )
    poses = Poses(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 2), torch.arange(1)
    )
    with torch.no_grad():
        Specimen(torch.zeros(16, 16, 16), 5.0).solvate(model="random").scatter(
            poses, voltage=300.0, objective_aperture=12.0, **propagation
        )
    tilt, particle = resolved
    for material in ("solvent", "specimen"):
        assert tilt.removal(material) == particle.removal(material)
    # Both loss channels are counted: shorter than the inelastic path alone.
    assert tilt.removal("solvent") < ice_inelastic_mfp(300.0)
    assert tilt.objective_aperture == 12.0


def test_aperture_requires_the_mfp_model():
    with pytest.raises(ValueError, match="objective_aperture"):
        make(blob(), objective_aperture=12.0)


def test_aperture_lowpasses_the_elastic_volume_isotropically():
    """
    Below 0.82 A/px a 12 mrad aperture lies inside the grid: the elastic
    volume keeps nothing beyond it in any direction, so no tilt sees any.
    """
    torch.manual_seed(0)
    volume = torch.rand(1, 16, 64, 64)
    model = make(volume, pixel_size=0.5, objective_aperture=12.0, propagation=MFP)
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
        assert model.absorption is None
        return
    assert model.absorption.min() == 0
    assert model.absorption.max() > 0
    empty = make(torch.zeros(1, 16, 64, 64), angles=ANGLES, propagation=propagation)
    assert torch.count_nonzero(empty.absorption) == 0
    clean = waves(empty).abs().square()
    # Exactly 1 in exact arithmetic; the padded canvas's float32 FFTs leave
    # ~1e-5 over a 60 degree tilt's slices, which no absorption could be.
    torch.testing.assert_close(clean, torch.ones_like(clean), rtol=0, atol=1e-4)
