"""Mean-free-path absorption and the objective aperture in micrographs."""

import math

import pytest
import torch

import specter
from specter.cpu_threads import limited_cpu_threads
from specter.ice import IceProfile
from specter.imagegenerator import MicrographGenerator
from specter.potential import (
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    ice_inelastic_mfp,
    inelastic_absorption_potential,
)
from specter.settings import Camera, Crowding, Ice, Optics, Propagation
from specter.specimen import MicrographSpecimenGenerator

MFP = Propagation(absorption_model="inelastic_mfp")
PROTEIN = Propagation(
    absorption_model="inelastic_mfp", inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A
)
DX = 5.0
NXY = 32


def ctf(n=1):
    return {
        "dfu": torch.full((n,), 1.0e4),
        "dfv": torch.full((n,), 1.0e4),
        "dfang": torch.zeros(n),
        "cs": torch.full((n,), 2.7e7),
    }


def template():
    v = torch.zeros(12, 12, 12)
    v[3:9, 3:9, 3:9] = 9.0
    return v


def specimen(ice=Ice(model="random"), nz=20, crowd=True, **kwargs):
    return MicrographSpecimenGenerator(
        template() if crowd else None,
        DX,
        NXY,
        nz=nz,
        crowding=Crowding(min_distance=40.0, n_points=3) if crowd else Crowding(),
        ice=ice,
        progressbars=False,
        **kwargs,
    )


def imager(spec, propagation=MFP, optics=None, **kwargs):
    return MicrographGenerator(
        spec,
        NXY,
        DX,
        ctf() if optics is not None else None,
        300.0,
        20.0,
        propagation=propagation,
        optics=optics,
        camera=Camera(noise_model=None, detector_model=None),
        verbose=False,
        progressbars=False,
        **kwargs,
    )


def run(model, seed=0):
    """One micrograph under a fixed seed, on one thread (the splat reorders)."""
    specter.seed(seed)
    with torch.no_grad(), limited_cpu_threads(1):
        model(torch.tensor([0]))
    return model.exitwaves


@pytest.mark.parametrize("propagation", [MFP, PROTEIN], ids=["uniform", "field"])
def test_pure_ice_micrograph_obeys_beer_lambert(propagation):
    """
    A slab of ice t thick transmits exp(-t / Lambda_ice) relative to the same
    ice unabsorbed: through the scalar factor when the field is uniform, and
    through the field when the specimen has its own mean free path (a pure-ice
    specimen occupies nothing, so the field is the ice's everywhere).
    """
    nz = 40
    absorbed = imager(specimen(nz=nz, crowd=False), propagation)
    reference = imager(specimen(nz=nz, crowd=False), Propagation())
    ratio = run(absorbed).abs().square().mean() / run(reference).abs().square().mean()
    assert (absorbed.absorption_potential is None) == (propagation is MFP)
    expected = math.exp(-nz * DX / ice_inelastic_mfp(300.0))
    # Measured: 6e-8 relative through the scalar, 7e-7 through the field.
    assert ratio.item() == pytest.approx(expected, rel=1e-5)


def test_ice_profile_absorbs_only_where_the_ice_is():
    """
    An ice profile leaves vacuum in the box, so the uniform case needs a
    field: each column transmits exp(-integral of the window / Lambda_ice).
    """
    nz = 40
    profile = IceProfile(mode="wedge", thickness_range=(60.0, 180.0), softness=4.0)
    absorbed = imager(specimen(Ice(model="random", profile=profile), nz, False))
    reference = imager(
        specimen(Ice(model="random", profile=profile), nz, False), Propagation()
    )
    wave, unabsorbed = run(absorbed), run(reference)
    window = profile.window(nz, NXY, DX)
    field = absorbed.absorption_potential[0]
    solvent = absorption_potential(ice_inelastic_mfp(300.0), 300.0)
    torch.testing.assert_close(field, solvent * window, rtol=1e-6, atol=1e-7)
    # Column by column, averaged along the ramp's constant-thickness lines.
    path = window.sum(0).mean(0) * DX
    ratio = wave.abs().square()[0].mean(0) / unabsorbed.abs().square()[0].mean(0)
    torch.testing.assert_close(
        ratio, torch.exp(-path / ice_inelastic_mfp(300.0)), rtol=1e-3, atol=0.0
    )


def test_specimen_field_is_inelastic_absorption_of_the_dry_specimen():
    """
    With ice, the field the blend writes equals inelastic_absorption_potential
    of the assembled DRY specimen, with the occupancy reference the ice uses.
    """
    spec = specimen(save_clean_exitwaves=True, molecular_mass=40_000.0)
    model = imager(spec, PROTEIN)
    run(model)
    dry = spec.clean_V
    assert dry.abs().max() > 0
    mfps = model._removal_mfps
    expected = inelastic_absorption_potential(
        dry,
        DX,
        300.0,
        mfp_solvent_A=mfps.removal("solvent"),
        mfp_specimen_A=mfps.removal("specimen"),
        full_potential=spec._occupancy_reference(),
    )
    torch.testing.assert_close(
        model.absorption_potential, expected, rtol=1e-5, atol=1e-6
    )
    # The specimen absorbs more than the ice it displaces.
    assert float(model.absorption_potential.max()) > float(expected.min()) * 1.2


def test_no_ice_leaves_only_the_specimen_absorbing():
    """In vacuum the uniform case absorbs nothing and the field is occ * V_spec."""
    uniform = imager(specimen(Ice()), MFP)
    reference = imager(specimen(Ice()), Propagation())
    assert uniform._uniform_absorption_V == 0.0
    torch.testing.assert_close(run(uniform), run(reference), rtol=0.0, atol=0.0)

    spec = specimen(Ice())
    model = imager(spec, PROTEIN)
    run(model)
    expected = inelastic_absorption_potential(
        model.volume,
        DX,
        300.0,
        mfp_solvent_A=float("inf"),
        mfp_specimen_A=model._removal_mfps.removal("specimen"),
        full_potential=spec._occupancy_reference(),
    )
    torch.testing.assert_close(model.absorption_potential, expected)
    assert float(model.absorption_potential.min()) == 0.0


@pytest.mark.parametrize("pad_fft", [False, True])
def test_uniform_scalar_equals_the_constant_field(pad_fft):
    """
    The scalar factor is exactly what propagating a constant field gives, in
    the reflect-padded margin as well as the box.
    """
    propagation = Propagation(absorption_model="inelastic_mfp", pad_fft=pad_fft)
    scalar = imager(specimen(), propagation)
    wave = run(scalar)
    assert scalar.absorption_potential is None
    field = imager(specimen(), propagation)
    run(field)  # the same specimen, from the same seed
    assert torch.equal(field.volume, scalar.volume)
    field.absorption_potential = torch.full_like(
        field.volume, scalar._uniform_absorption_V
    )
    field._uniform_absorption_V = 0.0
    with torch.no_grad(), limited_cpu_threads(1):
        field(torch.tensor([0]))
    torch.testing.assert_close(field.exitwaves, wave, rtol=1e-5, atol=1e-6)


def test_volume_specimen_builds_its_field_at_construction():
    """A pre-assembled volume with ice gets its field from the same blend."""
    torch.manual_seed(3)
    volume = torch.zeros(1, 20, NXY, NXY)
    volume[:, 6:14, 10:20, 10:20] = 8.0
    specter.seed(1)
    model = MicrographGenerator(
        volume.clone(),
        NXY,
        DX,
        None,
        300.0,
        20.0,
        propagation=PROTEIN,
        optics=None,
        ice=Ice(model="random"),
        camera=Camera(noise_model=None),
        verbose=False,
        progressbars=False,
    )
    mfps = model._removal_mfps
    expected = inelastic_absorption_potential(
        volume,
        DX,
        300.0,
        mfp_solvent_A=mfps.removal("solvent"),
        mfp_specimen_A=mfps.removal("specimen"),
    )
    torch.testing.assert_close(
        model.absorption_potential, expected, rtol=1e-5, atol=1e-6
    )


def test_aperture_lowpasses_the_specimen_and_absorbs_more():
    """
    At a voxel fine enough to carry scattering beyond the aperture, the
    specimen is low-passed per slice, and the aperture's loss is charged.
    """
    dx = 0.5
    optics = Optics(objective_aperture=12.0)
    spec = MicrographSpecimenGenerator(
        None, dx, 32, nz=8, ice=Ice(model="random"), progressbars=False
    )
    model = MicrographGenerator(
        spec,
        32,
        dx,
        ctf(),
        300.0,
        20.0,
        propagation=MFP,
        optics=optics,
        camera=Camera(noise_model=None),
        verbose=False,
        progressbars=False,
    )
    run(model)
    k = torch.fft.fftfreq(32, d=dx)
    kr = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    power = torch.fft.fft2(model.volume[0]).abs().square().sum(0)
    from specter.potential._absorption import _aperture_k

    beyond = kr > _aperture_k(12.0, 300.0) * 1.01
    assert power[beyond].max() < 1e-8 * power.max()
    without = imager(specimen(), MFP)._uniform_absorption_V
    assert model._uniform_absorption_V > without


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_slabwise_host_blend_writes_the_same_field(monkeypatch):
    """
    A host canvas blended slab by slab on the bank's device writes the field
    from the same occupancy read, matching the dry specimen's.
    """
    import specter.ice._blend as blend
    from specter.specimen._single_particle import AbsorptionRates

    calls = []
    slabwise = blend._blend_ice_slabwise
    monkeypatch.setattr(
        blend,
        "_blend_ice_slabwise",
        lambda *a, **k: calls.append(1) or slabwise(*a, **k),
    )

    spec = MicrographSpecimenGenerator(
        template(),
        DX,
        48,
        nz=24,
        crowding=Crowding(min_distance=40.0, n_points=3),
        ice=Ice(model="gd"),
        save_clean_exitwaves=True,
        move_to_cpu=True,
        progressbars=False,
    ).cuda()
    rates = AbsorptionRates(specimen=0.3, solvent=0.2)
    specter.seed(0)
    assembled = spec.assemble(absorption=rates)
    assert calls, "the host-canvas branch was not taken"
    assert assembled.volume.device.type == "cpu"
    assert assembled.absorption.device.type == "cpu"
    dry = spec.clean_V
    occupancy = (
        inelastic_absorption_potential(
            dry,
            DX,
            300.0,
            mfp_solvent_A=1.0,
            mfp_specimen_A=2.0,
            full_potential=spec._occupancy_reference(),
        )
        - absorption_potential(1.0, 300.0)
    ) / (absorption_potential(2.0, 300.0) - absorption_potential(1.0, 300.0))
    expected = occupancy * 0.3 + (1 - occupancy) * 0.2
    torch.testing.assert_close(assembled.absorption, expected, rtol=1e-5, atol=1e-6)
