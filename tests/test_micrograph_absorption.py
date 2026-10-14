"""Mean-free-path absorption and the objective aperture in micrographs."""

import math
from dataclasses import replace

import pytest
import torch

import specter
from specter.cpu_threads import limited_cpu_threads
from specter.ice import IceProfile
from specter.imaging import Field, Sample
from specter.imaging._stages import resolve_removal_mfps
from specter.potential import (
    INELASTIC_MFP_PROTEIN_A,
    absorption_potential,
    ice_inelastic_mfp,
    inelastic_absorption_potential,
    template_occupancy_reference,
)
from specter.settings import Crowding, Ice, Propagation
from specter.specimen import MicrographSpecimenGenerator

MFP = dict(absorption_model="inelastic_mfp")
PROTEIN = dict(
    absorption_model="inelastic_mfp", inelastic_mfp_specimen=INELASTIC_MFP_PROTEIN_A
)
DX = 5.0
NXY = 32


def template():
    v = torch.zeros(12, 12, 12)
    v[3:9, 3:9, 3:9] = 9.0
    return v


def field(ice=Ice(model="random"), nz=20, crowd=True, **kwargs):
    f = Field(template() if crowd else None, DX, NXY, nz=nz, **kwargs)
    if crowd:
        f = f.pack(Crowding(min_distance=40.0, n_points=3))
    return f.solvate(ice) if ice.model is not None else f


def draw(f, seed=0, **settings):
    """One micrograph under a fixed seed, on one thread (the splat reorders)."""
    with torch.no_grad(), limited_cpu_threads(1):
        return f.draw(voltage=300.0, seed=seed, **settings)


def scatter(sample, **propagation):
    with torch.no_grad():
        return sample.scatter(voltage=300.0, **propagation).pixels()


def removal(propagation):
    return resolve_removal_mfps(Propagation(**propagation), None, 300.0)


@pytest.mark.parametrize("propagation", [MFP, PROTEIN], ids=["uniform", "field"])
def test_pure_ice_micrograph_obeys_beer_lambert(propagation):
    """
    A slab of ice t thick transmits exp(-t / Lambda_ice) relative to the same
    ice unabsorbed: through the scalar factor when the field is uniform, and
    through the field when the specimen has its own mean free path (a pure-ice
    specimen occupies nothing, so the field is the ice's everywhere).
    """
    nz = 40
    absorbed = draw(field(nz=nz, crowd=False), **propagation)
    reference = draw(field(nz=nz, crowd=False))
    ratio = (
        scatter(absorbed).abs().square().mean()
        / scatter(reference).abs().square().mean()
    )
    assert (absorbed.absorption is None) == (propagation is MFP)
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
    absorbed = draw(field(Ice(model="random", profile=profile), nz, False), **MFP)
    reference = draw(field(Ice(model="random", profile=profile), nz, False))
    # On the field's own canvas, so the columns line up with the window; a
    # wedge would otherwise pad by default.
    wave = scatter(absorbed, pad_fft=False)
    unabsorbed = scatter(reference, pad_fft=False)
    window = profile.window(nz, NXY, DX)
    solvent = absorption_potential(ice_inelastic_mfp(300.0), 300.0)
    torch.testing.assert_close(
        absorbed.absorption[0], solvent * window, rtol=1e-6, atol=1e-7
    )
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
    sample = draw(field(molecular_mass=40_000.0), keep_dry=True, **PROTEIN)
    dry = sample.dry
    assert dry.abs().max() > 0
    mfps = removal(PROTEIN)
    expected = inelastic_absorption_potential(
        dry,
        DX,
        300.0,
        mfp_solvent_A=mfps.removal("solvent"),
        mfp_specimen_A=mfps.removal("specimen"),
        full_potential=template_occupancy_reference(template(), DX, 40_000.0),
    )
    torch.testing.assert_close(sample.absorption, expected, rtol=1e-5, atol=1e-6)
    # The specimen absorbs more than the ice it displaces.
    assert float(sample.absorption.max()) > float(expected.min()) * 1.2


def test_no_ice_leaves_only_the_specimen_absorbing():
    """In vacuum the uniform case absorbs nothing and the field is occ * V_spec."""
    uniform = draw(field(Ice()), **MFP)
    reference = draw(field(Ice()))
    assert uniform.uniform_absorption == 0.0 and uniform.absorption is None
    torch.testing.assert_close(scatter(uniform), scatter(reference), rtol=0.0, atol=0.0)

    sample = draw(field(Ice()), **PROTEIN)
    expected = inelastic_absorption_potential(
        sample.volume,
        DX,
        300.0,
        mfp_solvent_A=float("inf"),
        mfp_specimen_A=removal(PROTEIN).removal("specimen"),
        full_potential=template_occupancy_reference(template(), DX, None),
    )
    torch.testing.assert_close(sample.absorption, expected)
    assert float(sample.absorption.min()) == 0.0


@pytest.mark.parametrize("pad_fft", [False, True])
def test_uniform_scalar_equals_the_constant_field(pad_fft):
    """
    The scalar factor is exactly what propagating a constant field gives, in
    the reflect-padded margin as well as the box.
    """
    scalar = draw(field(), **MFP)
    assert scalar.absorption is None and scalar.uniform_absorption > 0.0
    constant = replace(
        scalar,
        absorption=torch.full_like(scalar.volume, scalar.uniform_absorption),
        uniform_absorption=0.0,
    )
    torch.testing.assert_close(
        scatter(constant, pad_fft=pad_fft),
        scatter(scalar, pad_fft=pad_fft),
        rtol=1e-5,
        atol=1e-6,
    )


def test_volume_specimen_builds_its_field_when_prepared():
    """A pre-assembled volume with ice gets its field from the same blend."""
    torch.manual_seed(3)
    volume = torch.zeros(1, 20, NXY, NXY)
    volume[:, 6:14, 10:20, 10:20] = 8.0
    with torch.no_grad(), limited_cpu_threads(1):
        sample = Sample.from_volume(volume.clone(), DX).prepare(
            Ice(model="random"), voltage=300.0, **PROTEIN
        )
    mfps = removal(PROTEIN)
    expected = inelastic_absorption_potential(
        volume,
        DX,
        300.0,
        mfp_solvent_A=mfps.removal("solvent"),
        mfp_specimen_A=mfps.removal("specimen"),
    )
    torch.testing.assert_close(sample.absorption, expected, rtol=1e-5, atol=1e-6)


def test_aperture_lowpasses_the_specimen_and_absorbs_more():
    """
    At a voxel fine enough to carry scattering beyond the aperture, the
    specimen is low-passed per slice, and the aperture's loss is charged.
    """
    dx = 0.5
    ice_only = Field(None, dx, 32, nz=8).solvate(model="random")
    sample = draw(ice_only, objective_aperture=12.0, **MFP)
    k = torch.fft.fftfreq(32, d=dx)
    kr = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    power = torch.fft.fft2(sample.volume[0]).abs().square().sum(0)
    from specter.potential._absorption import _aperture_k

    beyond = kr > _aperture_k(12.0, 300.0) * 1.01
    assert power[beyond].max() < 1e-8 * power.max()
    without = draw(ice_only, **MFP).uniform_absorption
    assert sample.uniform_absorption > without


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
