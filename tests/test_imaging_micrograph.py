"""
The micrograph objects (`specter.imaging.Field`/`Sample`) against the
generator they replaced.

The references under ``tests/test_data/imaging_micrograph_*.pt`` are the
retired `MicrographGenerator`'s outputs on these exact inputs and seeds,
written once while both existed. Exact without ice. With ice, to float
tolerance and without noise: the references were written while the CPU ice
splat still accumulated in thread-scheduling order, so they carry ~1e-5 V of
that noise, which Poisson counts would turn into different integers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.ice import IceProfile, RandomIcemaker
from specter.imaging import CTF, Exposure, Field, Sample
from specter.settings import Crowding, Envelopes, Ice, Optics, Propagation

REFERENCES = Path(__file__).parent / "test_data"

NXY = 48
PIXEL = 2.0
VOLTAGE = 300.0
DOSE = 30.0
N = 2


def _template() -> torch.Tensor:
    V = torch.zeros(12, 12, 12)
    V[4:8, 3:9, 4:8] = 5.0
    return V


def _ctf() -> CTF:
    return CTF.from_values(torch.tensor([9000.0, 13000.0]), cs=2.7)


def _retired_ctf(ctf: CTF, waves: object, profile: IceProfile | None) -> CTF:
    """
    `ctf` with the defocus the retired generator would have imaged at.

    It referenced a profile's defocus to the ice's mean top surface rather
    than to its scattering centroid, the plane a nominal defocus now refers
    to. Moving the defocus by the difference between the two shifts makes the
    comparison one of everything else, which is unchanged.

    It also propagated with the conjugate Fresnel propagator, under which an
    exit-face defocus ``A`` images each slice where the physical propagator's
    ``A + T + dz`` does (``T`` the column's depth, ``dz`` one slice), so the
    retired exit-face defocus ``dfu - z_top`` is moved by that much. ``T``
    comes from the wave's own shift, ``-T / 2`` plus the ice centroid's
    height.
    """
    if profile is None:
        return ctf
    _, z_top = profile.surfaces(NXY, PIXEL)
    centroid = profile.centroid(NXY, PIXEL)
    depth = 2.0 * (centroid - waves.defocus_shift)
    offset = waves.defocus_shift + depth + PIXEL - float(z_top.mean())
    params = dict(ctf.params)
    for name in ("dfu", "dfv"):
        params[name] = params[name] + offset
    return CTF(params, ctf.ids)


def _reference(name: str) -> object:
    """The retired generator's output for this case."""
    return torch.load(REFERENCES / f"imaging_micrograph_{name}.pt", weights_only=True)


def _compare(
    reference: str,
    *,
    ice: Ice = Ice(),
    crowding: Crowding = Crowding(min_distance=18.0),
    propagation: Propagation = Propagation(),
    optics: Optics = Optics(),
    envelopes: Envelopes = Envelopes(),
    noise: str | None = None,
    keep_dry: bool = False,
) -> None:
    icemaker = RandomIcemaker(PIXEL, NXY, 30) if ice.model == "random" else None
    field = Field(_template(), PIXEL, NXY, nz=30, crowding=crowding)
    if ice.model is not None:
        field = field.solvate(ice, icemaker=icemaker)
    else:
        field = Field(_template(), PIXEL, NXY, nz=30, crowding=crowding, ice=ice)
    if envelopes.dose_envelope and envelopes.dose_envelope_target == "specimen":
        field = field.damage()
    beam = dict(asdict_propagation(propagation))
    got, got_clean = [], []
    with torch.no_grad():
        for i in range(N):
            exposure = Exposure(DOSE)
            sample = field.draw(
                voltage=VOLTAGE,
                exposure=exposure,
                seed=7,
                index=i,
                keep_dry=keep_dry,
                objective_aperture=optics.objective_aperture,
                **beam,
            )
            ids = torch.tensor([i])
            waves = sample.scatter(voltage=VOLTAGE, ids=ids, **beam)
            if keep_dry:
                got_clean.append(
                    sample.clean().scatter(voltage=VOLTAGE, ids=ids, **beam).pixels()
                )
            ctf = _retired_ctf(_ctf()[i : i + 1], waves, ice.profile)
            image = waves.apply_ctf(ctf).intensity(
                temporal_coherence=envelopes.temporal_coherence
            )
            got.append(image.detect(exposure, noise=noise, seed=7).array)
    want = _reference(reference)
    assert isinstance(want, dict)
    if ice.model is None:
        assert torch.equal(torch.cat(got), want["images"])
    else:
        assert noise is None
        torch.testing.assert_close(torch.cat(got), want["images"], rtol=1e-5, atol=0)
    if keep_dry:
        assert torch.equal(torch.cat(got_clean), want["clean"])


def asdict_propagation(p: Propagation) -> dict:
    from dataclasses import asdict

    return asdict(p)


def test_crowded_solvated_noisy_micrographs_match_the_generator() -> None:
    _compare("ice_clean", ice=Ice(model="random"), keep_dry=True)


def test_an_ice_free_noisy_field_matches_the_generator_exactly() -> None:
    _compare("ice_free_noisy", noise="poisson")


@pytest.mark.parametrize("specimen_mfp", [None, 2460.0])
def test_material_absorption_matches_the_generator(specimen_mfp: float | None) -> None:
    """Uniform solvent absorption, and the field with a specimen mean free
    path, the latter through an objective aperture."""
    _compare(
        "absorb_field" if specimen_mfp else "absorb_uniform",
        ice=Ice(model="random"),
        propagation=Propagation(
            absorption_model="inelastic_mfp", inelastic_mfp_specimen=specimen_mfp
        ),
        optics=Optics(objective_aperture=12.0 if specimen_mfp else None),
    )


def test_an_ice_profile_matches_the_generator() -> None:
    profile = IceProfile(mode="wedge", thickness_range=(20.0, 50.0), angle=30.0)
    _compare(
        "profile",
        ice=Ice(model="random", profile=profile),
        # Unpadded, as the retired generator ran it; a wedge now pads by default.
        propagation=Propagation(absorption_model="inelastic_mfp", pad_fft=False),
    )


def test_specimen_damage_and_solvent_decorrelation_match_the_generator() -> None:
    _compare(
        "damage_motion",
        ice=Ice(model="random", motion_variance=0.3),
        envelopes=Envelopes(dose_envelope=True, dose_envelope_target="specimen"),
    )


def test_a_sample_refuses_another_dose() -> None:
    field = Field(_template(), PIXEL, NXY, nz=30).damage()
    sample = field.draw(voltage=VOLTAGE, exposure=Exposure(DOSE))
    with pytest.raises(ValueError, match="drawn for a dose"):
        sample.scatter(voltage=VOLTAGE, exposure=Exposure(DOSE + 1))


def _prebuilt() -> torch.Tensor:
    V = torch.zeros(1, 30, NXY, NXY)
    V[:, 10:20, 16:32, 18:30] = 5.0
    return V


@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "ice",
        "damage",
        "damage_ice",
        "absorb_uniform",
        "absorb_field",
        "profile",
    ],
)
def test_a_prepared_volume_matches_the_generator(case: str) -> None:
    """A pre-assembled volume, imaged untilted at two doses."""
    ice = Ice()
    envelopes = Envelopes()
    propagation = Propagation()
    optics = Optics()
    if case in ("ice", "damage_ice", "absorb_uniform", "absorb_field"):
        ice = Ice(model="random")
    if case == "profile":
        ice = Ice(
            model="random",
            profile=IceProfile(mode="wedge", thickness_range=(20.0, 50.0)),
        )
    if case.startswith("damage"):
        envelopes = Envelopes(dose_envelope=True, dose_envelope_target="specimen")
    if case.startswith("absorb") or case == "profile":
        propagation = Propagation(
            absorption_model="inelastic_mfp",
            inelastic_mfp_specimen=2460.0 if case == "absorb_field" else None,
            pad_fft=False,  # as the retired generator ran it
        )
    if case == "absorb_field":
        optics = Optics(objective_aperture=12.0)
    doses = torch.tensor([20.0, 40.0])
    icemaker = RandomIcemaker(PIXEL, NXY, 30) if ice.model is not None else None
    want = _reference(f"prepared_{case}")
    assert isinstance(want, torch.Tensor)
    torch.manual_seed(5)
    sample = Sample.from_volume(_prebuilt(), PIXEL).prepare(
        ice,
        voltage=VOLTAGE,
        icemaker=icemaker,
        damaged=envelopes.dose_envelope,
        objective_aperture=optics.objective_aperture,
        **asdict_propagation(propagation),
    )
    with torch.no_grad():
        for i in range(2):
            exposure = Exposure(doses[i : i + 1])
            waves = sample.scatter(
                voltage=VOLTAGE,
                ids=torch.tensor([i]),
                exposure=exposure,
                **asdict_propagation(propagation),
            )
            got = (
                waves.apply_ctf(_retired_ctf(_ctf()[i : i + 1], waves, ice.profile))
                .intensity()
                .detect(exposure, noise=None)
            )
            torch.testing.assert_close(got.array, want[i : i + 1], rtol=1e-5, atol=0)
