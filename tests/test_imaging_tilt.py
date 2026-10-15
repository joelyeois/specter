"""
The tilt-series objects (`Sample.tilt_series` -> `TiltSeries`) against the
generator they replaced.

The references under ``tests/test_data/imaging_tilt_*.pt`` were the retired
`TiltSeriesGenerator`'s images and exit waves on these exact inputs and
seeds, written while both existed. The multislice cases were re-recorded on
2026-10-15, when a tilt began keeping its FFT margin through the transfer
function rather than wrapping the field's edges into each other; a missing
reference is written on first run (delete one to re-record it).

Exact without ice. With ice, to float tolerance and without noise: the
references were written while the CPU ice splat still accumulated in
thread-scheduling order, so they carry ~1e-5 V of that noise, which Poisson
counts would turn into different integers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.ice import RandomIcemaker
from specter.imaging import CTF, Exposure, Poses, Sample
from specter.settings import Envelopes, Ice, Optics, Propagation

REFERENCES = Path(__file__).parent / "test_data"

NZ, NXY, MICROGRAPH = 12, 32, 32
PIXEL = 3.0
VOLTAGE = 300.0
ANGLES = [-40.0, -20.0, 0.0, 20.0, 40.0]
N = len(ANGLES)


def _volume() -> torch.Tensor:
    torch.manual_seed(0)
    V = torch.zeros(1, NZ, NXY, NXY)
    V[:, 4:8, 10:22, 12:20] = 4.0
    V[:, 5:7, 14:18, 6:26] = 7.0
    return V


def _ctf() -> CTF:
    return CTF.from_values(torch.linspace(20000.0, 24000.0, N), cs=2.7)


def _compare(
    reference: str,
    *,
    ice: Ice = Ice(),
    dose: float | torch.Tensor = 3.0,
    propagation: Propagation = Propagation(pad_fft=True),
    optics: Optics = Optics(),
    envelopes: Envelopes = Envelopes(),
    noise: str | None = None,
    absorption: torch.Tensor | None = None,
    exact: bool = True,
) -> None:
    icemaker = RandomIcemaker(PIXEL, NXY, NZ) if ice.model == "random" else None
    path = REFERENCES / f"imaging_tilt_{reference}.pt"
    exposure = Exposure.tilt_series(dose, N)
    on_specimen = envelopes.dose_envelope_target == "specimen"
    series = Sample.from_volume(_volume(), PIXEL).tilt_series(
        Poses.tilt_series(ANGLES),
        exposure,
        voltage=VOLTAGE,
        micrograph_size=MICROGRAPH,
        ice=ice,
        icemaker=icemaker,
        damaged=envelopes.dose_envelope and on_specimen,
        absorption=absorption,
        seed=7,
        objective_aperture=optics.objective_aperture,
        propagation=propagation,
        log=False,
    )
    got, waves_got = [], []
    with torch.no_grad():
        for i in range(N):
            waves = series.scatter(i)
            waves_got.append(waves.pixels())
            one = slice(i, i + 1)
            image = (
                waves.apply_ctf(
                    _ctf()[one],
                    dose_envelope=(
                        exposure[one]
                        if envelopes.dose_envelope and not on_specimen
                        else None
                    ),
                )
                .intensity(temporal_coherence=envelopes.temporal_coherence)
                .detect(exposure[one], noise=noise, seed=7)
            )
            got.append(image.array)
    got_t, waves_t = torch.cat(got), torch.stack(waves_got, dim=1)[0]
    if not path.exists():
        torch.save({"images": got_t, "waves": waves_t}, path)
        return
    want = torch.load(path, weights_only=True)
    if exact:
        assert torch.equal(waves_t, want["waves"])
        assert torch.equal(got_t, want["images"])
    else:
        assert noise is None
        torch.testing.assert_close(got_t, want["images"], rtol=1e-5, atol=0)


def test_an_ice_free_noisy_series_matches_the_generator_exactly() -> None:
    _compare("ice_free_noisy", noise="poisson")


def test_the_dose_envelope_on_the_transfer_function_matches() -> None:
    _compare("tf_dose_envelope", envelopes=Envelopes(dose_envelope=True))


@pytest.mark.parametrize("model", ["projection", "ctf"])
def test_the_depthless_models_match(model: str) -> None:
    """The linear model's Cc envelope acts on the real contrast rather than
    inside the transfer function: the same product, to float precision."""
    _compare(
        f"model_{model}",
        propagation=Propagation(scattering_model=model, alpha=0.1),
        exact=model != "ctf",
    )


def test_a_solvated_series_matches_the_generator() -> None:
    _compare("solvated", ice=Ice(model="random"), exact=False)


def test_per_tilt_damage_and_solvent_decorrelation_match() -> None:
    """Unequal doses: the ice is re-rendered per dose, the specimen damaged
    per tilt after the tilts before it."""
    _compare(
        "damage_motion",
        ice=Ice(model="random", motion_variance=0.3),
        dose=torch.tensor([3.0, 2.0, 3.0, 4.0, 3.0]),
        envelopes=Envelopes(dose_envelope=True, dose_envelope_target="specimen"),
        exact=False,
    )


def test_material_absorption_and_aperture_match() -> None:
    _compare(
        "absorb",
        ice=Ice(model="random"),
        propagation=Propagation(
            absorption_model="inelastic_mfp",
            inelastic_mfp_specimen=2460.0,
            pad_fft=False,  # the retired generator's default
        ),
        optics=Optics(objective_aperture=12.0),
        exact=False,
    )


def test_an_explicit_absorption_field_matches() -> None:
    absorption = _volume().clamp(min=0) * 0.05
    _compare("explicit_absorption", absorption=absorption)
