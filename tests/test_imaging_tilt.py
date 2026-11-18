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
The cases with ice were re-recorded on 2026-11-07, when `RandomIcemaker`
began convolving its waters periodically; they pin the objects since.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import roma
import torch

from specter.ice import RandomIcemaker
from specter.imaging import CTF, Exposure, Poses, Sample
from specter.imaging._tilt import fit_volume_to_tilt
from specter.settings import Envelopes, Ice, Optics, Propagation
from specter.tilt import (
    estimate_required_extent,
    estimate_required_nxy,
    volume_shape_for_tilt,
)

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


def _assert_rounding(got: torch.Tensor, want: torch.Tensor, noise: str | None) -> None:
    """
    Assert a stored reference is reproduced to float rounding.

    The reference was rendered with the full-spectrum resampler; the
    half-spectrum one moves an exit wave by about 2e-6, which in a
    Poisson-sampled image crosses a count boundary on a few pixels.
    """
    if noise is None or got.is_complex():
        scale = float(want.abs().max())
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5 * scale)
        return
    moved = got != want
    assert moved.float().mean() < 2e-3
    assert (got - want).abs().max() <= 1


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
        _assert_rounding(waves_t, want["waves"], None)
        _assert_rounding(got_t, want["images"], noise)
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


@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_tilt_range_widens_only_the_axis_across_the_tilt_axis(axis: str) -> None:
    nz, frame, tilt = 16, 32, 60.0
    quaternions = Poses.tilt_series([-tilt, 0.0, tilt], axis=axis).quaternions
    across = estimate_required_nxy(frame, nz, tilt)
    ny, nx = estimate_required_extent(frame, nz, roma.unitquat_to_rotmat(quaternions))
    assert (ny, nx) == ((across, frame) if axis == "x" else (frame, across))

    # Along the tilt axis the volume is filled to the canvas and its margins,
    # across it to the full coverage; an axis already that wide is left alone.
    volume = torch.rand(1, nz, 40, 40)
    fitted, coverage = fit_volume_to_tilt(
        volume, frame, quaternions, 8, 0, 0, True, log=False, canvas_nxy=48
    )
    wide, narrow = across + 16, 48 + 16
    assert fitted.shape[-2:] == ((wide, narrow) if axis == "x" else (narrow, wide))
    assert coverage.recommended_nxy_for_max_tilt == across
    refitted, _ = fit_volume_to_tilt(
        fitted, frame, quaternions, 8, 0, 0, True, log=False, canvas_nxy=48
    )
    assert refitted is fitted


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("field", [(16, 32, 32), (15, 33, 33)])
def test_a_volume_sized_for_a_tilt_range_needs_no_padding(
    axis: str, field: tuple[int, int, int]
) -> None:
    nz, frame, tilt = field[0], field[1], 60.0
    shape = volume_shape_for_tilt(field, tilt, axis)
    # The field of view is the centred crop: each axis grows by an even count.
    assert shape[0] == nz
    assert all((n - frame) % 2 == 0 and n > frame for n in shape[1:])

    quaternions = Poses.tilt_series([-tilt, 0.0, tilt], axis=axis).quaternions
    volume = torch.rand(1, *shape)
    fitted, _ = fit_volume_to_tilt(
        volume, frame, quaternions, 8, 0, 0, True, log=False, canvas_nxy=frame + 64
    )
    assert fitted is volume

    # One voxel narrower across the tilt axis and the range no longer fits.
    narrower = volume[..., 1:-1, :] if axis == "x" else volume[..., 1:-1]
    fitted, _ = fit_volume_to_tilt(
        narrower, frame, quaternions, 8, 0, 0, True, log=False, canvas_nxy=frame + 64
    )
    assert fitted.shape != narrower.shape


@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_volume_built_wide_across_the_tilt_axis_is_imaged_unpadded(
    axis: str,
) -> None:
    # Along the tilt axis the volume spans the padded canvas and its edge
    # margin. The reference holds the same material in a square box,
    # zero-filled beyond that, so the two series sample identical potential.
    nz, frame, wide, narrow = 12, 24, 120, 24 + 2 * 32 + 2 * 8
    torch.manual_seed(4)
    blobs = torch.nn.functional.avg_pool3d(
        torch.rand(1, 1, nz + 2, wide + 2, narrow + 2), 3, stride=1
    )[0]
    pad = (wide - narrow) // 2
    if axis == "y":
        blobs = blobs.transpose(-1, -2)
    square = torch.nn.functional.pad(
        blobs, (pad, pad, 0, 0) if axis == "x" else (0, 0, pad, pad)
    )

    def waves(volume: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        series = Sample.from_volume(4.0 * volume, PIXEL).tilt_series(
            Poses.tilt_series(ANGLES, axis=axis),
            Exposure.tilt_series(3.0, N),
            voltage=VOLTAGE,
            micrograph_size=frame,
            ice=Ice(model=None),
            log=False,
        )
        with torch.no_grad():
            array = torch.cat([series.scatter(i).array for i in range(N)])
        return array, tuple(series.volume.shape)

    got, shape = waves(blobs)
    want, _ = waves(square)
    assert shape == tuple(blobs.shape)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-4)
