"""
Golden-output regression tests for every imaging modality, and the settings
plumbing between the image-formation objects.

Outputs are compared against committed tensors in tests/test_data/. Missing
references fail and must be regenerated explicitly after reviewing an
intentional output change. The references were written by the generator
classes these objects replaced (`ImageGenerator`, `ImageGeneratorFromCoordinates`,
`MicrographGenerator`, `TiltSeriesGenerator`), so a match here is also the
proof that the objects reproduce them.
"""

from __future__ import annotations

import math

import pytest
import roma
import torch

from specter.aberrations import Aberration
from specter.arrays import center_crop
from specter.imaging import (
    CTF,
    ExitWave,
    Exposure,
    Field,
    Poses,
    Sample,
    Specimen,
    simulate_particles,
)
from specter.imaging import _stages as stages
from specter.scattering import IterativeScattering
from specter.settings import Crowding, Ice, Propagation, TiltGeometry

# These fixtures predate `coincidence_radius` being rescaled to mean a true
# effective exclusion radius (exclusion area pi*r^2, rather than the old
# grid-cell-indexing convention with area r^2/2). Expressing the old 1.8 via
# the exact conversion keeps the suppression grid bit-identical, so the
# fixtures still verify the physics rather than just the new parameterization.
_CR = 1.8 / math.sqrt(2 * math.pi)

VOLTAGE = 300.0
_IDENTITY = Poses(
    torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 2), torch.arange(1)
)


def test_missing_golden_reference_fails_without_writing(
    tmp_path, monkeypatch, save_or_compare
):
    import conftest

    monkeypatch.setattr(conftest, "FIXTURE_DIR", tmp_path / "references")
    with pytest.raises(AssertionError, match="Missing golden reference"):
        save_or_compare("missing", torch.ones(2, 2))
    assert not conftest.FIXTURE_DIR.exists()


@pytest.fixture
def small_coords():
    """Twenty carbon atoms near the origin."""
    torch.manual_seed(0)
    coords = torch.randn(20, 3) * 5.0
    atomic_numbers = torch.full((20,), 6, dtype=torch.long)
    return coords, atomic_numbers


@pytest.fixture
def ctf_params():
    return {
        "dfu": torch.tensor([5000.0]),
        "dfv": torch.tensor([5000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.0]),
        "tiltx": torch.tensor([0.0]),
        "tilty": torch.tensor([0.0]),
        "trefoil1": torch.tensor([0.0]),
        "trefoil2": torch.tensor([0.0]),
    }


def _tilt_volume() -> torch.Tensor:
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    return volume


def _tilt_series(
    angles: list[float],
    dose: float | torch.Tensor = 2.0,
    volume: torch.Tensor | None = None,
    micrograph_size: int = 32,
    **settings,
):
    """A tilt series of `_tilt_volume` about y, and its per-tilt exposure."""
    exposure = Exposure.tilt_series(dose, len(angles))
    series = Sample.from_volume(
        _tilt_volume() if volume is None else volume, 2.0
    ).tilt_series(
        Poses.tilt_series(angles, axis="y"),
        exposure,
        voltage=VOLTAGE,
        micrograph_size=micrograph_size,
        tilt=TiltGeometry(tilt_axis="y"),
        log=False,
        **settings,
    )
    return series, exposure


# ---------------------------------------------------------------------------
# Golden outputs
# ---------------------------------------------------------------------------


def test_particle_regression(small_volume, ctf_params, save_or_compare):
    """Particles: multislice scattering, random ice, coincidence loss, k3 detector."""
    torch.manual_seed(0)
    specimen = (
        Specimen(small_volume, 2.0).crowd(min_distance=60.0).solvate(model="random")
    )
    poses = Poses(
        torch.tensor([[0.7071, 0.0, 0.7071, 0.0]]),
        torch.tensor([[1.0, -1.0]]),
        torch.arange(1),
    )
    torch.manual_seed(0)
    images = simulate_particles(
        specimen,
        poses,
        CTF(ctf_params),
        Exposure(2.0, n_frames=10),
        voltage=VOLTAGE,
        scatter=dict(scattering_model="multislice", alpha=0.1),
        camera=dict(detector="k3_300kv", coincidence_radius=_CR),
    )
    save_or_compare("image_generator", images.array.cpu())


def test_particle_from_atoms_regression(small_coords, ctf_params, save_or_compare):
    """Particles rendered from atoms per pose: multislice, random ice, coincidence."""
    coords, atomic_numbers = small_coords
    torch.manual_seed(0)
    specimen = (
        Specimen.from_atoms(coords, atomic_numbers, 2.0, 16)
        .crowd(min_distance=40.0)
        .solvate(model="random")
    )
    torch.manual_seed(0)
    images = simulate_particles(
        specimen,
        _IDENTITY,
        CTF(ctf_params),
        Exposure(2.0, n_frames=10),
        voltage=VOLTAGE,
        scatter=dict(scattering_model="multislice", alpha=0.1),
        camera=dict(coincidence_radius=_CR),
    )
    save_or_compare("image_generator_from_coordinates", images.array.cpu())


def test_micrograph_regression(small_volume, ctf_params, save_or_compare):
    """Micrographs: a drawn field, projection scattering, random ice, coincidence."""
    torch.manual_seed(0)
    field = Field(
        small_volume,
        2.0,
        32,
        crowding=Crowding(min_distance=60.0, water_air_interface=True),
    ).solvate(Ice(model="random"))
    beam = dict(scattering_model="projection", alpha=0.1)
    exposure = Exposure(2.0, n_frames=10)
    torch.manual_seed(0)
    sample = field.draw(voltage=VOLTAGE, **beam)
    images = (
        sample.scatter(voltage=VOLTAGE, exposure=exposure, **beam)
        .apply_ctf(CTF(ctf_params))
        .intensity()
        .detect(exposure, coincidence_radius=_CR)
    )
    save_or_compare("micrograph_generator", images.array.cpu())


def test_tilt_series_regression(ctf_params, save_or_compare):
    """Tilt series: three tilts about y, projection scattering, coincidence loss."""
    series, exposure = _tilt_series(
        [-10.0, 0.0, 10.0],
        propagation=Propagation(scattering_model="projection", alpha=0.1),
    )
    exposure = Exposure.tilt_series(2.0, 3, n_frames=10)
    torch.manual_seed(0)
    images = [
        series.scatter(i)
        .apply_ctf(CTF(ctf_params))
        .intensity()
        .detect(exposure[i : i + 1], coincidence_radius=_CR)
        .array
        for i in range(3)
    ]
    save_or_compare("tilt_series_generator", torch.stack(images, dim=1).cpu())


# ---------------------------------------------------------------------------
# Ice blended into a pre-built volume
# ---------------------------------------------------------------------------


def _prepared_volume(small_volume_4d):
    volume = small_volume_4d.clone()
    sample = Sample.from_volume(volume.clone(), 2.0).prepare(
        Ice(model="random"), voltage=VOLTAGE
    )
    return sample.volume, volume


def _tilted_volume(small_volume_4d):
    series, _ = _tilt_series([0.0], ice=Ice(model="random"))
    return series.volume, _tilt_volume()


@pytest.mark.parametrize(
    ("make_volume", "particle"),
    [
        (_prepared_volume, (slice(12, 20), slice(12, 20), slice(12, 20))),
        (_tilted_volume, (slice(5, 11), slice(20, 28), slice(20, 28))),
    ],
    ids=["micrograph", "tilt_series"],
)
def test_ice_is_blended_into_a_prebuilt_volume(make_volume, particle, small_volume_4d):
    """A pre-built volume given an ice model has ice blended into it, matching
    its size and voxel size, only where the volume had little existing
    potential.

    For a tilt series this happens before the tilt-coverage padding; no
    padding is triggered at this geometry (the volume already meets the
    coverage), so the series' volume keeps the raw input's shape exactly.
    """
    torch.manual_seed(0)
    blended, original = make_volume(small_volume_4d)

    # Ice fills the existing volume, it does not grow it.
    assert blended.shape == original.shape
    # The particle region (above the ice mask threshold) is untouched.
    assert torch.equal(blended[(0, *particle)], original[(0, *particle)])
    # Elsewhere (originally near-zero potential), ice has been added.
    assert not torch.equal(blended, original)
    assert blended[0, 0, 0, 0] > 0


def test_a_prebuilt_volume_without_ice_is_unchanged(small_volume_4d):
    """With no ice, no absorption and no damage there is nothing to add."""
    original = small_volume_4d.clone()
    sample = Sample.from_volume(small_volume_4d.clone(), 2.0).prepare(voltage=VOLTAGE)
    assert torch.equal(sample.volume, original)


# ---------------------------------------------------------------------------
# The transfer function
# ---------------------------------------------------------------------------


def test_bfactor_damps_transfer_function():
    """B-factor envelope is a Fourier-space transfer-function damping term."""
    aberration = Aberration(
        n_pixels=16,
        pixel_size=2.0,
        voltage=300.0,
        aberration_model="nonlinear",
    )

    no_envelope = aberration.transfer_function({})
    zero_envelope = aberration.transfer_function({"bfactor": torch.tensor([0.0])})
    damped = aberration.transfer_function({"bfactor": torch.tensor([80.0])})

    assert torch.equal(no_envelope, zero_envelope)
    assert torch.equal(damped[:, 0, 0], no_envelope[:, 0, 0])
    assert (
        torch.abs(damped[:, 1:, 1:]).mean() < torch.abs(no_envelope[:, 1:, 1:]).mean()
    )


def test_aberration_bfactor_kwarg_matches_ctf_params_bfactor():
    """Aberration's constructor bfactor kwarg matches an equal ctf_params['bfactor']."""
    exitwave = torch.ones(1, 16, 16, dtype=torch.complex64)
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}

    img_via_ctf_params = Aberration(16, 2.0, 300.0)(
        exitwave, {**ctf_params, "bfactor": torch.tensor([80.0])}
    )
    img_via_kwarg = Aberration(16, 2.0, 300.0, bfactor=80.0)(exitwave, ctf_params)

    assert torch.allclose(img_via_kwarg, img_via_ctf_params)


def test_aberration_bfactor_kwarg_overrides_ctf_params_bfactor():
    """An explicit bfactor kwarg replaces (not adds to) ctf_params['bfactor']."""
    exitwave = torch.ones(1, 16, 16, dtype=torch.complex64)
    ctf_params = {
        "dfu": torch.tensor([5000.0]),
        "cs": torch.tensor([2.7]),
        "bfactor": torch.tensor([200.0]),
    }

    img_overridden_to_zero = Aberration(16, 2.0, 300.0, bfactor=0.0)(
        exitwave, ctf_params
    )
    img_no_bfactor = Aberration(16, 2.0, 300.0)(
        exitwave, {k: v for k, v in ctf_params.items() if k != "bfactor"}
    )

    assert torch.allclose(img_overridden_to_zero, img_no_bfactor)


def test_aberration_linear_model_specimen_absorption_true_matches_no_alpha():
    """specimen_absorption=True (default) must reproduce the pre-fix
    behaviour exactly -- amplitude contrast is assumed already baked into
    the exit wave upstream (true for every scattering_model except "ctf"),
    so this constructor's alpha must have no effect on the transfer
    function itself, regardless of its value."""
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    tf_alpha_0 = Aberration(16, 2.0, 300.0, aberration_model="linear", alpha=0.0)
    tf_alpha_1 = Aberration(16, 2.0, 300.0, aberration_model="linear", alpha=0.1)
    assert torch.allclose(
        tf_alpha_0.transfer_function(ctf_params),
        tf_alpha_1.transfer_function(ctf_params),
    )


def test_aberration_linear_model_specimen_absorption_false_applies_amp_contrast():
    """specimen_absorption=False folds -acos(alpha) into chi -- changes the
    transfer function even with no explicit ctf_params["phaseshift"], and
    differs between alpha values (unlike the specimen_absorption=True
    case above)."""
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}
    tf_alpha_0 = Aberration(
        16, 2.0, 300.0, aberration_model="linear", alpha=0.0, specimen_absorption=False
    )
    tf_alpha_1 = Aberration(
        16, 2.0, 300.0, aberration_model="linear", alpha=0.1, specimen_absorption=False
    )
    tf_no_absorption_guard = Aberration(
        16, 2.0, 300.0, aberration_model="linear", alpha=0.0
    )
    assert not torch.allclose(
        tf_alpha_0.transfer_function(ctf_params),
        tf_alpha_1.transfer_function(ctf_params),
    )
    assert not torch.allclose(
        tf_alpha_0.transfer_function(ctf_params),
        tf_no_absorption_guard.transfer_function(ctf_params),
    )


def test_aberration_frequency_grids_are_not_persisted():
    """Derived frequency grids follow the module across devices without being state.

    `Aberration` builds its frequency grids with `torch.meshgrid`, which
    returns stride-0 broadcast views; persisted, they cannot be restored by
    `load_state_dict`, which writes through them in place. They are pure
    functions of the constructor arguments, so they are not persisted at all.
    """
    aberration = Aberration(16, 2.0, 300.0, progressbars=False)
    persisted = set(aberration.state_dict())
    for name in ("k", "k2", "radian", "KY", "KX"):
        assert name in aberration._buffers, f"{name} must stay a buffer"
        assert name not in persisted, f"{name} is derived and must not be persisted"


def _image(small_volume, ctf_params, *, alpha: float, model: str) -> torch.Tensor:
    waves = Specimen(small_volume, 1.0).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model=model, alpha=alpha
    )
    return (
        waves.apply_ctf(CTF(ctf_params)).intensity().detect(Exposure(2.0), noise=None)
    )


def test_the_ctf_model_gets_its_amplitude_contrast_at_the_lens(
    small_volume, ctf_params
):
    """`scattering_model="ctf"` propagates a real projection with no absorptive
    part of its own, so its amplitude contrast is applied by the linear
    transfer function, with no explicit opt-in."""
    waves = Specimen(small_volume, 1.0).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model="ctf", alpha=0.0
    )
    assert waves.linear and not waves.array.is_complex()
    img_alpha_0 = _image(small_volume, ctf_params, alpha=0.0, model="ctf")
    img_alpha_1 = _image(small_volume, ctf_params, alpha=0.1, model="ctf")
    assert not torch.allclose(img_alpha_0.array, img_alpha_1.array)


def test_a_wave_model_is_not_given_its_amplitude_contrast_twice(
    small_volume, ctf_params
):
    """Multislice applies alpha upstream (`potential.apply_amplitude_contrast`),
    so the lens must not add it again: the wave's own alpha, carried for the
    linear model, has no effect on a nonlinear one."""
    waves = Specimen(small_volume, 1.0).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model="multislice", alpha=0.1
    )
    assert not waves.linear
    plain = ExitWave.apply_ctf(waves, CTF(ctf_params)).array
    from dataclasses import replace

    relabelled = replace(waves, alpha=0.5).apply_ctf(CTF(ctf_params)).array
    assert torch.equal(plain, relabelled)


# ---------------------------------------------------------------------------
# Envelopes: each reaches the stage that applies it, in every modality
# ---------------------------------------------------------------------------


_CC = dict(cc=2.7, energy_spread=0.8, deltaV_V=0.05e-6, deltaI_I=0.02e-6)


def _particle_waves(small_volume, small_coords) -> ExitWave:
    return Specimen(small_volume, 2.0).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model="projection"
    )


def _atoms_waves(small_volume, small_coords) -> ExitWave:
    coords, atomic_numbers = small_coords
    return Specimen.from_atoms(coords, atomic_numbers, 2.0, 16).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model="projection"
    )


def _micrograph_waves(small_volume, small_coords) -> ExitWave:
    sample = Sample.from_volume(small_volume, 2.0)
    return sample.scatter(voltage=VOLTAGE, scattering_model="projection")


def _tilt_waves(small_volume, small_coords) -> ExitWave:
    series, _ = _tilt_series(
        [-10.0, 0.0, 10.0], propagation=Propagation(scattering_model="projection")
    )
    return series.scatter(1)


@pytest.mark.parametrize(
    "make_waves",
    [_particle_waves, _atoms_waves, _micrograph_waves, _tilt_waves],
    ids=["particle", "from_atoms", "micrograph", "tilt_series"],
)
def test_envelopes_reach_the_stage_that_applies_them(
    make_waves, small_volume, small_coords, ctf_params
):
    """
    The spatial-coherence and dose envelopes act on the transfer function,
    and the Cc focal spread where intensities add: summed over by default,
    or as the transfer function's envelope under
    ``temporal_coherence="envelope"``. Every modality's exit wave reaches
    the same two stages.
    """
    from specter.aberrations import FocalSpread, focal_spread_sigma
    from specter.aberrations._envelopes import cc_envelope
    from specter.constants import energy_to_wavelength

    waves = make_waves(small_volume, small_coords)
    lens = waves.apply_ctf(
        CTF(ctf_params), convergence_angle=0.02, dose_envelope=Exposure(2.0)
    )
    s = waves.oversample
    reference = stages.aberrate(
        waves.array,
        Aberration(
            waves.pad_nxy * s,
            waves.pixel_size / s,
            VOLTAGE,
            convergence_angle=0.02,
            dose_envelope=True,
            dose_weighted=True,
            progressbars=False,
        ),
        {**ctf_params, "dose": torch.tensor([2.0])},
        s,
        waves.pad_nxy,
    )
    assert waves.defocus_shift == 0.0  # projection: no depth to offset from
    torch.testing.assert_close(lens.array, reference, rtol=0, atol=0)
    assert not torch.equal(lens.array, waves.apply_ctf(CTF(ctf_params)).array)

    wavelength = energy_to_wavelength(VOLTAGE)
    focal = FocalSpread.from_instrument(
        2.7e7, VOLTAGE * 1e3, 0.8, 0.05e-6, 0.02e-6, wavelength
    )
    assert focal.sigma_f == pytest.approx(
        focal_spread_sigma(2.7e7, 300e3, 0.8, 0.05e-6, 0.02e-6)
    )
    grid = lens.array.shape[-1] // lens.pad_nxy
    dx = lens.pixel_size / grid
    integrated = lens.intensity(**_CC).array
    torch.testing.assert_close(
        integrated, focal.intensity(lens.array, dx, grid), rtol=0, atol=0
    )

    k = torch.fft.fftfreq(lens.array.shape[-1], dx)
    envelope = cc_envelope(
        k[:, None] ** 2 + k[None, :] ** 2,
        wavelength,
        2.7e7,
        VOLTAGE * 1e3,
        0.8,
        0.05e-6,
        0.02e-6,
    )
    enveloped = lens.intensity(**_CC, temporal_coherence="envelope").array
    want = torch.fft.ifft2(torch.fft.fft2(lens.array) * envelope).abs().square()
    torch.testing.assert_close(enveloped, want)
    assert not torch.equal(enveloped, integrated)


def test_the_dose_envelope_changes_the_image(small_volume, ctf_params):
    """With a non-trivial dose, the dose envelope changes the simulated image."""
    waves = Specimen(small_volume, 2.0).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model="projection", alpha=0.1
    )
    exposure = Exposure(40.0)

    def image(envelope: Exposure | None) -> torch.Tensor:
        lens = waves.apply_ctf(CTF(ctf_params), dose_envelope=envelope)
        return lens.intensity().detect(exposure, noise=None).array

    assert not torch.allclose(image(None), image(exposure))


# ---------------------------------------------------------------------------
# The defocus convention
# ---------------------------------------------------------------------------


def test_the_ctf_keeps_the_nominal_defocus(small_volume, ctf_params):
    """
    Multislice returns the wave at the volume's exit face, half its Z extent
    from the midplane whose defocus the CryoSPARC/RELION value is. The exit
    face is the high-defocus end (an underfocused lens images a plane
    upstream), so the wave carries minus half the depth and `apply_ctf`
    subtracts it, so the `CTF` a caller holds, and writes back out, stays the
    nominal, midplane defocus.
    """
    original = ctf_params["dfu"].clone()
    ctf = CTF(ctf_params)
    waves = Specimen(small_volume, 2.0).scatter(
        _IDENTITY, voltage=VOLTAGE, scattering_model="multislice", alpha=0.1
    )
    expected = -small_volume.shape[0] * 2.0 / 2
    assert expected < 0
    assert waves.defocus_shift == pytest.approx(expected)

    lens = waves.apply_ctf(ctf)
    assert torch.equal(ctf.params["dfu"], original)
    shifted = CTF(
        {**ctf_params, "dfu": original - expected, "dfv": original - expected}
    )
    from dataclasses import replace

    unshifted = replace(waves, defocus_shift=0.0).apply_ctf(shifted)
    torch.testing.assert_close(lens.array, unshifted.array)


# ---------------------------------------------------------------------------
# Tilt geometry
# ---------------------------------------------------------------------------


def test_a_tilted_uniform_slab_images_flat_to_its_edges():
    """
    A laterally uniform slab must image flat at any tilt. The FFT joins the
    frame's opposite edges, which under tilt cross the slab at different
    depths, so an unpadded canvas darkens and rings at the edges (16% off at
    60 degrees). A tilt series pads by default, and the 32 px margin leaves
    0.3% from the oversampled slice interpolation.
    """
    nxy, px = 64, 4.0
    volume = torch.full((1, 50, 160, 160), 4.5)
    poses = Poses.tilt_series([60.0], axis="y")

    def edge_deviation(**propagation: object) -> float:
        series = Sample.from_volume(volume.clone(), px).tilt_series(
            poses,
            Exposure.tilt_series(2.0, 1),
            voltage=300.0,
            micrograph_size=nxy,
            tilt=TiltGeometry(tilt_axis="y"),
            log=False,
            **propagation,
        )
        with torch.no_grad():
            image = series.scatter(0).intensity(cc=None).pixels()[0]
        m = (image.shape[-1] - nxy) // 2
        columns = image[m : m + nxy, m : m + nxy].mean(dim=0)
        return float((columns / columns[16:-16].mean() - 1).abs().max())

    assert edge_deviation() < 0.01
    assert edge_deviation(pad_fft=False) > 0.05


def test_edge_margin_pads_beyond_the_geometric_minimum(ctf_params):
    """edge_margin adds real interpolation slack beyond the exact geometric minimum.

    Regression guard for a tilt-angle artifact: the geometric minimum leaves
    zero slack, so output pixels at the edge of the crop have a per-slice
    sampling footprint that lands exactly on the padded-volume boundary at
    the deepest Z, with no room for interpolation. That produces a real
    artifact -- a bright line through the origin in the FFT, aligned with the
    tilt axis. edge_margin fixes it by always padding beyond the geometric
    minimum, independent of taper_width (which only fades pixels that are
    provably never sampled, and has zero effect on output).
    """
    projection = Propagation(scattering_model="projection")
    no_margin, _ = _tilt_series([45.0], propagation=projection, edge_margin=0)
    default, _ = _tilt_series([45.0], propagation=projection)  # edge_margin=8

    assert default.volume.shape[-1] == no_margin.volume.shape[-1] + 2 * 8

    def image(series) -> torch.Tensor:
        lens = series.scatter(0).apply_ctf(CTF(ctf_params))
        return lens.intensity().detect(Exposure(2.0), noise=None).array

    assert not torch.allclose(image(no_margin), image(default))


def test_pad_fft_multislice_shapes_match(ctf_params):
    """pad_fft must not change the detected image's shape, and must not crash
    -- at any tilt.

    Regression guard for a tilt-angle artifact distinct from edge_margin's:
    multislice's per-slice Fresnel propagation is an FFT-based convolution
    that is periodic over whatever canvas it runs on. Propagating at exactly
    the output size gives it zero headroom; under tilt the resulting per-step
    wraparound compounds coherently into a visible artifact along all four
    frame edges. pad_fft pads once before the recursion; the wave and its
    intensity keep the padded canvas through the transfer function, also a
    circular convolution, and the detector crops the image to the field.

    This also covers the scenario that used to crash: the identity (0 deg)
    path previously did raw tensor indexing with no bounds handling, and the
    padded canvas wasn't guaranteed to fit inside the edge_margin-provisioned
    volume.
    """
    for angle in (0.0, 45.0):
        series, exposure = _tilt_series(
            [angle],
            propagation=Propagation(scattering_model="multislice", pad_fft=True),
        )
        assert series.scattering.nxy == series.nxy == 32

        waves = series.scatter(0)
        canvas = 32 + 2 * series.scattering.fft_pad_margin
        assert waves.pad_nxy == canvas
        intensity = waves.apply_ctf(CTF(ctf_params)).intensity()
        image = intensity.detect(exposure[0:1], noise=None)
        assert image.array.shape[-2:] == (32, 32)
        for tensor in (waves.pixels(), intensity.pixels()):
            assert tensor.shape[-2:] == (canvas, canvas)


def test_pad_fft_changes_the_exit_wave_under_tilt():
    """pad_fft materially changes multislice output at a non-trivial tilt,
    where the wraparound it fixes is present: it is not a silent no-op."""
    base, _ = _tilt_series(
        [45.0], propagation=Propagation(scattering_model="multislice", pad_fft=False)
    )
    padded, _ = _tilt_series(
        [45.0], propagation=Propagation(scattering_model="multislice", pad_fft=True)
    )
    field = center_crop(padded.scatter(0).pixels(), 32, dim=(-2, -1))
    assert not torch.allclose(base.scatter(0).pixels(), field)


def test_a_series_leaves_its_volume_where_it_was_given():
    """
    Preparing a series must not drag its volume onto a device: only
    `scatter` decides, when it knows whether the volume fits. A series is
    not a module, so nothing like a module-wide ``.to(device)`` can move it
    behind that decision either.
    """
    series, _ = _tilt_series([30.0])
    assert not isinstance(series, torch.nn.Module)
    assert series.volume.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_series_places_a_volume_that_fits():
    """A volume that comfortably fits ends up on the compute device on first
    use -- no flag needed for the common case."""
    series, _ = _tilt_series(
        [30.0], propagation=Propagation(scattering_model="multislice")
    )
    assert series.volume.device.type == "cpu"
    waves = series.scatter(0, device="cuda")
    assert series.volume.device.type == "cuda"
    assert torch.isfinite(waves.array).all()


def _streamed_series(monkeypatch):
    """A series whose volume's upload to the device raises an out-of-memory
    error, injected rather than provoked, and the same series uploaded."""
    projection = Propagation(scattering_model="multislice")
    fits, _ = _tilt_series([30.0], propagation=projection)
    want = fits.scatter(0, device="cuda").pixels().cpu()

    streamed, _ = _tilt_series([30.0], propagation=projection)
    original_to = torch.Tensor.to

    def fake_to(tensor_self, *args, **kw):
        if tensor_self is streamed.volume:
            raise torch.cuda.OutOfMemoryError("simulated OOM for test")
        return original_to(tensor_self, *args, **kw)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    return streamed, want


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_series_falls_back_to_the_host_when_its_volume_does_not_fit(monkeypatch):
    """When moving the volume to the device runs out of memory, the series
    still completes, with the volume kept on the host and the same result."""
    streamed, want = _streamed_series(monkeypatch)
    with pytest.warns(UserWarning, match="do not fit"):
        waves = streamed.scatter(0, device="cuda")
    assert streamed.volume.device.type == "cpu"  # the fallback kept it there
    assert torch.allclose(want, waves.pixels().cpu(), atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_streamed_series_still_computes_on_the_device(monkeypatch):
    """Falling back keeps the VOLUME on the host; the propagation still runs on
    the device, fetching geometry-bounded blocks, so GPU memory stays bounded
    without giving up the device."""
    streamed, _ = _streamed_series(monkeypatch)
    torch.cuda.reset_peak_memory_stats()
    with pytest.warns(UserWarning, match="do not fit"):
        waves = streamed.scatter(0, device="cuda")
    assert waves.array.is_cuda
    # The whole volume is ~147 KB here; the point is that the peak must not
    # scale with it.
    assert torch.cuda.max_memory_allocated() < 50_000_000


# ---------------------------------------------------------------------------
# Atoms rendered per pose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift_angstrom", [8.0, -6.0])
def test_an_atom_shift_is_in_angstrom(small_coords, shift_angstrom):
    """
    A requested shift moves the atoms by that many Angstrom.

    `translations_angstrom_to_torch` normalizes to grid_sample's [-1, 1]
    convention, which is meaningless for atom positions: routing the
    translation through it shrank every shift by a factor of
    n * pixel_size / 2. The shifts here are whole voxels at 2 A, so the
    rendered volume moves by exactly that many voxels, against x
    (translations are subtracted, in the lab frame).
    """
    coords, atomic_numbers = small_coords
    specimen = Specimen.from_atoms(coords, atomic_numbers, 2.0, 32)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def render(shift: float) -> torch.Tensor:
        poses = Poses(identity, torch.tensor([[shift, 0.0]]), torch.arange(1))
        return specimen.materialize(poses, voltage=VOLTAGE)[0]

    voxels = int(shift_angstrom / 2.0)
    torch.testing.assert_close(
        render(shift_angstrom),
        torch.roll(render(0.0), -voxels, dims=-1),
        atol=1e-4,
        rtol=0,
    )


def test_the_atom_and_volume_paths_agree(small_coords):
    """
    A specimen rendered from its atoms and one rotated as a volume must place
    the particle identically for the same pose.

    `Specimen.from_atoms` rotates the atoms and renders them; a volume
    specimen rotates a prebuilt map through `build_affine_matrix`. They are
    separate implementations of one convention, compared here through the
    same public `materialize`. Both are blurred first: a rendered atom is
    cusped, and trilinear rotation of a cusp differs from re-rendering it by
    more than a sharp comparison tolerates. Under a 3-voxel blur the two
    paths differ by at most 0.04 of the peak over random poses, against at
    least 0.51 with the rotation inverted or the shift flipped.
    """
    coords, atomic_numbers = small_coords
    n, voxel_size = 32, 1.0
    atoms = Specimen.from_atoms(coords, atomic_numbers, voxel_size, n)
    volume = Specimen(atoms.potential, voxel_size)
    poses = Poses(roma.random_unitquat(1), torch.tensor([[4.0, -2.5]]), torch.arange(1))

    def blurred(specimen: Specimen) -> torch.Tensor:
        V = specimen.materialize(poses, voltage=VOLTAGE)[0]
        k = torch.fft.fftfreq(n)
        k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
        return torch.fft.ifftn(
            torch.fft.fftn(V) * torch.exp(-2 * math.pi**2 * 9 * k2)
        ).real

    volume_a, volume_b = blurred(atoms), blurred(volume)
    peak = volume_a.max()
    diff = (volume_a - volume_b).abs()
    assert diff.max() < 0.1 * peak
    assert diff.mean() < 0.005 * peak


# ---------------------------------------------------------------------------
# Micrograph placement and batches
# ---------------------------------------------------------------------------


def test_a_sample_falls_back_to_the_host_when_it_does_not_fit(small_volume):
    """
    A specimen volume too large for the device streams from the host instead.

    A micrograph field is assembled on the host, and imaging it uploads it.
    At ``micrograph_size`` that upload can exceed the device, and it used to
    be unconditional -- which OOM'd the very device the field had just been
    assembled off. `IterativeScattering` accepts an off-device volume, so the
    fallback costs per-slice transfers rather than the run.

    The OOM is injected rather than provoked: reproducing it for real needs a
    volume larger than whatever GPU the suite happens to run on, and this must
    also run on CPU-only machines.
    """
    calls = {"n": 0}

    class _VolumeThatWontFit:
        """Stands in for a volume whose upload exhausts the device."""

        device = torch.device("cpu")

        def numel(self):
            return 8_400_000_000

        def element_size(self):
            return 4

        def to(self, *args, **kwargs):
            calls["n"] += 1
            raise torch.cuda.OutOfMemoryError("injected")

        def cpu(self):
            return self

    stub = _VolumeThatWontFit()
    sample = Sample(stub, 2.0)  # type: ignore[arg-type]

    with pytest.warns(UserWarning, match="does not fit|do not fit"):
        placed = sample.placed("cuda:0")

    assert calls["n"] == 1
    assert placed.volume is stub, "the volume must stay on the host"


def _clean_sample(volume):
    """A drawn micrograph field that keeps its ice-free volume."""
    field = Field(
        volume,
        2.0,
        32,
        crowding=Crowding(min_distance=60.0, water_air_interface=True),
    ).solvate(Ice(model="random"))
    return field.draw(voltage=VOLTAGE, keep_dry=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_clean_volume_streams_when_it_does_not_fit(small_volume, monkeypatch):
    """
    The clean volume is a whole second canvas, and its upload used to be an
    unconditional ``.to(device)`` beside the main volume's guarded one, so a
    run that had just fallen back to streaming the specimen then OOM'd on
    its ice-free twin. It goes through the same fallback.
    """
    torch.manual_seed(0)
    sample = _clean_sample(small_volume)
    want = sample.clean().scatter(voltage=VOLTAGE, device="cuda").pixels().cpu()

    clean = sample.dry
    original_to = torch.Tensor.to

    def fake_to(tensor_self, *args, **kw):
        if tensor_self is clean and "cuda" in str(args[:1] + tuple(kw.values())):
            raise torch.cuda.OutOfMemoryError("injected")
        return original_to(tensor_self, *args, **kw)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    with pytest.warns(UserWarning, match="do not fit"):
        got = sample.clean().scatter(voltage=VOLTAGE, device="cuda")
    assert torch.equal(got.pixels().cpu(), want)


def test_a_micrograph_batch_propagates_once(small_volume, ctf_params, monkeypatch):
    """
    Every image of a micrograph batch sees one specimen at one pose, so the
    batch's exit waves are identical: the volume is propagated once, and each
    image still gets its own CTF and noise -- bit-identical to propagating B
    copies.
    """
    B = 3
    torch.manual_seed(0)
    sample = _clean_sample(small_volume)
    seen = []
    scatter = IterativeScattering.forward

    def record(self, V, *args, **kw):
        seen.append(V.shape[0])
        return scatter(self, V, *args, **kw)

    monkeypatch.setattr(IterativeScattering, "forward", record)
    exposure = Exposure(torch.full((B,), 20.0))
    beam = dict(voltage=VOLTAGE, ids=torch.arange(B), exposure=exposure)
    with torch.no_grad():
        waves = sample.scatter(**beam)
        clean = sample.clean().scatter(**beam)
    assert seen == [1, 1]  # the iced and the clean exit wave, once each
    assert waves.array.shape[0] == B and clean.array.shape[0] == B

    # The same batch the old way: B copies through multislice.
    monkeypatch.setattr(IterativeScattering, "forward", scatter)
    propagator = IterativeScattering(
        32, 2.0, VOLTAGE, oversample=2, crop=False, progressbars=False
    )
    with torch.no_grad():
        copies = propagator(sample.volume.expand(B, -1, -1, -1), pose=0)
    assert torch.equal(waves.array, copies)

    params = {k: v.expand(B).clone() for k, v in ctf_params.items()}
    params["dfu"] = params["dfu"] + 100.0 * torch.arange(B)
    params["dfv"] = params["dfu"].clone()
    images = waves.apply_ctf(CTF(params)).intensity().detect(exposure, seed=1)
    assert not torch.equal(images.array[0], images.array[1])


# ---------------------------------------------------------------------------
# Ice kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entrypoint", ["volume", "atoms", "field", "prepared", "tilt_series"]
)
@pytest.mark.parametrize(
    ("ice", "expected"),
    [
        (Ice(model="random", parameterization="lobato"), "lobato"),
        (Ice(model="random"), "kirkland"),
    ],
    ids=["explicit_lobato", "shared_default"],
)
def test_every_entry_point_resolves_the_requested_ice_kernel(
    entrypoint, ice, expected, small_coords, monkeypatch
):
    """Every entry point builds the requested water kernel, including defaults."""
    from specter.ice._kernels import build_water_kernel

    n = 16
    volume = torch.zeros(n, n, n)
    if entrypoint == "volume":
        maker = Specimen(volume, 2.0).solvate(ice).icemaker
    elif entrypoint == "atoms":
        coords, atomic_numbers = small_coords
        maker = (
            Specimen.from_atoms(coords, atomic_numbers, 2.0, n).solvate(ice).icemaker
        )
    elif entrypoint == "field":
        maker = Field(volume, 2.0, n).solvate(ice).icemaker
    else:
        # These consume their icemaker at once, blending ice into a prebuilt
        # canvas. Inspect that actual input to the blend.
        import specter.imaging._field as field_module
        import specter.imaging._tilt as tilt_module

        seen = []

        def capture(volume, icemaker, *args, **kwargs):
            seen.append(icemaker)
            return volume

        module = field_module if entrypoint == "prepared" else tilt_module
        monkeypatch.setattr(module, "blend_ice_into_volume", capture)
        sample = Sample.from_volume(volume, 2.0)
        if entrypoint == "prepared":
            sample.prepare(ice, voltage=VOLTAGE)
        else:
            sample.tilt_series(
                Poses.tilt_series([0.0]),
                Exposure.tilt_series(2.0, 1),
                voltage=VOLTAGE,
                micrograph_size=n,
                ice=ice,
                log=False,
            )
        assert len(seen) == 1
        maker = seen[0]

    assert maker is not None
    assert maker.parameterization == expected
    torch.testing.assert_close(maker.ice_kernel, build_water_kernel(2.0, expected))


# ---------------------------------------------------------------------------
# Crowding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ice_thickness", [0.0, 200.0, 2000.0])
def test_crowding_slab_does_not_follow_ice_thickness(small_volume, ice_thickness):
    """
    `ice_thickness` sizes the canvas and the ice, not the neighbour slab.

    It is a contrast knob: raising it lowers contrast and deepens the solvent
    background. Letting it also deepen the crowd's ``max_distance_z`` would
    make one field change both the imaging conditions and the specimen, and
    it stacked ever more neighbours above each other as the ice grew -- 6
    duplicates at 0 A against 46 at 2000 A, measured through
    `run_particle_stack` at 128 px / 2 A. The slab is the template's own
    depth instead, which is what ``nz * pixel_size`` already evaluated to
    whenever the ice fitted the box, so only the growth is gone. A
    micrograph `Field` does scale with ``nz``, deliberately.
    """
    specimen = Specimen(
        small_volume, 2.0, ice=Ice(model=None, thickness=ice_thickness)
    ).crowd(min_distance=60.0)
    crowd = specimen._build_crowd(pad_fft=False)
    assert crowd is not None
    assert crowd.max_distance_z == small_volume.shape[0] * 2.0
    # The canvas itself still grows, or the knob would do nothing at all.
    assert specimen.nz == max(small_volume.shape[0], int(ice_thickness // 2.0))


def test_crowding_slab_from_atoms_does_not_follow_ice_thickness(small_coords):
    """A specimen rendered from atoms shares the rule; its template is nxy deep."""
    from dataclasses import replace

    coords, atomic_numbers = small_coords
    base = Specimen.from_atoms(coords, atomic_numbers, 2.0, 32)
    slabs = []
    for ice_thickness in (0.0, 2000.0):
        specimen = replace(base, ice=Ice(model=None, thickness=ice_thickness)).crowd(
            min_distance=60.0
        )
        crowd = specimen._build_crowd(pad_fft=False)
        assert crowd is not None
        slabs.append(crowd.max_distance_z)
    assert slabs == [32 * 2.0, 32 * 2.0]


# ---------------------------------------------------------------------------
# Per-tilt parameters
# ---------------------------------------------------------------------------


def test_each_tilt_uses_its_own_parameters(monkeypatch):
    """
    Per-tilt parameters are one entry per tilt and must be selected by the
    tilt, not by the series. Until 2026-09-08 every tilt reused tilt 0's
    defocus, dose and coincidence radius, and the dose envelope saw a
    pre-exposure of zero on every tilt, so a tilt series carried no
    accumulated radiation damage at all.
    """
    volume = torch.zeros(1, 16, 64, 64)
    volume[0, 5:11, 24:40, 24:40] = 30.0
    n = 3
    ctf = CTF(
        {
            "dfu": torch.full((n,), 5000.0),
            "dfv": torch.full((n,), 5000.0),
            "dfang": torch.zeros(n),
            "cs": torch.full((n,), 2.7e7),
        }
    )
    projection = Propagation(scattering_model="projection", alpha=0.1)

    def images(dose: torch.Tensor, envelope: bool) -> torch.Tensor:
        series, exposure = _tilt_series(
            [0.0, 0.0, 0.0],
            dose=dose,
            volume=volume.clone(),
            micrograph_size=64,
            propagation=projection,
        )
        out = []
        with torch.no_grad():
            for i in range(n):
                one = slice(i, i + 1)
                lens = series.scatter(i).apply_ctf(
                    ctf[one], dose_envelope=exposure[one] if envelope else None
                )
                out.append(lens.intensity().detect(exposure[one], noise=None).array)
        return torch.cat(out)

    dose = torch.tensor([1.0, 2.0, 4.0])
    means = images(dose, envelope=False).mean(dim=(-1, -2))
    # counts scale with each tilt's own dose: 1 : 2 : 4
    assert torch.allclose(means / means[0], dose / dose[0], rtol=1e-3)

    # the dose envelope sees the accumulated pre-exposure: the third tilt,
    # identical in geometry and dose to the first but taken after the others,
    # carries less high-frequency signal
    seen = []
    forward = Aberration.forward

    def spy(self, exitwave, ctf_params):
        seen.append(float(ctf_params["pre_exposure"].flatten()[0]))
        return forward(self, exitwave, ctf_params)

    monkeypatch.setattr(Aberration, "forward", spy)
    enveloped = images(torch.tensor([2.0, 60.0, 2.0]), envelope=True)
    assert seen == [0.0, 2.0, 62.0]
    k = torch.fft.fftfreq(64, d=2.0)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    hp = (k2 > (1 / 30.0) ** 2).float()

    def high_power(img):
        return float((torch.fft.fft2(img - img.mean()).abs() ** 2 * hp).sum())

    assert high_power(enveloped[2]) < 0.9 * high_power(enveloped[0])


def test_a_misspelt_setting_fails_loudly(small_volume):
    """A misspelt or moved setting must fail loudly, not become a silent no-op:
    `dose_envelope` belongs to `apply_ctf`, not to the scattering stage."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        Specimen(small_volume, 2.0).scatter(
            _IDENTITY, voltage=VOLTAGE, dose_envelope=True
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        _tilt_series([0.0], dose_envelope=True)
