"""
Golden-output regression tests for all four image generators.

On the first run, each test saves its output as a fixture under
tests/test_data/. Subsequent runs load the fixture and assert that
the output is numerically identical. To regenerate a fixture, delete
the corresponding .pt file and re-run.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import roma
import torch

from specter.aberrations import Aberration
from specter.imagegenerator import (
    ImageGenerator,
    ImageGeneratorFromCoordinates,
    MicrographGenerator,
    TiltSeriesGenerator,
)

FIXTURE_DIR = Path(__file__).parent / "test_data"

# These fixtures predate `coincidence_radius` being rescaled to mean a true
# effective exclusion radius (exclusion area pi*r^2, rather than the old
# grid-cell-indexing convention with area r^2/2). Expressing the old 1.8 via
# the exact conversion keeps the suppression grid bit-identical, so the
# fixtures still verify the physics rather than just the new parameterization.
_CR = 1.8 / math.sqrt(2 * math.pi)


def _save_or_compare(name: str, tensor: torch.Tensor) -> None:
    """Save tensor as a fixture on first run; compare on subsequent runs."""
    path = FIXTURE_DIR / f"{name}.pt"
    if path.exists():
        expected = torch.load(path, weights_only=True)
        assert torch.allclose(tensor.float(), expected.float(), atol=1e-4), (
            f"Regression failure for '{name}'. "
            "Delete the fixture file and re-run to regenerate."
        )
    else:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.cpu(), path)
        pytest.skip(f"Fixture '{name}.pt' generated — re-run to verify.")


@pytest.fixture
def small_volume():
    """3D cubic volume (32, 32, 32) with a simple phantom."""
    volume = torch.zeros(32, 32, 32)
    volume[12:20, 12:20, 12:20] = 50.0
    return volume


@pytest.fixture
def small_volume_4d():
    """4D cubic volume (1, 32, 32, 32) as returned by MicrographSpecimenGenerator."""
    volume = torch.zeros(1, 32, 32, 32)
    volume[0, 12:20, 12:20, 12:20] = 50.0
    return volume


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


def test_image_generator_regression(small_volume, ctf_params):
    """ImageGenerator: multislice scattering, random ice, coincidence loss, k3 detector."""
    torch.manual_seed(0)
    gen = ImageGenerator(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[0.7071, 0.0, 0.7071, 0.0]]),
        translations=torch.tensor([[1.0, -1.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="multislice",
        ice_model="random",
        alpha=0.1,
        coincidence_radius=_CR,
        n_frames=10,
        detector_model="k3_300kv",
        crowd_min_distance=60.0,
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    images = gen(torch.tensor([0]))
    _save_or_compare("image_generator", images.cpu())


def test_image_generator_from_coordinates_regression(small_coords, ctf_params):
    """ImageGeneratorFromCoordinates: voxelize coords, multislice, random ice, coincidence."""
    coords, atomic_numbers = small_coords
    torch.manual_seed(0)
    gen = ImageGeneratorFromCoordinates(
        coordinates=coords,
        atomic_numbers=atomic_numbers,
        nxy=16,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="multislice",
        ice_model="random",
        alpha=0.1,
        coincidence_radius=_CR,
        n_frames=10,
        crowd_min_distance=40.0,
        verbose=False,
    )
    torch.manual_seed(0)
    images = gen(torch.tensor([0]))
    _save_or_compare("image_generator_from_coordinates", images.cpu())


def test_micrograph_generator_regression(small_volume, ctf_params):
    """MicrographGenerator: projection scattering, random ice, coincidence."""
    torch.manual_seed(0)
    gen = MicrographGenerator(
        scattering_potential=small_volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="projection",
        ice_model="random",
        alpha=0.1,
        coincidence_radius=_CR,
        n_frames=10,
        crowd_min_distance=60.0,
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    images = gen(torch.tensor([0]))
    _save_or_compare("micrograph_generator", images.cpu())


def test_micrograph_generator_accepts_prebuilt_icemaker(small_volume, ctf_params):
    """A pre-built icemaker passed via icemaker= is reused, not rebuilt internally."""
    from specter.ice import RandomIcemaker

    icemaker = RandomIcemaker(dx=2.0, n=32, nz=32, progressbars=False)

    gen = MicrographGenerator(
        scattering_potential=small_volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="projection",
        icemaker=icemaker,
        alpha=0.1,
        coincidence_radius=_CR,
        n_frames=10,
        verbose=False,
        progressbars=False,
    )
    assert gen.specimen_gen.icemaker is icemaker
    images = gen(torch.tensor([0]))
    assert images.shape == (1, 32, 32)


def test_micrograph_generator_blends_ice_into_prebuilt_volume(
    small_volume_4d, ctf_params
):
    """MicrographGenerator(volume=..., ice_model=...) blends ice into volume at construction,
    matching its size/voxel size, only where volume had little existing potential."""
    torch.manual_seed(0)
    original_volume = small_volume_4d.clone()

    gen = MicrographGenerator(
        scattering_potential=None,
        volume=small_volume_4d.clone(),
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        ice_model="random",
        verbose=False,
        progressbars=False,
    )

    # Same shape as the input volume -- ice fills the existing volume, doesn't grow it.
    assert gen.volume.shape == original_volume.shape
    # Particle region (originally high potential, above the ice mask threshold) is
    # untouched.
    assert torch.equal(
        gen.volume[0, 12:20, 12:20, 12:20], original_volume[0, 12:20, 12:20, 12:20]
    )
    # Elsewhere (originally near-zero potential), ice has been added.
    assert not torch.equal(gen.volume, original_volume)
    assert gen.volume[0, 0, 0, 0] > 0


def test_micrograph_generator_volume_without_ice_model_is_unchanged(
    small_volume_4d, ctf_params
):
    """volume= with no ice_model/icemaker is registered as-is (no ice added)."""
    original_volume = small_volume_4d.clone()
    gen = MicrographGenerator(
        scattering_potential=None,
        volume=small_volume_4d.clone(),
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        verbose=False,
        progressbars=False,
    )
    assert torch.equal(gen.volume, original_volume)


def test_tilt_series_generator_regression(ctf_params):
    """TiltSeriesGenerator: 3-angle tilt series, coincidence loss, tilt_axis='y'."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([-10.0, 0.0, 10.0])

    torch.manual_seed(0)
    gen = TiltSeriesGenerator(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=angles,
        noise_model="poisson",
        scattering_model="projection",
        alpha=0.1,
        coincidence_radius=_CR,
        n_frames=10,
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    tilt_series, _, _ = gen.generate_tilt_series(torch.tensor([0]))
    _save_or_compare("tilt_series_generator", tilt_series.cpu())


def test_tilt_series_generator_blends_ice_into_volume(ctf_params):
    """TiltSeriesGenerator(volume=..., ice_model=...) blends ice into volume, matching its
    size/voxel size, before the class's own tilt-coverage padding is applied."""
    torch.manual_seed(0)
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    original_volume = volume.clone()

    gen = TiltSeriesGenerator(
        volume=volume.clone(),
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=torch.tensor([0.0]),
        ice_model="random",
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )

    # No tilt-coverage padding triggered at this geometry (available_nxy already
    # meets target_nxy), so gen.volume's shape matches the raw input exactly.
    assert gen.volume.shape == original_volume.shape
    # Particle region (originally high potential, above the ice mask threshold) is
    # untouched.
    assert torch.equal(
        gen.volume[0, 5:11, 20:28, 20:28], original_volume[0, 5:11, 20:28, 20:28]
    )
    # Elsewhere (originally near-zero potential), ice has been added.
    assert not torch.equal(gen.volume, original_volume)
    assert gen.volume[0, 0, 0, 0] > 0


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


def test_image_generator_ctf_scattering_model_gets_amp_contrast_by_default(
    small_volume, ctf_params
):
    """BaseImager._init_optics wires specimen_absorption=(scattering_model
    != "ctf") automatically -- scattering_model="ctf" must pick up
    amplitude contrast without any explicit opt-in."""
    kwargs = dict(
        scattering_potential=small_volume,
        pixel_size=1.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="ctf",
        noise_model=None,
        ice_model=None,
        detector_model=None,
        progressbars=False,
    )
    model_alpha_0 = ImageGenerator(alpha=0.0, **kwargs)
    assert model_alpha_0.aberration.specimen_absorption is False
    img_alpha_0 = model_alpha_0(torch.tensor([0]))
    img_alpha_1 = ImageGenerator(alpha=0.1, **kwargs)(torch.tensor([0]))
    assert not torch.allclose(img_alpha_0, img_alpha_1)


def test_image_generator_multislice_scattering_model_keeps_specimen_absorption():
    """scattering_model="multislice" already applies alpha upstream via
    potential.apply_amplitude_contrast -- Aberration's specimen_absorption must
    stay True (its default) so the amp-contrast offset isn't added again.
    aberration_model is derived from scattering_model (not independently
    settable), so this also confirms multislice -> "nonlinear"."""
    model = ImageGenerator(
        scattering_potential=torch.zeros(8, 8, 8),
        pixel_size=1.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params={"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])},
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="multislice",
        noise_model=None,
        ice_model=None,
        detector_model=None,
        alpha=0.1,
        progressbars=False,
    )
    assert model.aberration_model == "nonlinear"
    assert model.aberration.specimen_absorption is True


def test_aberration_rejects_per_image_ctf_params_as_constructor_kwargs():
    """dfu/cs/etc. genuinely vary per particle -- still not constructor args."""
    with pytest.raises(TypeError, match="dfu"):
        Aberration(16, 2.0, 300.0, dfu=5000.0)


def test_transfer_function_supports_batched_ctf_params():
    """Batched CTF parameters should broadcast over the Fourier grid."""
    aberration = Aberration(
        n_pixels=16,
        pixel_size=2.0,
        voltage=300.0,
        aberration_model="nonlinear",
    )

    transfer = aberration.transfer_function(
        {
            "dfu": torch.linspace(4000.0, 5000.0, 5),
            "dfv": torch.linspace(4100.0, 5100.0, 5),
            "dfang": torch.zeros(5),
        }
    )

    assert transfer.shape == (5, 16, 16)


def test_image_generator_plumbs_envelope_params(small_volume, ctf_params):
    """ImageGenerator forwards Cs/Cc/dose envelope params to its Aberration submodule."""
    gen = ImageGenerator(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        convergence_angle=0.02,
        cc=2.7e7,
        energy_spread=0.8,
        deltaV_V=0.05e-6,
        deltaI_I=0.02e-6,
        dose_envelope=True,
        verbose=False,
        progressbars=False,
    )
    assert gen.aberration.convergence_angle == 0.02
    assert gen.aberration.cc == 2.7e7
    assert gen.aberration.energy_spread == 0.8
    assert gen.aberration.deltaV_V == 0.05e-6
    assert gen.aberration.deltaI_I == 0.02e-6
    assert gen.aberration.dose_envelope is True


def test_image_generator_from_coordinates_plumbs_envelope_params(
    small_coords, ctf_params
):
    """ImageGeneratorFromCoordinates forwards Cs/Cc/dose envelope params to Aberration."""
    coords, atomic_numbers = small_coords
    gen = ImageGeneratorFromCoordinates(
        coordinates=coords,
        atomic_numbers=atomic_numbers,
        nxy=16,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        convergence_angle=0.02,
        cc=2.7e7,
        energy_spread=0.8,
        deltaV_V=0.05e-6,
        deltaI_I=0.02e-6,
        dose_envelope=True,
        verbose=False,
    )
    assert gen.aberration.convergence_angle == 0.02
    assert gen.aberration.cc == 2.7e7
    assert gen.aberration.energy_spread == 0.8
    assert gen.aberration.deltaV_V == 0.05e-6
    assert gen.aberration.deltaI_I == 0.02e-6
    assert gen.aberration.dose_envelope is True


def test_micrograph_generator_plumbs_envelope_params(small_volume, ctf_params):
    """MicrographGenerator forwards Cs/Cc/dose envelope params to its Aberration submodule."""
    gen = MicrographGenerator(
        scattering_potential=small_volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        convergence_angle=0.02,
        cc=2.7e7,
        energy_spread=0.8,
        deltaV_V=0.05e-6,
        deltaI_I=0.02e-6,
        dose_envelope=True,
        verbose=False,
        progressbars=False,
    )
    assert gen.aberration.convergence_angle == 0.02
    assert gen.aberration.cc == 2.7e7
    assert gen.aberration.energy_spread == 0.8
    assert gen.aberration.deltaV_V == 0.05e-6
    assert gen.aberration.deltaI_I == 0.02e-6
    assert gen.aberration.dose_envelope is True


def test_tilt_series_generator_plumbs_envelope_params(ctf_params):
    """TiltSeriesGenerator forwards Cs/Cc/dose envelope params to its Aberration submodule."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([-10.0, 0.0, 10.0])

    gen = TiltSeriesGenerator(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=angles,
        noise_model=None,
        scattering_model="projection",
        convergence_angle=0.02,
        cc=2.7e7,
        energy_spread=0.8,
        deltaV_V=0.05e-6,
        deltaI_I=0.02e-6,
        dose_envelope=True,
        verbose=False,
        progressbars=False,
    )
    assert gen.aberration.convergence_angle == 0.02
    assert gen.aberration.cc == 2.7e7
    assert gen.aberration.energy_spread == 0.8
    assert gen.aberration.deltaV_V == 0.05e-6
    assert gen.aberration.deltaI_I == 0.02e-6
    assert gen.aberration.dose_envelope is True


def test_image_generator_dose_envelope_changes_output(small_volume, ctf_params):
    """Enabling the dose envelope changes the simulated image given non-trivial dose."""
    kwargs = dict(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=40.0,
        noise_model=None,
        scattering_model="projection",
        alpha=0.1,
        verbose=False,
        progressbars=False,
    )

    gen_off = ImageGenerator(**kwargs, dose_envelope=False)
    gen_on = ImageGenerator(**kwargs, dose_envelope=True)

    image_off = gen_off(torch.tensor([0]))
    image_on = gen_on(torch.tensor([0]))

    assert not torch.allclose(image_off, image_on)


def test_image_generator_bfactor_none_and_zero_match(small_volume, ctf_params):
    """Providing None or 0.0 preserves the existing generator behavior."""
    kwargs = dict(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        alpha=0.1,
        verbose=False,
        progressbars=False,
    )

    gen_none = ImageGenerator(**kwargs, bfactor=None)
    gen_zero = ImageGenerator(**kwargs, bfactor=0.0)

    image_none = gen_none(torch.tensor([0]))
    image_zero = gen_zero(torch.tensor([0]))

    assert torch.equal(image_none, image_zero)


def test_micrograph_generator_potential_scale_changes_output(small_volume, ctf_params):
    """potential_scale multiplies the volume before scattering, so it must change output.

    Regression guard: previously accepted and stored as a buffer but never applied to the
    volume in MicrographGenerator.forward(), silently a no-op.
    """
    kwargs = dict(
        scattering_potential=None,
        volume=small_volume.unsqueeze(0),
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        verbose=False,
        progressbars=False,
    )

    gen_default = MicrographGenerator(**kwargs)
    gen_scaled = MicrographGenerator(**kwargs, potential_scale=2.0)

    image_default = gen_default(torch.tensor([0]))
    image_scaled = gen_scaled(torch.tensor([0]))

    assert not torch.allclose(image_default, image_scaled)


def test_tilt_series_generator_potential_scale_changes_output(ctf_params):
    """Same regression guard as above, for TiltSeriesGenerator.generate_tilt_series()."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([-10.0, 0.0, 10.0])
    kwargs = dict(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=angles,
        noise_model=None,
        scattering_model="projection",
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )

    gen_default = TiltSeriesGenerator(**kwargs)
    gen_scaled = TiltSeriesGenerator(**kwargs, potential_scale=2.0)

    tilt_series_default, _, _ = gen_default.generate_tilt_series(torch.tensor([0]))
    tilt_series_scaled, _, _ = gen_scaled.generate_tilt_series(torch.tensor([0]))

    assert not torch.allclose(tilt_series_default, tilt_series_scaled)


def test_ctf_params_dict_undoes_multislice_defocus_shift(small_volume, ctf_params):
    """
    Multislice evaluates the CTF at the volume's midplane, which requires
    internally shifting dfu/dfv by half the volume's Z extent (see
    _apply_defocus_shift). ctf_params_dict() is the external-facing accessor
    (used e.g. by create_particle_starfile_from_model) and must undo that
    shift, returning the original CryoSPARC/RELION-convention defocus rather
    than the internally-adjusted one.
    """
    original_dfu = ctf_params["dfu"].clone()
    gen = ImageGenerator(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="multislice",
        alpha=0.1,
        verbose=False,
        progressbars=False,
    )

    expected_shift = gen.nz * gen.pixel_size / 2
    assert expected_shift > 0
    assert torch.allclose(gen.dfu, original_dfu - expected_shift)
    assert torch.allclose(gen.ctf_params_dict()["dfu"], original_dfu)


def test_tilt_series_generator_edge_margin_pads_beyond_geometric_minimum(ctf_params):
    """edge_margin adds real interpolation slack beyond the exact geometric minimum.

    Regression guard for a tilt-angle artifact: required_nxy (the geometric minimum
    computed by _estimate_required_nxy) leaves zero slack, so output pixels at the
    edge of the crop have a per-slice sampling footprint that lands exactly on the
    padded-volume boundary at the deepest Z, with no room for interpolation. That
    produced a real artifact -- a bright line through the origin in the FFT, aligned
    with the tilt axis (see dev/tilt series/diag_line_source.png). edge_margin fixes
    it by always padding beyond the geometric minimum, independent of taper_width
    (which only fades pixels beyond required_nxy that are provably never sampled --
    verified to have zero effect on output, see dev/tilt series/ investigation).
    """
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([45.0])
    kwargs = dict(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=angles,
        noise_model=None,
        scattering_model="projection",
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )

    gen_no_margin = TiltSeriesGenerator(**kwargs, edge_margin=0)
    gen_default = TiltSeriesGenerator(**kwargs)  # edge_margin=8 by default

    assert gen_default.volume.shape[-1] == gen_no_margin.volume.shape[-1] + 2 * 8

    tilt_series_no_margin, _, _ = gen_no_margin.generate_tilt_series(torch.tensor([0]))
    tilt_series_default, _, _ = gen_default.generate_tilt_series(torch.tensor([0]))
    assert not torch.allclose(tilt_series_no_margin, tilt_series_default)


def test_tilt_series_generator_pad_fft_multislice_shapes_match(ctf_params):
    """pad_fft=True must not change output shapes, and must not crash -- at any tilt.

    Regression guard for a tilt-angle artifact distinct from edge_margin's:
    multislice's per-slice Fresnel propagation is an FFT-based convolution that is
    periodic over whatever canvas it runs on. Propagating at exactly the output size
    gives it zero headroom; under tilt (hundreds to 1000+ multislice steps), the
    resulting per-step wraparound compounds coherently into a real, visible artifact
    along all four frame edges (confirmed via direct comparison against
    scattering_model="projection", which cannot exhibit this artifact by
    construction, at both toy and production scale -- see
    dev/tilt series/verify_recursion_pad*.py).

    pad_fft=True fixes this inside IterativeScattering.multislice: pad once before
    the whole nz_new-step recursion, run the entire recursion at the padded canvas,
    crop back to the true output size once at the end. self.nxy (and
    self.iterative_scattering.nxy) are always the true output size -- there is no
    more separate pad_nxy concept for TiltSeriesGenerator, and self.aberration is
    never rebuilt.

    This also covers the exact scenario that used to crash: the identity/no-rotation
    (0deg) path previously did raw tensor indexing with no bounds handling, and the
    padded canvas wasn't guaranteed to fit inside the edge_margin-provisioned volume
    -- a RuntimeError shape mismatch. 0deg is included below specifically to guard
    against regressing that.
    """
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    kwargs = dict(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="multislice",
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )

    for angle in (0.0, 45.0):
        gen = TiltSeriesGenerator(**kwargs, angles=torch.tensor([angle]), pad_fft=True)
        assert gen.iterative_scattering.nxy == gen.nxy == gen.pad_nxy

        tilt_series, exitwaves, clean_images = gen.generate_tilt_series(
            torch.tensor([0])
        )
        for tensor in (tilt_series, exitwaves, clean_images):
            assert tensor.shape[-2:] == (32, 32)


def test_tilt_series_generator_pad_fft_changes_output_under_tilt(ctf_params):
    """pad_fft=True materially changes multislice output at a non-trivial tilt.

    Companion to the shapes test above: confirms pad_fft is not a silent no-op at
    45deg (where the artifact it fixes is actually present), while leaving 0deg
    numerically close to unchanged (no tilt-induced wraparound to fix there).
    """
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    kwargs = dict(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="multislice",
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )

    gen_base = TiltSeriesGenerator(**kwargs, angles=torch.tensor([45.0]), pad_fft=False)
    gen_pad = TiltSeriesGenerator(**kwargs, angles=torch.tensor([45.0]), pad_fft=True)
    _, exitwaves_base, _ = gen_base.generate_tilt_series(torch.tensor([0]))
    _, exitwaves_pad, _ = gen_pad.generate_tilt_series(torch.tensor([0]))
    assert not torch.allclose(exitwaves_base, exitwaves_pad)


def test_tilt_series_generator_volume_is_never_a_registered_buffer(ctf_params):
    """`volume` must never be in the module's buffer registry -- otherwise a
    later `.to(device)` on the whole module (the CLI pipelines' pattern)
    would unconditionally drag it onto the compute device regardless of
    whether it fits, before `generate_tilt_series` gets a say. Pure
    bookkeeping check, doesn't need CUDA."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    gen = TiltSeriesGenerator(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=torch.tensor([30.0]),
        verbose=False,
        progressbars=False,
    )
    assert "volume" not in dict(gen.named_buffers())
    assert gen.volume.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tilt_series_generator_auto_places_volume_that_fits(ctf_params):
    """A volume that comfortably fits should end up on the compute device
    automatically on first use -- no flag needed for the common case."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    gen = TiltSeriesGenerator(
        volume=volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=torch.tensor([30.0]),
        scattering_model="multislice",
        verbose=False,
        progressbars=False,
    ).to("cuda")
    # .to("cuda") alone must not have moved it -- only generate_tilt_series decides.
    assert gen.volume.device.type == "cpu"

    _, exitwaves, _ = gen.generate_tilt_series(torch.tensor([0]))
    assert gen.volume.device.type == "cuda"
    assert torch.isfinite(exitwaves).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tilt_series_generator_falls_back_to_windowed_when_volume_does_not_fit(
    ctf_params, monkeypatch
):
    """When moving `volume` to the compute device raises OOM, generation must
    still succeed by falling back to CPU-resident windowed fetches, matching
    the direct-GPU result. Simulates OOM deterministically (rather than
    requiring an actually huge volume) by intercepting `.to()` calls on this
    specific tensor."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    kwargs = dict(
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        angles=torch.tensor([30.0]),
        scattering_model="multislice",
        noise_model=None,
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )

    gen_gpu = TiltSeriesGenerator(volume=volume.clone(), **kwargs).to("cuda")
    _, exitwaves_gpu, _ = gen_gpu.generate_tilt_series(torch.tensor([0]))

    gen_streamed = TiltSeriesGenerator(volume=volume.clone(), **kwargs).to("cuda")
    assert gen_streamed.volume.device.type == "cpu"

    original_to = torch.Tensor.to

    def fake_to(tensor_self, *args, **kw):
        if tensor_self is gen_streamed.volume:
            raise torch.cuda.OutOfMemoryError("simulated OOM for test")
        return original_to(tensor_self, *args, **kw)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    torch.cuda.reset_peak_memory_stats()
    _, exitwaves_streamed, _ = gen_streamed.generate_tilt_series(torch.tensor([0]))
    peak_bytes = torch.cuda.max_memory_allocated()

    assert gen_streamed.volume.device.type == "cpu"  # fallback kept it on CPU
    assert torch.allclose(exitwaves_gpu.cpu(), exitwaves_streamed.cpu(), atol=1e-3)
    # the whole volume (16*48*48*4 bytes) would be ~147KB here -- tiny either
    # way, but the point is this must not scale with volume size, so just
    # sanity-check it's in the same ballpark as the windowed block, not
    # blown up by some accidental full-volume copy.
    assert peak_bytes < 50_000_000


# ---------------------------------------------------------------------------
# Translations must be applied in Angstrom, not in grid-normalized units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift_angstrom", [8.0, -6.0])
def test_from_coordinates_translation_is_in_angstrom(
    small_coords, ctf_params, shift_angstrom
):
    """
    A requested shift must move the atoms by that many Angstrom.

    `translations_angstrom_to_torch` normalizes to grid_sample's [-1, 1]
    convention, which is meaningless for atom positions: routing the
    translation through it shrank every shift by a factor of n * pixel_size / 2.
    """
    coords, atomic_numbers = small_coords
    gen = ImageGeneratorFromCoordinates(
        coordinates=coords,
        atomic_numbers=atomic_numbers,
        nxy=32,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0], [shift_angstrom, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        ice_model=None,
        verbose=False,
    )

    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    unshifted = gen.rotate(identity, torch.tensor([[0.0, 0.0]]))
    shifted = gen.rotate(identity, torch.tensor([[shift_angstrom, 0.0]]))

    # Translations are subtracted, and act along x only.
    delta = (shifted - unshifted).squeeze(0)
    expected = torch.tensor([-shift_angstrom, 0.0, 0.0])
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-5)


def test_from_coordinates_and_volume_paths_agree(small_coords, ctf_params):
    """
    The two generators must place a particle identically for the same pose.

    `ImageGeneratorFromCoordinates.rotate` transforms atoms; `ImageGenerator.
    rotate` transforms a prebuilt map through `build_affine_matrix`. They are
    separate implementations of one convention, and only this comparison
    exercises both public methods: the unit-level consistency test in
    `test_rotations.py` reimplements the coordinate transform inline, so it
    cannot see a regression inside either `rotate()`.
    """
    coords, atomic_numbers = small_coords
    n, voxel_size, sigma = 32, 1.0, 2.0

    axis = torch.arange(n, dtype=torch.float32) - n // 2
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")

    def voxelize(points: torch.Tensor) -> torch.Tensor:
        volume = torch.zeros(n, n, n)
        for p in points:
            r2 = (
                (xx - p[0] / voxel_size) ** 2
                + (yy - p[1] / voxel_size) ** 2
                + (zz - p[2] / voxel_size) ** 2
            )
            volume += torch.exp(-r2 / (2 * sigma**2))
        return volume

    quaternion = roma.random_unitquat(1)
    translation = torch.tensor([[4.0, -2.5]])
    common = dict(
        pixel_size=voxel_size,
        quaternions=quaternion,
        translations=translation,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model=None,
        scattering_model="projection",
        ice_model=None,
        verbose=False,
    )

    from_coords = ImageGeneratorFromCoordinates(
        coordinates=coords, atomic_numbers=atomic_numbers, nxy=n, **common
    )
    from_volume = ImageGenerator(scattering_potential=voxelize(coords), **common)

    volume_a = voxelize(from_coords.rotate(quaternion, translation).squeeze(0))
    volume_b = from_volume.rotate(quaternion, translation)[0]

    peak = volume_a.max()
    diff = (volume_a - volume_b).abs()
    assert diff.max() < 0.05 * peak
    assert diff.mean() < 0.002 * peak


def test_micrograph_volume_falls_back_to_host_when_it_does_not_fit(
    small_volume, ctf_params, monkeypatch
):
    """
    A specimen volume too large for the device streams from the host instead.

    `MicrographSpecimenGenerator` assembles with `move_to_cpu=True`, so the
    volume starts on the host and `forward` uploads it. At `micrograph_size`
    that upload can exceed the device, and it used to be unconditional -- which
    OOM'd the very device `move_to_cpu` had just moved the volume off.
    `IterativeScattering` accepts an off-device volume, so the fallback costs
    per-slice transfers rather than the run.

    The OOM is injected rather than provoked: reproducing it for real needs a
    volume larger than whatever GPU the suite happens to run on, and this must
    also exercise on CPU-only machines.
    """
    gen = MicrographGenerator(
        scattering_potential=small_volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        noise_model="none",
        scattering_model="projection",
        ice_model="random",
        alpha=0.1,
        crowd_min_distance=60.0,
        verbose=False,
        progressbars=False,
    )

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

    stub = _VolumeThatWontFit()
    gen.__dict__["volume"] = stub
    # `device` is a read-only Lightning property, so patch it on the class.
    monkeypatch.setattr(
        type(gen), "device", property(lambda self: torch.device("cuda:0"))
    )

    with pytest.warns(UserWarning, match="does not fit"):
        gen._ensure_volume_placed()

    assert calls["n"] == 1
    assert gen.volume is stub, "volume must stay on the host after a failed upload"

    # Warned once, not once per micrograph in the run.
    gen._ensure_volume_placed()
    assert calls["n"] == 2
    assert gen._warned_volume_on_host is True


def test_micrograph_unit_potential_scale_skips_the_extra_volume_copy(
    small_volume, ctf_params
):
    """
    `potential_scale=1` must not materialise a second copy of the volume.

    `V * scale` allocates another tensor the size of the whole specimen -- 33.5
    GB at the default config, which is what pushed a 4096-pixel micrograph over
    a 44 GB card even though the volume itself fits. Scaling by 1 is the
    default and is a no-op, so it is skipped; this pins that skipping it does
    not change the image.
    """

    def _run(scale):
        torch.manual_seed(0)
        gen = MicrographGenerator(
            scattering_potential=small_volume,
            micrograph_size=32,
            pixel_size=2.0,
            ctf_params=ctf_params,
            voltage=300.0,
            dose_per_angstrom=2.0,
            noise_model="none",
            scattering_model="projection",
            ice_model="random",
            alpha=0.1,
            crowd_min_distance=60.0,
            potential_scale=scale,
            verbose=False,
            progressbars=False,
        )
        torch.manual_seed(0)
        return gen(torch.tensor([0]))

    assert torch.allclose(_run(1.0), _run(1.0000001), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# state_dict round-trip
#
# `Aberration` builds its frequency grids with `torch.meshgrid`, which returns
# stride-0 broadcast views. Registered as persistent buffers, those cannot be
# restored: `load_state_dict` copies into a buffer in place, and writing
# through a stride whose elements alias one address is refused. They are pure
# functions of the constructor arguments, so they are not persisted at all.
# ---------------------------------------------------------------------------


def _plain_generator(volume, ctf_params):
    """A minimal ImageGenerator: no ice, no noise, no detector."""
    return ImageGenerator(
        scattering_potential=volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        scattering_model="multislice",
        alpha=0.1,
        verbose=False,
        progressbars=False,
    )


def test_aberration_frequency_grids_are_not_persisted(small_volume, ctf_params):
    """Derived frequency grids follow the module across devices without being state."""
    gen = _plain_generator(small_volume, ctf_params)
    buffers = gen.aberration._buffers
    persisted = set(gen.aberration.state_dict())
    for name in ("k", "k2", "radian", "KY", "KX"):
        assert name in buffers, f"{name} must stay a buffer so it follows the device"
        assert name not in persisted, f"{name} is derived and must not be persisted"


def test_image_generator_state_dict_round_trips(small_volume, ctf_params):
    """A model's own state_dict loads back into an identical model."""
    source = _plain_generator(small_volume, ctf_params)
    target = _plain_generator(small_volume, ctf_params)
    target.load_state_dict(source.state_dict(), strict=True)

    torch.manual_seed(0)
    expected = source.forward(torch.tensor([0]))
    torch.manual_seed(0)
    actual = target.forward(torch.tensor([0]))
    assert torch.allclose(expected, actual)
