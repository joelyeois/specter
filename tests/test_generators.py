"""
Golden-output regression tests for all four image generators.

On the first run, each test saves its output as a fixture under
tests/test_data/. Subsequent runs load the fixture and assert that
the output is numerically identical. To regenerate a fixture, delete
the corresponding .pt file and re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specter.aberrations import Aberration
from specter.imagegenerator import (
    ImageGenerator,
    ImageGeneratorFromCoordinates,
    MicrographGenerator,
    TiltSeriesGenerator,
)

FIXTURE_DIR = Path(__file__).parent / "test_data"


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
    vol = torch.zeros(32, 32, 32)
    vol[12:20, 12:20, 12:20] = 50.0
    return vol


@pytest.fixture
def small_volume_4d():
    """4D cubic volume (1, 32, 32, 32) as returned by TomogramGenerator."""
    vol = torch.zeros(1, 32, 32, 32)
    vol[0, 12:20, 12:20, 12:20] = 50.0
    return vol


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
        energy=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="multislice",
        ice_model="random",
        alpha=0.1,
        coincidence_radius=1.8,
        num_frames=10,
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
        energy=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="multislice",
        ice_model="random",
        alpha=0.1,
        coincidence_radius=1.8,
        num_frames=10,
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
        energy=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="projection",
        ice_model="random",
        alpha=0.1,
        coincidence_radius=1.8,
        num_frames=10,
        crowd_min_distance=60.0,
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    images = gen(torch.tensor([0]))
    _save_or_compare("micrograph_generator", images.cpu())


def test_micrograph_generator_accepts_prebuilt_icemaker(small_volume, ctf_params):
    """A pre-built IceBank passed via icemaker= is reused, not rebuilt internally."""
    from specter.ice import IceBank

    bank = IceBank(dx=2.0, n=32, method="random", num_unique=1)
    bank.build()

    gen = MicrographGenerator(
        scattering_potential=small_volume,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        energy=300.0,
        dose_per_angstrom=2.0,
        noise_model="poisson",
        scattering_model="projection",
        icemaker=bank,
        alpha=0.1,
        coincidence_radius=1.8,
        num_frames=10,
        verbose=False,
        progressbars=False,
    )
    assert gen.specimen_gen.icemaker is bank
    images = gen(torch.tensor([0]))
    assert images.shape == (1, 32, 32)


def test_tilt_series_generator_regression(ctf_params):
    """TiltSeriesGenerator: 3-angle tilt series, coincidence loss, tilt_axis='y'."""
    vol = torch.zeros(1, 16, 48, 48)
    vol[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([-10.0, 0.0, 10.0])

    torch.manual_seed(0)
    gen = TiltSeriesGenerator(
        vol=vol,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        energy=300.0,
        dose_per_angstrom=2.0,
        angles=angles,
        noise_model="poisson",
        scattering_model="projection",
        alpha=0.1,
        coincidence_radius=1.8,
        num_frames=10,
        tilt_axis="y",
        verbose=False,
        progressbars=False,
    )
    torch.manual_seed(0)
    tilt_series, _, _ = gen.generate_tilt_series(torch.tensor([0]))
    _save_or_compare("tilt_series_generator", tilt_series.cpu())


def test_bfactor_damps_transfer_function():
    """B-factor envelope is a Fourier-space transfer-function damping term."""
    aberration = Aberration(
        n_pixels=16,
        pixel_size=2.0,
        energy=300.0,
        aberration_model="holography",
    )

    no_envelope = aberration.transfer_function({})
    zero_envelope = aberration.transfer_function({"bfactor": torch.tensor([0.0])})
    damped = aberration.transfer_function({"bfactor": torch.tensor([80.0])})

    assert torch.equal(no_envelope, zero_envelope)
    assert torch.equal(damped[:, 0, 0], no_envelope[:, 0, 0])
    assert (
        torch.abs(damped[:, 1:, 1:]).mean() < torch.abs(no_envelope[:, 1:, 1:]).mean()
    )


def test_transfer_function_supports_batched_ctf_params():
    """Batched CTF parameters should broadcast over the Fourier grid."""
    aberration = Aberration(
        n_pixels=16,
        pixel_size=2.0,
        energy=300.0,
        aberration_model="holography",
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
        energy=300.0,
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
        energy=300.0,
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
        energy=300.0,
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
    vol = torch.zeros(1, 16, 48, 48)
    vol[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([-10.0, 0.0, 10.0])

    gen = TiltSeriesGenerator(
        vol=vol,
        micrograph_size=32,
        pixel_size=2.0,
        ctf_params=ctf_params,
        energy=300.0,
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
        energy=300.0,
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
        energy=300.0,
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
