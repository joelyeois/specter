"""End-to-end parity between aberration_backend="legacy" (aberrations.
Aberration) and aberration_backend="torch_ctf" (ctf.LegacyAberrationAdapter),
through the actual ImageGenerator forward pass -- not just the isolated
transfer-function classes tested elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from specter.imagegenerator import (
    ImageGenerator,
    MicrographGenerator,
    TiltSeriesGenerator,
)
from specter.settings import Camera, Optics, Propagation, TiltGeometry


@pytest.fixture
def realistic_ctf_params():
    """Every CTF term nonzero at once -- same values used by the
    ImageGenerator parity test above, single particle (matches how
    test_generators.py's MicrographGenerator/TiltSeriesGenerator
    regression fixtures call gen(torch.tensor([0]))/
    generate_tilt_series(torch.tensor([0]))."""
    return {
        "dfu": torch.tensor([5200.0]),
        "dfv": torch.tensor([4900.0]),
        "dfang": torch.tensor([12.0]),
        "cs": torch.tensor([2.7e7]),
        "phaseshift": torch.tensor([0.1]),
        "tiltx": torch.tensor([5e-4]),
        "tilty": torch.tensor([-2e-4]),
        "trefoil1": torch.tensor([0.3]),
        "trefoil2": torch.tensor([-0.15]),
    }


def _build(small_volume, ctf_params, aberration_backend, seed=0):
    torch.manual_seed(seed)
    gen = ImageGenerator(
        scattering_potential=small_volume,
        pixel_size=2.0,
        quaternions=torch.tensor([[0.7071, 0.0, 0.7071, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[1.0, -1.0], [0.0, 0.5]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        ice_model=None,
        verbose=False,
        progressbars=False,
        propagation=Propagation(scattering_model="multislice", alpha=0.1),
        optics=Optics(aberration_backend=aberration_backend),
        camera=Camera(noise_model=None),
    )
    torch.manual_seed(seed)
    return gen(torch.tensor([0, 1]))


def test_realistic_nonzero_ctf_params_match_across_backends(small_volume):
    """Every CTF term nonzero at once (defocus, astigmatism, Cs, trefoil,
    beam tilt, phase shift), two particles -- exercises the full
    conversion path, not just defocus/Cs like the plain regression
    fixture's all-zero ctf_params does."""
    ctf_params = {
        "dfu": torch.tensor([5200.0, 4800.0]),
        "dfv": torch.tensor([4900.0, 4600.0]),
        "dfang": torch.tensor([12.0, -30.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
        "phaseshift": torch.tensor([0.1, 0.0]),
        "tiltx": torch.tensor([5e-4, -3e-4]),
        "tilty": torch.tensor([-2e-4, 4e-4]),
        "trefoil1": torch.tensor([0.3, -0.2]),
        "trefoil2": torch.tensor([-0.15, 0.25]),
    }

    legacy_images = _build(small_volume, ctf_params, "legacy")
    torch_ctf_images = _build(small_volume, ctf_params, "torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


def test_ctf_scattering_model_selects_linear_aberration_model_both_backends(
    small_volume,
):
    """scattering_model="ctf" implies aberration_model="linear" (real-valued
    path), not just the default "nonlinear" complex path -- see
    aberrations.aberration_model_for_scattering -- and both backends must
    produce a real (non-complex), correctly-shaped image from it.

    This does NOT assert legacy/torch_ctf numeric parity, unlike the other
    tests in this file: end-to-end scattering_model="ctf" is not covered by
    test_ctf_legacy_adapter.py's aberration_model="linear" parity test
    either (that one only exercises specimen_absorption=True, the
    scattering_model != "ctf" case), and the two backends were found to
    disagree well outside float32 noise once specimen_absorption=False is
    actually exercised end-to-end -- a real, pre-existing gap in the
    torch_ctf backend, not something this test should paper over."""
    torch.manual_seed(0)
    ctf_params = {
        "dfu": torch.tensor([5000.0, 5000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    def build(backend):
        torch.manual_seed(0)
        gen = ImageGenerator(
            scattering_potential=small_volume,
            pixel_size=2.0,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            ctf_params=ctf_params,
            voltage=300.0,
            dose_per_angstrom=2.0,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="ctf", alpha=0.0),
            optics=Optics(aberration_backend=backend),
            camera=Camera(noise_model=None),
        )
        assert gen.aberration_model == "linear"
        torch.manual_seed(0)
        return gen(torch.tensor([0, 1]))

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")
    assert not legacy_images.is_complex()
    assert not torch_ctf_images.is_complex()
    assert legacy_images.shape == torch_ctf_images.shape


@pytest.mark.skipif(
    not __import__("os").path.exists(
        "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
    ),
    reason="real .cs file not available",
)
def test_real_csfile_particles_match_across_backends_end_to_end(small_volume):
    """First 5 real particles from the same .cs file used throughout this
    migration, through the actual ImageGenerator forward pass end to end
    -- the strongest available validation that aberration_backend
    switches nothing but which engine computes the transfer function."""
    from specter.io import extract_parameters_from_csfile

    CS_PATH = (
        "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
    )
    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_angstrom,
        ctf_params,
        scale,
        anisomag,
        indices,
        split,
    ) = extract_parameters_from_csfile(CS_PATH, halfset="all", n_particles=5)

    def build(backend):
        torch.manual_seed(0)
        gen = ImageGenerator(
            scattering_potential=small_volume,
            pixel_size=2.0,
            quaternions=rotations,
            translations=torch.zeros(5, 2),
            ctf_params=ctf_params,
            voltage=float(voltage_kv),
            dose_per_angstrom=2.0,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="multislice", alpha=float(alpha)),
            optics=Optics(aberration_backend=backend),
            camera=Camera(noise_model=None),
        )
        torch.manual_seed(0)
        return gen(torch.arange(5))

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape == (5, 32, 32)
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


def test_micrograph_generator_matches_across_backends(
    small_volume, realistic_ctf_params
):
    """Same scattering_potential/micrograph_size/pixel_size/projection-model
    setup as test_generators.py::test_micrograph_generator_regression, with
    nonzero realistic_ctf_params instead of that test's all-zero fixture so
    every CTF term is actually exercised."""

    def build(backend):
        torch.manual_seed(0)
        gen = MicrographGenerator(
            scattering_potential=small_volume,
            micrograph_size=32,
            pixel_size=2.0,
            ctf_params=realistic_ctf_params,
            voltage=300.0,
            dose_per_angstrom=2.0,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="projection", alpha=0.1),
            optics=Optics(aberration_backend=backend),
            camera=Camera(noise_model=None),
        )
        torch.manual_seed(0)
        return gen(torch.tensor([0]))

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


def test_tilt_series_generator_matches_across_backends(realistic_ctf_params):
    """Same volume/angles/tilt_axis='y'/projection-model setup as
    test_generators.py::test_tilt_series_generator_regression, with
    nonzero realistic_ctf_params instead of that test's all-zero fixture."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    angles = torch.tensor([-10.0, 0.0, 10.0])

    def build(backend):
        torch.manual_seed(0)
        gen = TiltSeriesGenerator(
            volume=volume.clone(),
            micrograph_size=32,
            pixel_size=2.0,
            ctf_params=realistic_ctf_params,
            voltage=300.0,
            dose_per_angstrom=2.0,
            angles=angles,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="projection", alpha=0.1),
            optics=Optics(aberration_backend=backend),
            camera=Camera(noise_model=None),
            tilt=TiltGeometry(tilt_axis="y"),
        )
        torch.manual_seed(0)
        tilt_series, _, _ = gen.generate_tilt_series(torch.tensor([0]))
        return tilt_series

    legacy_series = build("legacy")
    torch_ctf_series = build("torch_ctf")

    assert legacy_series.shape == torch_ctf_series.shape
    assert torch.allclose(legacy_series, torch_ctf_series, atol=1e-3)


_LPP_PARAMS = dict(
    NA=0.1,
    laser_wavelength_angstrom=10640.0,
    focal_length_angstrom=2e7,
    laser_xy_angle_deg=0.0,
    laser_xz_angle_deg=0.0,
    laser_long_offset_angstrom=0.0,
    laser_trans_offset_angstrom=0.0,
    laser_polarization_angle_deg=0.0,
    peak_phase_deg=90.0,
)


def test_lpp_params_end_to_end_through_image_generator(small_volume):
    """lpp_params, passed to ImageGenerator as a constructor kwarg (not a
    ctf_params dict key -- see ctf/_legacy.py), reaches the rendered image:
    a laser phase plate must change the output relative to the same run
    with no phase plate at all."""
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7e7])}

    def build(lpp_params):
        torch.manual_seed(0)
        gen = ImageGenerator(
            scattering_potential=small_volume,
            pixel_size=2.0,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 0.0]]),
            ctf_params=ctf_params,
            voltage=300.0,
            dose_per_angstrom=2.0,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="multislice", alpha=0.1),
            optics=Optics(aberration_backend="torch_ctf", lpp_params=lpp_params),
            camera=Camera(noise_model=None),
        )
        torch.manual_seed(0)
        return gen(torch.tensor([0]))

    without_lpp = build(None)
    with_lpp = build(_LPP_PARAMS)

    assert without_lpp.shape == with_lpp.shape
    assert not torch.allclose(without_lpp, with_lpp)


def test_lpp_params_with_legacy_backend_raises(small_volume):
    """aberrations.Aberration has no laser-phase-plate model -- lpp_params
    with the default aberration_backend='legacy' must raise at
    construction time, not silently have no effect."""
    with pytest.raises(ValueError, match="lpp_params"):
        ImageGenerator(
            scattering_potential=small_volume,
            pixel_size=2.0,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 0.0]]),
            ctf_params={"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7e7])},
            voltage=300.0,
            dose_per_angstrom=2.0,
            ice_model=None,
            verbose=False,
            progressbars=False,
            propagation=Propagation(scattering_model="multislice", alpha=0.1),
            optics=Optics(lpp_params=_LPP_PARAMS),
            camera=Camera(noise_model=None),
        )
