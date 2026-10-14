"""End-to-end parity between aberration_backend="legacy" (aberrations.
Aberration) and aberration_backend="torch_ctf" (ctf.LegacyAberrationAdapter),
through the whole image-formation chain (`specter.imaging`) -- not just the
isolated transfer-function classes tested elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from specter.imaging import CTF, Exposure, Field, Poses, Sample, Specimen
from specter.settings import Propagation, TiltGeometry


@pytest.fixture
def realistic_ctf_params():
    """Every CTF term nonzero at once, one image."""
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


def _particles(
    small_volume: torch.Tensor,
    ctf_params: dict,
    backend: str,
    *,
    scattering_model: str = "multislice",
    alpha: float = 0.1,
    lpp_params: dict | None = None,
    quaternions: torch.Tensor | None = None,
    shifts: torch.Tensor | None = None,
) -> torch.Tensor:
    n = len(next(iter(ctf_params.values())))
    if quaternions is None:
        quaternions = torch.tensor([[0.7071, 0.0, 0.7071, 0.0], [1.0, 0.0, 0.0, 0.0]])
    if shifts is None:
        shifts = torch.tensor([[1.0, -1.0], [0.0, 0.5]])
    poses = Poses(quaternions[:n], shifts[:n], torch.arange(n))
    with torch.no_grad():
        waves = Specimen(small_volume, 2.0).scatter(
            poses, voltage=300.0, scattering_model=scattering_model, alpha=alpha
        )
        waves = waves.apply_ctf(CTF(ctf_params), backend=backend, lpp_params=lpp_params)
        return waves.intensity().detect(Exposure(2.0), noise=None).array


def test_realistic_nonzero_ctf_params_match_across_backends(small_volume):
    """Every CTF term nonzero at once (defocus, astigmatism, Cs, trefoil,
    beam tilt, phase shift), two particles -- exercises the full
    conversion path, not just defocus/Cs."""
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

    legacy_images = _particles(small_volume, ctf_params, "legacy")
    torch_ctf_images = _particles(small_volume, ctf_params, "torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


def test_ctf_scattering_model_selects_linear_aberration_model_both_backends(
    small_volume,
):
    """scattering_model="ctf" implies the linear (real-valued) aberration
    model, not the default complex path -- see
    aberrations.aberration_model_for_scattering -- and both backends must
    produce a real, correctly-shaped image from it.

    This does NOT assert legacy/torch_ctf numeric parity, unlike the other
    tests in this file: the two backends were found to disagree well outside
    float32 noise once specimen_absorption=False is exercised end to end -- a
    real, pre-existing gap in the torch_ctf backend, not something this test
    should paper over."""
    ctf_params = {
        "dfu": torch.tensor([5000.0, 5000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)

    def build(backend: str) -> torch.Tensor:
        poses = Poses(identity, torch.zeros(2, 2), torch.arange(2))
        with torch.no_grad():
            waves = Specimen(small_volume, 2.0).scatter(
                poses, voltage=300.0, scattering_model="ctf", alpha=0.0
            )
            assert waves.linear
            waves = waves.apply_ctf(CTF(ctf_params), backend=backend)
            return waves.intensity().detect(Exposure(2.0), noise=None).array

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")
    assert not legacy_images.is_complex()
    assert not torch_ctf_images.is_complex()
    assert legacy_images.shape == torch_ctf_images.shape


def test_micrograph_matches_across_backends(small_volume, realistic_ctf_params):
    """A field of one template, projection model, every CTF term exercised."""

    def build(backend: str) -> torch.Tensor:
        sample = Field(small_volume, 2.0, 32).draw(
            voltage=300.0, scattering_model="projection", alpha=0.1
        )
        with torch.no_grad():
            waves = sample.scatter(
                voltage=300.0, scattering_model="projection", alpha=0.1
            )
            waves = waves.apply_ctf(CTF(realistic_ctf_params), backend=backend)
            return waves.intensity().detect(Exposure(2.0), noise=None).array

    legacy_images = build("legacy")
    torch_ctf_images = build("torch_ctf")

    assert legacy_images.shape == torch_ctf_images.shape
    assert torch.allclose(legacy_images, torch_ctf_images, atol=1e-3)


def test_tilt_series_matches_across_backends(realistic_ctf_params):
    """A three-tilt series about y, projection model, every CTF term."""
    volume = torch.zeros(1, 16, 48, 48)
    volume[0, 5:11, 20:28, 20:28] = 50.0
    angles = [-10.0, 0.0, 10.0]

    def build(backend: str) -> torch.Tensor:
        exposure = Exposure.tilt_series(2.0, 3)
        series = Sample.from_volume(volume.clone(), 2.0).tilt_series(
            Poses.tilt_series(angles, axis="y"),
            exposure,
            voltage=300.0,
            micrograph_size=32,
            tilt=TiltGeometry(tilt_axis="y"),
            propagation=Propagation(scattering_model="projection", alpha=0.1),
            log=False,
        )
        images = []
        with torch.no_grad():
            for i in range(3):
                waves = series.scatter(i).apply_ctf(
                    CTF(realistic_ctf_params), backend=backend
                )
                images.append(
                    waves.intensity().detect(exposure[i : i + 1], noise=None).array
                )
        return torch.cat(images)

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


def test_lpp_params_end_to_end_through_the_chain(small_volume):
    """lpp_params, given at apply_ctf (not as a ctf_params dict key -- see
    ctf/_legacy.py), reaches the rendered image: a laser phase plate must
    change the output relative to the same run with no phase plate at all."""
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7e7])}
    one = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def build(lpp_params: dict | None) -> torch.Tensor:
        return _particles(
            small_volume,
            ctf_params,
            "torch_ctf",
            lpp_params=lpp_params,
            quaternions=one,
            shifts=torch.zeros(1, 2),
        )

    without_lpp = build(None)
    with_lpp = build(_LPP_PARAMS)

    assert without_lpp.shape == with_lpp.shape
    assert not torch.allclose(without_lpp, with_lpp)


def test_lpp_params_with_legacy_backend_raises(small_volume):
    """aberrations.Aberration has no laser-phase-plate model -- lpp_params
    with the default backend='legacy' must raise, not silently have no
    effect."""
    poses = Poses(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 2), torch.arange(1)
    )
    waves = Specimen(small_volume, 2.0).scatter(poses, voltage=300.0, alpha=0.1)
    ctf = CTF({"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7e7])})
    with pytest.raises(ValueError, match="lpp_params"):
        waves.apply_ctf(ctf, lpp_params=_LPP_PARAMS)
