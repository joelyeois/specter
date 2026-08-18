"""End-to-end parity between aberration_backend="legacy" (aberrations.
Aberration) and aberration_backend="torch_ctf" (ctf.LegacyAberrationAdapter),
through Reconstructor and TomogramReconstructor -- the ghostbuster
counterpart to test_imagegenerator_backend_parity.py, which covers the
forward-model generators directly.
"""

from __future__ import annotations

import pytest
import roma
import torch

from specter.ghostbuster import Reconstructor, TomogramReconstructor

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


@pytest.fixture
def small_volume() -> torch.Tensor:
    volume = torch.zeros(16, 16, 16)
    volume[4:12, 4:12, 4:12] = 50.0
    return volume


@pytest.fixture
def realistic_ctf_params() -> dict[str, torch.Tensor]:
    """Every CTF term nonzero at once, matching
    test_imagegenerator_backend_parity.py's fixture of the same name."""
    return {
        "dfu": torch.tensor([5200.0]),
        "dfv": torch.tensor([4900.0]),
        "dfang": torch.tensor([12.0]),
        "cs": torch.tensor([2.7]),
        "phaseshift": torch.tensor([0.1]),
        "tiltx": torch.tensor([5e-4]),
        "tilty": torch.tensor([-2e-4]),
        "trefoil1": torch.tensor([0.3]),
        "trefoil2": torch.tensor([-0.15]),
    }


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------


def _build_reconstructor(small_volume, ctf_params, aberration_backend, **extra):
    torch.manual_seed(0)
    model = Reconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 0.0]]),
        ctf_params=ctf_params,
        voltage=300.0,
        dose_per_angstrom=2.0,
        alpha=0.1,
        scattering_model="projection",
        aberration_backend=aberration_backend,
        **extra,
    )
    torch.manual_seed(0)
    return model.forward(torch.tensor([0]))


def test_reconstructor_matches_across_backends(small_volume, realistic_ctf_params):
    """Every CTF term nonzero at once, through Reconstructor's actual forward
    pass (ImageGenerator built internally) -- not just the isolated
    ImageGenerator parity already covered elsewhere."""
    legacy = _build_reconstructor(small_volume, realistic_ctf_params, "legacy")
    torch_ctf = _build_reconstructor(small_volume, realistic_ctf_params, "torch_ctf")

    assert legacy.shape == torch_ctf.shape
    assert torch.allclose(legacy, torch_ctf, atol=1e-3)


def test_reconstructor_lpp_params_changes_output(small_volume):
    """lpp_params reaches the rendered image through Reconstructor, same as
    it does through a bare ImageGenerator."""
    ctf_params = {"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])}

    without_lpp = _build_reconstructor(
        small_volume, ctf_params, "torch_ctf", lpp_params=None
    )
    with_lpp = _build_reconstructor(
        small_volume, ctf_params, "torch_ctf", lpp_params=_LPP_PARAMS
    )

    assert without_lpp.shape == with_lpp.shape
    assert not torch.allclose(without_lpp, with_lpp)


def test_reconstructor_lpp_params_with_legacy_backend_raises(small_volume):
    """lpp_params with the default aberration_backend='legacy' must raise at
    construction time, same as it does for a bare ImageGenerator."""
    with pytest.raises(ValueError, match="lpp_params"):
        Reconstructor(
            V=small_volume,
            voxel_size=2.0,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            translations=torch.tensor([[0.0, 0.0]]),
            ctf_params={"dfu": torch.tensor([5000.0]), "cs": torch.tensor([2.7])},
            voltage=300.0,
            dose_per_angstrom=2.0,
            scattering_model="projection",
            lpp_params=_LPP_PARAMS,
        )


# ---------------------------------------------------------------------------
# TomogramReconstructor
# ---------------------------------------------------------------------------


@pytest.fixture
def tilt_quaternions() -> torch.Tensor:
    angles_deg = torch.tensor([-10.0, 0.0, 10.0])
    theta = torch.deg2rad(angles_deg)
    rotvecs = torch.stack(
        [theta, torch.zeros_like(theta), torch.zeros_like(theta)], dim=-1
    )
    return roma.rotvec_to_unitquat(rotvecs)


def _build_tomogram_reconstructor(
    small_volume, tilt_quaternions, ctf_params, aberration_backend, **extra
):
    torch.manual_seed(0)
    model = TomogramReconstructor(
        V=small_volume,
        voxel_size=2.0,
        quaternions=tilt_quaternions,
        translations=torch.zeros(3, 2),
        ctf_params=ctf_params,
        voltage=300.0,
        scattering_model="projection",
        aberration_backend=aberration_backend,
        **extra,
    )
    torch.manual_seed(0)
    return model.forward(0)


def test_tomogram_reconstructor_matches_across_backends(small_volume, tilt_quaternions):
    ctf_params = {
        "dfu": torch.full((3,), 5000.0),
        "cs": torch.full((3,), 2.7),
        "phaseshift": torch.full((3,), 0.1),
    }

    legacy = _build_tomogram_reconstructor(
        small_volume, tilt_quaternions, ctf_params, "legacy"
    )
    torch_ctf = _build_tomogram_reconstructor(
        small_volume, tilt_quaternions, ctf_params, "torch_ctf"
    )

    assert legacy.shape == torch_ctf.shape
    assert torch.allclose(legacy, torch_ctf, atol=1e-3)


def test_tomogram_reconstructor_lpp_params_with_legacy_backend_raises(
    small_volume, tilt_quaternions
):
    with pytest.raises(ValueError, match="lpp_params"):
        TomogramReconstructor(
            V=small_volume,
            voxel_size=2.0,
            quaternions=tilt_quaternions,
            translations=torch.zeros(3, 2),
            ctf_params={"dfu": torch.full((3,), 5000.0), "cs": torch.full((3,), 2.7)},
            voltage=300.0,
            scattering_model="projection",
            lpp_params=_LPP_PARAMS,
        )
