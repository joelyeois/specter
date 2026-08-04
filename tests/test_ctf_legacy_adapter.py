"""Parity tests for ctf._legacy: the compatibility bridge from specter's
legacy Angstrom-based ctf_params dict convention to the torch-ctf-backed
CTFParameters/TransferFunction.

Every conversion used here (dfu/dfv -> defocus/astigmatism, cs -> mm,
phaseshift -> degrees, trefoil1/trefoil2 -> Z33c/Z33s, tiltx/tilty ->
Z31c/Z31s) is already verified formula-by-formula against
aberrations.Aberration in test_ctf_transfer.py -- these tests check the
*bridge* (dict handling, optional keys, dose/bfactor plumbing, real
multi-particle .cs data end to end through the exact dict-based call
signature aberrations.Aberration.forward() uses), not the underlying
physics again.
"""

from __future__ import annotations

import pytest
import torch

from specter.aberrations import Aberration
from specter.ctf import LegacyAberrationAdapter

N_PIXELS = 16
PIXEL_SIZE = 1.5
VOLTAGE = 300.0


def _exitwave(n: int, seed: int, n_pixels: int = N_PIXELS) -> torch.Tensor:
    return torch.randn(
        n,
        n_pixels,
        n_pixels,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(seed),
    )


def test_full_ctf_params_dict_matches_old_aberration():
    exitwave = _exitwave(5, seed=0)
    ctf_params = {
        "dfu": torch.tensor([2883.8, 2882.4, 2918.7, 2938.5, 2937.4]),
        "dfv": torch.tensor([2812.3, 2811.0, 2847.4, 2867.3, 2866.3]),
        "dfang": torch.tensor([-40.05] * 5),
        "cs": torch.tensor([2.7e7] * 5),
        "tiltx": torch.tensor([-1.739e-4] * 5),
        "tilty": torch.tensor([-1.229e-4] * 5),
        "phaseshift": torch.tensor([0.0] * 5),
        "trefoil1": torch.tensor([-0.0629] * 5),
        "trefoil2": torch.tensor([0.2650] * 5),
    }

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography")
    old_out = old(exitwave, ctf_params)

    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography"
    )
    new_out = adapter(exitwave, ctf_params)

    assert old_out.shape == new_out.shape
    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_minimal_ctf_params_dict_matches_old_aberration():
    """Just dfu + cs -- matches how generate_tilt_series.py's minimal path
    constructs ctf_params (no astigmatism/beam-tilt/trefoil/phase-plate
    terms at all)."""
    exitwave = _exitwave(2, seed=1)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography")
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography"
    )
    new_out = adapter(exitwave, ctf_params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_empty_ctf_params_dict_matches_old_aberration():
    """No CTF terms at all -- matches BaseImager's ctf_params=None path
    (chi stays identically 0)."""
    exitwave = _exitwave(2, seed=2)
    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography")
    old_out = old(exitwave, {})
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography"
    )
    new_out = adapter(exitwave, {})

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_ctf_model_matches_old_aberration():
    exitwave = _exitwave(2, seed=3)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="ctf", alpha=0.0)
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="ctf"
    )
    new_out = adapter(exitwave, ctf_params)

    assert not old_out.is_complex()
    assert not new_out.is_complex()
    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_bfactor_matches_old_aberration():
    """bfactor is a LegacyAberrationAdapter *construction-time* argument
    (matching TransferFunction), not read per-call from the dict -- in
    practice ctf_params["bfactor"] is always the same constant value
    across every particle anyway (BaseImager expands a scalar bfactor to
    all n particles identically), so this is a no-op difference."""
    exitwave = _exitwave(2, seed=4)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography", bfactor=150.0
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography", bfactor=150.0
    )
    new_out = adapter(exitwave, ctf_params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_dose_envelope_matches_old_aberration():
    exitwave = _exitwave(2, seed=5)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
        "dose": torch.tensor([40.0, 45.0]),
    }

    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography", dose_envelope=True
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="holography", dose_envelope=True
    )
    new_out = adapter(exitwave, ctf_params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_convergence_angle_and_cc_envelopes_match_old_aberration():
    exitwave = _exitwave(2, seed=6)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    old = Aberration(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="holography",
        convergence_angle=1.0,
        cc=1.4e7,
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="holography",
        convergence_angle=1.0,
        cc=1.4e7,
    )
    new_out = adapter(exitwave, ctf_params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


@pytest.mark.skipif(
    not __import__("os").path.exists(
        "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
    ),
    reason="real .cs file not available",
)
def test_real_csfile_particles_match_old_aberration_end_to_end():
    """First 20 real particles from the same .cs file used throughout this
    migration, through the exact dict-based call signature
    aberrations.Aberration.forward() uses -- the same validation bar as
    everywhere else in this migration, applied to the compatibility
    bridge specifically."""
    from specter.io import extract_parameters_from_csfile

    CS_PATH = (
        "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
    )
    (
        voltage_kv,
        pixel_size,
        alpha,
        rotations,
        translations_A,
        ctf_params,
        scale,
        anisomag,
        indices,
        split,
    ) = extract_parameters_from_csfile(CS_PATH, return_class="all", n_particles=20)

    n_pixels = 128
    voltage = float(voltage_kv)
    px = float(pixel_size)
    exitwave = _exitwave(20, seed=7, n_pixels=n_pixels)

    old = Aberration(n_pixels, px, voltage, aberration_model="holography")
    old_out = old(exitwave, ctf_params)

    adapter = LegacyAberrationAdapter(
        n_pixels, px, voltage, aberration_model="holography"
    )
    new_out = adapter(exitwave, ctf_params)

    assert old_out.shape == new_out.shape == (20, n_pixels, n_pixels)
    assert torch.allclose(old_out, new_out, atol=1e-4)
