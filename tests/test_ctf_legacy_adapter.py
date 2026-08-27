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
from specter.ctf import CTFParameters, LegacyAberrationAdapter, TransferFunction

N_PIXELS = 16
PIXEL_SIZE = 1.5
VOLTAGE = 300.0

_LPP_KWARGS = dict(
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

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    old_out = old(exitwave, ctf_params)

    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear"
    )
    new_out = adapter(exitwave, ctf_params)

    assert old_out.shape == new_out.shape
    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_minimal_ctf_params_dict_matches_old_aberration():
    """Just dfu + cs -- matches how run_tilt_series's minimal path
    constructs ctf_params (no astigmatism/beam-tilt/trefoil/phase-plate
    terms at all)."""
    exitwave = _exitwave(2, seed=1)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear"
    )
    new_out = adapter(exitwave, ctf_params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_empty_ctf_params_dict_matches_old_aberration():
    """No CTF terms at all -- matches BaseImager's ctf_params=None path
    (chi stays identically 0)."""
    exitwave = _exitwave(2, seed=2)
    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    old_out = old(exitwave, {})
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear"
    )
    new_out = adapter(exitwave, {})

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_ctf_model_matches_old_aberration():
    exitwave = _exitwave(2, seed=3)
    ctf_params = {
        "dfu": torch.tensor([15000.0, 16000.0]),
        "cs": torch.tensor([2.7e7, 2.7e7]),
    }

    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="linear", alpha=0.0
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="linear"
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
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear", bfactor=150.0
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear", bfactor=150.0
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
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear", dose_envelope=True
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear", dose_envelope=True
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
        aberration_model="nonlinear",
        convergence_angle=1.0,
        cc=1.4e7,
    )
    old_out = old(exitwave, ctf_params)
    adapter = LegacyAberrationAdapter(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="nonlinear",
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
    ) = extract_parameters_from_csfile(CS_PATH, halfset="all", n_particles=20)

    n_pixels = 128
    voltage = float(voltage_kv)
    px = float(pixel_size)
    exitwave = _exitwave(20, seed=7, n_pixels=n_pixels)

    old = Aberration(n_pixels, px, voltage, aberration_model="nonlinear")
    old_out = old(exitwave, ctf_params)

    adapter = LegacyAberrationAdapter(
        n_pixels, px, voltage, aberration_model="nonlinear"
    )
    new_out = adapter(exitwave, ctf_params)

    assert old_out.shape == new_out.shape == (20, n_pixels, n_pixels)
    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_lpp_params_matches_direct_ctfparameters_construction():
    """lpp_params, passed as a LegacyAberrationAdapter *construction-time*
    argument (not a ctf_params dict key -- it's a single shared
    laser-instrument config, never per-particle), must match building
    CTFParameters(lpp_params=...) directly -- no old-Aberration comparison
    is possible here (Aberration has no LPP model at all), so this checks
    the bridge's *wiring* against the already-validated native-units path
    (see test_ctf_transfer.py's LPP tests)."""
    exitwave = _exitwave(1, seed=8)
    ctf_params = {"dfu": torch.tensor([15000.0]), "cs": torch.tensor([2.7e7])}

    adapter = LegacyAberrationAdapter(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="nonlinear",
        lpp_params=_LPP_KWARGS,
    )
    bridged_out = adapter(exitwave, ctf_params)

    params = CTFParameters(
        defocus=15000.0 / 1e4, spherical_aberration=2.7e7 / 1e7, lpp_params=_LPP_KWARGS
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    direct_out = tf(exitwave, params)

    assert torch.allclose(bridged_out, direct_out, atol=1e-6)


def test_lpp_params_overrides_stale_nonzero_phaseshift():
    """A stale nonzero "phaseshift" left over in a ctf_params dict must not
    raise CTFParameters's lpp_params/phase_shift mutual-exclusivity error,
    and must not affect the output -- a construction-time lpp_params
    always wins."""
    exitwave = _exitwave(1, seed=9)
    base_ctf_params = {"dfu": torch.tensor([15000.0]), "cs": torch.tensor([2.7e7])}
    ctf_params_with_stale_phaseshift = {
        **base_ctf_params,
        "phaseshift": torch.tensor([0.5]),
    }

    adapter = LegacyAberrationAdapter(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="nonlinear",
        lpp_params=_LPP_KWARGS,
    )
    out_without_phaseshift = adapter(exitwave, base_ctf_params)
    out_with_stale_phaseshift = adapter(exitwave, ctf_params_with_stale_phaseshift)

    assert torch.allclose(out_without_phaseshift, out_with_stale_phaseshift, atol=1e-6)


# ---------------------------------------------------------------------------
# Unsupported-parameter contract
#
# `torch_ctf` is an incomplete second implementation, kept for a migration off
# `"legacy"` rather than offered as a choice. What it cannot express it must
# refuse: a CryoSPARC .cs file carries tetrafoil, and ignoring it would give a
# plausible image at the wrong transfer function.
# ---------------------------------------------------------------------------


def test_every_aberration_term_is_classified_by_the_adapter() -> None:
    """A term added to Aberration must be mapped here or declared unsupported."""
    from specter.aberrations._aberration import _CTF_PARAM_NAMES
    from specter.ctf._legacy import _SUPPORTED_CTF_PARAMS, _UNSUPPORTED_CTF_PARAMS

    assert not (_SUPPORTED_CTF_PARAMS & _UNSUPPORTED_CTF_PARAMS)
    assert _SUPPORTED_CTF_PARAMS | _UNSUPPORTED_CTF_PARAMS == set(_CTF_PARAM_NAMES), (
        "a ctf_params term is neither mapped nor declared unsupported, so "
        "torch_ctf would silently ignore it"
    )


def test_unsupported_terms_raise_only_when_nonzero() -> None:
    """A zero-valued term has no effect either way, so it is accepted."""
    from specter.ctf._legacy import (
        _UNSUPPORTED_CTF_PARAMS,
        ctf_params_dict_to_parameters,
    )

    base = {
        "dfu": torch.tensor([10000.0]),
        "dfv": torch.tensor([10000.0]),
        "dfang": torch.tensor([0.0]),
        "cs": torch.tensor([2.7e7]),
    }
    kwargs = dict(pixel_size=1.0, image_shape=(32, 32), voltage=300.0)

    for name in sorted(_UNSUPPORTED_CTF_PARAMS):
        ctf_params_dict_to_parameters({**base, name: torch.tensor([0.0])}, **kwargs)
        with pytest.raises(NotImplementedError, match=name):
            ctf_params_dict_to_parameters({**base, name: torch.tensor([1e6])}, **kwargs)
