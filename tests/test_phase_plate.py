"""The phase-plate models: ``phase_plate`` = "ideal", "volta" or "laser".

``"volta"`` shifts frequency k by ``phase_shift * (1 - exp(-k^2 / (2 r^2)))``,
the model Parakeet 0.4.7 applies (``parakeet/simulate/phase_plate.py``,
``compute_phase_shift``; default ``radius`` 0.005 1/Å in ``parakeet/config.py``),
reimplemented below in NumPy exactly as Parakeet builds it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from specter.aberrations import (
    Aberration,
    phase_plate_cuton_profile,
    resolve_phase_plate,
)
from specter.ctf import LegacyAberrationAdapter
from specter.ctf._legacy import ctf_params_dict_to_parameters
from specter.ctf._parameters import CTFParameters
from specter.ctf._transfer import TransferFunction
from specter.imaging._engines import aberration_engine
from specter.settings import Optics

VOLTAGE = 300.0
DEFOCUS = 20000.0  # Å, underfocus
LPP = {"NA": 0.05, "laser_wavelength_angstrom": 10640.0}


def parakeet_phase_plate(
    shape: tuple[int, int], pixel_size: float, phase_shift: float, radius: float
) -> np.ndarray:
    """Parakeet 0.4.7's ``compute_phase_shift``, line for line."""
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    Y = (Y - shape[0] // 2) / (pixel_size * shape[0])
    X = (X - shape[1] // 2) / (pixel_size * shape[1])
    k = np.sqrt(X**2 + Y**2)
    return np.fft.ifftshift(
        np.exp(1j * phase_shift * (1 - np.exp(-(k**2) / (2 * radius**2))))
    )


def _legacy(
    n: int,
    px: float,
    phaseshift: torch.Tensor,
    model: str = "nonlinear",
    **plate: object,
) -> torch.Tensor:
    engine = Aberration(
        n,
        px,
        VOLTAGE,
        aberration_model=model,  # type: ignore[arg-type]
        alpha=0.1,
        **plate,  # type: ignore[arg-type]
    )
    dfu = torch.full_like(phaseshift, DEFOCUS)
    return engine.transfer_function({"dfu": dfu, "phaseshift": phaseshift})


def _torch_ctf(
    n: int,
    px: float,
    phaseshift: torch.Tensor,
    model: str = "nonlinear",
    **plate: object,
) -> torch.Tensor:
    adapter = LegacyAberrationAdapter(
        n,
        px,
        VOLTAGE,
        aberration_model=model,  # type: ignore[arg-type]
        **plate,  # type: ignore[arg-type]
    )
    params = ctf_params_dict_to_parameters(
        {"dfu": torch.full_like(phaseshift, DEFOCUS), "phaseshift": phaseshift},
        px,
        (n, n),
        VOLTAGE,
    )
    return adapter.transfer_function_module.transfer_function(params)


BACKENDS = {"legacy": _legacy, "torch_ctf": _torch_ctf}


# --- mode validation -------------------------------------------------------


def test_resolve_phase_plate_modes() -> None:
    """Modes resolve to their cut-on; None infers from the laser settings."""
    assert resolve_phase_plate("ideal", None, None) == ("ideal", None)
    assert resolve_phase_plate("volta", None, None) == ("volta", 0.005)
    assert resolve_phase_plate("volta", 0.01, None) == ("volta", 0.01)
    assert resolve_phase_plate("laser", None, LPP) == ("laser", None)
    assert resolve_phase_plate(None, None, None) == ("ideal", None)
    assert resolve_phase_plate(None, None, LPP) == ("laser", None)


@pytest.mark.parametrize(
    ("mode", "cuton", "lpp", "match"),
    [
        ("zernike", None, None, "must be one of"),
        ("ideal", 0.005, None, "does not apply"),
        ("laser", 0.005, LPP, "does not apply"),
        ("volta", 0.0, None, "positive"),
        ("volta", -1.0, None, "positive"),
        ("laser", None, None, "needs lpp_params"),
        ("ideal", None, LPP, "laser phase plate"),
        ("volta", None, LPP, "laser phase plate"),
        (None, 0.005, None, "does not apply"),
    ],
)
def test_resolve_phase_plate_refuses(
    mode: str | None, cuton: float | None, lpp: dict | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_phase_plate(mode, cuton, lpp)  # type: ignore[arg-type]


def test_every_entry_point_validates() -> None:
    """Each constructor taking the option refuses a bad combination."""
    with pytest.raises(ValueError, match="does not apply"):
        Aberration(32, 1.0, VOLTAGE, phase_plate_cuton=0.005)
    with pytest.raises(ValueError, match="torch_ctf"):
        Aberration(32, 1.0, VOLTAGE, phase_plate="laser")
    with pytest.raises(ValueError, match="laser phase plate"):
        LegacyAberrationAdapter(32, 1.0, VOLTAGE, lpp_params=LPP, phase_plate="volta")
    with pytest.raises(ValueError, match="needs lpp_params"):
        LegacyAberrationAdapter(32, 1.0, VOLTAGE, phase_plate="laser")
    with pytest.raises(ValueError, match="does not apply"):
        TransferFunction(32, 1.0, phase_plate="ideal", phase_plate_cuton=0.005)
    with pytest.raises(ValueError, match="laser phase plate"):
        Optics(aberration_backend="torch_ctf", lpp_params=LPP, phase_plate="volta")
    with pytest.raises(ValueError, match="positive"):
        Optics(phase_plate="volta", phase_plate_cuton=0.0)
    with pytest.raises(ValueError, match="does not apply"):
        Optics(phase_plate_cuton=0.005)
    # The laser plate's existing settings still infer "laser".
    Optics(aberration_backend="torch_ctf", lpp_params=LPP)
    assert LegacyAberrationAdapter(32, 1.0, VOLTAGE, lpp_params=LPP).lpp_params == LPP


def test_laser_mode_needs_laser_parameters_per_call() -> None:
    """A TransferFunction set to "laser" refuses parameters without lpp_params."""
    tf = TransferFunction(32, 1.0, phase_plate="laser")
    with pytest.raises(ValueError, match="needs lpp_params"):
        tf.transfer_function(CTFParameters(defocus=1.0, voltage=VOLTAGE))


# --- "ideal" is the existing behaviour --------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model", ["nonlinear", "linear"])
def test_ideal_is_the_existing_transfer_function(backend: str, model: str) -> None:
    """``phase_plate="ideal"`` is bit for bit the transfer function without it."""
    tf = BACKENDS[backend]
    ps = torch.tensor([math.pi / 2, math.pi / 4])
    torch.testing.assert_close(
        tf(64, 2.0, ps, model, phase_plate="ideal"),
        tf(64, 2.0, ps, model),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_vanishing_cuton_is_the_ideal_plate(backend: str) -> None:
    """A Volta cut-on far below the grid's first frequency is the ideal plate."""
    tf = BACKENDS[backend]
    ps = torch.tensor([math.pi / 2])
    ideal = tf(96, 3.0, ps, phase_plate="ideal")
    tiny = tf(96, 3.0, ps, phase_plate="volta", phase_plate_cuton=1e-7)
    torch.testing.assert_close(tiny, ideal, rtol=0, atol=1e-6)


# --- "volta" is Parakeet's plate -------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model", ["nonlinear", "linear"])
@pytest.mark.parametrize(("n", "px"), [(256, 3.37), (200, 1.0)])
def test_volta_default_matches_parakeet(
    backend: str, model: str, n: int, px: float
) -> None:
    """T(90°) / T(0°) under the default Volta plate is Parakeet's, on two grids."""
    tf = BACKENDS[backend]
    with_plate = tf(n, px, torch.tensor([math.pi / 2]), model, phase_plate="volta")
    without = tf(n, px, torch.tensor([0.0]), model, phase_plate="volta")
    ratio = (with_plate / without)[0].to(torch.complex128).numpy()
    expected = parakeet_phase_plate((n, n), px, math.pi / 2, 0.005)
    np.testing.assert_allclose(ratio, expected, rtol=0, atol=2e-5)


def test_cuton_profile_half_point() -> None:
    """The plate gives half its shift at k = r sqrt(2 ln 2)."""
    r = 0.005
    k = torch.tensor([0.0, r * math.sqrt(2 * math.log(2)), 1.0])
    torch.testing.assert_close(
        phase_plate_cuton_profile(k, r), torch.tensor([0.0, 0.5, 1.0])
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_phase_shifts_match_singles(backend: str) -> None:
    """Each image of a batch gets its own phase shift under the Volta plate."""
    tf = BACKENDS[backend]
    shifts = torch.tensor([0.0, math.pi / 4, math.pi / 2])
    batch = tf(80, 2.5, shifts, phase_plate="volta")
    for i, s in enumerate(shifts):
        single = tf(80, 2.5, s.reshape(1), phase_plate="volta")
        torch.testing.assert_close(batch[i : i + 1], single, rtol=0, atol=1e-6)


@pytest.mark.parametrize("backend", ["legacy", "torch_ctf"])
def test_engine_cache_keys_on_the_plate(backend: str) -> None:
    """Engines differing only in the phase plate are different cached modules."""
    args = (
        32,
        1.0,
        VOLTAGE,
        "nonlinear",
        None,
        None,
        False,
        True,
        backend,
        None,
        "cpu",
    )
    ideal = aberration_engine(*args, "ideal", None)
    volta = aberration_engine(*args, "volta", 0.005)
    assert ideal is not volta
    assert aberration_engine(*args, "volta", 0.005) is volta


# --- CryoSPARC parity of the ideal plate ------------------------------------


def _cryosparc_ctf(
    k: np.ndarray, df_angstrom: float, ps_rad: float, w: float
) -> np.ndarray:
    """CryoSPARC v4.7.1's 1D CTF, with its own wavelength constants."""
    volts, cs = VOLTAGE * 1e3, 2.7e7  # V; Cs mm -> Å
    lam = 12.2643247 / np.sqrt(volts + 0.978466e-6 * volts**2)
    chi = (
        np.pi * lam * df_angstrom * k**2
        - 0.5 * np.pi * cs * lam**3 * k**4
        + ps_rad
        - np.arccos(w)
    )
    return -np.cos(chi)


@pytest.mark.parametrize("df_um", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("ps_deg", [0.0, 45.0, 90.0])
@pytest.mark.parametrize("w", [0.0, 0.1])
def test_ideal_plate_matches_cryosparc(df_um: float, ps_deg: float, w: float) -> None:
    """The linear-model image transfer equals CryoSPARC v4.7.1's CTF.

    CryoSPARC's ``newctf.compute_ctf_chi`` (``cryosparc_compute/newctf.py``
    lines 23-28 and 148-156, v4.7.1) forms
    ``chi = pi lam df k^2 - 0.5 pi Cs lam^3 k^4 + ps - arccos(w)`` and the CTF
    ``-cos(chi)``, with ``lam = 12.2643247 / sqrt(V + 0.978466e-6 V^2)``
    (the legacy ``ctf.compute_ctf``, ``cryosparc_compute/ctf.py`` lines 22-50,
    is the same function written as ``-sqrt(1 - w^2) sin(chi) - w cos(chi)``).
    SPECTER's linear model without specimen absorption images a real
    projection as ``-imag`` of its transfer function. The residual above
    k = 0 is the two codes' different wavelength constants, ~6e-6 relative.
    """
    n, px = 512, 2.0
    tf = TransferFunction(n, px, aberration_model="linear", specimen_absorption=False)
    params = CTFParameters(
        defocus=df_um,
        spherical_aberration=2.7,
        voltage=VOLTAGE,
        phase_shift=ps_deg,
        amplitude_contrast=w,
    )
    transfer = tf.transfer_function(params).reshape(-1, n, n)[0]
    specter_ctf = -torch.imag(transfer[0, : n // 2]).double().numpy()
    k = np.arange(n // 2) / (n * px)
    expected = _cryosparc_ctf(k, df_um * 1e4, math.radians(ps_deg), w)
    np.testing.assert_allclose(specter_ctf[1:], expected[1:], rtol=0, atol=2e-3)
    np.testing.assert_allclose(specter_ctf[0], expected[0], rtol=0, atol=1e-6)
