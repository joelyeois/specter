"""Parity + new-capability tests for the torch-ctf-backed ctf/ package.

Parity tests compare CTFParameters + TransferFunction (new, torch-ctf-backed)
against aberrations.Aberration (old, hand-rolled) for the terms that exist
in both. The unit/convention mappings used here (defocus/cs Angstrom<->
micrometers/mm, and especially the trefoil/beam-tilt Zernike-coefficient
normalization by `zernike_rho_max`) were derived empirically -- see
ctf._units.zernike_rho_max's docstring for why the Zernike terms need a
grid-dependent rescale that isn't just a fixed unit conversion.

New-capability tests cover things Aberration cannot do at all: tetrafoil,
sparse/optional parameters, the specimen-absorption double-counting guard,
and gradient descent on an arbitrary subset of parameters.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from specter.aberrations import Aberration
from specter.constants import energy_to_wavelength
from specter.ctf import CTFParameters, TransferFunction
from specter.ctf._units import zernike_rho_max

N_PIXELS = 16
PIXEL_SIZE = 1.5
VOLTAGE = 300.0
WAVELENGTH = energy_to_wavelength(VOLTAGE)


def _old_transfer(
    ctf_params: dict, aberration_model: str = "nonlinear"
) -> torch.Tensor:
    kwargs = {"alpha": 0.0} if aberration_model == "linear" else {}
    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model=aberration_model, **kwargs
    )
    return old.transfer_function(ctf_params).squeeze()


def _new_transfer(
    params: CTFParameters,
    specimen_absorption: bool = False,
    aberration_model: str = "nonlinear",
) -> torch.Tensor:
    new = TransferFunction(
        N_PIXELS,
        PIXEL_SIZE,
        aberration_model=aberration_model,
        specimen_absorption=specimen_absorption,
    )
    return new.transfer_function(params).squeeze()


# ---------------------------------------------------------------------------
# Parity: defocus + astigmatism + Cs
# ---------------------------------------------------------------------------


def test_defocus_astigmatism_cs_matches_old_aberration():
    dfu, dfv, dfang = 20000.0, 18000.0, 35.0  # Angstrom, Angstrom, degrees
    cs_ang = 2.7e7  # Angstrom

    old = _old_transfer(
        {
            "dfu": torch.tensor(dfu),
            "dfv": torch.tensor(dfv),
            "dfang": torch.tensor(dfang),
            "cs": torch.tensor(cs_ang),
        }
    )

    params = CTFParameters(
        defocus=(dfu + dfv) / 2 / 1e4,
        astigmatism=(dfu - dfv) / 2 / 1e4,
        astigmatism_angle=dfang,
        spherical_aberration=cs_ang / 1e7,
        voltage=VOLTAGE,
    )
    new = _new_transfer(params)

    assert torch.allclose(old, new, atol=1e-4)


def test_isotropic_defocus_matches_old_aberration():
    """No astigmatism -- sanity check independent of the astigmatism-angle mapping."""
    df_ang = 15000.0
    old = _old_transfer({"dfu": torch.tensor(df_ang), "cs": torch.tensor(2.7e7)})
    params = CTFParameters(
        defocus=df_ang / 1e4, spherical_aberration=2.7e7 / 1e7, voltage=VOLTAGE
    )
    new = _new_transfer(params)
    assert torch.allclose(old, new, atol=1e-4)


def test_cs_only_matches_old_aberration():
    """Cs in isolation, no defocus at all."""
    cs_ang = 3.1e7
    old = _old_transfer({"dfu": torch.tensor(0.0), "cs": torch.tensor(cs_ang)})
    params = CTFParameters(
        defocus=0.0, spherical_aberration=cs_ang / 1e7, voltage=VOLTAGE
    )
    new = _new_transfer(params)
    assert torch.allclose(old, new, atol=1e-4)


@pytest.mark.parametrize(
    "dfang", [0.0, 30.0, 45.0, 60.0, 90.0, -30.0, 120.0, -75.0, 179.0]
)
def test_astigmatism_angle_sweep_matches_old_aberration(dfang):
    """astigmatism_angle=dfang (direct, no offset/sign conversion) must hold at
    every angle, not just the one value spot-checked above."""
    dfu, dfv = 22000.0, 17000.0
    old = _old_transfer(
        {
            "dfu": torch.tensor(dfu),
            "dfv": torch.tensor(dfv),
            "dfang": torch.tensor(dfang),
        }
    )
    params = CTFParameters(
        defocus=(dfu + dfv) / 2 / 1e4,
        astigmatism=(dfu - dfv) / 2 / 1e4,
        astigmatism_angle=dfang,
        spherical_aberration=0.0,
    )
    new = _new_transfer(params)
    # Slightly looser than the other parity tests' 1e-4: pure float32
    # trig-precision accumulation at some angles (e.g. 60 deg), not a
    # formula discrepancy -- values agree to ~4 decimal places regardless.
    assert torch.allclose(old, new, atol=3e-4)


# ---------------------------------------------------------------------------
# Parity: trefoil (via odd_zernike, rescaled by zernike_rho_max)
# ---------------------------------------------------------------------------


def test_trefoil_matches_old_aberration_via_zernike_conversion():
    trefoil1, trefoil2 = 0.6, -0.9  # specter's physical-k^3 convention

    old = _old_transfer(
        {
            "dfu": torch.tensor(0.0),
            "trefoil1": torch.tensor(trefoil1),
            "trefoil2": torch.tensor(trefoil2),
        }
    )

    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    params = CTFParameters(
        defocus=0.0,
        spherical_aberration=0.0,
        odd_zernike={"Z33c": -trefoil1 * rho_max**3, "Z33s": -trefoil2 * rho_max**3},
    )
    new = _new_transfer(params)

    assert torch.allclose(old, new, atol=1e-4)


@pytest.mark.parametrize(
    "trefoil1,trefoil2",
    [(0.6, -0.9), (-0.3, 0.4), (1.2, 0.0), (0.0, -0.7), (-1.5, -1.5)],
)
def test_trefoil_sweep_matches_old_aberration(trefoil1, trefoil2):
    old = _old_transfer(
        {
            "dfu": torch.tensor(0.0),
            "trefoil1": torch.tensor(trefoil1),
            "trefoil2": torch.tensor(trefoil2),
        }
    )
    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    params = CTFParameters(
        defocus=0.0,
        spherical_aberration=0.0,
        odd_zernike={"Z33c": -trefoil1 * rho_max**3, "Z33s": -trefoil2 * rho_max**3},
    )
    new = _new_transfer(params)
    assert torch.allclose(old, new, atol=1e-4)


# ---------------------------------------------------------------------------
# Parity: beam tilt (via odd_zernike Z31, *not* torch-ctf's own
# beam_tilt_mrad convenience path -- see module docstring / _units.py)
# ---------------------------------------------------------------------------


def test_beamtilt_matches_old_aberration_via_zernike_conversion():
    tiltx, tilty = 2e-5, -1.5e-5  # radians, small enough to avoid phase wraparound
    cs_ang = 2.7e7

    old = _old_transfer(
        {
            "dfu": torch.tensor(0.0),
            "cs": torch.tensor(cs_ang),
            "tiltx": torch.tensor(tiltx),
            "tilty": torch.tensor(tilty),
        }
    )

    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    prefactor = 2 * torch.pi * WAVELENGTH**2 * cs_ang * rho_max**3
    params = CTFParameters(
        defocus=0.0,
        spherical_aberration=cs_ang / 1e7,
        voltage=VOLTAGE,
        odd_zernike={"Z31c": -prefactor * tiltx, "Z31s": -prefactor * tilty},
    )
    new = _new_transfer(params)

    assert torch.allclose(old, new, atol=1e-4)


def test_beam_tilt_mrad_convenience_arg_does_not_match_specter_convention():
    """Documents a real gotcha: torch-ctf's own beam_tilt_mrad -> Zernike
    conversion does not include the zernike_rho_max rescale, so it does
    *not* reproduce specter's physical-k beamtilt formula. Manual
    odd_zernike (see test above) is required for parity."""
    tiltx, tilty = 2e-5, -1.5e-5
    cs_ang = 2.7e7

    old = _old_transfer(
        {
            "dfu": torch.tensor(0.0),
            "cs": torch.tensor(cs_ang),
            "tiltx": torch.tensor(tiltx),
            "tilty": torch.tensor(tilty),
        }
    )
    params = CTFParameters(
        defocus=0.0,
        spherical_aberration=cs_ang / 1e7,
        voltage=VOLTAGE,
        beam_tilt_mrad=[tiltx * 1e3, tilty * 1e3],
    )
    new = _new_transfer(params)

    assert not torch.allclose(old, new, atol=1e-3)


@pytest.mark.parametrize(
    "tiltx,tilty",
    [(2e-5, -1.5e-5), (-1e-5, 3e-5), (5e-6, 0.0), (0.0, -4e-5), (-3e-5, -3e-5)],
)
def test_beamtilt_sweep_matches_old_aberration(tiltx, tilty):
    cs_ang = 2.7e7
    old = _old_transfer(
        {
            "dfu": torch.tensor(0.0),
            "cs": torch.tensor(cs_ang),
            "tiltx": torch.tensor(tiltx),
            "tilty": torch.tensor(tilty),
        }
    )
    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    prefactor = 2 * torch.pi * WAVELENGTH**2 * cs_ang * rho_max**3
    params = CTFParameters(
        defocus=0.0,
        spherical_aberration=cs_ang / 1e7,
        voltage=VOLTAGE,
        odd_zernike={"Z31c": -prefactor * tiltx, "Z31s": -prefactor * tilty},
    )
    new = _new_transfer(params)
    assert torch.allclose(old, new, atol=1e-4)


# ---------------------------------------------------------------------------
# Parity: phase shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase_deg", [10.0, 40.0, 90.0, -25.0, 150.0])
def test_phase_shift_matches_old_aberration_linear_model(phase_deg):
    """Compared under aberration_model="linear": Aberration's phaseshift() only
    zeroes the DC pixel for the "nonlinear" model (see the next test) --
    "linear" mode applies the plain -phaseshift term everywhere, matching
    torch-ctf's calculate_total_phase_shift with no special-casing."""
    phase_rad = torch.deg2rad(torch.tensor(phase_deg))
    old = _old_transfer(
        {"dfu": torch.tensor(0.0), "phaseshift": phase_rad}, aberration_model="linear"
    )
    params = CTFParameters(defocus=0.0, spherical_aberration=0.0, phase_shift=phase_deg)
    new = _new_transfer(params, aberration_model="linear")
    assert torch.allclose(old, new, atol=1e-4)


def test_phase_shift_nonlinear_model_matches_old_aberration_including_dc():
    """TransferFunction._zero_dc_for_nonlinear ports Aberration's
    nonlinear-model DC-pinning ("phaseshift must be zero at DC for Fourier
    optics validity"), so this now matches everywhere, DC included -- not
    just "everywhere but DC" as it did before that fix."""
    phase_deg = 40.0
    phase_rad = torch.deg2rad(torch.tensor(phase_deg))
    old = _old_transfer(
        {"dfu": torch.tensor(0.0), "phaseshift": phase_rad},
        aberration_model="nonlinear",
    )
    params = CTFParameters(defocus=0.0, spherical_aberration=0.0, phase_shift=phase_deg)
    new = _new_transfer(params)

    assert torch.allclose(old, new, atol=1e-4)
    assert torch.allclose(new[0, 0], torch.tensor(1.0 + 0.0j), atol=1e-6)


def test_nonlinear_dc_pinning_also_covers_amplitude_contrast():
    """amplitude_contrast has no old-Aberration equivalent (dead code there),
    but it's the same k-independent-chi-term situation as phase_shift: left
    unpinned, it would be an inert global phase in nonlinear mode. Verify
    the DC pixel is pinned to 1+0j even when only amplitude_contrast (not
    phase_shift) is nonzero, and that non-DC pixels still carry its effect."""
    params = CTFParameters(defocus=1.0, amplitude_contrast=0.1)
    new = _new_transfer(params, specimen_absorption=False)

    assert torch.allclose(new[0, 0], torch.tensor(1.0 + 0.0j), atol=1e-6)
    # Away from DC, amplitude_contrast must still have a real effect (i.e.
    # this isn't pinning the whole array, just the DC pixel).
    baseline = _new_transfer(
        CTFParameters(defocus=1.0, amplitude_contrast=0.0), specimen_absorption=False
    )
    assert not torch.allclose(new[1, 1], baseline[1, 1], atol=1e-6)


def test_dc_pinning_preserves_gradients_correctly():
    """The DC pin must show up correctly in autograd: a learnable
    phase_shift's gradient should get zero contribution from the (now
    constant) DC pixel, but nonzero contribution from everywhere else."""
    params = CTFParameters(
        defocus=1.0, phase_shift=40.0, amplitude_contrast=0.0, learnable={"phase_shift"}
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    transfer = tf.transfer_function(params)

    # DC must be a plain constant (no grad_fn dependence on phase_shift):
    # perturbing phase_shift must not move it.
    grad_dc = torch.autograd.grad(
        transfer[0, 0].real, params.fields["phase_shift"].value, retain_graph=True
    )[0]
    assert torch.allclose(grad_dc, torch.tensor(0.0), atol=1e-6)

    # Some non-DC pixel must still depend on phase_shift.
    grad_offdc = torch.autograd.grad(
        transfer[2, 3].real, params.fields["phase_shift"].value, retain_graph=True
    )[0]
    assert grad_offdc.abs().item() > 1e-6


# ---------------------------------------------------------------------------
# Independent verification: amplitude contrast (no old-Aberration parity is
# possible here -- Aberration.alpha is accepted but never actually used, so
# there is nothing in the old code to compare against. Verified instead
# against the closed-form phase-shift-equivalence derivation:
# amplitude contrast Q is mathematically equivalent to adding a constant
# phase offset delta = arcsin(Q) to chi, independent of k.
# ---------------------------------------------------------------------------


def test_amplitude_contrast_real_ctf_equals_constant_Q():
    """With every other term zero, chi is identically -arcsin(Q) at all k, so
    the real weak-phase-object CTF -sin(chi) collapses to the constant Q
    everywhere -- a clean, closed-form check independent of any grid or
    frequency-dependent term."""
    from torch_ctf import calculate_ctf_2d

    Q = 0.1
    ctf = calculate_ctf_2d(
        defocus=0.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=VOLTAGE,
        spherical_aberration=0.0,
        amplitude_contrast=Q,
        phase_shift=0.0,
        pixel_size=PIXEL_SIZE,
        image_shape=(N_PIXELS, N_PIXELS),
        rfft=False,
        fftshift=False,
        return_complex_ctf=False,
    )
    assert torch.allclose(ctf, torch.full_like(ctf, Q), atol=1e-6)


def test_amplitude_contrast_complex_transfer_matches_specters_absorptive_potential_constant():
    """The complex transfer function should equal sqrt(1-Q^2) + iQ everywhere
    -- exactly specter's own potential.apply_amplitude_contrast's `c = sqrt(1-a^2)
    + i*a` scalar. This is the concrete numeric confirmation of the
    equivalence derived earlier: amplitude contrast applied via an
    absorptive specimen potential and amplitude contrast applied as a CTF
    phase-shift term are the same physics, evaluated at two different
    points in the pipeline."""
    from torch_ctf import calculate_ctf_2d

    Q = 0.1
    transfer = calculate_ctf_2d(
        defocus=0.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=VOLTAGE,
        spherical_aberration=0.0,
        amplitude_contrast=Q,
        phase_shift=0.0,
        pixel_size=PIXEL_SIZE,
        image_shape=(N_PIXELS, N_PIXELS),
        rfft=False,
        fftshift=False,
        return_complex_ctf=True,
    )
    expected = torch.full_like(transfer, (1 - Q**2) ** 0.5 + 1j * Q)
    assert torch.allclose(transfer, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# New capability: tetrafoil (Aberration's is an unimplemented stub, so this
# is verified against torch-ctf's own documented Zernike formula by direct
# reimplementation, not against specter -- there is no specter formula to
# compare against).
# ---------------------------------------------------------------------------


def test_tetrafoil_now_computable():
    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    params = CTFParameters(
        defocus=1.0,
        even_zernike={
            "Z44c": 0.2 * rho_max**4,
            "Z44s": -0.1 * rho_max**4,
            "Z60": 0.05 * rho_max**6,
        },
    )
    new = _new_transfer(params)
    assert torch.isfinite(new.real).all() and torch.isfinite(new.imag).all()
    # A nonzero tetrafoil term should make the transfer function anisotropic
    # (break the pure-defocus rotational symmetry).
    assert not torch.allclose(new, new.T, atol=1e-6)


def test_tetrafoil_matches_reimplemented_zernike_formula():
    """Reimplements torch_ctf.ctf_aberrations.apply_even_zernikes's
    documented formula (coeff * rho**4 * cos/sin(4*theta) for Z44c/Z44s,
    coeff * rho**6 for Z60) from scratch and checks the actual function
    against it -- a regression/self-consistency check on torch-ctf's own
    code, since specter has no tetrafoil implementation to compare to."""
    from torch_grid_utils.fftfreq_grid import fftfreq_grid
    from torch_grid_utils.polar_grid import fftfreq_grid_polar
    from torch_ctf.ctf_aberrations import apply_even_zernikes

    fft_freq_grid = fftfreq_grid(
        image_shape=(N_PIXELS, N_PIXELS),
        rfft=False,
        fftshift=False,
        spacing=PIXEL_SIZE,
        norm=False,
    )
    rho, theta = fftfreq_grid_polar(fft_freq_grid)

    z44c, z44s, z60 = 0.2, -0.1, 0.05
    total_phase_shift = torch.zeros_like(rho)
    actual = apply_even_zernikes(
        {"Z44c": z44c, "Z44s": z44s, "Z60": z60}, total_phase_shift, rho, theta
    )
    expected = (
        z44c * rho**4 * torch.cos(4 * theta)
        + z44s * rho**4 * torch.sin(4 * theta)
        + z60 * rho**6
    )
    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Flexibility: omitted parameters must still run
# ---------------------------------------------------------------------------


def test_defocus_only_runs():
    params = CTFParameters(defocus=1.0)
    new = _new_transfer(params)
    assert new.shape == (N_PIXELS, N_PIXELS)
    assert torch.isfinite(new.real).all()


def test_sparse_odd_zernike_trefoil_only_runs():
    params = CTFParameters(defocus=1.0, odd_zernike={"Z33c": 0.05})
    new = _new_transfer(params)
    assert torch.isfinite(new.real).all()
    assert "Z33s" not in params.torch_ctf_kwargs(PIXEL_SIZE, (N_PIXELS, N_PIXELS)).get(
        "odd_zernike_coeffs", {}
    )


def test_no_zernike_terms_omits_kwargs_entirely():
    params = CTFParameters(defocus=1.0)
    kwargs = params.torch_ctf_kwargs(PIXEL_SIZE, (N_PIXELS, N_PIXELS))
    assert "even_zernike_coeffs" not in kwargs
    assert "odd_zernike_coeffs" not in kwargs
    assert "beam_tilt_mrad" not in kwargs


# ---------------------------------------------------------------------------
# specimen_absorption double-counting guard (the "option a" policy)
# ---------------------------------------------------------------------------


def test_specimen_absorption_guard_zeroes_and_warns():
    baseline = CTFParameters(defocus=1.0, amplitude_contrast=0.0)
    baseline_out = _new_transfer(baseline, specimen_absorption=True)

    contaminated = CTFParameters(defocus=1.0, amplitude_contrast=0.1)
    with pytest.warns(UserWarning, match="specimen_absorption"):
        guarded_out = _new_transfer(contaminated, specimen_absorption=True)

    assert torch.allclose(baseline_out, guarded_out)


def test_specimen_absorption_false_lets_amplitude_contrast_through():
    baseline = CTFParameters(defocus=1.0, amplitude_contrast=0.0)
    baseline_out = _new_transfer(baseline, specimen_absorption=False)

    explicit = CTFParameters(defocus=1.0, amplitude_contrast=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit_out = _new_transfer(explicit, specimen_absorption=False)

    assert not torch.allclose(baseline_out, explicit_out)


# ---------------------------------------------------------------------------
# Gradient descent flexibility
# ---------------------------------------------------------------------------


def test_learnable_field_receives_gradient():
    params = CTFParameters(defocus=1.0, learnable={"defocus"})
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")

    # A constant exitwave has all its energy at the DC frequency, where every
    # defocus/Cs/Zernike term is exactly zero by construction -- gradients
    # would trivially vanish. Use broadband content so every term's
    # gradient is actually exercised.
    exitwave = torch.randn(
        1,
        N_PIXELS,
        N_PIXELS,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(0),
    )
    out = tf(exitwave, params)
    loss = (out.abs() ** 2).sum()
    loss.backward()

    defocus_param = params.fields["defocus"].value
    assert isinstance(defocus_param, torch.nn.Parameter)
    assert defocus_param.grad is not None
    assert torch.isfinite(defocus_param.grad).all()

    # Non-learnable fields must not accumulate gradients.
    assert not isinstance(params.fields["astigmatism"].value, torch.nn.Parameter)


def test_non_learnable_fields_stay_fixed_buffers():
    params = CTFParameters(defocus=1.0, odd_zernike={"Z33c": 0.05}, learnable=())
    assert not isinstance(params.fields["defocus"].value, torch.nn.Parameter)
    assert not isinstance(params.odd_zernike["Z33c"].value, torch.nn.Parameter)
    assert list(params.parameters()) == []


def test_per_field_learnable_selection():
    params = CTFParameters(
        defocus=1.0,
        odd_zernike={"Z33c": 0.05, "Z33s": 0.02},
        learnable={"defocus", "Z33c"},
    )
    assert isinstance(params.fields["defocus"].value, torch.nn.Parameter)
    assert isinstance(params.odd_zernike["Z33c"].value, torch.nn.Parameter)
    assert not isinstance(params.odd_zernike["Z33s"].value, torch.nn.Parameter)
    assert not isinstance(params.fields["astigmatism"].value, torch.nn.Parameter)


def test_per_field_optimizers_update_independently():
    """Answers: can different CTF params have different optimizers/LRs?
    Yes -- each learnable ParamField.value is an ordinary nn.Parameter, so
    ordinary per-parameter optimizer groups (or entirely separate
    optimizers, as Reconstructor already does for V/rotations/translations)
    work unchanged."""
    params = CTFParameters(
        defocus=1.0,
        odd_zernike={"Z33c": 0.05},
        learnable={"defocus", "Z33c"},
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")

    defocus_before = params.fields["defocus"].value.item()
    trefoil_before = params.odd_zernike["Z33c"].value.item()

    # Z33c's gradient is intrinsically much smaller than defocus's (rho is
    # normalized to [0, 1] vs. defocus's physical 1/Angstrom^2 scale), so
    # its learning rate needs to be correspondingly larger to produce a
    # visible update -- itself a demonstration of why per-field learning
    # rates are necessary, not just a nice-to-have.
    opt_defocus = torch.optim.SGD([params.fields["defocus"].value], lr=1.0)
    opt_trefoil = torch.optim.SGD([params.odd_zernike["Z33c"].value], lr=1e4)

    # A constant exitwave has all its energy at the DC frequency, where every
    # defocus/Cs/Zernike term is exactly zero by construction -- gradients
    # would trivially vanish. Use broadband content so every term's
    # gradient is actually exercised.
    exitwave = torch.randn(
        1,
        N_PIXELS,
        N_PIXELS,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(0),
    )
    loss = (tf(exitwave, params).abs() ** 2).sum()
    opt_defocus.zero_grad()
    opt_trefoil.zero_grad()
    loss.backward()
    opt_defocus.step()
    opt_trefoil.step()

    defocus_delta = abs(params.fields["defocus"].value.item() - defocus_before)
    trefoil_delta = abs(params.odd_zernike["Z33c"].value.item() - trefoil_before)
    assert defocus_delta > 0
    assert trefoil_delta > 0


# ---------------------------------------------------------------------------
# Parity: bfactor envelope (b_envelope is literally the same imported
# function on both sides -- this checks the two classes feed it the same k2
# grid, not the envelope math itself, which can't diverge).
# ---------------------------------------------------------------------------


def test_bfactor_envelope_matches_old_aberration_via_forward():
    """NOTE: Aberration's constructor-level `bfactor` convenience is only
    merged into ctf_params inside forward() (`if self.bfactor is not None:
    ctf_params = {**ctf_params, "bfactor": self.bfactor}`) -- calling
    Aberration.transfer_function() directly silently ignores it unless
    "bfactor" is already a key in the dict you pass. TransferFunction
    applies self.bfactor unconditionally inside transfer_function() itself,
    so it also takes effect when transfer_function() is called directly
    (e.g. for CTF-only diagnostics/visualization) -- a deliberate
    improvement, but it means the two classes' *transfer_function()* outputs
    only agree once bfactor is actually wired through, i.e. compare via
    forward(), which is how both are used in practice."""
    bfactor_val = 150.0  # Angstrom^2
    exitwave = torch.randn(
        1,
        N_PIXELS,
        N_PIXELS,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(3),
    )
    old = Aberration(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="nonlinear",
        bfactor=bfactor_val,
    )
    old_out = old.forward(exitwave, {"dfu": torch.tensor(12000.0)})

    params = CTFParameters(defocus=12000.0 / 1e4, spherical_aberration=0.0)
    new_tf = TransferFunction(
        N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear", bfactor=bfactor_val
    )
    new_out = new_tf.forward(exitwave, params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_bfactor_via_transfer_function_directly_is_a_known_api_difference():
    """Documents the asymmetry above concretely: with the same bfactor,
    calling .transfer_function() directly (not .forward()) gives *different*
    results between the two classes -- old ignores the constructor bfactor
    entirely here, new applies it. Not a bug in either class individually,
    but a real API-level difference to know about if anything ever calls
    .transfer_function() directly instead of .forward()."""
    bfactor_val = 150.0
    old = Aberration(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="nonlinear",
        bfactor=bfactor_val,
    )
    old_t = old.transfer_function({"dfu": torch.tensor(12000.0)}).squeeze()

    params = CTFParameters(defocus=12000.0 / 1e4, spherical_aberration=0.0)
    new_tf = TransferFunction(
        N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear", bfactor=bfactor_val
    )
    new_t = new_tf.transfer_function(params).squeeze()

    assert not torch.allclose(
        old_t, new_t, atol=1e-4
    )  # old: bfactor silently ignored here
    assert torch.allclose(
        old_t[0, 0], new_t[0, 0], atol=1e-4
    )  # DC: both unaffected by bfactor


# ---------------------------------------------------------------------------
# Parity: multi-particle batching (every test above used an implicit batch
# of 1 -- Reconstructor/ImageGenerator always pass per-particle tensors of
# shape (N,), N>1, so this is the actually-used code path, not tested
# anywhere else in this file).
# ---------------------------------------------------------------------------


def test_multi_particle_batch_matches_old_aberration():
    dfu = torch.tensor([12000.0, 15000.0, 9000.0, 20000.0])
    dfv = torch.tensor([11000.0, 15000.0, 8500.0, 17000.0])
    dfang = torch.tensor([10.0, 0.0, 45.0, -20.0])
    cs_ang = 2.7e7

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    old_t = old.transfer_function(
        {"dfu": dfu, "dfv": dfv, "dfang": dfang, "cs": torch.full((4,), cs_ang)}
    )

    params = CTFParameters(
        defocus=(dfu + dfv) / 2 / 1e4,
        astigmatism=(dfu - dfv) / 2 / 1e4,
        astigmatism_angle=dfang,
        spherical_aberration=cs_ang / 1e7,
        voltage=VOLTAGE,
    )
    new_tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    new_t = new_tf.transfer_function(params)

    assert old_t.shape == new_t.shape == (4, N_PIXELS, N_PIXELS)
    assert torch.allclose(old_t, new_t, atol=1e-4)


def test_multi_particle_indexing_selects_matching_subset():
    """CTFParameters.torch_ctf_kwargs' idx argument must select the same
    per-particle subset Aberration gets by slicing its ctf_params dict
    before calling transfer_function -- these need to agree for
    Reconstructor's per-batch particle indexing to be swappable."""
    dfu = torch.tensor([12000.0, 15000.0, 9000.0, 20000.0])
    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    idx = torch.tensor([2, 0])
    old_t = old.transfer_function({"dfu": dfu[idx]})

    params = CTFParameters(defocus=dfu / 1e4, spherical_aberration=0.0)
    new_tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    new_t = new_tf.transfer_function(params, idx=idx)

    assert torch.allclose(old_t, new_t, atol=1e-4)


# ---------------------------------------------------------------------------
# Parity: full forward() end-to-end (a real complex exitwave in, an
# aberrated wave out) -- every test above only compared the transfer
# function itself, not the fft2/multiply/ifft2 application to an actual
# wavefunction.
# ---------------------------------------------------------------------------


def test_forward_end_to_end_matches_old_aberration_nonlinear():
    exitwave = torch.randn(
        1,
        N_PIXELS,
        N_PIXELS,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(1),
    )
    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    old_out = old.forward(
        exitwave, {"dfu": torch.tensor(14000.0), "cs": torch.tensor(2.7e7)}
    )

    params = CTFParameters(
        defocus=14000.0 / 1e4, spherical_aberration=2.7e7 / 1e7, voltage=VOLTAGE
    )
    new_tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    new_out = new_tf.forward(exitwave, params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


def test_forward_end_to_end_matches_old_aberration_ctf_model():
    exitwave = torch.randn(
        1,
        N_PIXELS,
        N_PIXELS,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(2),
    )
    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="linear", alpha=0.0
    )
    old_out = old.forward(
        exitwave, {"dfu": torch.tensor(14000.0), "cs": torch.tensor(2.7e7)}
    )

    params = CTFParameters(
        defocus=14000.0 / 1e4, spherical_aberration=2.7e7 / 1e7, voltage=VOLTAGE
    )
    new_tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="linear")
    new_out = new_tf.forward(exitwave, params)

    assert torch.allclose(old_out, new_out, atol=1e-4)


# ---------------------------------------------------------------------------
# Explicitly not covered: Cc/Cs-spatial-coherence/dose envelopes have no
# TransferFunction equivalent yet at all (not even a silently-wrong no-op --
# the constructor simply has no such parameters, so passing them is a
# TypeError, not a silent mismatch). Confirmed here so a future signature
# change doesn't accidentally start silently ignoring them instead.
# ---------------------------------------------------------------------------


def test_cs_envelope_matches_old_aberration():
    """Spatial-coherence (beam convergence) envelope -- needs the
    defocus/Cs micrometers/mm -> Angstrom conversion at the call site."""
    dfu, cs_ang = 15000.0, 2.7e7
    old = _old_transfer(
        {"dfu": torch.tensor(dfu), "cs": torch.tensor(cs_ang)},
    )
    old_full = Aberration(
        N_PIXELS,
        PIXEL_SIZE,
        VOLTAGE,
        aberration_model="nonlinear",
        convergence_angle=1.0,
    )
    old_t = old_full.transfer_function(
        {"dfu": torch.tensor(dfu), "cs": torch.tensor(cs_ang)}
    ).squeeze()

    params = CTFParameters(
        defocus=dfu / 1e4, spherical_aberration=cs_ang / 1e7, voltage=VOLTAGE
    )
    new_tf = TransferFunction(
        N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear", convergence_angle=1.0
    )
    new_t = new_tf.transfer_function(params).squeeze()

    assert torch.allclose(old_t, new_t, atol=1e-4)
    # And it must actually do something (differ from the no-envelope case).
    assert not torch.allclose(old_t, old, atol=1e-3)


def test_cc_envelope_matches_old_aberration():
    """Temporal-coherence (chromatic aberration) envelope."""
    dfu, cs_ang = 15000.0, 2.7e7
    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear", cc=1.4e7
    )
    old_t = old.transfer_function(
        {"dfu": torch.tensor(dfu), "cs": torch.tensor(cs_ang)}
    ).squeeze()

    params = CTFParameters(
        defocus=dfu / 1e4, spherical_aberration=cs_ang / 1e7, voltage=VOLTAGE
    )
    new_tf = TransferFunction(
        N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear", cc=1.4e7
    )
    new_t = new_tf.transfer_function(params).squeeze()

    assert torch.allclose(old_t, new_t, atol=1e-4)


def test_dose_envelope_matches_old_aberration():
    """Grant & Grigorieff (2015) cumulative-dose envelope -- dose lives on
    CTFParameters (matching old Aberration reading it from the same
    per-image ctf_params dict as everything else), not as a
    TransferFunction constructor convenience like bfactor."""
    dfu, cs_ang = 15000.0, 2.7e7
    old = Aberration(
        N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear", dose_envelope=True
    )
    old_t = old.transfer_function(
        {
            "dfu": torch.tensor(dfu),
            "cs": torch.tensor(cs_ang),
            "dose": torch.tensor(40.0),
        }
    ).squeeze()

    params = CTFParameters(
        defocus=dfu / 1e4, spherical_aberration=cs_ang / 1e7, voltage=VOLTAGE, dose=40.0
    )
    new_tf = TransferFunction(
        N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear", dose_envelope=True
    )
    new_t = new_tf.transfer_function(params).squeeze()

    assert torch.allclose(old_t, new_t, atol=1e-4)


def test_dose_envelope_disabled_without_ctf_params_dose():
    """dose_envelope=True on TransferFunction but no dose on CTFParameters
    must be a no-op (matches old Aberration's `"dose" in ctf_params` gate),
    not an error."""
    params = CTFParameters(defocus=1.0, spherical_aberration=2.7)
    tf_with_envelope = TransferFunction(
        N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear", dose_envelope=True
    )
    tf_without = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    assert torch.allclose(
        tf_with_envelope.transfer_function(params), tf_without.transfer_function(params)
    )


# ---------------------------------------------------------------------------
# Regression: per-particle (batched) Zernike coefficients. Real .cs files
# give trefoil/beam-tilt-derived Zernike terms as per-particle tensors, not
# scalars -- this path was broken twice over until exercised against real
# data: CTFParameters.__init__ hard-cast Zernike values through float(),
# which rejects any tensor with more than one element; and torch_ctf's own
# apply_odd_zernikes/apply_even_zernikes initialize their phase accumulator
# unbatched (torch.zeros_like(rho)) and accumulate into it in place, which
# raises a broadcast error the moment a Zernike coefficient has a batch
# dimension at all -- a real upstream limitation, worked around in
# TransferFunction._call_calculate_ctf_2d by falling back to a per-particle
# loop only when a batched Zernike coefficient is actually present.
# ---------------------------------------------------------------------------


def test_multi_particle_trefoil_matches_old_aberration():
    """Same physics as test_trefoil_sweep_matches_old_aberration, but with
    a genuinely per-particle (not scalar) trefoil coefficient -- this is
    the shape that broke both bugs above."""
    trefoil1 = torch.tensor([0.6, -0.3, 1.2, 0.0, -1.5])
    trefoil2 = torch.tensor([-0.9, 0.4, 0.0, -0.7, -1.5])

    old = Aberration(N_PIXELS, PIXEL_SIZE, VOLTAGE, aberration_model="nonlinear")
    old_t = old.transfer_function(
        {"dfu": torch.zeros(5), "trefoil1": trefoil1, "trefoil2": trefoil2}
    )

    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    params = CTFParameters(
        defocus=torch.zeros(5),
        spherical_aberration=0.0,
        odd_zernike={"Z33c": -trefoil1 * rho_max**3, "Z33s": -trefoil2 * rho_max**3},
    )
    new_tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    new_t = new_tf.transfer_function(params)

    assert old_t.shape == new_t.shape == (5, N_PIXELS, N_PIXELS)
    assert torch.allclose(old_t, new_t, atol=1e-4)


def test_mixed_batched_and_scalar_zernike_coefficients():
    """Some Zernike terms per-particle, others shared/scalar -- both must
    resolve correctly through the same per-particle fallback loop."""
    rho_max = zernike_rho_max((N_PIXELS, N_PIXELS), PIXEL_SIZE)
    params = CTFParameters(
        defocus=torch.tensor([1.0, 1.2, 0.8]),
        spherical_aberration=0.0,
        even_zernike={
            "Z44c": torch.tensor([0.1, 0.2, -0.1]) * rho_max**4,
            "Z60": 0.05 * rho_max**6,
        },
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    out = tf.transfer_function(params)
    assert out.shape == (3, N_PIXELS, N_PIXELS)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()
    # The three particles must actually differ (different defocus and Z44c).
    assert not torch.allclose(out[0], out[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Uniform-batched-Zernike-coefficient collapse: cryoSPARC's .cs schema
# stores every CTF term as a flat per-particle array regardless of whether
# the underlying value is actually per-particle -- confirmed against a real
# .cs file's *entire* 30,515-particle dataset: exactly 1 unique
# trefoil/beam-tilt/Cs/phase-shift value each, vs. ~30,000+ unique defocus
# values. TransferFunction collapses a per-particle-*shaped* but
# uniform-*valued* Zernike coefficient to a single value so the fast
# vectorized calculate_ctf_2d path runs instead of the slow per-particle
# loop -- this is the common case in practice, not a rare one.
# ---------------------------------------------------------------------------


def test_collapse_if_uniform_helper():
    from specter.ctf._transfer import _collapse_if_uniform

    uniform = torch.full((5,), 0.3)
    collapsed = _collapse_if_uniform(uniform)
    assert collapsed.numel() == 1
    assert collapsed.item() == pytest.approx(0.3)

    nonuniform = torch.tensor([0.1, 0.2, 0.3])
    assert _collapse_if_uniform(nonuniform) is nonuniform

    scalar = torch.tensor(0.5)
    assert _collapse_if_uniform(scalar) is scalar

    learnable_uniform = torch.nn.Parameter(torch.full((5,), 0.3))
    assert _collapse_if_uniform(learnable_uniform) is learnable_uniform


def test_uniform_batched_trefoil_with_per_particle_defocus_matches_manual_loop():
    """Realistic .cs-file shape: trefoil stored per-particle but every value
    identical, alongside genuinely per-particle defocus. Must match the
    same per-particle reconstruction the (slow, unoptimized) loop would
    give -- the optimization must not change the result, only the path."""
    defocus = torch.tensor([0.8, 1.0, 1.2, 0.9, 1.1])
    trefoil_c_uniform = torch.full((5,), 0.15)

    params = CTFParameters(
        defocus=defocus,
        spherical_aberration=2.7,
        odd_zernike={"Z33c": trefoil_c_uniform},
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    out = tf.transfer_function(params)

    expected = torch.cat(
        [
            tf.transfer_function(
                CTFParameters(
                    defocus=defocus[i : i + 1],
                    spherical_aberration=2.7,
                    odd_zernike={"Z33c": torch.tensor(0.15)},
                )
            )
            for i in range(5)
        ],
        dim=0,
    )
    assert out.shape == (5, N_PIXELS, N_PIXELS)
    assert torch.allclose(out, expected, atol=1e-6)
    # Particles still differ from each other via defocus, despite the
    # shared/collapsed trefoil value.
    assert not torch.allclose(out[0], out[1], atol=1e-6)


def test_learnable_uniform_zernike_coefficient_gets_independent_gradients():
    """The safety guard: a *learnable* per-particle Zernike coefficient
    that happens to start out uniform must not be collapsed, or every
    particle but one would silently get zero gradient. Verified by
    checking each of N particles -- which see different defocus, so
    genuinely have independent loss contributions -- gets its own nonzero
    gradient, not just a single collapsed element.

    Applies the transfer function to an actual (broadband, non-constant)
    exit wave rather than testing on the bare transfer_function() output:
    exp(-i*chi) has unit magnitude everywhere by construction, so
    |transfer_function()|^2 is trivially constant and its gradient w.r.t.
    any CTF parameter is ~0 regardless of correctness -- not a meaningful
    gradient check on its own.
    """
    defocus = torch.tensor([0.8, 1.0, 1.2])
    params = CTFParameters(
        defocus=defocus,
        spherical_aberration=2.7,
        odd_zernike={"Z33c": torch.full((3,), 0.15)},
        learnable={"Z33c"},
    )
    coeff = params.odd_zernike["Z33c"].value
    assert isinstance(coeff, torch.nn.Parameter)

    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    exitwave = torch.randn(
        3,
        N_PIXELS,
        N_PIXELS,
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(0),
    )
    out = tf(exitwave, params)
    # Per-particle loss so each particle's gradient contribution is distinct.
    loss = (out.abs() ** 2 * torch.tensor([1.0, 2.0, 3.0]).view(-1, 1, 1)).sum()
    loss.backward()

    assert coeff.grad is not None
    assert torch.isfinite(coeff.grad).all()
    assert torch.all(coeff.grad.flatten() != 0), (
        "every particle's trefoil gradient must be independently nonzero -- "
        "a collapse bug would leave all but one element exactly zero"
    )


# ---------------------------------------------------------------------------
# Real-world validation: first 5 particles from an actual CryoSPARC .cs
# file, run through the real extraction/unit-conversion path
# (specter.io.extract_parameters_from_csfile), not hand-picked synthetic
# values. Skipped if the file isn't mounted (e.g. off the lab filesystem).
# ---------------------------------------------------------------------------

_REAL_CS_FILE = (
    "/scratch/loh/joel/empiar-10202/CS-aav2/J247/J247_passthrough_particles.cs"
)


@pytest.mark.skipif(
    not __import__("os").path.exists(_REAL_CS_FILE),
    reason=f"real .cs file not available: {_REAL_CS_FILE}",
)
def test_first_five_particles_of_real_csfile_match_old_aberration():
    import math

    from specter.io import extract_parameters_from_csfile

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
    ) = extract_parameters_from_csfile(_REAL_CS_FILE, halfset="all", n_particles=5)

    n_pixels = 256
    voltage = float(voltage_kv)
    px = float(pixel_size)

    old = Aberration(n_pixels, px, voltage, aberration_model="nonlinear")
    old_t = old.transfer_function(ctf_params)

    dfu, dfv, dfang = ctf_params["dfu"], ctf_params["dfv"], ctf_params["dfang"]
    cs_A = ctf_params["cs"]
    tiltx, tilty = ctf_params["tiltx"], ctf_params["tilty"]
    trefoil1, trefoil2 = ctf_params["trefoil1"], ctf_params["trefoil2"]

    rho_max = zernike_rho_max((n_pixels, n_pixels), px)
    prefactor = 2 * math.pi * old.wavelength**2 * cs_A * rho_max**3

    params = CTFParameters(
        defocus=(dfu + dfv) / 2 / 1e4,
        astigmatism=(dfu - dfv) / 2 / 1e4,
        astigmatism_angle=dfang,
        spherical_aberration=cs_A / 1e7,
        voltage=voltage,
        phase_shift=torch.rad2deg(ctf_params["phaseshift"]),
        amplitude_contrast=float(alpha),
        odd_zernike={
            "Z33c": -trefoil1 * rho_max**3,
            "Z33s": -trefoil2 * rho_max**3,
            "Z31c": -prefactor * tiltx,
            "Z31s": -prefactor * tilty,
        },
    )
    # specimen_absorption=True zeroes amplitude_contrast here, matching
    # old Aberration's alpha (accepted, never applied -- confirmed dead
    # code) -- this is the fair "matches old" comparison, not a claim that
    # amplitude contrast doesn't matter.
    new_tf = TransferFunction(
        n_pixels, px, aberration_model="nonlinear", specimen_absorption=True
    )
    with pytest.warns(UserWarning, match="specimen_absorption"):
        new_t = new_tf.transfer_function(params)

    assert old_t.shape == new_t.shape == (5, n_pixels, n_pixels)
    assert torch.allclose(old_t, new_t, atol=1e-4)


# ---------------------------------------------------------------------------
# Laser phase plate (torch_ctf.calc_LPP_ctf_2D). No old-Aberration parity is
# possible here -- Aberration only has a spatially-uniform phase_shift, not
# a physical LPP model -- and the underlying laser physics (relativistic
# ponderomotive phase from a standing wave) is specialized enough that
# reimplementing it independently isn't the right form of verification
# either. Instead: verify the *wiring* (matches a direct calc_LPP_ctf_2D
# call), verify DC-pinning/batching/validation all still hold, and check
# basic physical sanity (peak_phase_deg=0 is a near no-op).
# ---------------------------------------------------------------------------

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


def test_lpp_wiring_matches_direct_calc_LPP_ctf_2D():
    from torch_ctf import calc_LPP_ctf_2D

    params = CTFParameters(
        defocus=1.0, spherical_aberration=2.7, lpp_params=_LPP_KWARGS
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="linear")
    new_t = tf.transfer_function(params).squeeze()

    manual = calc_LPP_ctf_2D(
        defocus=1.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=VOLTAGE,
        spherical_aberration=2.7,
        amplitude_contrast=0.0,
        pixel_size=PIXEL_SIZE,
        image_shape=(N_PIXELS, N_PIXELS),
        rfft=False,
        fftshift=False,
        return_complex_ctf=True,
        **_LPP_KWARGS,
    )
    assert torch.allclose(new_t, manual, atol=1e-6)


def test_lpp_dc_pinned_in_nonlinear_mode():
    params = CTFParameters(
        defocus=1.0, spherical_aberration=2.7, lpp_params=_LPP_KWARGS
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    out = tf.transfer_function(params).squeeze()
    assert torch.allclose(out[0, 0], torch.tensor(1.0 + 0.0j), atol=1e-6)
    # Off-DC, the laser pattern must actually have an effect.
    assert not torch.allclose(out[3, 5], torch.tensor(1.0 + 0.0j), atol=1e-3)


def test_lpp_peak_phase_zero_is_a_real_upstream_nan_singularity():
    """torch_ctf.get_eta0_from_peak_phase_deg computes
    ``eta0 = eta0_test * peak_phase_deg / peak_phase_deg_test`` where both
    eta0_test and peak_phase_deg_test are themselves exactly zero when
    peak_phase_deg=0 (zero laser power) -- a genuine 0/0 upstream
    singularity, not a wiring bug on this side. Documented here so it isn't
    mistaken for a regression if hit again."""
    zero_power = {**_LPP_KWARGS, "peak_phase_deg": 0.0}
    params_lpp = CTFParameters(
        defocus=1.0, spherical_aberration=2.7, lpp_params=zero_power
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    out_lpp = tf.transfer_function(params_lpp).squeeze()
    assert torch.isnan(out_lpp).any()


def test_lpp_near_zero_peak_phase_is_near_noop():
    """Physical sanity check (not wiring): with negligible (but nonzero,
    avoiding the singularity above) laser power, the LPP transfer function
    should be close to the plain defocus-only CTF, not introduce spurious
    structure."""
    near_zero_power = {**_LPP_KWARGS, "peak_phase_deg": 1e-4}
    params_lpp = CTFParameters(
        defocus=1.0, spherical_aberration=2.7, lpp_params=near_zero_power
    )
    params_plain = CTFParameters(defocus=1.0, spherical_aberration=2.7)
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    out_lpp = tf.transfer_function(params_lpp).squeeze()
    out_plain = tf.transfer_function(params_plain).squeeze()
    assert torch.allclose(out_lpp, out_plain, atol=1e-3)


def test_lpp_mutually_exclusive_with_nonzero_phase_shift():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CTFParameters(defocus=1.0, phase_shift=10.0, lpp_params=_LPP_KWARGS)
    # A zero/default phase_shift alongside lpp_params is fine.
    CTFParameters(defocus=1.0, phase_shift=0.0, lpp_params=_LPP_KWARGS)
    CTFParameters(defocus=1.0, lpp_params=_LPP_KWARGS)


def test_lpp_missing_required_key_raises():
    with pytest.raises(ValueError, match="missing required keys"):
        CTFParameters(defocus=1.0, lpp_params={"NA": 0.1})


def test_lpp_unrecognized_key_raises():
    with pytest.raises(ValueError, match="unrecognized keys"):
        CTFParameters(defocus=1.0, lpp_params={**_LPP_KWARGS, "bogus_key": 1.0})


def test_lpp_combined_with_per_particle_defocus_and_trefoil():
    """LPP is always a single shared instrument config, but it must still
    compose correctly with genuinely per-particle terms (defocus, trefoil)
    -- exercising the same per-particle fallback loop the batched-Zernike
    regression tests use, now with calc_LPP_ctf_2D as the inner function."""
    from torch_ctf import calc_LPP_ctf_2D

    defocus = torch.tensor([0.8, 1.0, 1.2])
    trefoil_c = torch.tensor([0.1, -0.1, 0.05])
    params = CTFParameters(
        defocus=defocus,
        spherical_aberration=2.7,
        lpp_params=_LPP_KWARGS,
        odd_zernike={"Z33c": trefoil_c},
    )
    tf = TransferFunction(N_PIXELS, PIXEL_SIZE, aberration_model="nonlinear")
    out = tf.transfer_function(params)
    assert out.shape == (3, N_PIXELS, N_PIXELS)

    for i in range(3):
        manual = calc_LPP_ctf_2D(
            defocus=defocus[i].item(),
            astigmatism=0.0,
            astigmatism_angle=0.0,
            voltage=VOLTAGE,
            spherical_aberration=2.7,
            amplitude_contrast=0.0,
            pixel_size=PIXEL_SIZE,
            image_shape=(N_PIXELS, N_PIXELS),
            rfft=False,
            fftshift=False,
            odd_zernike_coeffs={"Z33c": trefoil_c[i]},
            return_complex_ctf=True,
            **_LPP_KWARGS,
        ).clone()
        manual[0, 0] = 1.0 + 0.0j  # DC pin, matching nonlinear mode
        assert torch.allclose(out[i], manual, atol=1e-6)
