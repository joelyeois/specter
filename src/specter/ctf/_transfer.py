"""Applies a torch-ctf-computed transfer function to a complex exit wave.

The piece torch-ctf deliberately doesn't provide: multiplying a computed
CTF into an actual wavefunction's Fourier transform (fft2 -> multiply ->
ifft2), plus specter's envelope stack, which has no torch-ctf equivalent
(that functionality lives in the separate, not-yet-adopted teamtomo
package torch-fourier-filter -- see aberrations._envelopes' own docstring).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch_ctf import calc_LPP_ctf_2D, calculate_ctf_2d

from ..aberrations import b_envelope, cc_envelope, cs_envelope, dose_envelope
from ..arrays import kgrid_2d
from ..constants import energy_to_wavelength
from ..fft import fft2, ifft2
from ._parameters import CTFParameters


class TransferFunction(nn.Module):
    """Applies a :class:`CTFParameters`-driven torch-ctf transfer function
    to a complex exit wave.

    Replaces ``aberrations.Aberration``'s phase-computation internals with
    torch-ctf, while keeping the parts torch-ctf doesn't provide (envelope
    application, the actual fft2/multiply/ifft2 step) local to specter.

    Parameters
    ----------
    n_pixels : int
        Exit wave side length in pixels.
    pixel_size : float
        Pixel size in Å.
    aberration_model : {"holography", "ctf"}, optional
        ``"holography"`` returns the full complex aberrated wave;
        ``"ctf"`` returns its real part. Both use the *complex* torch-ctf
        transfer function (``return_complex_ctf=True``), matching
        ``aberrations.Aberration``'s existing behaviour exactly. Adopting
        torch-ctf's dedicated real ``-sin(chi)`` weak-phase-object formula
        for a genuine projection-approximation exit wave is a deliberate,
        separate follow-up -- not done here, to keep this class a faithful,
        testable like-for-like replacement first.
    specimen_absorption : bool, optional
        If True (default), amplitude contrast is assumed to already be
        baked into the exit wave via ``scattering.complex_potential``
        (specter's default: alpha=0.1) *before* this transfer function is
        ever applied. Any nonzero ``ctf_params.amplitude_contrast`` is
        therefore zeroed here (with a warning) to avoid double-counting
        the same physical effect once at the specimen level and again at
        the lens level -- see the amplitude-contrast design discussion.
        Set False only when the exit wave came from a potential with no
        absorption applied upstream (e.g. alpha=0 was used everywhere).
    bfactor : float | torch.Tensor, optional
        Isotropic B-factor envelope in Å². None (default) disables
        it.
    convergence_angle : float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. None (default) disables it. Matches
        ``aberrations.Aberration``'s constructor-level convenience exactly.
    cc : float, optional
        Chromatic aberration coefficient in Å, used for the Cc
        (temporal coherence) envelope. None (default) disables it.
    energy_spread : float, optional
        FWHM of the beam energy spread in eV, used by the Cc envelope.
        Default 0.7.
    deltaV_V : float, optional
        Relative high-voltage instability, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I : float, optional
        Relative objective-lens current instability, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope : bool, optional
        Whether to apply the Grant & Grigorieff (2015) cumulative-dose
        envelope, using ``ctf_params.dose`` (see :class:`CTFParameters`).
        Default False.
    """

    def __init__(
        self,
        n_pixels: int,
        pixel_size: float,
        aberration_model: str = "holography",
        specimen_absorption: bool = True,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
    ) -> None:
        super().__init__()
        if aberration_model not in ("holography", "ctf"):
            raise ValueError(f"Unknown aberration_model: {aberration_model!r}")
        self.n_pixels = n_pixels
        self.pixel_size = pixel_size
        self.aberration_model = aberration_model
        self.specimen_absorption = specimen_absorption
        self.convergence_angle = convergence_angle
        self.cc = cc
        self.energy_spread = energy_spread
        self.deltaV_V = deltaV_V
        self.deltaI_I = deltaI_I
        self.dose_envelope = dose_envelope

        _, _, KX, KY = kgrid_2d(n_pixels, pixel_size)
        k2 = (KX**2 + KY**2).unsqueeze(0)
        self.register_buffer("k2", k2)
        self.register_buffer("k", torch.sqrt(k2))

        if bfactor is None:
            self.bfactor: torch.Tensor | None = None
        else:
            self.register_buffer(
                "bfactor", torch.as_tensor(bfactor, dtype=torch.float32).flatten()
            )

    def _resolve_amplitude_contrast(self, kwargs: dict) -> dict:
        """Enforce the specimen-absorption double-counting guard in place."""
        ac = kwargs["amplitude_contrast"]
        if self.specimen_absorption and torch.any(ac != 0):
            warnings.warn(
                "specimen_absorption=True but ctf_params.amplitude_contrast "
                "is nonzero -- amplitude contrast is assumed to already be "
                "baked into the exit wave via scattering.complex_potential, "
                "so this term is being zeroed here to avoid double-counting "
                "it at both the specimen and lens stages. Pass "
                "specimen_absorption=False if that assumption is wrong for "
                "this run.",
                stacklevel=3,
            )
            kwargs = {**kwargs, "amplitude_contrast": torch.zeros_like(ac)}
        return kwargs

    def _zero_dc_for_holography(self, transfer: torch.Tensor) -> torch.Tensor:
        """Pin the DC (k=0) transfer-function value to 1+0j for holography mode.

        Every genuinely k-dependent chi term (defocus, astigmatism, Cs,
        trefoil, beam tilt, tetrafoil, ...) is already exactly zero at k=0
        by construction. The only terms that are *not* -- phase_shift and
        amplitude_contrast -- are both physically properties of the
        *scattered* wave relative to the unscattered reference beam, not of
        the reference beam itself (a real phase plate shifts the scattered
        beam relative to the central beam; amplitude contrast describes how
        the scattered signal alone splits into absorption/phase channels --
        neither has anything to say about the unscattered beam).

        In holography mode the full complex wave is preserved, so any
        residual constant term left in chi at k=0 becomes a spatially
        uniform phase multiplying the *entire* real-space output (by the
        Fourier shift property) -- physically inert (no effect on intensity
        or on any interferometric/relative-phase quantity), so it must be
        removed for the phase-plate/amplitude-contrast terms to have their
        intended effect. This isn't optional or mode-tunable: it's the same
        DC-pinning aberrations.Aberration's phaseshift() already enforces
        for "holography" (there, only phase_shift could ever be nonzero at
        DC; amplitude_contrast was unimplemented). "ctf" mode does *not*
        get this treatment -- taking the real part downstream makes a
        uniform phase directly observable there, matching torch-ctf's own
        convention of never special-casing DC (see
        test_phase_shift_matches_old_aberration_ctf_model).
        """
        transfer = transfer.clone()
        transfer[..., 0, 0] = torch.ones(
            (), dtype=transfer.dtype, device=transfer.device
        )
        return transfer

    def _call_calculate_ctf_2d(self, kwargs: dict) -> torch.Tensor:
        """calculate_ctf_2d(**kwargs, return_complex_ctf=True) -- or
        calc_LPP_ctf_2D if ``kwargs["lpp_params"]`` is set (always a single
        shared laser-instrument config, never per-particle, so it needs no
        special-casing below beyond being passed through unchanged to every
        call) -- working around a real upstream limitation: torch_ctf's
        apply_odd_zernikes/apply_even_zernikes initialize their phase
        accumulator as ``torch.zeros_like(rho)`` (shape (H, W), *not*
        batched) and then accumulate into it in place, which raises a
        broadcast error the moment any Zernike coefficient is a
        per-particle tensor rather than a scalar -- confirmed against a
        real multi-particle .cs file, where trefoil/beam-tilt-derived
        Zernike coefficients are legitimately per-particle. Not fixable
        from this side without a torch_ctf PR, so when a batched Zernike
        coefficient is present this falls back to computing each
        particle's transfer function with its own scalar Zernike
        coefficients (the code path already proven correct) and stacks the
        results -- slower, but correct.

        Before that fallback is even considered, every *non-learnable*
        Zernike coefficient that's uniform across its batch dimension is
        collapsed to a single value (see ``_collapse_if_uniform``). This
        matters in practice: cryoSPARC's .cs schema stores every CTF term
        as a flat per-particle array regardless of whether the underlying
        value is actually per-particle, and trefoil/beam-tilt/Cs are
        physically per-optics-group properties (shared across potentially
        thousands of particles), not truly per-particle like defocus is --
        confirmed against a real .cs file, where a 200-particle sample had
        198 unique defocus values but exactly 1 unique trefoil/beam-tilt/Cs
        value. So the common case is a "batched" tensor that's secretly
        uniform, and collapsing it lets the fast vectorized path run
        instead of the slow per-particle loop below. Restricted to
        non-learnable coefficients specifically: collapsing a *learnable*
        per-particle tensor that happens to be uniformly initialized would
        route the gradient through only the one retained element, silently
        leaving every other element's gradient at zero instead of each
        particle getting its own -- a real correctness bug, not just a
        missed optimization, so learnable coefficients always take the
        slow-but-correct path regardless of current uniformity.
        """
        lpp_params = kwargs.pop("lpp_params", None)
        fn = calc_LPP_ctf_2D if lpp_params is not None else calculate_ctf_2d
        extra_kwargs = lpp_params or {}

        for key in ("even_zernike_coeffs", "odd_zernike_coeffs"):
            if kwargs.get(key):
                kwargs[key] = {
                    name: _prepare_zernike_coeff(coeff)
                    for name, coeff in kwargs[key].items()
                }

        zernike_dicts = [
            kwargs.get("even_zernike_coeffs"),
            kwargs.get("odd_zernike_coeffs"),
        ]
        batchsize = None
        for zdict in zernike_dicts:
            if not zdict:
                continue
            for coeff in zdict.values():
                if coeff.numel() > 1:
                    batchsize = coeff.numel()
        if batchsize is None:
            return fn(**kwargs, **extra_kwargs, return_complex_ctf=True)

        per_particle_outputs = []
        for i in range(batchsize):
            # Slice to a length-1 leading dim (not a bare 0-d scalar):
            # calculate_ctf_2d's internal `"... -> ... 1 1"` reshape only
            # *prepends* a new leading batch dim when the input already has
            # one dim to align past the trailing (1, 1) -- a 0-d input
            # reshapes to (1, 1), which has the *same* rank as the (H, W)
            # grid and so aligns with it directly (broadcasting to (H, W),
            # no batch dim at all), while a (1,)-shaped input reshapes to
            # (1, 1, 1), which outranks (H, W) and correctly broadcasts to
            # (1, H, W). Concatenating per-particle (H, W) outputs along
            # dim=0 silently produces a wrong (batchsize * H, W) shape;
            # keeping a (1, H, W) output per particle is what makes
            # concatenation reproduce the true (batchsize, H, W) shape.
            particle_kwargs = dict(kwargs)
            for key in ("even_zernike_coeffs", "odd_zernike_coeffs"):
                if kwargs.get(key):
                    particle_kwargs[key] = {
                        name: (
                            coeff.flatten()[i : i + 1] if coeff.numel() > 1 else coeff
                        )
                        for name, coeff in kwargs[key].items()
                    }
            for key in (
                "defocus",
                "astigmatism",
                "astigmatism_angle",
                "voltage",
                "spherical_aberration",
                "amplitude_contrast",
                *(["phase_shift"] if lpp_params is None else []),
            ):
                value = kwargs[key]
                if value.numel() > 1:
                    particle_kwargs[key] = value.flatten()[i : i + 1]
            per_particle_outputs.append(
                fn(**particle_kwargs, **extra_kwargs, return_complex_ctf=True)
            )
        return torch.cat(per_particle_outputs, dim=0)

    def transfer_function(
        self, ctf_params: CTFParameters, idx: torch.Tensor | slice | int | None = None
    ) -> torch.Tensor:
        """Compute the complex transfer function exp(-i*chi) for the given
        particle indices."""
        kwargs = ctf_params.torch_ctf_kwargs(
            self.pixel_size, (self.n_pixels, self.n_pixels), idx=idx
        )
        kwargs = self._resolve_amplitude_contrast(kwargs)
        transfer = self._call_calculate_ctf_2d(kwargs)

        if self.aberration_model == "holography":
            transfer = self._zero_dc_for_holography(transfer)

        if self.bfactor is not None:
            bfactor = self.bfactor.view(-1, 1, 1)
            if torch.any(bfactor != 0):
                transfer = transfer * b_envelope(self.k2, bfactor)

        # Cs/Cc/dose envelopes: aberrations._envelopes expects Å
        # throughout, but CTFParameters' defocus/spherical_aberration are
        # in torch-ctf's own units (micrometers/mm) -- converted here, at
        # the point of use, rather than changing CTFParameters' units.
        if self.convergence_angle is not None:
            cs_angstrom = (kwargs["spherical_aberration"] * 1e7).view(-1, 1, 1)
            # CTFParameters.defocus is already the (dfu + dfv) / 2 mean,
            # matching aberrations.Aberration's own defocus_avg.
            defocus_avg_angstrom = (kwargs["defocus"] * 1e4).view(-1, 1, 1)
            wavelength = energy_to_wavelength(kwargs["voltage"])
            transfer = transfer * cs_envelope(
                self.k,
                wavelength,
                cs_angstrom,
                defocus_avg_angstrom,
                self.convergence_angle,
            )

        if self.cc is not None:
            wavelength = energy_to_wavelength(kwargs["voltage"])
            voltage_v = kwargs["voltage"] * 1e3  # kV -> V (numerically eV-equivalent)
            transfer = transfer * cc_envelope(
                self.k2,
                wavelength,
                self.cc,
                voltage_v,
                self.energy_spread,
                self.deltaV_V,
                self.deltaI_I,
            )

        if self.dose_envelope and ctf_params.dose is not None:
            dose = ctf_params.dose.gather(idx).view(-1, 1, 1)
            transfer = transfer * dose_envelope(self.k, dose)

        return transfer

    def forward(
        self,
        exitwave: torch.Tensor,
        ctf_params: CTFParameters,
        idx: torch.Tensor | slice | int | None = None,
    ) -> torch.Tensor:
        """Apply the transfer function to a complex exit wave.

        Parameters
        ----------
        exitwave : torch.Tensor
            Complex-valued exit wave, shape (B, Y, X).
        ctf_params : CTFParameters
            CTF parameters for this batch.
        idx : torch.Tensor | slice | int | None, optional
            Which particles' parameters to use. None selects all (assumes
            `ctf_params`'s per-particle fields already match `exitwave`'s
            batch order/size).

        Returns
        -------
        torch.Tensor
            Aberrated exit wave. Complex for ``"holography"``, real for
            ``"ctf"``.
        """
        transfer = self.transfer_function(ctf_params, idx=idx)
        aberrated = ifft2(fft2(exitwave) * transfer)
        return torch.real(aberrated) if self.aberration_model == "ctf" else aberrated


def _collapse_if_uniform(coeff: torch.Tensor) -> torch.Tensor:
    """Collapse a >1-element, non-learnable, uniform-valued tensor down to
    a single element; pass everything else through unchanged.

    See :meth:`TransferFunction._call_calculate_ctf_2d` for why this
    matters and why it's restricted to non-learnable coefficients.
    """
    if (
        coeff.numel() > 1
        and not coeff.requires_grad
        and torch.all(coeff == coeff.flatten()[0])
    ):
        return coeff.flatten()[:1]
    return coeff


def _prepare_zernike_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """``_collapse_if_uniform`` then, if a single element remains, flatten
    it from ``_broadcast_for_grid``'s 3-D ``(1, 1, 1)`` shape down to 1-D
    ``(1,)``.

    Needed even for a genuinely single-particle call (never multi-particle
    to begin with, so ``_collapse_if_uniform`` never touches it):
    ``apply_odd_zernikes``/``apply_even_zernikes`` initialize their phase
    accumulator as ``torch.zeros_like(rho)`` (unbatched, shape ``(H, W)``)
    and accumulate in place. Broadcasting a *2-D* ``rho**3`` against a
    *3-D* ``(1, 1, 1)`` coefficient prepends a new leading batch dim
    (result ``(1, H, W)``), which the in-place accumulate can't absorb --
    the same upstream bug ``_call_calculate_ctf_2d`` documents for genuine
    multi-particle coefficients, but triggered here purely by shape rank,
    independent of batch size. A 1-D ``(1,)`` coefficient, by contrast,
    aligns against ``rho``'s trailing (W) dimension and broadcasts to
    ``(H, W)`` with no new dim -- confirmed via
    ``test_micrograph_generator_matches_across_backends``/
    ``test_tilt_series_generator_matches_across_backends``, both
    single-particle calls with nonzero trefoil/beam-tilt terms. Flattening
    is a reshape (a view), so it's always safe regardless of
    ``requires_grad``.
    """
    coeff = _collapse_if_uniform(coeff)
    return coeff.flatten() if coeff.numel() == 1 else coeff
