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

from ..aberrations import b_envelope
from ..arrays import kgrid_2d
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
        Pixel size in Angstrom.
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
        Isotropic B-factor envelope in Angstrom^2. None (default) disables
        it. The only envelope ported so far -- Cs/Cc/dose envelopes need
        unit conversions (CTFParameters' defocus/cs are in
        micrometers/mm, aberrations._envelopes expects Angstrom) that
        aren't wired up yet; see TODO in :meth:`forward`.
    """

    def __init__(
        self,
        n_pixels: int,
        pixel_size: float,
        aberration_model: str = "holography",
        specimen_absorption: bool = True,
        bfactor: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if aberration_model not in ("holography", "ctf"):
            raise ValueError(f"Unknown aberration_model: {aberration_model!r}")
        self.n_pixels = n_pixels
        self.pixel_size = pixel_size
        self.aberration_model = aberration_model
        self.specimen_absorption = specimen_absorption

        _, _, KX, KY = kgrid_2d(n_pixels, pixel_size)
        self.register_buffer("k2", (KX**2 + KY**2).unsqueeze(0))

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
        results -- slower, but correct. The fast vectorized call below
        still runs whenever every Zernike coefficient is scalar (shared
        across particles) or absent, which is the common case.
        """
        lpp_params = kwargs.pop("lpp_params", None)
        fn = calc_LPP_ctf_2D if lpp_params is not None else calculate_ctf_2d
        extra_kwargs = lpp_params or {}

        zernike_dicts = [
            kwargs.get("even_zernike_coeffs"),
            kwargs.get("odd_zernike_coeffs"),
        ]
        batch_size = None
        for zdict in zernike_dicts:
            if not zdict:
                continue
            for coeff in zdict.values():
                if coeff.numel() > 1:
                    batch_size = coeff.numel()
        if batch_size is None:
            return fn(**kwargs, **extra_kwargs, return_complex_ctf=True)

        per_particle_outputs = []
        for i in range(batch_size):
            # Slice to a length-1 leading dim (not a bare 0-d scalar):
            # calculate_ctf_2d's internal `"... -> ... 1 1"` reshape only
            # *prepends* a new leading batch dim when the input already has
            # one dim to align past the trailing (1, 1) -- a 0-d input
            # reshapes to (1, 1), which has the *same* rank as the (H, W)
            # grid and so aligns with it directly (broadcasting to (H, W),
            # no batch dim at all), while a (1,)-shaped input reshapes to
            # (1, 1, 1), which outranks (H, W) and correctly broadcasts to
            # (1, H, W). Concatenating per-particle (H, W) outputs along
            # dim=0 silently produces a wrong (batch_size * H, W) shape;
            # keeping a (1, H, W) output per particle is what makes
            # concatenation reproduce the true (batch_size, H, W) shape.
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
        # TODO: Cc/Cs-envelope and dose-envelope parity with
        # aberrations.Aberration -- needs defocus (um -> Angstrom, x1e4)
        # and spherical_aberration (mm -> Angstrom, x1e7) unit conversion
        # at the call site before reusing aberrations._envelopes.cs_envelope
        # / cc_envelope / dose_envelope unchanged.
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
