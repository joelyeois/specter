"""Parameter containers bridging specter's differentiable, per-particle
world and torch-ctf's pure, stateless functions.

torch-ctf's functions (``calculate_ctf_2d`` etc.) take plain floats/Tensors
and have no notion of "this one is learnable" -- that bookkeeping has to
live somewhere else. ``CTFParameters`` is that "somewhere else": it stores
each named CTF term as either a fixed buffer or an ``nn.Parameter``
(visible to an optimiser via ``.parameters()``), lets any term be omitted
(falling back to a physically neutral default), and materializes the exact
kwargs torch-ctf expects on demand.

Notes
-----
Units follow torch-ctf's own convention throughout -- defocus/astigmatism
in micrometers, voltage in kV, spherical_aberration in millimeters,
phase_shift in degrees, pixel_size in Angstrom, amplitude_contrast as a
[0, 1] fraction. This is *not* the same convention as specter's older
``ctf_params`` dict (``aberrations.Aberration``), which takes dfu/dfv/cs
directly in Angstrom. Callers migrating an existing Angstrom-based dict
must convert explicitly (dfu/dfv Angstrom -> micrometers: divide by 1e4;
cs Angstrom -> mm: divide by 1e7) -- no implicit conversion happens here.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

# Physically-neutral defaults for every scalar torch-ctf requires but that a
# caller may reasonably omit. `defocus` has no default -- it's the one term
# that must always be provided.
_DEFAULTS: dict[str, float] = {
    "astigmatism": 0.0,
    "astigmatism_angle": 0.0,
    "phase_shift": 0.0,
    "amplitude_contrast": 0.0,
    "voltage": 300.0,
    "spherical_aberration": 2.7,
}

_SCALAR_NAMES = (
    "defocus",
    "astigmatism",
    "astigmatism_angle",
    "voltage",
    "spherical_aberration",
    "amplitude_contrast",
    "phase_shift",
)

# Required keys for `lpp_params` -- exactly torch_ctf.calc_LPP_ctf_2D's laser
# geometry arguments. "dual_laser" is optional there (defaults False) so it's
# not required here either.
_LPP_PARAM_NAMES = (
    "NA",
    "laser_wavelength_angstrom",
    "focal_length_angstrom",
    "laser_xy_angle_deg",
    "laser_xz_angle_deg",
    "laser_long_offset_angstrom",
    "laser_trans_offset_angstrom",
    "laser_polarization_angle_deg",
    "peak_phase_deg",
)


class ParamField(nn.Module):
    """A single named CTF term: a storage tensor, optionally trainable,
    optionally shared across a group and gathered per-particle.

    Parameters
    ----------
    value : torch.Tensor
        Storage tensor. Shape () for a value shared by every particle,
        shape (N,) for a per-particle value, or shape (G,) for a
        per-group value combined with `group_index`.
    learnable : bool, optional
        If True, `value` is registered as an ``nn.Parameter`` (visible to
        an optimiser via ``.parameters()``). If False (default), it's a
        fixed buffer.
    group_index : torch.Tensor, optional
        Integer tensor of shape (N,) mapping each particle to a row of a
        group-shaped `value` (e.g. a micrograph/optics-group id) -- the
        RELION-style "refined per optics group, not per particle" case.
        None (default) means `value` is already per-particle, or global.
    per_particle : bool or None, optional
        Whether `value`'s leading axis indexes particles (so it should be
        sliced by `idx` in :meth:`gather`). None (default) infers this
        from ``value.ndim`` (0-d = global, >0-d = per-particle) or from
        `group_index` being given (always per-particle via gather).
        Pass explicitly for compound-shaped globals, e.g. a single
        ``[bx, by]`` beam-tilt vector of shape (2,) that is *not*
        per-particle despite having ndim > 0.
    """

    def __init__(
        self,
        value: torch.Tensor,
        learnable: bool = False,
        group_index: torch.Tensor | None = None,
        per_particle: bool | None = None,
    ) -> None:
        super().__init__()
        if learnable:
            self.value = nn.Parameter(value)
        else:
            self.register_buffer("value", value)

        if group_index is not None:
            self.register_buffer("_group_index", group_index)
            self._per_particle = True
        else:
            self._group_index = None
            self._per_particle = (
                value.dim() > 0 if per_particle is None else per_particle
            )

    def gather(self, idx: torch.Tensor | slice | int | None = None) -> torch.Tensor:
        """Resolve this field to a tensor for particle indices `idx`.

        `idx=None` (the default) selects every particle -- or, for a
        global/group-shared field, simply returns the shared value
        unindexed.
        """
        if not self._per_particle:
            return self.value
        if self._group_index is not None:
            gi = self._group_index if idx is None else self._group_index[idx]
            return self.value[gi]
        return self.value if idx is None else self.value[idx]


class CTFParameters(nn.Module):
    """Sparse, partially-learnable set of CTF parameters for torch-ctf.

    Only ``defocus`` is required. Every other scalar term defaults to a
    physically neutral value (see module-level ``_DEFAULTS``) when
    omitted -- ``CTFParameters(defocus=dfu)`` alone is valid and will run.
    Zernike terms are sparse dicts: a key that's absent is never computed
    by torch-ctf (matching its own None-means-skip convention in
    ``apply_even_zernikes``/``apply_odd_zernikes``), so
    ``odd_zernike={"Z33c": 0.1}`` enables trefoil alone without needing to
    touch beam tilt or any other higher-order term.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused. Required.
    astigmatism : float | torch.Tensor, optional
        Astigmatism magnitude in micrometers. Default 0.0.
    astigmatism_angle : float | torch.Tensor, optional
        Astigmatism angle in degrees. Default 0.0.
    voltage : float | torch.Tensor, optional
        Accelerating voltage in kV. Default 300.0.
    spherical_aberration : float | torch.Tensor, optional
        Spherical aberration in mm. Default 2.7.
    amplitude_contrast : float | torch.Tensor, optional
        Amplitude contrast fraction, in [0, 1]. Default 0.0 -- deliberately
        *not* the conventional ~0.07-0.1, because by default specter's own
        multislice/projection scattering already applies amplitude
        contrast at the specimen level via ``scattering.complex_potential``
        (see ``TransferFunction``'s ``specimen_absorption`` flag); a
        nonzero default here would silently double-count it for anyone
        using this class through :class:`TransferFunction` unchanged.
    phase_shift : float | torch.Tensor, optional
        Uniform (Volta/hole) phase-plate phase shift in degrees. Default
        0.0. Mutually exclusive with `lpp_params` -- a laser phase plate's
        spatially-varying pattern replaces a uniform phase_shift physically
        (torch_ctf.calc_LPP_ctf_2D doesn't take a phase_shift argument at
        all), so passing both a nonzero phase_shift and lpp_params raises.
    even_zernike, odd_zernike : dict[str, float | torch.Tensor], optional
        Sparse Zernike coefficients, e.g. ``{"Z44c": 0.1}`` or
        ``{"Z33c": 0.1, "Z33s": 0.2}``. None/omitted (default) means no
        higher-order aberrations at all.
    beam_tilt_mrad : Sequence[float] | torch.Tensor, optional
        ``[bx, by]`` in mrad. None (default) disables beam tilt.
    lpp_params : dict[str, float], optional
        Laser phase plate geometry, dispatching to
        ``torch_ctf.calc_LPP_ctf_2D`` instead of ``calculate_ctf_2d``.
        Always a single shared (non-per-particle) instrument configuration
        -- one physical laser setup per simulated dataset, not something
        that varies particle to particle. Must contain every key in
        ``{"NA", "laser_wavelength_angstrom", "focal_length_angstrom",
        "laser_xy_angle_deg", "laser_xz_angle_deg",
        "laser_long_offset_angstrom", "laser_trans_offset_angstrom",
        "laser_polarization_angle_deg", "peak_phase_deg"}``, plus an
        optional ``"dual_laser"`` bool (default False). None (default)
        disables it -- standard uniform-phase-shift / no-phase-plate path.
        Do not set ``peak_phase_deg`` to exactly 0.0 to mean "negligible
        laser power": torch_ctf.get_eta0_from_peak_phase_deg divides
        ``eta0_test * peak_phase_deg / peak_phase_deg_test`` where both the
        numerator and ``peak_phase_deg_test`` are themselves exactly zero
        at ``peak_phase_deg=0`` -- a genuine 0/0 upstream singularity that
        produces NaN, not a no-op. Use a small nonzero value (e.g. 1e-4) or
        omit ``lpp_params`` entirely instead.
    dose : float | torch.Tensor, optional
        Cumulative electron dose in e-/Angstrom^2, for
        :class:`~specter.ctf.TransferFunction`'s dose envelope (Grant &
        Grigorieff 2015). Not a torch-ctf argument at all -- like
        ``aberrations.Aberration``, dose describes a specific exposure, not
        a lens/specimen CTF property, so it's applied as a separate
        post-hoc envelope rather than folded into ``torch_ctf_kwargs()``.
        None (default) disables the dose envelope regardless of whether
        ``TransferFunction`` was constructed with ``dose_envelope=True``.
    learnable : Iterable[str], optional
        Names to promote to ``nn.Parameter`` -- any of the scalar names
        above, a Zernike key (e.g. ``"Z33c"``), or ``"beam_tilt_mrad"``.
        `lpp_params` is never learnable (forward-simulation-only, plain
        floats). Everything else is a fixed buffer. Default: nothing is
        learnable (pure forward simulation).
    group_index : dict[str, torch.Tensor], optional
        Per-field group index (see :class:`ParamField`) for terms shared
        across a group of particles (e.g.
        ``{"astigmatism": micrograph_id}``) rather than fully per-particle
        or fully global.
    """

    def __init__(
        self,
        defocus: float | torch.Tensor,
        astigmatism: float | torch.Tensor | None = None,
        astigmatism_angle: float | torch.Tensor | None = None,
        voltage: float | torch.Tensor | None = None,
        spherical_aberration: float | torch.Tensor | None = None,
        amplitude_contrast: float | torch.Tensor | None = None,
        phase_shift: float | torch.Tensor | None = None,
        even_zernike: dict[str, float | torch.Tensor] | None = None,
        odd_zernike: dict[str, float | torch.Tensor] | None = None,
        beam_tilt_mrad: Iterable[float] | torch.Tensor | None = None,
        lpp_params: dict[str, float] | None = None,
        dose: float | torch.Tensor | None = None,
        learnable: Iterable[str] = (),
        group_index: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        learnable = set(learnable)
        group_index = group_index or {}

        if lpp_params is not None:
            missing = set(_LPP_PARAM_NAMES) - lpp_params.keys()
            if missing:
                raise ValueError(
                    f"lpp_params is missing required keys: {sorted(missing)}"
                )
            extra = lpp_params.keys() - set(_LPP_PARAM_NAMES) - {"dual_laser"}
            if extra:
                raise ValueError(f"lpp_params has unrecognized keys: {sorted(extra)}")
            if (
                phase_shift is not None
                and float(torch.as_tensor(phase_shift).max()) != 0.0
            ):
                raise ValueError(
                    "lpp_params and a nonzero phase_shift are mutually exclusive -- "
                    "a laser phase plate's spatially-varying pattern replaces a "
                    "uniform phase_shift physically; torch_ctf.calc_LPP_ctf_2D "
                    "doesn't accept a phase_shift argument at all."
                )
            self.lpp_params: dict[str, float] | None = {
                "dual_laser": False,
                **lpp_params,
            }
        else:
            self.lpp_params = None

        provided = dict(
            defocus=defocus,
            astigmatism=astigmatism,
            astigmatism_angle=astigmatism_angle,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
            phase_shift=phase_shift,
        )
        self.fields = nn.ModuleDict()
        for name in _SCALAR_NAMES:
            v = provided[name]
            if v is None:
                if name not in _DEFAULTS:
                    raise ValueError(f"'{name}' has no default and must be provided")
                v = _DEFAULTS[name]
            self.fields[name] = ParamField(
                torch.as_tensor(v, dtype=torch.float32),
                learnable=name in learnable,
                group_index=group_index.get(name),
            )

        self.even_zernike = nn.ModuleDict(
            {
                k: ParamField(
                    torch.as_tensor(v, dtype=torch.float32),
                    learnable=k in learnable,
                    group_index=group_index.get(k),
                )
                for k, v in (even_zernike or {}).items()
            }
        )
        self.odd_zernike = nn.ModuleDict(
            {
                k: ParamField(
                    torch.as_tensor(v, dtype=torch.float32),
                    learnable=k in learnable,
                    group_index=group_index.get(k),
                )
                for k, v in (odd_zernike or {}).items()
            }
        )
        if beam_tilt_mrad is not None:
            self.beam_tilt_mrad: ParamField | None = ParamField(
                torch.as_tensor(beam_tilt_mrad, dtype=torch.float32),
                learnable="beam_tilt_mrad" in learnable,
                per_particle=False,
            )
        else:
            self.beam_tilt_mrad = None

        if dose is not None:
            self.dose: ParamField | None = ParamField(
                torch.as_tensor(dose, dtype=torch.float32),
                learnable="dose" in learnable,
                group_index=group_index.get("dose"),
            )
        else:
            self.dose = None

    def torch_ctf_kwargs(
        self,
        pixel_size: float,
        image_shape: tuple[int, int],
        idx: torch.Tensor | slice | int | None = None,
        rfft: bool = False,
        fftshift: bool = False,
    ) -> dict:
        """Materialize the exact kwargs ``torch_ctf.calculate_ctf_2d`` (and
        siblings) expect, gathered for the given particle indices.

        Parameters
        ----------
        pixel_size : float
            Pixel size in Angstrom.
        image_shape : tuple[int, int]
            Shape of the 2D image the CTF is being computed for.
        idx : torch.Tensor | slice | int | None, optional
            Particle indices to select. None (default) selects every
            particle.
        rfft, fftshift : bool, optional
            Passed straight through to torch-ctf. Default False for both,
            matching specter's existing full-complex-FFT convention.

        Returns
        -------
        dict
            Ready to pass as ``torch_ctf.calculate_ctf_2d(**kwargs)``.
        """
        out = {name: field.gather(idx) for name, field in self.fields.items()}
        out["pixel_size"] = pixel_size
        out["image_shape"] = image_shape
        out["rfft"] = rfft
        out["fftshift"] = fftshift
        # torch_ctf.apply_even_zernikes/apply_odd_zernikes multiply
        # `coeff * rho**n * trig(...)` directly with no reshaping of `coeff`
        # -- unlike the 7 "core" scalar terms above, which calculate_ctf_2d's
        # own _prepare_inputs reshapes to (..., 1, 1) internally. A
        # per-particle Zernike coefficient (shape (N,)) must therefore be
        # reshaped to (N, 1, 1) *here* so it broadcasts against the (H, W)
        # frequency grid, matching aberrations.Aberration's own
        # `.view(-1, 1, 1)` convention for the same terms.
        if len(self.even_zernike):
            out["even_zernike_coeffs"] = {
                k: _broadcast_for_grid(f.gather(idx))
                for k, f in self.even_zernike.items()
            }
        if len(self.odd_zernike):
            out["odd_zernike_coeffs"] = {
                k: _broadcast_for_grid(f.gather(idx))
                for k, f in self.odd_zernike.items()
            }
        if self.beam_tilt_mrad is not None:
            out["beam_tilt_mrad"] = self.beam_tilt_mrad.gather(idx)
        if self.lpp_params is not None:
            # calc_LPP_ctf_2D has no phase_shift parameter -- the laser
            # pattern replaces it entirely (enforced mutually exclusive at
            # construction time already).
            del out["phase_shift"]
            out["lpp_params"] = self.lpp_params
        return out


def _broadcast_for_grid(value: torch.Tensor) -> torch.Tensor:
    """Reshape a per-particle scalar (N,) to (N, 1, 1); leave a global
    scalar (0-d) untouched."""
    return value.view(-1, 1, 1) if value.dim() > 0 else value
