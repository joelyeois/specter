"""Compatibility bridge from specter's legacy, Angstrom-based ctf_params
dict convention (``aberrations.Aberration``, ``BaseImager._ctf_batch()``)
to the new torch-ctf-backed :class:`~specter.ctf.CTFParameters`.

This exists so ``BaseImager`` (and anything else holding a legacy
``ctf_params`` dict) can opt into the torch-ctf backend without every call
site needing to change: :class:`LegacyAberrationAdapter` has the exact same
call signature as ``aberrations.Aberration`` (``forward(exitwave,
ctf_params_dict)``), so ``self.aberration = LegacyAberrationAdapter(...)``
is the only change needed anywhere.

Every unit conversion and Zernike-coefficient mapping here (dfu/dfv
Angstrom -> defocus/astigmatism micrometers, cs Angstrom -> mm, phaseshift
radians -> degrees, trefoil1/trefoil2 -> Z33c/Z33s, tiltx/tilty ->
Z31c/Z31s) is exactly what's verified term-by-term against
``aberrations.Aberration`` in tests/test_ctf_transfer.py, including
against a real multi-particle CryoSPARC .cs file -- nothing new is
introduced here, this module only wires that already-validated conversion
into the legacy dict-based calling convention.

TODO (unit convention): the ``ctf_params`` dict handled here mirrors
CryoSPARC's own parameterization (dfu/dfv/cs in Angstrom, tiltx/tilty/
phaseshift in radians, trefoil in Angstrom^3) -- it is not torch-ctf's
native convention (defocus in um, Cs in mm, angles in degrees,
dimensionless Zernike coefficients normalized by the grid's rho_max, see
``_units.zernike_rho_max``). Someone coming from RELION, or who already
has parameters in torch-ctf's own units, has no equally convenient entry
point today -- ``CTFParameters`` accepts native units directly, so
nothing is physically blocked, but there is no wrapper analogous to
``LegacyAberrationAdapter`` for that convention. Build one (or document
constructing ``CTFParameters`` directly) once a concrete RELION-style
consumer shows up.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ..constants import energy_to_wavelength
from ._parameters import CTFParameters
from ._transfer import TransferFunction
from ._units import zernike_rho_max


def ctf_params_dict_to_parameters(
    ctf_params: dict[str, Any],
    pixel_size: float,
    image_shape: tuple[int, int],
    voltage: float,
) -> CTFParameters:
    """Convert a legacy Angstrom/radian-based ``ctf_params`` dict (the
    ``aberrations.Aberration`` / ``BaseImager._ctf_batch()`` convention)
    into a :class:`CTFParameters`.

    Parameters
    ----------
    ctf_params : dict[str, Any]
        Any subset of ``{"dfu", "dfv", "dfang", "cs", "phaseshift",
        "tiltx", "tilty", "trefoil1", "trefoil2", "dose"}``, matching
        exactly what ``BaseImager._ctf_batch()`` produces (and what
        ``aberrations.Aberration.transfer_function()`` accepts). Every key
        is optional -- an absent key means "no effect", the same as
        ``aberrations.Aberration``'s own ``ctf_params.get(key, self.zero)``
        fallback. ``"bfactor"`` is deliberately ignored here (see
        :class:`LegacyAberrationAdapter` -- it's a
        ``TransferFunction``-construction-time argument, not a per-call
        one, since in practice it's always the same constant value across
        every particle anyway).
    pixel_size : float
        Pixel size in Angstrom.
    image_shape : tuple[int, int]
        Shape of the 2D image the CTF is being computed for -- needed for
        the trefoil/beam-tilt Zernike-coefficient rescale (see
        ``zernike_rho_max``).
    voltage : float
        Accelerating voltage in kV.

    Returns
    -------
    CTFParameters
    """
    dfu = ctf_params.get("dfu")
    dfv = ctf_params.get("dfv", dfu)
    dfang = ctf_params.get("dfang", 0.0)
    cs_angstrom = ctf_params.get("cs", 0.0)
    phaseshift_rad = ctf_params.get("phaseshift", 0.0)
    tiltx = ctf_params.get("tiltx", 0.0)
    tilty = ctf_params.get("tilty", 0.0)
    trefoil1 = ctf_params.get("trefoil1", 0.0)
    trefoil2 = ctf_params.get("trefoil2", 0.0)

    if dfu is None:
        # No defocus term at all -- e.g. ctf_params=None upstream. Matches
        # Aberration's own behaviour: the "dfu"/"dfv"/"dfang" branch is
        # skipped entirely, chi_defocus stays 0.
        defocus_um: float | torch.Tensor = 0.0
        astigmatism_um: float | torch.Tensor = 0.0
    else:
        defocus_um = (dfu + dfv) / 2 / 1e4
        astigmatism_um = (dfu - dfv) / 2 / 1e4

    odd_zernike: dict[str, float | torch.Tensor] = {}
    if _nonzero(trefoil1) or _nonzero(trefoil2):
        rho_max = zernike_rho_max(image_shape, pixel_size)
        odd_zernike["Z33c"] = -torch.as_tensor(trefoil1) * rho_max**3
        odd_zernike["Z33s"] = -torch.as_tensor(trefoil2) * rho_max**3
    if _nonzero(tiltx) or _nonzero(tilty):
        rho_max = zernike_rho_max(image_shape, pixel_size)
        wavelength = energy_to_wavelength(voltage)
        prefactor = (
            2 * math.pi * wavelength**2 * torch.as_tensor(cs_angstrom) * rho_max**3
        )
        odd_zernike["Z31c"] = -prefactor * torch.as_tensor(tiltx)
        odd_zernike["Z31s"] = -prefactor * torch.as_tensor(tilty)

    dose = ctf_params.get("dose")

    return CTFParameters(
        defocus=defocus_um,
        astigmatism=astigmatism_um,
        astigmatism_angle=dfang,
        voltage=voltage,
        spherical_aberration=torch.as_tensor(cs_angstrom) / 1e7,
        phase_shift=torch.rad2deg(torch.as_tensor(phaseshift_rad)),
        odd_zernike=odd_zernike or None,
        dose=dose,
    )


def _nonzero(value: Any) -> bool:
    """True if `value` is anything but exactly zero (scalar or tensor)."""
    return bool(torch.any(torch.as_tensor(value) != 0))


class LegacyAberrationAdapter(nn.Module):
    """Drop-in replacement for ``aberrations.Aberration``, backed by
    :class:`TransferFunction`.

    Same call signature as ``Aberration`` (``forward(exitwave,
    ctf_params_dict)`` / ``__call__``), so any existing call site
    (``self.aberration(exitwave, ctf_batch)``) needs no changes -- only
    which class ``self.aberration`` is constructed from needs to change.
    Converts the legacy dict to :class:`CTFParameters` fresh on every
    call via :func:`ctf_params_dict_to_parameters`.
    """

    def __init__(
        self,
        n_pixels: int,
        pixel_size: float,
        voltage: float,
        aberration_model: str = "holography",
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
    ) -> None:
        super().__init__()
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.image_shape = (n_pixels, n_pixels)
        self.transfer_function_module = TransferFunction(
            n_pixels,
            pixel_size,
            aberration_model=aberration_model,
            bfactor=bfactor,
            convergence_angle=convergence_angle,
            cc=cc,
            energy_spread=energy_spread,
            deltaV_V=deltaV_V,
            deltaI_I=deltaI_I,
            dose_envelope=dose_envelope,
        )

    def forward(
        self, exitwave: torch.Tensor, ctf_params: dict[str, Any]
    ) -> torch.Tensor:
        params = ctf_params_dict_to_parameters(
            ctf_params, self.pixel_size, self.image_shape, self.voltage
        )
        return self.transfer_function_module(exitwave, params)
