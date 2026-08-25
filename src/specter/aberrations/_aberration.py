from __future__ import annotations

from typing import Any

import torch
import lightning as L

from . import _envelopes as env
from . import _functions as fn
from ..constants import energy_to_wavelength
from ..fft import fft2, ifft2

# Keys read out of the per-batch `ctf_params` dict passed to `forward()` --
# see `Aberration.transfer_function`. Never constructor arguments -- `bfactor`
# is the one exception, accepted as a constructor kwarg (see `__init__`) for
# consistency with `ImageGenerator`/`BaseImager`/`Reconstructor`, which all
# offer the same convenience with the same override-wins precedence.
_CTF_PARAM_NAMES = frozenset(
    {
        "dfu",
        "dfv",
        "dfang",
        "cs",
        "phaseshift",
        "tiltx",
        "tilty",
        "trefoil1",
        "trefoil2",
        "tetrafoil1",
        "tetrafoil2",
        "tetrafoil3",
        "tetrafoil4",
        "dose",
    }
)


class Aberration(L.LightningModule):
    """
    An aberration module to apply microscopy aberrations to the 2D exitwaves,
    following the conventions of Kirkland [1]_ and Penczek [2]_.

    Parameters
    ----------
    n_pixels : int
        Number of pixels in exitwave, (n_pixels, n_pixels).
    pixel_size: float
        Pixel size in Å.
    voltage: float
        Accelerating voltage of the electron beam in kV. Typical values are 100/120/200/300 kV.
    aberration_model: str, optional
        Specifies aberration model to use. Options include 'nonlinear' and 'linear'.
        Default is 'nonlinear'.
    alpha: float, optional
        The amplitude contrast ratio to use for the linear model. Common values
        are 0.07 and 0.1. Only has an effect on the transfer function when
        ``specimen_absorption=False`` (see below) -- required whenever
        ``aberration_model="linear"`` regardless, since :meth:`__init__`
        raises if it's missing, but with the default ``specimen_absorption``
        it's stored and otherwise unused.
    specimen_absorption: bool, optional
        If True (default), amplitude contrast is assumed to already be
        baked into the exit wave via ``scattering.complex_potential``
        upstream (true for every ``scattering_model`` except ``"ctf"``,
        which yields a real-valued, non-absorptive exit wave with nowhere
        else to represent it) -- so ``alpha`` is accepted but not applied
        again here, avoiding double-counting the same physical effect at
        both the specimen and lens stages. Set False to instead fold
        ``-acos(alpha)`` into the phase-shift term (CryoSPARC's own
        convention: ``phase_shift - arccos(amp_contrast)``), which is what
        an exit wave from ``scattering_model="ctf"`` needs. Mirrors
        ``ctf.TransferFunction``'s identically-named, identically-purposed
        parameter.
    convergence_angle: float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. Default is None (envelope disabled).
    cc: float, optional
        Chromatic aberration coefficient in Å, used for the Cc
        (temporal coherence) envelope. Default is None (envelope
        disabled).
    energy_spread: float, optional
        FWHM of the beam energy spread in eV, used by the Cc envelope.
        Default 0.7.
    deltaV_V: float, optional
        Relative high-voltage instability, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I: float, optional
        Relative objective-lens current instability, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope: bool, optional
        Whether to apply the Grant & Grigorieff (2015) cumulative-dose
        envelope, using the per-image ``dose`` key in ``ctf_params``.
        Default False.
    bfactor: float or torch.Tensor, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function, damping high-resolution signal. Scalar (applied to all
        particles) or per-particle tensor. If given, overrides any
        ``"bfactor"`` entry already present in the ``ctf_params`` passed to
        :meth:`forward` -- the same convenience and precedence offered by
        ``ImageGenerator``/``BaseImager``/``Reconstructor``. None or 0.0
        means no envelope. Default None.

    Notes
    -----
    Per-image CTF terms other than ``bfactor`` (``dfu``, ``dfv``, ``dfang``,
    ``cs``, ``phaseshift``, ``tiltx``, ``tilty``, ``trefoil1/2``,
    ``tetrafoil1-4``, ``dose``) are never constructor arguments -- they're
    passed to :meth:`forward` via the ``ctf_params`` dict, since they
    genuinely vary per particle. ``bfactor`` is the one exception: it's
    usually a single calibration value applied uniformly across a batch, so
    it gets the same constructor-level convenience as ``dose_per_angstrom``
    one level up, rather than requiring a hand-built per-particle tensor.

    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
       US, Boston, MA, 2010).
    .. [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
       Methods in Enzymology (Academic Press Inc., 2010) volume. 482, pp. 35–72.
    """

    def __init__(
        self,
        n_pixels: int,
        pixel_size: float,
        voltage: float,
        aberration_model: str = "nonlinear",
        alpha: float | None = None,
        specimen_absorption: bool = True,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        bfactor: float | torch.Tensor | None = None,
        progressbars: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            bad = _CTF_PARAM_NAMES & kwargs.keys()
            if bad:
                raise TypeError(
                    f"Aberration() does not accept {sorted(bad)} as constructor "
                    "arguments -- these are per-image CTF parameters, passed to "
                    "forward() via the `ctf_params` dict instead, e.g. "
                    "Aberration(...)(exitwave, {'dfu': ...})."
                )
            raise TypeError(
                f"Aberration() got unexpected keyword arguments: {sorted(kwargs)}"
            )
        super().__init__()
        self.n_pixels = n_pixels
        self.pixel_size = pixel_size

        # model params
        self.voltage = voltage
        self.wavelength = energy_to_wavelength(voltage)
        self.aberration_model = aberration_model

        # frequency coordinates -- KY varies along axis 0 (rows), KX along
        # axis 1 (columns), matching the row/col = y/x convention used
        # elsewhere in specter (see arrays.grid_2d/kgrid_2d).
        kx = torch.fft.fftfreq(n_pixels, pixel_size)
        KY, KX = torch.meshgrid(kx, kx, indexing="ij")
        k2 = KX**2 + KY**2
        radian = torch.arctan2(KX, KY)
        self.register_buffer("k", torch.sqrt(k2).unsqueeze(0))
        self.register_buffer("radian", radian.unsqueeze(0))
        self.register_buffer("k2", k2.unsqueeze(0))
        self.register_buffer("KY", KY)
        self.register_buffer("KX", KX)

        # dummy tensor for non-existent aberration terms
        self.register_buffer("zero", torch.tensor(0.0))

        self.convergence_angle = convergence_angle
        self.cc = cc
        self.energy_spread = energy_spread
        self.deltaV_V = deltaV_V
        self.deltaI_I = deltaI_I
        self.dose_envelope = dose_envelope

        if bfactor is None:
            self.bfactor: torch.Tensor | None = None
        else:
            self.register_buffer(
                "bfactor", torch.as_tensor(bfactor, dtype=torch.float32).flatten()
            )

        if aberration_model == "linear":
            if alpha is None:
                raise Exception("Specify alpha for the linear model.")
            else:
                self.alpha = alpha
        self.specimen_absorption = specimen_absorption

    def transfer_function(self, ctf_params: dict[str, Any]) -> torch.Tensor:
        """
        Compute transfer function from CTF parameters dictionary.

        Modular approach allowing flexible specification of aberration terms.
        Only computes contributions for parameters present in the dictionary.

        Parameters
        ----------
        ctf_params : dict
            Dictionary of CTF parameters. Supported keys:

            - 'dfu'/'dfv'/'dfang' : Defocus and astigmatism
            - 'cs' : Spherical aberration
            - 'phaseshift' : Phase shift (e.g., phase plate)
            - 'tiltx'/'tilty' : Beam tilt
            - 'trefoil1'/'trefoil2' : Trefoil aberration
            - 'tetrafoil1'/'tetrafoil2'/'tetrafoil3'/'tetrafoil4' : Tetrafoil
            - 'bfactor' : Isotropic B-factor envelope in Å²
            - 'dose' : Cumulative electron dose (fluence) in e⁻/Å², used by
              the dose envelope when ``dose_envelope=True``

        Returns
        -------
        transfer : torch.Tensor
            Complex-valued transfer function exp(-iχ) where χ is the
            total aberration phase.

        Notes
        -----
        Missing parameters default to zero (no contribution to aberration).
        """
        # total aberration phase
        chi = torch.zeros_like(self.k)

        # --- Defocus ---
        if any(k in ctf_params for k in ["dfu", "dfv", "dfang"]):
            dfu = ctf_params.get("dfu", self.zero).view(-1, 1, 1)

            # if dfv is not provided, use dfu
            dfv = ctf_params.get("dfv", dfu).view(-1, 1, 1)

            dfang = ctf_params.get("dfang", self.zero).view(-1, 1, 1)
            chi = chi + fn.defocus(
                self.k2, self.radian, self.wavelength, dfu, dfv, dfang
            )

        # --- Cs ---
        if "cs" in ctf_params:
            cs = ctf_params.get("cs", self.zero).view(-1, 1, 1)
            chi = chi + fn.cs(self.k, self.wavelength, cs)

        # --- Phaseshift (+ amplitude-contrast offset for the CTF model) ---
        needs_amp_contrast_offset = (
            self.aberration_model == "linear" and not self.specimen_absorption
        )
        if "phaseshift" in ctf_params or needs_amp_contrast_offset:
            phaseshift = ctf_params.get("phaseshift", self.zero).view(-1, 1, 1)
            alpha = torch.as_tensor(self.alpha) if needs_amp_contrast_offset else None
            chi = chi + fn.phaseshift(
                phaseshift, self.k, self.n_pixels, self.aberration_model, alpha=alpha
            )

        # --- Beam tilt ---
        if any(k in ctf_params for k in ["tiltx", "tilty"]):
            cs = ctf_params.get("cs", self.zero).view(-1, 1, 1)
            tiltx = ctf_params.get("tiltx", self.zero).view(-1, 1, 1)
            tilty = ctf_params.get("tilty", self.zero).view(-1, 1, 1)
            chi = chi + fn.beamtilt(
                self.k2, self.KY, self.KX, self.wavelength, cs, tiltx, tilty
            )

        # --- Trefoil ---
        if any(k in ctf_params for k in ["trefoil1", "trefoil2"]):
            trefoil1 = ctf_params.get("trefoil1", self.zero).view(-1, 1, 1)
            trefoil2 = ctf_params.get("trefoil2", self.zero).view(-1, 1, 1)
            chi = chi + fn.trefoil(self.k, self.radian, trefoil1, trefoil2)

        # --- Tetrafoil ---
        if any(
            k in ctf_params
            for k in ["tetrafoil1", "tetrafoil2", "tetrafoil3", "tetrafoil4"]
        ):
            tetrafoil1 = ctf_params.get("tetrafoil1", self.zero).view(-1, 1, 1)
            tetrafoil2 = ctf_params.get("tetrafoil2", self.zero).view(-1, 1, 1)
            tetrafoil3 = ctf_params.get("tetrafoil3", self.zero).view(-1, 1, 1)
            tetrafoil4 = ctf_params.get("tetrafoil4", self.zero).view(-1, 1, 1)
            chi = chi + fn.tetrafoil(
                self.k, self.radian, tetrafoil1, tetrafoil2, tetrafoil3, tetrafoil4
            )

        transfer = torch.exp(-1j * chi)

        if "bfactor" in ctf_params:
            bfactor = ctf_params["bfactor"].view(-1, 1, 1)
            if torch.any(bfactor != 0):
                transfer = transfer * env.b_envelope(self.k2, bfactor)

        if self.convergence_angle is not None:
            cs_val = ctf_params.get("cs", self.zero).view(-1, 1, 1)
            dfu = ctf_params.get("dfu", self.zero).view(-1, 1, 1)
            dfv = ctf_params.get("dfv", dfu).view(-1, 1, 1)
            defocus_avg = 0.5 * (dfu + dfv)
            transfer = transfer * env.cs_envelope(
                self.k, self.wavelength, cs_val, defocus_avg, self.convergence_angle
            )

        if self.cc is not None:
            voltage_v = self.voltage * 1e3  # kV -> V (numerically eV-equivalent)
            transfer = transfer * env.cc_envelope(
                self.k2,
                self.wavelength,
                self.cc,
                voltage_v,
                self.energy_spread,
                self.deltaV_V,
                self.deltaI_I,
            )

        if self.dose_envelope and "dose" in ctf_params:
            dose = ctf_params["dose"].view(-1, 1, 1)
            transfer = transfer * env.dose_envelope(self.k, dose)

        return transfer

    def forward(
        self, exitwave: torch.Tensor, ctf_params: dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply microscope aberrations to exit wave.

        Forward pass applying the transfer function to the input exit wave.

        Parameters
        ----------
        exitwave : torch.Tensor
            Complex-valued exit wave from scattering module, shape (B, Y, X).
        ctf_params : dict
            Dictionary of CTF parameters (see transfer_function for details).

        Returns
        -------
        aberrated_exitwave : torch.Tensor
            Exit wave after aberration. Complex-valued for the nonlinear model,
            real-valued for the linear model (projects to real for intensity imaging).

        Notes
        -----
        Applies transfer function in Fourier space:
        ψ_aberrated = FFT⁻¹[FFT[ψ] * T(CTF)]
        """
        if self.bfactor is not None:
            ctf_params = {**ctf_params, "bfactor": self.bfactor}
        self.tf = self.transfer_function(ctf_params)
        aberrated_exitwaves = ifft2(fft2(exitwave) * self.tf)
        if self.aberration_model == "linear":
            return torch.real(aberrated_exitwaves)
        elif self.aberration_model == "nonlinear":
            return aberrated_exitwaves
        else:
            raise ValueError(f"Unknown aberration_model: {self.aberration_model!r}")
