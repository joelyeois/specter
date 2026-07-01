from __future__ import annotations

from typing import Any

import torch
import lightning as L

from . import _envelopes as env
from . import _functions as fn
from ..fft import fft2, ifft2
from ..scattering import energy_to_wavelength


class Aberration(L.LightningModule):
    """
    An aberration module to apply microscopy aberrations to the 2D exitwaves.

    Parameters
    ----------
    n_pixels : int
        Number of pixels in exitwave, (n_pixels, n_pixels).
    pixel_size: float
        Pixel size in Å.
    energy: float
        Energy of the electron beam in keV. Typical values are 100/120/200/300 keV.
    aberration_model: str, optional
        Specifies aberration model to use. Options include 'holography' and 'ctf'.
        Default is 'holography'.
    alpha: float, optional
        The amplitude contrast ratio to use for the CTF model. Common values
        are 0.07 and 0.1.
    convergence_angle: float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. Default is None (envelope disabled).
    cc: float, optional
        Chromatic aberration coefficient in Angstrom, used for the Cc
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

    Notes
    -----
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
       US, Boston, MA, 2010).
    .. [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
       Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35–72.
    """

    def __init__(
        self,
        n_pixels: int,
        pixel_size: float,
        energy: float,
        aberration_model: str = "holography",
        alpha: float | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        progressbars: bool = True,
    ):
        super().__init__()
        self.n_pixels = n_pixels
        self.pixel_size = pixel_size

        # model params
        self.energy = energy
        self.wavelength = energy_to_wavelength(energy)
        self.aberration_model = aberration_model

        # frequency coordinates
        kx = torch.fft.fftfreq(n_pixels, pixel_size)
        kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
        k2 = kxx**2 + kyy**2
        radian = torch.arctan2(kyy, kxx)
        self.register_buffer("k", torch.sqrt(k2).unsqueeze(0))
        self.register_buffer("radian", radian.unsqueeze(0))
        self.register_buffer("k2", k2.unsqueeze(0))
        self.register_buffer("kxx", kxx)
        self.register_buffer("kyy", kyy)

        # dummy tensor for non-existent aberration terms
        self.register_buffer("zero", torch.tensor(0.0))

        self.convergence_angle = convergence_angle
        self.cc = cc
        self.energy_spread = energy_spread
        self.deltaV_V = deltaV_V
        self.deltaI_I = deltaI_I
        self.dose_envelope = dose_envelope

        if aberration_model == "ctf":
            if alpha is None:
                raise Exception("Specify alpha for CTF model.")
            else:
                self.alpha = alpha

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
            - 'dose' : Cumulative electron dose in e-/Angstrom^2, used by
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

        # --- Phaseshift ---
        if "phaseshift" in ctf_params:
            phaseshift = ctf_params.get("phaseshift", self.zero).view(-1, 1, 1)
            chi = chi + fn.phaseshift(
                phaseshift, self.k, self.n_pixels, self.aberration_model
            )

        # --- Beam tilt ---
        if any(k in ctf_params for k in ["tiltx", "tilty"]):
            cs = ctf_params.get("cs", self.zero).view(-1, 1, 1)
            tiltx = ctf_params.get("tiltx", self.zero).view(-1, 1, 1)
            tilty = ctf_params.get("tilty", self.zero).view(-1, 1, 1)
            chi = chi + fn.beamtilt(
                self.k2, self.kxx, self.kyy, self.wavelength, cs, tiltx, tilty
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
            chi = chi + fn.tetrafoil(tetrafoil1, tetrafoil2, tetrafoil3, tetrafoil4)

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
            voltage = self.energy * 1e3  # keV -> V (numerically eV-equivalent)
            transfer = transfer * env.cc_envelope(
                self.k2,
                self.wavelength,
                self.cc,
                voltage,
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
            Exit wave after aberration. Complex-valued for holography model,
            real-valued for CTF model (projects to real for intensity imaging).

        Notes
        -----
        Applies transfer function in Fourier space:
        ψ_aberrated = FFT⁻¹[FFT[ψ] * T(CTF)]
        """
        self.tf = self.transfer_function(ctf_params)
        aberrated_exitwaves = ifft2(fft2(exitwave) * self.tf)
        if self.aberration_model == "ctf":
            return torch.real(aberrated_exitwaves)
        elif self.aberration_model == "holography":
            return aberrated_exitwaves
        else:
            raise ValueError(f"Unknown aberration_model: {self.aberration_model!r}")
