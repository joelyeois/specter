from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from .fft_tools import fft2, ifft2
from .scattering import energy_to_wavelength


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

        if aberration_model == "ctf":
            if alpha is None:
                raise Exception("Specify alpha for CTF model.")
            else:
                self.alpha = alpha

    def _cs(self, cs: torch.Tensor) -> torch.Tensor:
        """
        Calculate spherical aberration phase contribution.

        Parameters
        ----------
        cs : torch.Tensor
            Spherical aberration coefficient.

        Returns
        -------
        chi_cs : torch.Tensor
            Phase contribution from spherical aberration.
        """
        return torch.pi / 2 * self.wavelength**3 * self.k**4 * cs

    def _defocus(
        self, dfu: torch.Tensor, dfv: torch.Tensor, dfang: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate defocus phase contribution including astigmatism.

        Parameters
        ----------
        dfu : torch.Tensor
            Defocus along first axis.
        dfv : torch.Tensor
            Defocus along second axis.
        dfang : torch.Tensor
            Astigmatism angle.

        Returns
        -------
        chi_defocus : torch.Tensor
            Phase contribution from defocus and astigmatism.
        """
        dfang = dfang / 180 * torch.pi
        df = 0.5 * (dfu + dfv + (dfv - dfu) * torch.cos(2 * (self.radian + dfang)))
        return -torch.pi * self.wavelength * self.k2 * df

    def _beamtilt(
        self, cs: torch.Tensor, tiltx: torch.Tensor, tilty: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate beam tilt phase contribution.

        Parameters
        ----------
        cs : torch.Tensor
            Spherical aberration coefficient.
        tiltx : torch.Tensor
            Beam tilt in x direction.
        tilty : torch.Tensor
            Beam tilt in y direction.

        Returns
        -------
        chi_tilt : torch.Tensor
            Phase contribution from beam tilt.
        """
        tilts = torch.sin(tilty) * self.kxx + torch.sin(tiltx) * self.kyy
        return -2 * torch.pi * self.wavelength**2 * cs * self.k2 * tilts

    def _trefoil(self, trefoil1: torch.Tensor, trefoil2: torch.Tensor) -> torch.Tensor:
        """
        Calculate trefoil (3-fold astigmatism) phase contribution.

        Parameters
        ----------
        trefoil1 : torch.Tensor
            First trefoil component.
        trefoil2 : torch.Tensor
            Second trefoil component.

        Returns
        -------
        chi_trefoil : torch.Tensor
            Phase contribution from trefoil aberration.
        """
        return trefoil1 * self.k**3 * torch.sin(
            3 * self.radian
        ) + trefoil2 * self.k**3 * torch.cos(3 * self.radian)

    def _tetrafoil(self, tetrafoil1, tetrafoil2, tetrafoil3, tetrafoil4):
        """
        Calculate tetrafoil (4-fold astigmatism) phase contribution.

        Parameters
        ----------
        tetrafoil1 : torch.Tensor
            First tetrafoil component.
        tetrafoil2 : torch.Tensor
            Second tetrafoil component.
        tetrafoil3 : torch.Tensor
            Third tetrafoil component.
        tetrafoil4 : torch.Tensor
            Fourth tetrafoil component.

        Returns
        -------
        chi_tetrafoil : torch.Tensor or None
            Phase contribution from tetrafoil aberration. Currently not implemented.
        """
        pass

    def _phaseshift(self, phaseshift: torch.Tensor) -> torch.Tensor:
        """
        Calculate phase shift contribution (e.g., from Volta phase plate).

        Parameters
        ----------
        phaseshift : torch.Tensor
            Phase shift value in radians.

        Returns
        -------
        chi_phaseshift : torch.Tensor
            Phase shift contribution. For holography model, DC component is
            set to zero to maintain Fourier optics validity.
        """
        if self.aberration_model == "holography":
            phaseshift = phaseshift * torch.ones_like(self.k)
            # phaseshift must be zero at DC for Fourier optics
            phaseshift[:, self.n_pixels // 2, self.n_pixels // 2] = 0
        return -phaseshift

    def aberration(
        self,
        cs: torch.Tensor,
        dfu: torch.Tensor,
        dfv: torch.Tensor,
        dfang: torch.Tensor,
        tiltx: torch.Tensor,
        tilty: torch.Tensor,
        phaseshift: torch.Tensor,
        tref1: torch.Tensor,
        tref2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate combined aberration phase terms.

        Computes gamma and phi for the transfer function.

        Parameters
        ----------
        cs : torch.Tensor
            Spherical aberration coefficient.
        dfu : torch.Tensor
            Defocus along first axis.
        dfv : torch.Tensor
            Defocus along second axis.
        dfang : torch.Tensor
            Astigmatism angle.
        tiltx : torch.Tensor
            Beam tilt in x direction.
        tilty : torch.Tensor
            Beam tilt in y direction.
        phaseshift : torch.Tensor
            Phase shift (e.g., from phase plate).
        tref1 : torch.Tensor
            First trefoil component.
        tref2 : torch.Tensor
            Second trefoil component.

        Returns
        -------
        gamma : torch.Tensor
            Axial aberration phase (defocus + Cs - phaseshift).
        phi : torch.Tensor
            Non-axial aberration phase (beam tilt + trefoil).
        """
        w = self.wavelength
        ang = self.radian
        k2 = self.k2
        k = self.k

        # defocus
        dfu = dfu.unsqueeze(1).unsqueeze(2)
        dfv = dfv.unsqueeze(1).unsqueeze(2)
        dfang = dfang.unsqueeze(1).unsqueeze(2)
        cs = cs.unsqueeze(1).unsqueeze(2)
        df = 0.5 * (dfu + dfv + (dfv - dfu) * torch.cos(2 * (ang + dfang)))
        gamma = torch.pi * w * k2 * (0.5 * cs * w**2 * k2 - df)

        # beamtilt
        tiltx = tiltx.unsqueeze(1).unsqueeze(2)
        tilty = tilty.unsqueeze(1).unsqueeze(2)
        tilt = (
            -2
            * torch.pi
            * w**2
            * cs
            * k2
            * (torch.sin(tilty) * self.kxx + torch.sin(tiltx) * self.kyy)
        )

        # trefoil
        tref1 = tref1.unsqueeze(1).unsqueeze(2)
        tref2 = tref2.unsqueeze(1).unsqueeze(2)
        trefoil = tref1 * k**3 * torch.sin(3 * ang) + tref2 * k**3 * torch.cos(3 * ang)

        phi = tilt + trefoil

        # phase shift
        phaseshift = phaseshift.unsqueeze(1).unsqueeze(2)
        if self.aberration_model == "holography":
            # phaseshift must be zero at DC for Fourier optics
            phaseshift = phaseshift * torch.ones_like(gamma)
            phaseshift[:, 0, 0] = 0
        return gamma - phaseshift, phi

    # def temporal_envelope(self, cc=0, I=1, dE=1, dI=0, dV=0):
    #     """
    #     Eq. 2.25 of Erni, R. (2010). Aberration-corrected imaging in transmission electron microscopy: An introduction
    #     Eq. 3.41 of Kirkland
    #     """
    #     if cc == 0:
    #         return 1
    #     else:
    #         # E [eV], I [A], and V [V] are the electron energy, lens currents, and acceleration voltage
    #         dC1 = cc * torch.sqrt(
    #             (dE / self.energy) ** 2 + 4 * (dI / I) ** 2 + (dV / self.energy) ** 2
    #         )
    #         Et = torch.exp(
    #             -0.5 * torch.pi**2 * self.wavelength**2 * dC1**2 * self.k2**2
    #         )
    #         return Et

    def transfer(
        self,
        cs: torch.Tensor,
        dfu: torch.Tensor,
        dfv: torch.Tensor,
        dfang: torch.Tensor,
        tiltx: torch.Tensor,
        tilty: torch.Tensor,
        phaseshift: torch.Tensor,
        tref1: torch.Tensor,
        tref2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute transfer function with all aberration parameters.

        Legacy method combining aberration calculation with transfer function.

        Parameters
        ----------
        cs : torch.Tensor
            Spherical aberration coefficient.
        dfu : torch.Tensor
            Defocus along first axis.
        dfv : torch.Tensor
            Defocus along second axis.
        dfang : torch.Tensor
            Astigmatism angle.
        tiltx : torch.Tensor
            Beam tilt in x direction.
        tilty : torch.Tensor
            Beam tilt in y direction.
        phaseshift : torch.Tensor
            Phase shift.
        tref1 : torch.Tensor
            First trefoil component.
        tref2 : torch.Tensor
            Second trefoil component.

        Returns
        -------
        transfer : torch.Tensor
            Complex-valued transfer function.

        Notes
        -----
        This method is deprecated in favor of transfer_function(ctf_params).
        """
        gamma, phi = self.aberration(
            cs, dfu, dfv, dfang, tiltx, tilty, phaseshift, tref1, tref2
        )
        if self.aberration_model == "ctf":
            trans = np.sqrt(1 - self.alpha**2) * torch.sin(
                gamma
            ) - self.alpha * torch.cos(gamma)
        elif self.aberration_model == "holography":
            trans = torch.exp(-1j * gamma)
        return trans * torch.exp(-1j * phi)

    # def forward(self, exitwave, cs, dfu, dfv, dfang, tiltx, tilty, phaseshift, tref1, tref2):
    #     f = self.transfer(cs, dfu, dfv, dfang, tiltx, tilty, phaseshift, tref1, tref2)
    #     aberrated_exitwaves = ifft2(fft2(exitwave) * f)
    #     if self.aberration_model == "ctf":
    #         return torch.real(aberrated_exitwaves)
    #     elif self.aberration_model == "holography":
    #         return aberrated_exitwaves

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
        chi = 0

        # --- Defocus ---
        if any(k in ctf_params for k in ["dfu", "dfv", "dfang"]):
            dfu = ctf_params.get("dfu", self.zero).view(-1, 1, 1)

            # if dfv is not provided, use dfu
            dfv = ctf_params.get("dfv", dfu).view(-1, 1, 1)

            dfang = ctf_params.get("dfang", self.zero).view(-1, 1, 1)
            chi += self._defocus(dfu, dfv, dfang)

        # --- Cs ---
        if "cs" in ctf_params:
            cs = ctf_params.get("cs", self.zero).view(-1, 1, 1)
            chi += self._cs(cs)

        # --- Phaseshift ---
        if "phaseshift" in ctf_params:
            phaseshift = ctf_params.get("phaseshift", self.zero).view(-1, 1, 1)
            chi += self._phaseshift(phaseshift)

        # --- Beam tilt ---
        if any(k in ctf_params for k in ["tiltx", "tilty"]):
            cs = ctf_params.get("cs", self.zero).view(-1, 1, 1)
            tiltx = ctf_params.get("tiltx", self.zero).view(-1, 1, 1)
            tilty = ctf_params.get("tilty", self.zero).view(-1, 1, 1)
            chi += self._beamtilt(cs, tiltx, tilty)

        # --- Trefoil ---
        if any(k in ctf_params for k in ["trefoil1", "trefoil2"]):
            trefoil1 = ctf_params.get("trefoil1", self.zero).view(-1, 1, 1)
            trefoil2 = ctf_params.get("trefoil2", self.zero).view(-1, 1, 1)
            chi += self._trefoil(trefoil1, trefoil2)

        # --- Tetrafoil ---
        if any(
            k in ctf_params
            for k in ["tetrafoil1", "tetrafoil2", "tetrafoil3", "tetrafoil4"]
        ):
            tetrafoil1 = ctf_params.get("tetrafoil1", self.zero).view(-1, 1, 1)
            tetrafoil2 = ctf_params.get("tetrafoil2", self.zero).view(-1, 1, 1)
            tetrafoil3 = ctf_params.get("tetrafoil3", self.zero).view(-1, 1, 1)
            tetrafoil4 = ctf_params.get("tetrafoil4", self.zero).view(-1, 1, 1)
            chi += self._tetrafoil(tetrafoil1, tetrafoil2, tetrafoil3, tetrafoil4)

        return torch.exp(-1j * chi)

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
        f = self.transfer_function(ctf_params)
        aberrated_exitwaves = ifft2(fft2(exitwave) * f)
        if self.aberration_model == "ctf":
            return torch.real(aberrated_exitwaves)
        elif self.aberration_model == "holography":
            return aberrated_exitwaves


class Detector(L.LightningModule):
    """
    A detector module to apply detector noise to images. Future work to include
    magnification and DQE functionality.

    Parameters
    ----------
    pixel_size : float
        Pixel size in Å.
    dose_per_angstrom : float
        Dose of the electron beam in e-/Å².
    aberration_model : str, optional
        Specifies aberration model to use. Options include 'holography' and 'ctf'.
        Default is 'holography'.
    noise_model : str, optional
        Specifies noise model. Currently only 'poisson' available. Default is None
        (no noise applied).
    magnification : float, optional
        Magnification factor. Default is None. (To-do: not yet implemented).
    dqe : bool, optional
        Detective quantum efficiency. Default is None. (To-do: not yet implemented).
    mtf : torch.Tensor, optional
        Modulation transfer function in Fourier space to apply to images.
        Default is None (no MTF applied).
    coincidence_radius : float, optional
        Coincidence radius for detector. Default 0.0.
    dose_per_angstrom_per_frame : float, optional
        Dose per frame for dose-fractionated noise. Default 1.0.
    """

    def __init__(
        self,
        pixel_size: float,
        dose_per_angstrom: float,
        aberration_model: str = "holography",
        noise_model: str | None = None,
        magnification: float | None = None,
        dqe: bool | None = None,
        mtf: torch.Tensor | None = None,
        coincidence_radius: float = 0.0,
        dose_per_angstrom_per_frame: float = 1.0,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.register_buffer("mtf", mtf)
        self.coincidence_radius = coincidence_radius
        self.dose_per_angstrom_per_frame = dose_per_angstrom_per_frame

    def image(self, aberrated_exitwave: torch.Tensor) -> torch.Tensor:
        """
        Convert aberrated exit wave to detector image.

        Parameters
        ----------
        aberrated_exitwave : torch.Tensor
            Aberrated exit wave from microscope aberration module, shape (B, Y, X).

        Returns
        -------
        images : torch.Tensor
            Detector image. For holography model, returns intensity (squared
            magnitude). For CTF model, returns dose-scaled image.
        """
        if self.aberration_model == "holography":
            images = torch.abs(aberrated_exitwave) ** 2
        elif self.aberration_model == "ctf":
            images = self.dose_per_pixel * (aberrated_exitwave + 1)
        return images

    def anisomagnify(
        self, images: torch.Tensor, anisomag: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply anisotropic magnification to images.

        Parameters
        ----------
        images : torch.Tensor
            Input images with shape (B, H, W) where B is batch size.
        anisomag : torch.Tensor
            2x2 anisotropic magnification matrix for each image in batch,
            shape (B, 2, 2).

        Returns
        -------
        magnified_images : torch.Tensor
            Images after anisotropic magnification, shape (B, H, W).
        """
        images = images.unsqueeze(1)
        B = len(images)

        # Identity matrix (Bx3x3)
        M_affine = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        M_affine = M_affine.to(images.device)
        M_affine[:, :2, :2] = anisomag

        # Convert to (B, 2, 3) format by repeating for all batch elements
        M_affine = M_affine[:, :2, :]  # Shape: (B, 2, 3)

        # Generate affine grid for the batch
        grid = F.affine_grid(M_affine, images.shape, align_corners=False)

        # Apply transformation using grid sampling
        images = F.grid_sample(images, grid, align_corners=False, padding_mode="border")
        images = torch.squeeze(images)
        return images

    def add_mtf(self, images: torch.Tensor, mtf: torch.Tensor) -> torch.Tensor:
        """
        Apply modulation transfer function (MTF) to images.

        Parameters
        ----------
        images : torch.Tensor
            Input images.
        mtf : torch.Tensor
            Modulation transfer function in Fourier space.

        Returns
        -------
        filtered_images : torch.Tensor
            Real-valued images after MTF application.
        """
        return torch.real(ifft2(fft2(images) * mtf))

    def forward(
        self,
        aberrated_exitwave: torch.Tensor,
        anisomag: torch.Tensor | None = None,
        nxy: int | None = None,
    ) -> torch.Tensor:
        """
        Simulate detection process: conversion to intensity, magnification, MTF, and noise.

        Parameters
        ----------
        aberrated_exitwave : torch.Tensor
            Aberrated exit wave from microscope aberration module.
        anisomag : torch.Tensor, optional
            Anisotropic magnification matrices.
        nxy : int, optional
            Output image size in pixels. If provided, center-crops the image.

        Returns
        -------
        images : torch.Tensor
            Simulated images after detection.
        """
        images = self.image(aberrated_exitwave)

        # Set default crop size
        if nxy is None:
            nxy = images.shape[2]

        # Crop if needed
        if nxy != images.shape[2]:
            H, W = images.shape[1], images.shape[2]
            cy, cx = H // 2, W // 2

            half = nxy // 2
            images = images[
                :,
                cy - half : cy + half + (nxy % 2),
                cx - half : cx + half + (nxy % 2),
            ]

        # Apply anisomagnification
        if anisomag is not None:
            images = self.anisomagnify(images, anisomag)

        # Apply detector MTF
        if self.mtf is not None:
            images = self.add_mtf(images, self.mtf)

        if self.noise_model is None:
            return images
        elif self.noise_model == "poisson":
            if images.ndim == 3:
                return torch.stack([self.apply_coincidence(img) for img in images])
            else:
                return self.apply_coincidence(images)

    # def apply_coincidence_frame(self, img_frame):
    #     """Apply Poisson + coincidence to a single frame."""
    #     if self.coincidence_radius <= 0.0:
    #         return torch.poisson(torch.clamp(img_frame, min=0.0))

    #     H, W = img_frame.shape
    #     electrons_per_pixel = torch.poisson(torch.clamp(img_frame, min=0.0))
    #     ys, xs = torch.nonzero(electrons_per_pixel, as_tuple=True)
    #     counts = electrons_per_pixel[ys, xs].long()

    #     if counts.sum() == 0:
    #         return torch.zeros_like(img_frame)

    #     coords = torch.cat([
    #         xs.repeat_interleave(counts).unsqueeze(-1) + torch.rand((counts.sum(), 1), device=img_frame.device),
    #         ys.repeat_interleave(counts).unsqueeze(-1) + torch.rand((counts.sum(), 1), device=img_frame.device)
    #     ], dim=1)

    #     if coords.shape[0] > 1:
    #         r_sq = self.coincidence_radius ** 2
    #         dist_sq = torch.cdist(coords, coords, p=2).pow(2)
    #         adj = (dist_sq < r_sq) & torch.triu(torch.ones_like(dist_sq, dtype=torch.bool), diagonal=1)
    #         conflicts = adj.any(dim=0)
    #         coords = coords[~conflicts]

    #     ix = coords[:, 0].long().clamp(0, W - 1)
    #     iy = coords[:, 1].long().clamp(0, H - 1)
    #     flat_idx = ix * H + iy
    #     pixels = torch.zeros(H * W, device=img_frame.device)
    #     pixels.scatter_add_(0, flat_idx, torch.ones_like(ix, dtype=torch.float32))
    #     return pixels.view(H, W)

    # def apply_coincidence(self, img):
    #     """
    #     Apply dose-fractionated Poisson + coincidence to the total-dose image.
    #     """
    #     if self.noise_model != "poisson" or self.dose_per_angstrom_per_frame <= 0.0:
    #         # Single-frame Poisson approximation
    #         return torch.poisson(torch.clamp(img, min=0.0))

    #     # Estimate total dose from image
    #     # Assuming the image is in e-/Å² units per pixel
    #     total_dose = img.sum() / (img.shape[0] * img.shape[1])  # e-/Å² average per pixel

    #     n_frames = max(1, int(torch.round(total_dose / self.dose_per_angstrom_per_frame)))

    #     img_per_frame = img / n_frames
    #     final_image = torch.zeros_like(img)
    #     for _ in range(n_frames):
    #         final_image += self.apply_coincidence_frame(img_per_frame)

    #     return final_image

    def apply_coincidence_frame(self, img_frame: torch.Tensor) -> torch.Tensor:
        """
        Apply Poisson + coincidence loss to a single frame using pixel-based convolution.

        Parameters
        ----------
        img_frame : torch.Tensor
            Expected electrons per pixel, shape (H, W).

        Returns
        -------
        electrons_per_pixel : torch.Tensor
            Simulated electron counts after coincidence loss.
        """
        # 1. Poisson electrons per pixel
        electrons_per_pixel = torch.poisson(torch.clamp(img_frame, min=0.0))

        # 2. Determine kernel size from coincidence radius
        # Kernel covers floor(radius) pixels in each direction
        if self.coincidence_radius > 0.0:
            r = int(torch.floor(torch.tensor(self.coincidence_radius)))
            kernel_size = 2 * r + 1
            kernel = torch.ones(
                (1, 1, kernel_size, kernel_size), device=img_frame.device
            )

            # Add batch & channel dims for conv2d
            neighbors = F.conv2d(
                electrons_per_pixel.unsqueeze(0).unsqueeze(0), kernel, padding=r
            )

            # Suppress electrons where neighborhood count > 1
            mask = neighbors > 1
            electrons_per_pixel[mask.squeeze(0).squeeze(0)] = 1

        return electrons_per_pixel

    def apply_coincidence(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply dose-fractionated Poisson + coincidence.

        Parameters
        ----------
        img : torch.Tensor
            Total-dose image, shape (H, W).

        Returns
        -------
        final_image : torch.Tensor
            Simulated image after dose-fractionated noise and coincidence.
        """
        if self.noise_model != "poisson" or self.dose_per_angstrom_per_frame <= 0.0:
            return torch.poisson(torch.clamp(img, min=0.0))

        # 1. Compute number of frames based on dose per frame
        avg_total_dose = img.sum() / (img.shape[0] * img.shape[1])  # e-/Å² per pixel
        n_frames = max(
            1, int(torch.round(avg_total_dose / self.dose_per_angstrom_per_frame))
        )

        # 2. Split total dose per frame
        img_per_frame = img / n_frames

        # 3. Sum dose-fractionated frames after coincidence
        final_image = torch.zeros_like(img)
        for _ in range(n_frames):
            final_image += self.apply_coincidence_frame(img_per_frame)

        return final_image
