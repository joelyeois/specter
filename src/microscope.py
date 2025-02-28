import lightning as L
import numpy as np
import torch
from fft_tools import fft2, fftn, ifft2, ifftn
from scattering import energy_to_wavelength


class Aberration(L.LightningModule):
    def __init__(
        self,
        n_pixels,
        pixel_size,
        energy,
        aberration_model="holography",
        alpha=None,
    ):
        """
        An aberration module to apply microscopy aberrations to the 2D exitwaves.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in exitwave, (n_pixels, n_pixels).
        pixel_size: float
            Pixel size in angstroms.
        energy: float
            Energy of the electron beam in keV. Typical values are 100/120/200/300 keV.
        aberration_model: str
            Specifies aberration model to use. Options include 'holography' and 'ctf'.
        alpha: float
            The amplitude contrast ratio to use for the CTF model. Common values
            are 0.07 and 0.1.

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
           US, Boston, MA, 2010).
           [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
           Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35–72.

        """
        super().__init__()
        self.n_pixels = n_pixels
        self.pixel_size = pixel_size

        # model params
        self.energy = energy
        self.wavelength = energy_to_wavelength(energy)
        self.aberration_model = aberration_model

        # frequency coordinates
        kx = torch.fft.fftshift(torch.fft.fftfreq(n_pixels, pixel_size))
        kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
        k2 = kxx**2 + kyy**2
        radian = torch.arctan2(kyy, kxx)
        self.register_buffer("radian", radian)
        self.register_buffer("k2", k2)
        self.register_buffer("kxx", kxx)
        self.register_buffer("kyy", kyy)

        if aberration_model == "ctf":
            if alpha is None:
                raise Exception("Specify alpha for CTF model.")
            else:
                self.alpha = alpha

    def aberration(self, cs, dfu, dfv, dfang, tiltx, tilty):
        w = self.wavelength
        ang = self.radian.unsqueeze(0)
        k2 = self.k2.unsqueeze(0)
        dfu = dfu.unsqueeze(1).unsqueeze(2)
        dfv = dfv.unsqueeze(1).unsqueeze(2)
        dfang = dfang.unsqueeze(1).unsqueeze(2)
        cs = cs.unsqueeze(1).unsqueeze(2)
        df = 0.5 * (dfu + dfv + (dfv - dfu) * torch.cos(2 * (ang + dfang)))
        gamma = torch.pi * w * k2 * (0.5 * cs * w**2 * k2 - df)

        # beamtilt
        tiltx = tiltx.unsqueeze(1).unsqueeze(2)
        tilty = tilty.unsqueeze(1).unsqueeze(2)
        phi = (
            -2
            * torch.pi
            * w**2
            * cs
            * k2
            * (torch.sin(tilty) * self.kxx + torch.sin(tiltx) * self.kyy)
        )
        return gamma, phi

    def transfer(self, cs, dfu, dfv, dfang, tiltx, tilty):
        gamma, phi = self.aberration(cs, dfu, dfv, dfang, tiltx, tilty)
        if self.aberration_model == "ctf":
            transfer = np.sqrt(1 - self.alpha**2) * torch.sin(
                gamma
            ) - self.alpha * torch.cos(gamma)
        elif self.aberration_model == "holography":
            transfer = torch.exp(-1j * gamma)
        return transfer * torch.exp(-1j * phi)

    def forward(self, exitwave, cs, dfu, dfv, dfang, tiltx, tilty):
        f = self.transfer(cs, dfu, dfv, dfang, tiltx, tilty)
        aberrated_exitwaves = ifft2(fft2(exitwave) * f)
        if self.aberration_model == "ctf":
            return torch.real(aberrated_exitwaves)
        elif self.aberration_model == "holography":
            return aberrated_exitwaves


class Detector(L.LightningModule):
    def __init__(
        self,
        pixel_size,
        dose_per_angstrom,
        aberration_model="holography",
        noise_model=None,
        magnification=None,
        dqe=None,
    ):
        """
        A detector module to apply detector noise to images. Future work to include
        magnification and DQE functionality.

        Parameters
        ----------
        pixel_size: float
            Pixel size in angstroms.
        dose_per_angstrom: float
            Dose of the electron beam in e-/A^2.
        aberration_model: str
            Specifies aberration model to use. Options include 'holography' and 'ctf'.
        noise_model : str
            Specifies noise model. Currently only 'poisson' available.
        magnification : float
            To-do.
        dqe : bool
            To-do.

        """
        super().__init__()
        self.pixel_size = pixel_size
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.aberration_model = aberration_model
        self.noise_model = noise_model

    def image(self, aberrated_exitwave):
        if self.aberration_model == "holography":
            images = torch.abs(aberrated_exitwave) ** 2
        elif self.aberration_model == "ctf":
            images = (
                aberrated_exitwave * np.sqrt(self.dose_per_pixel) + self.dose_per_pixel
            )
        return images

    def forward(self, aberrated_exitwave):
        images = self.image(aberrated_exitwave)
        if self.noise_model is None:
            return images
        elif self.noise_model == "poisson":
            return torch.poisson(images)