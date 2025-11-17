import lightning as L
import numpy as np
import torch
from .fft_tools import fft2, ifft2
from .scattering import energy_to_wavelength
import torch.nn.functional as F


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

    def _cs(self, cs):
        return torch.pi / 2 * self.wavelength**3 * self.k**4 * cs

    def _defocus(self, dfu, dfv, dfang):
        dfu = dfu
        dfv = dfv
        dfang = dfang
        df = 0.5 * (dfu + dfv + (dfv - dfu) * torch.cos(2 * (self.radian + dfang)))
        return -torch.pi * self.wavelength * self.k2 * df

    def _beamtilt(self, cs, tiltx, tilty):
        cs = cs
        tiltx = tiltx
        tilty = tilty
        tilts = torch.sin(tilty) * self.kxx + torch.sin(tiltx) * self.kyy
        return -2 * torch.pi * self.wavelength**2 * cs * self.k2 * tilts

    def _trefoil(self, trefoil1, trefoil2):
        trefoil1 = trefoil1
        trefoil2 = trefoil2
        tf1 = trefoil1 * self.k**3 * torch.sin(3 * self.radian)
        tf2 = trefoil2 * self.k**3 * torch.cos(3 * self.radian)
        return tf1 + tf2

    def _tetrafoil(self, tetrafoil1, tetrafoil2, tetrafoil3, tetrafoil4):
        pass

    def _phaseshift(self, phaseshift):
        phaseshift = phaseshift.unsqueeze(1).unsqueeze(2)
        if self.aberration_model == "holography":
            phaseshift = phaseshift * torch.ones_like(self.k)
            # phaseshift must be zero at DC for Fourier optics
            phaseshift[:, self.n_pixels // 2, self.n_pixels // 2] = 0
        return -phaseshift

    def aberration(self, cs, dfu, dfv, dfang, tiltx, tilty, phaseshift, tref1, tref2):
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

    def transfer(self, cs, dfu, dfv, dfang, tiltx, tilty, phaseshift, tref1, tref2):
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

    def transfer_function(self, ctf_params):
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

    def forward(self, exitwave, ctf_params):
        f = self.transfer_function(ctf_params)
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
        mtf=None,
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
        self.register_buffer("mtf", mtf)

    def image(self, aberrated_exitwave):
        if self.aberration_model == "holography":
            images = torch.abs(aberrated_exitwave) ** 2
        elif self.aberration_model == "ctf":
            images = self.dose_per_pixel * (aberrated_exitwave + 1)
        return images

    def anisomagnify(self, images, anisomag):
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

    def add_mtf(self, images, mtf):
        return torch.real(ifft2(fft2(images) * mtf))

    def forward(self, aberrated_exitwave, anisomag=None, nxy=None):
        images = self.image(aberrated_exitwave)

        if nxy is not None:
            # crop away any padded areas first.
            images = images[:, nxy // 2 : -nxy // 2, nxy // 2 : -nxy // 2]

        # Apply anisomagnification
        if anisomag is not None:
            images = self.anisomagnify(images, anisomag)

        # Apply detector MTF
        if self.mtf is not None:
            images = self.add_mtf(images, self.mtf)

        if self.noise_model is None:
            return images
        elif self.noise_model == "poisson":
            return torch.poisson(torch.clamp(images, min=0.0))
