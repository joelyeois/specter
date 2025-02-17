import filters
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_tools import fft2, ifft2, fftn, ifftn


rest_mass_energy = 511.0e3  # [eV]
hc = 12.398e3  # [eV * Å]


def energy_to_wavelength(energy):
    """Converts electron energy [keV] to wavelength [Å]"""
    ev = energy * 1e3
    return hc / np.sqrt(ev * (ev + 2.0 * rest_mass_energy))


def interaction_parameter(energy):
    """Calculates the interaction parameter [1/ÅV]: Kirkland Eq.(5.6)."""
    w = energy_to_wavelength(energy)
    ev = energy * 1e3
    return (
        2.0
        * torch.pi
        / (w * ev)
        * ((ev + rest_mass_energy) / (ev + 2.0 * rest_mass_energy))
    )


def complex_potential(v, alpha=0.1):
    """Applies amplitude ratio, \alpha, to create complex potential"""
    return np.sqrt(1 - alpha**2) * v + 1j * (alpha * v)


class Scattering(L.LightningModule):
    def __init__(
        self,
        n_pixels,
        pixel_size,
        energy,
        dose_per_angstrom,
        scattering_model="multislice",
        klim=None,
        flip_curvature=False,
    ):
        """
        A scattering module to compute the 2D exitwave from a 3D scattering
        potential. Various scattering modes are available.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in volume, (n_pixels, n_pixels, n_pixels). Assumes same
            number of pixels in all three axes for now.
        pixel_size: float
            Pixel size in angstroms. Assumes dz is also pixel_size for now.
        energy: float
            Energy of the electron beam in keV. Typical values are 100/120/200/300 keV.
        dose_per_angstrom: float
            Dose of the electron beam in e-/A^2.
        scattering_mode: str
            Specifies scattering model to use. Options include 'multislice',
            'firstborn', 'projection' and 'ctf', in order of increasing approximations.
        klim: float
            Kirkland [1] explains that setting klim = 0.66 is necessary to avoid
            aliasing for FFT methods (multislice and first Born). But this numerically
            lowers the spatial frequency information in the resultant exitwaves, so
            default is set to None.
        flip_curvature: bool
            This corresponds to positive/negative Ewald sphere curvature
            ambiguity. Set to False for positive, and True for negative (CryoSPARC).
            Only affects multislice and first Born models.

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
           US, Boston, MA, 2010).

        """
        super().__init__()
        self.n_pixels = n_pixels
        self.pixel_size = pixel_size

        # model params
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.wavelength = energy_to_wavelength(energy)
        self.sigma = interaction_parameter(energy)
        self.scattering_model = scattering_model
        self.flip_curvature = flip_curvature

        # frequency coordinates
        kx = torch.fft.fftshift(torch.fft.fftfreq(n_pixels, pixel_size))
        kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
        k = torch.sqrt(kxx**2 + kyy**2)

        # Fresnel transfer function for multislice
        if scattering_model == "multislice":
            F = torch.exp(1j * torch.pi * self.wavelength * pixel_size * k**2)
            self.register_buffer("F", F)

        # Fresnel transfer function for first Born
        if scattering_model == "firstborn":
            F = []
            for i in range(n_pixels):
                f = torch.exp(1j * torch.pi * self.wavelength * pixel_size * 
                              (n_pixels - i) * k**2)
                F.append(f)
            F = torch.stack(F)
            self.register_buffer("F", F)

        # Kirkland bandlimit
        self.klim = klim
        if klim is not None:
            kmask = filters.circle2d(n_pixels, int(n_pixels * klim))[None, ...]
            self.register_buffer("kmask", kmask)
        else:
            self.kmask = 1

    def multislice(self, V):
        if self.flip_curvature:
            V = torch.flip(V, dims=(1,))
        exitwave = np.sqrt(self.dose_per_pixel)

        # iterate across z-planes of 3D potentials.
        for i in range(V.size(1)):
            # transmission function
            t = torch.exp(1j * self.sigma * V[:, i])

            # multiply with incident wave
            wv = t * exitwave

            # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
            exitwave = ifft2(fft2(wv) * self.F * self.kmask)
        return exitwave

    def firstborn(self, V):
        if self.flip_curvature:
            V = torch.flip(V, dims=(1,))

        V_f = fft2(V)
        exitwave_f = self.sigma * V_f * self.F[None, ...]
        exitwave = ifft2(exitwave_f)
        exitwave = torch.sum(exitwave, 1)  # sum along Z
        exitwave = 1 + 1j * exitwave

        # multiply with dose
        exitwave *= np.sqrt(self.dose_per_pixel)
        return exitwave

    def projection(self, V):
        exitwave = np.sqrt(self.dose_per_pixel) * torch.exp(
            1j * self.sigma * torch.sum(V, 1)
        )
        return exitwave

    def ctf(self, V):
        projection = 2 * self.sigma * torch.sum(V, 1)
        return projection

    def forward(self, V):
        """
        V is batch of 3D real-valued potentials with shape (B x Z x X x Y), outputs 
        a batch of 2D exitwaves with shape (B x Y x X). The CTF scattering model 
        outputs projected potential instead of exitwave.

        Note that the CTF model does not require computing the complex-valued
        potentials since it is built into the aberration function.

        Parameters
        ----------
        V : tensor
            Batch of 3D potentials.

        Returns
        -------
        psi : tensor
            Batch of 2D exitwaves / projected potentials.
        """
        if self.scattering_model == "multislice":
            V = complex_potential(V)
            return self.multislice(V)
        elif self.scattering_model == "projection":
            V = complex_potential(V)
            return self.projection(V)
        elif self.scattering_model == "firstborn":
            V = complex_potential(V)
            return self.firstborn(V)
        elif self.scattering_model == "ctf":
            return self.ctf(V)