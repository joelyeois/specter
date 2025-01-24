import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

fft2 = lambda array: torch.fft.fftshift(
    torch.fft.fft2(torch.fft.ifftshift(array))
)
ifft2 = lambda array: torch.fft.fftshift(
    torch.fft.ifft2(torch.fft.ifftshift(array))
)
fftn = lambda array: torch.fft.fftshift(
    torch.fft.fftn(torch.fft.ifftshift(array))
)
ifftn = lambda array: torch.fft.fftshift(
    torch.fft.ifftn(torch.fft.ifftshift(array))
)

rest_mass_energy = torch.tensor(511.0e3)  # [eV]
hc = torch.tensor(12.398e3)  # [eV * Å]

def energy_to_wavelength(energy):
    """Converts electron energy [keV] to wavelength [Å]"""
    ev = energy * 1e3
    return hc / torch.sqrt(ev * (ev + 2.0 * rest_mass_energy))


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


def complex_potential(v, alpha=torch.tensor(0.1)):
    """Applies amplitude ratio, \alpha, to create complex potential"""
    return torch.sqrt(1 - alpha**2) * v + 1j * (alpha * v)

class Scattering(L.LightningModule):
    def __init__(
        self,
        size,
        voxel_size,
        energy=200,
        dose_per_area=20,
        projectmode="multislice",
        klim=None,
        flipcurvature=False,
    ):
        super().__init__()
        nz, ny, nx = size
        self.voxel_size = voxel_size

        self.energy = energy
        self.dose_per_area = dose_per_area
        self.wavelength = energy_to_wavelength(energy)
        self.sigma = interaction_parameter(energy)
        self.projectmode = projectmode
        self.flipcurvature = flipcurvature

        # frequency coordinates
        kx = torch.fft.fftfreq(nx, voxel_size)
        ky = torch.fft.fftfreq(ny, voxel_size)
        kxx, kyy = torch.meshgrid(kx, ky, indexing="ij")
        kxx = torch.fft.fftshift(kxx)
        kyy = torch.fft.fftshift(kyy)

        kx = torch.stack([kyy, kxx], dim=-1)
        self.register_buffer("kx", kx)

        k = torch.sqrt(kxx**2 + kyy**2)
        self.register_buffer("k", k)

        # Fresnel transfer function
        H = torch.exp(1j * torch.pi * self.wavelength * voxel_size * self.k**2)
        self.register_buffer("H", H)

        # kirkland bandlimit
        if klim is not None:
            self.klim = klim
            kmask = torch.from_numpy(shapes.circle2d(nx, int(nx * klim)))[None, ...]
        else:
            kmask = torch.tensor(1)
        self.register_buffer("kmask", kmask)

    def multislice(self, V):
        # V is already rotated and is shape (B x Z x X x Y)
        if self.flipcurvature:
            V = torch.flip(V, dims=(1,))
        exitwave = torch.sqrt(
            torch.tensor(self.dose_per_area)
        )  # this should actually be expected dose per pixel
        for i in range(V.size(1)):
            # transmission function
            t = torch.exp(1j * self.sigma * V[:, i])

            # multiply with incident wave
            wv = t * exitwave

            # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
            exitwave = ifft2(fft2(wv) * self.H * self.kmask)
        return exitwave  # (B x X x Y)

    def firstborn(self, V):
        # V is already rotated and is shape (B x Z x X x Y)
        if self.flipcurvature:
            V = torch.flip(V, dims=(1,))
        exitwave = 1
        n = V.size(1)
        for i in range(n):
            # Fresnel transfer function
            F = torch.exp(
                    1j
                    * torch.pi
                    * self.wavelength
                    * self.voxel_size
                    * (n - i)
                    * self.k**2
                )
            # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
            exitwave += 1j * ifft2(fft2(self.sigma * V[:, i]) * F[None, ...])

        # multiply with dose
        exitwave *= torch.sqrt(torch.tensor(self.dose_per_area))
        return exitwave  # (B x X x Y)

    def fastfirstborn(self, V):
        # V is already rotated and is shape (B x Z x X x Y)
        if self.flipcurvature:
            V = torch.flip(V, dims=(1,))
        exitwave = 1
        n = V.size(1)
        exitwave_f = 0
        for i in range(n):
            # Fresnel transfer function
            F = torch.exp(
                    1j
                    * torch.pi
                    * self.wavelength
                    * self.voxel_size
                    * (n - i)
                    * self.k**2
                )
            # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
            exitwave_f += fft2(self.sigma * V[:, i]) * F[None, ...]
        exitwave += 1j * ifft2(exitwave_f)

        # multiply with dose
        exitwave *= torch.sqrt(torch.tensor(self.dose_per_area))
        return exitwave  # (B x X x Y)

    def projection(self, V):
        # V is already rotated and is shape (B x Z x X x Y)
        exitwave = torch.sqrt(torch.tensor(self.dose_per_area)) * torch.exp(
            1j * self.sigma * torch.sum(V, 1)
        )
        return exitwave  # (B x X x Y)

    # def ctf(self, V):
    #     # V is already rotated and is shape (B x Z x X x Y)
    #     projection = (
    #         torch.sqrt(torch.tensor(self.dose_per_area)) * self.sigma * torch.sum(V, 1)
    #     )
    #     return projection  # (B x X x Y)

    def ctf(self, V):
        # V is already rotated and is shape (B x Z x X x Y)
        projection = 2 * self.sigma * torch.sum(V, 1)
        return projection  # (B x X x Y)

    def forward(self, V):
        if self.projectmode == "multislice":
            return self.multislice(V)
        elif self.projectmode == "projection":
            return self.projection(V)
        elif self.projectmode == "firstborn":
            return self.firstborn(V)
        elif self.projectmode == "fastfirstborn":
            return self.fastfirstborn(V)
        elif self.projectmode == "ctf":
            return self.ctf(V)