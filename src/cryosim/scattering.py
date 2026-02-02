from . import filters
from .rotations import VolumeRotator, build_affine_matrix
import lightning as L
import numpy as np
import torch
from .fft_tools import fft2, ifft2
from rich.progress import track

rest_mass_energy = 511.0e3  # [eV]
hc = 12.398e3  # [eV * Å]


def energy_to_wavelength(energy):
    """
    Convert electron energy to de Broglie wavelength.

    Uses the relativistic formula for electron wavelength calculation.

    Parameters
    ----------
    energy : float
        Electron beam energy in keV.

    Returns
    -------
    wavelength : float
        De Broglie wavelength in Å.

    Notes
    -----
    Uses the relativistic formula:
    λ = hc / sqrt(E * (E + 2*m_e*c²))
    where m_e*c² = 511 keV (rest mass energy of electron).
    """
    ev = energy * 1e3
    return hc / np.sqrt(ev * (ev + 2.0 * rest_mass_energy))


def interaction_parameter(energy):
    """
    Calculate the electron-specimen interaction parameter.

    Computes the interaction constant σ for electron scattering, following
    Kirkland Eq. (5.6).

    Parameters
    ----------
    energy : float
        Electron beam energy in keV.

    Returns
    -------
    sigma : torch.Tensor
        Interaction parameter in units of 1/(Å·V).

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Eq. (5.6), Springer US, Boston, MA, 2010.
    """
    w = energy_to_wavelength(energy)
    ev = energy * 1e3
    return (
        2.0
        * torch.pi
        / (w * ev)
        * ((ev + rest_mass_energy) / (ev + 2.0 * rest_mass_energy))
    )


def complex_potential(v, alpha=0.1):
    """
    Apply amplitude ratio to create complex potential.

    Converts real-valued potential to complex potential using amplitude
    contrast ratio α, following weak phase object approximation with
    absorption.

    Parameters
    ----------
    v : torch.Tensor
        Real-valued scattering potential.
    alpha : float, optional
        Amplitude contrast ratio. Typical values are 0.07-0.1.
        Default is 0.1.

    Returns
    -------
    complex_v : torch.Tensor
        Complex-valued potential with real part scaled by sqrt(1-α²)
        and imaginary part scaled by α.

    Notes
    -----
    The complex potential is given by:
    V = sqrt(1-α²)*v + i*α*v
    """
    scale_real = (1 - alpha**2) ** 0.5
    return torch.complex(scale_real * v, alpha * v)


class Scattering(L.LightningModule):
    def __init__(
        self,
        nxy,
        pixel_size,
        energy,
        dose_per_angstrom,
        scattering_model="multislice",
        klim=None,
        flip_curvature=False,
        nz=None,
        alpha=0.0,
        progressbars=True,
    ):
        """
        A scattering module to compute the 2D exitwave from a 3D scattering
        potential. Various scattering modes are available.

        Parameters
        ----------
        nxy : int
            Number of pixels in x and y dimensions, (nxy, nxy).
        pixel_size : float
            Pixel size in Ångströms. Assumes dz equals pixel_size.
        energy : float
            Electron beam energy in keV. Typical values are 100, 120, 200, or 300 keV.
        dose_per_angstrom : float
            Electron dose in e-/Å^2.
        scattering_model : str, optional
            Scattering model to use. Options: 'multislice', 'firstborn',
            'projection', 'ctf' (in order of increasing approximations).
            Default is 'multislice'.
        klim : float, optional
            Bandlimit parameter for Kirkland's FFT aliasing prevention.
            Setting klim=0.66 prevents aliasing but reduces spatial frequency
            content. Default is None (no bandlimiting).
        flip_curvature : bool, optional
            Ewald sphere curvature sign. False for positive curvature,
            True for negative (CryoSPARC convention). Only affects multislice
            and first Born models. Default is False.
        nz : int, optional
            Number of slices in z dimension. Required for firstborn model.
            Default is None.
        alpha : float, optional
            Amplitude contrast ratio. Default is 0.0.
        progressbars : bool, optional
            Whether to display progress bars during computation. Default is True.

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
           Springer US, Boston, MA, 2010.

        """
        super().__init__()
        self.nxy = nxy
        self.nz = nz
        self.pixel_size = pixel_size

        # model params
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.wavelength = energy_to_wavelength(energy)
        self.sigma = interaction_parameter(energy)
        self.scattering_model = scattering_model
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.progressbars = progressbars

        # frequency coordinates
        kx = torch.fft.fftshift(torch.fft.fftfreq(nxy, pixel_size))
        kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
        k = torch.sqrt(kxx**2 + kyy**2)

        # Fresnel transfer function for multislice
        if scattering_model == "multislice":
            # original
            # F = torch.exp(-1j * torch.pi * self.wavelength * pixel_size * k**2)
            F = torch.exp(1j * torch.pi * self.wavelength * pixel_size * k**2)

            # self.register_buffer("F", F)
            self.register_buffer("F_real", F.real)
            self.register_buffer("F_imag", F.imag)

        # Fresnel transfer function for first Born
        if scattering_model == "firstborn" or scattering_model == "rytov":
            F = []
            # for i in tqdm(range(nz), desc='Create first Born propagators', leave=False):
            for i in track(
                range(nz),
                description="Create first Born propagators",
                transient=True,
                disable=not (self.progressbars),
            ):
                # original
                # f = torch.exp(-1j * torch.pi * self.wavelength * pixel_size *
                #               (nz - i) * k**2)
                f = torch.exp(
                    1j * torch.pi * self.wavelength * pixel_size * (nz - i) * k**2
                )
                F.append(f)
            F = torch.stack(F)
            # self.register_buffer("F", F)
            self.register_buffer("F_real", F.real)
            self.register_buffer("F_imag", F.imag)

        # Kirkland bandlimit
        self.klim = klim
        if klim is not None:
            kmask = filters.circle2d(nxy, int(nxy * klim))[None, ...]
            self.register_buffer("kmask", kmask)
        else:
            self.kmask = 1

    def multislice(self, V):
        """
        Compute exit wave using multislice algorithm.

        Iteratively propagates an electron wave through slices of the 3D
        potential, accounting for both transmission through each slice and
        Fresnel propagation between slices.

        Parameters
        ----------
        V : torch.Tensor
            Complex-valued 3D potential volume with shape (B, Z, Y, X) where
            B is batch size, Z is number of slices.

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave with shape (B, Y, X).

        Notes
        -----
        The multislice algorithm alternates between:
        1. Transmission: ψ * exp(iσV)
        2. Propagation: FFT[ψ] * F * FFT⁻¹
        where F is the Fresnel propagator and σ is the interaction parameter.
        """
        F = self.F_real + 1j * self.F_imag
        if self.flip_curvature:
            V = torch.flip(V, dims=(1,))
        exitwave = np.sqrt(self.dose_per_pixel)

        # iterate across z-planes of 3D potentials.
        # for i in tqdm(range(V.size(1)), desc='Multislicing', leave=False):
        for i in track(
            range(V.size(1)),
            description="Multislicing",
            transient=True,
            disable=not (self.progressbars),
        ):
            # transmission function
            # t = torch.exp(1j * self.sigma * complex_potential(V[:, i], alpha=self.alpha).to(self.device))
            t = torch.exp(1j * self.sigma * V[:, i].to(self.device))

            # multiply with incident wave
            wv = t * exitwave

            # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
            exitwave = ifft2(fft2(wv) * F * self.kmask)
        return exitwave

    def multislice_and_tilt(
        self, V, angle_deg, slice_batch_size: int = 1, return_slices: bool = False
    ):
        """
        Perform multislice on a tilted volume for tomography.

        Instead of rotating the entire volume (computationally expensive),
        samples slices on-the-fly from the tilted coordinate system.

        Parameters
        ----------
        V : torch.Tensor
            Potential volume with shape (B, Z, Y, X).
        angle_deg : float
            Tilt angle in degrees around the X-axis.
        slice_batch_size : int, optional
            Number of slices to sample and move to GPU at once.
            Higher values improve PCIe efficiency but use more VRAM.
            Default is 1.
        return_slices : bool, optional
            If True, returns the stack of sampled slices along with the exit wave.
            This is useful for debugging and verification. Default is False.

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave from tilted projection, shape (B, Y, X).
        slices : torch.Tensor (optional)
            If return_slices is True, returns (B, nz_new, Y, X) tensor of sampled slices.

        Notes
        -----
        Uses the VolumeRotator to extract tilted slices in blocks.
        The tilt is performed around the X-axis following tomography conventions.
        """
        B, Z, Y, X = V.shape
        device = self.device

        # Convert angle to radians for internal use if needed,
        # but build_affine_matrix/Rotation handles it.
        theta_rad = torch.deg2rad(torch.tensor(angle_deg, device=device))
        c, s = torch.cos(theta_rad), torch.sin(theta_rad)

        # Determine new Z range (projected extent along new Z axis)
        new_z_extent = (Y * abs(s) + Z * abs(c)) * self.pixel_size

        # Number of slices in new Z direction, keeping step size = pixel_size
        nz_new = int(np.ceil(new_z_extent.item() / self.pixel_size))

        # Setup VolumeRotator
        # Rotator should be on V.device for grid_sample efficiency (usually CPU for large V)
        rotator = VolumeRotator(
            nz=Z, ny=Y, nx=X, origin="relion", padding_mode="reflection"
        ).to(V.device)

        # Rotation matrix around X-axis
        # R_x(theta) = [[1, 0, 0], [0, c, -s], [0, s, c]]
        # Matrix must be on V.device for rotator
        R = (
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                device=V.device,
                dtype=V.dtype,
            )
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        # Build affine matrix
        theta_matrix = build_affine_matrix(R)

        # Prepare multislice variables on GPU
        F = self.F_real + 1j * self.F_imag
        exitwave = torch.full(
            (B, self.nxy, self.nxy),
            np.sqrt(self.dose_per_pixel),
            device=device,
            dtype=torch.complex64,
        )

        # Optional: collect slices
        all_slices = [] if return_slices else None

        # Iterate over new Z slices
        indices = torch.arange(nz_new, device=V.device)
        pbar = track(
            range(nz_new),
            description=f"Multislicing (Tilt {angle_deg:.1f})",
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block = None
        for i in pbar:
            # Sample a new block if needed
            if i % slice_batch_size == 0:
                batch_end = min(i + slice_batch_size, nz_new)
                batch_indices = indices[i:batch_end]
                slice_indices = batch_indices - (nz_new - 1) / 2

                slices_block = rotator.sample_rotated_slices(
                    V,
                    theta_matrix,
                    slice_indices=slice_indices,
                    roi_size=(self.nxy, self.nxy),
                    padding_mode="zeros",
                )

                if return_slices:
                    all_slices.append(slices_block.cpu())

                slices_block = slices_block.to(device)  # (B, K, nxy, nxy)

            # Extract single slice from the block
            slice_sample = slices_block[:, i % slice_batch_size]

            # Multislice propagation
            t = torch.exp(
                1j * self.sigma * complex_potential(slice_sample, alpha=self.alpha)
            )
            wv = t * exitwave
            exitwave = ifft2(fft2(wv) * F * self.kmask)

        if return_slices:
            return exitwave, torch.cat(all_slices, dim=1)
        return exitwave

    def rytov(self, V):
        """
        Compute exit wave using Rytov approximation.

        Treats scattering as a single
        event. Faster than multislice but less accurate for thick specimens.

        Parameters
        ----------
        V : torch.Tensor
            Complex-valued 3D potential volume with shape (B, Z, Y, X).

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave with shape (B, Y, X).

        Notes
        -----
        The first Born approximation computes the exit wave as:
        ψ = exp(i Σ_z [F(z) * V(z)])
        where F(z) accounts for Fresnel propagation from slice z to the exit plane.
        """
        F = self.F_real + 1j * self.F_imag
        if self.flip_curvature:
            V = torch.flip(V, dims=(1,))

        t = torch.exp(1j * self.sigma * V)  # (B x Z x X x Y)
        exitwaves = ifft2(fft2(t) * F[None, ...])  # propagate each slice
        exitwave = torch.prod(exitwaves, 1)  # product along Z

        # multiply with dose
        exitwave = exitwave * self.dose_per_pixel**0.5
        return exitwave  # (B x X x Y)

    def firstborn(self, V):
        """
        Compute exit wave using first Born approximation.

        Weak phase object approximation that treats scattering as a single
        event. Faster than multislice but less accurate for thick specimens.

        Parameters
        ----------
        V : torch.Tensor
            Complex-valued 3D potential volume with shape (B, Z, Y, X).

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave with shape (B, Y, X).

        Notes
        -----
        The first Born approximation computes the exit wave as:
        ψ = 1 + i Σ_z [F(z) * V(z)]
        where F(z) accounts for Fresnel propagation from slice z to the exit plane.
        """
        F = self.F_real + 1j * self.F_imag
        if self.flip_curvature:
            V = torch.flip(V, dims=(1,))

        V_f = fft2(V)
        exitwave_f = self.sigma * V_f * F[None, ...]
        exitwave = ifft2(exitwave_f)
        exitwave = torch.sum(exitwave, 1)  # sum along Z
        exitwave = 1 + 1j * exitwave

        # multiply with dose
        exitwave *= np.sqrt(self.dose_per_pixel)
        return exitwave

    def projection(self, V):
        """
        Compute exit wave using projection approximation.

        Phase object approximation that projects the 3D potential onto a 2D
        plane, ignoring Fresnel propagation effects.

        Parameters
        ----------
        V : torch.Tensor
            Complex-valued 3D potential volume with shape (B, Z, Y, X).

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave with shape (B, Y, X).

        Notes
        -----
        The exit wave is computed as:
        ψ = exp(i σ Σ_z V(z))
        This is valid only for thin specimens where propagation effects are negligible.
        """
        exitwave = np.sqrt(self.dose_per_pixel) * torch.exp(
            1j * self.sigma * torch.sum(V, 1)
        )
        return exitwave

    def ctf(self, V):
        """
        Compute projected potential for CTF-based imaging.

        Returns the projected potential (not exit wave) for use with
        contrast transfer function (CTF) imaging model.

        Parameters
        ----------
        V : torch.Tensor
            Real-valued 3D potential volume with shape (B, Z, Y, X).

        Returns
        -------
        projection : torch.Tensor
            Real-valued 2D projected potential with shape (B, Y, X).

        Notes
        -----
        Computes: 2σ Σ_z V(z)
        The factor of 2 accounts for the phase-contrast imaging relationship.
        CTF is applied separately in the aberration module.
        """
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
            V = complex_potential(V, alpha=self.alpha)
            return self.multislice(V)
        elif self.scattering_model == "rytov":
            V = complex_potential(V, alpha=self.alpha)
            return self.rytov(V)
        elif self.scattering_model == "projection":
            V = complex_potential(V, alpha=self.alpha)
            return self.projection(V)
        elif self.scattering_model == "firstborn":
            V = complex_potential(V, alpha=self.alpha)
            return self.firstborn(V)
        elif self.scattering_model == "ctf":
            return self.ctf(V)
