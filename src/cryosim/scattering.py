from . import filters
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
            Number of pixels in x and y dimensions, (nz, nxy, nxy).
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

    def multislice_and_tilt(self, V, angle_deg):
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

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave from tilted projection, shape (B, Y, X).

        Notes
        -----
        Uses trilinear interpolation (grid_sample) to extract tilted slices.
        The tilt is performed around the X-axis following tomography conventions.
        """
        B, Z, Y, X = V.shape
        device = self.device

        # Convert angle to radians
        theta = torch.deg2rad(angle_deg)
        c, s = torch.cos(theta), torch.sin(theta)

        # Determine new Z range (projected extent along new Z axis)
        # Original corners (relative to center)
        # We assume pixel_size is isotropic for simplicity in extent calculation
        # If not, we should multiply by self.pixel_size
        # But grid_sample works in normalized coords [-1, 1], so we can work in pixel units
        # if we handle aspect ratio correctly.
        # Let's work in pixel units for the grid, but physical units for Z-steps.

        # Dimensions
        # z_dim = Z * self.pixel_size
        # y_dim = Y * self.pixel_size

        # New Z extent (bounding box height after rotation)
        # z_new = |z * cos - y * sin| is wrong.
        # p_L = R p_S. z_L = y_S * sin + z_S * cos
        # Max extent is sum of projections of half-widths
        new_z_extent = (Y * abs(s) + Z * abs(c)) * self.pixel_size

        # Number of slices in new Z direction
        # We keep step size = pixel_size
        nz_new = int(np.ceil(new_z_extent.item() / self.pixel_size))

        # Start z_L (centered at 0)
        z_L_start = -(nz_new - 1) * self.pixel_size / 2

        # Pre-calculate coordinate grid for a single slice (x_L, y_L)
        # x_L corresponds to X axis (unchanged by X-rotation)
        # y_L corresponds to Y axis (rotated)
        # We want the output image to be (Y, X) size (or (nxy, nxy))
        # The detector is usually fixed size.
        # If we want to capture the whole projected volume, we might need larger Y.
        # But usually in tomography, the field of view is fixed (the detector).
        # So we iterate over the detector grid (self.nxy, self.nxy).

        ny_out, nx_out = self.nxy, self.nxy

        # Grid in L frame (detector plane)
        # shape (ny, nx)
        y_L_grid, x_L_grid = torch.meshgrid(
            (torch.arange(ny_out) - ny_out // 2) * self.pixel_size,
            (torch.arange(nx_out) - nx_out // 2) * self.pixel_size,
            indexing="ij",
        )

        # Flatten for batch processing if needed, or keep as (ny, nx)
        # We need (B, ny, nx, 3) for grid_sample

        # Prepare multislice variables
        F = self.F_real + 1j * self.F_imag
        exitwave = np.sqrt(self.dose_per_pixel)  # Scalar or broadcastable

        # Iterate over new Z slices
        for i in track(
            range(nz_new),
            description=f"Multislicing (Tilt {angle_deg:.1f})",
            transient=True,
            disable=not (self.progressbars),
        ):
            z_L = z_L_start + i * self.pixel_size

            # Coordinates in L frame: (x_L, y_L, z_L)
            # We need to map these to S frame (x_S, y_S, z_S)
            # Inverse rotation: p_S = R^T p_L
            # R_x(theta) = [[1, 0, 0], [0, c, -s], [0, s, c]]
            # R^T = [[1, 0, 0], [0, c, s], [0, -s, c]]
            # x_S = x_L
            # y_S = c * y_L + s * z_L
            # z_S = -s * y_L + c * z_L

            x_S = x_L_grid  # (ny, nx)
            y_S = c * y_L_grid + s * z_L
            z_S = -s * y_L_grid + c * z_L

            # Normalize to [-1, 1] for grid_sample
            # Note: grid_sample expects (x, y, z) order in the last dimension
            # And it expects normalized coordinates in range [-1, 1].
            # 1.0 corresponds to index (size-1). -1.0 corresponds to index 0.
            # Actually, align_corners=True (default) means -1 is left edge center, 1 is right edge center.
            # align_corners=False means -1 is left boundary, 1 is right boundary.
            # Let's use align_corners=True to match standard pixel coordinates if we mapped correctly.
            # The spatial extent of the volume V is defined by its shape (Z, Y, X).
            # Center is at (0,0,0).
            # Extent is [-X*ps/2, X*ps/2] roughly.
            # Normalized coord = 2 * coord / (size * ps) ?
            # More precisely: index = (coord / ps) + size/2
            # norm = (index / (size-1)) * 2 - 1

            # Let's do it via indices for clarity
            # index_x = x_S / self.pixel_size + (X - 1) / 2
            # norm_x = (index_x / (X - 1)) * 2 - 1
            # Simplify: norm_x = x_S / (self.pixel_size * (X - 1) / 2)

            norm_x = x_S / (self.pixel_size * (X - 1) / 2)
            norm_y = y_S / (self.pixel_size * (Y - 1) / 2)
            norm_z = z_S / (self.pixel_size * (Z - 1) / 2)

            # Stack to (1, ny, nx, 3) -> (B, ny, nx, 3)
            # grid_sample expects (x, y, z) in last dim
            grid = (
                torch.stack([norm_x, norm_y, norm_z], dim=-1)
                .unsqueeze(0)
                .expand(B, -1, -1, -1)
            )

            # Sample V
            # V is (B, Z, Y, X). grid_sample expects (B, C, D_in, H_in, W_in)
            # We treat V as (B, 1, Z, Y, X)
            # grid is (B, D_out, H_out, W_out, 3)
            # Here output is 2D slice (1, ny, nx). So D_out=1.
            # We can reshape grid to (B, 1, ny, nx, 3)
            grid = grid.unsqueeze(1)

            # Sample
            # padding_mode='reflection' handles the "infinite ice" assumption
            slice_sample = torch.nn.functional.grid_sample(
                V.unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            # slice_sample shape: (B, 1, 1, ny, nx) -> squeeze to (B, ny, nx)
            slice_sample = slice_sample.squeeze(1).squeeze(1).to(device)

            # Multislice propagation
            t = torch.exp(
                1j * self.sigma * complex_potential(slice_sample, alpha=self.alpha)
            )
            wv = t * exitwave
            exitwave = ifft2(fft2(wv) * F * self.kmask)

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
