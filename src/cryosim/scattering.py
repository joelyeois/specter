from . import filters
import lightning as L
import numpy as np
import torch
from .fft_tools import fft2, ifft2
from rich.progress import track

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
    """Applies amplitude ratio, α, to create complex potential (PyTorch version)"""
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
            Number of pixels in volume, (nz, nxy, nxy).
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
        if scattering_model == "firstborn":
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
        Perform multislice on a volume tilted by angle_deg around the X-axis.
        Instead of rotating the volume (expensive), we sample slices on the fly.

        Args:
            V: (B, Z, Y, X) potential volume
            angle_deg: float, tilt angle in degrees
        """
        B, Z, Y, X = V.shape
        device = self.device

        # Convert angle to radians
        theta = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
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
                padding_mode="zeros",
                align_corners=True,
            )
            # slice_sample shape: (B, 1, 1, ny, nx) -> squeeze to (B, ny, nx)
            slice_sample = slice_sample.squeeze(1).squeeze(1).to(device)

            # Multislice propagation
            t = torch.exp(1j * self.sigma * slice_sample)
            wv = t * exitwave
            exitwave = ifft2(fft2(wv) * F * self.kmask)

        return exitwave

    def firstborn(self, V):
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
            V = complex_potential(V, alpha=self.alpha)
            return self.multislice(V)
        elif self.scattering_model == "projection":
            V = complex_potential(V, alpha=self.alpha)
            return self.projection(V)
        elif self.scattering_model == "firstborn":
            V = complex_potential(V, alpha=self.alpha)
            return self.firstborn(V)
        elif self.scattering_model == "ctf":
            return self.ctf(V)
