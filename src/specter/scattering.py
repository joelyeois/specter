from __future__ import annotations


import torch
import lightning as L
from .progress import track

from . import filters
from .fft_tools import fft2, ifft2
from .rotations import VolumeRotator, build_affine_matrix

rest_mass_energy = 511.0e3  # [eV]
hc = 12.398e3  # [eV * Å]


def energy_to_wavelength(energy: float) -> float:
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
    ev = energy * 1.0e3  # [eV]
    return hc / (ev * (ev + 2.0 * rest_mass_energy)) ** 0.5


def interaction_parameter(energy: float) -> torch.Tensor:
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


def complex_potential(v: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
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
    V = v * (sqrt(1-α²) + i*α)
    """
    if alpha == 0.0:
        return v
    c = (1 - alpha**2) ** 0.5 + 1j * alpha
    return v * c


class Scattering(L.LightningModule):
    def __init__(
        self,
        nxy: int,
        pixel_size: float,
        energy: float,
        scattering_model: str = "multislice",
        klim: float | None = None,
        flip_curvature: bool = False,
        nz: int | None = None,
        alpha: float = 0.0,
        progressbars: bool = True,
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
        self.wavelength = energy_to_wavelength(energy)
        self.sigma = interaction_parameter(energy)
        self.scattering_model = scattering_model
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.progressbars = progressbars

        # frequency coordinates
        kx = torch.fft.fftfreq(nxy, pixel_size)
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

    def multislice(self, V: torch.Tensor) -> torch.Tensor:
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
        exitwave = 1.0

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

    def rytov(self, V: torch.Tensor) -> torch.Tensor:
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
        return exitwave  # (B x X x Y)

    def firstborn(self, V: torch.Tensor) -> torch.Tensor:
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
        return exitwave

    def projection(self, V: torch.Tensor) -> torch.Tensor:
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
        V_sum = torch.sum(V, 1)
        exitwave = torch.exp(1j * self.sigma * V_sum)
        return exitwave

    def ctf(self, V: torch.Tensor) -> torch.Tensor:
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

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """
        Perform scattering on a batch of 3D potentials.

        V is batch of 3D real-valued potentials with shape (B x Z x X x Y), outputs
        a batch of 2D exitwaves with shape (B x Y x X). The CTF scattering model
        outputs projected potential instead of exitwave.

        Note that the CTF model does not require computing the complex-valued
        potentials since it is built into the aberration function.

        Parameters
        ----------
        V : torch.Tensor
            Batch of 3D potentials.

        Returns
        -------
        psi : torch.Tensor
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


class IterativeScattering(L.LightningModule):
    """
    Scattering module that computes the 2D exitwave from a 3D scattering
    potential using on-the-fly slice sampling. This is highly memory efficient
    as it avoids rotating the entire 3D volume.
    """

    def __init__(
        self,
        nxy: int,
        pixel_size: float,
        energy: float,
        scattering_model: str = "multislice",
        klim: float | None = None,
        flip_curvature: bool = False,
        alpha: float = 0.0,
        progressbars: bool = True,
    ):
        """
        Parameters
        ----------
        nxy : int
            Number of pixels in x and y dimensions.
        pixel_size : float
            Pixel size in Å.
        energy : float
            Electron beam energy in keV.
        scattering_model : str
            Scattering model to use ('multislice', 'firstborn', 'rytov', 'projection', 'ctf').
        klim : float, optional
            Bandlimit parameter.
        flip_curvature : bool
            Whether to flip the curvature of the Ewald sphere.
        alpha : float
            Amplitude contrast ratio.
        progressbars : bool
            Whether to show progress bars.
        """
        super().__init__()
        self.nxy = nxy
        self.pixel_size = pixel_size
        self.energy = energy
        self.wavelength = energy_to_wavelength(energy)
        self.sigma = interaction_parameter(energy)
        self.scattering_model = scattering_model
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.progressbars = progressbars

        # frequency coordinates
        kx = torch.fft.fftfreq(nxy, pixel_size)
        kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
        k = torch.sqrt(kxx**2 + kyy**2)
        self.register_buffer("k2", k**2)

        # Kirkland bandlimit
        self.klim = klim
        if klim is not None:
            kmask = filters.circle2d(nxy, int(nxy * klim))[None, ...]
            self.register_buffer("kmask", kmask)
        else:
            self.kmask = 1

        # Precompute single-step Fresnel propagator if using multislice
        F_step = torch.exp(1j * torch.pi * self.wavelength * pixel_size * k**2)
        self.register_buffer("F_step_real", F_step.real)
        self.register_buffer("F_step_imag", F_step.imag)

    def _is_identity(self, theta_matrix: torch.Tensor) -> bool:
        """Check if the affine matrix is identity (no rotation or translation)."""
        B = theta_matrix.shape[0]
        identity = (
            torch.eye(3, 4, device=theta_matrix.device, dtype=theta_matrix.dtype)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        return torch.allclose(theta_matrix, identity, atol=1e-6)

    def _setup_tilt(
        self, V: torch.Tensor, theta_matrix: torch.Tensor
    ) -> tuple[int, VolumeRotator]:
        """Helper to calculate rotation parameters and setup VolumeRotator."""
        B, Z, Y, X = V.shape

        # Determine nz_new for arbitrary rotation
        # R is the rotation matrix (B, 3, 3) from output to input coordinates.
        R = theta_matrix[:, :3, :3]

        # Corners of the volume in pixel units relative to center.
        # VolumeRotator expects (z, y, x) or (x, y, z)?
        # Grid sample uses (x, y, z).
        corners = torch.tensor(
            [
                [-X / 2, -Y / 2, -Z / 2],
                [X / 2, -Y / 2, -Z / 2],
                [-X / 2, Y / 2, -Z / 2],
                [X / 2, Y / 2, -Z / 2],
                [-X / 2, -Y / 2, Z / 2],
                [X / 2, -Y / 2, Z / 2],
                [-X / 2, Y / 2, Z / 2],
                [X / 2, Y / 2, Z / 2],
            ],
            device=V.device,
            dtype=V.dtype,
        ).t()  # (3, 8)

        # Output Z-axis in input frame is R[:, 2].
        # P_out = R.T @ (P_in - T)
        # z_out = R.T[2, :] @ (P_in - T)
        # We ignore T for nz_new as it just shifts the center of sampling.
        rotated_corners = torch.bmm(
            R.transpose(1, 2), corners.unsqueeze(0).expand(B, -1, -1)
        )
        z_min = rotated_corners[:, 2, :].min(dim=1).values
        z_max = rotated_corners[:, 2, :].max(dim=1).values

        # Max extent across the batch
        nz_new = int(torch.ceil((z_max - z_min).max()).item())
        nz_new = max(1, nz_new)

        rotator = VolumeRotator(
            nz=Z,
            ny=Y,
            nx=X,
            origin="relion",
            padding_mode="reflection",
            init_base_grid=False,  # sample_rotated_slices builds its own grid; base_grid would OOM for large volumes
        ).to(V.device)

        return nz_new, rotator

    def _get_propagator(self, distance_slices: float) -> torch.Tensor:
        """Compute the Fresnel propagator for a given distance in slices."""
        F = torch.exp(
            1j
            * torch.pi
            * self.wavelength
            * self.pixel_size
            * distance_slices
            * self.k2
        )
        return F

    def multislice(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative multislice on a transformed volume.
        """
        is_identity = self._is_identity(theta_matrix)
        if is_identity:
            nz_new = V.shape[1]
            rotator = None
        else:
            nz_new, rotator = self._setup_tilt(V, theta_matrix)

        B = V.shape[0]
        device = self.device

        F = self.F_step_real + 1j * self.F_step_imag
        exitwave = torch.ones(
            B, self.nxy, self.nxy, device=device, dtype=torch.complex64
        )

        indices = torch.arange(nz_new, device=V.device)
        if self.flip_curvature:
            indices = torch.flip(indices, dims=(0,))

        pbar = track(
            range(nz_new),
            description="Multislice (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block = None
        y_start = (V.shape[-2] - self.nxy) // 2
        x_start = (V.shape[-1] - self.nxy) // 2

        for i in pbar:
            if is_identity:
                slice_sample = V[
                    :,
                    indices[i],
                    y_start : y_start + self.nxy,
                    x_start : x_start + self.nxy,
                ].to(device)
            else:
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
                    ).to(device)
                slice_sample = slices_block[:, i % slice_batch_size]

            slice_complex = complex_potential(slice_sample, alpha=self.alpha)
            t = torch.exp(1j * self.sigma * slice_complex)
            exitwave = ifft2(fft2(t * exitwave) * F * self.kmask)
        return exitwave

    def projection(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative projection approximation on a transformed volume.
        """
        is_identity = self._is_identity(theta_matrix)
        if is_identity:
            nz_new = V.shape[1]
            rotator = None
        else:
            nz_new, rotator = self._setup_tilt(V, theta_matrix)

        B = V.shape[0]
        device = self.device

        total_potential = torch.zeros((B, self.nxy, self.nxy), device=device)
        indices = torch.arange(nz_new, device=V.device)

        pbar = track(
            range(nz_new),
            description="Projection (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        y_start = (V.shape[-2] - self.nxy) // 2
        x_start = (V.shape[-1] - self.nxy) // 2

        for i in pbar:
            if is_identity:
                total_potential += V[
                    :,
                    indices[i],
                    y_start : y_start + self.nxy,
                    x_start : x_start + self.nxy,
                ].to(device)
            else:
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
                    ).to(device)
                total_potential += slices_block[:, i % slice_batch_size]

        total_complex = complex_potential(total_potential, alpha=self.alpha)
        exitwave = torch.exp(1j * self.sigma * total_complex)
        return exitwave

    def rytov(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative Rytov approximation on a transformed volume.
        """
        is_identity = self._is_identity(theta_matrix)
        if is_identity:
            nz_new = V.shape[1]
            rotator = None
        else:
            nz_new, rotator = self._setup_tilt(V, theta_matrix)

        B = V.shape[0]
        device = self.device

        exitwave = torch.ones(
            (B, self.nxy, self.nxy), device=device, dtype=torch.complex64
        )
        indices = torch.arange(nz_new, device=V.device)
        if self.flip_curvature:
            indices = torch.flip(indices, dims=(0,))

        pbar = track(
            range(nz_new),
            description="Rytov (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block = None
        y_start = (V.shape[-2] - self.nxy) // 2
        x_start = (V.shape[-1] - self.nxy) // 2

        for i in pbar:
            if is_identity:
                slice_sample = V[
                    :,
                    indices[i],
                    y_start : y_start + self.nxy,
                    x_start : x_start + self.nxy,
                ].to(device)
            else:
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
                    ).to(device)
                slice_sample = slices_block[:, i % slice_batch_size]

            slice_complex = complex_potential(slice_sample, alpha=self.alpha)
            # Propagate transmission of slice i to exit plane
            # Distance is nz_new - i
            F_i = self._get_propagator(float(nz_new - i))
            exitwave = exitwave * torch.exp(
                ifft2(fft2(1j * self.sigma * slice_complex) * F_i)
            )

        return exitwave

    def firstborn(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative first Born approximation on a transformed volume.
        """
        is_identity = self._is_identity(theta_matrix)
        if is_identity:
            nz_new = V.shape[1]
            rotator = None
        else:
            nz_new, rotator = self._setup_tilt(V, theta_matrix)

        B = V.shape[0]
        device = self.device

        total_scattered = torch.zeros(
            (B, self.nxy, self.nxy), device=device, dtype=torch.complex64
        )
        indices = torch.arange(nz_new, device=V.device)
        if self.flip_curvature:
            indices = torch.flip(indices, dims=(0,))

        pbar = track(
            range(nz_new),
            description="First Born (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block = None
        y_start = (V.shape[-2] - self.nxy) // 2
        x_start = (V.shape[-1] - self.nxy) // 2

        for i in pbar:
            if is_identity:
                slice_sample = V[
                    :,
                    indices[i],
                    y_start : y_start + self.nxy,
                    x_start : x_start + self.nxy,
                ].to(device)
            else:
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
                    ).to(device)
                slice_sample = slices_block[:, i % slice_batch_size]

            slice_complex = complex_potential(slice_sample, alpha=self.alpha)
            F_i = self._get_propagator(float(nz_new - i))
            total_scattered += ifft2(fft2(slice_complex) * F_i)

        exitwave = 1 + 1j * self.sigma * total_scattered
        return exitwave

    def ctf(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute iterative projected potential (for CTF) with transformed sampling.
        """
        is_identity = self._is_identity(theta_matrix)
        if is_identity:
            nz_new = V.shape[1]
            rotator = None
        else:
            nz_new, rotator = self._setup_tilt(V, theta_matrix)

        B = V.shape[0]
        device = self.device

        total_potential = torch.zeros((B, self.nxy, self.nxy), device=device)
        indices = torch.arange(nz_new, device=V.device)

        pbar = track(
            range(nz_new),
            description="CTF Projection (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        y_start = (V.shape[-2] - self.nxy) // 2
        x_start = (V.shape[-1] - self.nxy) // 2

        for i in pbar:
            if is_identity:
                total_potential += V[
                    :,
                    indices[i],
                    y_start : y_start + self.nxy,
                    x_start : x_start + self.nxy,
                ].to(device)
            else:
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
                    ).to(device)
                total_potential += slices_block[:, i % slice_batch_size]

        return 2 * self.sigma * total_potential

    def forward(
        self, V: torch.Tensor, pose: float | torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Forward pass for iterative scattering.

        Parameters
        ----------
        V : torch.Tensor
            Input 3D potential volume.
        pose : float or torch.Tensor
            If float, interpreted as tilt angle in degrees around X.
            If torch.Tensor, interpreted as affine matrix (B, 3, 4).
        slice_batch_size : int
            Number of slices to batch for sampling.
        """
        if isinstance(pose, (int, float)) or (
            isinstance(pose, torch.Tensor) and pose.ndim == 0
        ):
            # Backward compatibility for tilt angle
            angle_deg = pose
            B = V.shape[0]
            device = V.device
            theta_rad = torch.deg2rad(torch.as_tensor(angle_deg, device=device))
            c, s = torch.cos(theta_rad), torch.sin(theta_rad)
            R = (
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                    device=device,
                    dtype=V.dtype,
                )
                .unsqueeze(0)
                .expand(B, -1, -1)
            )
            theta_matrix = build_affine_matrix(R)
        else:
            theta_matrix = pose.to(V.device)

        if self.scattering_model == "ctf":
            return self.ctf(V, theta_matrix, slice_batch_size)

        if self.scattering_model == "multislice":
            return self.multislice(V, theta_matrix, slice_batch_size)
        elif self.scattering_model == "rytov":
            return self.rytov(V, theta_matrix, slice_batch_size)
        elif self.scattering_model == "projection":
            return self.projection(V, theta_matrix, slice_batch_size)
        elif self.scattering_model == "firstborn":
            return self.firstborn(V, theta_matrix, slice_batch_size)
        else:
            raise ValueError(f"Unknown scattering model: {self.scattering_model}")
