from __future__ import annotations


import torch
import lightning as L
from torch.utils.checkpoint import checkpoint as _gradient_checkpoint
from .progress import track

from .arrays import disk2d
from .constants import energy_to_wavelength, interaction_parameter
from .fft import fft2, ifft2
from .rotations import VolumeRotator, build_affine_matrix


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
        ews_curvature_sign: str = "negative",
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
        ews_curvature_sign : str, optional
            Ewald sphere curvature sign matching CryoSPARC's convention.
            ``'negative'`` (default) or ``'positive'``. Affects multislice,
            Rytov, and first Born models.
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
        self.ews_curvature_sign = ews_curvature_sign
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
        if scattering_model in ("firstborn", "rytov", "kinematic"):
            if nz is None:
                raise ValueError(
                    f"nz is required for scattering_model={scattering_model!r}."
                )
            F_slices = []
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
                F_slices.append(f)
            F = torch.stack(F_slices)
            # self.register_buffer("F", F)
            self.register_buffer("F_real", F.real)
            self.register_buffer("F_imag", F.imag)

        # Kirkland bandlimit
        self.klim = klim
        if klim is not None:
            kmask = disk2d(nxy, int(nxy * klim))[None, ...]
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
        if self.ews_curvature_sign == "negative":
            V = torch.flip(V, dims=(1,))
        exitwave: torch.Tensor | None = None

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
            t = torch.exp(1j * self.sigma * self.pixel_size * V[:, i].to(self.device))

            # multiply with incident wave (unit incident wave on the first slice)
            wv = t if exitwave is None else t * exitwave

            # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
            exitwave = ifft2(fft2(wv) * F * self.kmask)
        assert exitwave is not None, "V must have at least one z-slice"
        return exitwave

    def rytov(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute exit wave using Rytov approximation.

        Faster than multislice but less accurate for thick specimens.

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
        The Rytov approximation computes the exit wave as:
        ψ = exp(iσ Σ_z [F(z) ★ V(z)])
        where F(z) is the Fresnel propagator from slice z to the exit plane
        and ★ denotes convolution (Fourier-domain multiplication).
        This is the exponentiated Born series, reducing to the first Born
        approximation for small V.
        """
        F = self.F_real + 1j * self.F_imag
        if self.ews_curvature_sign == "negative":
            V = torch.flip(V, dims=(1,))

        scattered = ifft2(fft2(1j * self.sigma * self.pixel_size * V) * F[None, ...])
        exitwave = torch.exp(torch.sum(scattered, dim=1))
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
        if self.ews_curvature_sign == "negative":
            V = torch.flip(V, dims=(1,))

        V_f = fft2(V)
        exitwave_f = self.sigma * self.pixel_size * V_f * F[None, ...]
        exitwave = ifft2(exitwave_f)
        exitwave = torch.sum(exitwave, 1)  # sum along Z
        exitwave = 1 + 1j * exitwave
        return exitwave

    def kinematic(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute exit wave using kinematic scattering approximation.

        Each slice sees only the incident plane wave (single scattering).
        Uses the full transmission function per slice, without the weak-phase
        linearisation applied by the ``firstborn`` method.

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
        The exit wave is the first-order term of the multislice Born series:

        ψ = 1 + Σ_z F⁻¹{ F[exp(iσΔz V_z) − 1] · F_z }

        where F_z propagates from slice z to the exit plane. Compared to
        ``firstborn``, the per-slice amplitude ``exp(iσΔz V_z) − 1`` is kept
        exact rather than linearised to ``iσΔz V_z``. The two are equivalent
        for small phase per slice (σΔz V ≪ 1) but diverge for thick slices
        or dense material.
        """
        F = self.F_real + 1j * self.F_imag
        if self.ews_curvature_sign == "negative":
            V = torch.flip(V, dims=(1,))

        t = torch.exp(1j * self.sigma * self.pixel_size * V) - 1
        exitwave = ifft2(fft2(t) * F[None, ...])
        exitwave = 1 + torch.sum(exitwave, 1)
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
        exitwave = torch.exp(1j * self.sigma * self.pixel_size * V_sum)
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
        projection = 2 * self.sigma * self.pixel_size * torch.sum(V, 1)
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
        elif self.scattering_model == "kinematic":
            V = complex_potential(V, alpha=self.alpha)
            return self.kinematic(V)
        elif self.scattering_model == "ctf":
            return self.ctf(V)
        else:
            raise ValueError(f"Unknown scattering_model: {self.scattering_model!r}")


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
        ews_curvature_sign: str = "negative",
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
        ews_curvature_sign : str
            Ewald sphere curvature sign matching CryoSPARC's convention.
            ``'negative'`` (default) or ``'positive'``. Affects multislice,
            Rytov, and first Born models.
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
        self.ews_curvature_sign = ews_curvature_sign
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
            kmask = disk2d(nxy, int(nxy * klim))[None, ...]
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

    # ------------------------------------------------------------------
    # Shared slice-fetching machinery
    #
    # Every iterative scattering model (multislice, rytov, firstborn,
    # kinematic, projection, parallel_rytov) needs to: resolve how many
    # Z-slices to traverse for the given rotation, decide the traversal
    # order (Ewald sphere curvature sign), and fetch each slice either
    # directly from V (identity transform) or by resampling the rotated
    # volume. These helpers factor that out so each model method only
    # has to express its own per-slice physics.
    # ------------------------------------------------------------------

    def _resolve_tilt_setup(
        self, V: torch.Tensor, theta_matrix: torch.Tensor
    ) -> tuple[int, VolumeRotator | None, bool]:
        """
        Resolve the number of Z-slices to traverse and the rotator needed
        to sample them, short-circuiting rotated-volume setup entirely for
        the identity transform.

        Returns
        -------
        nz_new : int
            Number of Z-slices to traverse (equals V's Z-extent for identity).
        rotator : VolumeRotator or None
            None for the identity transform (slices are read directly from V).
        is_identity : bool
        """
        is_identity = self._is_identity(theta_matrix)
        if is_identity:
            return V.shape[1], None, is_identity
        nz_new, rotator = self._setup_tilt(V, theta_matrix)
        return nz_new, rotator, is_identity

    def _roi_start(self, V: torch.Tensor) -> tuple[int, int]:
        """Top-left pixel of the centered (nxy, nxy) ROI within V's (Y, X) extent."""
        y_start = (V.shape[-2] - self.nxy) // 2
        x_start = (V.shape[-1] - self.nxy) // 2
        return y_start, x_start

    def _slice_processing_order(
        self, nz_new: int, device: str | torch.device
    ) -> torch.Tensor:
        """Z-slice traversal order; negative EWS curvature reverses it."""
        indices = torch.arange(nz_new, device=device)
        if self.ews_curvature_sign == "negative":
            indices = torch.flip(indices, dims=(0,))
        return indices

    def _fetch_volume_slices(
        self,
        V: torch.Tensor,
        slice_positions: torch.Tensor,
        is_identity: bool,
        rotator: VolumeRotator | None,
        theta_matrix: torch.Tensor,
        nz_new: int,
        y_start: int,
        x_start: int,
        device: str | torch.device,
    ) -> torch.Tensor:
        """
        Fetch Z-slices of `V` at the given processing-order positions.

        For the identity transform, slices are read directly out of `V`
        (no resampling needed). Otherwise they are resampled from the
        rotated volume via `rotator.sample_rotated_slices`.

        Parameters
        ----------
        slice_positions : torch.Tensor
            1D tensor of slice indices (in processing order), shape (K,).

        Returns
        -------
        torch.Tensor
            Real-valued slices, shape (B, K, nxy, nxy).
        """
        if is_identity:
            return V[
                :,
                slice_positions,
                y_start : y_start + self.nxy,
                x_start : x_start + self.nxy,
            ].to(device)
        assert rotator is not None
        slice_indices = slice_positions.float() - (nz_new - 1) / 2
        return rotator.sample_rotated_slices(
            V,
            theta_matrix,
            slice_indices=slice_indices,
            roi_size=(self.nxy, self.nxy),
            padding_mode="zeros",
        ).to(device)

    def _iter_slices(
        self,
        V: torch.Tensor,
        theta_matrix: torch.Tensor,
        slice_batch_size: int,
        description: str,
    ):
        """
        Yield ``(i, nz_new, slice_sample)`` for each Z-slice of a (possibly
        rotated) volume, in EWS processing order.

        For the identity transform, each slice is fetched individually
        (cheap direct indexing). Otherwise slices are fetched in
        ``slice_batch_size``-sized batches via `_fetch_volume_slices` and
        cached, trading memory for fewer `sample_rotated_slices` calls.

        Yields
        ------
        i : int
            Loop index (0-indexed position in processing order).
        nz_new : int
            Total number of slices being traversed.
        slice_sample : torch.Tensor
            Real-valued slice, shape (B, nxy, nxy).
        """
        nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
        device = self.device
        y_start, x_start = self._roi_start(V)
        indices = self._slice_processing_order(nz_new, device=V.device)

        pbar = track(
            range(nz_new),
            description=description,
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block: torch.Tensor | None = None
        for i in pbar:
            if is_identity:
                slice_sample = self._fetch_volume_slices(
                    V,
                    indices[i : i + 1],
                    is_identity,
                    rotator,
                    theta_matrix,
                    nz_new,
                    y_start,
                    x_start,
                    device,
                )[:, 0]
            else:
                if i % slice_batch_size == 0:
                    batch_end = min(i + slice_batch_size, nz_new)
                    slices_block = self._fetch_volume_slices(
                        V,
                        indices[i:batch_end],
                        is_identity,
                        rotator,
                        theta_matrix,
                        nz_new,
                        y_start,
                        x_start,
                        device,
                    )
                assert slices_block is not None
                slice_sample = slices_block[:, i % slice_batch_size]

            yield i, nz_new, slice_sample

    def multislice(
        self,
        V: torch.Tensor,
        theta_matrix: torch.Tensor,
        slice_batch_size: int = 1,
        checkpoint_chunks: int | None = None,
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative multislice on a transformed volume.

        Parameters
        ----------
        V : torch.Tensor
            Input 3D potential volume, shape ``(B, Z, Y, X)``.
        theta_matrix : torch.Tensor
            Affine rotation matrices, shape ``(B, 3, 4)``.
        slice_batch_size : int
            Number of slices to sample from V in one ``sample_rotated_slices``
            call. Larger values trade memory for fewer kernel launches.
        checkpoint_chunks : int or None
            If set, the slice loop is split into chunks of this size and each
            chunk is run under ``torch.utils.checkpoint`` (gradient
            checkpointing). Only the exitwave at chunk boundaries is retained
            for backprop; intermediate wave states are recomputed during the
            backward pass. This reduces activation memory from
            ``O(nz_new)`` to ``O(checkpoint_chunks)`` at the cost of one
            extra forward pass per chunk during backward.
            ``None`` disables checkpointing (original behaviour).
        """
        B = V.shape[0]
        device = self.device

        F = self.F_step_real + 1j * self.F_step_imag
        exitwave = torch.ones(
            B, self.nxy, self.nxy, device=device, dtype=torch.complex64
        )

        if checkpoint_chunks is None:
            for i, _, slice_sample in self._iter_slices(
                V, theta_matrix, slice_batch_size, "Multislice (Iterative)"
            ):
                slice_complex = complex_potential(slice_sample, alpha=self.alpha)
                t = torch.exp(1j * self.sigma * self.pixel_size * slice_complex)
                exitwave = ifft2(fft2(t * exitwave) * F * self.kmask)
        else:
            nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
            y_start, x_start = self._roi_start(V)
            indices = self._slice_processing_order(nz_new, device=V.device)

            # Gradient-checkpointed loop.
            # Constants captured in the closure; only (exitwave, V) are
            # passed as explicit differentiable inputs so that
            # use_reentrant=False can track them properly.
            _alpha = self.alpha
            _sigma = self.sigma
            _pixel_size = self.pixel_size
            _F = F
            _kmask = self.kmask

            pbar = track(
                range(0, nz_new, checkpoint_chunks),
                description="Multislice (Checkpointed)",
                transient=True,
                disable=not (self.progressbars),
            )
            for chunk_start in pbar:
                chunk_end = min(chunk_start + checkpoint_chunks, nz_new)
                _ci = indices[chunk_start:chunk_end]
                _cnz = nz_new
                _cid = is_identity
                _crot = rotator
                _ctheta = theta_matrix
                _cy, _cx = y_start, x_start

                def _make_chunk(
                    ci: torch.Tensor,
                    cnz: int,
                    cid: bool,
                    crot: VolumeRotator | None,
                    ctheta: torch.Tensor,
                    cy: int,
                    cx: int,
                ):
                    def _chunk(exitwave: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
                        ew = exitwave
                        dev = ew.device
                        for j in range(len(ci)):
                            s = self._fetch_volume_slices(
                                V, ci[j : j + 1], cid, crot, ctheta, cnz, cy, cx, dev
                            )[:, 0]
                            sc = complex_potential(s, alpha=_alpha)
                            t = torch.exp(1j * _sigma * _pixel_size * sc)
                            ew = ifft2(fft2(t * ew) * _F * _kmask)
                        return ew

                    return _chunk

                chunk_fn = _make_chunk(_ci, _cnz, _cid, _crot, _ctheta, _cy, _cx)
                exitwave = _gradient_checkpoint(
                    chunk_fn, exitwave, V, use_reentrant=False
                )

        return exitwave

    def projection(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative projection approximation on a transformed volume.
        """
        nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
        B = V.shape[0]
        device = self.device

        total_potential = torch.zeros((B, self.nxy, self.nxy), device=device)
        # Note: unlike the other iterative models, projection sums slices in
        # a commutative accumulation, so it does not apply the EWS curvature
        # traversal order.
        indices = torch.arange(nz_new, device=V.device)
        y_start, x_start = self._roi_start(V)

        pbar = track(
            range(nz_new),
            description="Projection (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block: torch.Tensor | None = None
        for i in pbar:
            if is_identity:
                total_potential += self._fetch_volume_slices(
                    V,
                    indices[i : i + 1],
                    is_identity,
                    rotator,
                    theta_matrix,
                    nz_new,
                    y_start,
                    x_start,
                    device,
                )[:, 0]
            else:
                if i % slice_batch_size == 0:
                    batch_end = min(i + slice_batch_size, nz_new)
                    slices_block = self._fetch_volume_slices(
                        V,
                        indices[i:batch_end],
                        is_identity,
                        rotator,
                        theta_matrix,
                        nz_new,
                        y_start,
                        x_start,
                        device,
                    )
                assert slices_block is not None
                total_potential += slices_block[:, i % slice_batch_size]

        total_complex = complex_potential(total_potential, alpha=self.alpha)
        exitwave = torch.exp(1j * self.sigma * self.pixel_size * total_complex)
        return exitwave

    def rytov(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative Rytov approximation on a transformed volume.
        """
        B = V.shape[0]
        device = self.device

        exitwave = torch.ones(
            (B, self.nxy, self.nxy), device=device, dtype=torch.complex64
        )

        for i, nz_new, slice_sample in self._iter_slices(
            V, theta_matrix, slice_batch_size, "Rytov (Iterative)"
        ):
            slice_complex = complex_potential(slice_sample, alpha=self.alpha)
            # Propagate transmission of slice i to exit plane
            # Distance is nz_new - i
            F_i = self._get_propagator(float(nz_new - i))
            exitwave = exitwave * torch.exp(
                ifft2(fft2(1j * self.sigma * self.pixel_size * slice_complex) * F_i)
            )

        return exitwave

    def parallel_rytov(
        self,
        V: torch.Tensor,
        theta_matrix: torch.Tensor,
        checkpoint_chunks: int | None = None,
    ) -> torch.Tensor:
        """
        Compute exit wave using a fully-parallel Rytov approximation.

        Unlike ``rytov`` (which still loops slice-by-slice), this method
        exploits the linearity of the Rytov sum to process all slices in
        batched FFT operations, with optional gradient checkpointing to
        bound memory.

        The exit wave is:

        .. math::
            \\psi = \\exp\\!\\left(
                \\sum_{z} \\mathcal{F}^{-1}\\!\\left[
                    \\mathcal{F}[i\\sigma\\,\\Delta z\\,V_z] \\cdot F_z
                \\right]
            \\right)

        Because the exponent is a *sum* over independent slices, the
        computation parallelises trivially across the Z dimension.

        Parameters
        ----------
        V : torch.Tensor
            Input 3D potential volume, shape ``(B, Z, Y, X)``.
        theta_matrix : torch.Tensor
            Affine rotation matrices, shape ``(B, 3, 4)``.
        checkpoint_chunks : int or None
            Chunk size along Z for gradient checkpointing.  Each chunk
            processes ``checkpoint_chunks`` slices with a single batched
            FFT pair and accumulates into a running phase sum.  Only the
            phase sum at chunk boundaries (12 MB) and the chunk's slice
            inputs are retained; intermediate FFT activations are
            recomputed during backward.  Suggested value: ``50``.
            ``None`` processes all slices in one pass (fast but high
            memory).

        Notes
        -----
        Accuracy: Rytov ≈ multislice when σ·Δz·V ≪ 1 per slice (valid
        for 300 keV electrons through typical cryo-ET ice thicknesses at
        5 Å/px).
        """
        nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
        device = self.device
        y_start, x_start = self._roi_start(V)

        # Processing order: negative EWS sign flips the slice traversal
        indices = self._slice_processing_order(nz_new, device=V.device)

        # Each slice j (processing order) propagates distance (nz_new - j)
        # to the exit plane.
        distances = torch.arange(nz_new, 0, -1, dtype=torch.float32, device=device)

        # Running phase accumulator — shape (B, nxy, nxy) complex64
        B = V.shape[0]
        phase_sum = torch.zeros(
            B, self.nxy, self.nxy, device=device, dtype=torch.complex64
        )

        _alpha = self.alpha
        _sigma = self.sigma
        _pixel_size = self.pixel_size
        _wavelength = self.wavelength
        _k2 = self.k2

        chunk_size = checkpoint_chunks if checkpoint_chunks is not None else nz_new

        pbar = track(
            range(0, nz_new, chunk_size),
            description="Rytov (Parallel)",
            transient=True,
            disable=not (self.progressbars),
        )
        for chunk_start in pbar:
            chunk_end = min(chunk_start + chunk_size, nz_new)
            _ci = indices[chunk_start:chunk_end]  # volume slice indices
            _dist = distances[chunk_start:chunk_end]  # Fresnel distances

            _cid = is_identity
            _crot = rotator
            _ctheta = theta_matrix
            _cy, _cx = y_start, x_start
            _cnz = nz_new

            def _make_chunk(ci, dist, cid, crot, ctheta, cy, cx, cnz):
                def _chunk(phase_sum: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
                    dev = phase_sum.device
                    slices = self._fetch_volume_slices(
                        V, ci, cid, crot, ctheta, cnz, cy, cx, dev
                    )
                    # (B, K, nxy, nxy) real → complex potential
                    slices_c = complex_potential(slices, alpha=_alpha)
                    # Fresnel kernels for this chunk — recomputed each time
                    # (cheap, no learnable params)
                    d = dist.to(dev)
                    F_chunk = torch.exp(
                        1j
                        * torch.pi
                        * _wavelength
                        * _pixel_size
                        * d[:, None, None]
                        * _k2[None, ...]
                    )  # (K, nxy, nxy)
                    # Batched FFT → Fresnel multiply → batched IFFT
                    scattered = ifft2(
                        fft2(1j * _sigma * _pixel_size * slices_c)
                        * F_chunk.unsqueeze(0)
                    )  # (B, K, nxy, nxy)
                    return phase_sum + scattered.sum(dim=1)

                return _chunk

            chunk_fn = _make_chunk(_ci, _dist, _cid, _crot, _ctheta, _cy, _cx, _cnz)
            if checkpoint_chunks is not None:
                phase_sum = _gradient_checkpoint(
                    chunk_fn, phase_sum, V, use_reentrant=False
                )
            else:
                phase_sum = chunk_fn(phase_sum, V)

        return torch.exp(phase_sum)

    def firstborn(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative first Born approximation on a transformed volume.
        """
        B = V.shape[0]
        device = self.device

        total_scattered = torch.zeros(
            (B, self.nxy, self.nxy), device=device, dtype=torch.complex64
        )

        for i, nz_new, slice_sample in self._iter_slices(
            V, theta_matrix, slice_batch_size, "First Born (Iterative)"
        ):
            slice_complex = complex_potential(slice_sample, alpha=self.alpha)
            F_i = self._get_propagator(float(nz_new - i))
            total_scattered += ifft2(fft2(slice_complex) * F_i)

        exitwave = 1 + 1j * self.sigma * self.pixel_size * total_scattered
        return exitwave

    def kinematic(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute exit wave using iterative kinematic scattering approximation.

        Like ``firstborn`` but uses the full per-slice transmission function
        ``exp(iσΔz V) − 1`` instead of the weak-phase linearisation ``iσΔz V``.
        """
        B = V.shape[0]
        device = self.device

        total_scattered = torch.zeros(
            (B, self.nxy, self.nxy), device=device, dtype=torch.complex64
        )

        for i, nz_new, slice_sample in self._iter_slices(
            V, theta_matrix, slice_batch_size, "Kinematic (Iterative)"
        ):
            slice_complex = complex_potential(slice_sample, alpha=self.alpha)
            t = torch.exp(1j * self.sigma * self.pixel_size * slice_complex) - 1
            F_i = self._get_propagator(float(nz_new - i))
            total_scattered += ifft2(fft2(t) * F_i)

        exitwave = 1 + total_scattered
        return exitwave

    def ctf(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute iterative projected potential (for CTF) with transformed sampling.
        """
        nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
        B = V.shape[0]
        device = self.device

        total_potential = torch.zeros((B, self.nxy, self.nxy), device=device)
        # Note: like projection, this is a commutative sum, so it does not
        # apply the EWS curvature traversal order.
        indices = torch.arange(nz_new, device=V.device)
        y_start, x_start = self._roi_start(V)

        pbar = track(
            range(nz_new),
            description="CTF Projection (Iterative)",
            transient=True,
            disable=not (self.progressbars),
        )

        slices_block: torch.Tensor | None = None
        for i in pbar:
            if is_identity:
                total_potential += self._fetch_volume_slices(
                    V,
                    indices[i : i + 1],
                    is_identity,
                    rotator,
                    theta_matrix,
                    nz_new,
                    y_start,
                    x_start,
                    device,
                )[:, 0]
            else:
                if i % slice_batch_size == 0:
                    batch_end = min(i + slice_batch_size, nz_new)
                    slices_block = self._fetch_volume_slices(
                        V,
                        indices[i:batch_end],
                        is_identity,
                        rotator,
                        theta_matrix,
                        nz_new,
                        y_start,
                        x_start,
                        device,
                    )
                assert slices_block is not None
                total_potential += slices_block[:, i % slice_batch_size]

        return 2 * self.sigma * self.pixel_size * total_potential

    def forward(
        self,
        V: torch.Tensor,
        pose: float | torch.Tensor,
        slice_batch_size: int = 1,
        checkpoint_chunks: int | None = None,
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
            return self.multislice(V, theta_matrix, slice_batch_size, checkpoint_chunks)
        elif self.scattering_model == "rytov_parallel":
            return self.parallel_rytov(V, theta_matrix, checkpoint_chunks)
        elif self.scattering_model == "rytov":
            return self.rytov(V, theta_matrix, slice_batch_size)
        elif self.scattering_model == "projection":
            return self.projection(V, theta_matrix, slice_batch_size)
        elif self.scattering_model == "firstborn":
            return self.firstborn(V, theta_matrix, slice_batch_size)
        elif self.scattering_model == "kinematic":
            return self.kinematic(V, theta_matrix, slice_batch_size)
        else:
            raise ValueError(f"Unknown scattering model: {self.scattering_model}")
