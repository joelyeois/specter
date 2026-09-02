from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence


import torch
import lightning as L
from torch.utils.checkpoint import checkpoint as _gradient_checkpoint
from .progress import track

from .arrays import center_crop
from .constants import energy_to_wavelength, interaction_parameter
from .potential import apply_amplitude_contrast
from .fft import fft2, ifft2
from .rotations import VolumeRotator, build_affine_matrix


# Number of z-slices whose transmission functions are evaluated per batched
# torch.exp call in Scattering.multislice. Purely a performance knob: output
# and gradients are bitwise identical for any value, so it is deliberately not
# exposed as a constructor argument.
#
# 8 is the knee of the measured curve. Going 1 -> 8 is worth 16-60% (box
# 64-256, batch 1-16); going 8 -> 64 is <= 11% and inside run-to-run noise,
# while forward peak memory grows linearly in the chunk size (+3% at 8, +118%
# unchunked). Backward peak memory is flat in it, since autograd retains every
# slice's transmission function either way. Measured on an L40.
_MULTISLICE_SLICE_CHUNK = 8


class Scattering(L.LightningModule):
    def __init__(
        self,
        nxy: int,
        pixel_size: float,
        voltage: float,
        scattering_model: str = "multislice",
        klim: float | None = None,
        ews_curvature_sign: str = "negative",
        nz: int | None = None,
        alpha: float = 0.0,
        progressbars: bool = True,
    ):
        """
        A scattering module to compute the 2D exitwave from a 3D scattering
        potential. Various scattering modes are available, following the
        multislice formalism of Kirkland [1]_.

        Parameters
        ----------
        nxy : int
            Number of pixels in x and y dimensions, (nxy, nxy).
        pixel_size : float
            Pixel size in Å. Assumes dz equals pixel_size.
        voltage : float
            Electron beam accelerating voltage in kV. Typical values are 100,
            120, 200, or 300 kV.
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
        self.voltage = voltage
        self.wavelength = energy_to_wavelength(voltage)
        self.sigma = interaction_parameter(voltage)
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
            self.register_buffer("F_real", F.real, persistent=False)
            self.register_buffer("F_imag", F.imag, persistent=False)

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
            self.register_buffer("F_real", F.real, persistent=False)
            self.register_buffer("F_imag", F.imag, persistent=False)

        # Kirkland bandlimit. Built directly against k (already in the same
        # native/unshifted FFT frequency order F is in, DC at index 0) --
        # not via disk2d(), whose disk is centered at index n//2 (fftshift
        # convention) and would keep the Nyquist corner while masking out
        # DC if multiplied straight into an unshifted spectrum.
        self.klim = klim
        if klim is not None:
            k_nyquist = 1.0 / (2.0 * pixel_size)
            kmask = (k <= klim * k_nyquist).to(torch.float32)[None, ...]
            self.register_buffer("kmask", kmask, persistent=False)
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
            3D potential volume with shape (B, Z, Y, X) where B is batch
            size, Z is number of slices. Either real-valued, in which case
            amplitude contrast (``self.alpha``) is applied to each slice
            chunk as it is consumed, or already complex (absorptive), in
            which case it is used as given.

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

        A real ``V`` is complexified one chunk at a time rather than whole:
        ``V * (sqrt(1 - alpha^2) + 1j * alpha)`` is elementwise, so the
        result is bitwise identical, but the whole-volume form holds a
        complex64 copy of the entire padded volume for the duration of the
        loop -- 4 GB for a 512-pixel box with ``pad_fft`` -- for a value
        each slice consumes once.
        """
        F = self.F_real + 1j * self.F_imag

        # Fold the bandlimit into the propagator once, rather than once per
        # slice. Bitwise-exact only because kmask is binary (0.0/1.0, see
        # __init__), so the reassociation from (psi_k * F) * kmask to
        # psi_k * (F * kmask) is exact; a soft/apodised mask would not be.
        Fk = F * self.kmask

        exitwave: torch.Tensor | None = None

        # Iterate across z-planes of 3D potentials. The recursion is inherently
        # sequential (each slice transmits the previous slice's wave), but the
        # transmission functions themselves are independent, so they are
        # evaluated a chunk at a time and consumed via unbind. This is purely a
        # performance restructuring -- output and gradients are bitwise
        # identical to the per-slice form. See _MULTISLICE_SLICE_CHUNK.
        #
        # A negative Ewald-sphere sign traverses the slices in reverse. That is
        # done by walking the chunks backwards and flipping each one as it is
        # consumed, not by `torch.flip(V, dims=(1,))` up front: the whole-volume
        # flip is a full copy of the padded volume (6 GB for three 512-pixel
        # boxes), for a slice order the loop can produce for free.
        chunks: Sequence[torch.Tensor] = V.split(_MULTISLICE_SLICE_CHUNK, dim=1)
        reverse = self.ews_curvature_sign == "negative"
        if reverse:
            chunks = chunks[::-1]
        for chunk in track(
            chunks,
            description="Multislicing",
            transient=True,
            disable=not (self.progressbars),
        ):
            # transmission functions for the whole chunk, in one kernel.
            # .to() here rather than per slice keeps a volume that lives off
            # the compute device streaming in bounded-size blocks.
            chunk = chunk.to(self.device)
            if reverse:
                chunk = chunk.flip(1)
            if not chunk.is_complex():
                chunk = apply_amplitude_contrast(chunk, alpha=self.alpha)
            t_chunk = torch.exp(1j * self.sigma * self.pixel_size * chunk)

            for t in t_chunk.unbind(1):
                # multiply with incident wave (unit incident wave on the first slice)
                wv = t if exitwave is None else t * exitwave

                # propagate wave to next slice, also applies Kirkland's 0.66 bandlimit
                exitwave = ifft2(fft2(wv) * Fk)
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

        # Sum over slices in Fourier space, then take a single inverse FFT.
        # The inverse FFT is linear, so it commutes with the sum over z: this
        # is the same quantity as summing nz inverse-transformed planes, but
        # costs one inverse FFT instead of nz. See the note in
        # IterativeScattering.parallel_rytov, which does the same.
        scattered_k = (fft2(1j * self.sigma * self.pixel_size * V) * F[None, ...]).sum(
            dim=1
        )
        exitwave = torch.exp(ifft2(scattered_k))
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
        # Sum along Z in Fourier space -- one inverse FFT instead of nz. The
        # inverse FFT is linear, so it commutes with the sum (see rytov).
        exitwave = ifft2(torch.sum(exitwave_f, 1))
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
        # Sum along Z in Fourier space -- one inverse FFT instead of nz. The
        # inverse FFT is linear, so it commutes with the sum (see rytov).
        exitwave = 1 + ifft2(torch.sum(fft2(t) * F[None, ...], 1))
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
            # amplitude contrast is applied per chunk inside multislice --
            # see its docstring for why not here.
            return self.multislice(V)
        elif self.scattering_model == "rytov":
            V = apply_amplitude_contrast(V, alpha=self.alpha)
            return self.rytov(V)
        elif self.scattering_model == "projection":
            V = apply_amplitude_contrast(V, alpha=self.alpha)
            return self.projection(V)
        elif self.scattering_model == "firstborn":
            V = apply_amplitude_contrast(V, alpha=self.alpha)
            return self.firstborn(V)
        elif self.scattering_model == "kinematic":
            V = apply_amplitude_contrast(V, alpha=self.alpha)
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
        voltage: float,
        scattering_model: str = "multislice",
        klim: float | None = None,
        ews_curvature_sign: str = "negative",
        alpha: float = 0.0,
        progressbars: bool = True,
        roi_padding_mode: str = "zeros",
        pad_fft: bool = False,
        fft_pad_margin: int = 16,
    ):
        """
        Parameters
        ----------
        nxy : int
            Number of pixels in x and y dimensions.
        pixel_size : float
            Pixel size in Å.
        voltage : float
            Electron beam accelerating voltage in kV.
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
        roi_padding_mode : str, optional
            ``padding_mode`` passed to ``grid_sample`` (via
            ``VolumeRotator.sample_rotated_slices``) in ``_fetch_volume_slices``, for
            ROI queries (under rotation) that fall outside the input volume's extent.
            ``'zeros'`` (default) treats out-of-bounds queries as vacuum, which is the
            physically correct assumption for a discrete object surrounded by empty
            space, but wrong for a continuous medium like a vitreous-ice lamella (which
            has no vacuum -- real content everywhere) sampled with an ROI larger than
            the requested volume's own padding: it introduces a real, measurable
            density cliff exactly at the volume boundary (a discontinuity with slowly
            decaying, ~1/k, Fourier content, spread broadly rather than confined to a
            single spatial frequency). ``'reflection'`` or ``'border'`` avoid the value
            discontinuity (at the cost of a gradient kink, ~1/k^2, a much gentler
            artifact) by mirroring or clamping the real content instead. See
            ``TiltSeriesGenerator``'s ``pad_fft`` for where this matters in practice.
        pad_fft : bool, optional
            For ``scattering_model="multislice"`` only: give the per-slice FFT-based
            Fresnel propagation extra canvas headroom for the whole ``nz_new``-step
            recursion, rather than propagating at exactly ``nxy`` with zero headroom.
            Zero headroom lets each step's circular convolution wrap slightly at the
            same fixed frame boundary every step; under tilt (hundreds to 1000+ steps),
            that small per-step wraparound compounds coherently into a real, visible
            artifact along all four frame edges (confirmed via direct comparison against
            ``scattering_model="projection"``, which cannot exhibit this artifact by
            construction, at both toy and production scale -- see ``dev/tilt series/``).
            The fix is to pad once, run the *entire* recursion at the padded size, and
            crop back to ``nxy`` only once at the end -- padding and cropping every
            individual step was also tested and made things worse, since it discards
            genuinely-propagating field content on every iteration instead of only
            preventing wraparound. Default False.
        fft_pad_margin : int, optional
            Padding added on each side of the propagation canvas when ``pad_fft=True``,
            i.e. the recursion runs at ``nxy + 2 * fft_pad_margin``. Always filled with
            zeros (not reflection): the region a tilted slice's flanks sample is real
            vacuum, not continuing material, so reflecting would fabricate density that
            isn't there. Validated at 16-32px -- the result is identical across that
            whole range (converged), including at production scale (nz=368, ~1000+
            multislice steps), so this does not need to scale with volume size or step
            count. Default 16.
        """
        super().__init__()
        self.nxy = nxy
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.wavelength = energy_to_wavelength(voltage)
        self.sigma = interaction_parameter(voltage)
        self.scattering_model = scattering_model
        self.ews_curvature_sign = ews_curvature_sign
        self.alpha = alpha
        self.progressbars = progressbars
        self.roi_padding_mode = roi_padding_mode
        self.pad_fft = pad_fft
        self.fft_pad_margin = fft_pad_margin

        # frequency coordinates
        kx = torch.fft.fftfreq(nxy, pixel_size)
        kxx, kyy = torch.meshgrid(kx, kx, indexing="ij")
        k = torch.sqrt(kxx**2 + kyy**2)
        self.register_buffer("k2", k**2, persistent=False)

        # Kirkland bandlimit -- see the note in Scattering.__init__: built
        # directly against k (native/unshifted FFT order), not disk2d().
        self.klim = klim
        if klim is not None:
            k_nyquist = 1.0 / (2.0 * pixel_size)
            kmask = (k <= klim * k_nyquist).to(torch.float32)[None, ...]
            self.register_buffer("kmask", kmask, persistent=False)
        else:
            self.kmask = 1

        # Precompute single-step Fresnel propagator if using multislice
        F_step = torch.exp(1j * torch.pi * self.wavelength * pixel_size * k**2)
        self.register_buffer("F_step_real", F_step.real, persistent=False)
        self.register_buffer("F_step_imag", F_step.imag, persistent=False)

        # Second propagator/bandlimit at the padded canvas size, used only when
        # pad_fft=True (see multislice()).
        self.padded_nxy = nxy + 2 * fft_pad_margin if pad_fft else nxy
        if pad_fft:
            kx_p = torch.fft.fftfreq(self.padded_nxy, pixel_size)
            kxx_p, kyy_p = torch.meshgrid(kx_p, kx_p, indexing="ij")
            k_p = torch.sqrt(kxx_p**2 + kyy_p**2)
            F_step_p = torch.exp(1j * torch.pi * self.wavelength * pixel_size * k_p**2)
            self.register_buffer("F_step_padded_real", F_step_p.real, persistent=False)
            self.register_buffer("F_step_padded_imag", F_step_p.imag, persistent=False)
            if klim is not None:
                k_nyquist = 1.0 / (2.0 * pixel_size)
                kmask_p = (k_p <= klim * k_nyquist).to(torch.float32)[None, ...]
                self.register_buffer("kmask_padded", kmask_p, persistent=False)
            else:
                self.kmask_padded = 1

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

    def _roi_start(
        self, V: torch.Tensor, roi_size: int | None = None
    ) -> tuple[int, int]:
        """Top-left pixel of the centered (roi_size, roi_size) ROI within V's (Y, X) extent."""
        size = self.nxy if roi_size is None else roi_size
        y_start = (V.shape[-2] - size) // 2
        x_start = (V.shape[-1] - size) // 2
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
        roi_size: int | None = None,
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
        roi_size : int, optional
            Size of the square ROI to fetch. Defaults to ``self.nxy``; callers
            that need a larger canvas (e.g. ``multislice`` with ``pad_fft=True``)
            pass it explicitly.

        Returns
        -------
        torch.Tensor
            Real-valued slices, shape (B, K, roi_size, roi_size).
        """
        size = self.nxy if roi_size is None else roi_size
        if is_identity:
            B = V.shape[0]
            K = slice_positions.shape[0]
            Y, X = V.shape[-2], V.shape[-1]
            y_end, x_end = y_start + size, x_start + size
            # Requested window may exceed V's actual extent (e.g. a padded canvas
            # requested from a volume only sized for the unpadded ROI). Anything
            # outside V is genuine vacuum -- zero-fill it instead of silently
            # truncating (which used to produce a shape mismatch downstream).
            y0, y1 = max(y_start, 0), min(y_end, Y)
            x0, x1 = max(x_start, 0), min(x_end, X)
            if y0 == y_start and y1 == y_end and x0 == x_start and x1 == x_end:
                return V[:, slice_positions, y_start:y_end, x_start:x_end].to(device)
            out = torch.zeros(B, K, size, size, device=device, dtype=V.dtype)
            if y1 > y0 and x1 > x0:
                out[:, :, y0 - y_start : y1 - y_start, x0 - x_start : x1 - x_start] = V[
                    :, slice_positions, y0:y1, x0:x1
                ].to(device)
            return out
        assert rotator is not None
        slice_indices = slice_positions.float() - (nz_new - 1) / 2
        # device= lets sample_rotated_slices interpolate directly on `device`
        # via a small windowed transfer when V lives elsewhere -- e.g. when
        # TiltSeriesGenerator's volume didn't fit on the compute device and
        # stayed on CPU (see its `_ensure_volume_placed`). See its docstring.
        # When V and device already match, this is exactly the original
        # direct grid_sample, unchanged.
        return rotator.sample_rotated_slices(
            V,
            theta_matrix,
            slice_indices=slice_indices,
            roi_size=(size, size),
            padding_mode=self.roi_padding_mode,
            device=device,
        )

    def _iter_slices(
        self,
        V: torch.Tensor,
        theta_matrix: torch.Tensor,
        slice_batchsize: int,
        description: str,
        roi_size: int | None = None,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        """
        Yield ``(i, nz_new, slice_sample)`` for each Z-slice of a (possibly
        rotated) volume, in EWS processing order.

        For the identity transform, each slice is fetched individually
        (cheap direct indexing). Otherwise slices are fetched in
        ``slice_batchsize``-sized batches via `_fetch_volume_slices` and
        cached, trading memory for fewer `sample_rotated_slices` calls.

        Parameters
        ----------
        roi_size : int, optional
            Size of the square ROI to fetch per slice. Defaults to ``self.nxy``;
            ``multislice`` passes ``self.padded_nxy`` when ``pad_fft=True``.

        Yields
        ------
        i : int
            Loop index (0-indexed position in processing order).
        nz_new : int
            Total number of slices being traversed.
        slice_sample : torch.Tensor
            Real-valued slice, shape (B, roi_size, roi_size).
        """
        nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
        device = self.device
        y_start, x_start = self._roi_start(V, roi_size=roi_size)
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
                    roi_size=roi_size,
                )[:, 0]
            else:
                if i % slice_batchsize == 0:
                    batch_end = min(i + slice_batchsize, nz_new)
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
                        roi_size=roi_size,
                    )
                assert slices_block is not None
                slice_sample = slices_block[:, i % slice_batchsize]

            yield i, nz_new, slice_sample

    def multislice(
        self,
        V: torch.Tensor,
        theta_matrix: torch.Tensor,
        slice_batchsize: int = 1,
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
        slice_batchsize : int
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

        if self.pad_fft:
            F = self.F_step_padded_real + 1j * self.F_step_padded_imag
            kmask = self.kmask_padded
            canvas = self.padded_nxy
            roi_size: int | None = canvas
        else:
            F = self.F_step_real + 1j * self.F_step_imag
            kmask = self.kmask
            canvas = self.nxy
            roi_size = None

        # Fold the bandlimit into the propagator once, rather than once per
        # slice -- the same reassociation `Scattering.multislice` already makes,
        # and exact for the same reason: kmask is binary (0.0/1.0), or the
        # Python int 1 when klim is None, in which case `psi_k * F * kmask` was
        # spending a whole extra full-size complex kernel multiplying by one.
        # Measured bitwise-identical, and 1.08-1.18x on the slice step at
        # tilt-series (1200^2), micrograph (4096^2) and particle-stack shapes.
        Fk = F * kmask

        exitwave = torch.ones(B, canvas, canvas, device=device, dtype=torch.complex64)

        if checkpoint_chunks is None:
            for i, _, slice_sample in self._iter_slices(
                V,
                theta_matrix,
                slice_batchsize,
                "Multislice (Iterative)",
                roi_size=roi_size,
            ):
                slice_complex = apply_amplitude_contrast(slice_sample, alpha=self.alpha)
                t = torch.exp(1j * self.sigma * self.pixel_size * slice_complex)
                exitwave = ifft2(fft2(t * exitwave) * Fk)
        else:
            nz_new, rotator, is_identity = self._resolve_tilt_setup(V, theta_matrix)
            y_start, x_start = self._roi_start(V, roi_size=roi_size)
            indices = self._slice_processing_order(nz_new, device=V.device)

            # Gradient-checkpointed loop.
            # Constants captured in the closure; only (exitwave, V) are
            # passed as explicit differentiable inputs so that
            # use_reentrant=False can track them properly.
            _alpha = self.alpha
            _sigma = self.sigma
            _pixel_size = self.pixel_size
            _Fk = Fk
            _roi_size = roi_size

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
                    roi: int | None,
                ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                    def _chunk(exitwave: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
                        ew = exitwave
                        dev = ew.device
                        for j in range(len(ci)):
                            s = self._fetch_volume_slices(
                                V,
                                ci[j : j + 1],
                                cid,
                                crot,
                                ctheta,
                                cnz,
                                cy,
                                cx,
                                dev,
                                roi_size=roi,
                            )[:, 0]
                            sc = apply_amplitude_contrast(s, alpha=_alpha)
                            t = torch.exp(1j * _sigma * _pixel_size * sc)
                            ew = ifft2(fft2(t * ew) * _Fk)
                        return ew

                    return _chunk

                chunk_fn = _make_chunk(
                    _ci, _cnz, _cid, _crot, _ctheta, _cy, _cx, _roi_size
                )
                exitwave = _gradient_checkpoint(
                    chunk_fn, exitwave, V, use_reentrant=False
                )

        if self.pad_fft:
            exitwave = center_crop(exitwave, self.nxy, dim=(-2, -1))
        return exitwave

    def projection(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batchsize: int = 1
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
                if i % slice_batchsize == 0:
                    batch_end = min(i + slice_batchsize, nz_new)
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
                total_potential += slices_block[:, i % slice_batchsize]

        total_complex = apply_amplitude_contrast(total_potential, alpha=self.alpha)
        exitwave = torch.exp(1j * self.sigma * self.pixel_size * total_complex)
        return exitwave

    def rytov(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batchsize: int = 1
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
            V, theta_matrix, slice_batchsize, "Rytov (Iterative)"
        ):
            slice_complex = apply_amplitude_contrast(slice_sample, alpha=self.alpha)
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

        Mathematically identical to ``rytov``, but restructured for
        throughput and bounded memory -- see "When to use this over
        ``rytov``" below.

        The exit wave is:

        .. math::
            \\psi = \\exp\\!\\left(
                \\sum_{z} \\mathcal{F}^{-1}\\!\\left[
                    \\mathcal{F}[i\\sigma\\,\\Delta z\\,V_z] \\cdot F_z
                \\right]
            \\right)

        Because the exponent is a *sum* over independent slices -- unlike
        multislice's ``psi_i = t_i * psi_{i-1}``, no slice's contribution
        depends on any other slice's result -- the computation
        parallelises trivially across the Z dimension.

        When to use this over ``rytov``
        --------------------------------
        ``rytov`` computes the exact same sum, but as a Python loop over
        slices, one small FFT pair per slice, with no way to bound
        backward's memory: every slice's activations are retained for
        gradients regardless of how many slices there are. That's fine
        for a handful of slices (e.g. an untilted or mildly tilted single
        particle). It stops being fine for a tilted cryo-ET volume, where
        ``nz_new`` can reach the hundreds to low thousands: this method
        batches those slices into a handful of large, vectorized FFT
        calls instead of many small sequential ones (faster forward
        pass, independent of gradients), and ``checkpoint_chunks`` lets
        backward's memory scale with chunk size instead of ``nz_new``.
        Both benefits come from the same independence-across-slices
        property that makes multislice's true recursion unable to use
        this approach at all -- multislice's ``checkpoint_chunks``
        exists for the memory side of this same problem, but has no
        batching counterpart to reach for, since its slices are not
        independent.

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

            def _make_chunk(
                ci: torch.Tensor,
                dist: torch.Tensor,
                cid: bool,
                crot: VolumeRotator | None,
                ctheta: torch.Tensor,
                cy: int,
                cx: int,
                cnz: int,
            ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                def _chunk(phase_sum: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
                    dev = phase_sum.device
                    slices = self._fetch_volume_slices(
                        V, ci, cid, crot, ctheta, cnz, cy, cx, dev
                    )
                    # (B, K, nxy, nxy) real → complex potential
                    slices_c = apply_amplitude_contrast(slices, alpha=_alpha)
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
                    # Batched FFT → Fresnel multiply → sum over slices (still in
                    # Fourier space, where IFFT is linear and commutes with the
                    # sum) → single IFFT. Summing before the IFFT is
                    # mathematically identical to summing after it, but avoids
                    # ever backpropagating a broadcast-expanded gradient (from
                    # the slice-dim sum) through an FFT: MKL's FFT backward
                    # raises "Inconsistent configuration parameters" on such
                    # inputs.
                    scattered = fft2(
                        1j * _sigma * _pixel_size * slices_c
                    ) * F_chunk.unsqueeze(0)
                    return phase_sum + ifft2(scattered.sum(dim=1))

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
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batchsize: int = 1
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
            V, theta_matrix, slice_batchsize, "First Born (Iterative)"
        ):
            slice_complex = apply_amplitude_contrast(slice_sample, alpha=self.alpha)
            F_i = self._get_propagator(float(nz_new - i))
            total_scattered += ifft2(fft2(slice_complex) * F_i)

        exitwave = 1 + 1j * self.sigma * self.pixel_size * total_scattered
        return exitwave

    def kinematic(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batchsize: int = 1
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
            V, theta_matrix, slice_batchsize, "Kinematic (Iterative)"
        ):
            slice_complex = apply_amplitude_contrast(slice_sample, alpha=self.alpha)
            t = torch.exp(1j * self.sigma * self.pixel_size * slice_complex) - 1
            F_i = self._get_propagator(float(nz_new - i))
            total_scattered += ifft2(fft2(t) * F_i)

        exitwave = 1 + total_scattered
        return exitwave

    def ctf(
        self, V: torch.Tensor, theta_matrix: torch.Tensor, slice_batchsize: int = 1
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
                if i % slice_batchsize == 0:
                    batch_end = min(i + slice_batchsize, nz_new)
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
                total_potential += slices_block[:, i % slice_batchsize]

        return 2 * self.sigma * self.pixel_size * total_potential

    def forward(
        self,
        V: torch.Tensor,
        pose: float | torch.Tensor,
        slice_batchsize: int = 1,
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
        slice_batchsize : int
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
            return self.ctf(V, theta_matrix, slice_batchsize)

        if self.scattering_model == "multislice":
            return self.multislice(V, theta_matrix, slice_batchsize, checkpoint_chunks)
        elif self.scattering_model == "rytov_parallel":
            return self.parallel_rytov(V, theta_matrix, checkpoint_chunks)
        elif self.scattering_model == "rytov":
            return self.rytov(V, theta_matrix, slice_batchsize)
        elif self.scattering_model == "projection":
            return self.projection(V, theta_matrix, slice_batchsize)
        elif self.scattering_model == "firstborn":
            return self.firstborn(V, theta_matrix, slice_batchsize)
        elif self.scattering_model == "kinematic":
            return self.kinematic(V, theta_matrix, slice_batchsize)
        else:
            raise ValueError(f"Unknown scattering model: {self.scattering_model}")
