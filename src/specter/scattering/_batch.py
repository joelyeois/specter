"""
`Scattering`: exit waves from a batch of already-rotated potentials.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import lightning as L
import torch

from ..constants import energy_to_wavelength, interaction_parameter
from ..fft import fft2, ifft2
from ..potential import apply_amplitude_contrast
from ..progress import track
from . import _kernels
from ._kernels import (
    bandlimit_mask,
    frequency_grid,
    fresnel_propagator,
    phase_scale,
)
from specter.options import EwaldSphereSign, ScatteringModel


class Scattering(L.LightningModule):
    def __init__(
        self,
        nxy: int,
        pixel_size: float,
        voltage: float,
        scattering_model: ScatteringModel = "multislice",
        klim: float | None = None,
        ews_curvature_sign: EwaldSphereSign = "negative",
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

        k = frequency_grid(nxy, pixel_size)
        k2 = k**2

        # Fresnel transfer function for multislice: one slice step.
        if scattering_model == "multislice":
            F = fresnel_propagator(k2, self.wavelength, pixel_size)
            self.register_buffer("F_real", F.real, persistent=False)
            self.register_buffer("F_imag", F.imag, persistent=False)

        # Fresnel transfer function for first Born
        if scattering_model in ("firstborn", "rytov", "kinematic"):
            if nz is None:
                raise ValueError(
                    f"nz is required for scattering_model={scattering_model!r}."
                )
            # One propagator per slice, from that slice to the exit plane.
            F = torch.stack(
                [
                    fresnel_propagator(k2, self.wavelength, pixel_size * (nz - i))
                    for i in track(
                        range(nz),
                        description="Create first Born propagators",
                        transient=True,
                        disable=not (self.progressbars),
                    )
                ]
            )
            self.register_buffer("F_real", F.real, persistent=False)
            self.register_buffer("F_imag", F.imag, persistent=False)

        self.klim = klim
        kmask = bandlimit_mask(k, pixel_size, klim)
        if isinstance(kmask, torch.Tensor):
            self.register_buffer("kmask", kmask, persistent=False)
        else:
            self.kmask = kmask

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
        # identical to the per-slice form. See _kernels._MULTISLICE_SLICE_CHUNK.
        #
        # A negative Ewald-sphere sign traverses the slices in reverse. That is
        # done by walking the chunks backwards and flipping each one as it is
        # consumed, not by `torch.flip(V, dims=(1,))` up front: the whole-volume
        # flip is a full copy of the padded volume (6 GB for three 512-pixel
        # boxes), for a slice order the loop can produce for free.
        chunks: Sequence[torch.Tensor] = V.split(
            _kernels._MULTISLICE_SLICE_CHUNK, dim=1
        )
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

    def _propagator(self, V: torch.Tensor) -> torch.Tensor:
        """
        The complex first-Born propagator stack ``(Z, Y, X)`` on `V`'s device
        and complex dtype, in slice order matching the traversal.

        Cached: it was rebuilt from ``F_real``/``F_imag`` on every call, a
        1 GB complex volume at a 512 box, and the reversed traversal for
        ``ews_curvature_sign="negative"`` flipped the *volume* instead --
        a full copy of ``V`` in forward and again in backward -- when
        flipping the constant propagator once is the same sum.
        """
        cdtype = torch.complex128 if V.dtype == torch.float64 else torch.complex64
        key = (V.device, cdtype, self.ews_curvature_sign)
        cached = getattr(self, "_propagator_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        F = (self.F_real + 1j * self.F_imag).to(device=V.device, dtype=cdtype)
        if self.ews_curvature_sign == "negative":
            F = F.flip(0)
        self._propagator_cache = (key, F)
        return F

    def _fourier_slice_sum(
        self, V: torch.Tensor, per_slice: Callable[[torch.Tensor], torch.Tensor] | None
    ) -> torch.Tensor:
        """
        ``sum_z fft2(g(V_z)) * F_z`` over the slices, a z-chunk at a time.

        The three single-scatter models differ only in ``g`` (identity, or
        ``exp(i sigma dz V) - 1`` for kinematic) and in what multiplies the
        result. Chunking bounds the working set to one chunk's complex
        spectrum instead of the whole volume's: the unchunked form held
        the complex volume, its spectrum and the product at once, 3 x 3.2 GB
        for three 512-boxes, and that was most of a reconstruction step's
        device memory. Autograd keeps only each chunk's propagator view.

        Parameters
        ----------
        V : torch.Tensor
            ``(B, Z, Y, X)``, real or complex.
        per_slice : callable, optional
            Applied to each chunk before the FFT. None means the identity.

        Returns
        -------
        torch.Tensor
            ``(B, Y, X)`` complex.
        """
        F = self._propagator(V)
        B, nz, ny, nx = V.shape
        chunk = max(1, _kernels._SLICE_SUM_CHUNK_ELEMENTS // (B * ny * nx))
        acc: torch.Tensor | None = None
        # `split`, not `V[:, z0:z1]`: a narrow slice's backward materialises
        # a full-size zero gradient per chunk (the O(Z^2) trap 41e0125 took
        # out of multislice), 17 ms and 1.6 GB per chunk at a 512 box, while
        # split's backward is one concatenation.
        for z0, block in zip(range(0, nz, chunk), V.split(chunk, dim=1)):
            if per_slice is not None:
                block = per_slice(block)
            term = (fft2(block) * F[None, z0 : z0 + chunk]).sum(dim=1)
            acc = term if acc is None else acc + term
        assert acc is not None, "V must have at least one z-slice"
        return acc

    def rytov(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute exit wave using Rytov approximation.

        Faster than multislice but less accurate for thick specimens.

        Parameters
        ----------
        V : torch.Tensor
            3D potential volume with shape (B, Z, Y, X). Real, in which
            case the amplitude-contrast factor ``self.alpha`` is applied
            (as a scalar on the slice sum, the model being linear in V);
            or already complex (absorptive), used as given.

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
        approximation for small V. The sum over slices is taken in Fourier
        space and inverse-transformed once (the inverse FFT is linear, so it
        commutes with the sum), a z-chunk at a time; see
        :meth:`_fourier_slice_sum`.
        """
        scattered_k = self._fourier_slice_sum(V, None)
        exitwave = torch.exp(ifft2(self._phase_scale(V) * scattered_k))
        return exitwave  # (B x X x Y)

    def _phase_scale(self, V: torch.Tensor) -> complex:
        """See :func:`phase_scale`."""
        return phase_scale(V, self.sigma, self.pixel_size, self.alpha)

    def firstborn(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute exit wave using first Born approximation.

        Parameters
        ----------
        V : torch.Tensor
            3D potential volume with shape (B, Z, Y, X), real (the
            amplitude-contrast factor is applied here) or complex.

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D scattered wave with shape (B, Y, X).

        Notes
        -----
        The first Born approximation computes the scattered wave as:

        ψ = 1 + iσ Σ_z [F(z) ★ V(z)]

        Valid only for weak scatterers where multiple scattering is
        negligible. Same slice sum as :meth:`rytov`, with the exponent
        linearised.
        """
        scattered_k = self._fourier_slice_sum(V, None)
        return 1 + ifft2(self._phase_scale(V) * scattered_k)

    def kinematic(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute the exit wave in the kinematic (single-scattering) approximation.

        Parameters
        ----------
        V : torch.Tensor
            3D potential volume with shape (B, Z, Y, X), real (the
            amplitude-contrast factor is applied here) or complex.

        Returns
        -------
        exitwave : torch.Tensor
            Complex-valued 2D exit wave with shape (B, Y, X).

        Notes
        -----
        ψ = 1 + Σ_z F⁻¹{ F[exp(iσΔz V_z) − 1] · F_z }

        Each slice scatters once and propagates to the exit plane. Unlike
        ``firstborn``, the per-slice amplitude ``exp(iσΔz V_z) − 1`` is kept
        exact rather than linearised, so it stays valid for stronger
        potentials as long as multiple scattering between slices is
        negligible.
        """
        factor = self._phase_scale(V)

        def transmit(block: torch.Tensor) -> torch.Tensor:
            return torch.exp(factor * block) - 1

        return 1 + ifft2(self._fourier_slice_sum(V, transmit))

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
            # amplitude contrast folded into the slice sum's scalar (the
            # model is linear in V), not materialised as a complex volume.
            return self.rytov(V)
        elif self.scattering_model == "projection":
            V = apply_amplitude_contrast(V, alpha=self.alpha)
            return self.projection(V)
        elif self.scattering_model == "firstborn":
            return self.firstborn(V)
        elif self.scattering_model == "kinematic":
            return self.kinematic(V)
        elif self.scattering_model == "ctf":
            return self.ctf(V)
        else:
            raise ValueError(f"Unknown scattering_model: {self.scattering_model!r}")
