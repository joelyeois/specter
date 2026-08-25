from __future__ import annotations

from typing import Any, Literal, Sequence

import roma
import torch
import torch.nn.functional as F
from ..progress import status, track

from .. import rotations
from .. import tilt as tilt_geometry
from ..ice import IceBank, RandomIcemaker, blend_ice_into_volume, resolve_icemaker
from ._micrograph import MicrographGenerator
from ..scattering import IterativeScattering


class TiltSeriesGenerator(MicrographGenerator):
    """
    Generates tilt series images by tilting the specimen volume.

    Accepts either a list of tilt angles or explicit quaternions/translations.
    The volume is rotated for each tilt via an affine transformation passed to
    ``IterativeScattering``.

    Parameters
    ----------
    volume : torch.Tensor
        Pre-assembled specimen volume of shape (1, Z, Y, X) -- e.g. the
        output of
        :func:`~specter.pipelines.build_tomogram_generator`/`specter build
        tomogram`. If ``ice_model``
        or ``icemaker`` is given, ice is blended into ``volume`` (matching its
        own size and voxel size, masked to voxels with little existing
        scattering potential -- see ``ice_model`` below) before any of this
        class's own tilt-coverage/taper padding is applied, so that padding
        still sees, and extends, the ice-filled volume. Unlike this class's
        other tensors, ``volume`` is never moved by ``.to(device)`` -- it stays
        wherever it started (typically CPU) until the first
        :meth:`generate_tilt_series` call, which tries moving it to the
        compute device and transparently falls back to leaving it on CPU
        (streaming small, geometry-bounded blocks per Z-chunk instead of
        holding the whole volume in GPU memory) if it doesn't fit. This is
        automatic -- there's no flag to set for a volume too large for the
        GPU, and no penalty for one that comfortably fits.
    micrograph_size : int or tuple[int, int]
        Output image size in pixels (must be square).
    pixel_size : float
        Pixel size in Å.
    ctf_params : dict[str, torch.Tensor]
        CTF parameters; each value is a 1-D tensor of length n.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per tilt image in e⁻/Å². Scalar, or a
        1-D tensor of length n giving a separate dose for each tilt.
    quaternions : torch.Tensor, optional
        Explicit rotation quaternions of shape (N_tilts, 4). Mutually
        exclusive with ``angles``.
    translations : torch.Tensor, optional
        Per-tilt XY translations in Å, shape (N_tilts, 2), ordered [tx, ty]
        (matching ``rlnOriginXAngst`` / ``rlnOriginYAngst``). Works with both
        ``quaternions`` and ``angles``; defaults to zero shifts.
    angles : torch.Tensor or sequence of float, optional
        Tilt angles in degrees. Mutually exclusive with ``quaternions``.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices, shape (n, 2, 2).
    ice_model : str, optional
        Ice generation algorithm used to blend ice into ``volume`` (see
        ``volume`` above): ``'gd'`` (samples from the pre-generated
        :class:`~specter.ice.IceBank` cache) or ``'random'`` (instant,
        cheap :class:`~specter.ice.RandomIcemaker` placement). ``None``
        (default) or ``'none'`` adds no ice. Ignored when ``icemaker`` is
        provided.
    ice_cache_dir : str, optional
        Directory of cached ice configs for ``ice_model='gd'`` (see
        :func:`specter.ice.build_ice_cache`). Defaults to the bundled
        ``ice_data/ice_cache``. Ignored for other ``ice_model`` values or
        when ``icemaker`` is provided.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to blend into ``volume`` directly. When
        supplied, ``ice_model`` and ``ice_cache_dir`` are both ignored.
    ice_relax_steps : int, optional
        Forwarded to :meth:`~specter.ice.IceBank.generate_big_ice` when
        ``ice_model='gd'`` (or an ``IceBank`` ``icemaker``): number of local
        MLBOP relaxation steps used to heal tile seams. Default 0 (no
        relaxation, matching ``IceBank.generate_big_ice``'s own default) --
        tilt series volumes are typically large/tiled often enough that
        seam relaxation cost adds up, and the un-relaxed seams have not been
        a problem in practice for this class's usage. Set higher for
        production-quality seams. Ignored for ``RandomIcemaker``.
    scattering_model : str, optional
        Scattering model passed to ``IterativeScattering``. Default 'multislice'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal-space frequency limit.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.0.
    pad_fft : bool, optional
        Give multislice's per-slice FFT-based Fresnel propagation extra canvas
        headroom for the whole tilt recursion, rather than propagating at exactly
        the output size with zero headroom. At zero headroom, each step's circular
        convolution wraps slightly at the same fixed frame boundary every step;
        under tilt (hundreds to 1000+ multislice steps), that small per-step
        wraparound compounds coherently into a real, visible artifact along all
        four frame edges. Confirmed fixed by padding once, running the *entire*
        recursion at the padded size, and cropping back to the true output size
        only once at the end (validated against ``scattering_model="projection"``,
        which cannot exhibit this artifact by construction, at both toy and
        production scale -- nz=368, ~1000+ multislice steps, see
        ``dev/tilt series/verify_recursion_pad*.py``). Has no effect on
        ``scattering_model`` other than ``"multislice"``. Default False -- this is
        a real, confirmed fix, but flipping the *default* on requires re-validating
        against the full real-data survey, which has not been redone with this
        implementation.
    fft_pad_margin : int, optional
        Padding added on each side of the propagation canvas when ``pad_fft=True``.
        Always zero-filled (a tilted slice's flanks are real vacuum, not continuing
        ice, so reflecting would fabricate density that isn't physically present).
        Validated at 16-32px -- identical result across that whole range (already
        converged), independent of volume size or multislice step count. Default 16.
    chunk_size : int, optional
        Chunk size for ``MicrographSpecimenGenerator`` (unused here but passed to parent).
    detector_model : str, optional
        Detector MTF model ('k3_300kv', 'k3_200kv', 'perfect', None).
    progressbars : bool, optional
        Show progress bars. Default True.
    verbose : bool, optional
        Emit debug-level log messages. Default True.
    slice_batchsize : int, optional
        Number of Z slices propagated together. Default 1.
    pad_volume : bool, optional
        Automatically pad volume in XY when it is too small for the requested
        tilt coverage (reflect padding). Default True.
    edge_margin : int, optional
        Extra reflect-padded pixels on each XY side, added on top of the exact
        geometrically-required tilt coverage. The geometric minimum computed by
        ``_estimate_required_nxy`` leaves *zero* slack: output pixels at the edge of
        the crop have a per-slice sampling footprint that, at the deepest Z, lands
        exactly on the padded-volume boundary, with no room for interpolation. That
        produces a real, tilt-axis-aligned artifact -- visible in the image as
        banding at the two edges perpendicular to the tilt axis, and in its Fourier
        transform as a bright line through the origin along the tilt-axis direction
        (confirmed by rotating ``tilt_axis`` and watching the line rotate with it).
        A small default margin removes that slack deficit; empirically, ~8px cut the
        radial-power-spectrum shape-correlation gap between 0deg and a 45deg tilt
        roughly in half on a 192px/92-slice test volume (see dev/tilt series/ for the
        sweep). This is intentionally independent of ``taper_width``: tapering the
        *extra* margin beyond the geometric minimum has no effect on the output
        (those pixels are provably never sampled), so it cannot substitute for this.
        Default 8.
    taper_width : int, optional
        Extra reflect-padded apron pixels on each XY side beyond tilt-coverage
        padding (already inclusive of ``edge_margin``), with a cosine taper applied.
        Note: since this fades pixels strictly beyond the geometrically-required
        coverage (now ``edge_margin``-inflated), those pixels are never read by the
        interpolation that produces the output -- this taper has no measurable
        effect on the result and exists only to avoid a literal hard edge at the
        outermost boundary of the padded array itself. Default 0.
    z_taper_width : int, optional
        Cosine taper width in Z pixels at the top and bottom of the volume. Applied
        after ``z_edge_margin``'s zero-padding (if any), so a nonzero value here
        smooths the transition into that new zero region rather than merely dimming
        the sample's own edge slices. Default 0.
    z_edge_margin : int, optional
        Zero-pads this many pixels onto each Z edge before any tapering. Unlike XY (a
        cryo-ET sample continues laterally past any crop -- see ``edge_margin``), Z
        really is bounded by vacuum above/below the ice, so zero is the physically
        correct fill here -- but a hard step to zero is still a discontinuity that
        produces the same kind of tilt-induced Fourier artifact XY does (tilting mixes
        X and Z), so pair this with ``z_taper_width`` to smooth the transition rather
        than leaving a cliff. Matters most when ``pad_fft=True`` inflates the working
        canvas well beyond what ``edge_margin`` alone covers. Default 0 (off).
    tilt_axis : str, optional
        Axis around which the sample tilts ('x' or 'y'). Default 'x'.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Default 0.0.
    n_frames : int, optional
        Number of detector frames to simulate. Default None.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    convergence_angle : float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. Default None (envelope disabled).
    cc : float, optional
        Chromatic aberration coefficient in Å, used for the Cc
        (temporal coherence) envelope. Default None (envelope disabled).
    energy_spread : float, optional
        FWHM of the beam energy spread in eV, used by the Cc envelope.
        Default 0.7.
    deltaV_V : float, optional
        Relative high-voltage instability, used by the Cc envelope.
        Default 0.06e-6.
    deltaI_I : float, optional
        Relative objective-lens current instability, used by the Cc
        envelope. Default 0.01e-6.
    dose_envelope : bool, optional
        Whether to apply the Grant & Grigorieff (2015) cumulative-dose
        envelope, using ``dose_per_angstrom``. Default False.
    """

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        volume: torch.Tensor,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any],
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        angles: torch.Tensor | Sequence[float] | None = None,
        anisomag: torch.Tensor | None = None,
        ice_model: str | None = None,
        ice_cache_dir: str | None = None,
        icemaker: IceBank | RandomIcemaker | None = None,
        ice_relax_steps: int = 0,
        scattering_model: str = "multislice",
        noise_model: str | None = "poisson",
        klim: float | None = None,
        alpha: float = 0.0,
        pad_fft: bool = False,
        fft_pad_margin: int = 16,
        chunk_size: int = 1,
        detector_model: str | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        slice_batchsize: int = 1,
        pad_volume: bool = True,
        edge_margin: int = 8,
        z_edge_margin: int = 0,
        taper_width: int = 0,
        z_taper_width: int = 0,
        tilt_axis: str = "x",
        coincidence_radius: float | torch.Tensor = 0.0,
        n_frames: int | None = None,
        bfactor: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        aberration_backend: Literal["legacy", "torch_ctf"] = "legacy",
        lpp_params: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        if volume is None:
            raise ValueError("'volume' must be provided for TiltSeriesGenerator.")

        volume_icemaker = resolve_icemaker(
            ice_model,
            pixel_size,
            nxy=volume.shape[-1],
            nz=volume.shape[-3],
            ice_cache_dir=ice_cache_dir,
            icemaker=icemaker,
        )
        if volume_icemaker is not None:
            # Blend ice into the raw input volume before any of this class's own
            # tilt-coverage/z-vacuum/taper padding below, so that padding still
            # operates on (and, for the reflect-padded XY margin, naturally
            # extends) the ice-filled volume, and the z_edge_margin/taper
            # regions -- which are meant to be real vacuum, not ice -- stay
            # untouched.
            if verbose:
                print(
                    f"[TiltSeriesGenerator] Adding ice to volume using {ice_model} model"
                )
            with torch.no_grad(), status("Tiling ice volume", disable=not progressbars):
                volume = blend_ice_into_volume(
                    volume, volume_icemaker, pixel_size, relax_steps=ice_relax_steps
                )

        if isinstance(micrograph_size, int):
            desired_nxy = micrograph_size
        elif (
            isinstance(micrograph_size, (tuple, list))
            and len(micrograph_size) == 2
            and micrograph_size[0] == micrograph_size[1]
        ):
            desired_nxy = micrograph_size[0]
        else:
            raise ValueError("micrograph_size must have same dimensions in x and y.")

        self.tilt_axis = tilt_axis.lower()
        if self.tilt_axis not in ["x", "y"]:
            raise ValueError(f"Unsupported tilt_axis: {tilt_axis}. Use 'x' or 'y'.")

        max_tilt_angle_deg = tilt_geometry.infer_max_tilt_from_inputs(
            angles=angles, quaternions=quaternions
        )

        nz_input = int(volume.shape[-3])
        available_nxy = int(min(volume.shape[-2], volume.shape[-1]))
        required_nxy = tilt_geometry.estimate_required_nxy(
            desired_nxy=desired_nxy,
            nz=nz_input,
            max_tilt_angle_deg=max_tilt_angle_deg,
        )
        # required_nxy is the exact geometric minimum -- zero interpolation slack for
        # crop-edge output pixels (see edge_margin's docstring). Inflate it before
        # taper_width (a separate, purely cosmetic apron) gets added on top.
        required_nxy_padded = required_nxy + 2 * int(edge_margin)
        target_nxy = required_nxy_padded + 2 * taper_width
        self.recommended_nxy_for_max_tilt = required_nxy
        self.edge_margin = int(edge_margin)
        self.max_tilt_angle_deg = float(max_tilt_angle_deg)
        self.max_allowed_tilt_deg_for_volume = (
            tilt_geometry.estimate_max_allowed_tilt_deg(
                desired_nxy=desired_nxy, nz=nz_input, available_nxy=available_nxy
            )
        )
        self.max_allowed_nxy = tilt_geometry.estimate_max_allowed_nxy(
            available_nxy=available_nxy,
            nz=nz_input,
            max_tilt_angle_deg=max_tilt_angle_deg,
        )

        if available_nxy < target_nxy:
            if pad_volume:
                volume = tilt_geometry.pad_volume_xy_for_tilt(
                    volume, target_nxy, available_nxy
                )
                msg = (
                    "[TiltSeriesGenerator] Volume XY too small for requested tilt coverage"
                    + (" and taper" if taper_width > 0 else "")
                    + f"; padded (reflect) from {available_nxy} to {volume.shape[-1]} px in XY.\n"
                    f"  micrograph_size={desired_nxy}, requested_max_tilt={self.max_tilt_angle_deg:.2f} deg, "
                    f"required_volume_nxy>={required_nxy}, edge_margin={edge_margin} "
                    f"(-> required_nxy_padded>={required_nxy_padded})"
                )
                if taper_width > 0:
                    msg += f", target_nxy (with taper)>={target_nxy}"
                print(msg + ".")
            else:
                print(
                    "[TiltSeriesGenerator] Input volume XY may be too small for requested tilt "
                    "coverage; proceeding anyway (pad_volume=False).\n"
                    f"  micrograph_size={desired_nxy}, volume_shape={tuple(volume.shape)}, "
                    f"requested_max_tilt={self.max_tilt_angle_deg:.2f} deg,\n"
                    f"  required_volume_nxy>={required_nxy}, current_volume_nxy={available_nxy}, \n"
                    f"  max_allowed_tilt_with_current_volume\u2248{self.max_allowed_tilt_deg_for_volume:.2f} deg,\n"
                    f"  max_allowed_nxy\u2248{self.max_allowed_nxy}."
                )

        if z_edge_margin > 0:
            # Unlike XY (where the sample genuinely continues past our crop -- reflect is
            # correct) or edge_margin (a small interpolation-slack margin), Z really is
            # bounded by vacuum above/below the ice: zero-fill is the *physically correct*
            # answer here, not an approximation. But "correct value" and "safe to use
            # abruptly" are different things -- multislice's FFT-based propagation doesn't
            # care whether a discontinuity is physically justified, only that it's a
            # discontinuity. A hard step from real ice/protein density to exactly zero,
            # right at the true Z boundary, still produces the same class of artifact once
            # the sample is tilted (rotation mixes X and Z, so this Z-edge shows up in the
            # tilted image just like the XY boundary did). So: extend with real zeros
            # (correct), then taper across the new seam (z_taper_width, right below) so the
            # transition is gradual rather than a cliff -- mirroring how a real ice/vacuum
            # interface isn't a mathematical step function either.
            volume = F.pad(
                volume,
                (0, 0, 0, 0, z_edge_margin, z_edge_margin),
                mode="constant",
                value=0.0,
            )
            print(
                f"[TiltSeriesGenerator] Zero-padded {z_edge_margin} px at each Z edge "
                f"(physically correct vacuum above/below the sample)."
            )

        if taper_width > 0 or z_taper_width > 0:
            volume = tilt_geometry.apply_volume_cosine_taper(
                volume, taper_xy=int(taper_width), taper_z=int(z_taper_width)
            )
            if taper_width > 0:
                print(
                    f"[TiltSeriesGenerator] Applied cosine-taper over {taper_width} px "
                    f"at the XY edges."
                )
            if z_taper_width > 0:
                print(
                    f"[TiltSeriesGenerator] Applied cosine-taper over {z_taper_width} px "
                    f"at the Z edges (top/bottom)."
                )

        super().__init__(
            scattering_potential=None,
            micrograph_size=micrograph_size,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            volume=volume,
            anisomag=anisomag,
            scattering_model=scattering_model,
            noise_model=noise_model,
            klim=klim,
            alpha=alpha,
            # NOT pad_fft: MicrographGenerator/BaseImager's own pad_fft mechanism
            # (pad_nxy -> whole-volume XY padding, aberration built at pad_nxy) is
            # specific to MicrographGenerator.forward(), which TiltSeriesGenerator
            # never calls (generate_tilt_series uses self.iterative_scattering
            # directly). Forwarding pad_fft here would inflate self.pad_nxy and build
            # self.aberration at that size, mismatching the exitwave that
            # self.iterative_scattering now always returns at self.nxy. This class's
            # own pad_fft controls IterativeScattering's internal multislice-canvas
            # padding only (see below), entirely independent of the parent's.
            pad_fft=False,
            chunk_size=chunk_size,
            detector_model=detector_model,
            progressbars=progressbars,
            verbose=verbose,
            slice_batchsize=slice_batchsize,
            coincidence_radius=coincidence_radius,
            n_frames=n_frames,
            bfactor=bfactor,
            convergence_angle=convergence_angle,
            cc=cc,
            energy_spread=energy_spread,
            deltaV_V=deltaV_V,
            deltaI_I=deltaI_I,
            dose_envelope=dose_envelope,
            aberration_backend=aberration_backend,
            lpp_params=lpp_params,
            **kwargs,
        )
        # MicrographGenerator.__init__ (just above) registered self.volume as a
        # buffer, which would otherwise be dragged onto the compute device by
        # any later `.to(device)` call on this module (e.g. the CLI pipelines'
        # `TiltSeriesGenerator(...).to(device_target)` pattern) -- forcing the
        # whole volume into GPU memory unconditionally, whether or not it fits.
        # Un-registering it here means it simply stays wherever it already was
        # (typically CPU, e.g. straight from `torch.load`/`mrcfile`) until
        # `generate_tilt_series` below decides what to do with it: try moving
        # it to the compute device (fast -- IterativeScattering's rotated-slice
        # fetch runs directly against it), falling back to leaving it on CPU
        # and streaming small windowed blocks per Z-chunk instead (slower per
        # step, but with a GPU memory footprint set by the query geometry, not
        # by the volume's size -- see dev/tilt series/windowed_streaming_*.py)
        # if it doesn't fit. No flag needed: this is automatic and safe at any
        # volume size, so there's nothing for a caller to get wrong.
        volume_value = self.volume
        del self._buffers["volume"]
        self.volume = volume_value

        self.slice_batchsize = slice_batchsize
        # pad_fft=True (multislice only) gives the per-slice FFT-based Fresnel
        # propagation extra canvas headroom for the *entire* nz_new-step recursion,
        # padding once before the loop and cropping back to self.nxy once at the end
        # -- see IterativeScattering.multislice for the mechanism. Validated this
        # gives an artifact-free result matching scattering_model="projection" (which
        # cannot exhibit this artifact by construction) at both toy and production
        # scale (nz=368, ~1000+ multislice steps, see dev/tilt series/). Always uses
        # zeros for the padded region (not reflection): a tilted slice's flanks are
        # real vacuum there, not continuing ice, so reflecting would fabricate density
        # that isn't physically present.
        #
        # Unlike the old implementation, this does not inflate self.nxy/the volume-ROI
        # sampling size at all -- self.iterative_scattering.nxy is always the true
        # output size, so the exitwave it returns is already self.nxy-sized whenever
        # pad_fft=True, and self.aberration (built at self.nxy by the parent's
        # _init_optics()) never needs to be rebuilt.
        self.iterative_scattering = IterativeScattering(
            self.nxy,
            pixel_size,
            voltage,
            scattering_model=scattering_model,
            klim=klim,
            alpha=alpha,
            progressbars=progressbars,
            roi_padding_mode="zeros",
            pad_fft=pad_fft,
            fft_pad_margin=fft_pad_margin,
        )

        if quaternions is not None:
            self.register_buffer("quaternions", torch.as_tensor(quaternions))
            self.register_buffer(
                "translations",
                torch.as_tensor(translations)
                if translations is not None
                else torch.zeros(len(quaternions), 2),
            )
            self.angles = None
        elif angles is not None:
            self.angles = torch.as_tensor(angles)
            B = len(self.angles)
            theta_rad = torch.deg2rad(self.angles)

            if self.tilt_axis == "x":
                rotvecs = torch.stack(
                    [
                        theta_rad,
                        torch.zeros_like(theta_rad),
                        torch.zeros_like(theta_rad),
                    ],
                    dim=-1,
                )
            else:  # 'y'
                rotvecs = torch.stack(
                    [
                        torch.zeros_like(theta_rad),
                        theta_rad,
                        torch.zeros_like(theta_rad),
                    ],
                    dim=-1,
                )

            quats = roma.rotvec_to_unitquat(rotvecs)
            self.register_buffer("quaternions", quats)
            self.register_buffer(
                "translations",
                torch.as_tensor(translations)
                if translations is not None
                else torch.zeros(B, 2),
            )
        else:
            raise ValueError("Either 'angles' or 'quaternions' must be provided.")

    # ------------------------------------------------------------------ #
    # Forward methods                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_volume_placed(self) -> None:
        """
        Try moving ``self.volume`` onto the compute device once; fall back to
        leaving it on CPU (streaming small windowed blocks per Z-chunk via
        ``VolumeRotator.sample_rotated_slices``' ``device`` param instead --
        see this class's docstring for ``volume``) if it doesn't fit. A no-op on
        every call after the first, once ``self.volume`` is settled on some
        device.
        """
        if self.volume.device == self.device:
            return
        try:
            self.volume = self.volume.to(self.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if self.verbose:
                gb = self.volume.numel() * self.volume.element_size() / 1e9
                print(
                    f"[TiltSeriesGenerator] volume ({gb:.1f} GB) does not fit on "
                    f"{self.device}; keeping it on CPU and streaming windowed "
                    "per-chunk fetches instead (slower per step, bounded GPU "
                    "memory regardless of volume size)."
                )

    def generate_tilt_series(
        self, idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a complete tilt series for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices (selects CTF/anisomag parameters).

        Returns
        -------
        tilt_series : torch.Tensor
            Detected images, shape (B, N_tilts, Y, X).
        exitwaves : torch.Tensor
            Exit waves, shape (B, N_tilts, Y, X).
        clean_images : torch.Tensor
            ``|detector_waves|²`` before noise, shape (B, N_tilts, Y, X).
        """
        self._ensure_volume_placed()

        tilt_series = []
        exitwaves = []
        clean_images = []
        B = len(idx) if isinstance(idx, torch.Tensor) else 1
        n_tilts = len(self.quaternions)

        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1).to(self.volume.device)
        volume_scaled = self.volume * scale

        for i in track(
            range(n_tilts),
            description="Generating tilt series.",
            disable=not self.progressbars,
        ):
            Q = self.quaternions[i].unsqueeze(0).expand(B, -1)
            T = self.translations[i].unsqueeze(0).expand(B, -1)

            R_mat = roma.unitquat_to_rotmat(Q)
            if R_mat.ndim == 2:
                R_mat = R_mat.unsqueeze(0)
            T_torch = rotations.translations_angstrom_to_torch(
                T, self.volume.shape[-1], self.pixel_size
            )
            theta_matrix = rotations.build_affine_matrix(R_mat, T_torch)

            exitwave = self.iterative_scattering(
                volume_scaled, theta_matrix, slice_batchsize=self.slice_batchsize
            )

            ctf_batch = self._ctf_batch(idx)
            if self.scattering_model not in ["projection", "ctf"]:
                ctf_batch = tilt_geometry.shift_ctf_defocus_for_tilt(
                    ctf_batch,
                    tuple(self.volume.shape),
                    theta_matrix,
                    self.nz,
                    self.pixel_size,
                )

            detector_waves = self.aberration(exitwave, ctf_batch)

            dose_batch = self.dose_per_angstrom[idx]
            cr_batch = self.coincidence_radius[idx]
            if self.anisomag is None:
                image = self.detector(detector_waves, dose_batch, cr_batch, nxy=None)
            else:
                image = self.detector(
                    detector_waves, dose_batch, cr_batch, self.anisomag[idx], nxy=None
                )

            tilt_series.append(image.detach().cpu())
            exitwaves.append(exitwave.detach().cpu())
            clean_images.append(torch.abs(detector_waves.detach().cpu()) ** 2)

        return (
            torch.stack(tilt_series, dim=1),
            torch.stack(exitwaves, dim=1),
            torch.stack(clean_images, dim=1),
        )
