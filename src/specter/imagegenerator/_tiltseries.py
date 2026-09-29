"""
`TiltSeriesGenerator`: a cryo-ET tilt series from a specimen volume, one
multislice pass per tilt with the geometry padding each tilt needs.
"""

from __future__ import annotations

from specter import logger

from dataclasses import replace
from collections.abc import Callable
from typing import Any, Sequence

import roma
import torch
from ..progress import status, track

from .. import rotations
from .. import tilt as tilt_geometry
from ..ice import IceBank, RandomIcemaker, blend_ice_into_volume, resolve_icemaker
from ..potential import aperture_lowpass_isotropic
from ..settings import Camera, Envelopes, Ice, Optics, Propagation, TiltGeometry
from ._base import (
    RemovalMFPs,
    mfp_absorption_field,
    read_dose_weights,
    resolve_removal_mfps,
    solvent_exposure_filter,
)
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
        tomogram`. Its Z extent is taken to *be* the specimen's thickness, not
        a box the specimen sits somewhere inside: the defocus correction
        measures from the volume's midplane to its Z face
        (:func:`~specter.aberrations.defocus_midplane_shift`), so vacuum padded
        on in Z would be read as specimen and shift the simulated defocus by
        half its thickness. Pass the specimen at its true Z extent and let this
        class do any padding it needs. If ``ice_model``
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
    ctf_params : dict[str, torch.Tensor] or None
        Per-tilt CTF parameters; each value is a 1-D tensor of length n.
        Required unless ``optics`` is ``None``.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per tilt image in e⁻/Å². Scalar, or a
        1-D tensor of length n giving a separate dose for each tilt.
    propagation : Propagation, optional
        How the exit wave is computed. ``pad_fft`` here gives the multislice
        recursion FFT headroom of ``fft_pad_margin`` pixels: at zero headroom
        each step's circular convolution wraps slightly at the same frame
        boundary, and over hundreds of tilted steps that compounds into a
        visible artifact along all four edges (validated against
        ``scattering_model="projection"``, which cannot exhibit it, at toy
        and production scale). Default ``Propagation()``.
    optics : Optics, optional
        The aberration stage; ``None`` skips it. Default ``Optics()``.
    envelopes : Envelopes, optional
        Coherence and radiation-damage envelopes. With ``dose_envelope``,
        each tilt is a plain short exposure of ``dose_per_angstrom`` taken
        after the pre-exposure accumulated by the tilts before it, and the
        envelope is evaluated over exactly that interval (see
        :func:`specter.aberrations.dose_envelope`); tilts are assumed to be
        in acquisition order. Default ``Envelopes()``.
    camera : Camera, optional
        The detector chain. Default ``Camera()``.
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
    ice : Ice, optional
        The amorphous ice blended into ``volume`` at construction (see
        ``volume`` above). ``thickness`` is ignored: the volume's Z extent is
        the specimen's thickness. Default ``Ice()``, no ice.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to blend into ``volume`` directly. When
        supplied, ``ice.model`` and ``ice.cache_dir`` are ignored.
    absorption_potential : torch.Tensor, optional
        Explicit nonnegative imaginary potential in volts, with the same shape,
        dtype and device as ``volume`` before geometry padding. Compute it
        from the dry specimen and the solvent support, not from ice-filled
        density. It receives the same tilt padding and edge taper as the
        elastic volume and requires multislice with ``alpha=0``. An explicit
        field takes precedence: it supplies all absorption, the mean-free-path
        settings are not used to build or modify it, and ``potential_scale``
        scales only the elastic potential. With an objective aperture the
        field is expected to include the aperture loss, since the elastic
        volume is low-passed at the aperture regardless of where the field
        came from. Default None, which builds the field from the
        mean-free-path settings under
        ``Propagation(absorption_model="inelastic_mfp")`` (see Notes) and
        uses no field under ``"alpha"``.
    fft_pad_margin : int, optional
        Padding added on each side of the propagation canvas when
        ``propagation.pad_fft`` is set.
        Always zero-filled (a tilted slice's flanks are real vacuum, not continuing
        ice, so reflecting would fabricate density that isn't physically present).
        Validated at 16-32px -- identical result across that whole range (already
        converged), independent of volume size or multislice step count. Default 16.
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
        roughly in half on a 192px/92-slice test volume. This is intentionally
        independent of ``taper_width``: tapering the
        *extra* margin beyond the geometric minimum has no effect on the output
        (those pixels are provably never sampled), so it cannot substitute for this.
        Default 8.
    tilt : TiltGeometry, optional
        Tilt axis and the XY/Z cosine tapers applied to the volume before
        imaging. The XY taper fades pixels beyond the geometrically required
        coverage (already inclusive of ``edge_margin``), which the output
        never samples, so it only avoids a hard edge at the padded array's
        own boundary; the Z taper fades the sample's outermost slices, which
        tilting turns into an in-plane discontinuity the propagator sees.
        Default ``TiltGeometry()``.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Default 0.0.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.

    Notes
    -----
    Under ``Propagation(absorption_model="inelastic_mfp")`` absorption is
    modelled as in the particle generators. The imaginary potential is
    :func:`~specter.potential.inelastic_absorption_potential` of the DRY
    ``volume``, read before ice is blended in, with the same occupancy
    :func:`~specter.ice.blend_ice_into_volume` uses to weight the ice. Its
    rates are resolved by :func:`~specter.imagegenerator._base.resolve_removal_mfps`:
    the solvent takes ``inelastic_mfp_solvent`` or the ice value at this
    voltage (:func:`~specter.potential.ice_inelastic_mfp`), the specimen
    takes ``inelastic_mfp_specimen``, and an objective aperture adds its
    elastic loss to each as a rate. The solvent term is present only when
    ice is blended. The field is padded and tapered with the volume and
    propagated alongside it by
    :meth:`~specter.scattering.IterativeScattering.multislice_absorptive`,
    one slice at a time in the beam frame.

    Two cases differ from the particle path by construction. Without a
    specimen mean free path the field is uniform, but it is built as a
    constant over the volume's box rather than applied as a scalar: a tilted
    slice extends past the rotated slab into vacuum, where nothing may
    absorb, so the path length must come from the geometry. Without ice and
    without a specimen mean free path nothing absorbs and no field is built,
    as in the particle path. An objective aperture low-passes the elastic
    volume with :func:`~specter.potential.aperture_lowpass_isotropic`, a
    spherical filter in 3D frequency space that commutes with every tilt,
    where the particle path filters each beam-frame slice; the two keep the
    same scattering on the Ewald sphere.

    The field is a second real volume of the padded size, so it doubles the
    resident memory of the specimen. Both are placed on the compute device
    together, or both streamed from the host when they do not fit.

    ``Envelopes(dose_envelope_target="specimen")`` damages the specimen and
    not its ice, per tilt: tilt ``i`` is imaged with the dry specimen under
    the envelope of its own dose after the pre-exposure of the tilts before
    it, plus the ice, which is weighted by the undamaged specimen's
    occupancy. The padded dry volume is kept as its 3D spectrum
    (:class:`~specter.fft.RadialSpectrum`), so each tilt's damaged volume
    costs one inverse transform; resident memory is four volumes (the
    volume, the spectrum, its inverse-transform scratch and the ice), all
    placed on the device together or streamed from the host together.
    ``Ice(motion_variance=...)`` filters the ice once, at the dose of one
    tilt, before it is blended: the coherence between two frames depends
    only on the dose between them, so every tilt keeps the same fraction of
    the ice structure. That requires the same dose on every tilt.
    """

    _dose_weighted = False  # a tilt is a plain sum, not an exposure-filtered one
    _defers_dose_series = True  # built from the dry volume, below

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        volume: torch.Tensor,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any] | None,
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        angles: torch.Tensor | Sequence[float] | None = None,
        anisomag: torch.Tensor | None = None,
        propagation: Propagation = Propagation(),
        optics: Optics | None = Optics(),
        envelopes: Envelopes = Envelopes(),
        camera: Camera = Camera(),
        ice: Ice = Ice(),
        icemaker: IceBank | RandomIcemaker | None = None,
        fft_pad_margin: int = 16,
        progressbars: bool = True,
        verbose: bool = True,
        slice_batchsize: int = 1,
        pad_volume: bool = True,
        edge_margin: int = 8,
        tilt: TiltGeometry = TiltGeometry(),
        coincidence_radius: float | torch.Tensor = 0.0,
        bfactor: float | torch.Tensor | None = None,
        absorption_potential: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if volume is None:
            raise ValueError("'volume' must be provided for TiltSeriesGenerator.")

        mfp_absorption = propagation.absorption_model == "inelastic_mfp"
        objective_aperture = None if optics is None else optics.objective_aperture
        if objective_aperture is not None and not mfp_absorption:
            raise ValueError(
                "Optics(objective_aperture=...) requires "
                "Propagation(absorption_model='inelastic_mfp'): under 'alpha' the "
                "fitted amplitude contrast already stands in for aperture loss, "
                "and applying both would count it twice."
            )
        if (mfp_absorption or absorption_potential is not None) and (
            propagation.scattering_model != "multislice" or propagation.alpha != 0
        ):
            raise ValueError(
                "TiltSeriesGenerator's absorption field (absorption_model="
                "'inelastic_mfp' or an explicit absorption_potential) requires "
                "multislice and alpha=0"
            )

        if absorption_potential is not None:
            if (
                absorption_potential.shape != volume.shape
                or absorption_potential.dtype != volume.dtype
                or absorption_potential.device != volume.device
                or not absorption_potential.is_floating_point()
                or not torch.isfinite(absorption_potential).all()
                or (absorption_potential < 0).any()
            ):
                raise ValueError(
                    "absorption_potential must be finite, nonnegative, real and "
                    "match volume shape, dtype and device"
                )

        self.ice = ice
        volume_icemaker = resolve_icemaker(
            ice.model,
            pixel_size,
            nxy=volume.shape[-1],
            nz=volume.shape[-3],
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=ice.parameterization,
        )
        # Resolved before the parent's constructor, which is handed "alpha"
        # below: the field must be read off the DRY specimen, before ice is
        # blended into it.
        removal_mfps = resolve_removal_mfps(propagation, optics, voltage)
        if absorption_potential is None and removal_mfps is not None:
            with torch.no_grad():
                absorption_potential = self._mfp_absorption_field(
                    volume,
                    pixel_size,
                    removal_mfps,
                    has_solvent=volume_icemaker is not None,
                )
        damages_specimen = (
            envelopes.dose_envelope and envelopes.dose_envelope_target == "specimen"
        )
        if ice.motion_variance is not None and (
            envelopes.dose_envelope and not damages_specimen
        ):
            raise ValueError(
                "Ice(motion_variance=...) with the dose envelope on the "
                "transfer function would take the solvent's structure away "
                "twice: the envelope fades the water ring, and the exposure "
                "filter decorrelates it. Set "
                "Envelopes(dose_envelope_target='specimen')."
            )
        ice_filter = self._tilt_solvent_filter(
            ice, camera, pixel_size, dose_per_angstrom
        )
        # Kept for the damage series: the dose envelope on the specimen acts
        # on the dry specimen, never on the ice blended into it.
        dry = volume if damages_specimen else None
        volume = self._blend_ice(
            volume,
            ice,
            volume_icemaker,
            pixel_size,
            verbose,
            progressbars,
            ice_filter=ice_filter,
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

        self.tilt = tilt
        self.tilt_axis = tilt.tilt_axis
        taper_width = tilt.taper_width
        z_taper_width = tilt.z_taper_width

        volume = self._fit_volume_to_tilt(
            volume,
            desired_nxy,
            angles,
            quaternions,
            edge_margin,
            taper_width,
            z_taper_width,
            pad_volume,
        )
        if dry is not None:
            dry = self._fit_volume_to_tilt(
                dry,
                desired_nxy,
                angles,
                quaternions,
                edge_margin,
                taper_width,
                z_taper_width,
                pad_volume,
            )
        if absorption_potential is not None:
            absorption_potential = self._fit_volume_to_tilt(
                absorption_potential,
                desired_nxy,
                angles,
                quaternions,
                edge_margin,
                taper_width,
                z_taper_width,
                pad_volume,
            )
        # Scattering beyond the aperture is charged as absorption
        # (`RemovalMFPs.removal`), so the share of it the grid carries is
        # filtered out of the elastic potential, as the particle path does.
        # The filter is isotropic in 3D rather than per beam-frame slice, so
        # that it commutes with every tilt (see aperture_lowpass_isotropic).
        if objective_aperture is not None:
            with torch.no_grad():
                volume = aperture_lowpass_isotropic(
                    volume, pixel_size, objective_aperture, voltage
                )
                if dry is not None:
                    dry = aperture_lowpass_isotropic(
                        dry, pixel_size, objective_aperture, voltage
                    )

        super().__init__(
            specimen=volume,
            micrograph_size=micrograph_size,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            anisomag=anisomag,
            # NOT pad_fft: MicrographGenerator/BaseImager's own pad_fft mechanism
            # (pad_nxy -> whole-volume XY padding, aberration built at pad_nxy) is
            # specific to MicrographGenerator.forward(), which TiltSeriesGenerator
            # never calls (generate_tilt_series uses self.iterative_scattering
            # directly). Forwarding pad_fft here would inflate self.pad_nxy and build
            # self.aberration at that size, mismatching the exitwave that
            # self.iterative_scattering always returns at self.nxy. This class's
            # own pad_fft controls IterativeScattering's internal multislice-canvas
            # padding only (see below), entirely independent of the parent's.
            # The parent cannot construct material fields and rejects the MFP
            # model; this class builds the field above and propagates it
            # itself. The aperture is stripped as well, because the
            # parent accepts one only under the MFP model; both are restored
            # below.
            propagation=replace(propagation, pad_fft=False, absorption_model="alpha"),
            optics=(
                replace(optics, objective_aperture=None)
                if optics is not None and objective_aperture is not None
                else optics
            ),
            envelopes=envelopes,
            camera=camera,
            progressbars=progressbars,
            verbose=verbose,
            slice_batchsize=slice_batchsize,
            coincidence_radius=coincidence_radius,
            bfactor=bfactor,
            **kwargs,
        )
        self.propagation = propagation
        self.optics = optics
        self.absorption_model = propagation.absorption_model
        self.objective_aperture = objective_aperture
        self._removal_mfps = removal_mfps
        # Like volume, keep this outside registered buffers for bounded-memory
        # paired placement rather than unconditional Module.to() uploads.
        self.absorption_potential = absorption_potential
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
        # by the volume's size) if it doesn't fit. No flag needed: this is
        # automatic and safe at any
        # volume size, so there's nothing for a caller to get wrong.
        volume_value = self.volume
        del self._buffers["volume"]
        self.volume = volume_value
        if dry is not None:
            # Rendered into for every tilt, so it must be this class's own
            # canvas and not the caller's tensor.
            if self.volume is dry or self.volume.data_ptr() == dry.data_ptr():
                self.volume = self.volume.clone()
            with torch.no_grad():
                self._init_dose_series(dry, self.volume if volume_icemaker else dry)
            del dry

        self.slice_batchsize = slice_batchsize
        # pad_fft=True (multislice only) gives the per-slice FFT-based Fresnel
        # propagation extra canvas headroom for the *entire* nz_new-step recursion,
        # padding once before the loop and cropping back to self.nxy once at the end
        # -- see IterativeScattering.multislice for the mechanism. Validated this
        # gives an artifact-free result matching scattering_model="projection" (which
        # cannot exhibit this artifact by construction) at both toy and production
        # scale (nz=368, ~1000+ multislice steps). Always uses zeros for the padded
        # region (not reflection): a tilted slice's flanks are real vacuum there,
        # not continuing ice, so reflecting would fabricate density that isn't
        # physically present.
        #
        # The padding is internal to multislice: self.iterative_scattering.nxy is
        # always the true output size, so the exitwave it returns is already
        # self.nxy-sized whenever pad_fft=True, and self.aberration (built at
        # self.nxy by the parent's _init_optics()) never needs to be rebuilt.
        self.iterative_scattering = IterativeScattering(
            self.nxy,
            pixel_size,
            voltage,
            scattering_model=self.scattering_model,
            klim=self.klim,
            alpha=self.alpha,
            progressbars=progressbars,
            roi_padding_mode="zeros",
            pad_fft=propagation.pad_fft,
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

        # Exposure already delivered when each tilt starts: the summed dose of
        # the tilts before it, in index (= acquisition) order. Read by
        # _ctf_batch into ctf_params["pre_exposure"] for the dose envelope.
        n_tilts = len(self.quaternions)
        per_tilt = self.dose_per_angstrom.flatten().expand(n_tilts).to(torch.float32)
        self.register_buffer(
            "pre_exposure", torch.cumsum(per_tilt, 0) - per_tilt, persistent=False
        )

    # ------------------------------------------------------------------ #
    # Forward methods                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tilt_solvent_filter(
        ice: Ice,
        camera: Camera,
        pixel_size: float,
        dose_per_angstrom: float | torch.Tensor,
    ) -> Callable[[torch.Tensor], None] | None:
        """
        The solvent-exposure filter for the tilts' ice, or None.

        Each tilt is its own short exposure, a sum of its own frames, and the
        ice's coherence between two frames depends only on the dose between
        them, not on what came before. Every tilt therefore keeps the same
        fraction of its ice structure, the one set by the dose of one tilt,
        and a single filtered ice serves the series. That requires every tilt
        to receive the same dose.
        """
        if ice.motion_variance is None:
            return None
        doses = torch.as_tensor(dose_per_angstrom, dtype=torch.float32).flatten()
        if not torch.allclose(doses, doses[0].expand_as(doses)):
            raise ValueError(
                "Ice(motion_variance=...) on a tilt series needs the same dose on "
                "every tilt: one filtered ice volume serves the whole series"
            )
        weights, max_frequency = read_dose_weights(camera)
        return solvent_exposure_filter(
            ice, camera, pixel_size, float(doses[0]), weights, max_frequency
        )

    @staticmethod
    def _blend_ice(
        volume: torch.Tensor,
        ice: Ice,
        icemaker: IceBank | RandomIcemaker | None,
        pixel_size: float,
        verbose: bool,
        progressbars: bool,
        ice_filter: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """
        Blend ice into the raw input volume before any tilt-coverage or
        taper padding, so the padding operates on (and, for the
        reflect-padded XY margin, extends) the ice-filled volume.

        ``icemaker`` is the one :func:`~specter.ice.resolve_icemaker` returned
        for this volume, resolved by the caller because whether there is ice
        also decides whether the absorption field has a solvent term.
        """
        if icemaker is not None:
            if verbose:
                logger.info(f"Adding ice to volume using {ice.model} model")
            with torch.no_grad(), status("Tiling ice volume", disable=not progressbars):
                volume = blend_ice_into_volume(
                    volume,
                    icemaker,
                    pixel_size,
                    relax_steps=ice.relax_steps,
                    ice_filter=ice_filter,
                )

        return volume

    @staticmethod
    def _mfp_absorption_field(
        dry: torch.Tensor,
        pixel_size: float,
        removal_mfps: RemovalMFPs,
        has_solvent: bool,
    ) -> torch.Tensor | None:
        """
        The mean-free-path absorption field of the dry specimen, or None.

        The same field the particle generators build
        (:func:`~specter.imagegenerator._base.mfp_absorption_field`), read off
        the specimen before ice is blended into it, with the occupancy
        reference :func:`~specter.ice.blend_ice_into_volume` uses to weight
        the ice, so specimen and solvent absorb in the voxels that hold them.

        Unlike the particle path, the uniform case (no specimen mean free
        path) is a field and not a scalar. A scalar factorises out of the
        transmission function only when every slice is full of material. A
        tilted slice is not: past the edges of the rotated slab it samples
        vacuum, which the slicer zero-fills. A constant field over the
        volume's box is sampled with the volume, so it absorbs where the
        tilted beam crosses ice and nowhere else, and its path length grows
        as ``t / cos(theta)`` as the ice's does.

        Parameters
        ----------
        dry : torch.Tensor
            The specimen potential before ice is blended in, shape
            ``(1, Z, Y, X)``.
        pixel_size : float
            Voxel size in Angstrom.
        removal_mfps : RemovalMFPs
            The resolved mean free paths.
        has_solvent : bool
            Whether ice will be blended into the volume. Without it the
            specimen is surrounded by vacuum, which does not absorb.

        Returns
        -------
        torch.Tensor or None
            The absorption potential in volts, or None when nothing absorbs:
            no ice and no specimen mean free path, the case in which the
            particle path's uniform absorption is zero too.
        """
        if removal_mfps.inelastic_specimen is None and not has_solvent:
            return None
        return mfp_absorption_field(dry, pixel_size, removal_mfps, has_solvent)

    def _fit_volume_to_tilt(
        self,
        volume: torch.Tensor,
        desired_nxy: int,
        angles: torch.Tensor | Sequence[float] | None,
        quaternions: torch.Tensor | None,
        edge_margin: int,
        taper_width: int,
        z_taper_width: int,
        pad_volume: bool,
    ) -> torch.Tensor:
        """
        Size the volume for the tilt range: record the XY the range needs
        (`recommended_nxy_for_max_tilt`, `max_allowed_*`), reflect-pad the
        volume to it when asked, and apply the edge tapers.
        """
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
                    "Volume XY too small for requested tilt coverage"
                    + (" and taper" if taper_width > 0 else "")
                    + f"; padded (reflect) from {available_nxy} to {volume.shape[-1]} px in XY.\n"
                    f"  micrograph_size={desired_nxy}, requested_max_tilt={self.max_tilt_angle_deg:.2f} deg, "
                    f"required_volume_nxy>={required_nxy}, edge_margin={edge_margin} "
                    f"(-> required_nxy_padded>={required_nxy_padded})"
                )
                if taper_width > 0:
                    msg += f", target_nxy (with taper)>={target_nxy}"
                logger.info(msg + ".")
            else:
                logger.info(
                    "Input volume XY may be too small for requested tilt "
                    "coverage; proceeding anyway (pad_volume=False).\n"
                    f"  micrograph_size={desired_nxy}, volume_shape={tuple(volume.shape)}, "
                    f"requested_max_tilt={self.max_tilt_angle_deg:.2f} deg,\n"
                    f"  required_volume_nxy>={required_nxy}, current_volume_nxy={available_nxy}, \n"
                    f"  max_allowed_tilt_with_current_volume\u2248{self.max_allowed_tilt_deg_for_volume:.2f} deg,\n"
                    f"  max_allowed_nxy\u2248{self.max_allowed_nxy}."
                )

        if taper_width > 0 or z_taper_width > 0:
            volume = tilt_geometry.apply_volume_cosine_taper(
                volume, taper_xy=int(taper_width), taper_z=int(z_taper_width)
            )
            if taper_width > 0:
                logger.info(
                    f"Applied cosine-taper over {taper_width} px at the XY edges."
                )
            if z_taper_width > 0:
                logger.info(
                    f"Applied cosine-taper over {z_taper_width} px "
                    f"at the Z edges (top/bottom)."
                )

        return volume

    def _tilt_index(
        self,
        i: int,
        idx: torch.Tensor | int,
        batch: int,
        tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Index selecting tilt ``i`` of a per-tilt parameter tensor, broadcast
        over the volume batch.

        Per-tilt tensors carry one entry per tilt (``len == n_tilts``); a
        scalar parameter carries one entry, which every tilt shares. Only a
        tensor whose length is neither is a per-volume parameter and keeps
        ``idx``. ``tensor`` defaults to the ``dfu`` buffer, so the CTF batch
        and the dose/coincidence batches resolve the same way.
        """
        if tensor is None:
            tensor = getattr(self, "dfu", self.dose_per_angstrom)
        n = tensor.shape[0] if tensor.ndim else 1
        device = tensor.device if tensor.ndim else None
        if n == len(self.quaternions):
            return torch.full((batch,), i, dtype=torch.long, device=device)
        if n == 1:
            return torch.zeros(batch, dtype=torch.long, device=device)
        return idx if isinstance(idx, torch.Tensor) else torch.tensor([idx])

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
        damaged = self._specimen_spectrum is not None
        if not damaged:
            volume_scaled = self.volume * scale

        for i in track(
            range(n_tilts),
            description="Generating tilt series.",
            disable=not self.progressbars,
        ):
            if damaged:
                # This tilt's damage state: its own dose, after the
                # pre-exposure of the tilts before it.
                dose_i = self.dose_per_angstrom[
                    self._tilt_index(i, idx, 1, self.dose_per_angstrom)
                ]
                self._render_exposure(float(dose_i), float(self.pre_exposure[i]))
                volume_scaled = self.volume * scale.to(self.volume.device)
            Q = self.quaternions[i].unsqueeze(0).expand(B, -1)
            T = self.translations[i].unsqueeze(0).expand(B, -1)

            R_mat = roma.unitquat_to_rotmat(Q)
            if R_mat.ndim == 2:
                R_mat = R_mat.unsqueeze(0)
            T_torch = rotations.translations_angstrom_to_torch(
                T, self.volume.shape[-1], self.pixel_size
            )
            theta_matrix = rotations.build_affine_matrix(R_mat, T_torch)

            if self.absorption_potential is None:
                exitwave = self.iterative_scattering(
                    volume_scaled, theta_matrix, slice_batchsize=self.slice_batchsize
                )
            else:
                exitwave = self.iterative_scattering.multislice_absorptive(
                    volume_scaled,
                    self.absorption_potential,
                    theta_matrix,
                    slice_batchsize=self.slice_batchsize,
                )

            # Per-tilt parameters (defocus, dose, pre-exposure, coincidence
            # radius) are stored one entry per TILT, so they are selected by
            # the tilt index, not by ``idx``, which indexes the volume batch.
            # Until 2026-09-08 they were indexed by ``idx`` here, so every
            # tilt silently reused tilt 0's defocus and dose and the dose
            # envelope saw a pre-exposure of zero on every tilt.
            tilt_idx = self._tilt_index(i, idx, B)
            ctf_batch = self._ctf_batch(tilt_idx)
            if self.scattering_model not in ["projection", "ctf"]:
                ctf_batch = tilt_geometry.shift_ctf_defocus_for_tilt(
                    ctf_batch,
                    tuple(self.volume.shape),
                    theta_matrix,
                    self.nz,
                    self.pixel_size,
                )

            detector_waves = self._aberrate(exitwave, ctf_batch)

            dose_batch = self.dose_per_angstrom[
                self._tilt_index(i, idx, B, self.dose_per_angstrom)
            ]
            cr_batch = self.coincidence_radius[
                self._tilt_index(i, idx, B, self.coincidence_radius)
            ]
            anisomag = None if self.anisomag is None else self.anisomag[idx]
            image = self.detector(
                detector_waves, dose_batch, cr_batch, anisomag, nxy=None
            )

            tilt_series.append(image.detach().cpu())
            exitwaves.append(exitwave.detach().cpu())
            # |psi|^2 on the device: the host then receives a real image
            # rather than a complex one, and the CPU does no elementwise work.
            clean_images.append((detector_waves.detach().abs() ** 2).cpu())

        return (
            torch.stack(tilt_series, dim=1),
            torch.stack(exitwaves, dim=1),
            torch.stack(clean_images, dim=1),
        )
