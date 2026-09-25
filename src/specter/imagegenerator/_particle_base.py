"""
`ParticleGeneratorBase`: what the two single-particle generators share --
crowding, solvation with ice, and the per-batch volume pipeline.
"""

from __future__ import annotations

import contextlib
from typing import cast

import torch

from specter import logger

from ..arrays import compute_nz
from ..cpu_threads import limited_cpu_threads
from ..crowding import CrowdWithDuplicates
from ..ice import IceBank, RandomIcemaker, resolve_icemaker
from ..ice._blend import IceSlabBlender
from ..potential import (
    aperture_lowpass,
    apply_dose_damage,
    potential_occupancy_slabs,
    template_occupancy_reference,
)
from ..scattering import Scattering
from ..settings import Crowding, Ice, Propagation
from ._base import BaseImager

__all__ = ["ParticleGeneratorBase"]

#: Voxel budget for one solvate slab, and the slice count it is capped at.
#: The occupancy blur reads `halo` slices past each slab on both sides, so
#: the redundant work is (chunk + 2 halo) / chunk: 2x at the 16 slices a
#: 2**24 budget gave a 1024^2 canvas, 1.25x at 64. Past 64 the slab only
#: grows the blur's transposed intermediates (a whole 256-slice volume in
#: one slab took a 256-pixel box from 2.4 to 3.3 GB) for a halo saving that
#: has already flattened out.
_SOLVATE_SLAB_VOXELS = 2**26
_SOLVATE_MAX_SLICES = 64


def _solvate_chunk_slices(nxy: int) -> int:
    return max(1, min(_SOLVATE_SLAB_VOXELS // (nxy * nxy), _SOLVATE_MAX_SLICES))


def _reflect_index(n: int, pad: int, device: torch.device) -> torch.Tensor:
    """
    Indices into an axis of length `n` that reproduce
    ``F.pad(..., (pad, pad), mode="reflect")`` along it by gathering.

    Parameters
    ----------
    n : int
        Axis length before padding.
    pad : int
        Padding on each side; must be smaller than `n`, as for ``F.pad``.
    device : torch.device
        Device for the index tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(n + 2 * pad,)``, int64.
    """
    if pad >= n:
        raise ValueError(f"reflect padding {pad} must be smaller than the axis {n}")
    i = torch.arange(-pad, n + pad, device=device)
    i = torch.where(i < 0, -i, i)
    return torch.where(i >= n, 2 * (n - 1) - i, i)


class ParticleGeneratorBase(BaseImager):
    """
    Base class for particle image generators.

    Extends ``BaseImager`` with the particle-specific imaging pipeline:
    crowding → potential scaling → ice (solvation) → scatter → aberrate →
    detect.  ``ImageGenerator`` and ``ImageGeneratorFromCoordinates`` both
    inherit from this class.

    The concrete subclass is responsible for building the rotated volume ``V``
    of shape (B, Z, Y, X) and passing it to ``process_volume``.
    """

    quaternions: torch.Tensor
    translations: torch.Tensor

    # The dose envelope can act on the specimen's potential before the
    # solvent is added (Envelopes(dose_envelope_target="specimen")).
    _supports_specimen_damage: bool = True

    #: The template's mass in daltons, set by the concrete generator; None
    #: when only a bare volume was given, which keeps the fixed reference.
    molecular_mass: float | None = None
    _occupancy_reference_V: float | None = None

    def _set_molecular_mass(self, molecular_mass: float | None) -> None:
        """Record the template's mass, in daltons, hydrogens included."""
        if molecular_mass is not None and molecular_mass <= 0:
            raise ValueError(f"molecular_mass must be positive, got {molecular_mass}")
        self.molecular_mass = molecular_mass
        self._occupancy_reference_V = None

    def _occupancy_reference(self) -> float:
        """
        The potential at which a voxel of this template reads as full, V.

        Solved once, by :func:`~specter.potential.template_occupancy_reference`,
        so the particle displaces exactly its own volume of ice whatever
        scattering factors rendered it. Computed on first use rather than at
        construction, so it runs on the device the template has been moved
        to (0.04 s on a GPU against 4 s on the CPU for a 512^3 box), and so a
        DDP rank's zero placeholder is never read before rank 0's volume has
        been broadcast.
        """
        if self._occupancy_reference_V is None:
            self._occupancy_reference_V = template_occupancy_reference(
                self.V, self.pixel_size, self.molecular_mass
            )
        return self._occupancy_reference_V

    icemaker: IceBank | RandomIcemaker | None

    def _init_ice_geometry(
        self,
        nxy: int,
        template_nz: int,
        pixel_size: float,
        ice: Ice,
        propagation: Propagation,
    ) -> None:
        """
        Set the ice and box-geometry attributes, before ``BaseImager.__init__``.

        ``pad_nxy`` adds an ``nxy // 2`` FFT margin on each side when
        ``propagation.pad_fft`` is on, and ``nz`` is the template's depth
        grown to hold ``ice.thickness`` (:func:`~specter.arrays.compute_nz`).

        Parameters
        ----------
        nxy : int
            Lateral box size of the template, in pixels.
        template_nz : int
            Depth of the template, in slices.
        pixel_size : float
            Pixel size in Å.
        ice : Ice
            Ice settings.
        propagation : Propagation
            Propagation settings; only ``pad_fft`` is read.
        """
        self.pad_fft = propagation.pad_fft
        self.ice = ice
        self.ice_thickness = ice.thickness
        self.ice_relax_steps = ice.relax_steps
        self.pad_nxy = nxy + (nxy // 2) * 2 if self.pad_fft else nxy
        self.nz = compute_nz(template_nz, ice.thickness, pixel_size)

    def _init_crowding(
        self, crowding: Crowding, ice: Ice, template_nz: int, pixel_size: float
    ) -> None:
        """
        Record the crowding settings, resolving ``crowd_max_distance_z``.

        The default is the TEMPLATE's depth, ``template_nz * pixel_size``,
        not ``self.nz * pixel_size``: the neighbour slab must not follow
        ``ice.thickness``. The two agree until the ice is deeper than the
        box, which is what makes this a no-op for every run that was not
        growing its crowding.

        Parameters
        ----------
        crowding : Crowding
            Crowding settings.
        ice : Ice
            Ice settings; ``model`` becomes ``ice_model``.
        template_nz : int
            Depth of the template, in slices.
        pixel_size : float
            Pixel size in Å.
        """
        self.ice_model = ice.model
        self.crowding = crowding
        self.crowd_max_distance_z = (
            crowding.max_distance_z
            if crowding.max_distance_z is not None
            else template_nz * pixel_size
        )
        self.crowd_min_distance = crowding.min_distance

    def _init_icemaker(
        self,
        ice: Ice,
        icemaker: IceBank | RandomIcemaker | None,
        pixel_size: float,
        progressbars: bool,
    ) -> None:
        """
        Resolve ``self.icemaker`` from the ice settings or a prebuilt one.

        Must follow :meth:`_init_crowding`, which sets ``ice_model``; a
        prebuilt `icemaker` overrides it with its own method.

        Parameters
        ----------
        ice : Ice
            Ice settings.
        icemaker : IceBank or RandomIcemaker or None
            Prebuilt icemaker, used as given when not None.
        pixel_size : float
            Pixel size in Å.
        progressbars : bool
            Whether building the icemaker shows progress bars.
        """
        self.ice_parameterization = ice.parameterization
        self.icemaker = resolve_icemaker(
            self.ice_model,
            pixel_size,
            self.nxy,
            self.nz,
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=self.ice_parameterization,
            progressbars=progressbars,
        )
        if icemaker is not None:
            self.ice_model = icemaker.method

    def _build_crowd(
        self,
        template: torch.Tensor,
        crowding: Crowding,
        progressbars: bool,
        move_to_cpu: bool = False,
    ) -> CrowdWithDuplicates:
        """
        Build the crowding stage for a template from a `Crowding` bundle.

        ``nxy_out`` is the padded box when ``pad_fft`` is on, and
        ``max_distance_z`` is the already-resolved
        ``self.crowd_max_distance_z`` (the template's own depth by default).
        """
        if crowding.min_distance is None:
            raise ValueError("crowding.min_distance is required to build a crowd.")
        return CrowdWithDuplicates(
            template,
            self.pixel_size,
            crowding.min_distance,
            nxy_out=self.pad_nxy if self.pad_fft else self.nxy,
            nz_out=self.nz,
            max_distance_z=self.crowd_max_distance_z,
            max_distance_xy=crowding.max_distance_xy,
            method=crowding.method,
            n_points=crowding.n_points if crowding.n_points is not None else torch.inf,
            seed=crowding.seed,
            move_to_cpu=move_to_cpu,
            progressbars=progressbars,
            chunk_size=crowding.chunk_size,
            water_air_interface=crowding.water_air_interface,
            sigma_frac=crowding.sigma_frac,
            peak_amplitude=crowding.peak_amplitude,
            baseline=crowding.baseline,
        )

    def _build_scattering(self) -> Scattering:
        """The whole-volume propagator, from ``self.propagation``."""
        return Scattering(
            self.pad_nxy,
            self.pixel_size,
            self.voltage,
            scattering_model=self.scattering_model,
            klim=self.klim,
            ews_curvature_sign=self.ews_curvature_sign,
            nz=self.nz,
            alpha=self.alpha,
            progressbars=self.progressbars,
        )

    def free_fraction_field(
        self, V: torch.Tensor, potential_scale: torch.Tensor | float = 1.0
    ) -> torch.Tensor:
        """
        The fraction of each voxel still available to water, ``1 - occupancy``.

        Computed from `V` as given, so a caller can take it from the
        UNDAMAGED specimen and hand it to :meth:`solvate` after the specimen
        has been damaged. A molecule occupies the same volume whatever its
        radiation history, and the dose envelope conserves the potential's
        integral while spreading it, so reading occupancy after damage admits
        ice into a solid interior (peak core occupancy 1.00 -> 0.81 on 1A6M
        at 50 e-/A^2).

        Stored as float16: a weight in [0, 1] multiplying the ice, where half
        precision costs ~5e-4 relative against a blur that is itself an
        approximation, at half the memory of a full-precision canvas.

        Parameters
        ----------
        V : torch.Tensor
            Specimen potential, ``(B, Z, Y, X)``, already scaled.
        potential_scale : torch.Tensor or float, optional
            The scale `V` carries, divided back out of the occupancy
            reference so a contrast knob cannot change the water budget.

        Returns
        -------
        torch.Tensor
            ``(B, Z, Y, X)`` float16 in [0, 1], on `V`'s device.
        """
        scale = torch.as_tensor(
            potential_scale, dtype=V.dtype, device=V.device
        ).reshape(-1, 1, 1, 1)
        full = self._occupancy_reference() * scale
        chunk = _solvate_chunk_slices(V.shape[-1])
        free = torch.empty(V.shape, dtype=torch.float16, device=V.device)
        for b in range(V.shape[0]):
            full_b = full[b : b + 1] if full.shape[0] > 1 else full
            for start, end, occ in potential_occupancy_slabs(
                V[b : b + 1], self.pixel_size, chunk, full_potential=full_b
            ):
                free[b : b + 1, start:end] = (
                    occ.neg_().add_(1.0).clamp_(0.0, 1.0).to(torch.float16)
                )
                del occ
        return free

    def _apply_solvent_exposure(self, ice: torch.Tensor) -> None:
        """
        Filter the ice's fluctuation, in place, to what survives the exposure.

        A no-op unless ``Ice(motion_variance=...)`` is set. The whole batch
        shares one dose, since one filter serves the batch's ice canvas.
        """
        if self.ice.motion_variance is None:
            return
        from ..ice._exposure import apply_solvent_exposure

        doses = torch.as_tensor(self.dose_per_angstrom).flatten()
        if doses.numel() > 1 and not torch.allclose(doses, doses[0].expand_as(doses)):
            raise ValueError(
                "Ice(motion_variance=...) needs one dose for every image: the "
                "exposure filter acts on a whole batch's ice canvas at once"
            )
        apply_solvent_exposure(
            ice,
            self.pixel_size,
            float(doses[0]),
            self.ice.motion_variance,
            self.detector.n_frames or 1,
            self.detector.dose_weights,
            self._dose_weights_max_frequency,
        )

    def solvate(
        self,
        V: torch.Tensor,
        potential_scale: torch.Tensor | float = 1.0,
        free: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Embed the volume in amorphous ice.

        V must already be Z-padded (via ``pad_volume``) before calling this.
        Ice is XY-padded (by reflection, per slab, see Notes) to match the
        FFT-padded size when ``pad_fft`` is True.

        Parameters
        ----------
        V : torch.Tensor
            Volume potential of shape (B, Z, Y, X), already multiplied by
            `potential_scale`.
        potential_scale : torch.Tensor or float, optional
            The scale ``V`` has already been multiplied by, broadcastable to
            ``(B, 1, 1, 1)``. Divided back out when reading occupancy, so a
            contrast knob cannot change how much water the specimen
            displaces. Default 1.0.
        free : torch.Tensor or None, optional
            Precomputed free fraction (:meth:`free_fraction_field`) to weight
            the ice by, instead of reading occupancy off `V`. How the
            specimen-damage path keeps occupancy from the undamaged
            specimen. Default None.

        Returns
        -------
        V : torch.Tensor
            ``V`` with ice blended in, modified in place.

        Notes
        -----
        Blended a z-slab at a time, for the same reason
        :func:`~specter.ice.blend_ice_into_volume` is. The whole-volume
        spelling holds V, the ice, the occupancy, the weight and two
        clamp/subtract temporaries at once -- six canvases, which at
        ``nxy=512`` with ``pad_fft`` is 12 GB and is what made this OOM.
        Slabbing bounds everything but V to one slab each.

        The ice is generated at the UNPADDED box and, under ``pad_fft``,
        reflect-padded into the FFT margin. That padding is never
        materialised: each slab gathers its padded rows and columns from the
        unpadded ice by index (``_reflect_index``), so the ice costs one
        unpadded canvas rather than a padded one (0.5 GB against 2 GB at
        ``nxy=512``). The occupancy weighting and the slab-boundary
        bookkeeping are :class:`~specter.ice._blend.IceSlabBlender`'s.

        Occupancy is read from the SCALED volume with the scale divided
        back out, rather than being computed before the scale and carried
        down. Both give the identical field -- the blur is linear, so
        ``blur(s * V) / s == blur(V)`` -- but carrying it costs a whole
        extra canvas, and the point is only ever to keep `potential_scale`
        out of the water budget.
        """
        if isinstance(self.icemaker, IceBank):
            ice = self.icemaker.generate_big_ice(
                n=self.nxy,
                dx=self.pixel_size,
                nz=self.nz,
                batchsize=len(V),
                relax_steps=self.ice_relax_steps,
            ).to(V.device)
        else:
            # Only reached with an icemaker: `process_volume` skips solvation
            # without one.
            icemaker = cast(RandomIcemaker, self.icemaker)
            ice = icemaker.generate_ice(batchsize=len(V)).to(V.device)
        # On the pure ice canvas, before the particle's hole is cut: the
        # hole moves with the particle and never decorrelates.
        self._apply_solvent_exposure(ice)

        pad = self.nxy // 2 if self.pad_fft else 0
        ridx = _reflect_index(ice.shape[-1], pad, ice.device) if pad else None

        scale = torch.as_tensor(
            potential_scale, dtype=V.dtype, device=V.device
        ).reshape(-1, 1, 1, 1)
        # A voxel reading `full` volts of specimen is full; having divided
        # the scale out, that reference moves with it.
        full = self._occupancy_reference() * scale

        nz = V.shape[1]
        nxy = V.shape[-1]
        chunk = _solvate_chunk_slices(nxy)
        # One batch item at a time: the slab temporaries (the pristine copy,
        # the blur's transposed passes, the weighted ice) would otherwise all
        # scale with the batch, and at three 512-pixel boxes that was 7 GB
        # over the live volume. Per item they stay at the single-particle
        # figure, with no extra halo work since the blur is separable per item.
        for b in range(V.shape[0]):
            Vb = V[b : b + 1]
            ice_b = ice[b : b + 1]
            full_b = full[b : b + 1] if full.shape[0] > 1 else full
            # `full` carries the per-image scale, and it has to be divided in
            # HERE rather than after: potential_occupancy clamps to [0, 1]
            # against whatever reference it is given, so dividing afterwards
            # would clamp against the wrong one.
            blender = IceSlabBlender(self.pixel_size, full_potential=full_b)
            for start in range(0, nz, chunk):
                end = min(start + chunk, nz)
                slab = ice_b[:, start:end]
                if ridx is not None:
                    slab = slab.index_select(-2, ridx).index_select(-1, ridx)
                else:
                    slab = slab.clone()
                blender.add(
                    Vb, slab, start, end, free=None if free is None else free[b : b + 1]
                )
                del slab
        return V

    def process_volume(
        self,
        V: torch.Tensor,
        idx: torch.Tensor | int,
    ) -> torch.Tensor:
        """
        Run the particle imaging pipeline on a prepared volume.

        Parameters
        ----------
        V : torch.Tensor
            Rotated and padded volume of shape (B, Z, Y, X).
        idx : torch.Tensor or int
            Batch indices used to select per-image parameters.

        Returns
        -------
        images : torch.Tensor
            Simulated detector images.
        """
        # With the volume on the device, every CPU tensor op left in the
        # batch is a small one (tile placement, crop extraction, per-frame
        # detector bookkeeping), and each pays torch's per-op sync cost in
        # proportion to the pool. Capping once here also spares the scoped
        # caps inside (crowd insertion, ice tiling, Poisson-disk sampling)
        # their ~12 ms pool resize each, which at 64 x 2 resizes per batch
        # was a third of a 256 px batch. A CPU run keeps the full pool for
        # its multislice and blur, and caps only the crowd loop below.
        cap = limited_cpu_threads() if V.is_cuda else contextlib.nullcontext()
        with cap:
            return self._process_volume(V, idx)

    def _process_volume(self, V: torch.Tensor, idx: torch.Tensor | int) -> torch.Tensor:
        if hasattr(self, "crowd"):
            if self.verbose:
                logger.info("Adding crowding molecules to volume")
            # Stamped straight onto V[i]: a separate accumulator is a second
            # padded canvas held only to be added once, and a host copy of
            # it is a 2 GB pageable device-to-host transfer per particle,
            # 40% of the forward pass at a 512-pixel box.
            # `crowd` registered the template as its own buffer, so after
            # `.to(device)` there are two device copies of it (0.5 GB at
            # 512^3). Alias the crowd's onto ours here rather than at
            # construction: Module.to() re-copies every buffer separately, so
            # an alias made before the move would be split again by it.
            if self.crowd.V is not self.V:
                self.crowd.V = self.V
            # One thread cap for the whole batch. Each duplicate insertion
            # caps the pool itself, but resizing a 128-thread pool costs
            # ~12 ms each way, and per chunk that was 21 s of a 28 s
            # 64-particle run at 128 px (3.5x the simulation). Under this
            # outer cap the inner ones find the pool already small and do
            # nothing.
            with torch.no_grad(), limited_cpu_threads():
                for i in range(len(V)):
                    self.crowd(into=V[i])

        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1)
        # Skipped when every scale is 1 (the default): `V * scale` is a
        # second whole canvas, and the caller's own reference keeps the
        # first one alive alongside it. Same guard MicrographGenerator uses.
        if not self._potential_scale_is_unity:
            V = V * scale

        # `clean_exitwaves` is the ice-free, absorption-free reference, so it
        # is deliberately propagated from the real potential, and from the
        # undamaged one: it is the reference any damage is measured against.
        if getattr(self, "save_clean_exitwaves", False):
            self.clean_exitwaves = self.scattering(V)

        # Read the material fraction while V is still the specimen alone:
        # `solvate` writes into its input, and occupancy off a solvated volume
        # is full everywhere, which would give the whole box the specimen's
        # mean free path.
        with torch.no_grad():
            # The reference is solved only when a field will read it: a run
            # with no specimen absorption field never needs one.
            needs_field = (
                self.absorption_model == "inelastic_mfp"
                and self.propagation.inelastic_mfp_specimen is not None
            )
            v_ab = (
                self._absorption_field(
                    V, full_potential=self._occupancy_reference() * scale
                )
                if needs_field
                else None
            )

        has_ice = getattr(self, "icemaker", None) is not None
        free = None
        if self._damages_potential:
            # Occupancy first, from the UNDAMAGED specimen: how much water a
            # voxel displaces is a question about the molecule's volume, not
            # its radiation history, and the envelope spreads the potential
            # outwards while conserving its integral. Then the envelope, on
            # the specimen (and its crowding neighbours) alone, so the ice
            # added afterwards keeps its structure.
            if has_ice:
                with torch.no_grad():
                    free = self.free_fraction_field(V, potential_scale=scale)
            if self.verbose:
                logger.info("Applying dose damage to the specimen potential")
            pre = getattr(self, "pre_exposure", None)
            with torch.no_grad():
                V = apply_dose_damage(
                    V,
                    self.pixel_size,
                    self.dose_per_angstrom[idx],
                    pre_exposure=0.0 if pre is None else pre[idx],
                    weighted=self._dose_weighted,
                    voltage=self.voltage,
                    # The exposure's real frame structure and, where the run
                    # has them, its real per-frequency weights: the envelope
                    # that survives a movie is the weight-average of the
                    # frames' own damage states. See potential/_damage.py.
                    n_frames=self.n_frames,
                    frame_weights=self.detector.dose_weights,
                    frame_weights_max_frequency=self._dose_weights_max_frequency,
                )

        if has_ice:
            if self.verbose:
                logger.info(f"Adding ice to volume using {self.ice_model} model")
            with torch.no_grad():
                V = self.solvate(V, potential_scale=scale, free=free)
            del free

        # Scattering beyond the objective aperture is charged as absorption
        # (`_removal_mfp`), so the share of it the grid carries is filtered
        # out rather than propagated as well. A no-op at 1 A/px and coarser.
        if self.objective_aperture is not None:
            with torch.no_grad():
                V = aperture_lowpass(
                    V, self.pixel_size, self.objective_aperture, self.voltage
                )

        if v_ab is not None:
            V = torch.complex(V, v_ab)
            del v_ab
        # Set here rather than in `_build_scattering`: the propagator is
        # constructed before `icemaker` is, and whether there is a medium to
        # absorb in is part of the answer.
        self.scattering.uniform_absorption = self._uniform_absorption

        if self.verbose:
            logger.info(f"Applying scattering using {self.scattering_model} model")
        self.exitwaves = self.scattering(V)

        if self.verbose:
            logger.info(f"Applying aberrations using {self.aberration_model} model")
        self.detector_waves = self._aberrate(self.exitwaves, self._ctf_batch(idx))

        if self.verbose:
            logger.info(f"Applying detector and noise using {self.noise_model} model")
        dose_batch = self.dose_per_angstrom[idx]
        cr_batch = self.coincidence_radius[idx]
        if self.anisomag is None:
            images = self.detector(
                self.detector_waves, dose_batch, cr_batch, nxy=self.nxy
            )
        else:
            images = self.detector(
                self.detector_waves, dose_batch, cr_batch, self.anisomag[idx], self.nxy
            )
        return images
