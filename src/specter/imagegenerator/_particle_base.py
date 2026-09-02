from __future__ import annotations

import torch

from specter import logger

from ..ice import IceBank
from ..potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    occupancy_blur_halo_voxels,
    potential_occupancy,
)
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

    def solvate(
        self, V: torch.Tensor, potential_scale: torch.Tensor | float = 1.0
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
        ``nxy=512``). The blur's halo reaches into the previous slab, which
        has already had its ice added; the weighted ice that slab received
        is kept (``prev_tail``) and subtracted back so the occupancy is read
        off the pristine potential, exactly as when the padded canvas held
        it.

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
            ice = self.icemaker.generate_ice(batchsize=len(V)).to(V.device)

        pad = self.nxy // 2 if self.pad_fft else 0
        ridx = _reflect_index(ice.shape[-1], pad, ice.device) if pad else None

        scale = torch.as_tensor(
            potential_scale, dtype=V.dtype, device=V.device
        ).reshape(-1, 1, 1, 1)
        # A voxel reading `full` volts of specimen is full; having divided
        # the scale out, that reference moves with it.
        full = FULL_OCCUPANCY_POTENTIAL_V * scale

        nz = V.shape[1]
        nxy = V.shape[-1]
        chunk = _solvate_chunk_slices(nxy)
        # The blur reads past its slab -- see blend_ice_into_volume.
        halo = occupancy_blur_halo_voxels(self.pixel_size)
        # One batch item at a time: the slab temporaries (the pristine copy,
        # the blur's transposed passes, the weighted ice) would otherwise all
        # scale with the batch, and at three 512-pixel boxes that was 7 GB
        # over the live volume. Per item they stay at the single-particle
        # figure, with no extra halo work since the blur is separable per item.
        for b in range(V.shape[0]):
            Vb = V[b : b + 1]
            ice_b = ice[b : b + 1]
            full_b = full[b : b + 1] if full.shape[0] > 1 else full
            prev_tail: torch.Tensor | None = None
            for start in range(0, nz, chunk):
                end = min(start + chunk, nz)
                lo, hi = max(0, start - halo), min(nz, end + halo)
                # The halo overlaps the PREVIOUS slab, which has already had
                # its ice added. Reading that directly would see inflated
                # potential, infer a fuller voxel, and starve every chunk
                # boundary of ice. `prev_tail` holds exactly what was added
                # there, so subtracting it back recovers the pristine
                # potential for one slab-sized copy.
                src = Vb[:, lo:hi].clone()
                if lo < start:
                    assert prev_tail is not None
                    src[:, : start - lo] -= prev_tail[:, -(start - lo) :]
                # `full` carries the per-image scale, and it has to be divided
                # in HERE rather than after: potential_occupancy clamps to
                # [0, 1] against whatever reference it is given, so dividing
                # afterwards would clamp against the wrong one.
                occ = potential_occupancy(src, self.pixel_size, full_potential=full_b)[
                    :, start - lo : start - lo + (end - start)
                ]
                del src
                slab = ice_b[:, start:end]
                if ridx is not None:
                    slab = slab.index_select(-2, ridx).index_select(-1, ridx)
                else:
                    slab = slab.clone()
                # In place, and reusing `occ`: `(1 - occ).clamp(0, 1)` would
                # allocate two more slabs to produce a value consumed once.
                slab.mul_(occ.neg_().add_(1.0))
                Vb[:, start:end].add_(slab)
                # Running tail of the last `halo` weighted slices, across as
                # many previous slabs as the halo spans: the slab can be
                # narrower than the halo on a very large canvas (the slab
                # budget is fixed in voxels), so the previous slab alone
                # would not cover it.
                if halo > 0:
                    tail = (
                        slab
                        if prev_tail is None
                        else torch.cat([prev_tail, slab], dim=1)
                    )
                    prev_tail = tail[:, -halo:].clone()
                    del tail
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
        if hasattr(self, "crowd"):
            if self.verbose:
                logger.info("Adding crowding molecules to volume")
            # Stamped straight onto V[i]: a separate accumulator is a second
            # padded canvas held only to be added once. (This used to also
            # keep a CPU copy of that canvas on `self.volumes`, which nothing
            # read and which, as a 2 GB pageable device-to-host copy per
            # particle, was 40% of the forward pass at a 512-pixel box.)
            # `crowd` registered the template as its own buffer, so after
            # `.to(device)` there are two device copies of it (0.5 GB at
            # 512^3). Alias the crowd's onto ours here rather than at
            # construction: Module.to() re-copies every buffer separately, so
            # an alias made before the move would be split again by it.
            if self.crowd.V is not self.V:
                self.crowd.V = self.V
            with torch.no_grad():
                for i in range(len(V)):
                    self.crowd(into=V[i])

        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1)
        # Skipped when every scale is 1 (the default): `V * scale` is a
        # second whole canvas, and the caller's own reference keeps the
        # first one alive alongside it. Same guard MicrographGenerator uses.
        if not self._potential_scale_is_unity:
            V = V * scale

        if getattr(self, "save_clean_exitwaves", False):
            self.clean_exitwaves = self.scattering(V)

        if getattr(self, "icemaker", None) is not None:
            if self.verbose:
                logger.info(f"Adding ice to volume using {self.ice_model} model")
            with torch.no_grad():
                V = self.solvate(V, potential_scale=scale)

        if self.verbose:
            logger.info(f"Applying scattering using {self.scattering_model} model")
        self.exitwaves = self.scattering(V)

        if self.verbose:
            logger.info(f"Applying aberrations using {self.aberration_model} model")
        self.detector_waves = self.aberration(self.exitwaves, self._ctf_batch(idx))

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
