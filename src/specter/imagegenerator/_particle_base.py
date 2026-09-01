from __future__ import annotations

import torch
import torch.nn.functional as F

from specter import logger

from ..ice import IceBank
from ..potential import (
    FULL_OCCUPANCY_POTENTIAL_V,
    occupancy_blur_halo_voxels,
    potential_occupancy,
)
from ._base import BaseImager

__all__ = ["ParticleGeneratorBase"]


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
        Ice is XY-padded here to match the FFT-padded size when ``pad_fft``
        is True.

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
        Slabbing bounds everything but V and the ice to one slab each.

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

        if self.pad_fft:
            ice = F.pad(
                ice,
                (self.nxy // 2, self.nxy // 2, self.nxy // 2, self.nxy // 2, 0, 0),
                mode="reflect",
            )

        scale = torch.as_tensor(
            potential_scale, dtype=V.dtype, device=V.device
        ).reshape(-1, 1, 1, 1)
        # A voxel reading `full` volts of specimen is full; having divided
        # the scale out, that reference moves with it.
        full = FULL_OCCUPANCY_POTENTIAL_V * scale

        nz = V.shape[1]
        nxy = V.shape[-1]
        chunk = max(1, 2**24 // (nxy * nxy))
        # The blur reads past its slab -- see blend_ice_into_volume.
        halo = occupancy_blur_halo_voxels(self.pixel_size)
        for start in range(0, nz, chunk):
            end = min(start + chunk, nz)
            lo, hi = max(0, start - halo), min(nz, end + halo)
            # The halo overlaps the PREVIOUS slab, which has already had its
            # ice added. Reading that directly would see inflated potential,
            # infer a fuller voxel, and starve every chunk boundary of ice.
            # `ice` holds exactly what was added, so subtracting it back
            # recovers the pristine potential for one slab-sized copy.
            src = V[:, lo:hi].clone()
            if lo < start:
                src[:, : start - lo] -= ice[:, lo:start]
            # `full` carries the per-image scale, and it has to be divided in
            # HERE rather than after: potential_occupancy clamps to [0, 1]
            # against whatever reference it is given, so dividing afterwards
            # would clamp against the wrong one.
            occ = potential_occupancy(src, self.pixel_size, full_potential=full)[
                :, start - lo : start - lo + (end - start)
            ]
            del src
            # In place, and reusing `occ`: `(1 - occ).clamp(0, 1)` would
            # allocate two more slabs to produce a value consumed once.
            ice[:, start:end].mul_(occ.neg_().add_(1.0))
            V[:, start:end].add_(ice[:, start:end])
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
            with torch.no_grad():
                for i in range(len(V)):
                    volumes = self.crowd()
                    if not isinstance(volumes, float):
                        self.volumes = volumes.detach().cpu()
                    V[i] += volumes

        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1)
        # Skipped when every scale is 1 (the default): `V * scale` is a
        # second whole canvas, and the caller's own reference keeps the
        # first one alive alongside it. Same guard MicrographGenerator uses.
        if not bool(torch.all(scale == 1.0)):
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
