from __future__ import annotations

import torch
import torch.nn.functional as F

from specter import logger

from ..ice import IceBank
from ..potential import potential_occupancy
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
        self, V: torch.Tensor, occupancy: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Embed the volume in amorphous ice.

        V must already be Z-padded (via ``pad_volume``) before calling this.
        Ice is XY-padded here to match the FFT-padded size when ``pad_fft``
        is True.

        Parameters
        ----------
        V : torch.Tensor
            Volume potential of shape (B, Z, Y, X).
        occupancy : torch.Tensor, optional
            Fraction of each voxel the specimen already fills, same shape as
            ``V``. When given, the ice weight is ``1 - occupancy``; otherwise
            it is inferred from ``V`` itself. Default None.

        Returns
        -------
        V : torch.Tensor
            Volume with ice blended in via icemask.
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

        # Per-voxel fraction of space still free for water, not a binary
        # eligibility mask. `process_volume` normally hands the field down
        # already built; this fallback covers a direct caller of `solvate`.
        # It reads V, which has been scaled by `potential_scale` by then --
        # which is exactly why `process_volume` builds its own beforehand
        # rather than leaving it to here.
        if occupancy is None:
            occupancy = potential_occupancy(V, self.pixel_size)
        weight = (1.0 - occupancy).clamp(0.0, 1.0)
        ice.mul_(weight)
        V = V.add_(ice)
        if hasattr(self, "icemask"):
            self.icemask = weight.detach().cpu()
        return V

    def process_volume(
        self,
        V: torch.Tensor,
        idx: torch.Tensor | int,
        occupancy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the particle imaging pipeline on a prepared volume.

        Parameters
        ----------
        V : torch.Tensor
            Rotated and padded volume of shape (B, Z, Y, X).
        idx : torch.Tensor or int
            Batch indices used to select per-image parameters.
        occupancy : torch.Tensor, optional
            Fraction of each voxel the specimen already fills, same shape and
            pose as ``V``, forwarded to :meth:`solvate`. Default None, which
            infers the ice weight from the potential instead.

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

        # Built here, BEFORE the scale, and from the volume as it now stands
        # -- crowding duplicates included. Two reasons it cannot wait until
        # `solvate`:
        #
        #   - `potential_scale` is an optical knob. Reading occupancy off a
        #     scaled V would let it silently change how much water the
        #     specimen displaces, which is not something a contrast factor
        #     gets to do.
        #   - a caller-supplied `occupancy` describes the TARGET particle
        #     only. Crowding duplicates are summed in above and are invisible
        #     to it, so without this they would take ice at full strength
        #     straight through their middles. Combining by maximum keeps the
        #     sharp geometric field wherever it exists and covers the rest.
        if getattr(self, "icemaker", None) is not None:
            with torch.no_grad():
                inferred = potential_occupancy(V, self.pixel_size)
                occupancy = (
                    inferred
                    if occupancy is None
                    else torch.maximum(occupancy, inferred)
                )

        scale = self.potential_scale[idx].reshape(-1, 1, 1, 1)
        V = V * scale

        if getattr(self, "save_clean_exitwaves", False):
            self.clean_exitwaves = self.scattering(V)

        if getattr(self, "icemaker", None) is not None:
            if self.verbose:
                logger.info(f"Adding ice to volume using {self.ice_model} model")
            with torch.no_grad():
                V = self.solvate(V, occupancy=occupancy)

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
