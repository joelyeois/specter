from __future__ import annotations

import torch
import torch.nn.functional as F

from specter import logger

from ..ice import IceBank, ice_occupancy_weight
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

    def solvate(self, V: torch.Tensor) -> torch.Tensor:
        """
        Embed the volume in amorphous ice.

        V must already be Z-padded (via ``pad_volume``) before calling this.
        Ice is XY-padded here to match the FFT-padded size when ``pad_fft``
        is True.

        Parameters
        ----------
        V : torch.Tensor
            Volume potential of shape (B, Z, Y, X).

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
        # eligibility mask -- see ice_occupancy_weight for why the old
        # `V < 0.05 * V.max()` rule coupled every voxel's ice to whatever
        # the densest thing in the volume happened to be.
        weight = ice_occupancy_weight(V)
        ice.mul_(weight)
        V = V.add_(ice)
        if hasattr(self, "icemask"):
            self.icemask = weight.detach().cpu()
        return V

    def process_volume(self, V: torch.Tensor, idx: torch.Tensor | int) -> torch.Tensor:
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
        V = V * scale

        if getattr(self, "save_clean_exitwaves", False):
            self.clean_exitwaves = self.scattering(V)

        if getattr(self, "icemaker", None) is not None:
            if self.verbose:
                logger.info(f"Adding ice to volume using {self.ice_model} model")
            with torch.no_grad():
                V = self.solvate(V)

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
