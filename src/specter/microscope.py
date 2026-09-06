"""
Aberration and detector models.

The detector's coincidence-loss model (:meth:`Detector.apply_coincidence`) is
an original, deliberately simplified spatial simulation -- it is not an
implementation of any published closed-form theory. For an analytical
treatment of the same phenomenon (exact mean/variance of recorded counts and
events versus incoming rate, via Roach's statistical-overlap model), and for
the DQE/SNR consequences, see the reference below. Note that such closed-form
per-pixel statistics carry no spatial-correlation information and therefore do
not by themselves reproduce the low-spatial-frequency dip that coincidence
loss imprints on a power spectrum.

References
----------
Zambon, P. (2024). Modeling the impact of coincidence loss on count rate
statistics and noise performance in counting detectors for imaging
applications. Frontiers in Physics, 12, 1408430.
https://doi.org/10.3389/fphy.2024.1408430
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from .fft import fft2, ifft2
from .progress import track
from specter.options import AberrationModel, NoiseModel


class Detector(L.LightningModule):
    """
    A detector module to apply detector noise to images.

    Dose and coincidence radius are not stored at construction time; they must be
    supplied per-batch via ``forward()``. This allows per-image randomisation of
    both quantities from the parent image generator.

    Parameters
    ----------
    pixel_size : float
        Pixel size in Å.
    aberration_model : str, optional
        Specifies aberration model to use. Options include 'nonlinear' and 'linear'.
        Default is 'nonlinear'.
    noise_model : str, optional
        Specifies noise model. Currently only 'poisson' available. Default is None
        (no noise applied).
    mtf : torch.Tensor, optional
        Modulation transfer function in Fourier space to apply to images.
        Must be normalised so ``MTF(0) == 1``; it is a pure blur and conserves
        total counts. Default is None (no MTF applied).
    dqe0 : float, optional
        Zero-frequency detective quantum efficiency, i.e. the fraction of
        incident electrons the detector records at all. Scales the expected
        electron count, so it reduces the mean *and* the shot noise
        consistently (thinning a Poisson process leaves a Poisson process).
        Default 1.0 (an ideal counter: every electron is recorded).

        Applied to the signal in :meth:`image`, so it takes effect regardless
        of ``noise_model`` -- it is a property of detection, not of noise.
        Use :func:`specter.detectors.dqe0_for_detector` to get the published
        value for a named detector.
    n_frames : int, optional
        Number of frames for dose-fractionated noise.
    progressbars : bool, optional
        Whether to show progress bars. Default True.

    Notes
    -----
    ``mtf`` and ``dqe0`` are the two halves of a measured DQE curve and should
    be taken from the same detector: ``dqe0 = DQE(0)`` and
    ``MTF(k) = sqrt(DQE(k)/DQE(0))``. Setting one without the other silently
    mis-models the detector.
    """

    def __init__(
        self,
        pixel_size: float,
        aberration_model: AberrationModel = "nonlinear",
        noise_model: NoiseModel | None = None,
        mtf: torch.Tensor | None = None,
        dqe0: float = 1.0,
        n_frames: int | None = None,
        progressbars: bool = True,
    ):
        super().__init__()
        if not 0.0 < dqe0 <= 1.0:
            raise ValueError(f"dqe0 must be in (0, 1], got {dqe0}")
        self.pixel_size = pixel_size
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.register_buffer("mtf", mtf, persistent=False)
        self.dqe0 = dqe0
        self.n_frames = n_frames
        self.progressbars = progressbars

    def image(
        self, aberrated_exitwave: torch.Tensor, dose: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert aberrated exit wave to detector image.

        Parameters
        ----------
        aberrated_exitwave : torch.Tensor
            Aberrated exit wave from microscope aberration module, shape (B, Y, X).
        dose : torch.Tensor
            Total dose per image in e⁻/Å², shape (B,). Used for CTF model
            scaling.

        Returns
        -------
        images : torch.Tensor
            Detector image. For the nonlinear model, returns intensity
            (squared magnitude). For the linear model, returns dose-scaled
            image.

        Notes
        -----
        ``dqe0`` is folded into the dose here rather than applied downstream:
        an electron that is never detected cannot contribute to the image, be
        blurred by the MTF, or block a neighbour via coincidence loss. Scaling
        the expected count is exactly equivalent to thinning the arrival
        process, so the shot noise stays correct once ``torch.poisson`` is
        applied later.
        """
        dose_per_pixel = dose * self.pixel_size**2 * self.dqe0  # (B,)
        if self.aberration_model == "nonlinear":
            images = dose_per_pixel[:, None, None] * torch.abs(aberrated_exitwave) ** 2
        elif self.aberration_model == "linear":
            images = dose_per_pixel[:, None, None] * (aberrated_exitwave + 1)
        return images

    def anisomagnify(
        self, images: torch.Tensor, anisomag: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply anisotropic magnification to images.

        Parameters
        ----------
        images : torch.Tensor
            Input images with shape (B, H, W) where B is batch size.
        anisomag : torch.Tensor
            2x2 anisotropic magnification matrix for each image in batch,
            shape (B, 2, 2).

        Returns
        -------
        magnified_images : torch.Tensor
            Images after anisotropic magnification, shape (B, H, W).
        """
        images = images.unsqueeze(1)
        B = len(images)

        # Identity matrix (Bx3x3)
        M_affine = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        M_affine = M_affine.to(images.device)
        M_affine[:, :2, :2] = anisomag

        # Convert to (B, 2, 3) format by repeating for all batch elements
        M_affine = M_affine[:, :2, :]  # Shape: (B, 2, 3)

        # Generate affine grid for the batch
        grid = F.affine_grid(M_affine, list(images.shape), align_corners=False)

        # Apply transformation using grid sampling
        images = F.grid_sample(images, grid, align_corners=False, padding_mode="border")
        images = torch.squeeze(images, dim=1)
        return images

    def add_mtf(self, images: torch.Tensor, mtf: torch.Tensor) -> torch.Tensor:
        """
        Apply modulation transfer function (MTF) to images.

        Parameters
        ----------
        images : torch.Tensor
            Input images.
        mtf : torch.Tensor
            Modulation transfer function in Fourier space.

        Returns
        -------
        filtered_images : torch.Tensor
            Real-valued images after MTF application.
        """
        return torch.real(ifft2(fft2(images) * mtf))

    def forward(
        self,
        aberrated_exitwave: torch.Tensor,
        dose: torch.Tensor,
        coincidence_radius: torch.Tensor,
        anisomag: torch.Tensor | None = None,
        nxy: int | None = None,
    ) -> torch.Tensor:
        """
        Simulate detection process: conversion to intensity, magnification, MTF, and noise.

        Parameters
        ----------
        aberrated_exitwave : torch.Tensor
            Aberrated exit wave from microscope aberration module.
        dose : torch.Tensor
            Total dose per image in e⁻/Å², shape (B,).
        coincidence_radius : torch.Tensor
            Coincidence radius per image in pixels, shape (B,).
        anisomag : torch.Tensor, optional
            Anisotropic magnification matrices.
        nxy : int, optional
            Output image size in pixels. If provided, center-crops the image.

        Returns
        -------
        images : torch.Tensor
            Simulated images after detection.
        """
        images = self.image(aberrated_exitwave, dose)

        # Set default crop size
        if nxy is None:
            nxy = images.shape[2]

        # Crop if needed
        if nxy != images.shape[2]:
            H, W = images.shape[1], images.shape[2]
            cy, cx = H // 2, W // 2

            half = nxy // 2
            images = images[
                :,
                cy - half : cy + half + (nxy % 2),
                cx - half : cx + half + (nxy % 2),
            ]

        # Apply anisomagnification
        if anisomag is not None:
            images = self.anisomagnify(images, anisomag)

        # Apply detector MTF
        if self.mtf is not None:
            images = self.add_mtf(images, self.mtf)

        if self.noise_model is None:
            return images
        else:
            return torch.stack(
                [
                    self.apply_coincidence(img, d.item(), r.item())
                    for img, d, r in zip(images, dose, coincidence_radius)
                ]
            )

    def apply_detector_physics(
        self,
        intensity_map: torch.Tensor,
        pixel_size: float,
        dose_per_angstrom_sq_per_frame: float,
        coinc_radius_pixels: float = 0.6,
    ) -> torch.Tensor:
        """
        Simulates a single frame of a Direct Electron Detector (DED), using a
        randomized square-cell grid to suppress coincident arrivals.

        Parameters
        ----------
        intensity_map : torch.Tensor
            2D Tensor (normalized psi^2 from multislice).
        pixel_size : float
            Size of one pixel in Å.
        dose_per_angstrom_sq_per_frame : float
            Physical dose in this single frame, in e⁻/Å² (the image total
            divided by ``n_frames``).
        coinc_radius_pixels : float, optional
            Effective coincidence exclusion radius in pixels -- see
            :meth:`apply_coincidence` for the definition and calibration.

        Notes
        -----
        The suppression grid's cell side is ``r * sqrt(pi)`` so that the cell
        *area* equals ``pi * r**2``, i.e. the area of the exclusion disc of
        radius ``r``. Since the electron-loss rate depends on the exclusion
        *area* (not on the cell's shape), this makes ``coinc_radius_pixels``
        numerically equivalent to the radius of a true pairwise exclusion
        disc, rather than a shape-specific grid parameter. Verified against an
        exact O(n^2) pairwise implementation: the fitted effective area of
        this rule matches ``pi * r**2`` to within 0.4% over r = 1-3 px, and
        the pairwise rule to within 0.2% -- see
        ``tests/test_detector_coincidence.py``.

        Resolving exclusion per cell also keeps coincidence *locally bounded*:
        it cannot chain transitively across the frame the way a pairwise
        connected-component rule does, which is what keeps the model stable at
        the high dose rates it was calibrated against.
        """
        device = intensity_map.device

        # Nominal cell side, chosen so cell area == pi*r^2 (see Notes above).
        cell_size_nominal = coinc_radius_pixels * math.sqrt(math.pi)

        # 0. Pad map to avoid edge artifacts (a few cell widths)
        pad = int(np.ceil(cell_size_nominal * 3))
        orig_h, orig_w = intensity_map.shape
        # Use reflect padding to keep intensity levels consistent at the edge
        intensity_map = F.pad(
            intensity_map.unsqueeze(0).unsqueeze(0),
            (pad, pad, pad, pad),
            mode="reflect",
        ).squeeze()
        det_h, det_w = intensity_map.shape

        # 1. Convert physical dose to expected electron count
        # Use original (unpadded) dimensions: Poisson-per-pixel sampling means
        # padded pixels inflate the per-pixel lambda in the original region.
        dose_per_pixel = dose_per_angstrom_sq_per_frame * (pixel_size**2)
        total_expected = dose_per_pixel * orig_h * orig_w

        # 2. Sample landing positions from intensity map + sub-pixel jitter
        # expected electrons per pixel
        lambda_map = intensity_map * total_expected

        # sample electrons per pixel
        counts = torch.poisson(lambda_map)

        # get pixels that received electrons
        iy, ix = torch.nonzero(counts, as_tuple=True)
        n_per_pixel = counts[iy, ix].long()

        # total electrons
        n_e = int(n_per_pixel.sum().item())
        if n_e == 0:
            return torch.zeros_like(intensity_map)[pad:-pad, pad:-pad]

        # repeat pixel coordinates
        ix = ix.repeat_interleave(n_per_pixel)
        iy = iy.repeat_interleave(n_per_pixel)

        coords = torch.stack([ix.float(), iy.float()], dim=1)

        # add subpixel jitter
        coords += torch.rand((n_e, 2), device=device)

        # 3. Assign electrons to coincidence grid cells
        # Introduce a slight randomization to the cell size to mimic detector variation
        cell_size = cell_size_nominal * (
            1 + 0.05 * torch.randn(1, device=device)
        ).clamp(0.8, 1.2)

        # Random shift to prevent the grid from "locking" onto specific pixels
        shift = (torch.rand(2, device=device) - 0.5) * cell_size
        shifted_coords = coords + shift

        # Calculate cell coordinates
        cell_x = (shifted_coords[:, 0] / cell_size).floor().long()
        cell_y = (shifted_coords[:, 1] / cell_size).floor().long()

        # 2. Robust cell_id generation
        # Normalize to zero by subtracting the minimums found in this specific frame
        cx_min, cy_min = cell_x.min(), cell_y.min()
        grid_w = (cell_x.max() - cx_min) + 1

        # Generate unique 1D hash for each grid cell
        cell_id = (cell_y - cy_min) * grid_w + (cell_x - cx_min)

        # 4. Coincidence suppression
        perm = torch.randperm(n_e, device=device)
        sort_idx = torch.argsort(cell_id[perm], stable=True)
        sorted_cell_id = cell_id[perm][sort_idx]

        is_first = torch.ones(n_e, dtype=torch.bool, device=device)
        is_first[1:] = sorted_cell_id[1:] != sorted_cell_id[:-1]
        coords_kept = coords[perm][sort_idx][is_first]

        # 5. Bin into detector pixels
        ix_f = coords_kept[:, 0].long().clamp(0, det_w - 1)
        iy_f = coords_kept[:, 1].long().clamp(0, det_h - 1)
        flat_idx = iy_f * det_w + ix_f
        pixels = torch.zeros(det_h * det_w, dtype=intensity_map.dtype, device=device)
        pixels.scatter_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=intensity_map.dtype)
        )

        return pixels.reshape(det_h, det_w)[pad:-pad, pad:-pad]

    def apply_coincidence(
        self, img: torch.Tensor, dose: float, coincidence_radius: float
    ) -> torch.Tensor:
        """
        Apply dose-fractionated Poisson + coincidence.

        Parameters
        ----------
        img : torch.Tensor
            Total-dose image, shape (H, W).
        dose : float
            Total dose for this image in e⁻/Å².
        coincidence_radius : float
            Effective coincidence exclusion radius in pixels: within a single
            readout frame, an arriving electron is lost if it lands inside the
            exclusion area (``pi * r**2``) of one already recorded. If <= 0,
            plain Poisson noise is applied with no coincidence loss.

            This is a *physical* radius -- the lateral scale over which one
            electron's charge cloud renders the detector unable to resolve a
            second arrival -- so it converts directly to real units by
            multiplying by the detector's physical pixel pitch.

        Returns
        -------
        final_image : torch.Tensor
            Simulated image after dose-fractionated noise and coincidence.

        Notes
        -----
        Calibrated against beam-only Falcon 4i micrographs spanning
        0.15-31.29 e-/px/s, with the detector's counting efficiency modelled
        explicitly as ``dqe0=0.92``: ``coincidence_radius = 2.0`` px (~28 um
        at the 4096^2 sensor's 14 um pitch) reproduces the measured
        detected-electron yield to ~3% RMSE across the full dose range,
        together with the characteristic low-spatial-frequency dip in the
        power spectrum (``manuscript/coincidence-loss-exp.ipynb``). An
        earlier fit of 2.394 px was made without DQE(0) and absorbed that
        ~8% loss into the radius; using it together with ``dqe0=0.92`` counts
        the loss twice. The radius is a property of the camera unit and its
        counting configuration, so treat this value as a prior and
        re-calibrate from beam-only or empty-ice frames when a dataset
        provides them.

        This is a deliberately simplified, *locally bounded* model: exclusion
        is resolved per grid cell, so coincidence cannot chain transitively
        across the frame. Roach-style statistical-overlap models (see the
        module docstring's Zambon reference) instead merge any connected chain
        of overlapping events into one, which is a good description at low
        flux but percolates into a single frame-spanning cluster at the
        higher dose rates measured here.

        The radius is a true exclusion radius: the suppression grid's cell
        side is ``r * sqrt(pi / 2)``, so the effective exclusion area is
        ``pi * r**2`` (a value that indexed the cell side as ``r / sqrt(2)``
        would be ``sqrt(2 * pi)`` times larger).
        """
        if self.noise_model != "poisson":
            return img

        if coincidence_radius <= 0.0:
            return torch.poisson(torch.clamp(img, min=0.0))

        if self.n_frames is None:
            raise ValueError(
                "n_frames must be set on the Detector to apply dose-fractionated "
                "coincidence loss (coincidence_radius > 0)."
            )
        n_frames = self.n_frames

        # Clamp before normalizing, matching the coincidence_radius<=0 branch
        # above -- otherwise any negative pixel survives the division below.
        img = torch.clamp(img, min=0.0)
        img_sum = img.sum()
        if img_sum <= 0.0:
            # No expected electrons anywhere (e.g. an all-zero specimen
            # volume -- can happen with scattering_model="ctf", which has no
            # vacuum baseline, unlike multislice's exp(i*sigma*dz*V) == 1 at
            # V=0). img / img_sum would be 0/0 = NaN here; the physically
            # correct output for zero expected signal is a blank frame.
            return torch.zeros_like(img)
        intensity_map = img / img_sum

        # Use img_sum to derive the effective dose so that the radius>0 path
        # agrees with torch.poisson(img) at radius=0: both correctly account
        # for real electron absorption from alpha (imaginary potential), and
        # B-factor attenuation is treated consistently across both paths.
        dose_effective = (
            img_sum / (self.pixel_size**2 * img.shape[0] * img.shape[1])
        ).item()

        final_image = torch.zeros_like(img)
        for _ in track(
            range(n_frames),
            description="Applying coincidence loss",
            transient=True,
            disable=not (self.progressbars),
        ):
            final_image += self.apply_detector_physics(
                intensity_map,
                self.pixel_size,
                dose_effective / n_frames,
                coinc_radius_pixels=coincidence_radius,
            )

        return final_image
