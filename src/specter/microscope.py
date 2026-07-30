from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from .fft import fft2, ifft2
from .progress import track


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
        Specifies aberration model to use. Options include 'holography' and 'ctf'.
        Default is 'holography'.
    noise_model : str, optional
        Specifies noise model. Currently only 'poisson' available. Default is None
        (no noise applied).
    mtf : torch.Tensor, optional
        Modulation transfer function in Fourier space to apply to images.
        Default is None (no MTF applied).
    num_frames : int, optional
        Number of frames for dose-fractionated noise.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    """

    def __init__(
        self,
        pixel_size: float,
        aberration_model: str = "holography",
        noise_model: str | None = None,
        mtf: torch.Tensor | None = None,
        num_frames: int | None = None,
        progressbars: bool = True,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.register_buffer("mtf", mtf)
        self.num_frames = num_frames
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
            Dose per image in e-/Å², shape (B,). Used for CTF model scaling.

        Returns
        -------
        images : torch.Tensor
            Detector image. For holography model, returns intensity (squared
            magnitude). For CTF model, returns dose-scaled image.
        """
        dose_per_pixel = dose * self.pixel_size**2  # (B,)
        if self.aberration_model == "holography":
            images = dose_per_pixel[:, None, None] * torch.abs(aberrated_exitwave) ** 2
        elif self.aberration_model == "ctf":
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
            Dose per image in e-/Å², shape (B,).
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
        pixel_size_angstrom: float,
        dose_per_angstrom_sq_per_frame: float,
        coinc_radius_pixels: float = 1.5,
    ) -> torch.Tensor:
        """
        Simulates a single frame of a Direct Electron Detector (DED).

        Parameters
        ----------
        intensity_map : torch.Tensor
            2D Tensor (normalized psi^2 from multislice).
        pixel_size_angstrom : float
            Size of one pixel in Angstroms (e.g., 1.0).
        dose_per_angstrom_sq_per_frame : float
            Physical dose (e/A^2).
        coinc_radius_pixels : float, optional
            Dead-time radius in pixels.
        """
        det_h, det_w = intensity_map.shape
        device = intensity_map.device

        # 1. Convert physical dose to expected electrons in this frame
        dose_per_pixel = dose_per_angstrom_sq_per_frame * (pixel_size_angstrom**2)
        total_expected_electrons = dose_per_pixel * (det_h * det_w)

        # 2. Poisson number of incident electrons
        n_e = int(
            torch.poisson(torch.tensor(total_expected_electrons, device=device)).item()
        )
        if n_e <= 0:
            return torch.zeros_like(intensity_map)

        # 3. Sample landing coordinates from intensity_map distribution
        prob_dist = intensity_map.reshape(-1)
        prob_dist = prob_dist / prob_dist.sum()
        indices = torch.multinomial(prob_dist, n_e, replacement=True)

        iy = (indices // det_w).float()
        ix = (indices % det_w).float()
        coords = torch.stack([ix, iy], dim=1)
        coords += torch.rand((n_e, 2), device=device)

        # 4. Coincidence suppression (greedy, as requested)
        keep = torch.ones(n_e, dtype=torch.bool, device=device)
        r_sq = float(coinc_radius_pixels) * float(coinc_radius_pixels)
        for i in range(n_e):
            if not keep[i]:
                continue
            if i + 1 >= n_e:
                break
            d_sq = torch.sum((coords[i + 1 :] - coords[i]) ** 2, dim=1)
            close = d_sq < r_sq
            keep[i + 1 :][close] = False

        coords = coords[keep]

        # 5. Bin into detector pixels
        pixels = torch.zeros_like(intensity_map)
        ix_f = coords[:, 0].long().clamp(0, det_w - 1)
        iy_f = coords[:, 1].long().clamp(0, det_h - 1)
        pixels.index_put_(
            (iy_f, ix_f),
            torch.ones_like(ix_f, dtype=intensity_map.dtype),
            accumulate=True,
        )
        return pixels

    def apply_detector_physics_fast(
        self,
        intensity_map: torch.Tensor,
        pixel_size_angstrom: float,
        dose_per_angstrom_sq_per_frame: float,
        coinc_radius_pixels: float = 1.5,
    ) -> torch.Tensor:
        device = intensity_map.device

        # 0. Pad map to avoid edge artifacts
        pad = int(np.ceil(coinc_radius_pixels * 2))
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
        dose_per_pixel = dose_per_angstrom_sq_per_frame * (pixel_size_angstrom**2)
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
        base_cell_size = coinc_radius_pixels / (2**0.5)
        cell_size = base_cell_size * (1 + 0.05 * torch.randn(1, device=device)).clamp(
            0.8, 1.2
        )

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
            Total dose for this image in e-/Å².
        coincidence_radius : float
            Coincidence radius in pixels. If <= 0, plain Poisson noise is applied.

        Returns
        -------
        final_image : torch.Tensor
            Simulated image after dose-fractionated noise and coincidence.
        """
        if self.noise_model != "poisson":
            return img

        if coincidence_radius <= 0.0:
            return torch.poisson(torch.clamp(img, min=0.0))

        if self.num_frames is None:
            raise ValueError(
                "num_frames must be set on the Detector to apply dose-fractionated "
                "coincidence loss (coincidence_radius > 0)."
            )
        n_frames = self.num_frames
        intensity_map = img / img.sum()

        # Use img.sum() to derive the effective dose so that the radius>0 path
        # agrees with torch.poisson(img) at radius=0: both correctly account
        # for real electron absorption from alpha (imaginary potential), and
        # B-factor attenuation is treated consistently across both paths.
        dose_effective = (
            img.sum() / (self.pixel_size**2 * img.shape[0] * img.shape[1])
        ).item()

        final_image = torch.zeros_like(img)
        for _ in track(
            range(n_frames),
            description="Applying coincidence loss",
            transient=True,
            disable=not (self.progressbars),
        ):
            final_image += self.apply_detector_physics_fast(
                intensity_map,
                self.pixel_size,
                dose_effective / n_frames,
                coinc_radius_pixels=coincidence_radius,
            )

        return final_image
