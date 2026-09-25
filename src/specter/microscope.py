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
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from .fft import fft2, ifft2
from .progress import track
from specter.options import AberrationModel, NoiseModel


# Bound the dense coincidence lookup to 8 MiB and, below, four cells per
# arriving electron. Sparse frames and tiny radii retain the sorting path.
_COINCIDENCE_MAX_DENSE_CELLS = 2**20


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
        dose_weights: torch.Tensor | None = None,
        dose_weights_max_frequency: float | None = None,
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
        if dose_weights is not None:
            # The weights' first dimension IS the motion-correction job's own
            # fractionation, so it is data and takes precedence over a config
            # value the same way a .cs file's pixel size, voltage and amplitude
            # contrast do. `n_frames` otherwise defaults to 40, which is a
            # convention rather than anything read from the movie -- EER stores
            # ~2100 hardware frames and the grouping is a processing choice.
            #
            # Getting this wrong is silent, not loud: `apply_coincidence` loops
            # over n_frames and indexes weights[i] while `_frame_weight_grids`
            # normalises by the weights' own count, so a mismatch applies a
            # prefix of the weights under the wrong normalisation and the
            # filter stops preserving the signal.
            weight_frames = int(dose_weights.shape[0])
            if n_frames is not None and weight_frames != int(n_frames):
                warnings.warn(
                    f"dose_weights carries {weight_frames} frames but "
                    f"n_frames={n_frames}; using {weight_frames}, since the "
                    "weights record the motion-correction job's own "
                    "fractionation. Set n_frames to match to silence this.",
                    stacklevel=2,
                )
            self.n_frames = weight_frames
        self.register_buffer("dose_weights", dose_weights, persistent=False)
        self.dose_weights_max_frequency = dose_weights_max_frequency
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
        return self.detect_expected_counts(
            images, dose, coincidence_radius, anisomag, nxy
        )

    def from_intensity(
        self,
        intensity: torch.Tensor,
        dose: torch.Tensor,
        coincidence_radius: torch.Tensor,
        anisomag: torch.Tensor | None = None,
        nxy: int | None = None,
    ) -> torch.Tensor:
        """Detect an incoherent, dose-averaged intensity (vacuum intensity = 1).

        Used after integrating frozen configurations. Does not take a square
        root or invent a coherent wave. DQE, pixel area and dose enter once.
        """
        if intensity.is_complex() or intensity.ndim != 3:
            raise ValueError("intensity must be real with shape (B,Y,X)")
        if not torch.isfinite(intensity).all() or (intensity < 0).any():
            raise ValueError("intensity must be finite and nonnegative")
        images = intensity * (dose * self.pixel_size**2 * self.dqe0)[:, None, None]
        return self.detect_expected_counts(
            images, dose, coincidence_radius, anisomag, nxy
        )

    def detect_expected_counts(
        self,
        images: torch.Tensor,
        dose: torch.Tensor,
        coincidence_radius: torch.Tensor,
        anisomag: torch.Tensor | None = None,
        nxy: int | None = None,
    ) -> torch.Tensor:
        """Apply spatial detector response and sampling to expected counts.

        Input counts already include dose, pixel area and detection efficiency.
        Noise is drawn only here, after configuration integration.
        """
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

        # Apply detector MTF -- deliberately BEFORE the Poisson draw below, which is
        # what makes this a COUNTING detector. `apply_coincidence` samples arrivals and
        # deposits one count per surviving electron into one pixel, so the recorded
        # noise is white however wide the MTF is: a Poisson point process whose points
        # are independently displaced is still a Poisson point process. The blur lives
        # in the ensemble mean (hence MTF < 1) and not in any single event. That is the
        # right model for K2/K3/Falcon counting mode -- every MTF bundled in
        # detectors.py is a counting-mode curve -- and it is the published behaviour:
        # a counting detector's NPS is flat "due to the counting mode which assigns
        # detected electrons to single pixels" (Ruskin, Yu & Grigorieff 2013,
        # J. Struct. Biol. 184, 385-393).
        #
        # Do not "fix" this ordering. Drawing the noise first and blurring afterwards
        # models an INTEGRATING sensor, where one electron's charge cloud really is
        # split between neighbouring pixels, so the noise carries the kernel and NPS
        # becomes proportional to MTF^2 (ibid., Eq. 11). That is a different camera,
        # not a bug fix, and it is only self-consistent alongside the Landau/Swank
        # spread of deposited charge and the removal of coincidence loss, which is a
        # failure of the event finder and has no meaning without one.
        if self.mtf is not None:
            images = self.add_mtf(images, self.mtf)

        if self.noise_model is None:
            return images
        # One host transfer per argument for the whole batch rather than two
        # `.item()` syncs per image, and the frame-weight grids built once:
        # every image in the batch shares their shape and device.
        radii = coincidence_radius.tolist()
        weights = (
            self._frame_weight_grids(
                (int(images.shape[-2]), int(images.shape[-1])), images.device
            )
            if self.noise_model == "poisson"
            else None
        )
        return torch.stack(
            [self._apply_coincidence(img, r, weights) for img, r in zip(images, radii)]
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
        ix = ix.repeat_interleave(n_per_pixel, output_size=n_e)
        iy = iy.repeat_interleave(n_per_pixel, output_size=n_e)

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

        # The random shift moves coordinates by at most half a cell, so
        # cell indices start at -1. Bound the other end using the smallest
        # allowed cell width; this avoids a GPU scalar read to size the lookup.
        grid_w = math.ceil(det_w / (cell_size_nominal * 0.8)) + 2
        grid_h = math.ceil(det_h / (cell_size_nominal * 0.8)) + 2
        n_cells = grid_w * grid_h
        perm = torch.randperm(n_e, device=device)
        if n_cells <= min(_COINCIDENCE_MAX_DENSE_CELLS, 4 * n_e):
            cell_id = (cell_y + 1) * grid_w + (cell_x + 1)
            permuted_cells = cell_id[perm]
            order = torch.arange(n_e, device=device)
            first = torch.full((n_cells,), n_e, dtype=torch.long, device=device)
            first.scatter_reduce_(
                0, permuted_cells, order, reduce="amin", include_self=True
            )
            # The minimum shuffled rank is exactly the first entry a stable
            # sort by cell would retain. Deposit zero for rejected arrivals
            # to avoid a dynamic boolean gather and its CUDA synchronization.
            weights = (first[permuted_cells] == order).to(intensity_map.dtype)
            coords_kept = coords[perm]
        else:
            # Memory scales with arrivals here, even when tiny radii imply
            # billions of mostly empty cells. Keep the original stable sort.
            cx_min, cy_min = cell_x.min(), cell_y.min()
            occupied_grid_w = (cell_x.max() - cx_min) + 1
            cell_id = (cell_y - cy_min) * occupied_grid_w + (cell_x - cx_min)
            sort_idx = torch.argsort(cell_id[perm], stable=True)
            sorted_cell_id = cell_id[perm][sort_idx]
            is_first = torch.ones(n_e, dtype=torch.bool, device=device)
            is_first[1:] = sorted_cell_id[1:] != sorted_cell_id[:-1]
            coords_kept = coords[perm][sort_idx][is_first]
            weights = torch.ones(
                len(coords_kept), dtype=intensity_map.dtype, device=device
            )

        # 5. Bin into detector pixels
        ix_f = coords_kept[:, 0].long().clamp(0, det_w - 1)
        iy_f = coords_kept[:, 1].long().clamp(0, det_h - 1)
        flat_idx = iy_f * det_w + ix_f
        pixels = torch.zeros(det_h * det_w, dtype=intensity_map.dtype, device=device)
        pixels.scatter_add_(0, flat_idx, weights)

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
            Total dose for this image in e⁻/Å². Unused: the expected counts
            per frame are derived from ``img`` itself, which already carries
            the dose. Kept for the callers that pass it positionally.
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
        side is ``r * sqrt(pi)``, so the cell area, and with it the effective
        exclusion area, is ``pi * r**2`` (see
        :meth:`apply_detector_physics`).
        """
        if self.noise_model != "poisson":
            return img
        weights = self._frame_weight_grids(
            (int(img.shape[-2]), int(img.shape[-1])), img.device
        )
        return self._apply_coincidence(img, coincidence_radius, weights)

    def _apply_coincidence(
        self,
        img: torch.Tensor,
        coincidence_radius: float,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Body of :meth:`apply_coincidence`, with the frame-weight grids given.

        Parameters
        ----------
        img : torch.Tensor
            Total-dose image, shape (H, W).
        coincidence_radius : float
            Coincidence exclusion radius in pixels.
        weights : torch.Tensor or None
            ``self._frame_weight_grids(img.shape, img.device)``, hoisted so a
            batch builds it once rather than once per image.

        Returns
        -------
        torch.Tensor
            Simulated image after dose-fractionated noise and coincidence.
        """
        if self.noise_model != "poisson":
            return img

        if coincidence_radius <= 0.0 and self.dose_weights is None:
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
        # Both scalars in one host transfer. The effective dose is divided on
        # the device, in the image's own precision, exactly as before.
        img_sum_host, dose_effective = torch.stack(
            [img_sum, img_sum / (self.pixel_size**2 * img.shape[0] * img.shape[1])]
        ).tolist()
        if img_sum_host <= 0.0:
            # No expected electrons anywhere (e.g. an all-zero specimen
            # volume -- can happen with scattering_model="ctf", which has no
            # vacuum baseline, unlike multislice's exp(i*sigma*dz*V) == 1 at
            # V=0). img / img_sum would be 0/0 = NaN here; the physically
            # correct output for zero expected signal is a blank frame.
            return torch.zeros_like(img)
        intensity_map = img / img_sum

        # dose_effective above is derived from img_sum so that the radius>0
        # path agrees with torch.poisson(img) at radius=0: both correctly
        # account for real electron absorption from alpha (imaginary
        # potential), and B-factor attenuation is treated consistently across
        # both paths.

        final_image = torch.zeros_like(img)
        accum_k = None
        for i in track(
            range(n_frames),
            description="Applying coincidence loss",
            transient=True,
            disable=not (self.progressbars),
        ):
            frame = self.apply_detector_physics(
                intensity_map,
                self.pixel_size,
                dose_effective / n_frames,
                coinc_radius_pixels=coincidence_radius,
            )
            if weights is None:
                final_image += frame
            else:
                # Sum the frames in Fourier space under the exposure filter's
                # own per-frequency weights, rather than adding them equally.
                # Every frame carries the same expected signal, so weights
                # normalised to sum to n_frames leave the signal untouched and
                # multiply the noise POWER by sum(w^2)/n_frames -- the rising
                # floor a signal-preserving dose weighting leaves behind, and
                # the one thing `dose_envelope` cannot express, since it
                # attenuates the signal and leaves the noise white.
                f = torch.fft.rfft2(frame)
                accum_k = (
                    f * weights[i] if accum_k is None else accum_k + f * weights[i]
                )
        if accum_k is not None:
            final_image = torch.fft.irfft2(accum_k, s=tuple(img.shape[-2:]))

        return final_image

    def _frame_weight_grids(
        self, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor | None:
        """
        Per-frame, per-frequency weights on this image's rfft2 grid.

        `dose_weights` is stored as ``(n_frames, n_bins)`` over a radial axis
        running to the Nyquist **of the movie it was computed on**, which is
        not in general the Nyquist of the particles. Super-resolution and EER
        movies are the usual case: EMPIAR-11377's Falcon 4i weights span twice
        the Nyquist of its 0.731 A/px particles, so reading them as if the
        axis ended at the particles' own Nyquist stretches the steep half of
        the curve across the whole measurable band and overstates the noise
        gain threefold (6.76x against a measured 2.22x at 0.9 Nyquist).

        `dose_weights_max_frequency` is the frequency its last bin sits at,
        in 1/Angstrom, so the mapping is done in absolute frequency and a
        stack downsampled or Fourier-cropped after motion correction still
        gets the weights that apply at the frequencies it kept.
        :func:`~specter.io.load_dose_weights` derives it from the job's own
        files. Without it the axis is assumed to end at the image's Nyquist,
        which is right only by coincidence.

        Parameters
        ----------
        shape : tuple of int
            Image shape ``(Y, X)``.
        device : torch.device
            Device for the returned grids.

        Returns
        -------
        torch.Tensor or None
            Shape ``(n_frames, Y, X // 2 + 1)``, normalised so the weights at
            each frequency sum to ``n_frames``; None when no weights are set.
        """
        if self.dose_weights is None:
            return None
        w = self.dose_weights.to(device).float()
        n_frames, n_bins = w.shape
        ky = torch.fft.fftfreq(shape[0], device=device)
        kx = torch.fft.rfftfreq(shape[1], device=device)
        # Fraction of the WEIGHTS' Nyquist, which is what the radial axis
        # indexes -- not the image's, when the two differ.
        # Absolute frequency, in 1/Angstrom, mapped onto the weights' own
        # axis. Keying on frequency rather than on a fraction of somebody's
        # Nyquist is what makes a later crop or downsample harmless.
        k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) / self.pixel_size
        max_freq = self.dose_weights_max_frequency
        if max_freq is None:
            max_freq = 1.0 / (2.0 * self.pixel_size)
        r = k / max_freq
        idx = torch.clamp((r * (n_bins - 1)).round().long(), 0, n_bins - 1)
        grids = w[:, idx.reshape(-1)].reshape(n_frames, *idx.shape)
        total = grids.sum(dim=0, keepdim=True).clamp(min=1e-12)
        return grids * (n_frames / total)
