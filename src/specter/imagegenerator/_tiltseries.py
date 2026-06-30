from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from ..progress import track

from .. import rotations
from ._micrograph import MicrographGenerator
from ..rotations import Rotation
from ..scattering import IterativeScattering


class TiltSeriesGenerator(MicrographGenerator):
    """
    Generates tilt series images by tilting the specimen volume.

    Accepts either a list of tilt angles or explicit quaternions/translations.
    The volume is rotated for each tilt via an affine transformation passed to
    ``IterativeScattering``.

    Parameters
    ----------
    vol : torch.Tensor
        Pre-assembled specimen volume of shape (1, Z, Y, X).
    micrograph_size : int or tuple[int, int]
        Output image size in pixels (must be square).
    pixel_size : float
        Pixel size in Å.
    ctf_params : dict[str, torch.Tensor]
        CTF parameters; each value is a 1-D tensor of length n.
    energy : float
        Electron beam energy in kV.
    dose_per_angstrom : float or torch.Tensor
        Electron dose per Å². Scalar or 1-D tensor of length n.
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
    scattering_model : str, optional
        Scattering model passed to ``IterativeScattering``. Default 'multislice'.
    aberration_model : str, optional
        Aberration model. Default 'holography'.
    noise_model : str, optional
        Noise model. Default 'poisson'.
    klim : float, optional
        Reciprocal-space frequency limit.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.0.
    pad_fft : bool, optional
        Whether to XY-pad the volume for FFT antialiasing. Default False.
    chunk_size : int, optional
        Chunk size for ``TomogramGenerator`` (unused here but passed to parent).
    move_to_cpu : bool, optional
        Move volume to CPU after setup. Default False (tries GPU first).
    detector_model : str, optional
        Detector MTF model ('k3_300kv', 'k3_200kv', 'perfect', None).
    progressbars : bool, optional
        Show progress bars. Default True.
    verbose : bool, optional
        Emit debug-level log messages. Default True.
    slice_batch_size : int, optional
        Number of Z slices propagated together. Default 1.
    pad_volume : bool, optional
        Automatically pad volume in XY when it is too small for the requested
        tilt coverage (reflect padding). Default True.
    taper_width : int, optional
        Extra reflect-padded apron pixels on each XY side beyond tilt-coverage
        padding, with a cosine taper applied. Default 0.
    z_taper_width : int, optional
        Cosine taper width in Z pixels at the top and bottom of the volume.
        Default 0.
    tilt_axis : str, optional
        Axis around which the sample tilts ('x' or 'y'). Default 'x'.
    coincidence_radius : float or torch.Tensor, optional
        Coincidence radius in pixels. Default 0.0.
    num_frames : int, optional
        Number of detector frames to simulate. Default None.
    bfactor_envelope : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    convergence_angle : float, optional
        Beam convergence semi-angle in milliradians, used for the Cs
        (spatial coherence) envelope. Default None (envelope disabled).
    cc : float, optional
        Chromatic aberration coefficient in Angstrom, used for the Cc
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
    # Static helpers for geometry calculations                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_required_nxy(
        desired_nxy: int, nz: int, max_tilt_angle_deg: float
    ) -> int:
        """Minimum XY size so the tilted projection still covers ``desired_nxy`` pixels."""
        theta_rad = torch.deg2rad(max_tilt_angle_deg)
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)
        return int(torch.ceil((desired_nxy + nz * sin_t) / cos_t))

    @staticmethod
    def _estimate_max_allowed_nxy(
        available_nxy: int, nz: int, max_tilt_angle_deg: float
    ) -> int:
        """Maximum output XY achievable given the available volume at this tilt."""
        theta_rad = torch.deg2rad(max_tilt_angle_deg)
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)
        return int(torch.ceil(available_nxy * cos_t - nz * sin_t))

    @staticmethod
    def _estimate_max_allowed_tilt_deg(
        desired_nxy: int, nz: int, available_nxy: int
    ) -> float:
        """Largest tilt (degrees) such that the available volume still covers ``desired_nxy``."""
        thetas_deg = torch.linspace(0.0, 89.9, 4000)
        thetas_rad = torch.deg2rad(thetas_deg)
        spans = available_nxy * torch.cos(thetas_rad) - nz * torch.sin(thetas_rad)
        valid = spans >= desired_nxy
        if not bool(valid.any()):
            return 0.0
        return float(thetas_deg[valid][-1].item())

    @staticmethod
    def _infer_max_tilt_from_inputs(angles=None, quaternions=None) -> float:
        """Infer max tilt magnitude in degrees from provided poses."""
        if angles is not None:
            return angles.abs().max()
        if quaternions is not None:
            rotvecs = Rotation.from_quat(torch.as_tensor(quaternions)).as_rotvec()
            max_angle_rad = torch.linalg.norm(rotvecs, dim=-1).max()
            return max_angle_rad * (180.0 / torch.pi)
        return torch.tensor([0.0])

    @staticmethod
    def _pad_vol_xy_for_tilt(
        vol: torch.Tensor, required_nxy: int, available_nxy: int
    ) -> torch.Tensor:
        """
        Pad ``vol`` symmetrically in XY using reflect mode to reach ``required_nxy``.

        Parameters
        ----------
        vol : torch.Tensor
            Volume of shape (..., Z, Y, X).
        required_nxy : int
            Target XY size after padding.
        available_nxy : int
            Current XY extent of ``vol``.

        Returns
        -------
        vol : torch.Tensor
            Reflect-padded volume.
        """
        pad_each_side = (required_nxy - available_nxy + 1) // 2
        return F.pad(
            vol,
            (pad_each_side, pad_each_side, pad_each_side, pad_each_side, 0, 0),
            mode="reflect",
        )

    @staticmethod
    def _get_cosine_window(n: int, taper_px: int, device, dtype) -> torch.Tensor:
        """1-D cosine window of length ``n`` with ``taper_px``-wide fade at each end."""
        win = torch.ones(n, device=device, dtype=dtype)
        taper_px = min(taper_px, n // 2)
        if taper_px <= 0:
            return win
        ramp = 0.5 * (
            1
            - torch.cos(
                torch.pi * torch.linspace(0, 1, taper_px, device=device, dtype=dtype)
            )
        )
        win[:taper_px] = ramp
        win[-taper_px:] = ramp.flip(0)
        return win

    @staticmethod
    def _apply_cosine_taper(
        vol: torch.Tensor, taper_xy: int = 0, taper_z: int = 0
    ) -> torch.Tensor:
        """
        Apply a cosine taper to the XY and/or Z edges of the volume.

        Parameters
        ----------
        vol : torch.Tensor
            Volume of shape (..., Z, Y, X).
        taper_xy : int
            Taper width in XY pixels. 0 to skip.
        taper_z : int
            Taper width in Z pixels. 0 to skip.

        Returns
        -------
        vol : torch.Tensor
            Volume with taper applied.
        """
        if taper_xy <= 0 and taper_z <= 0:
            return vol

        nz, ny, nx = vol.shape[-3], vol.shape[-2], vol.shape[-1]
        device, dtype = vol.device, vol.dtype
        mask = torch.ones(1, device=device, dtype=dtype)

        if taper_xy > 0:
            win_y = TiltSeriesGenerator._get_cosine_window(ny, taper_xy, device, dtype)
            win_x = TiltSeriesGenerator._get_cosine_window(nx, taper_xy, device, dtype)
            mask = mask * win_y[:, None] * win_x[None, :]

        if taper_z > 0:
            win_z = TiltSeriesGenerator._get_cosine_window(nz, taper_z, device, dtype)
            mask = (
                win_z[:, None, None] * mask if mask.ndim == 2 else win_z[:, None, None]
            )

        return vol * mask

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        vol: torch.Tensor,
        micrograph_size: int | tuple[int, int],
        pixel_size: float,
        ctf_params: dict[str, Any],
        energy: float,
        dose_per_angstrom: float | torch.Tensor,
        quaternions: torch.Tensor | None = None,
        translations: torch.Tensor | None = None,
        angles: torch.Tensor | Sequence[float] | None = None,
        anisomag: torch.Tensor | None = None,
        scattering_model: str = "multislice",
        aberration_model: str = "holography",
        noise_model: str = "poisson",
        klim: float | None = None,
        alpha: float = 0.0,
        pad_fft: bool = False,
        chunk_size: int | None = None,
        move_to_cpu: bool = False,
        detector_model: str | None = None,
        progressbars: bool = True,
        verbose: bool = True,
        slice_batch_size: int = 1,
        pad_volume: bool = True,
        taper_width: int = 0,
        z_taper_width: int = 0,
        tilt_axis: str = "x",
        coincidence_radius: float | torch.Tensor = 0.0,
        num_frames: int | None = None,
        bfactor_envelope: float | torch.Tensor | None = None,
        convergence_angle: float | None = None,
        cc: float | None = None,
        energy_spread: float = 0.7,
        deltaV_V: float = 0.06e-6,
        deltaI_I: float = 0.01e-6,
        dose_envelope: bool = False,
        **kwargs: Any,
    ):
        if vol is None:
            raise ValueError("'vol' must be provided for TiltSeriesGenerator.")

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

        max_tilt_angle_deg = self._infer_max_tilt_from_inputs(
            angles=angles, quaternions=quaternions
        )

        nz_input = int(vol.shape[-3])
        available_nxy = int(min(vol.shape[-2], vol.shape[-1]))
        required_nxy = self._estimate_required_nxy(
            desired_nxy=desired_nxy,
            nz=nz_input,
            max_tilt_angle_deg=max_tilt_angle_deg,
        )
        target_nxy = required_nxy + 2 * taper_width
        self.recommended_nxy_for_max_tilt = required_nxy
        self.max_tilt_angle_deg = float(max_tilt_angle_deg)
        self.max_allowed_tilt_deg_for_volume = self._estimate_max_allowed_tilt_deg(
            desired_nxy=desired_nxy, nz=nz_input, available_nxy=available_nxy
        )
        self.max_allowed_nxy = self._estimate_max_allowed_nxy(
            available_nxy=available_nxy,
            nz=nz_input,
            max_tilt_angle_deg=max_tilt_angle_deg,
        )

        if available_nxy < target_nxy:
            if pad_volume:
                vol = self._pad_vol_xy_for_tilt(vol, target_nxy, available_nxy)
                msg = (
                    "[TiltSeriesGenerator] Volume XY too small for requested tilt coverage"
                    + (" and taper" if taper_width > 0 else "")
                    + f"; padded (reflect) from {available_nxy} to {vol.shape[-1]} px in XY.\n"
                    f"  micrograph_size={desired_nxy}, requested_max_tilt={self.max_tilt_angle_deg:.2f} deg, "
                    f"required_volume_nxy>={required_nxy}"
                )
                if taper_width > 0:
                    msg += f", target_nxy (with taper)>={target_nxy}"
                print(msg + ".")
            else:
                print(
                    "[TiltSeriesGenerator] Input volume XY may be too small for requested tilt "
                    "coverage; proceeding anyway (pad_volume=False).\n"
                    f"  micrograph_size={desired_nxy}, volume_shape={tuple(vol.shape)}, "
                    f"requested_max_tilt={self.max_tilt_angle_deg:.2f} deg,\n"
                    f"  required_volume_nxy>={required_nxy}, current_volume_nxy={available_nxy}, \n"
                    f"  max_allowed_tilt_with_current_volume\u2248{self.max_allowed_tilt_deg_for_volume:.2f} deg,\n"
                    f"  max_allowed_nxy\u2248{self.max_allowed_nxy}."
                )

        if taper_width > 0 or z_taper_width > 0:
            vol = self._apply_cosine_taper(
                vol, taper_xy=int(taper_width), taper_z=int(z_taper_width)
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
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            vol=vol,
            anisomag=anisomag,
            scattering_model=scattering_model,
            aberration_model=aberration_model,
            noise_model=noise_model,
            klim=klim,
            alpha=alpha,
            pad_fft=pad_fft,
            chunk_size=chunk_size,
            move_to_cpu=move_to_cpu,
            detector_model=detector_model,
            progressbars=progressbars,
            verbose=verbose,
            slice_batch_size=slice_batch_size,
            coincidence_radius=coincidence_radius,
            num_frames=num_frames,
            bfactor_envelope=bfactor_envelope,
            convergence_angle=convergence_angle,
            cc=cc,
            energy_spread=energy_spread,
            deltaV_V=deltaV_V,
            deltaI_I=deltaI_I,
            dose_envelope=dose_envelope,
            **kwargs,
        )
        # self.register_buffer("vol", vol)

        self.slice_batch_size = slice_batch_size
        self.iterative_scattering = IterativeScattering(
            desired_nxy,
            pixel_size,
            energy,
            scattering_model=scattering_model,
            klim=klim,
            alpha=alpha,
            progressbars=progressbars,
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

            quats = Rotation.from_rotvec(rotvecs).as_quat()
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

    def get_nz_tilt(self, V: torch.Tensor, theta_matrix: torch.Tensor) -> int:
        """
        Number of Z slices needed to fully cover the tilted volume.

        Parameters
        ----------
        V : torch.Tensor
            Volume of shape (B, Z, Y, X).
        theta_matrix : torch.Tensor
            Affine transformation matrix of shape (B, 3, 4) or (B, 4, 4).

        Returns
        -------
        nz_new : int
            Number of slices.
        """
        B, Z, Y, X = V.shape
        device = V.device
        theta_matrix = theta_matrix.to(device)
        R = theta_matrix[:, :3, :3]

        corners = torch.tensor(
            [
                [-X / 2, -Y / 2, -Z / 2],
                [X / 2, -Y / 2, -Z / 2],
                [-X / 2, Y / 2, -Z / 2],
                [X / 2, Y / 2, -Z / 2],
                [-X / 2, -Y / 2, Z / 2],
                [X / 2, -Y / 2, Z / 2],
                [-X / 2, Y / 2, Z / 2],
                [X / 2, Y / 2, Z / 2],
            ],
            device=device,
            dtype=V.dtype,
        ).t()  # (3, 8)

        rotated_corners = torch.bmm(
            R.transpose(1, 2), corners.unsqueeze(0).expand(B, -1, -1)
        )
        z_min = rotated_corners[:, 2, :].min(dim=1).values
        z_max = rotated_corners[:, 2, :].max(dim=1).values
        nz_new = int(torch.ceil((z_max - z_min).max()).item())
        return max(1, nz_new)

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
            |detector_waves|² before noise, shape (B, N_tilts, Y, X).
        """

        tilt_series = []
        exitwaves = []
        clean_images = []
        B = len(idx)
        n_frames = len(self.quaternions)

        for i in track(
            range(n_frames),
            description="Generating tilt series.",
            disable=not self.progressbars,
        ):
            Q = self.quaternions[i].unsqueeze(0).expand(B, -1)
            T = self.translations[i].unsqueeze(0).expand(B, -1)

            R_mat = Rotation.from_quat(Q).as_matrix()
            if R_mat.ndim == 2:
                R_mat = R_mat.unsqueeze(0)
            T_torch = rotations.translations_angstrom_to_torch(
                T, self.vol.shape[-1], self.pixel_size
            )
            theta_matrix = rotations.build_affine_matrix(R_mat, T_torch)

            exitwave = self.iterative_scattering(
                self.vol, theta_matrix, slice_batch_size=self.slice_batch_size
            )

            ctf_batch = self._ctf_batch(idx)
            if self.scattering_model not in ["projection", "ctf"]:
                nz_new = self.get_nz_tilt(self.vol, theta_matrix)
                z_offset = (nz_new - self.nz) * self.pixel_size / 2.0
                if "dfu" in ctf_batch:
                    ctf_batch["dfu"] = ctf_batch["dfu"] - z_offset
                if "dfv" in ctf_batch:
                    ctf_batch["dfv"] = ctf_batch["dfv"] - z_offset

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
