from __future__ import annotations

from typing import Sequence

import lightning as L
import torch
import torch.nn.functional as F

from ..fft import fft3, ifft3


def _normalize_slice_indices(
    slice_indices: torch.Tensor | Sequence[int] | int, device: str | torch.device
) -> torch.Tensor:
    """Coerce `slice_indices` to a 1D tensor of shape (K,)."""
    if not isinstance(slice_indices, torch.Tensor):
        slice_indices = torch.as_tensor(slice_indices, device=device)
    if slice_indices.ndim == 0:
        slice_indices = slice_indices.unsqueeze(0)
    return slice_indices


def _resolve_roi(
    roi_center: tuple[int, int] | None,
    roi_size: tuple[int, int] | None,
    ny: int,
    nx: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Fill in ROI center/size defaults (full volume, centered) where unset."""
    if roi_size is None:
        roi_size = (ny, nx)
    if roi_center is None:
        roi_center = (ny // 2, nx // 2)
    return roi_center, roi_size


def _build_roi_query_points(
    slice_indices: torch.Tensor,
    roi_center: tuple[int, int],
    roi_size: tuple[int, int],
    ny: int,
    nx: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build (x, y, z) pixel-unit query points, relative to the volume center,
    for every (slice, ROI pixel) combination.

    Returns
    -------
    torch.Tensor
        Shape (K, ny_roi, nx_roi, 3), last dim is (x, y, z).
    """
    ny_roi, nx_roi = roi_size
    cy, cx = roi_center
    K = slice_indices.shape[0]

    z_rel = slice_indices.to(dtype=dtype, device=device)  # (K,)
    y_rel_center = float(cy - ny // 2)
    x_rel_center = float(cx - nx // 2)

    y_rel_grid = torch.arange(ny_roi, device=device, dtype=dtype) - (ny_roi // 2)
    x_rel_grid = torch.arange(nx_roi, device=device, dtype=dtype) - (nx_roi // 2)
    yy_rel, xx_rel = torch.meshgrid(y_rel_grid, x_rel_grid, indexing="ij")

    x_pix = (x_rel_center + xx_rel).unsqueeze(0).expand(K, -1, -1)
    y_pix = (y_rel_center + yy_rel).unsqueeze(0).expand(K, -1, -1)
    z_pix = z_rel.view(K, 1, 1).expand(-1, ny_roi, nx_roi)

    return torch.stack([x_pix, y_pix, z_pix], dim=-1)  # (K, ny_roi, nx_roi, 3)


def _prepare_volume_for_grid_sample(V: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize a (Z,Y,X) / (B,Z,Y,X) / (B,1,Z,Y,X) volume to (B,1,Z,Y,X)."""
    if V.ndim == 3:
        return V.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    elif V.ndim == 4:  # (B, Z, Y, X)
        return V.unsqueeze(1)
    else:  # (B, 1, Z, Y, X)
        return V


class VolumeRotator(L.LightningModule):
    """
    3D volume rotator with cached base grid and RELION / PyTorch center conventions.
    """

    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        origin: str = "relion",
        align_corners: bool = False,
        padding_mode: str = "border",
        mode: str = "real",
        init_base_grid=True,
    ):
        """
        Initialize a 3D VolumeRotator with cached base grid and rotation center.

        This class supports rotating 3D volumes in either real space or Fourier space.
        The base grid and rotation center are cached as buffers for efficiency.

        Parameters
        ----------
        nz, ny, nx : int
            Dimensions of the volume along the Z, Y, and X axes.
        origin : str, optional
            Convention for the rotation origin. Options:
            - "relion": sets the origin at [nz//2, ny//2, nx//2] following RELION convention.
            - "center": sets the origin at the center of the volume using PyTorch convention.
            Default is "relion".
        align_corners : bool, optional
            Passed to `torch.nn.functional.affine_grid` and `grid_sample`. Determines
            how the normalized coordinates are aligned with the voxel corners.
            Default is False.
        padding_mode : str, optional
            Passed to `torch.nn.functional.grid_sample`. Options include "zeros", "border", or "reflection".
            Default is "border".
        mode : str, optional
            Default rotation mode for `forward`. Options:
            - "real": rotate in real space.
            - "fourier": rotate in Fourier space.
            Default is "real".

        Raises
        ------
        ValueError
            If `origin` is not one of ("relion", "center") or `mode` is not one of ("real", "fourier").

        Attributes
        ----------
        base_grid : torch.Tensor (buffer)
            Identity sampling grid for the volume, used to construct per-batch grids.
        center : torch.Tensor (buffer)
            The voxel coordinates of the rotation center.
        """
        super().__init__()

        # Validate inputs
        if origin not in ("relion", "center"):
            raise ValueError(f"Unknown origin: {origin}. Must be 'relion' or 'center'.")
        if mode not in ("real", "fourier"):
            raise ValueError(f"Unknown mode: {mode}. Must be 'real' or 'fourier'.")

        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.origin = origin
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.mode = mode  # default rotation mode

        # -------------------------------
        # Rotation center
        # -------------------------------
        # Calculate center based on dimensions and conventions
        if origin == "relion":
            cz, cy, cx = nz // 2, ny // 2, nx // 2
        else:
            cz, cy, cx = (nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2

        if self.align_corners:
            center = torch.tensor(
                [
                    2 * cx / (nx - 1) - 1,
                    2 * cy / (ny - 1) - 1,
                    2 * cz / (nz - 1) - 1,
                ]
            )
        else:
            center = torch.tensor(
                [
                    2 * (cx + 0.5) / nx - 1,
                    2 * (cy + 0.5) / ny - 1,
                    2 * (cz + 0.5) / nz - 1,
                ]
            )
        self.register_buffer("center", center.view(1, 1, 3))

        if init_base_grid:
            self._build_base_grid()

        # -------------------------------
        # Precompute XY grid for slice sampling
        # -------------------------------
        y = torch.arange(ny)
        x = torch.arange(nx)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy_grid = torch.stack([xx, yy], dim=-1).float()  # (Y, X, 2)
        self.register_buffer("xy_grid", xy_grid)  # buffer, reused for all slice calls

    # ------------------------------------------------------------------
    # Core grid construction
    # ------------------------------------------------------------------
    def _build_base_grid(self) -> None:
        """
        Build base grid.
        """
        eye = torch.eye(3, 4).unsqueeze(0)  # (1, 3, 4)
        base_grid = F.affine_grid(
            eye, (1, 1, self.nz, self.ny, self.nx), align_corners=self.align_corners
        )
        self.register_buffer("base_grid", base_grid)

    def _isotropic_scale(
        self, device: str | torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Per-axis scale factors mapping normalized [-1,1] deltas to isotropic pixel-like units."""
        return torch.tensor(
            [(self.nx - 1) / 2, (self.ny - 1) / 2, (self.nz - 1) / 2],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3)

    def _rotate_normalized_grid(
        self,
        grid: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the isotropic rotate-and-translate transform to normalized
        (grid_sample-convention) sampling points.

        Recenters on `self.center` (RELION convention only), rescales to
        isotropic pixel-like units, rotates by R, undoes the rescale/recenter,
        then translates by t. Shared by `_build_grid` (identity base grid) and
        `sample_rotated_slices` (ROI query points mapped into this same
        normalized convention).

        Parameters
        ----------
        grid : torch.Tensor
            Normalized sampling coordinates, shape (B, N, 3).
        R : torch.Tensor
            Rotation matrices, shape (B, 3, 3).
        t : torch.Tensor
            Translations in normalized units, shape (B, 3).
        scale : torch.Tensor
            Per-axis isotropic scale factors, shape (1, 1, 3).

        Returns
        -------
        torch.Tensor
            Rotated normalized sampling coordinates, shape (B, N, 3).
        """
        if self.origin == "relion":
            grid = (grid - self.center) * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + self.center
            grid = grid + t.unsqueeze(1)
        else:
            grid = grid * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + t.unsqueeze(1)
        return grid

    def _build_grid(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Build a sampling grid from cached base grid and affine parameters.

        theta: (B, 3, 4)
        Returns: (B, Z, Y, X, 3)
        """
        B = theta.shape[0]
        R = theta[..., :3]  # (B, 3, 3)
        t = theta[..., 3]  # (B, 3)

        if not hasattr(self, "base_grid"):
            self._build_base_grid()
        grid = self.base_grid.expand(B, -1, -1, -1, -1)
        grid = grid.view(B, -1, 3)

        scale = self._isotropic_scale(theta.device, theta.dtype)
        grid = self._rotate_normalized_grid(grid, R, t, scale)

        grid = grid.view(B, self.nz, self.ny, self.nx, 3)
        return grid

    # ------------------------------------------------------------------
    # Real-space rotation
    # ------------------------------------------------------------------
    def rotate_real(self, V: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rotate a volume in real space.
        V: (Z, Y, X)
        theta: (B, 3, 4)
        Returns: (B, Z, Y, X)
        """
        B = theta.shape[0]
        grid = self._build_grid(theta)

        V = V.unsqueeze(0).unsqueeze(1)  # (1, 1, Z, Y, X)
        V = V.expand(B, 1, self.nz, self.ny, self.nx)

        V_rot = F.grid_sample(
            V,
            grid,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        )

        return V_rot.squeeze(1)  # (B, Z, Y, X)

    # ------------------------------------------------------------------
    # Fourier-space rotation
    # ------------------------------------------------------------------
    def rotate_fourier(self, V: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rotate a volume in Fourier space via real/imag decomposition.
        """
        V_f = fft3(V, shift=True)  # complex, (Z, Y, X)

        V_f_rot_real = self.rotate_real(V_f.real, theta)
        V_f_rot_imag = self.rotate_real(V_f.imag, theta)

        V_f_rot = torch.complex(V_f_rot_real, V_f_rot_imag)
        V_rot = ifft3(V_f_rot, shift=True)

        return V_rot.real  # (B, Z, Y, X)

    # ------------------------------------------------------------------
    # Forward with optional per-call mode override
    # ------------------------------------------------------------------
    def forward(
        self, V: torch.Tensor, theta: torch.Tensor, mode: str | None = None
    ) -> torch.Tensor:
        """
        Forward pass with optional mode override.
        mode: "real" or "fourier"; defaults to self.mode
        """
        if mode is None:
            mode = self.mode

        if mode not in ("real", "fourier"):
            raise ValueError(f"Unknown mode: {mode}. Must be 'real' or 'fourier'.")

        if mode == "real":
            return self.rotate_real(V, theta)
        else:
            return self.rotate_fourier(V, theta)

    # ------------------------------------------------------------------
    # Sample a single tilted slice
    # ------------------------------------------------------------------
    def sample_rotated_slices(
        self,
        V: torch.Tensor,
        theta: torch.Tensor,
        slice_indices: torch.Tensor | Sequence[int] | int,
        roi_center: tuple[int, int] | None = None,
        roi_size: tuple[int, int] | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        """
        Sample multiple Z-slices from a rotated volume, restricted to a ROI in XY.
        The slice_index=0 corresponds to the center of the volume.

        Parameters
        ----------
        V : (Z,Y,X) torch.Tensor
            Input 3D volume.
        theta : (B,3,4) torch.Tensor
            Affine rotation matrix.
        slice_indices : (K,) or list or int
            One or more slice indices along Z of the rotated volume, relative to the center.
        roi_center : tuple (cy, cx)
            Center pixel of the region of interest in Y,X coordinates of the original volume.
            Defaults to full volume center.
        roi_size : tuple (ny_roi, nx_roi)
            Size of the region to sample.
            Defaults to full volume size.
        padding_mode : str
            Passed to grid_sample. Defaults to self.padding_mode.

        Returns
        -------
        slices_roi : (B, K, ny_roi, nx_roi) torch.Tensor
            Interpolated 2D slices of the rotated volume, cropped to ROI.
        """
        if padding_mode is None:
            padding_mode = self.padding_mode

        ny, nx = self.ny, self.nx
        B = theta.shape[0]
        dtype, device = V.dtype, V.device

        slice_indices = _normalize_slice_indices(slice_indices, device)
        K = slice_indices.shape[0]
        roi_center, roi_size = _resolve_roi(roi_center, roi_size, ny, nx)
        ny_roi, nx_roi = roi_size

        points_pix = _build_roi_query_points(
            slice_indices, roi_center, roi_size, ny, nx, device, dtype
        )  # (K, ny_roi, nx_roi, 3)
        points_pix_flat = (
            points_pix.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, -1, 3)
        )

        # -------------------------------
        # Apply rotation in pixel space
        # -------------------------------
        R = theta[..., :3]  # (B, 3, 3)
        t = theta[..., 3]  # (B, 3) normalized
        scale = self._isotropic_scale(device, dtype)

        if self.origin == "relion":
            # Normalize points to [-1, 1] relative to V center
            points_norm = points_pix_flat / scale + self.center
        else:
            # PyTorch center convention
            points_norm = points_pix_flat / scale
        grid = self._rotate_normalized_grid(points_norm, R, t, scale)

        grid = grid.view(B, K, ny_roi, nx_roi, 3)

        # -------------------------------
        # Sample volume
        # -------------------------------
        V_in = _prepare_volume_for_grid_sample(V, B)

        slices_roi = F.grid_sample(
            V_in, grid, align_corners=self.align_corners, padding_mode=padding_mode
        )

        return slices_roi[:, 0]  # (B, K, ny_roi, nx_roi)
