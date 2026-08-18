from __future__ import annotations

from typing import Sequence

import lightning as L
import torch
import torch.nn.functional as F

from ..fft import fft3, ifft3
from ._volume import (
    apply_fourier_translation,
    fourier_origin_displacement,
    split_affine_translation,
)


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


def _prepare_volume_for_grid_sample(V: torch.Tensor, batchsize: int) -> torch.Tensor:
    """Normalize a (Z,Y,X) / (B,Z,Y,X) / (B,1,Z,Y,X) volume to (B,1,Z,Y,X)."""
    if V.ndim == 3:
        return V.unsqueeze(0).unsqueeze(0).expand(batchsize, -1, -1, -1, -1)
    elif V.ndim == 4:  # (B, Z, Y, X)
        return V.unsqueeze(1)
    else:  # (B, 1, Z, Y, X)
        return V


def _windowed_grid_sample(
    V: torch.Tensor,
    grid: torch.Tensor,
    nz: int,
    ny: int,
    nx: int,
    align_corners: bool,
    padding_mode: str,
    device: torch.device,
    margin: int,
) -> torch.Tensor:
    """
    Interpolate `grid` against `V` without moving the whole of `V` to `device`.

    `grid` (already fully computed, normalized sample coordinates into `V`,
    shape (B, K, ny_roi, nx_roi, 3)) is a rigid rotation of an axis-aligned
    box, so it is itself a rotated ("slanted") box in `V`'s coordinate frame
    -- not axis-aligned, so it can't be sliced directly. Instead this takes
    the smallest axis-aligned bounding box that CONTAINS it (padded by
    `margin` px so bilinear's 1px neighbor footprint never reaches the
    block's own artificial edge), transfers only that small block to
    `device`, and runs grid_sample there. Wherever the box is clipped to
    V's true edges (rather than the query's own extent), padding_mode's
    behaviour is reproduced exactly, since the block's edge then coincides
    with the real volume edge. See dev/tilt series/windowed_streaming_
    prototype.py (and the geometry figure in that investigation) for the
    derivation and validation this was lifted from.

    Parameters
    ----------
    V : torch.Tensor
        (Z,Y,X) / (B,Z,Y,X) / (B,1,Z,Y,X), on any device (typically CPU).
    grid : torch.Tensor
        Normalized grid_sample-convention coordinates into `V`, shape
        (B, K, ny_roi, nx_roi, 3), on `V`'s device.
    margin : int
        Extra padding (px) around the query's exact pixel-space bounding
        box. 2 is generous headroom for bilinear interpolation; validated
        exact (vs. a full-volume-on-GPU reference) to ~1e-4 absolute error
        (float32 round-off from the extra normalize/renormalize step).

    Returns
    -------
    torch.Tensor
        (B, K, ny_roi, nx_roi), on `device`.
    """

    def unnorm(g: torch.Tensor, size: int) -> torch.Tensor:
        return (g + 1) * (size - 1) / 2 if align_corners else ((g + 1) * size - 1) / 2

    x_pix = unnorm(grid[..., 0], nx)
    y_pix = unnorm(grid[..., 1], ny)
    z_pix = unnorm(grid[..., 2], nz)

    x0, x1 = int(x_pix.min().floor()) - margin, int(x_pix.max().ceil()) + margin + 1
    y0, y1 = int(y_pix.min().floor()) - margin, int(y_pix.max().ceil()) + margin + 1
    z0, z1 = int(z_pix.min().floor()) - margin, int(z_pix.max().ceil()) + margin + 1
    x0c, x1c = max(x0, 0), min(x1, nx)
    y0c, y1c = max(y0, 0), min(y1, ny)
    z0c, z1c = max(z0, 0), min(z1, nz)

    B, K, ny_roi, nx_roi = grid.shape[:4]
    if x1c <= x0c or y1c <= y0c or z1c <= z0c:
        # Every query point in this batch falls outside V entirely (e.g. an
        # extreme-tilt Z-chunk at the sweep's edge) -- padding_mode="zeros"
        # semantics, the only mode this degenerate case can reach exactly.
        return torch.zeros(B, K, ny_roi, nx_roi, device=device, dtype=V.dtype)

    block = V[..., z0c:z1c, y0c:y1c, x0c:x1c].to(device, non_blocking=True)
    bz, by, bx = block.shape[-3:]

    def renorm(pix: torch.Tensor, offset: int, size: int) -> torch.Tensor:
        local = pix - offset
        return (
            (2 * local / (size - 1) - 1)
            if align_corners
            else ((2 * local + 1) / size - 1)
        )

    grid_local = torch.stack(
        [renorm(x_pix, x0c, bx), renorm(y_pix, y0c, by), renorm(z_pix, z0c, bz)], dim=-1
    ).to(device)

    V_in = _prepare_volume_for_grid_sample(block, B)
    slices_roi = F.grid_sample(
        V_in, grid_local, align_corners=align_corners, padding_mode=padding_mode
    )
    return slices_roi[:, 0]


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
        cz: float
        cy: float
        cx: float
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

        # The spectrum's DC term always lands at n // 2 under fftshift, so the
        # Fourier path rotates about the RELION centre whatever `origin` is.
        if self.align_corners:
            center_dc = torch.tensor(
                [
                    2 * (nx // 2) / (nx - 1) - 1,
                    2 * (ny // 2) / (ny - 1) - 1,
                    2 * (nz // 2) / (nz - 1) - 1,
                ]
            )
        else:
            center_dc = torch.tensor(
                [
                    2 * (nx // 2 + 0.5) / nx - 1,
                    2 * (ny // 2 + 0.5) / ny - 1,
                    2 * (nz // 2 + 0.5) / nz - 1,
                ]
            )
        self.register_buffer("center_dc", center_dc.view(1, 1, 3))

        if init_base_grid:
            self._build_base_grid()

    # ------------------------------------------------------------------
    # Core grid construction
    # ------------------------------------------------------------------
    def _build_base_grid(self) -> None:
        """
        Build base grid.
        """
        eye = torch.eye(3, 4).unsqueeze(0)  # (1, 3, 4)
        base_grid = F.affine_grid(
            eye, [1, 1, self.nz, self.ny, self.nx], align_corners=self.align_corners
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
        origin: str | None = None,
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
        if (origin or self.origin) == "relion":
            grid = (grid - self.center_dc) * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + self.center_dc
            grid = grid + t.unsqueeze(1)
        else:
            grid = grid * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + t.unsqueeze(1)
        return grid

    def _build_grid(
        self, theta: torch.Tensor, origin: str | None = None
    ) -> torch.Tensor:
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
        grid = self._rotate_normalized_grid(grid, R, t, scale, origin=origin)

        grid = grid.view(B, self.nz, self.ny, self.nx, 3)
        return grid

    # ------------------------------------------------------------------
    # Real-space rotation
    # ------------------------------------------------------------------
    def rotate_real(
        self, V: torch.Tensor, theta: torch.Tensor, origin: str | None = None
    ) -> torch.Tensor:
        """
        Rotate a volume in real space.
        V: (Z, Y, X)
        theta: (B, 3, 4)
        origin: overrides self.origin for this call; used by `rotate_fourier`,
            whose spectrum must always be rotated about the DC term.
        Returns: (B, Z, Y, X)
        """
        B = theta.shape[0]
        grid = self._build_grid(theta, origin=origin)

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

        Any translation carried by `theta` is applied as a phase ramp rather than
        by resampling the spectrum, which would modulate the density instead of
        moving it. `self.origin` is folded into the same ramp: the rotation itself
        is always centred on the spectrum's DC term at ``n // 2``.
        """
        V_f = fft3(V, shift=True)  # complex, (Z, Y, X)

        theta_rot, displacement = split_affine_translation(theta)
        if self.origin == "center":
            displacement = displacement + fourier_origin_displacement(
                theta, self.nz, self.ny, self.nx
            )

        V_f_rot_real = self.rotate_real(V_f.real, theta_rot, origin="relion")
        V_f_rot_imag = self.rotate_real(V_f.imag, theta_rot, origin="relion")

        V_f_rot = apply_fourier_translation(
            torch.complex(V_f_rot_real, V_f_rot_imag), displacement
        )
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
        device: str | torch.device | None = None,
        window_margin: int = 2,
    ) -> torch.Tensor:
        """
        Sample multiple Z-slices from a rotated volume, restricted to a ROI in XY.
        The slice_index=0 corresponds to the center of the volume.

        Parameters
        ----------
        V : (Z,Y,X) torch.Tensor
            Input 3D volume. May live on a different device than `device`
            below (e.g. CPU, to avoid holding a huge volume in GPU memory)
            -- see `device`.
        theta : (B,3,4) torch.Tensor
            Affine rotation matrix. May be on any device; internally moved
            to `V`'s device for the (cheap) grid construction.
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
        device : str or torch.device, optional
            Device the returned slices should live on. Defaults to `V`'s own
            device (original behaviour: grid_sample runs directly against
            `V` wherever it is). When given and different from `V`'s device,
            the interpolation itself still runs on `device` -- but rather
            than transferring the whole of `V` there first, only the small
            axis-aligned bounding box the rotated query actually touches is
            sliced out of `V` and transferred (see `_windowed_grid_sample`).
            This keeps `device` memory bounded by the query's own geometry
            (ROI size x `slice_indices` extent), never by `V`'s total size --
            validated flat at ~25-30MB from a 66MB volume up to a 67GB one,
            vs. a whole-volume transfer's memory scaling linearly with `V`
            and eventually OOMing (see dev/tilt series/windowed_streaming_
            scaling.py). Use this whenever `V` is deliberately kept off the
            compute device -- e.g. `TiltSeriesGenerator` falls back to this
            automatically (see its `_ensure_volume_placed`) when its volume
            doesn't fit on the GPU alongside a `.to(device)` call.
        window_margin : int, optional
            Only used when `device` triggers the windowed path above: extra
            padding (px) around the query's exact pixel-space bounding box,
            for bilinear interpolation's 1px neighbor footprint. Default 2.

        Returns
        -------
        slices_roi : (B, K, ny_roi, nx_roi) torch.Tensor
            Interpolated 2D slices of the rotated volume, cropped to ROI, on
            `device` (or `V`'s own device if `device` is None).
        """
        if padding_mode is None:
            padding_mode = self.padding_mode

        ny, nx = self.ny, self.nx
        B = theta.shape[0]
        dtype = V.dtype
        theta = theta.to(V.device)
        target_device = torch.device(device) if device is not None else V.device

        slice_indices = _normalize_slice_indices(slice_indices, V.device)
        K = slice_indices.shape[0]
        roi_center, roi_size = _resolve_roi(roi_center, roi_size, ny, nx)
        ny_roi, nx_roi = roi_size

        points_pix = _build_roi_query_points(
            slice_indices, roi_center, roi_size, ny, nx, V.device, dtype
        )  # (K, ny_roi, nx_roi, 3)
        points_pix_flat = (
            points_pix.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, -1, 3)
        )

        # -------------------------------
        # Apply rotation in pixel space
        # -------------------------------
        R = theta[..., :3]  # (B, 3, 3)
        t = theta[..., 3]  # (B, 3) normalized
        scale = self._isotropic_scale(V.device, dtype)

        if self.origin == "relion":
            # Normalize points to [-1, 1] relative to V center
            points_norm = points_pix_flat / scale + self.center.to(V.device, dtype)
        else:
            # PyTorch center convention
            points_norm = points_pix_flat / scale
        grid = self._rotate_normalized_grid(points_norm, R, t, scale)

        grid = grid.view(B, K, ny_roi, nx_roi, 3)

        # -------------------------------
        # Sample volume
        # -------------------------------
        if target_device == V.device:
            V_in = _prepare_volume_for_grid_sample(V, B)
            slices_roi = F.grid_sample(
                V_in, grid, align_corners=self.align_corners, padding_mode=padding_mode
            )
            return slices_roi[:, 0]  # (B, K, ny_roi, nx_roi)

        return _windowed_grid_sample(
            V,
            grid,
            self.nz,
            ny,
            nx,
            self.align_corners,
            padding_mode,
            target_device,
            window_margin,
        )
