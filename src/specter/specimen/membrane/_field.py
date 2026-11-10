"""
The signed distance field a membrane's mid-surface is read through.

`MembraneField` holds the signed distance ``phi`` to one membrane's
mid-surface on a regular grid: negative inside, positive outside, zero on
the surface. The geometry that produces it lives in ``_geometry``; this
module is the contract the transmembrane placement and the region
classification read it through.

Coordinates follow the rest of Specter: physical points are ``(x, y, z)``
triples, dense grids are ``(Z, Y, X)`` tensors, and ``MembraneField.origin_xyz``
is the physical location of grid index ``(0, 0, 0)``. A grid built about the
physical origin puts that origin on voxel ``n // 2`` of each axis
(:func:`_centred_origin_xyz`), the render canvas's convention, which is also
where ``soft_voxelize_coordinates`` and ``clip_insert_bounds`` put it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MembraneField:
    """
    Dense signed field over a physical working grid.

    Parameters
    ----------
    phi : torch.Tensor
        Signed field, shape ``(Z, Y, X)``. Negative inside the membrane
        solid, positive outside, zero at the mid-surface.
    spacing_angstrom : float
        Isotropic voxel spacing of ``phi``, Å.
    origin_xyz : torch.Tensor
        Physical ``(x, y, z)`` location of grid index ``(0, 0, 0)``, Å.
    clipped_at_boundary : bool, optional
        Whether the organelle's solid interior touched the working grid's
        own boundary during construction, as a plain checkable flag so a
        higher-level caller (e.g.
        ``TomogramSpecimenGenerator``) can programmatically skip compositing
        a visibly-truncated instance. Default ``False``.
    """

    phi: torch.Tensor
    spacing_angstrom: float
    origin_xyz: torch.Tensor
    clipped_at_boundary: bool = False

    def sample(self, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Trilinearly interpolate ``phi`` at arbitrary physical points.

        Parameters
        ----------
        points_xyz : torch.Tensor
            Physical ``(x, y, z)`` points, shape ``(..., 3)``, Å.

        Returns
        -------
        torch.Tensor
            Interpolated field values, shape ``(...)``. Points outside the
            grid are clamped to the nearest boundary value.
        """
        device = self.phi.device
        dtype = self.phi.dtype
        flat = points_xyz.reshape(-1, 3).to(device=device, dtype=dtype)
        grid = self._normalized_grid(flat)
        volume = self.phi[None, None]
        sampled = F.grid_sample(
            volume, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(points_xyz.shape[:-1])

    def gradient(
        self, points_xyz: torch.Tensor, eps_angstrom: float | None = None
    ) -> torch.Tensor:
        """
        Unit outward normal at arbitrary physical points via finite differences.

        Parameters
        ----------
        points_xyz : torch.Tensor
            Physical ``(x, y, z)`` points, shape ``(..., 3)``, Å.
        eps_angstrom : float, optional
            Finite-difference step, Å. Default ``0.5 * spacing_angstrom``.

        Returns
        -------
        torch.Tensor
            Unit vectors pointing from negative (inside) to positive
            (outside) ``phi``, shape ``(..., 3)``.
        """
        eps = eps_angstrom if eps_angstrom is not None else 0.5 * self.spacing_angstrom
        device = points_xyz.device
        dtype = points_xyz.dtype
        offsets = torch.eye(3, device=device, dtype=dtype) * eps
        components = []
        for axis in range(3):
            plus = self.sample(points_xyz + offsets[axis])
            minus = self.sample(points_xyz - offsets[axis])
            components.append((plus - minus) / (2.0 * eps))
        grad = torch.stack(components, dim=-1)
        norm = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp_min(1e-8)
        return grad / norm

    def _normalized_grid(self, points_xyz: torch.Tensor) -> torch.Tensor:
        device = self.phi.device
        origin = self.origin_xyz.to(device=device, dtype=points_xyz.dtype)
        nz, ny, nx = self.phi.shape
        idx_x = (points_xyz[:, 0] - origin[0]) / self.spacing_angstrom
        idx_y = (points_xyz[:, 1] - origin[1]) / self.spacing_angstrom
        idx_z = (points_xyz[:, 2] - origin[2]) / self.spacing_angstrom
        norm_x = 2.0 * (idx_x + 0.5) / nx - 1.0
        norm_y = 2.0 * (idx_y + 0.5) / ny - 1.0
        norm_z = 2.0 * (idx_z + 0.5) / nz - 1.0
        return torch.stack([norm_x, norm_y, norm_z], dim=-1).reshape(1, -1, 1, 1, 3)


def _centred_origin_xyz(
    shape_zyx: tuple[int, ...], spacing_angstrom: float
) -> torch.Tensor:
    """
    Physical ``(x, y, z)`` of voxel ``(0, 0, 0)`` of a grid centred on the origin.

    Centred in the render canvas's convention: voxel ``k`` of an axis of
    ``n`` voxels sits at ``(k - n // 2) * spacing_angstrom``, so physical
    ``(0, 0, 0)`` is exactly on voxel ``n // 2`` whatever the parity of
    ``n``. That is the voxel `~specter.arrays.clip_insert_bounds` puts on a
    local grid's insertion point, so a membrane grid built about this origin
    and inserted at a canvas voxel renders its physical origin on that
    voxel. ``-n * spacing / 2``, the grid's geometric centre, is half a voxel
    away on an odd axis.

    Parameters
    ----------
    shape_zyx : tuple of int
        Grid shape ``(Z, Y, X)``.
    spacing_angstrom : float
        Voxel spacing, Å.

    Returns
    -------
    torch.Tensor
        Origin ``(x, y, z)``, Å, float32, shape ``(3,)``.
    """
    nz, ny, nx = shape_zyx
    return -(
        torch.tensor([nx // 2, ny // 2, nz // 2], dtype=torch.float32)
        * spacing_angstrom
    )


def _grid_points_xyz(
    shape_zyx: tuple[int, int, int],
    spacing_angstrom: float,
    origin_xyz: torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Physical ``(x, y, z)`` of every voxel of a grid, shape ``(Z, Y, X, 3)``.

    Voxel ``(k, j, i)`` is at ``origin_xyz + (i, j, k) * spacing_angstrom``.
    Every grid the membrane package builds about the physical origin takes
    `origin_xyz` from `_centred_origin_xyz`, the canvas convention.
    """
    nz, ny, nx = shape_zyx
    iz = torch.arange(nz, device=device, dtype=dtype)
    iy = torch.arange(ny, device=device, dtype=dtype)
    ix = torch.arange(nx, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(iz, iy, ix, indexing="ij")
    origin = origin_xyz.to(device=device, dtype=dtype)
    x = origin[0] + xx * spacing_angstrom
    y = origin[1] + yy * spacing_angstrom
    z = origin[2] + zz * spacing_angstrom
    return torch.stack([x, y, z], dim=-1)


__all__ = ["MembraneField"]
