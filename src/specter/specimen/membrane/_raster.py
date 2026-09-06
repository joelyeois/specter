"""
Rasterize a membrane's geometry and calibrated potential onto an output grid.

Combines a :class:`~specter.specimen.membrane._field.MembraneField` (organic
shape, on its own fine working grid) with a
:class:`~specter.specimen.membrane._profile.BilayerProfile` (calibrated
``psi(d)``) into a physical density volume at whatever voxel size and grid
placement the caller requests.

Critically, this anti-aliases before resampling: when the requested output
voxel size is coarser than the field's own working spacing, the fine-grid
density is first low-pass filtered (a Gaussian matched to the output voxel
footprint) and only then interpolated onto the output grid. Naively
point-sampling the fine density directly (skipping the filter) aliases:
the bilayer's real, physical peak-to-peak leaflet separation is fixed, but
sparse point-sampling of its few-Å-wide features turns into what looks
like a growing or distorted gap as voxel size increases, with no physical
meaning. Filtering
first makes coarser voxel sizes correctly show the two leaflets blurring
together, which is the real, expected behavior as resolution drops.
"""

from __future__ import annotations

import torch

from ...filters import gaussian_blur3d
from ._field import MembraneField, _grid_points_xyz
from ._profile import BilayerProfile


def rasterize_membrane_density(
    field: MembraneField,
    profile: BilayerProfile,
    target_shape: tuple[int, int, int],
    target_spacing_angstrom: float,
    target_origin_xyz: torch.Tensor | None = None,
    antialias_sigma_angstrom: float | None = None,
) -> torch.Tensor:
    """
    Rasterize ``psi(field.phi)`` onto an output grid, anti-aliased.

    Parameters
    ----------
    field : MembraneField
        Membrane geometry, on its own fine working grid (see
        :class:`~specter.specimen.membrane._field.MembraneField`, produced
        by e.g. :func:`~specter.specimen.membrane.
        _field_spherical_harmonics.generate_membrane_field_spherical_harmonics`).
    profile : BilayerProfile
        Calibrated bilayer potential profile (see
        :func:`~specter.specimen.membrane._profile.compute_bilayer_profile`).
    target_shape : tuple of int
        Output grid shape.
    target_spacing_angstrom : float
        Output voxel size, Å.
    target_origin_xyz : torch.Tensor, optional
        Physical ``(x, y, z)`` location of output grid index ``(0, 0, 0)``,
        Å. Default centers the output grid on the physical origin,
        matching ``field``'s own centered-origin convention.
    antialias_sigma_angstrom : float, optional
        Gaussian blur sigma applied to the fine density before resampling,
        Å. Default ``0.5 * target_spacing_angstrom`` whenever the output is
        coarser than ``field``'s own working spacing (a conservative
        approximation of a box filter matched to the output voxel
        footprint -- see module docstring), 0 otherwise (no anti-aliasing
        needed when the output is the same resolution as or finer than the
        working field; interpolation alone suffices there).

    Returns
    -------
    torch.Tensor
        Density volume, shape ``target_shape``.
    """
    density_fine = profile(field.phi)

    if antialias_sigma_angstrom is None:
        antialias_sigma_angstrom = (
            0.5 * target_spacing_angstrom
            if target_spacing_angstrom > field.spacing_angstrom
            else 0.0
        )
    if antialias_sigma_angstrom > 0:
        sigma_vox = antialias_sigma_angstrom / field.spacing_angstrom
        # Zeros past the faces: the density ends inside the box.
        density_fine = gaussian_blur3d(density_fine, sigma_vox, pad_mode="constant")

    filtered_field = MembraneField(
        phi=density_fine,
        spacing_angstrom=field.spacing_angstrom,
        origin_xyz=field.origin_xyz,
    )

    if target_origin_xyz is None:
        extent = (
            torch.tensor(
                [target_shape[2], target_shape[1], target_shape[0]],
                dtype=torch.float32,
            )
            * target_spacing_angstrom
        )
        target_origin_xyz = -0.5 * extent

    points_xyz = _grid_points_xyz(
        target_shape, target_spacing_angstrom, target_origin_xyz, field.phi.device
    )
    return filtered_field.sample(points_xyz)


__all__ = ["rasterize_membrane_density"]
