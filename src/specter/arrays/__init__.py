"""Array utilities: grids, voxelization, radial profiles/NPS, padding/cropping, and block tiling."""

from __future__ import annotations

from ._grids import (
    ball3d,
    disk2d,
    grid_1d,
    grid_2d,
    grid_3d,
    kgrid_1d,
    kgrid_2d,
    kgrid_3d,
    radial_grid_2d,
    radial_grid_3d,
    radial_kgrid_2d,
    radial_kgrid_3d,
    real_to_kgrid_3d,
)
from ._nps import compute_nps_1d, compute_nps_2d, compute_nps_3d
from ._padding import (
    centered_pad,
    center_crop,
    clip_insert_bounds,
    coarse_occupancy_mask,
    compute_nz,
    downsample,
    fourier_crop,
    pad_to_common_shape,
    pad_volume,
    radial_symmetrize,
)
from ._profiles import radial_profile_2d, radial_profile_3d
from ._tiling import tile_volume_from_blocks, tile_volume_from_blocks_blended
from ._voxelize import (
    soft_voxelize_coordinates,
    soft_voxelize_coordinates_into,
    soft_voxelize_xy_coordinates,
)

__all__ = [
    "ball3d",
    "centered_pad",
    "center_crop",
    "clip_insert_bounds",
    "coarse_occupancy_mask",
    "compute_nps_1d",
    "compute_nps_2d",
    "compute_nps_3d",
    "compute_nz",
    "disk2d",
    "downsample",
    "fourier_crop",
    "grid_1d",
    "grid_2d",
    "grid_3d",
    "kgrid_1d",
    "kgrid_2d",
    "kgrid_3d",
    "pad_to_common_shape",
    "pad_volume",
    "radial_grid_2d",
    "radial_grid_3d",
    "radial_kgrid_2d",
    "radial_kgrid_3d",
    "radial_profile_2d",
    "radial_profile_3d",
    "radial_symmetrize",
    "real_to_kgrid_3d",
    "soft_voxelize_coordinates",
    "soft_voxelize_coordinates_into",
    "soft_voxelize_xy_coordinates",
    "tile_volume_from_blocks",
    "tile_volume_from_blocks_blended",
]
