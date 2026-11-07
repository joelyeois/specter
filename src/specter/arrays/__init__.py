"""Array utilities: grids, voxelization, radial profiles, padding/cropping, stamping, and block tiling."""

from __future__ import annotations

from ._grids import (
    ball3d,
    disk2d,
    grid_1d,
    grid_2d,
    grid_3d,
    kgrid_1d,
    kgrid_2d,
    radial_grid_2d,
    radial_grid_3d,
    real_to_kgrid_3d,
    rfft_frequency_squared,
)
from ._reduce import count_nonzero_chunked
from ._stamp import insert_particles_into_micrograph
from ._padding import (
    center_crop,
    clip_insert_bounds,
    compute_nz,
    fourier_crop,
    pad_volume,
    radial_symmetrize,
)
from ._profiles import radial_profile_2d, radial_profile_3d
from ._tiling import tile_volume_from_blocks_blended
from ._voxelize import (
    soft_voxelize_coordinates,
    soft_voxelize_coordinates_into,
    soft_voxelize_xy_coordinates,
)

__all__ = [
    "ball3d",
    "center_crop",
    "clip_insert_bounds",
    "count_nonzero_chunked",
    "compute_nz",
    "disk2d",
    "fourier_crop",
    "grid_1d",
    "grid_2d",
    "grid_3d",
    "insert_particles_into_micrograph",
    "kgrid_1d",
    "kgrid_2d",
    "pad_volume",
    "radial_grid_2d",
    "radial_grid_3d",
    "radial_profile_2d",
    "radial_profile_3d",
    "radial_symmetrize",
    "real_to_kgrid_3d",
    "rfft_frequency_squared",
    "soft_voxelize_coordinates",
    "soft_voxelize_coordinates_into",
    "soft_voxelize_xy_coordinates",
    "tile_volume_from_blocks_blended",
]
