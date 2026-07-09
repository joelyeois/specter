"""
Quaternion-based 3D rotations, coordinate/volume transforms, and the
`VolumeRotator` module used for sampling (possibly tilted) slices out of a
3D volume.

Split across:
    _rotation.py        - Rotation class, translate_coordinates
    _transforms.py       - coordinate/quaternion utilities built on Rotation
    _volume.py           - rotate_volume, rotate_volume_fourier, affine matrix helpers
    _volume_rotator.py   - VolumeRotator (LightningModule) and its private helpers

Everything is re-exported here so existing `from .rotations import X` /
`from specter.rotations import X` usage is unaffected by the split.
"""

from ._rotation import Rotation, translate_coordinates
from ._transforms import (
    random_quaternion,
    random_rotation_matrix,
    random_rotvec,
    rotate_coordinates,
    rotations_angular_difference,
)
from ._volume import (
    build_affine_matrix,
    rotate_volume,
    rotate_volume_fourier,
    translations_angstrom_to_torch,
)
from ._volume_rotator import VolumeRotator
from ._volume_rotator import _build_roi_query_points as _build_roi_query_points
from ._volume_rotator import _normalize_slice_indices as _normalize_slice_indices
from ._volume_rotator import (
    _prepare_volume_for_grid_sample as _prepare_volume_for_grid_sample,
)
from ._volume_rotator import _resolve_roi as _resolve_roi

__all__ = [
    "Rotation",
    "translate_coordinates",
    "rotate_coordinates",
    "random_quaternion",
    "random_rotvec",
    "random_rotation_matrix",
    "rotations_angular_difference",
    "rotate_volume",
    "rotate_volume_fourier",
    "translations_angstrom_to_torch",
    "build_affine_matrix",
    "VolumeRotator",
]
