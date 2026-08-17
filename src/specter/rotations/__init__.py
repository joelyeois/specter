"""
Rotation representations are handled directly via the `roma` library
(xyzw quaternion convention, matching roma's default). This package keeps
only what roma doesn't provide: coordinate/volume transforms and the
`VolumeRotator` module used for sampling (possibly tilted) slices out of a
3D volume.

Split across:
    _rotation.py        - translate_coordinates, rotate_coordinates (thin roma wrapper)
    _random.py           - random quaternion/rotvec/matrix generators, rotations_angular_difference (thin roma wrappers)
    _volume.py           - rotate_volume, rotate_volume_fourier, affine matrix helpers
    _volume_rotator.py   - VolumeRotator (LightningModule) and its private helpers

Everything is re-exported here so existing `from .rotations import X` /
`from specter.rotations import X` usage is unaffected by the split.

References
----------
Brégier, R. (2021). Deep Regression on Manifolds: A 3D Rotation Case Study.
2021 International Conference on 3D Vision (3DV). https://arxiv.org/abs/2103.16317
roma source: https://github.com/naver/roma
"""

from ._random import (
    random_quaternion,
    random_rotation_matrix,
    random_rotvec,
    rotations_angular_difference,
)
from ._rotation import rotate_coordinates, translate_coordinates
from ._volume import (
    apply_fourier_translation,
    build_affine_matrix,
    fourier_origin_displacement,
    rotate_volume,
    rotate_volume_fourier,
    split_affine_translation,
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
    "translate_coordinates",
    "rotate_coordinates",
    "random_quaternion",
    "random_rotvec",
    "random_rotation_matrix",
    "rotations_angular_difference",
    "rotate_volume",
    "rotate_volume_fourier",
    "split_affine_translation",
    "apply_fourier_translation",
    "fourier_origin_displacement",
    "translations_angstrom_to_torch",
    "build_affine_matrix",
    "VolumeRotator",
]
