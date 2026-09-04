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

`_volume_rotator` is the only module here that imports `lightning`, and
`lightning` costs ~6 s on top of `torch` at import time (it pulls in
`torchmetrics` and `matplotlib`). It is therefore re-exported *lazily*, via
a PEP 562 module `__getattr__`, so that importing this package for
`rotate_volume` or a quaternion helper -- none of which need Lightning --
does not pay for it. `from specter.rotations import VolumeRotator` still
works and still returns the same object; it just loads the submodule on
first access. Consumers that genuinely want the rotator (`scattering.py`,
`imagegenerator/`) are `LightningModule`s themselves and so lose nothing.
Do not "simplify" this back to a plain top-level import.

References
----------
Brégier, R. (2021). Deep Regression on Manifolds: A 3D Rotation Case Study.
2021 International Conference on 3D Vision (3DV). https://arxiv.org/abs/2103.16317
roma source: https://github.com/naver/roma
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._random import (
    random_quaternion,
    random_rotation_matrix,
    random_rotvec,
    rotations_angular_difference,
)
from ._rotation import rotate_coordinates, translate_coordinates
from ._volume import (
    affine_sampling_grid,
    apply_fourier_translation,
    build_affine_matrix,
    fourier_origin_displacement,
    rotate_volume,
    rotate_volume_fourier,
    split_affine_translation,
    translations_angstrom_to_torch,
)

if TYPE_CHECKING:
    from ._volume_rotator import (
        VolumeRotator as VolumeRotator,
        _build_roi_query_points as _build_roi_query_points,
        _normalize_slice_indices as _normalize_slice_indices,
        _prepare_volume_for_grid_sample as _prepare_volume_for_grid_sample,
        _resolve_roi as _resolve_roi,
    )

#: Names served from ``_volume_rotator`` on first attribute access. See the
#: module docstring for why they are not imported eagerly.
_LAZY_VOLUME_ROTATOR_NAMES = frozenset(
    {
        "VolumeRotator",
        "_build_roi_query_points",
        "_normalize_slice_indices",
        "_prepare_volume_for_grid_sample",
        "_resolve_roi",
    }
)


def __getattr__(name: str) -> Any:
    """
    Resolve a ``_volume_rotator`` export on first access (PEP 562).

    Parameters
    ----------
    name : str
        Attribute being looked up on the ``specter.rotations`` module.

    Returns
    -------
    Any
        The requested object, also cached into the module globals so that
        subsequent lookups skip this function entirely.

    Raises
    ------
    AttributeError
        If ``name`` is not one of this package's lazy exports.
    """
    if name in _LAZY_VOLUME_ROTATOR_NAMES:
        from . import _volume_rotator

        value = getattr(_volume_rotator, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    List the module's attributes, including the not-yet-loaded lazy ones.

    Returns
    -------
    list of str
        Sorted attribute names, so tab completion and ``dir()`` show
        ``VolumeRotator`` whether or not it has been accessed yet.
    """
    return sorted(set(globals()) | _LAZY_VOLUME_ROTATOR_NAMES)


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
    "affine_sampling_grid",
]
