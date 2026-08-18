"""
Tilt-series geometry, shared by the forward model
(:class:`~specter.imagegenerator.TiltSeriesGenerator`) and the inverse
problem (:class:`~specter.ghostbuster.TomogramReconstructor`), plus AreTomo3
tilt-geometry file I/O.

Split across:
    _geometry.py  - pure functions of shape/pose: tilt-coverage sizing,
                     reflect-padding, cosine tapering, the rotated-corner
                     Z-extent used for multislice depth + defocus-shift
                     correction.
    _aretomo3.py   - AreTomo3 .aln/.xf/global-shifts CSV parsing into
                      quaternions/translations.
"""

from ._aretomo3 import (
    read_aretomo3_aln,
    read_aretomo3_global_shifts,
    read_aretomo3_xf,
    tilt_to_quaternions,
)
from ._geometry import (
    apply_volume_cosine_taper,
    estimate_max_allowed_nxy,
    estimate_max_allowed_tilt_deg,
    estimate_required_nxy,
    infer_max_tilt_from_inputs,
    nz_tilt_for_pose,
    pad_volume_xy_for_tilt,
    shift_ctf_defocus_for_tilt,
)

__all__ = [
    "apply_volume_cosine_taper",
    "estimate_max_allowed_nxy",
    "estimate_max_allowed_tilt_deg",
    "estimate_required_nxy",
    "infer_max_tilt_from_inputs",
    "nz_tilt_for_pose",
    "pad_volume_xy_for_tilt",
    "shift_ctf_defocus_for_tilt",
    "read_aretomo3_aln",
    "read_aretomo3_global_shifts",
    "read_aretomo3_xf",
    "tilt_to_quaternions",
]
