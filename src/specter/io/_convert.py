"""
CryoSPARC ``.cs`` to RELION ``.star`` metadata conversion.

Reads the raw CryoSPARC columns rather than going through
:func:`~specter.io.extract_parameters_from_csfile`, which folds beam shift
*and* the real-space anisotropic-magnification correction into the particle
shifts for specter's own forward model. The magnification matrix travels to
RELION in the optics block, so pre-applying it here as well would make RELION
count it twice.

Conventions, and where each is pinned
-------------------------------------
Origins
    CryoSPARC's ``alignments3D/shift`` (px) times ``alignments3D/psize_A``
    equals RELION's ``rlnOriginXAngst`` with no sign flip: both readers in
    this package land their shifts in the same internal slot untransformed
    (``_cryosparc.py`` and ``_relion.py``), and that slot is documented in
    terms of ``rlnOriginXAngst`` in
    :func:`~specter.rotations.translations_angstrom_to_torch`.
Beam shift
    RELION has no beam-shift column because it does not need one: an image
    shift and an origin offset are the same operation. ``ctf/shift_A`` folds
    into the origin losslessly; only its provenance is lost.
Beam tilt
    ``ctf/tilt_A`` is an Angstrom-scale coefficient with ``chi = -2*pi *
    lambda^2 * tilt_A * k^3 * cos(theta)``. specter parameterises the same
    term as ``sin(tilt)`` with a ``2*pi*Cs*lambda^2`` prefactor (see
    :func:`specter.aberrations._functions.beamtilt`), so the angle is
    ``arcsin(tilt_A / Cs)``. RELION stores it in milliradians.
Anisotropic magnification
    CryoSPARC and RELION both define the matrix in Fourier space, so the
    conversion is the identity offset alone -- CryoSPARC stores ``M - I``,
    RELION stores ``M``. No inverse and no transpose: those belong to the
    Fourier-to-real-space change of frame that ``_cryosparc.py`` performs for
    ghostbuster, and they have no place in a format conversion.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import roma
import starfile
import torch
from cryosparc.dataset import Dataset

# Optics values that define a RELION optics group. Particles sharing all of
# them are pooled into one row of the optics block.
_OPTICS_COLUMNS = (
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnImagePixelSize",
    "rlnImageSize",
    "rlnImageDimensionality",
)

# Aberrations CryoSPARC records that have no resolution-independent RELION
# equivalent. RELION's rlnOddZernike/rlnEvenZernike coefficients are defined
# against proper Zernike radial polynomials normalised to a user-set maximum
# resolution, so there is no box-size-independent mapping from these.
_UNMAPPED_ABERRATIONS = {
    "ctf/trefoil_A": "trefoil",
    "ctf/tetra_A": "tetrafoil",
}


def _has(dataset: Any, key: str) -> bool:
    """Whether ``dataset`` carries ``key``, without assuming its API."""
    try:
        dataset[key]
    except (KeyError, ValueError, TypeError):
        return False
    return True


def _fields(dataset: Any) -> list[str]:
    """Column names of a ``Dataset`` or of a plain mapping."""
    fields = getattr(dataset, "fields", None)
    return list(fields()) if callable(fields) else list(dataset)


def _merge_passthrough(dataset: Any, passthrough: Any) -> dict[str, Any]:
    """
    Combine a particles ``.cs`` with its passthrough file.

    CryoSPARC splits a restack (and several other jobs) across two files:
    the particles ``.cs`` carries the image address, the passthrough carries
    the pose and CTF that came in from upstream. Neither alone has what a
    ``.star`` needs. They are joined on ``uid`` rather than row order, which
    the two files are not obliged to share.
    """
    main_uid = np.asarray(dataset["uid"])
    other_uid = np.asarray(passthrough["uid"])

    sorter = np.argsort(other_uid)
    pos = np.searchsorted(other_uid, main_uid, sorter=sorter)
    order = sorter[np.clip(pos, 0, len(other_uid) - 1)]
    if len(other_uid) != len(main_uid) or not np.array_equal(
        other_uid[order], main_uid
    ):
        raise KeyError(
            "The passthrough file does not describe the same particles as the "
            "particles file: their uid columns do not match. Pass the "
            "passthrough that belongs to this job."
        )

    merged = {key: dataset[key] for key in _fields(dataset)}
    for key in _fields(passthrough):
        if key not in merged:
            merged[key] = np.asarray(passthrough[key])[order]
    return merged


def _as_str_array(values: Any) -> list[str]:
    """CryoSPARC stores paths as bytes; RELION wants text."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _euler_degrees(pose: np.ndarray) -> np.ndarray:
    """``alignments3D/pose`` (Rodrigues rotvec) to RELION ZYZ Euler degrees."""
    R = roma.rotvec_to_rotmat(torch.as_tensor(pose, dtype=torch.float64))
    return roma.rotmat_to_euler("ZYZ", R, degrees=True).numpy()


def _warn_unmapped(dataset: Any) -> None:
    dropped = [
        name
        for key, name in _UNMAPPED_ABERRATIONS.items()
        if _has(dataset, key) and not np.allclose(dataset[key], 0.0)
    ]
    if dropped:
        warnings.warn(
            f"Dropping {' and '.join(dropped)} coefficients: RELION expresses "
            "higher-order aberrations as Zernike coefficients normalised to a "
            "chosen maximum resolution, so there is no box-size-independent "
            "conversion from CryoSPARC's Angstrom-scale coefficients. Every "
            "other CTF term is preserved.",
            UserWarning,
            stacklevel=3,
        )


def _optics_per_particle(dataset: Any, n: int) -> pd.DataFrame:
    """The optics values of each particle, before grouping."""
    columns: dict[str, Any] = {
        "rlnVoltage": np.asarray(dataset["ctf/accel_kv"], dtype=np.float64),
        # Millimetres in both formats.
        "rlnSphericalAberration": np.asarray(dataset["ctf/cs_mm"], dtype=np.float64),
        "rlnAmplitudeContrast": np.asarray(
            dataset["ctf/amp_contrast"], dtype=np.float64
        ),
        "rlnImagePixelSize": np.asarray(
            dataset["alignments3D/psize_A"], dtype=np.float64
        ),
    }
    if _has(dataset, "blob/shape"):
        columns["rlnImageSize"] = np.asarray(dataset["blob/shape"])[:, 0].astype(int)
    columns["rlnImageDimensionality"] = np.full(n, 2, dtype=int)

    # ctf/anisomag holds M - I, flattened row-major. All-zero means isotropic,
    # in which case the columns are left out rather than written as identity.
    if _has(dataset, "ctf/anisomag"):
        aniso = np.asarray(dataset["ctf/anisomag"], dtype=np.float64).reshape(n, 4)
        if not np.allclose(aniso, 0.0):
            magmat = aniso + np.array([1.0, 0.0, 0.0, 1.0])
            for i, name in enumerate(("00", "01", "10", "11")):
                columns[f"rlnMagMat{name}"] = magmat[:, i]

    return pd.DataFrame(columns)


def _group_optics(per_particle: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Collapse per-particle optics to unique rows, numbered from 1 in order of
    first appearance, and return the group each particle belongs to.
    """
    optics = per_particle.drop_duplicates().reset_index(drop=True)
    optics.insert(0, "rlnOpticsGroup", np.arange(1, len(optics) + 1))
    merged = per_particle.merge(
        optics, on=list(per_particle.columns), how="left", sort=False
    )
    return optics, merged["rlnOpticsGroup"].to_numpy()


def _particles_table(
    dataset: Any,
    n: int,
    group_ids: np.ndarray,
    cs_angstrom: np.ndarray,
    image_prefix: str | None = None,
    image_basename: bool = False,
) -> pd.DataFrame:
    """Per-particle pose, CTF and image-address columns."""
    psize = np.asarray(dataset["alignments3D/psize_A"], dtype=np.float64)
    shift_px = np.asarray(dataset["alignments3D/shift"], dtype=np.float64)
    origins = shift_px * psize[:, None]
    if _has(dataset, "ctf/shift_A"):
        origins = origins - np.asarray(dataset["ctf/shift_A"], dtype=np.float64)

    euler = _euler_degrees(np.asarray(dataset["alignments3D/pose"]))
    idx = np.asarray(dataset["blob/idx"], dtype=np.int64)
    paths = _as_str_array(dataset["blob/path"])
    # Stripping before prefixing so the two compose into a full relocation.
    if image_basename:
        paths = [os.path.basename(p) for p in paths]
    if image_prefix is not None:
        # blob/path is relative to the CryoSPARC project directory, so a
        # bare conversion only resolves for a reader started there.
        paths = [os.path.join(image_prefix, p) for p in paths]

    columns: dict[str, Any] = {
        # CryoSPARC addresses images by (path, idx) with idx 0-based; RELION
        # uses a 1-based "index@stack" string.
        "rlnImageName": [f"{i + 1:06d}@{p}" for i, p in zip(idx, paths)],
        "rlnOpticsGroup": group_ids,
        "rlnAngleRot": euler[:, 0],
        "rlnAngleTilt": euler[:, 1],
        "rlnAnglePsi": euler[:, 2],
        "rlnOriginXAngst": origins[:, 0],
        "rlnOriginYAngst": origins[:, 1],
        "rlnDefocusU": np.asarray(dataset["ctf/df1_A"], dtype=np.float64),
        "rlnDefocusV": np.asarray(dataset["ctf/df2_A"], dtype=np.float64),
        "rlnDefocusAngle": np.degrees(
            np.asarray(dataset["ctf/df_angle_rad"], dtype=np.float64)
        ),
    }

    if _has(dataset, "ctf/phase_shift_rad"):
        columns["rlnPhaseShift"] = np.degrees(
            np.asarray(dataset["ctf/phase_shift_rad"], dtype=np.float64)
        )

    if _has(dataset, "ctf/tilt_A"):
        tilt = np.asarray(dataset["ctf/tilt_A"], dtype=np.float64)
        tilt_rad = np.arcsin(tilt / cs_angstrom[:, None])
        columns["rlnBeamTiltX"] = tilt_rad[:, 0] * 1e3
        columns["rlnBeamTiltY"] = tilt_rad[:, 1] * 1e3

    if _has(dataset, "alignments3D/alpha"):
        columns["rlnCtfScalefactor"] = np.asarray(
            dataset["alignments3D/alpha"], dtype=np.float64
        )

    if _has(dataset, "alignments3D/split"):
        # CryoSPARC labels half-sets 0/1, RELION 1/2.
        columns["rlnRandomSubset"] = (
            np.asarray(dataset["alignments3D/split"], dtype=np.int64) + 1
        )

    if _has(dataset, "micrograph_blob/path"):
        columns["rlnMicrographName"] = _as_str_array(dataset["micrograph_blob/path"])

    return pd.DataFrame(columns)


def convert_csfile_to_starfile(
    csfile_path: str,
    star_path: str,
    passthrough_path: str | None = None,
    image_prefix: str | None = None,
    image_basename: bool = False,
    overwrite: bool = True,
) -> None:
    """
    Convert a CryoSPARC particle ``.cs`` file to a RELION 3.1 ``.star`` file.

    Metadata only: no image stack is read, copied or rewritten.
    ``rlnImageName`` points at the ``.mrcs`` files the ``.cs`` already
    references, addressing each row by its own ``blob/idx`` so that a stack
    whose row order differs from the metadata's is still described correctly.

    Parameters
    ----------
    csfile_path : str
        Path to the CryoSPARC particle ``.cs`` file.
    star_path : str
        Path of the ``.star`` file to write. A single file holding two data
        blocks, ``optics`` and ``particles``.
    passthrough_path : str, optional
        Path to the job's ``*_passthrough_particles.cs``. Several job types
        -- restack among them -- split their output, leaving the image
        address in the particles file and the pose and CTF in the
        passthrough. Give both and they are joined on ``uid``. Default is
        None, for a ``.cs`` that already carries every column.
    image_prefix : str, optional
        Prepended to each ``blob/path``. CryoSPARC records image paths
        relative to the project directory (``J423/restack/batch_0.mrc``), so
        without this the ``.star`` only resolves for a reader started in
        that directory. Pass the project directory to get absolute paths.
        Default is None, which writes the paths through unchanged.
    image_basename : bool, optional
        Reduce each image path to its filename, dropping the directory.
        CryoSPARC's particle importer takes the stack directory as its own
        parameter and resolves images by filename within it, so the stored
        directory is noise there. Applied before ``image_prefix``, so the
        two compose into a full relocation. Default is False.
    overwrite : bool, optional
        Overwrite ``star_path`` if it exists. Default is True.

    Notes
    -----
    Trefoil and tetrafoil are dropped with a warning; see
    ``_UNMAPPED_ABERRATIONS``. Every other CTF term, the pose, the half-set
    labels and the per-particle scale factor are preserved.
    """
    dataset: Any = Dataset.load(csfile_path)
    if passthrough_path is not None:
        dataset = _merge_passthrough(dataset, Dataset.load(passthrough_path))

    required = ("alignments3D/pose", "alignments3D/shift", "alignments3D/psize_A")
    missing = [key for key in required if not _has(dataset, key)]
    if not _has(dataset, "blob/idx"):
        missing.append("blob/idx")
    if missing:
        hint = (
            "A particle .cs file from a refinement job is expected."
            if passthrough_path is not None
            else "If this job wrote a *_passthrough_particles.cs alongside it, "
            "pass that too: CryoSPARC splits the image address and the "
            "pose/CTF across the two files."
        )
        raise KeyError(
            f"{csfile_path} is missing required column(s): {', '.join(missing)}. {hint}"
        )

    n = len(np.asarray(dataset["alignments3D/psize_A"]))
    _warn_unmapped(dataset)

    cs_angstrom = np.asarray(dataset["ctf/cs_mm"], dtype=np.float64) * 1e7
    optics, group_ids = _group_optics(_optics_per_particle(dataset, n))
    particles = _particles_table(
        dataset, n, group_ids, cs_angstrom, image_prefix, image_basename
    )

    starfile.write(
        {"optics": optics, "particles": particles}, star_path, overwrite=overwrite
    )
