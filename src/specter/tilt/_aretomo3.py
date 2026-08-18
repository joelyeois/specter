"""
Parsers for AreTomo3's tilt-geometry output formats (``.aln``, global-shift
CSV, ``.xf``) into quaternions and shifts for ``TiltSeriesGenerator``.

References
----------
Zheng, S., Wolff, G., Greenan, G., Chen, Z., Faas, F. G. A., Bárcena, M.,
Koster, A. J., Cheng, Y., & Agard, D. A. (2022). AreTomo: An integrated
software package for automated marker-free, motion-corrected cryo-electron
tomographic alignment and reconstruction. Journal of Structural Biology: X, 6,
100068. https://doi.org/10.1016/j.yjsbx.2022.100068
AreTomo3 source: https://github.com/czimaginginstitute/AreTomo3
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import roma
import torch


def tilt_to_quaternions(
    tilt_angles_deg: torch.Tensor | Sequence[float],
    rot_deg: float,
) -> torch.Tensor:
    """
    Convert AreTomo3 tilt angles and tilt-axis angle to rotation quaternions.

    AreTomo3 parameterises the tilt geometry with a per-series tilt-axis
    angle (the ``ROT`` column in the ``.aln`` file) and per-tilt tilt angles
    (the ``TILT`` column).  This function converts that representation to
    unit quaternions suitable for ``TiltSeriesGenerator(quaternions=...)``.

    Using quaternions (rather than ``TiltSeriesGenerator(angles=...)``) is
    necessary when the tilt axis is not exactly horizontal (``'x'``) or
    vertical (``'y'``), which is typical for real data.

    Parameters
    ----------
    tilt_angles_deg : sequence of float or torch.Tensor, shape (N,)
        Per-tilt tilt angles in degrees (``TILT`` column from the ``.aln``
        file or ``tilt_angles`` key from :func:`read_aretomo3_aln`).
    rot_deg : float
        Tilt-axis angle in degrees measured from horizontal in the image
        plane (``ROT`` column from the ``.aln`` file — constant across all
        tilts for a given series).  0° = horizontal (≈ SPECTER ``'x'``),
        ±90° = vertical (≈ SPECTER ``'y'``).

    Returns
    -------
    quaternions : torch.Tensor, shape (N, 4)
        Unit quaternions in ``[x, y, z, w]`` order, one per tilt.  Pass
        directly as the ``quaternions`` argument of ``TiltSeriesGenerator``.

    Examples
    --------
    >>> result = read_aretomo3_aln("TS_001.aln", pixel_size=1.35)
    >>> quats = tilt_to_quaternions(result["tilt_angles"], result["tilt_axis"][0].item())
    >>> tsg = TiltSeriesGenerator(
    ...     volume=volume,
    ...     quaternions=quats,
    ...     translations=result["translations"],
    ...     ...
    ... )
    """
    theta = torch.deg2rad(torch.as_tensor(tilt_angles_deg, dtype=torch.float32))  # (N,)
    phi = torch.deg2rad(torch.tensor(rot_deg, dtype=torch.float32))  # scalar

    # Unit tilt axis in the image plane: (cos ROT, sin ROT, 0)
    axis = torch.stack([phi.cos(), phi.sin(), torch.tensor(0.0)], dim=0)  # (3,)

    rotvecs = theta.unsqueeze(-1) * axis.unsqueeze(0)  # (N, 3)
    return roma.rotvec_to_unitquat(rotvecs)  # (N, 4)


def read_aretomo3_aln(
    aln_path: str,
    pixel_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read per-tilt alignment parameters from an AreTomo3 ``.aln`` file.

    Parameters
    ----------
    aln_path : str
        Path to the ``*.aln`` file produced by AreTomo3.
    pixel_size : float
        Pixel size in Å used to convert shifts from pixels to Å.

    Returns
    -------
    quaternions : torch.Tensor, shape (N_tilts, 4)
        Per-tilt rotation quaternions ``[x, y, z, w]`` encoding the tilt
        angle and tilt-axis direction (``ROT`` column).  Pass directly as
        ``TiltSeriesGenerator(quaternions=...)``.
    translations : torch.Tensor, shape (N_tilts, 2)
        Per-tilt XY translations in Å, ordered ``[tx, ty]``.  Pass
        directly as ``TiltSeriesGenerator(translations=...)``.

    Notes
    -----
    Shifts are stored in the ``.aln`` file in full-resolution pixels and
    converted to Å by multiplying by ``pixel_size``.  The tilt-axis angle
    (``ROT`` column, constant per series) is used to build the quaternions
    via :func:`tilt_to_quaternions`.

    Examples
    --------
    >>> quats, translations = read_aretomo3_aln("TS_001.aln", pixel_size=1.35)
    >>> tsg = TiltSeriesGenerator(
    ...     volume=volume,
    ...     quaternions=quats,
    ...     translations=translations,
    ...     ...
    ... )
    """
    global_rows: list[list[float]] = []

    with open(aln_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Local Alignment" in line:
                    break
                continue
            global_rows.append([float(v) for v in line.split()])

    if not global_rows:
        raise ValueError(f"No global alignment data found in {aln_path}")

    data = np.array(global_rows, dtype=np.float32)

    tilt_axis = data[0, 1]
    shift_pixels = torch.tensor(data[:, 3:5], dtype=torch.float32)
    tilt_angles = torch.tensor(data[:, 9], dtype=torch.float32)

    translations = shift_pixels * pixel_size
    quaternions = tilt_to_quaternions(tilt_angles, float(tilt_axis))

    return quaternions, translations


def read_aretomo3_global_shifts(
    csv_path: str,
    pixel_size: float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Read per-tilt alignment shifts from an AreTomo3 global alignment CSV.

    Parses the ``*_AT_GL.csv`` file written by AreTomo3 into
    ``<outdir>/<prefix>_Log/``.  Returns the shifts as a
    ``TiltSeriesGenerator``-compatible ``translations`` tensor in Å.

    Parameters
    ----------
    csv_path : str
        Path to the ``*_AT_GL.csv`` file.
    pixel_size : float, optional
        Pixel size in Å.  When ``None`` (default), the value embedded in
        column 4 of the CSV is used.  Pass an explicit value to override it.

    Returns
    -------
    dict with keys:

    ``translations`` : torch.Tensor, shape (N_tilts, 2)
        Per-tilt XY translations in Å, ordered ``[tx, ty]``, ready for
        direct use as the ``translations`` argument of
        ``TiltSeriesGenerator``.
    ``tilt_angles`` : torch.Tensor, shape (N_tilts,)
        Tilt angles in degrees (in the order written by AreTomo3,
        typically acquisition order).
    ``tilt_axis`` : torch.Tensor, shape (N_tilts,)
        Per-tilt tilt-axis angles in degrees.
    ``shift_pixels`` : torch.Tensor, shape (N_tilts, 2)
        Raw shifts ``[sx, sy]`` in full-resolution pixels before the
        Å conversion.

    Notes
    -----
    **Units**: AreTomo3 stores all shifts in full-resolution pixels.
    The binning factor used internally during XCF measurement is
    immediately reversed in ``CProjAlignMain::mMeasure`` before the
    shifts are stored in ``CAlignParam``, so no binning correction is
    needed here.

    **Sign convention**: AreTomo3 ``shift_x > 0`` applies
    ``g(x) = f(x + sx)`` (Fourier phase shift), which moves image
    content in the −x direction.  SPECTER's ``TiltSeriesGenerator``
    with ``tx > 0`` (Å) samples the volume at ``x + tx / pixel_size``,
    also moving projected content in the −x direction.  The conventions
    therefore match directly:

    .. code-block:: text

        tx = shift_x × pixel_size
        ty = shift_y × pixel_size

    **CSV column layout** (written by ``CAreTomoMain::mLogGlobalShift``)::

        col 0 : tilt index (0-based)
        col 1 : acquisition index
        col 2 : tilt angle (degrees)
        col 3 : pixel size (Å/pixel)
        col 4 : cumulative dose (e⁻/Å²)
        col 5 : tilt-axis angle (degrees)
        col 6 : shift_x (full-resolution pixels)
        col 7 : shift_y (full-resolution pixels)

    Examples
    --------
    >>> result = read_aretomo3_global_shifts("TS_001_AT_GL.csv")
    >>> tsg = TiltSeriesGenerator(
    ...     volume=volume,
    ...     angles=result["tilt_angles"],
    ...     translations=result["translations"],
    ...     ...
    ... )
    """
    data = np.loadtxt(csv_path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    tilt_angles = torch.tensor(data[:, 2], dtype=torch.float32)
    csv_pixel_size = float(data[0, 3])
    tilt_axis = torch.tensor(data[:, 5], dtype=torch.float32)
    shift_pixels = torch.tensor(data[:, 6:8], dtype=torch.float32)

    ps = pixel_size if pixel_size is not None else csv_pixel_size
    translations = shift_pixels * ps

    return {
        "translations": translations,
        "tilt_angles": tilt_angles,
        "tilt_axis": tilt_axis,
        "shift_pixels": shift_pixels,
    }


def read_aretomo3_xf(
    xf_path: str,
    pixel_size: float,
) -> dict[str, torch.Tensor]:
    """
    Read per-tilt alignment shifts from an AreTomo3 IMOD ``.xf`` file.

    The ``.xf`` format stores a 2-D affine transform per tilt that
    combines tilt-axis de-rotation with the image-plane shift.  This
    function inverts the rotation to recover the raw image-plane shifts
    and returns them in a ``TiltSeriesGenerator``-compatible form.

    Parameters
    ----------
    xf_path : str
        Path to the ``*_st.xf`` file produced by AreTomo3.
    pixel_size : float
        Pixel size in Å, used to convert shifts from pixels to Å.

    Returns
    -------
    dict with keys:

    ``translations`` : torch.Tensor, shape (N_tilts, 2)
        Per-tilt XY translations in Å, ordered ``[tx, ty]``, ready for
        direct use as the ``translations`` argument of
        ``TiltSeriesGenerator``.
    ``shift_pixels`` : torch.Tensor, shape (N_tilts, 2)
        Raw image-plane shifts ``[sx, sy]`` in full-resolution pixels,
        recovered by undoing the tilt-axis rotation embedded in the
        ``.xf`` entries.
    ``rotation_matrices`` : torch.Tensor, shape (N_tilts, 2, 2)
        The 2×2 rotation matrices extracted from the ``.xf`` file,
        one per tilt.

    Notes
    -----
    **IMOD .xf format** — each line is::

        a11  a12  a21  a22  dx  dy

    where ``[[a11, a12], [a21, a22]] = R(−tilt_axis)`` and the encoded
    shifts ``[dx, dy]`` are::

        [dx, dy] = R(−tilt_axis) @ [−sx, −sy]

    Inverting gives the raw image-plane shifts::

        [sx, sy] = −Rᵀ(−tilt_axis) @ [dx, dy]
                 = −[[a11, a21], [a12, a22]] @ [dx, dy]

    **Sign convention** and **units**: same as
    :func:`read_aretomo3_global_shifts`.  Prefer that function when the
    ``*_AT_GL.csv`` is available — it provides shifts directly without
    needing to invert the rotation.

    **Dark frames**: AreTomo3 writes identity-matrix lines (``1 0 0 1 0 0``)
    for dark (excluded) tilts in the ``.xf`` file.  The resulting
    ``translations`` for those rows will be ``[0, 0]``, which is
    harmless but the tilt ordering follows the raw MRC stack, not
    acquisition order.
    """
    data = np.loadtxt(xf_path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    a11, a12, a21, a22 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    dx, dy = data[:, 4], data[:, 5]

    # Recover raw image-plane shift: [sx, sy] = -R^T @ [dx, dy]
    sx = -(a11 * dx + a21 * dy)
    sy = -(a12 * dx + a22 * dy)

    shift_pixels = torch.tensor(np.stack([sx, sy], axis=-1), dtype=torch.float32)
    translations = shift_pixels * pixel_size

    rotation_matrices = torch.tensor(
        np.stack([a11, a12, a21, a22], axis=-1).reshape(-1, 2, 2),
        dtype=torch.float32,
    )

    return {
        "translations": translations,
        "shift_pixels": shift_pixels,
        "rotation_matrices": rotation_matrices,
    }
