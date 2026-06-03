from __future__ import annotations

import numpy as np
import torch


def read_aretomo3_aln(
    aln_path: str,
    pixel_size: float,
) -> dict[str, torch.Tensor]:
    """
    Read per-tilt alignment parameters from an AreTomo3 ``.aln`` file.

    The ``.aln`` file is AreTomo3's primary alignment output.  It
    contains tilt angles, tilt-axis angles, and per-tilt XY shifts for
    the global alignment, plus an optional local (patch-based) alignment
    section.  This function reads the global section and returns the
    shifts as a ``TiltSeriesGenerator``-compatible ``translations``
    tensor.

    Parameters
    ----------
    aln_path : str
        Path to the ``*.aln`` file produced by AreTomo3.
    pixel_size : float
        Pixel size in Å used to convert shifts from pixels to Å.

    Returns
    -------
    dict with keys:

    ``translations`` : torch.Tensor, shape (N_tilts, 2)
        Per-tilt XY translations in Å, ordered ``[tx, ty]``, ready for
        direct use as the ``translations`` argument of
        ``TiltSeriesGenerator``.
    ``tilt_angles`` : torch.Tensor, shape (N_tilts,)
        Tilt angles in degrees (column ``TILT``), in the order they
        appear in the file (sorted by tilt angle ascending).
    ``tilt_axis`` : torch.Tensor, shape (N_tilts,)
        Per-tilt tilt-axis angles in degrees (column ``ROT``).
    ``sec_indices`` : torch.Tensor, shape (N_tilts,)
        1-based section indices into the raw MRC stack (column ``SEC``).
    ``shift_pixels`` : torch.Tensor, shape (N_tilts, 2)
        Raw shifts ``[TX, TY]`` in full-resolution pixels before the
        Å conversion.

    Notes
    -----
    **Units**: TX and TY in the ``.aln`` file are the raw
    ``CAlignParam`` shifts in full-resolution pixels, written directly
    by ``CSaveAlignFile::mSaveGlobal``.  The same shifts appear in the
    ``*_AT_GL.csv`` log file; the two files are equivalent for the
    global alignment columns.  No pixel-size information is embedded in
    the ``.aln`` file itself, so ``pixel_size`` must be supplied.

    **Sign convention**: same as :func:`read_aretomo3_global_shifts`.
    TX > 0 shifts projected content in the +x direction; ty > 0 shifts
    in the +y direction.  The direct conversion is::

        tx = TX × pixel_size
        ty = TY × pixel_size

    **Global section column layout** (``CSaveAlignFile::mSaveGlobal``)::

        col 0 : SEC   — 1-based section index in the raw MRC stack
        col 1 : ROT   — tilt-axis angle (degrees)
        col 2 : GMAG  — global magnification (always 1.0)
        col 3 : TX    — shift_x (full-resolution pixels)
        col 4 : TY    — shift_y (full-resolution pixels)
        col 5 : SMEAN — intensity mean scale (always 1.0)
        col 6 : SFIT  — intensity fit scale  (always 1.0)
        col 7 : SCALE — additional scale     (always 1.0)
        col 8 : BASE  — intensity base       (always 0.0)
        col 9 : TILT  — tilt angle (degrees)

    The ``# Local Alignment`` comment marks the start of the patch-based
    local alignment section; lines after it are ignored by this function.

    Examples
    --------
    >>> result = read_aretomo3_aln("TS_001.aln", pixel_size=1.35)
    >>> tsg = TiltSeriesGenerator(
    ...     vol=vol,
    ...     angles=result["tilt_angles"],
    ...     translations=result["translations"],
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
                # Stop at the local alignment section
                if "Local Alignment" in line:
                    break
                continue
            global_rows.append([float(v) for v in line.split()])

    if not global_rows:
        raise ValueError(f"No global alignment data found in {aln_path}")

    data = np.array(global_rows, dtype=np.float32)

    sec_indices = torch.tensor(data[:, 0], dtype=torch.int32)
    tilt_axis = torch.tensor(data[:, 1], dtype=torch.float32)
    shift_pixels = torch.tensor(data[:, 3:5], dtype=torch.float32)
    tilt_angles = torch.tensor(data[:, 9], dtype=torch.float32)

    translations = shift_pixels * pixel_size

    return {
        "translations": translations,
        "tilt_angles": tilt_angles,
        "tilt_axis": tilt_axis,
        "sec_indices": sec_indices,
        "shift_pixels": shift_pixels,
    }


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
    ...     vol=vol,
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
