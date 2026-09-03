"""Reconcile a refinement file with images that were Fourier-cropped after extraction.

A particle set records the box and pixel size it was extracted at. Users
routinely bin the images afterwards (a 360 px box at 0.57 Å becomes 200 px
at 1.03 Å) without touching the metadata, and every downstream reader would
then render the structure into a box that is too small by the same factor.
`recorded_box` reads the box the metadata claims; `rescale_metadata` writes
a copy whose pixel size and pixel-unit shifts match a different box, so the
ordinary extractors see a consistent file.
"""

from __future__ import annotations

import numpy as np


def recorded_box(metadata_path: str) -> int | None:
    """
    The image box size a refinement file records, or None if it does not.

    A CryoSPARC ``.cs`` carries ``blob/shape``; a RELION 3.1+ ``.star``
    carries ``rlnImageSize`` in its optics block. A passthrough file without
    image references records nothing, and the caller has to trust the images.
    """
    if metadata_path.endswith(".cs"):
        from cryosparc.dataset import Dataset

        d = Dataset.load(metadata_path)
        if "blob/shape" not in d.fields():
            return None
        return int(np.asarray(d["blob/shape"])[0][0])
    import starfile

    data = starfile.read(metadata_path)
    if isinstance(data, dict) and "optics" in data and "rlnImageSize" in data["optics"]:
        return int(data["optics"]["rlnImageSize"].iloc[0])
    return None


def rescale_metadata(
    metadata_path: str, new_box: int, out_path: str, current_box: int | None = None
) -> float:
    """
    Write a copy of ``metadata_path`` whose pixel size describes images
    Fourier-cropped (or padded) to ``new_box`` pixels.

    The physical field of view is unchanged, so the pixel size scales by
    ``current_box / new_box`` and every shift expressed in pixels scales the
    other way; shifts in Ångström are untouched.

    Parameters
    ----------
    metadata_path : str
        The original ``.cs`` or ``.star``.
    new_box : int
        Box size of the images actually in hand.
    out_path : str
        Where to write the rescaled copy (same format as the input).
    current_box : int, optional
        The box the file's pixel size currently describes. Unset reads it
        from the file (`recorded_box`), which a passthrough without image
        references cannot supply.

    Returns
    -------
    float
        The new pixel size in Å.
    """
    box = current_box if current_box is not None else recorded_box(metadata_path)
    if box is None:
        raise ValueError(f"{metadata_path} records no box size; cannot rescale it")
    factor = box / new_box
    if metadata_path.endswith(".cs"):
        from cryosparc.dataset import Dataset

        d = Dataset.load(metadata_path)
        new_psize = None
        for key in ("alignments3D/psize_A", "blob/psize_A", "alignments2D/psize_A"):
            if key in d.fields():
                d[key] = np.asarray(d[key], dtype=np.float32) * factor
                new_psize = float(np.asarray(d[key])[0])
        for key in ("alignments3D/shift", "alignments2D/shift"):
            if key in d.fields():
                d[key] = np.asarray(d[key], dtype=np.float32) / factor
        if "blob/shape" in d.fields():
            d["blob/shape"] = np.full_like(np.asarray(d["blob/shape"]), new_box)
        d.save(out_path)
        assert new_psize is not None
        return new_psize
    import starfile

    data = starfile.read(metadata_path)
    if isinstance(data, dict) and "optics" in data:
        optics = data["optics"]
        optics["rlnImagePixelSize"] = optics["rlnImagePixelSize"].astype(float) * factor
        if "rlnImageSize" in optics:
            optics["rlnImageSize"] = new_box
        new_psize = float(optics["rlnImagePixelSize"].iloc[0])
    else:
        table = data if not isinstance(data, dict) else list(data.values())[-1]
        table["rlnImagePixelSize"] = table["rlnImagePixelSize"].astype(float) * factor
        new_psize = float(table["rlnImagePixelSize"].iloc[0])
    starfile.write(data, out_path, overwrite=True)
    return new_psize
