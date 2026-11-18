"""Load a pre-built specimen volume from disk, for `TiltSeriesConfig`'s
``volume_path`` -- the "user already has a specimen volume" path, as opposed
to building one in-process via `build_tomogram_generator`/`specter build
tomogram` or `MicrographSpecimenGenerator`.
"""

from __future__ import annotations

import os

import torch


#: Prefix of the MRC header label a tomogram built wider than its field of
#: view carries (`field_of_view_label`).
_FIELD_OF_VIEW_LABEL = "specter field_of_view_yx="


def field_of_view_label(field_of_view: tuple[int, int]) -> str:
    """
    The MRC header label recording a volume's field of view.

    Parameters
    ----------
    field_of_view : tuple of int
        ``(Y, X)`` voxels of the centred part of the volume meant to be
        imaged.

    Returns
    -------
    str
        The label, read back by `volume_field_of_view`.
    """
    return f"{_FIELD_OF_VIEW_LABEL}{int(field_of_view[0])},{int(field_of_view[1])}"


def volume_field_of_view(path: str) -> tuple[int, int] | None:
    """
    The field of view a specimen volume was built around, if it records one.

    `specter build tomogram` with ``max_tilt_angle`` set builds a volume
    wider than the field of view, so that a tilted view images real
    specimen beyond the field's edges, and writes the field's size into
    the MRC header.

    Parameters
    ----------
    path : str
        Path to the volume file.

    Returns
    -------
    tuple of int or None
        ``(Y, X)`` voxels of the centred field of view, or None when the
        file records none (any ``.pt`` file, and any volume that is its
        own field of view).
    """
    if os.path.splitext(path)[1].lower() not in (".mrc", ".mrcs"):
        return None
    import mrcfile

    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        labels = mrc.get_labels()
    for label in labels:
        if label.startswith(_FIELD_OF_VIEW_LABEL):
            ny, nx = label[len(_FIELD_OF_VIEW_LABEL) :].split(",")
            return int(ny), int(nx)
    return None


def load_specimen_volume(path: str) -> torch.Tensor:
    """
    Load a pre-built specimen volume, already in scattering-potential units.

    Parameters
    ----------
    path : str
        Path to the volume file. ``.mrc``/``.mrcs`` are loaded via
        ``mrcfile`` (its on-disk storage order already matches specter's own
        ``(Z, Y, X)`` convention -- no axis swap needed). ``.pt`` is loaded
        via ``torch.load``.

    Returns
    -------
    torch.Tensor
        The volume, shape ``(Z, Y, X)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If ``path``'s extension isn't one of ``.mrc``, ``.mrcs``, ``.pt``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mrc", ".mrcs"):
        from ..io import read_mrc

        volume, _ = read_mrc(path)
    elif ext == ".pt":
        volume = torch.as_tensor(torch.load(path), dtype=torch.float32)
    else:
        raise ValueError(
            f"load_specimen_volume: unsupported file extension {ext!r} for "
            f"{path!r}. Expected one of: .mrc, .mrcs, .pt."
        )
    return volume
