"""Load a pre-built specimen volume from disk, for `TiltSeriesConfig`'s
``volume_path`` -- the "user already has a specimen volume" path, as opposed
to building one in-process via `CryoETSpecimenGenerator` (polnet placement)
or `MicrographSpecimenGenerator`.
"""

from __future__ import annotations

import os

import torch


def load_specimen_volume(path: str) -> torch.Tensor:
    """
    Load a pre-built specimen volume, already in scattering-potential units.

    Parameters
    ----------
    path : str
        Path to the volume file. ``.mrc``/``.mrcs`` are loaded via
        ``mrcfile`` (its on-disk storage order already matches specter's own
        ``(Z, Y, X)`` convention -- no axis swap needed, see
        ``specimen/polnet_bridge.py``'s note on this). ``.pt`` is loaded via
        ``torch.load``.

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
        import mrcfile

        with mrcfile.open(path, permissive=True) as mrc:
            volume = torch.as_tensor(mrc.data.copy(), dtype=torch.float32)
    elif ext == ".pt":
        volume = torch.as_tensor(torch.load(path), dtype=torch.float32)
    else:
        raise ValueError(
            f"load_specimen_volume: unsupported file extension {ext!r} for "
            f"{path!r}. Expected one of: .mrc, .mrcs, .pt."
        )
    return volume
