"""
Bundled ice reference data, shipped inside the package.

Holds the pre-optimised :class:`~specter.ice.IceBank` library (``ice_cache/``)
plus the MD reference data every :class:`~specter.ice.GradientSKIcemaker` run
needs for its default S(k) target: ``lda_80k_frame799_full_coords.pt`` (a
frame of LDA-80K amorphous ice) and the precomputed
``mdsim_f_radial_avg_*.pt`` radial averages.

This lives under ``src/specter/`` rather than at the repository root so that
it ships with the package, following ``specter/atom_data``'s precedent. The
previous root-level ``ice-data/`` directory was located at runtime by walking
three directories up from the importing module, which resolves to the
repository root only for an editable install from a checkout; installed as a
wheel it pointed inside the virtualenv's ``lib/``, so neither the bundled ice
library nor the default S(k) target could be found. Resolving through
:mod:`importlib.resources` instead asks the import system where the data is,
which is correct for every install layout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

#: Subdirectory holding the pre-optimised `IceBank` configuration library.
ICE_CACHE_DIRNAME = "ice_cache"


def bundled_ice_data(*parts: str) -> Path:
    """
    Filesystem path to a bundled ice data file or directory.

    Parameters
    ----------
    *parts : str
        Path components below this package, e.g. ``"ice_cache"`` or
        ``"lda_80k_frame799_full_coords.pt"``. With no arguments, returns the
        package directory itself.

    Returns
    -------
    pathlib.Path
        Absolute path. Callers that glob a directory or hand the path to
        :func:`torch.load` need a real filesystem path, which this provides
        for any normally installed package (wheel or editable). It would not
        for an unextracted zipimport/egg, which specter does not support.
    """
    root = resources.files(__name__)
    for part in parts:
        root = root.joinpath(part)
    return Path(str(root))
