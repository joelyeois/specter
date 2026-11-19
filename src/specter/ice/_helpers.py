"""
Small helpers for the ice generators: the number density of amorphous ice
and a few array utilities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

# The water constants live in `specter.constants`, which `potential` imports
# without importing `ice`; re-exported here for the ice generators.
from ..constants import (  # noqa: F401
    avogadro,
    density_of_amorphous_ice,
    molar_mass_of_water,
    ndensity_of_amorphous_ice,
)


GridSize = int | Sequence[int] | np.ndarray | torch.Tensor
"""An XY grid size: one number for a square grid, or ``(ny, nx)``."""


def grid_yx(n: GridSize) -> tuple[int, int]:
    """
    Read an XY grid size given as one number or as a pair.

    Parameters
    ----------
    n : int or sequence of int
        Voxels along both axes, or ``(ny, nx)`` as a tuple, list, array or
        tensor.

    Returns
    -------
    tuple of int
        ``(ny, nx)``.

    Raises
    ------
    ValueError
        If `n` is not one or two whole numbers.
    """
    values = np.asarray(n.cpu() if isinstance(n, torch.Tensor) else n).reshape(-1)
    if values.size not in (1, 2) or not np.all(values == np.floor(values)):
        raise ValueError(
            f"a grid size is one whole number or a (ny, nx) pair, got {n!r}"
        )
    ny, nx = (int(values[0]), int(values[-1]))
    return ny, nx


def rfftn(array: torch.Tensor) -> torch.Tensor:
    """
    Compute N-dimensional real-input Fourier transform with centering.

    Wraps torch.fft.rfftn with FFT shifting to ensure the zero-frequency component
    is centered, handling the last dimension which is complex-valued differently.

    Parameters
    ----------
    array : torch.Tensor
        Input real-valued tensor.

    Returns
    -------
    fft : torch.Tensor
        Complex-valued tensor containing the Fourier coefficients.
        Zero frequency is centered.
    """
    return torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2),
    )
