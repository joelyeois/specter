"""
Output helpers for a reconstruction run: MRC volumes, plot3d previews and
FSC figures.
"""

from __future__ import annotations

from specter import logger

from pathlib import Path
from typing import TYPE_CHECKING

import mrcfile
import torch

if TYPE_CHECKING:
    from ..plots import FSCCurve

# ---------------------------------------------------------------------------
# Volume/figure saving helpers shared by Reconstructor and TomogramReconstructor
# ---------------------------------------------------------------------------


def save_volume_mrc(path: Path, v: torch.Tensor, voxel_size: float) -> None:
    """
    Save a reconstructed volume as an MRC file.

    Parameters
    ----------
    path : Path
        Output ``.mrc`` file path.
    v : torch.Tensor
        Volume, shape ``(Z, Y, X)``. Moved to CPU/float32 if not already.
    voxel_size : float
        Voxel size in Å, written into the MRC header.
    """
    v = v.detach().cpu().float()
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(v.numpy())
        mrc.voxel_size = voxel_size


def save_plot3d_preview(path: Path, v: torch.Tensor, title: str) -> None:
    """
    Save a ``plot3d`` preview image of a volume. Silently skips on failure.

    Parameters
    ----------
    path : Path
        Output image path.
    v : torch.Tensor
        Volume to render, shape ``(Z, Y, X)``.
    title : str
        Figure title.
    """
    try:
        import matplotlib.pyplot as plt

        from ..plots import plot3d

        fig = plot3d(v, title=title, show=False)
        assert fig is not None
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"plot3d preview skipped: {exc}")


def save_halfmap_fsc_figure(
    path: Path,
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    voxel_size: float,
    fsc_mask: torch.Tensor | None = None,
) -> FSCCurve | None:
    """
    Compute and save a gold-standard half-map FSC figure. Silently skips on failure.

    Parameters
    ----------
    path : Path
        Output image path.
    volume_a, volume_b : torch.Tensor
        The two half-set volumes, shape ``(Z, Y, X)``.
    voxel_size : float
        Voxel size in Å.
    fsc_mask : torch.Tensor, optional
        Gives a masked curve as well.

    Returns
    -------
    FSCCurve or None
        The curve drawn, with its resolutions at FSC = 0.143, or None if the
        figure could not be produced.
    """
    try:
        from ..plots import HALFMAP_FSC_THRESHOLD, fsc_curve

        curve = fsc_curve(
            volume_a, volume_b, voxel_size, HALFMAP_FSC_THRESHOLD, fsc_mask
        )
        save_fsc_figure(path, [curve], ["gold-standard"], voxel_size, halfmap=True)
        return curve
    except Exception as exc:
        logger.warning(f"half-map FSC plot skipped: {exc}")
        return None


def save_fsc_figure(
    path: Path,
    curves: list[FSCCurve],
    labels: list[str],
    voxel_size: float,
    halfmap: bool = False,
) -> None:
    """
    Draw FSC curves already computed and save the figure. Silently skips on failure.

    Parameters
    ----------
    path : Path
        Output image path.
    curves : list of FSCCurve
        From `specter.plots.fsc_curve`.
    labels : list of str
        One legend label per curve.
    voxel_size : float
        Voxel size in Å.
    halfmap : bool
        Half-map curves (FSC = 0.143) rather than map-to-model (0.5).
    """
    try:
        import matplotlib.pyplot as plt

        from ..plots import (
            HALFMAP_FSC_THRESHOLD,
            MAP_TO_MODEL_FSC_THRESHOLD,
            plot_fsc_curves,
        )

        threshold, title = (
            (HALFMAP_FSC_THRESHOLD, "Half-map FSCs")
            if halfmap
            else (MAP_TO_MODEL_FSC_THRESHOLD, "Map-to-model FSCs")
        )
        fig = plot_fsc_curves(curves, labels, voxel_size, threshold, title)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"FSC plot skipped: {exc}")
