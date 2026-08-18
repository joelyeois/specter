from __future__ import annotations

from pathlib import Path

import mrcfile
import torch

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
        print(f"[ghostbuster] plot3d preview skipped: {exc}")


def save_fsc_figure(
    path: Path,
    v: torch.Tensor,
    fsc_ref: torch.Tensor,
    voxel_size: float,
    label: str,
    fsc_mask: torch.Tensor | float | None = None,
    cryosparc_ref: torch.Tensor | None = None,
) -> None:
    """
    Compute and save a map-to-model FSC figure. Silently skips on failure.

    Parameters
    ----------
    path : Path
        Output image path.
    v : torch.Tensor
        Reconstructed volume, shape ``(Z, Y, X)``.
    fsc_ref : torch.Tensor
        Reference volume to compute FSC against.
    voxel_size : float
        Voxel size in Å.
    label : str
        Legend label for ``v`` in the figure.
    fsc_mask : torch.Tensor or float, optional
        Mask applied before FSC computation.
    cryosparc_ref : torch.Tensor, optional
        Additional CryoSPARC reference volume, plotted alongside ``v`` when given.
    """
    try:
        import matplotlib.pyplot as plt

        from ..plots import plot_map_to_model_fsc

        # Move references to volume's device for GPU FSC computation.
        device = v.device
        fsc_ref_dev = (
            fsc_ref.detach().to(device).float()
            if isinstance(fsc_ref, torch.Tensor)
            else fsc_ref
        )
        fsc_mask_dev = (
            fsc_mask.detach().to(device) if isinstance(fsc_mask, torch.Tensor) else None
        )

        volumes = [v]
        labels = [label]

        if cryosparc_ref is not None:
            cs_ref_dev = (
                cryosparc_ref.detach().to(device).float()
                if isinstance(cryosparc_ref, torch.Tensor)
                else cryosparc_ref
            )
            volumes.append(cs_ref_dev)
            labels.append("CryoSPARC")

        fig = plot_map_to_model_fsc(
            volumes,
            fsc_ref_dev,
            voxel_size=voxel_size,
            mask=fsc_mask_dev,
            labels=labels,
            show=False,
        )
        assert fig is not None
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[ghostbuster] FSC plot skipped: {exc}")
