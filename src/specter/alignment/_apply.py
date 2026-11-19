"""Resampling of a volume in an aligned frame."""

from __future__ import annotations

import torch

from ..rotations import rotate_volume_exact


def apply_alignment(
    volume: torch.Tensor, *, rotation: torch.Tensor, centre: torch.Tensor
) -> torch.Tensor:
    """
    Resample a volume in an aligned frame.

    Applies a pose found on one volume to another on the same grid, such as a
    half-map or a mask of the map that was aligned.

    Parameters
    ----------
    volume : torch.Tensor
        Real cubic volume (N, N, N).
    rotation : torch.Tensor
        Rotation (3, 3) taking aligned-frame coordinates (x, y, z) to
        input-frame coordinates (`Alignment.rotation`).
    centre : torch.Tensor
        Origin of the aligned frame in the input volume, (3,), in voxels
        (x, y, z) from the central voxel (`Alignment.centre`).

    Returns
    -------
    torch.Tensor
        ``volume`` sampled at ``rotation @ (x - c0) + c0 + centre``, with
        ``c0`` the central voxel ``N // 2``, through
        `specter.rotations.rotate_volume_exact`: nothing is interpolated.
        Differentiable in `volume`.
    """
    n = volume.shape[-1]
    if volume.shape != (n, n, n):
        raise ValueError(f"expected a cubic volume, got shape {tuple(volume.shape)}")
    theta = torch.cat([rotation, (centre * 2 / n)[:, None]], -1)[None].to(volume)
    return rotate_volume_exact(volume, theta)[0]
