"""Radial (circular/spherical) averaging of 2D images and 3D volumes."""

from __future__ import annotations

from typing import Literal, overload

import torch


@overload
def radial_profile_3d(
    data: torch.Tensor,
    center: tuple[float, float, float] | None = None,
    return_r: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def radial_profile_3d(
    data: torch.Tensor,
    center: tuple[float, float, float] | None = None,
    return_r: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def radial_profile_3d(
    data: torch.Tensor,
    center: tuple[float, float, float] | None = None,
    return_r: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the radial average (spherical average) of a 3D volume.

    Parameters
    ----------
    data : torch.Tensor
        3D tensor of shape (m, n, o).
    center : tuple of float, optional
        Center of the radial profile. Defaults to the integer center of the volume (m//2, n//2, o//2).
    return_r : bool, optional
        If True, also return the radius indices. Default is False.

    Returns
    -------
    radialprofile : torch.Tensor
        Radial average.
    r : torch.Tensor, optional
        Radius indices if return_r=True.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D tensor.")

    m, n, o = data.shape
    device = data.device

    # Default integer center
    if center is None:
        center = (m // 2, n // 2, o // 2)

    # Create coordinate grids relative to center
    z = torch.arange(m, device=device) - center[0]
    y = torch.arange(n, device=device) - center[1]
    x = torch.arange(o, device=device) - center[2]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Compute radial distances and integer bins
    r = torch.sqrt(xx**2 + yy**2 + zz**2)
    r_bin = r.round().long().flatten()
    data_flat = data.flatten()

    # Sum values per bin and count voxels per bin
    max_r = int(r_bin.max().item()) + 1
    sum_bin = torch.bincount(r_bin, weights=data_flat, minlength=max_r)
    count_bin = torch.bincount(r_bin, minlength=max_r)

    # Avoid division by zero
    radialprofile = sum_bin / count_bin.clamp(min=1)

    if return_r:
        return torch.arange(max_r, device=device), radialprofile
    else:
        return radialprofile


@overload
def radial_profile_2d(
    data: torch.Tensor,
    center: tuple[float, float] | None = None,
    return_r: Literal[False] = False,
) -> torch.Tensor: ...
@overload
def radial_profile_2d(
    data: torch.Tensor,
    center: tuple[float, float] | None = None,
    return_r: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def radial_profile_2d(
    data: torch.Tensor,
    center: tuple[float, float] | None = None,
    return_r: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the radial average (circular average) of a 2D image.

    Parameters
    ----------
    data : torch.Tensor
        2D tensor of shape (m, n).
    center : tuple of float, optional
        Center of the radial profile. Defaults to the integer center of the image (m//2, n//2).
    return_r : bool, optional
        If True, also return the radius indices. Default is False.

    Returns
    -------
    radialprofile : torch.Tensor
        Radial average.
    r : torch.Tensor, optional
        Radius indices if return_r=True.
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D tensor.")

    m, n = data.shape
    device = data.device

    # Default integer center
    if center is None:
        center = (m // 2, n // 2)

    # Create coordinate grids relative to center
    y = torch.arange(m, device=device) - center[0]
    x = torch.arange(n, device=device) - center[1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Compute radial distances and integer bins
    r = torch.sqrt(xx**2 + yy**2)
    r_bin = r.round().long().flatten()
    data_flat = data.flatten()

    # Sum values per bin and count pixels per bin
    max_r = int(r_bin.max().item()) + 1
    sum_bin = torch.bincount(r_bin, weights=data_flat, minlength=max_r)
    count_bin = torch.bincount(r_bin, minlength=max_r)

    # Avoid division by zero
    radialprofile = sum_bin / count_bin.clamp(min=1)

    if return_r:
        return torch.arange(max_r, device=device), radialprofile
    else:
        return radialprofile
