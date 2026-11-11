"""Radial (circular/spherical) averaging of 2D images and 3D volumes."""

from __future__ import annotations

from typing import Literal, overload

import torch


def _radial_profile(
    data: torch.Tensor, center: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Average `data` over shells of rounded index distance from `center`.

    The squared distance is summed from the last axis to the first
    (``x**2 + y**2 + z**2``). Returns the shell radii and the profile.
    """
    device = data.device
    axes = [
        torch.arange(size, device=device) - c
        for size, c in zip(data.shape, center, strict=True)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")

    # Radial distance of every element, rounded to its shell.
    squared = grids[-1] ** 2
    for g in reversed(grids[:-1]):
        squared = squared + g**2
    r_bin = torch.sqrt(squared).round().long().flatten()

    # Sum values and count elements per shell; empty shells average to 0.
    max_r = int(r_bin.max().item()) + 1
    sum_bin = torch.bincount(r_bin, weights=data.flatten(), minlength=max_r)
    count_bin = torch.bincount(r_bin, minlength=max_r)
    return torch.arange(max_r, device=device), sum_bin / count_bin.clamp(min=1)


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
    if center is None:
        center = (m // 2, n // 2, o // 2)
    r, radialprofile = _radial_profile(data, center)
    return (r, radialprofile) if return_r else radialprofile


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
    if center is None:
        center = (m // 2, n // 2)
    r, radialprofile = _radial_profile(data, center)
    return (r, radialprofile) if return_r else radialprofile
