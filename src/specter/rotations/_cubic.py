"""
Cubic B-spline interpolation of a volume at arbitrary points.

Trilinear interpolation attenuates fine detail wherever a sample point falls
between voxels, and a bent specimen is sampled between voxels along all three
axes. A cubic B-spline that interpolates the volume keeps most of that detail.
It takes two steps [1]_: the volume is replaced once by its spline
coefficients (`cubic_bspline_coefficients_`), and each point then reads the
64 coefficients around it (`cubic_grid_sample`). The 64 reads are folded into
eight trilinear ones [2]_, so the sampling stays on ``grid_sample``.

References
----------
.. [1] Unser, M. (1999). Splines: A perfect fit for signal and image
   processing. *IEEE Signal Processing Magazine*, 16(6), 22–38.
   [doi:10.1109/79.799930](https://doi.org/10.1109/79.799930)
.. [2] Ruijters, D., ter Haar Romeny, B. M., & Suetens, P. (2008). Efficient
   GPU-based texture interpolation using uniform B-splines. *Journal of
   Graphics Tools*, 13(4), 61–69.
   [doi:10.1080/2151237X.2008.10129269](https://doi.org/10.1080/2151237X.2008.10129269)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from specter.options import GridSamplePadding


@torch.no_grad()
def cubic_bspline_coefficients_(volume: torch.Tensor) -> torch.Tensor:
    """
    Replace a volume, in place, by its cubic B-spline coefficients.

    The coefficients ``c`` solve ``(c[i-1] + 4 c[i] + c[i+1]) / 6 = v[i]``
    along each of the last three axes, with ``c`` zero outside the volume,
    which is what ``grid_sample``'s zero padding reads there. The tridiagonal
    system is solved plane by plane, so no second volume is allocated.

    Parameters
    ----------
    volume : torch.Tensor
        ``(..., Z, Y, X)``, overwritten.

    Returns
    -------
    torch.Tensor
        `volume`, now holding the coefficients.
    """
    for dim in (-3, -2, -1):
        n = volume.shape[dim]
        # Thomas algorithm for the constant (1, 4, 1) system.
        upper = [0.25]
        for _ in range(n - 1):
            upper.append(1.0 / (4.0 - upper[-1]))
        volume.select(dim, 0).mul_(1.5)
        for i in range(1, n):
            plane = volume.select(dim, i)
            plane.mul_(6.0).sub_(volume.select(dim, i - 1)).mul_(upper[i])
        for i in range(n - 2, -1, -1):
            volume.select(dim, i).sub_(volume.select(dim, i + 1), alpha=upper[i])
    return volume


def _fetch_pair(
    x: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Two linear fetches that together weight the four coefficients around `x`.

    Returns
    -------
    tuple
        ``(weight, position)`` for the lower and the upper fetch, with the
        positions in voxels.
    """
    i = torch.floor(x)
    f = x - i
    w0 = (1 - f) ** 3 / 6
    w1 = (3 * f**3 - 6 * f**2 + 4) / 6
    w3 = f**3 / 6
    g0 = w0 + w1
    g1 = 1 - g0
    return (g0, i - 1 + w1 / g0), (g1, i + 1 + w3 / g1)


def cubic_grid_sample(
    coefficients: torch.Tensor,
    grid: torch.Tensor,
    align_corners: bool,
    padding_mode: GridSamplePadding = "zeros",
) -> torch.Tensor:
    """
    ``grid_sample`` with cubic B-spline interpolation, for a 3D volume.

    Parameters
    ----------
    coefficients : torch.Tensor
        ``(B, C, Z, Y, X)`` spline coefficients, from
        `cubic_bspline_coefficients_`.
    grid : torch.Tensor
        ``(B, K, H, W, 3)`` sample points in ``grid_sample``'s normalised
        ``(x, y, z)`` convention.
    align_corners : bool
        ``grid_sample``'s convention for `grid`.
    padding_mode : {"zeros", "border", "reflection"}, optional
        Passed to each fetch. The coefficients are solved for ``"zeros"``.

    Returns
    -------
    torch.Tensor
        ``(B, C, K, H, W)``.
    """
    sizes = coefficients.shape[-1], coefficients.shape[-2], coefficients.shape[-3]
    pairs = []
    for axis, size in enumerate(sizes):
        g = grid[..., axis]
        voxels = (g + 1) * (size - 1) / 2 if align_corners else ((g + 1) * size - 1) / 2
        pairs.append(
            [
                (w, 2 * p / (size - 1) - 1 if align_corners else (2 * p + 1) / size - 1)
                for w, p in _fetch_pair(voxels)
            ]
        )
    out: torch.Tensor | None = None
    for wz, pz in pairs[2]:
        for wy, py in pairs[1]:
            for wx, px in pairs[0]:
                fetched = F.grid_sample(
                    coefficients,
                    torch.stack((px, py, pz), dim=-1),
                    mode="bilinear",
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                fetched = fetched * (wx * wy * wz).unsqueeze(1)
                out = fetched if out is None else out.add_(fetched)
    assert out is not None
    return out
