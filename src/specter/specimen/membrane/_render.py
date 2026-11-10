"""
Render a membrane's calibrated potential onto an output grid.

The bilayer potential is a function of the signed distance to the
mid-surface alone, ``psi(d)``. The distance is held on the coarse `Grid`
(`_geometry`), interpolated by a cubic B-spline at each output voxel, and
looked up in a table of `BilayerProfile`. Only the output blocks the
membrane passes through are evaluated, so the cost follows the membrane's
area and the device holds the coarse fields plus one batch of blocks,
whatever the output resolution.

The table is the profile averaged over one output voxel along the normal.
A voxel coarser than the profile's features therefore shows the two
leaflets merging, where point-sampling the profile would alias them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ._field import _centred_origin_xyz
from ._geometry import Grid, bspline_taps, prefilter
from ._profile import BilayerProfile

# Output voxels evaluated per batch of blocks.
_BATCH_VOXELS = 1 << 24
# Spacing of the profile lookup table, A.
_TABLE_STEP = 0.25


def profile_table(
    profile: BilayerProfile, voxel_size: float, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, float, float]:
    """
    ``psi(d)`` on a uniform table, averaged over one voxel along the normal.

    Parameters
    ----------
    profile : BilayerProfile
        Calibrated bilayer profile.
    voxel_size : float
        Output voxel size, Å.
    device : torch.device or str, optional
        Device of the returned table. Default ``"cpu"``.

    Returns
    -------
    table : torch.Tensor
        Potential at ``d0 + i * step``, zero past the profile's ends.
    d0 : float
        Signed distance of the first entry, Å.
    step : float
        Table spacing, Å.
    """
    dist = profile.distance_angstrom.detach().cpu()
    lo, hi = float(dist.min()), float(dist.max())
    half = max(abs(lo), abs(hi)) + voxel_size
    d = torch.arange(-half, half + _TABLE_STEP, _TABLE_STEP)
    cpu = BilayerProfile(distance_angstrom=dist, psi=profile.psi.detach().cpu())
    psi = cpu(d).to(torch.float32)
    psi[(d < lo) | (d > hi)] = 0.0
    w = max(1, int(round(voxel_size / _TABLE_STEP)))
    if w > 1:
        psi = F.avg_pool1d(
            F.pad(psi.view(1, 1, -1), (w // 2, w - 1 - w // 2), mode="replicate"),
            w,
            stride=1,
        ).view(-1)
    return psi.to(device), -half, _TABLE_STEP


def render_bilayer(
    surfaces: Sequence[torch.Tensor],
    grid: Grid,
    profile: BilayerProfile,
    shape_zyx: tuple[int, int, int],
    voxel_size: float,
    origin_xyz: torch.Tensor | Sequence[float] | None = None,
    block: int = 64,
    out_device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Bilayer potential of one or more membranes on an output grid.

    Parameters
    ----------
    surfaces : sequence of torch.Tensor
        Signed distance of each membrane's mid-surface on `grid`, indexed
        ``(x, y, z)``. Their potentials add.
    grid : Grid
        Coarse grid the distances live on.
    profile : BilayerProfile
        Calibrated ``psi(d)``.
    shape_zyx : tuple of int
        Output shape ``(Z, Y, X)``.
    voxel_size : float
        Output voxel size, Å.
    origin_xyz : torch.Tensor or sequence of float, optional
        Physical ``(x, y, z)`` of output voxel ``(0, 0, 0)``, Å. Default
        puts the physical origin on voxel ``n // 2`` of each axis.
    block : int, optional
        Edge of the cubic blocks the output is evaluated in, voxels.
        Default 64.
    out_device : torch.device or str, optional
        Device of the returned volume. Default the grid's.

    Returns
    -------
    torch.Tensor
        Potential, shape `shape_zyx`.
    """
    dev = grid.device
    out_device = dev if out_device is None else torch.device(out_device)
    if origin_xyz is None:
        origin_xyz = _centred_origin_xyz(shape_zyx, voxel_size)
    lo = torch.as_tensor(origin_xyz, dtype=torch.float32).to(dev)
    shape = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
    out = torch.zeros(shape_zyx, dtype=torch.float32, device=out_device)
    table, d0, step = profile_table(profile, voxel_size, dev)
    band = -d0
    span = block * voxel_size
    n_blocks = torch.tensor([int(math.ceil(s / block)) for s in shape], device=dev)
    size = torch.tensor(grid.shape, device=dev, dtype=torch.float32)
    local = torch.arange(block, device=dev, dtype=torch.float32) * voxel_size
    batch = max(1, _BATCH_VOXELS // block**3)
    view = [(-1, block, 1, 1), (-1, 1, block, 1), (-1, 1, 1, block)]
    for phi in surfaces:
        coef = prefilter(phi)[None, None]
        near = (phi.abs() < band + 2 * grid.h).nonzero().float() * grid.h + grid.origin
        ids: set[tuple[int, ...]] = set()
        # A node's cell reaches one spacing either way.
        for shift in (-grid.h, grid.h):
            b = torch.floor((near + shift - lo + voxel_size / 2) / span).long()
            ok = ((b >= 0) & (b < n_blocks)).all(1)
            ids.update(map(tuple, torch.unique(b[ok], dim=0).tolist()))
        todo = sorted(ids)
        for i0 in range(0, len(todo), batch):
            chunk = todo[i0 : i0 + batch]
            n = len(chunk)
            org = torch.tensor(chunk, device=dev, dtype=torch.float32) * span + lo
            taps = [
                bspline_taps(
                    (org[:, k].view(-1, 1) + local.view(1, -1) - grid.origin[k])
                    / grid.h
                )
                for k in range(3)
            ]
            d = torch.zeros(n, block, block, block, device=dev)
            for corner in range(8):
                sel = [(corner >> k) & 1 for k in range(3)]
                w: torch.Tensor | float = 1.0
                pos = []
                for k in range(3):
                    gk, pk = taps[k][sel[k]]
                    w = w * gk.view(view[k])
                    pos.append(
                        (pk / (size[k] - 1) * 2 - 1)
                        .view(view[k])
                        .expand(n, block, block, block)
                    )
                d += (
                    w
                    * F.grid_sample(
                        coef.expand(n, 1, *phi.shape),
                        torch.stack(pos[::-1], -1),
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )[:, 0]
                )
            t = ((d - d0) / step).clamp_(0, table.numel() - 1.001)
            i = t.long()
            frac = t - i
            val = (table[i] * (1 - frac) + table[i + 1] * frac).to(out_device)
            for j, (bx, by, bz) in enumerate(chunk):
                x0, y0, z0 = bx * block, by * block, bz * block
                nx = min(x0 + block, shape[0]) - x0
                ny = min(y0 + block, shape[1]) - y0
                nz = min(z0 + block, shape[2]) - z0
                out[z0 : z0 + nz, y0 : y0 + ny, x0 : x0 + nx] += val[
                    j, :nx, :ny, :nz
                ].permute(2, 1, 0)
    return out


def render_interior(
    phi: torch.Tensor,
    grid: Grid,
    shape_zyx: tuple[int, int, int],
    voxel_size: float,
    origin_xyz: torch.Tensor | Sequence[float] | None = None,
    out_device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Which output voxels lie inside a membrane's solid.

    The signed distance is interpolated trilinearly, a z-slab at a time.
    Its error is far below the bilayer's half-thickness for a surface with
    the regularised reach, so a voxel misclassified near the mid-surface is
    one the bilayer's shell already covers.

    Parameters
    ----------
    phi : torch.Tensor
        Signed distance of the membrane's mid-surface on `grid`, indexed
        ``(x, y, z)``, negative inside the solid.
    grid : Grid
        Coarse grid `phi` lives on.
    shape_zyx : tuple of int
        Output shape ``(Z, Y, X)``.
    voxel_size : float
        Output voxel size, Å.
    origin_xyz : torch.Tensor or sequence of float, optional
        Physical ``(x, y, z)`` of output voxel ``(0, 0, 0)``, Å. Default
        puts the physical origin on voxel ``n // 2`` of each axis.
    out_device : torch.device or str, optional
        Device of the returned mask. Default the grid's.

    Returns
    -------
    torch.Tensor
        Boolean mask, shape `shape_zyx`, True inside the solid.
    """
    dev = grid.device
    out_device = dev if out_device is None else torch.device(out_device)
    if origin_xyz is None:
        origin_xyz = _centred_origin_xyz(shape_zyx, voxel_size)
    lo = torch.as_tensor(origin_xyz, dtype=torch.float32).to(dev)
    nz, ny, nx = shape_zyx
    out = torch.zeros(shape_zyx, dtype=torch.bool, device=out_device)
    # (Z, Y, X) as (D, H, W), so a sample point's (x, y, z) indexes (W, H, D).
    field = phi.permute(2, 1, 0)[None, None].to(torch.float32)
    size = torch.tensor(grid.shape, device=dev, dtype=torch.float32)

    def _norm(n: int, k: int) -> torch.Tensor:
        p = lo[k] + torch.arange(n, device=dev, dtype=torch.float32) * voxel_size
        return (p - grid.origin[k]) / grid.h / (size[k] - 1) * 2 - 1

    gx, gy = _norm(nx, 0), _norm(ny, 1)
    gz_all = _norm(nz, 2)
    slab = max(1, _BATCH_VOXELS // max(1, nx * ny))
    for z0 in range(0, nz, slab):
        gz = gz_all[z0 : z0 + slab]
        pts = torch.stack(
            torch.broadcast_tensors(
                gx.view(1, 1, -1), gy.view(1, -1, 1), gz.view(-1, 1, 1)
            ),
            -1,
        )
        val = F.grid_sample(
            field, pts[None], mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]
        out[z0 : z0 + gz.numel()] = (val < 0).to(out_device)
    return out


__all__ = ["profile_table", "render_bilayer", "render_interior"]
