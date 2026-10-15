"""
Rotation-minimizing (parallel-transport) frames along a path.

Both filament orientations (``_placement.filament_orientations``, which
adds the helical twist on top) and tubes use them. A **tube** needs them
most: its protofilaments sit at fixed azimuths around the axis, so any roll
drift between consecutive path points shears them relative to one another
and the lattice comes apart along the bend.

A parallel-transport frame carries one reference normal along the path,
removing at each step only the component that the new tangent has made
invalid. That is the minimum rotation needed to stay orthogonal, so no
spurious twist accumulates: a discrete form of Bishop's relatively
parallel adapted frame [1]_.

References
----------
.. [1] Bishop, R. L. (1975). There is more than one way to frame a curve.
   *The American Mathematical Monthly*, 82(3), 246–251.
   https://doi.org/10.1080/00029890.1975.11993807
"""

from __future__ import annotations

import torch


def parallel_transport_frames(positions_xyz: torch.Tensor) -> torch.Tensor:
    """
    Per-point orthonormal frames that do not twist along the path.

    Parameters
    ----------
    positions_xyz : torch.Tensor
        Path points, shape ``(n, 3)``, physical ``(x, y, z)``.

    Returns
    -------
    torch.Tensor
        Rotation matrices, shape ``(n, 3, 3)``, whose columns are
        ``[normal, binormal, tangent]``. Column 2 (``R[:, :, 2]``) is the
        unit tangent, so ``R[i] @ [0, 0, 1]`` points along the path -- the
        same convention `filament_orientations` uses. A single-point path
        gets the identity.

    Notes
    -----
    The tangent at point ``i`` points towards ``i + 1``; the last point
    reuses its predecessor's tangent, matching `filament_tangents`.
    """
    n = positions_xyz.shape[0]
    dtype, device = positions_xyz.dtype, positions_xyz.device
    if n == 1:
        return torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

    diffs = positions_xyz[1:] - positions_xyz[:-1]
    tangents = diffs / diffs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tangents = torch.cat([tangents, tangents[-1:]], dim=0)

    # Seed normal: any direction not parallel to the first tangent.
    seed = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    if bool((seed * tangents[0]).sum().abs() > 0.9):
        seed = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

    frames = torch.empty((n, 3, 3), dtype=dtype, device=device)
    normal = seed
    for i in range(n):
        # Project out whatever the new tangent has invalidated -- this is
        # the parallel transport step, and it is what keeps the frame from
        # spinning about the tangent.
        normal = normal - (normal * tangents[i]).sum() * tangents[i]
        norm = normal.norm()
        if float(norm) < 1e-6:
            # Degenerate only if the path doubled back exactly; reseed.
            normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
            normal = normal - (normal * tangents[i]).sum() * tangents[i]
            norm = normal.norm()
        normal = normal / norm.clamp_min(1e-8)
        binormal = torch.linalg.cross(tangents[i], normal)
        frames[i] = torch.stack([normal, binormal, tangents[i]], dim=1)
    return frames
