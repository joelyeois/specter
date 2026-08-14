"""
Delaunay worker for ``_carbon._alpha_shape``'s blocked alpha-complex build.

Deliberately imports only numpy and scipy. Workers run under the
``"spawn"`` multiprocessing context (see ``_carbon._alpha_shape``), which
freshly imports this module in every worker process -- pulling torch in
here would cost several seconds per process for no benefit, since none of
this needs a GPU.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def circumspheres(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Circumcenter and circumradius of tetrahedra, in closed form.

    Uses the cross-product formula rather than a batched 3x3 LU solve --
    pure elementwise arithmetic, and degenerate slivers fall out as
    non-finite radii instead of needing a separate determinant mask.

    Parameters
    ----------
    v : np.ndarray
        (T, 4, 3) tetrahedron vertices.

    Returns
    -------
    tuple of np.ndarray
        (T, 3) circumcenters and (T,) circumradii; slivers get ``inf``.
    """
    a = v[:, 0]
    u, w, t = v[:, 1] - a, v[:, 2] - a, v[:, 3] - a
    cross_wt, cross_tu, cross_uw = np.cross(w, t), np.cross(t, u), np.cross(u, w)
    num = (
        (u * u).sum(1)[:, None] * cross_wt
        + (w * w).sum(1)[:, None] * cross_tu
        + (t * t).sum(1)[:, None] * cross_uw
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = num / (2.0 * (u * cross_wt).sum(1)[:, None])
    r = np.linalg.norm(rel, axis=1)
    r[~np.isfinite(r)] = np.inf
    return a + rel, r


def block_job(
    args: tuple[np.ndarray, np.ndarray, float, float, float, float, float],
) -> np.ndarray:
    """
    Triangulate one halo block and return the tets it owns.

    A tetrahedron is owned by the block whose core contains its
    circumcenter, which partitions the accepted set exactly -- see
    ``_carbon._alpha_shape``'s docstring for why this is exact, not an
    approximation.

    Parameters
    ----------
    args : tuple
        ``(points, global_indices, x0, x1, y0, y1, alpha)`` -- `points` are
        the block's halo point set and `global_indices` maps its rows back
        to the full cloud.

    Returns
    -------
    np.ndarray
        (K, 4) tetrahedra as global vertex indices.
    """
    pts, gidx, x0, x1, y0, y1, alpha = args
    if pts.shape[0] < 4:
        return np.empty((0, 4), dtype=np.int64)
    simplices = Delaunay(pts).simplices
    c, r = circumspheres(pts[simplices])
    keep = (
        (r < alpha)
        & (c[:, 0] >= x0)
        & (c[:, 0] < x1)
        & (c[:, 1] >= y0)
        & (c[:, 1] < y1)
    )
    return gidx[simplices[keep]]
