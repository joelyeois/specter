"""
Coordinate sampling and statistics: Poisson-disk neighbour placement in 2-D
and 3-D, and the radial distribution function.
"""

from __future__ import annotations

from typing import Literal

import torch

from .cpu_threads import limited_cpu_threads


def poisson_disk_neighbors(
    min_distance: float,
    n_points: int | float = torch.inf,
    box: tuple[int, int] = (256, 256),  # (height, width)
    k: int = 30,
    seed: Literal["origin", "random"] = "origin",
) -> torch.Tensor:
    """
    2D Poisson-disk sampling in a rectangular box centered at the origin.

    Parameters
    ----------
    min_distance : float
        Minimum spacing between points.
    n_points : int
        Total number of points to generate (including seed).
    box : tuple of int
        (height, width) of the bounding box in pixels. Origin is at (0,0).
        Valid coordinates are y in [-H/2, H/2), x in [-W/2, W/2).
    k : int
        Number of candidate points to try per active point.
    seed : {"origin", "random"}
        If "origin", first point is at the center (0,0).
        If "random", first point is chosen uniformly inside the box.

    Returns
    -------
    pts : (m,2) torch.Tensor
        Sampled 2D coordinates, including the seed point.
    """
    H, W = box
    y_min, y_max = -H // 2, H // 2
    x_min, x_max = -W // 2, W // 2

    # initialize first point
    if seed == "origin":
        first_point = torch.tensor([0.0, 0.0])
        n_points += 1  # don't count origin.
    elif seed == "random":
        y = (y_max - y_min) * torch.rand(1) + y_min
        x = (x_max - x_min) * torch.rand(1) + x_min
        first_point = torch.tensor([y.item(), x.item()])
    else:
        raise ValueError("seed must be 'origin' or 'random'")

    pts = [first_point]
    active = [0]

    # Every op in this loop is on a few hundred elements; see cpu_threads.
    with limited_cpu_threads(1):
        while active and len(pts) < n_points:
            idx = int(torch.randint(len(active), (1,)).item())
            center_point = pts[active[idx]]

            # generate k candidates in annulus [min_distance, 2*min_distance]
            theta = torch.rand(k) * 2 * torch.pi
            radius = min_distance + min_distance * torch.rand(k)
            candidates = center_point.unsqueeze(0) + torch.stack(
                (radius * torch.cos(theta), radius * torch.sin(theta)), dim=1
            )

            # reject candidates outside the centered box
            mask = (
                (candidates[:, 0] >= y_min)
                & (candidates[:, 0] < y_max)
                & (candidates[:, 1] >= x_min)
                & (candidates[:, 1] < x_max)
            )
            candidates = candidates[mask]

            if candidates.shape[0] == 0:
                active.pop(idx)
                continue

            # distance check against all existing points
            pts_tensor = torch.stack(pts)
            diff = candidates[:, None, :] - pts_tensor[None, :, :]
            dist2 = (diff**2).sum(dim=2)
            min_dist2, _ = dist2.min(dim=1)
            candidates = candidates[min_dist2 >= min_distance**2]

            if candidates.shape[0] > 0:
                pts.append(candidates[0])
                active.append(len(pts) - 1)
            else:
                active.pop(idx)

        if n_points == torch.inf:
            return torch.stack(pts)
        else:
            return torch.stack(pts[: int(n_points)])


def poisson_disk_neighbors_3d(
    min_distance: float,
    n_points: int | float = torch.inf,
    box: tuple[float, float, float] = (256.0, 256.0, 256.0),  # (D,H,W)
    k: int = 30,
    seed: Literal["origin", "random"] = "origin",
) -> torch.Tensor:
    """
    Generate 3D points using Poisson-disk sampling within a 3D tensor volume.

    This function produces points in a (D, H, W) volume such that no two points
    are closer than `min_distance` pixels, creating a uniform but spatially
    separated distribution.

    Parameters
    ----------
    min_distance : float
        Minimum allowed distance between points, in Å.
    n_points : int or torch.inf, optional
        Maximum number of points to generate. Default is infinite (fill the volume).
    box : tuple of float, optional
        Dimensions of the 3D volume in Å, as (D, H, W). Default is (256, 256, 256).
    k : int, optional
        Number of candidate points to generate around each active point. Higher
        values produce denser sampling. Default is 30.
    seed : {'origin', 'random'}, optional
        Determines the initial seed point:
        - 'origin': starts at (0, 0, 0)
        - 'random': starts at a random location within the box

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3), where each row is a point coordinate (x, y, z) in Å.

    Notes
    -----
    - The coordinate system is centered at (0,0,0), with ranges:
      x ∈ [-W/2, W/2], y ∈ [-H/2, H/2], z ∈ [-D/2, D/2].
    - The function uses Bridson's Poisson-disk sampling; the neighbour test is
      one vectorised distance check against every accepted point.
    """
    D, H, W = box
    z_min, z_max = -D / 2, D / 2
    y_min, y_max = -H / 2, H / 2
    x_min, x_max = -W / 2, W / 2

    # initialize first point
    if seed == "origin":
        first_point = torch.tensor([0.0, 0.0, 0.0])  # x,y,z
        n_points += 1
    elif seed == "random":
        x = (x_max - x_min) * torch.rand(1) + x_min
        y = (y_max - y_min) * torch.rand(1) + y_min
        z = (z_max - z_min) * torch.rand(1) + z_min
        first_point = torch.tensor([x.item(), y.item(), z.item()])
    else:
        raise ValueError("seed must be 'origin' or 'random'")

    pts = [first_point]
    active = [0]
    # Accepted points as one (N, 3) tensor, for the vectorised distance test
    # below. Grown by concatenation, which is O(N^2) in total but for the
    # point counts a Poisson-disk fill produces (tens to a few thousand) that
    # is far below the cost of the Python-level grid scan it replaces: the
    # earlier cell-grid neighbour search did up to 125 ``.item()`` reads per
    # candidate, ~30k for the ~8 points a single-particle box holds, 0.18 s.
    pts_t = first_point.unsqueeze(0)

    # Every op in this loop is on a few hundred elements; see cpu_threads.
    with limited_cpu_threads(1):
        while active and len(pts) < n_points:
            idx = int(torch.randint(len(active), (1,)).item())
            center_point = pts[active[idx]]

            # generate k candidates in spherical shell
            phi = torch.acos(2 * torch.rand(k) - 1)
            theta = 2 * torch.pi * torch.rand(k)
            r = min_distance * (1 + torch.rand(k))

            dx = r * torch.sin(phi) * torch.cos(theta)
            dy = r * torch.sin(phi) * torch.sin(theta)
            dz = r * torch.cos(phi)
            candidates = center_point.unsqueeze(0) + torch.stack([dx, dy, dz], dim=1)

            # filter candidates in tensor bounds (z,y,x)
            mask = (
                (candidates[:, 0] >= x_min)
                & (candidates[:, 0] < x_max)
                & (candidates[:, 1] >= y_min)
                & (candidates[:, 1] < y_max)
                & (candidates[:, 2] >= z_min)
                & (candidates[:, 2] < z_max)
            )
            candidates = candidates[mask]

            if candidates.shape[0] == 0:
                active.pop(idx)
                continue

            # A candidate is accepted if no accepted point lies within
            # min_distance of it; the first such candidate (in draw order) is
            # taken, exactly as the per-candidate grid scan did.
            nearest = torch.cdist(candidates, pts_t).min(dim=1).values
            ok = torch.nonzero(~(nearest < min_distance)).flatten()
            if len(ok) > 0:
                new_pt = candidates[ok[0]]
                pts.append(new_pt)
                pts_t = torch.cat([pts_t, new_pt.unsqueeze(0)], dim=0)
                active.append(len(pts) - 1)
            else:
                active.pop(idx)

        if seed == "origin":
            if len(pts) == 1:
                return torch.empty((0, 3))
            pts = pts[1:]  # don't include origin
            return torch.stack(pts if n_points == torch.inf else pts[: int(n_points)])
        elif seed == "random":
            return torch.stack(pts if n_points == torch.inf else pts[: int(n_points)])


def radial_distribution_function(
    coords: torch.Tensor,
    volume: float,
    dr: float = 0.5,
    r_max: float | None = None,
    number_density: float | None = None,
    chunk_size: int | None = None,
    approximate: bool = False,
    n_samples: int = 1_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the radial distribution function (RDF) g(r) from atomic coordinates.

    g(r) is defined as the probability of finding a particle at distance r
    relative to the bulk average::

        g(r) = (1 / (4π r² ρ N)) * Σ_i Σ_{j≠i} δ(r - |r_i - r_j|)

    Coordinates are assumed to be centred at the origin. Any atom lying outside
    the cubic box of side ``volume^(1/3)`` is silently discarded before
    computing pairwise distances, so the density normalisation remains
    self-consistent. No periodic boundary conditions are applied.

    Three computation modes (controlled by ``chunk_size`` and ``approximate``):

    - **Exact / full** (``approximate=False``, ``chunk_size=None``):
      ``torch.pdist``, O(N²) memory — fastest for N ≲ 10 000 on GPU.
    - **Exact / chunked** (``approximate=False``, ``chunk_size=int``):
      row-blocked ``torch.cdist``, lower peak memory — use for large N on CPU.
    - **Approximate** (``approximate=True``):
      O(N) random pair sampling; scales to very large N at the cost of
      statistical noise proportional to ``1/√n_samples``.

    Parameters
    ----------
    coords : torch.Tensor
        Atom positions, shape (N, 3), in Å.
    volume : float
        System volume in Å³ used for number-density normalisation.
    dr : float, optional
        Histogram bin width in Å. Default 0.5.
    r_max : float, optional
        Maximum radius to include in Å. Defaults to the cube-root of
        ``volume`` (i.e. the side length of a cube with that volume).
    number_density : float, optional
        Override number density ρ (particles / Å³). If ``None``,
        computed as ``len(coords) / volume``.
    chunk_size : int or None, optional
        Row block size for chunked exact mode. Ignored when
        ``approximate=True``. If ``None``, uses ``torch.pdist``.
    approximate : bool, optional
        If ``True``, use random-pair-sampling approximation. Default False.
    n_samples : int, optional
        Number of random pairs drawn in approximate mode. Default 1 000 000.

    Returns
    -------
    r : torch.Tensor
        Bin-centre radii in Å, shape (n_bins,).
    g_r : torch.Tensor
        Radial distribution function values, shape (n_bins,).
    """
    device = coords.device

    half_side = (volume ** (1.0 / 3.0)) / 2.0
    mask = (coords.abs() <= half_side).all(dim=1)
    coords = coords[mask]

    N = coords.shape[0]

    if r_max is None:
        r_max = volume ** (1.0 / 3.0)
    if number_density is None:
        number_density = N / volume

    bins = torch.arange(0, r_max + dr, dr, device=device)
    hist = torch.zeros(len(bins) - 1, device=device)

    # ------------------------------------------------------------------
    # 1. Approximate mode — O(n_samples) random pair sampling
    # ------------------------------------------------------------------
    if approximate:
        total_unordered = N * (N - 1) // 2
        i = torch.randint(0, N, (n_samples,), device=device)
        j = torch.randint(0, N - 1, (n_samples,), device=device)
        j = j + (j >= i).to(dtype=j.dtype)
        dists = torch.norm(coords[i] - coords[j], dim=1)
        idx = torch.bucketize(dists, bins) - 1
        idx = idx[(idx >= 0) & (idx < hist.numel())]
        hist.index_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))
        S_eff = dists.numel()
        if S_eff > 0:
            hist *= total_unordered / S_eff

    # ------------------------------------------------------------------
    # 2. Exact mode — full O(N²) via torch.pdist
    # ------------------------------------------------------------------
    elif chunk_size is None:
        dists = torch.pdist(coords)
        idx = torch.bucketize(dists, bins) - 1
        idx = idx[(idx >= 0) & (idx < hist.numel())]
        hist.index_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))

    # ------------------------------------------------------------------
    # 3. Exact mode — chunked O(N²) via torch.cdist (lower peak memory)
    # ------------------------------------------------------------------
    else:
        for start in range(0, N, chunk_size):
            ci = coords[start : start + chunk_size]
            cj = coords[start:]
            m = ci.shape[0]
            if m == 0:
                continue
            D = torch.cdist(ci, cj)
            D_block = D[:, :m]
            D_tail = D[:, m:]
            if m > 1:
                tri = torch.triu(
                    torch.ones((m, m), dtype=torch.bool, device=device), diagonal=1
                )
                dists = torch.cat([D_block[tri], D_tail.reshape(-1)])
            else:
                dists = D_tail.reshape(-1)
            idx = torch.bucketize(dists, bins) - 1
            idx = idx[(idx >= 0) & (idx < hist.numel())]
            hist.index_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))

    r = bins[:-1] + dr / 2
    shell_volume = 4 * torch.pi * r**2 * dr
    # Every branch above histograms each pair once: pdist and the chunked
    # upper-triangle enumerate unordered pairs, and the sampled branch
    # rescales its draws by N(N-1)/2 to match. The definition above sums over
    # ordered pairs (i, j != i), which counts each twice, hence the 2. Without
    # it an ideal gas plateaus at 0.5 rather than 1.
    g_r = 2.0 * hist / (number_density * N * shell_volume)
    return r, g_r
