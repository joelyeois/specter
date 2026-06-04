from __future__ import annotations

import torch


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
    relative to the bulk average:

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
    g_r = hist / (number_density * N * shell_volume)
    return r, g_r
