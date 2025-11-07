import matplotlib.pyplot as plt
import torch


def plot3d(vol, title=None, vmin=None):
    fig, axes = plt.subplots(1, 3, dpi=200, constrained_layout=True, figsize=(8, 3.6))
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(vol.sum(i), vmin=vmin)
        ax.set(xticks=[], yticks=[], title=f"projection along axis {i}")
        fig.colorbar(im, ax=ax, location="bottom")
    if title is not None:
        plt.suptitle(title, fontsize=15)
    plt.show()


def plot_slices(
    vol, start_idx=0, end_idx=None, axis=0, ylabel=None, vmin=None, vmax=None
):
    nslices = 5
    sh = vol.shape
    if end_idx is None:
        idx_length = sh[axis] - start_idx - 1
        end_idx = start_idx + idx_length
    else:
        idx_length = end_idx - start_idx
    indices = torch.linspace(start_idx, end_idx, nslices).type(torch.int)

    fig, axes = plt.subplots(1, 5, dpi=200, figsize=(8, 2.5), constrained_layout=True)
    for ax, i in zip(axes.ravel(), indices):
        ax.set(xticks=[], yticks=[])
        if axis == 0:
            im = ax.imshow(vol[i, :, :], vmin=vmin, vmax=vmax)
        elif axis == 1:
            im = ax.imshow(vol[:, i, :], vmin=vmin, vmax=vmax)
        elif axis == 2:
            im = ax.imshow(vol[:, :, i], vmin=vmin, vmax=vmax)
        ax.set_title(f"Slice {i}")
        fig.colorbar(im, ax=ax, location="bottom")

    if ylabel is not None:
        axes[0].set_ylabel(ylabel)
    plt.suptitle(f"Slices along axis {axis}")
    plt.show()


# def radial_distribution_function(
#     coords, volume, dr=0.5, r_max=None,
#     number_density=0.03142228327508648,
#     plot=True, device=None, chunk_size=None
# ):
#     """
#     Computes radial distribution function (RDF) from 3D coordinates.

#     Parameters
#     ----------
#     coords : torch.Tensor, shape (N, 3)
#         Atom coordinates.
#     volume : float
#         Volume in A^3.
#     dr : float
#         Width of radial shell/bin in A.
#     r_max : float, optional
#         Maximum radius cutoff. Default: box length.
#     number_density : float
#         N/V in [num / A^3]. Default is for amorphous ice.
#     plot : bool
#         Whether to plot the RDF.
#     device : torch.device or str, optional
#         Device for computation (CPU or CUDA).
#     chunk_size : int or None
#         If None, compute all pairwise distances at once (exact RDF).
#         If int, compute RDF in chunks to save memory.
#     """

#     if device is None:
#         device = coords.device
#     coords = coords.to(device)

#     N = coords.shape[0]
#     if r_max is None:
#         r_max = volume ** (1/3)  # length of cubic box
#     bins = torch.arange(0, r_max + dr, dr, device=device)

#     hist = torch.zeros(len(bins)-1, device=device)

#     if chunk_size is None:
#         # --- Full RDF (all pairs at once) ---
#         dists = torch.cdist(coords, coords)  # (N, N)
#         mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
#         dists = dists[mask]
#         hist = torch.histc(dists, bins=len(bins)-1, min=0, max=r_max)

#     else:
#         # --- Chunked RDF (exact but memory-efficient) ---
#         for i in tqdm(range(0, N, chunk_size)):
#             ci = coords[i:i+chunk_size]  # (≤chunk_size, 3)
#             dists = torch.cdist(ci, coords)  # (len(ci), N)

#             # Mask out self-distances only for the overlapping diagonal part
#             mask = torch.ones_like(dists, dtype=torch.bool)
#             rows = torch.arange(len(ci), device=device)
#             cols = torch.arange(i, i+len(ci), device=device)
#             mask[rows, cols] = 0

#             dists = dists[mask]
#             hist += torch.histc(dists, bins=len(bins)-1, min=0, max=r_max)

#     # RDF normalization
#     r = bins[:-1] + dr/2
#     shell_volume = 4 * torch.pi * r**2 * dr
#     g_r = hist / (number_density * N * shell_volume)

#     # Move to CPU for plotting/output
#     r, g_r = r.cpu().numpy(), g_r.cpu().numpy()

#     if plot:
#         plt.plot(r, g_r)
#         plt.xlabel('r (Å)')
#         plt.ylabel('g(r)')
#         plt.title('Radial Distribution Function')
#         plt.xlim([0, r_max])
#         plt.show()

#     return r, g_r


def radial_distribution_function(
    coords,
    volume,
    dr=0.5,
    r_max=None,
    number_density=0.03142228327508648,
    chunk_size=None,
    approximate=False,
    n_samples=1000000,
    plot=True,
):
    """
    Computes the radial distribution function (RDF) from atomic coordinates.

    The RDF g(r) is defined as:

        g(r) = (1 / (4 * pi * r^2 * rho * N)) *
               < sum_i sum_{j != i} delta(r - |r_i - r_j|) >

    where
      - N is the number of particles
      - rho = N / V is the number density
      - r_i are the particle coordinates
      - delta is the Dirac delta function
      - <> indicates averaging

    Intuitively, g(r) measures how the local density at distance r
    compares to the average density of the system.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3) coordinates in Å.
    volume : float
        Simulation box volume in Å^3.
    dr : float, optional
        Bin width in Å. Default = 0.5.
    r_max : float, optional
        Maximum radius to compute. Default = cubic box length.
    number_density : float, optional
        Number density in [num / Å^3].
    chunk_size : int or None
        If None, use torch.pdist (fast, exact).
        If int, compute in chunks (exact, lower memory).
    approximate : bool
        If True, approximate RDF by random sampling O(N).
    n_samples : int
        Number of random pairs in approximate mode.
    plot : bool
        If True, plots g(r).

    Returns
    -------
    r : torch.tensor
        Bin centers.
    g_r : torch.tensor
        Radial distribution function values.
    """

    device = coords.device
    N = coords.shape[0]
    if r_max is None:
        r_max = volume ** (1 / 3)  # cubic box length

    # bins and histogram container
    bins = torch.arange(0, r_max + dr, dr, device=device)
    hist = torch.zeros(len(bins) - 1, device=device)

    # ------------------------
    # 1. Approximate mode
    # ------------------------
    if approximate:
        device = coords.device
        N = coords.shape[0]
        total_unordered = N * (N - 1) // 2

        # draw n_samples ordered pairs but guaranteed i != j
        # efficient trick: draw j from [0..N-2] and bump up where j >= i
        i = torch.randint(0, N, (n_samples,), device=device)
        j = torch.randint(0, N - 1, (n_samples,), device=device)
        j = j + (j >= i).to(dtype=j.dtype)  # now j != i for all entries

        # distances for sampled ordered pairs
        dists = torch.norm(coords[i] - coords[j], dim=1)

        # bin sampled distances
        idx = torch.bucketize(dists, bins) - 1
        idx = idx[(idx >= 0) & (idx < hist.numel())]
        hist.index_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))

        # scale histogram to estimate counts over all unordered pairs
        S_eff = dists.numel()  # actual sampled ordered pairs
        if S_eff > 0:
            hist *= total_unordered / S_eff

    # ------------------------
    # 2. Exact mode: pdist
    # ------------------------
    elif chunk_size is None:
        dists = torch.pdist(coords)  # (N*(N-1)/2,)
        idx = torch.bucketize(dists, bins) - 1
        idx = idx[(idx >= 0) & (idx < hist.numel())]
        hist.index_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))

    # ------------------------
    # 3. Exact mode: chunked cdist
    # ------------------------
    else:
        for i in range(0, N, chunk_size):
            ci = coords[i : i + chunk_size]  # (m, 3)
            cj = coords[i:]  # (N-i, 3)
            m = ci.shape[0]
            if m == 0:
                continue

            D = torch.cdist(ci, cj)  # (m, N-i)

            # split into in-block vs tail
            D_block = D[:, :m]
            D_tail = D[:, m:]

            # strictly upper-triangle of block
            if m > 1:
                tri = torch.triu(
                    torch.ones((m, m), dtype=torch.bool, device=device), diagonal=1
                )
                d_block = D_block[tri]
                dists = torch.cat([d_block, D_tail.reshape(-1)])
            else:
                dists = D_tail.reshape(-1)

            # binning
            idx = torch.bucketize(dists, bins) - 1
            idx = idx[(idx >= 0) & (idx < hist.numel())]
            hist.index_add_(0, idx, torch.ones_like(idx, dtype=hist.dtype))

    # ------------------------
    # Normalize RDF
    # ------------------------
    r = bins[:-1] + dr / 2
    shell_volume = 4 * torch.pi * r**2 * dr
    g_r = hist / (number_density * N * shell_volume)

    if plot:
        plt.plot(r.cpu().numpy(), g_r.cpu().numpy())
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function")
        plt.xlim([0, r_max])
        plt.show()

    return r, g_r
