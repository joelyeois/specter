import matplotlib.pyplot as plt
import torch

def plot3d(vol, title=None):
    fig, axes = plt.subplots(1, 3, dpi=200, constrained_layout=True, figsize=(8,3.6))
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(vol.sum(i))
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
    idx_interval = idx_length // nslices
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

def radial_distribution_function(coords, volume, dr=0.5, r_max=None,
                                 number_density=0.03142228327508648, plot=True):
    """
        Computes radial distribution function from the list of coordinates.

        Parameters
        ----------
        coordinates: tensor
            Shape of (N, 3).
        volume: float
            Volume in A^3.
        dr: float
            Width of radial shell or bin in A
        r_max: float
            Maximum radius to plot
        number_density: float
            N/V in [num / A^3]. Default assumes amorphous ice.
        """

    N = coords.shape[0]
    if r_max is None:
        r_max = volume**(1/3) # length of cube.
    bins = torch.arange(0, r_max + dr, dr)
    
    # Compute pairwise distances using cdist
    distances = torch.cdist(coords, coords)  # (N, N)
    
    # Keep upper triangle only
    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
    distances = distances[mask]
    
    # Histogram
    hist = torch.histc(distances, bins=len(bins)-1, min=0, max=r_max)
    
    # RDF normalization
    r = bins[:-1] + dr/2
    shell_volume = 4 * torch.pi * r**2 * dr
    g_r = hist / (number_density * N * shell_volume)
    
    # Plot
    if plot:
        plt.plot(r.numpy(), g_r.numpy())
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.title('Radial Distribution Function')
        plt.xlim([0, r_max])
        plt.show()
    return r.numpy(), g_r.numpy()