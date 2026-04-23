from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from .arrays import radial_profile_2d
from .coords import radial_distribution_function


def plot3d(
    vol: torch.Tensor,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
) -> None:
    """
    Plot 3 orthogonal projections of a 3D volume.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume tensor.
    title : str or None, optional
        Super title for the plot.
    vmin : float or None, optional
        Minimum value for colormap scaling.
    vmax : float or None, optional
        Maximum value for colormap scaling.
    cmap : str or None, optional
        Matplotlib colormap name. Default is None (uses matplotlib default).
    """
    fig, axes = plt.subplots(1, 3, dpi=200, constrained_layout=True, figsize=(8, 3.6))
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(vol.sum(i), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set(xticks=[], yticks=[], title=f"projection along axis {i}")
        fig.colorbar(im, ax=ax, location="bottom")
    if title is not None:
        plt.suptitle(title, fontsize=15)
    plt.show()


try:
    import lightning as L
    from IPython.display import display

    class VolumeMonitorCallback(L.Callback):
        """Lightning callback that plots the reconstructed volume every N training steps."""

        def __init__(self, every_n_steps: int = 100):
            self.every_n_steps = every_n_steps
            self._display_handle = None

        def _plot_volume(self, pl_module, title):
            vol = pl_module.V.data.detach().cpu().float()
            fig, axes = plt.subplots(
                1, 3, dpi=200, constrained_layout=True, figsize=(8, 3.6)
            )
            for i, ax in enumerate(axes.ravel()):
                im = ax.imshow(vol.sum(i), cmap="bone")
                ax.set(xticks=[], yticks=[], title=f"projection along axis {i}")
                fig.colorbar(im, ax=ax, location="bottom")
            plt.suptitle(title, fontsize=15)
            if self._display_handle is None:
                self._display_handle = display(fig, display_id=True)
            else:
                self._display_handle.update(fig)
            plt.close(fig)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % self.every_n_steps != 0:
                return
            total = (
                trainer.max_steps
                if trainer.max_steps > 0
                else trainer.estimated_stepping_batches
            )
            self._plot_volume(pl_module, title=f"Step {trainer.global_step} / {total}")

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            # Plot after the previous epoch's symmetry has been applied
            if batch_idx == 0 and trainer.current_epoch > 0:
                total = (
                    trainer.max_steps
                    if trainer.max_steps > 0
                    else trainer.estimated_stepping_batches
                )
                self._plot_volume(
                    pl_module,
                    title=f"Epoch {trainer.current_epoch} (Step {trainer.global_step} / {total})",
                )

        def on_train_end(self, trainer, pl_module):
            # Plot the final epoch after symmetry
            total = (
                trainer.max_steps
                if trainer.max_steps > 0
                else trainer.estimated_stepping_batches
            )
            self._plot_volume(
                pl_module,
                title=f"Epoch {trainer.current_epoch + 1} (Step {trainer.global_step} / {total})",
            )

except ImportError:
    pass


def plot_slices(
    vol: torch.Tensor,
    start_idx: int = 0,
    end_idx: int | None = None,
    axis: int = 0,
    ylabel: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Plot slices of a 3D volume along a specified axis.

    Parameters
    ----------
    vol : torch.Tensor
        3D volume tensor.
    start_idx : int, optional
        Starting slice index. Default 0.
    end_idx : int or None, optional
        Ending slice index. If None, computed from nslices.
    axis : int, optional
        Axis along which to slice (0, 1, or 2). Default 0.
    ylabel : str or None, optional
        Label for the y-axis of the first subplot.
    vmin : float or None, optional
        Minimum value for colormap.
    vmax : float or None, optional
        Maximum value for colormap.
    """
    nslices = 5
    sh = vol.shape
    if end_idx is None:
        end_idx = sh[axis] - 1
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


def plot_rdf(
    coords: torch.Tensor,
    volume: float,
    dr: float = 0.5,
    r_max: float | None = None,
    number_density: float | None = None,
    chunk_size: int | None = None,
    approximate: bool = False,
    n_samples: int = 1_000_000,
    ax: plt.Axes | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute and plot the radial distribution function g(r).

    Delegates computation to :func:`specter.coords.radial_distribution_function`.

    Parameters
    ----------
    coords : torch.Tensor
        Atom positions, shape (N, 3), in Å.
    volume : float
        System volume in Å³.
    dr : float, optional
        Bin width in Å. Default 0.5.
    r_max : float, optional
        Maximum radius in Å. Defaults to cube-root of ``volume``.
    number_density : float, optional
        Override number density (particles / Å³). If ``None``, computed
        from ``len(coords) / volume``.
    chunk_size : int or None, optional
        Row block size for chunked exact mode. If ``None``, uses
        ``torch.pdist`` (full O(N²)).
    approximate : bool, optional
        Use random-pair-sampling approximation. Default False.
    n_samples : int, optional
        Pairs drawn in approximate mode. Default 1 000 000.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If ``None``, uses the current axes.

    Returns
    -------
    r : torch.Tensor
        Bin-centre radii in Å.
    g_r : torch.Tensor
        Radial distribution function values.
    """
    r, g_r = radial_distribution_function(
        coords,
        volume=volume,
        dr=dr,
        r_max=r_max,
        number_density=number_density,
        chunk_size=chunk_size,
        approximate=approximate,
        n_samples=n_samples,
    )
    if ax is None:
        ax = plt.gca()
    ax.plot(r.cpu().numpy(), g_r.cpu().numpy())
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    if r_max is not None:
        ax.set_xlim([0, r_max])
    return r, g_r


def plot_particle_stack(
    images: torch.Tensor,
    pixel_size: float,
    defocus_values: torch.Tensor | None = None,
    max_images: int = 5,
) -> None:
    """
    Plot cryo-EM image diagnostics: raw images, FFT magnitude, and radial power spectra.

    Parameters
    ----------
    images : torch.Tensor
        Stack of images, shape (N, H, W).
    pixel_size : float
        Pixel size in Angstroms.
    defocus_values : torch.Tensor or None, optional
        Defocus values in Angstroms, shape (N,). Default is None.
    max_images : int, optional
        Maximum number of images to display. Default is 5.
    """
    n = min(len(images), max_images)
    images = images[:n]

    # scale figure width proportionally to number of images
    fig_w = max(2, 2 * n)
    fig_h = 6

    fig, axes = plt.subplots(
        3,
        n,
        dpi=200,
        constrained_layout=True,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [2, 2, 1]},
    )

    # ensure axes is always 2D even for n=1
    if n == 1:
        axes = axes[:, None]

    # share y axis across radial profile row
    for ax in axes[2, 1:]:
        ax.sharey(axes[2, 0])
        plt.setp(ax.get_yticklabels(), visible=False)

    for i in range(n):
        img = images[i]
        img_centered = img - torch.mean(img)
        fft_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(img_centered)))

        # --- row 0: raw image ---
        ax = axes[0, i]
        im = ax.imshow(img, cmap="gray")
        ax.set(xticks=[], yticks=[])
        if defocus_values is not None:
            ax.set_title(
                f"Defocus: {defocus_values[i] / 10000:.2f} $\\mu$m", fontsize=6
            )
        fig.colorbar(im, ax=ax, location="bottom")

        # --- row 1: FFT magnitude ---
        ax = axes[1, i]
        im = ax.imshow(fft_mag, cmap="gray")
        ax.set(xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, location="bottom")

        # --- row 2: radial profile ---
        ax = axes[2, i]
        profile = radial_profile_2d(fft_mag)
        n_bins = len(profile)

        ax.plot(profile)

        tick_positions = [int(j * (n_bins - 1) / 4) for j in range(5)]
        tick_labels = []
        for pos in tick_positions:
            if pos == 0:
                tick_labels.append("∞")
            else:
                freq = (pos / (n_bins - 1)) * (1 / (2 * pixel_size))
                tick_labels.append(f"{1 / freq:.1f}")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=5)
        ax.set_xlim(0, n_bins - 1)

    axes[0, 0].set(ylabel="Images")
    axes[1, 0].set(ylabel="FFT")
    axes[2, 0].set(ylabel="FFT Radial profile")
    axes[2, n // 2].set_xlabel("Resolution (Å)")

    plt.show()
