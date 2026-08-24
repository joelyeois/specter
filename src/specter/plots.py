from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import torch
from PIL import Image as PILImage

from .arrays import radial_profile_2d, radial_symmetrize
from .coords import radial_distribution_function
from .fft import fourier_shell_correlation

# seaborn's "deep" categorical palette, hardcoded to avoid a seaborn dependency
# for two matplotlib-only plotting helpers.
_DEEP_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def _deep_palette(n_colors: int) -> list[str]:
    """Cycle through :data:`_DEEP_PALETTE` to get ``n_colors`` colors."""
    return [_DEEP_PALETTE[i % len(_DEEP_PALETTE)] for i in range(n_colors)]


def _despine(ax: matplotlib.axes.Axes, offset: float = 0) -> None:
    """Hide top/right spines and move the rest outward, seaborn-``despine``-style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", offset))


def plot3d(
    volume: torch.Tensor,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure | None:
    """
    Plot 3 orthogonal projections of a 3D volume.

    Parameters
    ----------
    volume : torch.Tensor
        3D volume tensor.
    title : str or None, optional
        Super title for the plot.
    vmin : float or None, optional
        Minimum value for colormap scaling.
    vmax : float or None, optional
        Maximum value for colormap scaling.
    cmap : str or None, optional
        Matplotlib colormap name. Default is None (uses matplotlib default).
    show : bool, optional
        Whether to display the figure immediately. Set ``False`` to get the
        figure object back (e.g. for saving to disk). Default is True.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when ``show=True`` (figure is displayed and closed).
        The figure object when ``show=False``; caller is responsible for
        closing it.
    """
    fig, axes = plt.subplots(1, 3, dpi=200, constrained_layout=True, figsize=(8, 3.6))
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(volume.sum(i), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set(xticks=[], yticks=[], title=f"projection along axis {i}")
        fig.colorbar(im, ax=ax, location="bottom")
    if title is not None:
        fig.suptitle(title, fontsize=15)
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


try:
    import lightning as L
    from IPython.display import display

    class VolumeMonitorCallback(L.Callback):
        """Lightning callback that plots the reconstructed volume every N training steps."""

        def __init__(self, every_n_steps: int = 100):
            self.every_n_steps = every_n_steps
            self._display_handle = None

        def _plot_volume(self, pl_module: L.LightningModule, title: str) -> None:
            volume = pl_module.V.data.detach().cpu().float()
            fig, axes = plt.subplots(
                1, 3, dpi=200, constrained_layout=True, figsize=(8, 3.6)
            )
            for i, ax in enumerate(axes.ravel()):
                im = ax.imshow(volume.sum(i), cmap="bone")
                ax.set(xticks=[], yticks=[], title=f"projection along axis {i}")
                fig.colorbar(im, ax=ax, location="bottom")
            plt.suptitle(title, fontsize=15)
            if self._display_handle is None:
                self._display_handle = display(fig, display_id=True)
            else:
                self._display_handle.update(fig)
            plt.close(fig)

        def on_train_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            # Rank-0-only under multi-GPU (DDP): pl_module.V is already
            # identical across replicas (DDP gradient all-reduce every
            # step), so every rank would otherwise plot/display redundantly.
            if not trainer.is_global_zero:
                return
            if trainer.global_step % self.every_n_steps != 0:
                return
            total = (
                trainer.max_steps
                if trainer.max_steps > 0
                else trainer.estimated_stepping_batches
            )
            self._plot_volume(pl_module, title=f"Step {trainer.global_step} / {total}")

        def on_train_batch_start(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            batch: Any,
            batch_idx: int,
        ) -> None:
            if not trainer.is_global_zero:
                return
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

        def on_train_end(
            self, trainer: L.Trainer, pl_module: L.LightningModule
        ) -> None:
            if not trainer.is_global_zero:
                return
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
    volume: torch.Tensor,
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
    volume : torch.Tensor
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
    sh = volume.shape
    if end_idx is None:
        end_idx = sh[axis] - 1
    indices = torch.linspace(start_idx, end_idx, nslices).type(torch.int)

    fig, axes = plt.subplots(1, 5, dpi=200, figsize=(8, 2.5), constrained_layout=True)
    for ax, i in zip(axes.ravel(), indices):
        ax.set(xticks=[], yticks=[])
        if axis == 0:
            im = ax.imshow(volume[i, :, :], vmin=vmin, vmax=vmax)
        elif axis == 1:
            im = ax.imshow(volume[:, i, :], vmin=vmin, vmax=vmax)
        elif axis == 2:
            im = ax.imshow(volume[:, :, i], vmin=vmin, vmax=vmax)
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
        ax.set_xlim((0, r_max))
    return r, g_r


def plot_particle_stack(
    images: torch.Tensor,
    pixel_size: float,
    defocus_values: torch.Tensor | None = None,
    max_images: int = 5,
    fftmax: float | None = None,
) -> None:
    """
    Plot cryo-EM image diagnostics: raw images, FFT magnitude, and radial power spectra.

    Parameters
    ----------
    images : torch.Tensor
        Stack of images, shape (N, H, W).
    pixel_size : float
        Pixel size in Å.
    defocus_values : torch.Tensor or None, optional
        Defocus values in Å, shape (N,). Default is None.
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

        # --- row 1: FFT magnitude (left half = radially symmetrized, right half = actual) ---
        ax = axes[1, i]
        fft_sym = radial_symmetrize(fft_mag)
        half = fft_mag.shape[1] // 2
        fft_display = torch.cat([fft_sym[:, :half], fft_mag[:, half:]], dim=1)
        im = ax.imshow(fft_display, cmap="gray", vmax=fftmax)
        ax.set(xticks=[], yticks=[])
        kw = dict(
            fontsize=7,
            color="white",
            va="top",
            bbox=dict(boxstyle="square,pad=0.1", fc="black", alpha=0.4, lw=0),
        )
        ax.text(2, 2, "radial avg", ha="left", **kw)
        ax.text(fft_display.shape[1] - 2, 2, "actual", ha="right", **kw)
        ax.axvline(half - 0.5, color="white", linewidth=0.6, linestyle=":")

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
    if fftmax is not None:
        axes[2, 0].set_ylim(top=fftmax)
    axes[2, n // 2].set_xlabel("Resolution (Å)")

    plt.show()


#: Gold-standard half-map FSC threshold (two independent half-set reconstructions).
HALFMAP_FSC_THRESHOLD = 0.143
#: Map-to-model FSC threshold (a reconstruction against a known reference).
MAP_TO_MODEL_FSC_THRESHOLD = 0.5


def fsc_resolution(
    k: torch.Tensor,
    fsc: torch.Tensor,
    threshold: float,
    k_max: float | None = None,
) -> str:
    """
    Resolution at the last FSC threshold crossing, linearly interpolated.

    The *last* crossing rather than the first: a noisy FSC can dip below the
    threshold and recover, and the conservative reading is the highest
    frequency beyond which it stays below.

    Parameters
    ----------
    k : torch.Tensor
        Spatial frequencies (1/Å), one per shell.
    fsc : torch.Tensor
        FSC value per shell, same length as ``k``.
    threshold : float
        Crossing to interpolate -- `HALFMAP_FSC_THRESHOLD` for two half-sets,
        `MAP_TO_MODEL_FSC_THRESHOLD` against a known reference.
    k_max : float, optional
        Ignore shells above this frequency (pass the Nyquist frequency to stop
        a crossing in the aliased tail from being reported as a resolution).

    Returns
    -------
    str
        Resolution formatted as ``"1.397 Å"``, or ``">Nyquist"`` when the FSC
        never crosses the threshold.
    """
    kf = k.cpu().float()
    ff = fsc.cpu().float()
    last_k_cross: torch.Tensor | None = None
    for i in range(len(ff) - 1):
        if k_max is not None and kf[i] > k_max:
            break
        if ff[i] >= threshold > ff[i + 1]:
            t = (threshold - ff[i]) / (ff[i + 1] - ff[i])
            last_k_cross = kf[i] + t * (kf[i + 1] - kf[i])
    if last_k_cross is not None:
        return f"{1.0 / last_k_cross.item():.3f} Å"
    return ">Nyquist"


def resolution_between(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    voxel_size: float,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> str:
    """
    Resolution at an FSC threshold between two volumes, optionally masked.

    The numeric counterpart to `plot_map_to_model_fsc`/`plot_halfmap_fsc`,
    for callers that want to record a resolution rather than draw one. Those
    two return only the *unmasked* resolution even when given a mask, so a
    masked figure and a masked number cannot come from the same call.

    Parameters
    ----------
    volume_a, volume_b : torch.Tensor
        Volumes to correlate, shape ``(Z, Y, X)``. Two half-sets for a
        gold-standard resolution, or a reconstruction and its reference for
        map-to-model.
    voxel_size : float
        Voxel size in Å.
    threshold : float
        `HALFMAP_FSC_THRESHOLD` for two half-sets, `MAP_TO_MODEL_FSC_THRESHOLD`
        against a known reference.
    mask : torch.Tensor, optional
        Applied to both volumes before correlation. None correlates them as-is.

    Returns
    -------
    str
        Resolution formatted as ``"1.397 Å"``, or ``">Nyquist"``.
    """
    a = volume_a.detach().cpu().float()
    b = volume_b.detach().cpu().float()
    if mask is not None:
        m = mask.detach().cpu().float()
        a, b = a * m, b * m
    k, fsc = fourier_shell_correlation(a, b, pixelsize=voxel_size)
    return fsc_resolution(k, fsc, threshold, k_max=1.0 / (2.0 * voxel_size))


def plot_map_to_model_fsc(
    volumes: torch.Tensor | list[torch.Tensor],
    reference: torch.Tensor,
    voxel_size: float,
    mask: torch.Tensor | None = None,
    labels: list[str] | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure | None:
    """
    Compute and plot map-to-model Fourier Shell Correlations.

    Each input volume is correlated against ``reference``. When ``mask`` is
    provided, both an unmasked and a masked FSC are computed for each volume.
    The resolution at FSC = 0.5 is interpolated from each curve and appended
    to its legend label.

    Parameters
    ----------
    volumes : torch.Tensor or list of torch.Tensor
        Input volumes. Either a ``(B, Z, Y, X)`` batch tensor or a list of
        ``(Z, Y, X)`` tensors.
    reference : torch.Tensor
        Reference volume, shape ``(Z, Y, X)``.
    voxel_size : float
        Voxel size in Å. Determines the Nyquist frequency and the x-axis scale.
    mask : torch.Tensor, optional
        Binary or soft mask, shape ``(Z, Y, X)``. When provided, a second
        (masked) FSC is computed for each volume by multiplying both the input
        and the reference with the mask before correlation.
    labels : list of str, optional
        One label per volume. Defaults to ``["volume 0", "volume 1", ...]``.
    show : bool, optional
        Whether to display the figure immediately. Set ``False`` to get the
        figure object back (e.g. for saving to disk). Default is True.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when ``show=True`` (figure is displayed and closed).
        The figure object when ``show=False``; caller is responsible for
        closing it.
    """
    # Normalise to a flat list of (Z, Y, X) tensors
    if isinstance(volumes, torch.Tensor):
        volume_list: list[torch.Tensor] = (
            list(volumes) if volumes.ndim == 4 else [volumes]
        )
    else:
        volume_list = list(volumes)

    n_volumes = len(volume_list)
    if labels is None:
        labels = [f"vol {i}" for i in range(n_volumes)]

    def _res_at_half(k_arr: torch.Tensor, fsc_arr: torch.Tensor) -> str:
        return fsc_resolution(k_arr, fsc_arr, MAP_TO_MODEL_FSC_THRESHOLD)

    nyquist = 1.0 / (2.0 * voxel_size)
    palette = _deep_palette(max(n_volumes, 1))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for i, (volume, label) in enumerate(zip(volume_list, labels)):
        color = palette[i]
        volume_f = volume.float()
        ref_f = reference.float()

        k, fsc = fourier_shell_correlation(volume_f, ref_f, pixelsize=voxel_size)
        ax.plot(
            k.cpu(),
            fsc.cpu(),
            color=color,
            ls="-",
            lw=1.8,
            label=f"{label} ({_res_at_half(k, fsc)})",
        )

        if mask is not None:
            m = mask.float()
            k_m, fsc_m = fourier_shell_correlation(
                volume_f * m, ref_f * m, pixelsize=voxel_size
            )
            ax.plot(
                k_m.cpu(),
                fsc_m.cpu(),
                color=color,
                ls="--",
                lw=1.8,
                label=f"{label} masked ({_res_at_half(k_m, fsc_m)})",
            )

    # FSC = 0.5 criterion line
    ax.axhline(0.5, color="#888888", ls=":", lw=1.0, zorder=0)

    # X-axis: evenly spaced frequency ticks from 0 to Nyquist, labelled in Å
    tick_k = torch.linspace(0, nyquist, 6).tolist()
    tick_labels = ["∞"] + [f"{1.0 / k:.2f}" for k in tick_k[1:]]

    ax.set(
        xticks=tick_k,
        xticklabels=tick_labels,
        xlabel="resolution (Å)",
        title="Map-to-model FSCs",
        ylim=[-0.05, 1.05],
        xlim=[0, nyquist],
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.8)
    _despine(ax, offset=6)
    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_halfmap_fsc(
    volumes_A: torch.Tensor | list[torch.Tensor],
    volumes_B: torch.Tensor | list[torch.Tensor],
    voxel_size: float,
    mask: torch.Tensor | None = None,
    labels: list[str] | None = None,
    show: bool = True,
) -> tuple[matplotlib.figure.Figure | None, list[str]]:
    """
    Compute and plot half-map Fourier Shell Correlations.

    Each volume in volumes_A is correlated against the corresponding volume in volumes_B.
    When ``mask`` is provided, both an unmasked and a masked FSC are computed for
    each volume pair. The (unmasked) resolution at the gold-standard FSC = 0.143
    threshold is interpolated from each curve, appended to its legend label, and
    returned alongside the figure.

    Parameters
    ----------
    volumes_A : torch.Tensor or list of torch.Tensor
        First set of input volumes. Either a ``(B, Z, Y, X)`` batch tensor or a list of
        ``(Z, Y, X)`` tensors.
    volumes_B : torch.Tensor or list of torch.Tensor
        Second set of input volumes. Either a ``(B, Z, Y, X)`` batch tensor or a list of
        ``(Z, Y, X)`` tensors. Must have the same number of volumes as ``volumes_A``.
    voxel_size : float
        Voxel size in Å. Determines the Nyquist frequency and the x-axis scale.
    mask : torch.Tensor, optional
        Binary or soft mask, shape ``(Z, Y, X)``. When provided, a second
        (masked) FSC is computed for each volume pair by multiplying both the input
        and the reference with the mask before correlation.
    labels : list of str, optional
        One label per volume pair. Defaults to ``["pair 0", "pair 1", ...]``.
    show : bool, optional
        Whether to display the figure immediately. Set ``False`` to get the
        figure object back (e.g. for saving to disk). Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        ``None`` when ``show=True`` (figure is displayed and closed).
        The figure object when ``show=False``; caller is responsible for
        closing it.
    resolutions : list of str
        Unmasked resolution at the FSC=0.143 crossing for each volume pair,
        e.g. ``"3.201 Å"`` or ``">Nyquist"`` if the curve never drops below
        0.143 within the Nyquist limit. Same order as ``labels``.

    Raises
    ------
    ValueError
        If the number of volumes in volumes_A and volumes_B do not match.
    """
    # Normalise to flat lists of (Z, Y, X) tensors
    if isinstance(volumes_A, torch.Tensor):
        volume_list_A: list[torch.Tensor] = (
            list(volumes_A) if volumes_A.ndim == 4 else [volumes_A]
        )
    else:
        volume_list_A = list(volumes_A)

    if isinstance(volumes_B, torch.Tensor):
        volume_list_B: list[torch.Tensor] = (
            list(volumes_B) if volumes_B.ndim == 4 else [volumes_B]
        )
    else:
        volume_list_B = list(volumes_B)

    # Check that the number of volumes matches
    if len(volume_list_A) != len(volume_list_B):
        raise ValueError(
            f"Number of volumes in volumes_A ({len(volume_list_A)}) must match "
            f"the number in volumes_B ({len(volume_list_B)})"
        )

    n_volumes = len(volume_list_A)
    if labels is None:
        labels = [f"pair {i}" for i in range(n_volumes)]

    nyquist = 1.0 / (2.0 * voxel_size)

    def _res_at_half(k_arr: torch.Tensor, fsc_arr: torch.Tensor) -> str:
        return fsc_resolution(k_arr, fsc_arr, HALFMAP_FSC_THRESHOLD, k_max=nyquist)

    palette = _deep_palette(max(n_volumes, 1))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    resolutions: list[str] = []

    for i, (volume_a, volume_b, label) in enumerate(
        zip(volume_list_A, volume_list_B, labels)
    ):
        color = palette[i]
        volume_a_f = volume_a.float()
        volume_b_f = volume_b.float()

        k, fsc = fourier_shell_correlation(volume_a_f, volume_b_f, pixelsize=voxel_size)
        resolution = _res_at_half(k, fsc)
        resolutions.append(resolution)
        ax.plot(
            k.cpu(),
            fsc.cpu(),
            color=color,
            ls="-",
            lw=1.8,
            label=f"{label} ({resolution})",
        )

        if mask is not None:
            m = mask.float()
            k_m, fsc_m = fourier_shell_correlation(
                volume_a_f * m, volume_b_f * m, pixelsize=voxel_size
            )
            ax.plot(
                k_m.cpu(),
                fsc_m.cpu(),
                color=color,
                ls="--",
                lw=1.8,
                label=f"{label} masked ({_res_at_half(k_m, fsc_m)})",
            )

    # FSC = 0.143 criterion line
    ax.axhline(0.143, color="#888888", ls=":", lw=1.0, zorder=0)

    # X-axis: evenly spaced frequency ticks from 0 to Nyquist, labelled in Å
    tick_k = torch.linspace(0, nyquist, 6).tolist()
    tick_labels = ["∞"] + [f"{1.0 / k:.2f}" for k in tick_k[1:]]

    ax.set(
        xticks=tick_k,
        xticklabels=tick_labels,
        xlabel="resolution (Å)",
        title="Half-map FSCs",
        ylim=[-0.05, 1.05],
        xlim=[0, nyquist],
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.8)
    _despine(ax, offset=6)
    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None, resolutions
    return fig, resolutions


def _extract_epoch_number(filename: str) -> int | None:
    """
    Extract epoch number from FSC or volume PNG filename.

    Parameters
    ----------
    filename : str
        Filename like "fsc_001_B.png" or "volume_042_A.png"

    Returns
    -------
    int or None
        Epoch number if matched, None otherwise.
    """
    match = re.search(r"(?:fsc|volume)_(\d+)(?:_[A-Z])?", filename)
    return int(match.group(1)) if match else None


def _discover_fsc_images(
    job_folder: str | Path, suffix: str | None = None
) -> list[tuple[int, Path]]:
    """
    Find FSC PNG files in job_folder/epochs/, sorted by epoch number.

    Parameters
    ----------
    job_folder : str or Path
        Job folder path.
    suffix : str or None
        If specified, only return files matching this suffix (e.g., "A" for fsc_001_A.png).

    Returns
    -------
    list of (epoch_number, Path) tuples
        Sorted by epoch number ascending.

    Raises
    ------
    FileNotFoundError
        If job_folder/epochs/ does not exist.
    """
    job_folder = Path(job_folder)
    epochs_dir = job_folder / "epochs"

    if not epochs_dir.exists():
        raise FileNotFoundError(f"Epochs directory not found: {epochs_dir}")

    # Find matching FSC files
    pattern = f"fsc_*_{suffix}.png" if suffix else "fsc_*.png"
    fsc_files = []
    for fsc_file in sorted(epochs_dir.glob(pattern)):
        epoch = _extract_epoch_number(fsc_file.name)
        if epoch is not None:
            fsc_files.append((epoch, fsc_file))

    # Sort by epoch number
    fsc_files.sort(key=lambda x: x[0])
    return fsc_files


def _discover_volume_images(job_folder: str | Path) -> list[tuple[int, Path]]:
    """
    Find volume PNG files in job_folder/epochs/, sorted by epoch number.

    Parameters
    ----------
    job_folder : str or Path
        Job folder path.

    Returns
    -------
    list of (epoch_number, Path) tuples
        Sorted by epoch number ascending.

    Raises
    ------
    FileNotFoundError
        If job_folder/epochs/ does not exist.
    """
    job_folder = Path(job_folder)
    epochs_dir = job_folder / "epochs"

    if not epochs_dir.exists():
        raise FileNotFoundError(f"Epochs directory not found: {epochs_dir}")

    # Find volume files
    volume_files = []
    for volume_file in sorted(epochs_dir.glob("volume_*.png")):
        epoch = _extract_epoch_number(volume_file.name)
        if epoch is not None:
            volume_files.append((epoch, volume_file))

    # Sort by epoch number
    volume_files.sort(key=lambda x: x[0])
    return volume_files


try:
    import ipywidgets as widgets
    from IPython.display import display, Markdown

    def visualize_job_epochs(
        job_folder: str | Path,
        suffix: str | None = None,
        image_width: int = 8,
    ) -> None:
        """
        Display FSC and volume images side-by-side with synchronized epoch slider.

        Interactive Jupyter visualization of reconstruction progress. FSC and volume
        PNG images are displayed for each epoch with a synchronized slider control.
        Requires ipywidgets to be installed.

        Parameters
        ----------
        job_folder : str or Path
            Path to the job folder (e.g., "~/specter-data/empiar-10202/J002").
        suffix : str, optional
            Filter FSC images by suffix (e.g., "A" shows only fsc_001A.png).
            If None, all FSC images are shown.
        image_width : int, optional
            Display width of each image in inches. Default is 8.

        Raises
        ------
        FileNotFoundError
            If job_folder/epochs/ does not exist.
        ValueError
            If no FSC or volume images are found.

        Examples
        --------
        >>> visualize_job_epochs("~/specter-data/empiar-10202/J002")
        >>> visualize_job_epochs("~/specter-data/empiar-10202/J002", suffix="A")
        """
        job_folder = Path(job_folder).expanduser()

        # Discover images
        try:
            fsc_files = _discover_fsc_images(job_folder, suffix=suffix)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e

        try:
            volume_files = _discover_volume_images(job_folder)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e

        if not fsc_files:
            suffix_str = f" with suffix '{suffix}'" if suffix else ""
            raise ValueError(
                f"No FSC images found{suffix_str} in {job_folder / 'epochs'}"
            )
        if not volume_files:
            raise ValueError(f"No volume images found in {job_folder / 'epochs'}")

        # Create epoch lists (use max range, show N/A for missing)
        fsc_epochs = {epoch: path for epoch, path in fsc_files}
        volume_epochs = {epoch: path for epoch, path in volume_files}
        all_epochs = sorted(set(fsc_epochs.keys()) | set(volume_epochs.keys()))

        # Create matplotlib figures for display
        fig_fsc, ax_fsc = plt.subplots(
            figsize=(image_width, image_width * 0.75), dpi=100
        )
        fig_volume, ax_volume = plt.subplots(
            figsize=(image_width, image_width * 0.75), dpi=100
        )
        plt.close(fig_fsc)
        plt.close(fig_volume)

        # Create output widgets
        output_fsc = widgets.Output()
        output_volume = widgets.Output()

        def update_display(epoch_idx: int) -> None:
            """Update both images for the selected epoch index."""
            epoch = all_epochs[epoch_idx]

            output_fsc.clear_output(wait=True)
            output_volume.clear_output(wait=True)

            with output_fsc:
                if epoch in fsc_epochs:
                    img_fsc = PILImage.open(fsc_epochs[epoch])
                    ax_fsc.clear()
                    ax_fsc.imshow(img_fsc)
                    ax_fsc.set_title(f"FSC Epoch {epoch}")
                    ax_fsc.axis("off")
                    fig_fsc.tight_layout()
                    display(fig_fsc)
                else:
                    display(Markdown(f"**FSC**: No image for epoch {epoch}"))

            with output_volume:
                if epoch in volume_epochs:
                    img_volume = PILImage.open(volume_epochs[epoch])
                    ax_volume.clear()
                    ax_volume.imshow(img_volume)
                    ax_volume.set_title(f"Volume Epoch {epoch}")
                    ax_volume.axis("off")
                    fig_volume.tight_layout()
                    display(fig_volume)
                else:
                    display(Markdown(f"**Volume**: No image for epoch {epoch}"))

        # Create slider
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(all_epochs) - 1,
            step=1,
            description="Epoch:",
            continuous_update=False,
        )

        # Update on slider change
        def on_slider_change(change: dict) -> None:
            update_display(change["new"])

        slider.observe(on_slider_change, names="value")

        # Create navigation buttons
        btn_prev = widgets.Button(description="◀", layout=widgets.Layout(width="50px"))
        btn_next = widgets.Button(description="▶", layout=widgets.Layout(width="50px"))

        def on_prev_clicked(btn: widgets.Button) -> None:
            if slider.value > 0:
                slider.value -= 1

        def on_next_clicked(btn: widgets.Button) -> None:
            if slider.value < len(all_epochs) - 1:
                slider.value += 1

        btn_prev.on_click(on_prev_clicked)
        btn_next.on_click(on_next_clicked)

        # Layout: title, images side-by-side, slider with buttons
        header = widgets.HTML(f"<h3>Job: {job_folder.name}</h3>")
        images_hbox = widgets.HBox([output_fsc, output_volume])
        slider_hbox = widgets.HBox([btn_prev, slider, btn_next])

        vbox = widgets.VBox([header, images_hbox, slider_hbox])

        # Initial display
        update_display(0)
        display(vbox)

except ImportError:

    def visualize_job_epochs(
        job_folder: str | Path,
        suffix: str | None = None,
        image_width: int = 8,
    ) -> None:
        raise ImportError(
            "visualize_job_epochs() requires ipywidgets. "
            "Install with: pip install ipywidgets"
        )
