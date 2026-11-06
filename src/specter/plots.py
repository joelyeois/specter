"""
Plotting helpers: volumes, slices, particle stacks, radial distributions
and FSC curves with their resolution readouts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import torch

from .arrays import radial_profile_2d, radial_symmetrize
from .coords import radial_distribution_function
from .fft import fourier_shell_correlation, fsc_crossing

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


#: Gold-standard half-map FSC threshold (two independent half-set reconstructions;
#: Rosenthal & Henderson 2003).
HALFMAP_FSC_THRESHOLD = 0.143
#: Map-to-model FSC threshold (a reconstruction against a known reference): the
#: correlation with a perfect reference that a half-map FSC of 0.143 corresponds
#: to (Rosenthal & Henderson 2003).
MAP_TO_MODEL_FSC_THRESHOLD = 0.5


def fsc_resolution(
    k: torch.Tensor,
    fsc: torch.Tensor,
    threshold: float,
    k_max: float | None = None,
) -> str:
    """
    Resolution at the first FSC threshold crossing, linearly interpolated.

    The *first* crossing, as RELION and CryoSPARC report it
    (`specter.fft.fsc_crossing`). A masked FSC rises again toward Nyquist
    when both maps share the mask, because the mask correlates their noise;
    reading the last crossing then reports that artefact as resolution
    (1.79 A against a true ~1.91 A on EMPIAR-13203).

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
        Ignore shells above this frequency, for a curve that runs past
        Nyquist. `specter.fft.fourier_shell_correlation`'s stops there.

    Returns
    -------
    str
        Resolution formatted as ``"1.397 Å"``, or ``">Nyquist"`` when the FSC
        never crosses the threshold.

    References
    ----------
    .. [1] Rosenthal, P. B., & Henderson, R. (2003). Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy. *Journal of Molecular
       Biology*, 333(4), 721–745. https://doi.org/10.1016/j.jmb.2003.07.013
    """
    kf = k.detach().cpu().double()
    ff = fsc.detach().cpu()
    if k_max is not None:
        keep = int((kf <= k_max).sum())
        kf, ff = kf[:keep], ff[:keep]
    if len(ff) < 2 or not bool((ff[1:] < threshold).any()):
        return ">Nyquist"
    shell = fsc_crossing(ff, threshold)
    i = min(int(shell), len(kf) - 2)
    k_cross = kf[i] + (shell - i) * (kf[i + 1] - kf[i])
    return f"{1.0 / float(k_cross):.3f} Å"


@dataclass(frozen=True)
class FSCCurve:
    """
    One FSC between two volumes and its resolution, optionally also masked.

    Attributes
    ----------
    k, fsc : torch.Tensor
        Frequencies (1/Å) and the unmasked FSC, from
        `specter.fft.fourier_shell_correlation`.
    resolution : str
        At the curve's threshold, as `fsc_resolution` formats it.
    fsc_masked : torch.Tensor or None
        The FSC inside the mask, when one was given.
    resolution_masked : str or None
        Its resolution.
    """

    k: torch.Tensor
    fsc: torch.Tensor
    resolution: str
    fsc_masked: torch.Tensor | None = None
    resolution_masked: str | None = None


def fsc_curve(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    voxel_size: float,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> FSCCurve:
    """
    The FSC of two volumes and its resolution, plus the masked pair with a mask.

    Parameters
    ----------
    volume_a, volume_b : torch.Tensor
        Volumes to correlate, shape ``(N, N, N)``; `volume_b` is moved to
        `volume_a`'s device.
    voxel_size : float
        Voxel size in Å.
    threshold : float
        `HALFMAP_FSC_THRESHOLD` or `MAP_TO_MODEL_FSC_THRESHOLD`.
    mask : torch.Tensor, optional
        Gives the masked curve as well.

    Returns
    -------
    FSCCurve
    """
    a = volume_a.detach().float()
    b = volume_b.detach().to(a.device).float()
    k, fsc = fourier_shell_correlation(a, b, pixel_size=voxel_size)
    curve = FSCCurve(k, fsc, fsc_resolution(k, fsc, threshold))
    if mask is None:
        return curve
    _, fsc_m = fourier_shell_correlation(
        a, b, pixel_size=voxel_size, mask=mask.detach().float()
    )
    return replace(
        curve, fsc_masked=fsc_m, resolution_masked=fsc_resolution(k, fsc_m, threshold)
    )


def resolution_between(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    voxel_size: float,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> str:
    """
    Resolution at an FSC threshold between two volumes, optionally masked.

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
    a = volume_a.detach().float()
    b = volume_b.detach().to(a.device).float()
    m = None if mask is None else mask.detach().float()
    k, fsc = fourier_shell_correlation(a, b, pixel_size=voxel_size, mask=m)
    return fsc_resolution(k, fsc, threshold)


def plot_fsc_curves(
    curves: list[FSCCurve],
    labels: list[str],
    voxel_size: float,
    threshold: float,
    title: str,
) -> matplotlib.figure.Figure:
    """
    Draw FSC curves, a dashed masked curve beside each that has one.

    Parameters
    ----------
    curves : list of FSCCurve
        From `fsc_curve`.
    labels : list of str
        One per curve; each legend entry adds the curve's resolution.
    voxel_size : float
        Voxel size in Å, which sets the Nyquist frequency the axis ends at.
    threshold : float
        Drawn as a dotted line.
    title : str
        The axes' title.

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it.
    """
    nyquist = 1.0 / (2.0 * voxel_size)
    palette = _deep_palette(max(len(curves), 1))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for color, curve, label in zip(palette, curves, labels):
        k = curve.k.cpu()
        ax.plot(
            k,
            curve.fsc.cpu(),
            color=color,
            ls="-",
            lw=1.8,
            label=f"{label} ({curve.resolution})",
        )
        if curve.fsc_masked is not None:
            ax.plot(
                k,
                curve.fsc_masked.cpu(),
                color=color,
                ls="--",
                lw=1.8,
                label=f"{label} masked ({curve.resolution_masked})",
            )
    ax.axhline(threshold, color="#888888", ls=":", lw=1.0, zorder=0)

    # X-axis: evenly spaced frequency ticks from 0 to Nyquist, labelled in Å
    tick_k = torch.linspace(0, nyquist, 6).tolist()
    tick_labels = ["∞"] + [f"{1.0 / k:.2f}" for k in tick_k[1:]]
    ax.set(
        xticks=tick_k,
        xticklabels=tick_labels,
        xlabel="resolution (Å)",
        title=title,
        ylim=[-0.05, 1.05],
        xlim=[0, nyquist],
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.8)
    _despine(ax, offset=6)
    fig.tight_layout()
    return fig


def _volume_list(volumes: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """A ``(B, Z, Y, X)`` batch or a list of ``(Z, Y, X)`` volumes, as a list."""
    if isinstance(volumes, torch.Tensor):
        return list(volumes) if volumes.ndim == 4 else [volumes]
    return list(volumes)


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
        One label per volume. Defaults to ``["vol 0", "vol 1", ...]``.
    show : bool, optional
        Whether to display the figure immediately. Set ``False`` to get the
        figure object back (e.g. for saving to disk). Default is True.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when ``show=True`` (figure is displayed and closed).
        The figure object when ``show=False``; caller is responsible for
        closing it.

    References
    ----------
    .. [1] Rosenthal, P. B., & Henderson, R. (2003). Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy. *Journal of Molecular
       Biology*, 333(4), 721–745. https://doi.org/10.1016/j.jmb.2003.07.013
    """
    volume_list = _volume_list(volumes)
    if labels is None:
        labels = [f"vol {i}" for i in range(len(volume_list))]
    curves = [
        fsc_curve(v, reference, voxel_size, MAP_TO_MODEL_FSC_THRESHOLD, mask)
        for v in volume_list
    ]
    fig = plot_fsc_curves(
        curves, labels, voxel_size, MAP_TO_MODEL_FSC_THRESHOLD, "Map-to-model FSCs"
    )
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
        0.143 within the Nyquist limit. Same order as ``labels``. For the
        masked resolutions as well, use `fsc_curve`.

    Raises
    ------
    ValueError
        If the number of volumes in volumes_A and volumes_B do not match.

    References
    ----------
    .. [1] Rosenthal, P. B., & Henderson, R. (2003). Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy. *Journal of Molecular
       Biology*, 333(4), 721–745. https://doi.org/10.1016/j.jmb.2003.07.013
    """
    volume_list_A = _volume_list(volumes_A)
    volume_list_B = _volume_list(volumes_B)
    if len(volume_list_A) != len(volume_list_B):
        raise ValueError(
            f"Number of volumes in volumes_A ({len(volume_list_A)}) must match "
            f"the number in volumes_B ({len(volume_list_B)})"
        )
    if labels is None:
        labels = [f"pair {i}" for i in range(len(volume_list_A))]
    curves = [
        fsc_curve(a, b, voxel_size, HALFMAP_FSC_THRESHOLD, mask)
        for a, b in zip(volume_list_A, volume_list_B)
    ]
    fig = plot_fsc_curves(
        curves, labels, voxel_size, HALFMAP_FSC_THRESHOLD, "Half-map FSCs"
    )
    resolutions = [c.resolution for c in curves]
    if show:
        plt.show()
        plt.close(fig)
        return None, resolutions
    return fig, resolutions
