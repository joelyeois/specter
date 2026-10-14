"""
Shared plumbing for the walkthrough scripts in docs-figures/walkthrough/.

Each walkthrough page in docs/walkthrough/ has one script here. The code a
reader sees on the page is pulled out of that script by pymdownx.snippets
(``--8<-- "docs-figures/walkthrough/<page>.py:<section>"``), so what is shown
is what made the figures. Plotting lives here rather than in the scripts, so
the sections a page quotes stay about the physics.

Run every page with ``uv run python docs-figures/walkthrough/<page>.py``;
figures land in docs/assets/images/walkthrough/, and the text a section
prints (an array shape, a timing, a table) in docs-includes/walkthrough/, from
where the page includes it under the code that printed it. The quoted
code says ``device="cuda"``, as a reader's would; pick the GPU with
``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from specter.plots import _DEEP_PALETTE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images" / "walkthrough"
TEXT_DIR = REPO_ROOT / "docs-includes" / "walkthrough"

PALETTE = list(_DEEP_PALETTE)
"""Seaborn's "deep" palette, the one every docs figure uses."""

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.grid": False,
    }
)


def save(fig: plt.Figure, name: str) -> Path:
    """
    Write `fig` to docs/assets/images/walkthrough/<name>.png and close it.

    The PNG is stored as a 256-colour palette image: a figure of noisy
    greyscale panels is four times smaller that way and indistinguishable on
    screen, which matters for a docs tree of several dozen of them.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    quantize_png(path)
    print(f"wrote {path.relative_to(REPO_ROOT)}", file=sys.__stdout__)
    return path


def quantize_png(path: Path) -> None:
    """Rewrite the PNG at `path` as an optimised 256-colour palette image."""
    image = Image.open(path).convert("RGB")
    image.quantize(
        colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
    ).save(path, optimize=True)


class Transcript:
    """
    Keep what each quoted section prints, for the page to show under it.

    ``transcript.start("load")`` before a section sends everything printed
    until the next ``start`` (or ``close``) to the terminal and to
    docs-includes/walkthrough/<page>-load.txt. The page includes that file in
    a ``text`` block under the code, so the output shown is always the output
    of the run that made the figures, and the quoted code prints with a
    plain ``print``.
    """

    def __init__(self, page: str) -> None:
        self.page = page
        self.name: str | None = None
        self.buffer = io.StringIO()
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text: str) -> int:
        self.stdout.write(text)
        if self.name is not None:
            self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.stdout.flush()

    def start(self, name: str) -> None:
        """Begin recording the section `name`, closing the previous one."""
        self._flush_section()
        self.name = name

    def close(self) -> None:
        """Write the last section and restore the terminal."""
        self._flush_section()
        sys.stdout = self.stdout

    def _flush_section(self) -> None:
        text = self.buffer.getvalue()
        if self.name is not None and text.strip():
            TEXT_DIR.mkdir(parents=True, exist_ok=True)
            (TEXT_DIR / f"{self.page}-{self.name}.txt").write_text(
                text, encoding="utf-8"
            )
        self.name = None
        self.buffer = io.StringIO()


def as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """A detached host copy of `x`."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def show_image(
    ax: plt.Axes,
    image: torch.Tensor | np.ndarray,
    *,
    title: str | None = None,
    pixel_size: float | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: str | None = None,
    scalebar: float | None = None,
) -> matplotlib.image.AxesImage:
    """
    Draw a 2D array with no ticks.

    Parameters
    ----------
    pixel_size : float, optional
        In Angstrom; needed for `scalebar`.
    colorbar : str, optional
        Label for a colour bar beside the panel.
    scalebar : float, optional
        Length of a scale bar in Angstrom, drawn bottom left.
    """
    data = as_numpy(image)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(colorbar)
    if scalebar and pixel_size:
        # Black on the white background of an inverted map, white otherwise.
        color = "black" if cmap.endswith("_r") else "white"
        ny, nx = data.shape[-2:]
        length = scalebar / pixel_size
        x0, y0 = 0.06 * nx, 0.92 * ny
        ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2, solid_capstyle="butt")
        label = f"{scalebar / 10:g} nm" if scalebar >= 10 else f"{scalebar:g} Å"
        ax.text(
            x0 + length / 2,
            y0 - 0.025 * ny,
            label,
            color=color,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return im


def image_row(
    images: Sequence[torch.Tensor | np.ndarray],
    titles: Sequence[str] | None = None,
    *,
    panel: float = 2.2,
    shared_scale: bool = False,
    **kwargs: object,
) -> tuple[plt.Figure, np.ndarray]:
    """One row of `show_image` panels, optionally on a common grey scale."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(panel * n, panel), squeeze=False)
    axes = axes[0]
    if shared_scale:
        stack = np.stack([as_numpy(i) for i in images])
        kwargs.setdefault("vmin", float(np.percentile(stack, 0.5)))
        kwargs.setdefault("vmax", float(np.percentile(stack, 99.5)))
    for i, (ax, image) in enumerate(zip(axes, images)):
        show_image(ax, image, title=None if titles is None else titles[i], **kwargs)
    fig.tight_layout()
    return fig, axes


def power_spectrum(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Centred log power spectrum of a 2D image, mean removed."""
    data = as_numpy(image)
    data = data - data.mean()
    ps = np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2
    return np.log1p(ps / ps.mean())


def radial_average(
    image: torch.Tensor | np.ndarray, pixel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotational average of a centred 2D array.

    Returns
    -------
    k : np.ndarray
        Bin centres in 1/Angstrom.
    profile : np.ndarray
        Mean of `image` in each one-pixel-wide ring.
    """
    data = as_numpy(image)
    ny, nx = data.shape
    y, x = np.indices(data.shape)
    r = np.hypot(y - ny // 2, x - nx // 2).astype(int)
    total = np.bincount(r.ravel(), data.ravel())
    count = np.bincount(r.ravel())
    profile = total / np.maximum(count, 1)
    k = np.arange(len(profile)) / (nx * pixel_size)
    keep = k <= 0.5 / pixel_size
    return k[keep], profile[keep]


def radial_power(
    image: torch.Tensor | np.ndarray, pixel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotationally averaged power spectrum of a 2D image, mean removed."""
    data = as_numpy(image)
    data = data - data.mean()
    ps = np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2 / data.size
    return radial_average(ps, pixel_size)
