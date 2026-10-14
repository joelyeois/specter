"""
Thumbnails and quoted code for docs/basic-examples.md.

Each quoted section is a complete, copyable template; the thumbnail beside it
is what that template produced.

Run with: uv run python docs-figures/walkthrough/basic_examples.py
"""

from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import as_numpy, save, show_image


def lowpass(x: torch.Tensor, pixel_size: float, resolution: float) -> torch.Tensor:
    """A Gaussian low-pass to `resolution` Å, for display only."""
    x = x.float()
    k = torch.fft.fftfreq(x.shape[-1], d=pixel_size, device=x.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(-k2 * resolution**2)).real


def unpadded(image) -> torch.Tensor:
    """The first image of `image` on its pixel grid, the FFT margin cropped."""
    pixels = image.pixels()[0]
    m = (image.pad_nxy - image.nxy) // 2
    return pixels[m : m + image.nxy, m : m + image.nxy]


def thumbnail(images: list[np.ndarray], name: str, cols: int = 3) -> None:
    """A square grid of `images`, no gaps, for a gallery card."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3, 3 * rows / cols))
    for ax, image in zip(np.ravel(axes), images):
        lo, hi = np.percentile(image, [1, 99])
        show_image(ax, image, vmin=lo, vmax=hi)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02, hspace=0.02)
    save(fig, name)


# The templates write their outputs to the working directory; keep them out
# of the repository.
os.chdir(tempfile.mkdtemp(prefix="specter-basic-examples-"))

# --8<-- [start:particles]
from specter.imaging import CTF, Exposure, Poses, Structure, simulate_particles

n = 9
specimen = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256, device="cuda")
specimen = specimen.crowd(min_distance=190.0).solvate(model="gd", thickness=300.0)

images = simulate_particles(
    specimen,
    Poses.random(n, shift=2.0),
    CTF.sample(n, defocus=[10000.0, 20000.0]),
    Exposure(dose=40.0),
    voltage=300.0,
    seed=0,
    camera={"detector": "k3_300kv"},
)
images.save("particles.mrcs")
# --8<-- [end:particles]

thumbnail(
    [as_numpy(lowpass(i, 1.0, 10.0)) for i in images.array],
    "basic-particles",
)

# --8<-- [start:micrograph]
from specter.imaging import CTF, Exposure, Field, Structure

structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=1.0, box=256, device="cuda")

field = Field(
    template.potential.cpu(),
    voxel_size=1.0,
    nxy=1024,
    molecular_mass=structure.molecular_mass,
)
field = field.pack(min_distance=192.0).solvate(model="gd", thickness=500.0)
sample = field.draw(voltage=300.0, seed=0, device="cuda")

exposure = Exposure(dose=40.0, n_frames=40)
micrograph = (
    sample.scatter(voltage=300.0, device="cuda")
    .apply_ctf(CTF.from_values(defocus=15000.0))
    .intensity(cc=2.7)
    .detect(exposure, detector="k3_300kv", seed=0)
)
micrograph.save("micrograph.mrc")
# --8<-- [end:micrograph]

thumbnail(
    [as_numpy(lowpass(micrograph.array[0], 1.0, 12.0))], "basic-micrograph", cols=1
)
del sample, field, micrograph

# --8<-- [start:tiltseries]
from specter.imaging import CTF, Exposure, Field, Image, Poses, Sample, Structure
from specter.settings import Ice

structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=4.0, box=64, device="cuda")

# A 600 Å slab of particles, wide enough to stay covered at ±60°.
field = Field(
    template.potential.cpu(),
    voxel_size=4.0,
    nxy=1056,
    nz=150,
    molecular_mass=structure.molecular_mass,
)
volume = field.pack(min_distance=192.0).draw(voltage=300.0, seed=0).volume
sample = Sample.from_volume(volume, voxel_size=4.0)

angles = [0.0]
for step in range(5, 61, 5):  # dose-symmetric: 0, +5, -5, +10, -10, ...
    angles += [float(step), float(-step)]
poses = Poses.tilt_series(angles)
exposure = Exposure.tilt_series(3.0, len(angles), n_frames=10)

series = sample.tilt_series(
    poses,
    exposure,
    voltage=300.0,
    micrograph_size=384,
    ice=Ice(model="gd"),
    absorption_model="inelastic_mfp",
    seed=0,
)
ctf = CTF.from_values(defocus=30000.0, n=len(series))
noiseless, tilts = [], []
for i in range(len(series)):
    wave = series.scatter(i, device="cuda")
    image = wave.apply_ctf(ctf[i : i + 1]).intensity(cc=2.7)
    noiseless.append(image)
    tilts.append(image.detect(exposure[i : i + 1], seed=0))
Image.cat(tilts).save("tiltseries.mrcs")
# --8<-- [end:tiltseries]

order = sorted(range(len(angles)), key=lambda i: angles[i])
picks = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
thumbnail(
    [as_numpy(lowpass(unpadded(noiseless[i]), 4.0, 12.0)) for i in picks],
    "basic-tiltseries",
    cols=2,
)
del sample, series, tilts, noiseless


def class_thumbnail() -> None:
    """The EMPIAR-11377 mixed 2D classification, compact, for its card."""
    import pandas as pd

    from _common import PALETTE, REPO_ROOT

    df = pd.read_csv(
        REPO_ROOT / "docs-figures" / "data" / "empiar-11377-mixed-2dclasses.csv"
    )
    counts = df.groupby(["class_2d", "source"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index][:20]
    fig, ax = plt.subplots(figsize=(3, 3))
    x = np.arange(len(counts))
    ax.bar(x, counts["EMPIAR-11377"], color=PALETTE[0], label="EMPIAR-11377")
    ax.bar(
        x,
        counts["SPECTER"],
        bottom=counts["EMPIAR-11377"],
        color=PALETTE[3],
        label="SPECTER",
    )
    ax.set_xticks([])
    ax.set_xlabel("2D class")
    ax.set_ylabel("Particles")
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "basic-match")


class_thumbnail()
