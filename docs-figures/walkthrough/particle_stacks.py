"""
Figures and printed output for docs/walkthrough/particle-stacks.md.

Run with: uv run python docs-figures/walkthrough/particle_stacks.py
"""

from __future__ import annotations

import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import Transcript, as_numpy, save, show_image

transcript = Transcript("particle-stacks")


def lowpass(image: torch.Tensor, pixel_size: float, resolution: float) -> np.ndarray:
    """`image` with a Gaussian low-pass at `resolution` Angstrom, for display."""
    data = as_numpy(image)
    k = np.fft.fftfreq(data.shape[-1], pixel_size)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    sigma = 1.0 / resolution
    return np.fft.ifft2(np.fft.fft2(data) * np.exp(-k2 / (2 * sigma**2))).real


def gallery(
    images: torch.Tensor,
    titles: list[str],
    name: str,
    *,
    resolution: float | None = None,
    cols: int = 8,
) -> None:
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.15 * cols, 1.35 * rows), squeeze=False
    )
    for i, ax in enumerate(axes.ravel()):
        if i >= len(images):
            ax.axis("off")
            continue
        img = images[i] if resolution is None else lowpass(images[i], 1.0, resolution)
        img = as_numpy(img)
        lo, hi = np.percentile(img, [1, 99])
        show_image(ax, img, vmin=lo, vmax=hi)
        ax.set_title(titles[i], fontsize=7)
    fig.tight_layout(pad=0.3)
    save(fig, name)


transcript.start("setup")
# --8<-- [start:setup]
import torch

import specter
from specter.imaging import CTF, Exposure, Poses, Structure, simulate_particles

structure = Structure.from_pdb("6bdf")
specimen = structure.render(voxel_size=1.0, box=256, device="cuda")
specimen = specimen.crowd(min_distance=192.0)  # Å, the largest diameter of 6bdf
specimen = specimen.solvate(model="gd", thickness=500.0)

for step in specimen.plan(voltage=300.0):
    print(step)
# --8<-- [end:setup]

transcript.start("stack")
# --8<-- [start:stack]
import time

n = 16
specter.seed(0)  # poses and CTF terms are drawn from the global generator
poses = Poses.random(n, shift=2.0)
ctf = CTF.sample(
    n,
    defocus=[8000.0, 20000.0],
    astigmatism=[0.0, 300.0],
    astigmatism_angle=[0.0, 180.0],
)
exposure = Exposure(dose=40.0, n_frames=40)

start = time.perf_counter()
images = simulate_particles(
    specimen,
    poses,
    ctf,
    exposure,
    voltage=300.0,
    seed=0,
    batch_size=4,
    coherence={"cc": 2.7},
    camera={"detector": "falcon4i_300kv", "coincidence_radius": 0.22},
)
torch.cuda.synchronize()
print(f"{tuple(images.array.shape)} in {time.perf_counter() - start:.1f} s")
print(f"mean {images.array.mean():.1f} counts/px")
# --8<-- [end:stack]

defocus_um = [f"{d / 1e4:.2f} µm" for d in ctf.params["dfu"].tolist()]
gallery(images.array, defocus_um, "particle-stacks-gallery", resolution=8.0)

transcript.start("noiseless")
# --8<-- [start:noiseless]
clean = simulate_particles(
    specimen,
    poses,
    ctf,
    exposure,
    voltage=300.0,
    seed=0,
    batch_size=4,
    coherence={"cc": 2.7},
    camera={"detector": "falcon4i_300kv", "noise": None},
)
# --8<-- [end:noiseless]

fig, axes = plt.subplots(2, 6, figsize=(8.4, 3.0))
for col in range(6):
    raw = lowpass(images.array[col], 1.0, 8.0)
    lo, hi = np.percentile(raw, [1, 99])
    show_image(axes[0, col], raw, vmin=lo, vmax=hi, title=defocus_um[col])
    smooth = lowpass(clean.array[col], 1.0, 8.0)
    lo, hi = np.percentile(smooth, [1, 99])
    show_image(axes[1, col], smooth, vmin=lo, vmax=hi)
axes[0, 0].set_ylabel("Detected")
axes[1, 0].set_ylabel("noise=None")
axes[1, 0].yaxis.label.set_visible(True)
fig.tight_layout(pad=0.4)
save(fig, "particle-stacks-noiseless")

transcript.start("reproducible")


# --8<-- [start:reproducible]
def run(poses, ctf, *, seed, batch_size):
    return simulate_particles(
        specimen,
        poses,
        ctf,
        exposure,
        voltage=300.0,
        seed=seed,
        batch_size=batch_size,
        coherence={"cc": 2.7},
        camera={"detector": "falcon4i_300kv", "noise": None},
    ).array


one_at_a_time = run(poses, ctf, seed=0, batch_size=1)
a_slice = run(poses[8:12], ctf[8:12], seed=0, batch_size=4)
another_seed = run(poses, ctf, seed=1, batch_size=4)

for name, other, reference in [
    ("batch_size=1 vs 4", one_at_a_time, clean.array),
    ("particles 8-11 alone", a_slice, clean.array[8:12]),
    ("seed=1 vs seed=0", another_seed, clean.array),
]:
    differ = (other != reference).float().mean()
    scale = (other - reference).abs().max() / reference.std()
    print(
        f"{name:22s} pixels that differ {differ:6.1%}   "
        f"largest difference {scale:.1e} of the image std"
    )
# --8<-- [end:reproducible]

transcript.start("normalize")
# --8<-- [start:normalize]
from specter.image import normalize_particles

particles, means, stds = normalize_particles(images.array.cpu())
particles = -particles  # protein bright on a dark background

print(f"before: mean {means[:4].numpy().round(1)}, std {stds[:4].numpy().round(2)}")
print(
    f"after:  mean {particles.mean((1, 2))[:4].numpy().round(3)}, "
    f"std {particles.std((1, 2))[:4].numpy().round(3)}"
)
# --8<-- [end:normalize]

fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5))
raw = as_numpy(images.array[2])
lo, hi = np.percentile(raw, [1, 99])
show_image(axes[0], raw, vmin=lo, vmax=hi, title="Counts", colorbar="counts")
show_image(
    axes[1],
    particles[2],
    vmin=-2.5,
    vmax=2.5,
    title="Normalised, inverted",
    colorbar="standard deviations",
)
smooth = lowpass(particles[2], 1.0, 8.0)
lo, hi = np.percentile(smooth, [1, 99])
show_image(
    axes[2],
    smooth,
    vmin=lo,
    vmax=hi,
    title="Same, 8 Å low-pass",
    pixel_size=1.0,
    scalebar=50,
)
fig.tight_layout()
save(fig, "particle-stacks-normalize")

os.chdir(tempfile.mkdtemp())
transcript.start("save")
# --8<-- [start:save]
import mrcfile

images.save("particles.mrcs")

with mrcfile.open("particles.mrcs") as mrc:
    print(mrc.data.shape, f"voxel size {mrc.voxel_size.x:.2f} Å")
# --8<-- [end:save]

transcript.close()
