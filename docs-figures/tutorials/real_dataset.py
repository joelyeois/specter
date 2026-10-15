"""
Figures and printed output for docs/tutorials/real-dataset.md.

The data are 64 particles of EMPIAR-11377 (a translating 70S ribosome, PDB
8B0X), under docs-figures/data/:

- empiar-11377-64particles-128px.mrcs: the first 64 images of the 512 px,
  0.731 A CryoSPARC restack (job J214), Fourier-cropped to 128 px at 2.924 A
  and stored as float16. Kept in the restack's polarity, which its
  ``blob/sign`` records as -1.
- empiar-11377-64particles-pdbframe.cs: the same 64 rows of the J214
  passthrough with the poses of job J205, which is downstream of an Align 3D
  against the deposited map and so in the PDB model's frame.
- empiar-11377-64particles-unaligned.cs: the same rows with J214's own
  poses, from a refinement never aligned to the model (a constant 142 degree
  rotation from the PDB frame).

Both .cs files were rescaled to the 128 px images with
`specter.match.rescale_metadata`.

Run with: uv run python docs-figures/tutorials/real_dataset.py
"""

from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import (
    PALETTE,
    REPO_ROOT,
    Transcript,
    as_numpy,
    radial_power,
    save,
    show_image,
)

DATA = REPO_ROOT / "docs-figures" / "data"
transcript = Transcript("real_dataset")


def lowpass(x: torch.Tensor, pixel_size: float, resolution: float) -> torch.Tensor:
    """A Gaussian low-pass to `resolution` Å, for display only."""
    x = x.float()
    k = torch.fft.fftfreq(x.shape[-1], d=pixel_size, device=x.device)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(-k2 * resolution**2)).real


def zscore(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True)


# The quoted code names the files as a reader would have them, in the
# working directory.
os.chdir(DATA)

transcript.start("parameters")
# --8<-- [start:parameters]
from specter.io import extract_parameters_from_csfile

(
    voltage,
    pixel_size,
    alpha,
    quaternions,
    shifts,
    ctf_params,
    scale,
    anisomag,
    ids,
    halfsets,
) = extract_parameters_from_csfile("empiar-11377-64particles-pdbframe.cs")

print(f"{len(ids)} particles at {float(voltage):g} kV, {float(pixel_size):.3f} Å/px")
print(f"amplitude contrast {float(alpha):g}")
print(f"defocus {ctf_params['dfu'].min():.0f} to {ctf_params['dfu'].max():.0f} Å")
print(f"Cs {ctf_params['cs'][0] / 1e7:g} mm, terms: {sorted(ctf_params)}")
print(f"anisotropic magnification of particle 0:\n{anisomag[0]}")
# --8<-- [end:parameters]

transcript.start("images")
# --8<-- [start:images]
from specter.match import load_experimental_images

# Read in the polarity the detector recorded: the restack stores the opposite,
# which CryoSPARC writes as blob/sign = -1, and the loader multiplies by it.
real = load_experimental_images(
    "empiar-11377-64particles-pdbframe.cs",
    images_path="empiar-11377-64particles-128px.mrcs",
)
print(f"{tuple(real.shape)} images, {real.dtype}")
# --8<-- [end:images]

transcript.start("simulate")
# --8<-- [start:simulate]
from specter.imaging import CTF, Exposure, Poses, Structure, simulate_particles

structure = Structure.from_pdb("8b0x")
template = structure.render(voxel_size=float(pixel_size), box=128, device="cuda")
specimen = template.crowd(min_distance=structure.pdb.max_diameter).solvate(
    model="gd", thickness=0.0
)

poses = Poses(quaternions, shifts, ids)
ctf = CTF(ctf_params, ids)
simulated = simulate_particles(
    specimen,
    poses,
    ctf,
    Exposure(dose=40.0),
    voltage=float(voltage),
    seed=0,
    batch_size=16,
    scatter={"alpha": float(alpha)},
    camera={
        "detector": "falcon4i_300kv",
        "detector_pixel_size": 0.731,  # the camera's own pixel, before binning
        "anisomag": anisomag,
    },
)
print(
    f"{tuple(simulated.array.shape)} simulated, mean {simulated.array.mean():.1f} counts/px"
)
# --8<-- [end:simulate]


# ---------------------------------------------------------------------------
# Figure: real and simulated particles side by side.
show = [0, 1, 2, 3, 4, 5]
fig, axes = plt.subplots(2, len(show), figsize=(1.8 * len(show), 3.9))
for col, i in enumerate(show):
    for row, (label, stack) in enumerate(
        [("EMPIAR-11377", real), ("SPECTER", simulated.array.cpu())]
    ):
        first = row == 1 and col == 0
        show_image(
            axes[row, col],
            as_numpy(zscore(lowpass(stack[i], float(pixel_size), 12.0))),
            vmin=-2.5,
            vmax=2.5,
            pixel_size=float(pixel_size) if first else None,
            scalebar=100 if first else None,
        )
        if col == 0:
            axes[row, col].set_ylabel(label)
    axes[0, col].set_title(f"{ctf_params['dfu'][i] / 1e4:.2f} µm")
fig.tight_layout()
save(fig, "real_dataset-particles")

# Thumbnail: two particles, real over simulated.
fig, axes = plt.subplots(2, 2, figsize=(3, 3))
for col, i in enumerate([1, 3]):
    for row, stack in enumerate([real, simulated.array.cpu()]):
        img = as_numpy(zscore(lowpass(stack[i], float(pixel_size), 12.0)))
        show_image(axes[row, col], img, vmin=-2.5, vmax=2.5)
        for spine in axes[row, col].spines.values():
            spine.set_visible(False)
fig.subplots_adjust(0, 0, 1, 1, wspace=0.03, hspace=0.03)
save(fig, "real_dataset-thumbnail")

transcript.start("pose-check")
# --8<-- [start:pose-check]
from specter.match import matched_index_correlation

checks = {}
for frame in ("pdbframe", "unaligned"):
    _, _, _, q, t, terms, _, aniso, idx, _ = extract_parameters_from_csfile(
        f"empiar-11377-64particles-{frame}.cs"
    )
    # Noiseless, without ice or neighbours: only the view is being tested.
    views = simulate_particles(
        template,
        Poses(q, t, idx),
        CTF(terms, idx),
        Exposure(dose=40.0),
        voltage=float(voltage),
        batch_size=16,
        scatter={"alpha": float(alpha)},
        camera={"noise": None, "anisomag": aniso},
    )
    checks[frame] = views
    result = matched_index_correlation(views.array.cpu(), real, float(pixel_size))
    print(
        f"{frame:10s} matched {result.matched:.3f}  shuffled {result.shuffled:.3f}  "
        f"{result.fraction_above:4.0%} of pairs above  z = {result.z_score:5.1f}  "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
# --8<-- [end:pose-check]

# ---------------------------------------------------------------------------
# Figure: per-particle matched correlation against many shuffled pairings,
# the statistic the check summarises.
from specter.match._metrics import _bandpass, _xcorr_max

e = _bandpass(real, float(pixel_size), 60.0, 15.0)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
bins = np.linspace(-0.05, 0.6, 40)
for ax, (frame, title) in zip(
    axes,
    [("pdbframe", "Poses in the model's frame"), ("unaligned", "Poses never aligned")],
):
    s = _bandpass(checks[frame].array.cpu(), float(pixel_size), 60.0, 15.0)
    matched = as_numpy(_xcorr_max(s, e, 12))
    if frame == "pdbframe":
        best = [int(i) for i in np.argsort(matched)[::-1][:3]]
    shuffled = np.concatenate(
        [
            as_numpy(
                _xcorr_max(
                    s,
                    e[
                        torch.randperm(
                            len(e), generator=torch.Generator().manual_seed(k)
                        )
                    ],
                    12,
                )
            )
            for k in range(50)
        ]
    )
    ax.hist(shuffled, bins=bins, density=True, color="0.7", label="shuffled pairs")
    ax.hist(
        matched,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.6,
        color=PALETTE[0],
        label="matched pairs",
    )
    ax.set(title=title, xlabel="Peak correlation, 60-15 Å band")
axes[0].set_ylabel("Density")
axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, "real_dataset-pose-check")

# Figure: what a pose in the wrong frame looks like, for the three particles
# whose model-frame view correlates best with the data.
picks = best
fig, axes = plt.subplots(len(picks), 3, figsize=(5.4, 1.85 * len(picks)))
for row, i in enumerate(picks):
    panels = [
        ("EMPIAR-11377", real[i]),
        ("Model's frame", checks["pdbframe"].array[i].cpu()),
        ("Never aligned", checks["unaligned"].array[i].cpu()),
    ]
    for col, (label, img) in enumerate(panels):
        data = as_numpy(zscore(lowpass(img, float(pixel_size), 25.0)))
        show_image(
            axes[row, col], data, vmin=-2.5, vmax=2.5, title=label if row == 0 else None
        )
fig.tight_layout()
save(fig, "real_dataset-views")

# Figure: rotationally averaged power of the two stacks, each image z-scored.
power = {}
for label, stack in [("EMPIAR-11377", real), ("SPECTER", simulated.array.cpu())]:
    spectra = [radial_power(img, float(pixel_size)) for img in zscore(stack)]
    k = spectra[0][0]
    power[label] = np.mean([p for _, p in spectra], axis=0)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for color, (label, p) in zip([PALETTE[0], PALETTE[3]], power.items()):
    ax.semilogy(k[1:], p[1:], color=color, lw=1.2, label=label)
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power (z-scored images)",
    title="Mean rotationally averaged power, 64 particles",
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "real_dataset-spectra")
ratio = power["SPECTER"] / power["EMPIAR-11377"]
band_ratios = []
for lo, hi in [(0.0, 1 / 50), (1 / 50, 1 / 20), (1 / 20, 1 / 10), (1 / 10, 1 / 5.85)]:
    band = (k >= lo) & (k < hi)
    label = f"{1 / hi:.0f}-{1 / lo:.0f} Å" if lo > 0 else f"> {1 / hi:.0f} Å"
    band_ratios.append(f"{label:>10s}: simulated / real power {ratio[band].mean():.2f}")

# The stack is written to a scratch working directory, not the repository.
os.chdir(tempfile.mkdtemp())

transcript.start("star")
# --8<-- [start:star]
from specter.io import create_particle_starfile, extract_parameters_from_starfile

create_particle_starfile(
    simulated.array.cpu(),
    rotations=poses.quaternions,
    translations=poses.shifts,
    ctf_params=ctf.params,
    alpha=float(alpha),
    voltage=float(voltage),
    dx=float(pixel_size),
    dose_per_angstrom=40.0,
    output_dir="simulated",
    filename="particles",
)
_, _, _, q2, t2, terms2, *_ = extract_parameters_from_starfile(
    "simulated/particles.star"
)
import roma

R1, R2 = roma.unitquat_to_rotmat(quaternions), roma.unitquat_to_rotmat(q2)
print(f"pose round trip: max rotation-matrix difference {(R1 - R2).abs().max():.1e}")
print(f"shift round trip: max difference {(t2 - shifts).abs().max():.1e} Å")
print(
    f"defocus round trip: max difference {(terms2['dfu'] - ctf_params['dfu']).abs().max():.1e} Å"
)
# --8<-- [end:star]

transcript.close()
print("\n".join(band_ratios))
