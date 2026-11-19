"""
Figures and printed output for docs/tutorials/tilt-series-engines.md.

Run with: uv run python docs-figures/tutorials/tilt_series_engines.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, save, show_image, thumbnail

transcript = Transcript("tilt_series_engines")


def lowpass(image: torch.Tensor, resolution: float, pixel_size: float) -> np.ndarray:
    """`image` with a Gaussian low-pass to `resolution` Å, for display only."""
    data = as_numpy(image).astype(np.float64)
    ky = np.fft.fftfreq(data.shape[-2], pixel_size)[:, None]
    kx = np.fft.fftfreq(data.shape[-1], pixel_size)[None, :]
    envelope = np.exp(-((kx**2 + ky**2) * resolution**2))
    return np.fft.ifft2(np.fft.fft2(data) * envelope).real


transcript.start("specimen")
# --8<-- [start:specimen]
import torch

import specter
from specter.imaging import CTF, Exposure, Field, Poses, Sample, Structure
from specter.settings import Ice

px, n = 4.0, 256  # voxel size in Å, and the size of each tilt image

structure = Structure.from_pdb("6bdf")
template = structure.render(voxel_size=px, box=64, device="cuda")
field = Field(
    template.potential.cpu(),
    voxel_size=px,
    nxy=704,
    nz=64,
    molecular_mass=structure.molecular_mass,
).pack(min_distance=structure.pdb.max_diameter)
dry = field.draw(voltage=300.0, seed=0).volume  # no ice: (1, Z, Y, X)

angles = [0.0]
for step in range(6, 61, 6):  # dose-symmetric: 0, +6, -6, +12, -12, ...
    angles += [float(step), float(-step)]
poses = Poses.tilt_series(angles)
exposure = Exposure.tilt_series(3.0, len(angles))
ctf = CTF.from_values(defocus=30000.0, n=len(angles))
print(f"specimen {tuple(dry.shape)}, {len(angles)} tilts to ±60°")
# --8<-- [end:specimen]

transcript.start("objects")
# --8<-- [start:objects]
specter.seed(0)  # the ice is drawn from the global stream
series = Sample.from_volume(dry, voxel_size=px).tilt_series(
    poses, exposure, voltage=300.0, micrograph_size=n, ice=Ice(model="gd")
)
reference = []
for i in range(len(series)):
    wave = series.scatter(i, device="cuda")
    image = wave.apply_ctf(ctf[i : i + 1]).intensity(cc=2.7)
    reference.append(image.detect(exposure[i : i + 1], noise=None).array[0])
reference = torch.stack(reference)
print(f"objects: {tuple(reference.shape)} expected counts")
# --8<-- [end:objects]

transcript.start("ice")
# --8<-- [start:ice]
from specter.ice import blend_ice_into_volume, resolve_icemaker

specter.seed(0)
nz, ny, nx = dry.shape[-3:]
icemaker = resolve_icemaker("gd", px, nxy=(ny, nx), nz=nz)
with torch.no_grad():
    volume = blend_ice_into_volume(dry, icemaker, px)
print(
    f"mean potential {float(dry.mean()):.2f} V dry, {float(volume.mean()):.2f} V in ice"
)
# --8<-- [end:ice]

transcript.start("fit")
# --8<-- [start:fit]
import roma

from specter import tilt

margin = 32  # the zero-filled FFT margin of the propagation canvas, per side
edge = 8  # interpolation slack beyond the geometric coverage, per side

R = roma.unitquat_to_rotmat(poses.quaternions)
need_y, need_x = tilt.estimate_required_extent(n, nz, R)
canvas = n + 2 * margin
widest = max(need_y, need_x)
target = (
    min(widest, max(need_y, canvas)) + 2 * edge,
    min(widest, max(need_x, canvas)) + 2 * edge,
)
volume = tilt.pad_volume_xy_for_tilt(volume, target)
print(f"±60° about x samples {need_y} voxels along y of a {nz}-voxel slab")
print(f"with margins {target[0]} x {target[1]} in Y x X; the volume is {ny} x {nx}")
# --8<-- [end:fit]

transcript.start("engines")
# --8<-- [start:engines]
from specter.aberrations import Aberration, FocalSpread
from specter.constants import energy_to_wavelength
from specter.detectors import Detector
from specter.scattering import IterativeScattering

oversample = 2
scattering = IterativeScattering(
    n,
    px,
    300.0,
    scattering_model="multislice",
    oversample=oversample,
    crop=False,  # keep the propagation grid; the detector samples it down
    pad_fft=True,
    fft_pad_margin=margin,
    keep_pad=True,  # keep the margin through the lens; the detector crops it
    roi_padding_mode="zeros",  # outside the volume is vacuum
    progressbars=False,
).to("cuda")
wide = scattering.output_nxy  # the canvas: n + 2 * margin pixels

lens = Aberration(
    wide * oversample,
    px / oversample,
    300.0,
    specimen_absorption=True,
    progressbars=False,
).to("cuda")
focal = FocalSpread.from_instrument(
    2.7e7,  # Cc, Angstrom
    300e3,  # beam energy, eV
    0.7,  # energy spread, FWHM eV
    0.06e-6,  # high-tension instability
    0.01e-6,  # lens-current instability
    energy_to_wavelength(300.0),
)
camera = Detector(px, oversample=oversample, progressbars=False).to("cuda")
print(f"canvas {wide} px, propagated on {wide * oversample}")
# --8<-- [end:engines]

transcript.start("tilt")
# --8<-- [start:tilt]
from specter import rotations
from specter.aberrations import defocus_midplane_shift

volume = volume.to("cuda")
doses = exposure.doses(len(angles)).to("cuda")
no_loss = torch.zeros(1, device="cuda")


def record(i):
    """Tilt `i`: its expected counts, tilted depth and defocus shift."""
    # The pose as a sampling matrix on the fitted volume's grid.
    shift_xy = rotations.translations_angstrom_to_torch(
        poses.shifts[i : i + 1].to("cuda"), volume.shape[-1], px
    )
    theta = rotations.build_affine_matrix(R[i : i + 1].to("cuda"), shift_xy)

    with torch.no_grad():
        psi = scattering(volume, theta)

    # The wave left the tilted column's exit face; the nominal defocus is
    # that of the specimen's midplane, half the tilted depth above it.
    depth = tilt.nz_tilt_for_pose(tuple(volume.shape), theta)
    shift = defocus_midplane_shift(nz, px) - (depth - nz) * px / 2.0
    params = {k: v.to("cuda") for k, v in ctf[i : i + 1].params.items()}
    params["dfu"] = params["dfu"] - shift
    params["dfv"] = params["dfv"] - shift

    intensity = focal.intensity(lens(psi, params), px / oversample, oversample)
    expected = doses[i] * px**2 * camera.dqe0 * intensity
    counts = camera.detect_expected_counts(expected, doses[i : i + 1], no_loss, nxy=n)
    return counts[0], depth, shift


by_hand, depths, shifts = [], [], []
for i in range(len(angles)):
    counts, depth, shift = record(i)
    by_hand.append(counts)
    depths.append(depth)
    shifts.append(shift)
by_hand = torch.stack(by_hand)

difference = (by_hand - reference).abs().max() / reference.abs().max()
print(f"by hand: {tuple(by_hand.shape)}; largest difference {difference:.1e}")
for a in (0.0, 30.0, 60.0):
    i = angles.index(a)
    print(
        f"{a:+3.0f}°: {depths[i]:3d} slices, {depths[i] * px:4.0f} Å; "
        f"defocus shift {shifts[i]:5.0f} Å"
    )
# --8<-- [end:tilt]

# ---------------------------------------------------------------------------
# Figure: objects against the hand-built chain, at three tilts.
show = [0.0, 30.0, 60.0]
fig, axes = plt.subplots(2, len(show), figsize=(7.6, 5.3))
for col, a in enumerate(show):
    i = angles.index(a)
    ref = lowpass(reference[i], 12.0, px)
    hand = lowpass(by_hand[i], 12.0, px)
    lo, hi = np.percentile(ref, [0.5, 99.5])
    show_image(
        axes[0, col],
        ref,
        title=f"{a:+.0f}°",
        vmin=lo,
        vmax=hi,
        pixel_size=px if col == 0 else None,
        scalebar=250 if col == 0 else None,
    )
    show_image(axes[1, col], hand, vmin=lo, vmax=hi)
axes[0, 0].set_ylabel("Objects")
axes[1, 0].set_ylabel("Engines by hand")
fig.tight_layout()
save(fig, "tilt_series_engines-equal")
thumbnail(
    lowpass(by_hand[angles.index(30.0)], 12.0, px), "tilt_series_engines", cmap="gray"
)

# Figure: the tilted depth and the defocus shift against tilt angle.
ang = np.array(angles)
order = np.argsort(ang)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0))
ax1.plot(ang[order], np.array(depths)[order] * px, "o-", color=PALETTE[0], ms=3, lw=0.8)
ax1.axhline(nz * px, color="0.5", lw=0.8, ls=":")
ax1.set(
    xlabel="Tilt angle (°)",
    ylabel="Propagated depth (Å)",
    title="Depth of the tilted column",
)
ax2.plot(ang[order], np.array(shifts)[order], "o-", color=PALETTE[1], ms=3, lw=0.8)
ax2.set(
    xlabel="Tilt angle (°)",
    ylabel="Defocus shift (Å)",
    title="Exit face to midplane",
)
fig.tight_layout()
save(fig, "tilt_series_engines-depth")

transcript.close()
