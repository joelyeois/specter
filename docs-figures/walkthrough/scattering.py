"""
Figures and printed output for docs/walkthrough/scattering.md.

Run with: uv run python docs-figures/walkthrough/scattering.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import (
    PALETTE,
    Transcript,
    as_numpy,
    radial_average,
    save,
    show_image,
)

transcript = Transcript("scattering")


def relative_phase(psi: torch.Tensor) -> torch.Tensor:
    """The phase of `psi` relative to its mean wave, in radians."""
    return torch.angle(psi / psi.mean())


def shell_power(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Rotationally averaged power of a real 2D array, mean removed."""
    data = as_numpy(x).astype(np.float64)
    data = data - data.mean()
    ps = np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2
    return radial_average(ps, 1.0)


transcript.start("constants")
# --8<-- [start:constants]
from specter.constants import energy_to_wavelength, interaction_parameter

print("voltage   wavelength   sigma")
for voltage in [100.0, 120.0, 200.0, 300.0]:
    wavelength = energy_to_wavelength(voltage)  # Å
    sigma = interaction_parameter(voltage)  # rad / (V·Å)
    print(f"{voltage:4.0f} kV   {wavelength * 100:6.4f} pm   {sigma:.3e} rad/(V·Å)")
# --8<-- [end:constants]

transcript.start("setup")
# --8<-- [start:setup]
import torch

from specter.imaging import Poses, Structure

torch.manual_seed(0)
specimen = Structure.from_pdb("6bdf").render(voxel_size=1.0, box=256, device="cuda")
# --8<-- [end:setup]

transcript.start("poses")
# --8<-- [start:poses]
poses = Poses.random(4, shift=20.0)

print("quaternions", tuple(poses.quaternions.shape))
print("shifts (Å)\n", poses.shifts)
print("ids", poses.ids)
# --8<-- [end:poses]

transcript.start("materialize")
# --8<-- [start:materialize]
volumes = specimen.materialize(poses, voltage=300.0)
print(tuple(volumes.shape))  # (image, Z, Y, X)
# --8<-- [end:materialize]

fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5))
for i, ax in enumerate(axes):
    show_image(
        ax,
        volumes[i].sum(0),
        cmap="gray_r",
        title=f"Pose {i}",
        pixel_size=1.0,
        scalebar=50 if i == 0 else None,
    )
fig.tight_layout()
save(fig, "scattering-poses")
del volumes

transcript.start("scatter")
# --8<-- [start:scatter]
pose = poses[:1]
waves = specimen.scatter(pose, voltage=300.0, seed=0)

print(f"array    {tuple(waves.array.shape)} {waves.array.dtype}")
print(f"pixels   {tuple(waves.pixels().shape)}")
print(f"oversample {waves.oversample}, pixel size {waves.pixel_size} Å")
print(f"defocus shift {waves.defocus_shift} Å")
# --8<-- [end:scatter]

transcript.start("ice")
# --8<-- [start:ice]
solvated = specimen.solvate(model="gd", thickness=500.0)
waves_ice = solvated.scatter(pose, voltage=300.0, seed=0)

for name, w in [("no ice", waves), ("500 Å ice", waves_ice)]:
    psi = w.pixels()[0]
    print(
        f"{name:10s} mean phase {torch.angle(psi.mean()):+.3f} rad, "
        f"amplitude std {psi.abs().std():.4f}"
    )
# --8<-- [end:ice]

fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.4))
for row, (name, w) in enumerate([("No ice", waves), ("500 Å ice", waves_ice)]):
    psi = w.pixels()[0]
    show_image(
        axes[row, 0],
        psi.abs(),
        title=f"{name}: amplitude",
        colorbar=" ",
        pixel_size=1.0,
        scalebar=50 if row == 0 else None,
    )
    show_image(
        axes[row, 1], relative_phase(psi), title=f"{name}: phase", colorbar="rad"
    )
fig.tight_layout()
save(fig, "scattering-exit-wave")

transcript.start("models")
# --8<-- [start:models]
models = ["multislice", "rytov", "firstborn", "projection"]
exit_waves, intensities = {}, {}
for thickness in [None, 500.0, 1000.0, 2000.0]:
    s = specimen
    if thickness is not None:
        s = specimen.solvate(model="gd", thickness=thickness)
    for model in models:
        w = s.scatter(pose, voltage=300.0, seed=0, scattering_model=model)
        exit_waves[thickness, model] = w.pixels()[0]
        intensities[thickness, model] = w.intensity(cc=None).pixels()[0]


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm())


print("ice (Å)   model        mean I   std I   corr. with multislice")
for (thickness, model), intensity in intensities.items():
    reference = intensities[thickness, "multislice"]
    label = "none" if thickness is None else f"{thickness:.0f}"
    flat = intensity.std() < 1e-4
    pattern = "  flat" if flat else f"{correlation(intensity, reference):6.3f}"
    print(
        f"{label:>7s}   {model:10s}  {intensity.mean():7.3f}  "
        f"{intensity.std():6.3f}   {pattern}"
    )
# --8<-- [end:models]

fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), layout="constrained")
phase_ms = as_numpy(relative_phase(exit_waves[500.0, "multislice"]))
intensity_ms = as_numpy(intensities[500.0, "multislice"])
phase_range = np.percentile(phase_ms, [0.5, 99.5])
intensity_range = np.percentile(intensity_ms, [0.5, 99.5])
for col, model in enumerate(models):
    psi = exit_waves[500.0, model]
    intensity = as_numpy(intensities[500.0, model])
    im_phase = show_image(
        axes[0, col],
        relative_phase(psi),
        title=f"{model}: phase",
        vmin=phase_range[0],
        vmax=phase_range[1],
        pixel_size=1.0,
        scalebar=50 if col == 0 else None,
    )
    # first Born's mean is off the shared scale; give it its own, same width
    offset = intensity.mean() - intensity_ms.mean() if model == "firstborn" else 0.0
    show_image(
        axes[1, col],
        intensity,
        title=f"{model}: |ψ|², mean {intensity.mean():.2f}",
        vmin=intensity_range[0] + offset,
        vmax=intensity_range[1] + offset,
    )
fig.colorbar(im_phase, ax=list(axes[0]), fraction=0.02, pad=0.02, label="Phase (rad)")
save(fig, "scattering-models")

thicknesses = [None, 500.0, 1000.0, 2000.0]
xs = np.arange(len(thicknesses))
xlabels = ["none", "500", "1000", "2000"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9))
for color, model in zip(PALETTE, models):
    means, corrs = [], []
    for t in thicknesses:
        intensity = intensities[t, model]
        means.append(float(intensity.mean()))
        if model == "projection":
            corrs.append(np.nan)  # a flat image has no pattern to correlate
        else:
            corrs.append(float(correlation(intensity, intensities[t, "multislice"])))
    ax1.semilogy(xs, means, "o-", color=color, lw=1.2, ms=4, label=model)
    ax2.plot(xs, corrs, "o-", color=color, lw=1.2, ms=4, label=model)
ax1.set(
    xticks=xs,
    xticklabels=xlabels,
    xlabel="Ice thickness (Å)",
    ylabel="Mean exit-wave intensity",
    title="Conservation of electrons",
)
ax2.set(
    xticks=xs,
    xticklabels=xlabels,
    xlabel="Ice thickness (Å)",
    ylabel="Correlation with multislice",
    title="Intensity pattern",
    ylim=(0, 1.05),
)
ax1.legend(frameon=False)
fig.tight_layout()
save(fig, "scattering-model-accuracy")

transcript.start("oversample")
# --8<-- [start:oversample]
thick = specimen.solvate(model="gd", thickness=2000.0)
images = {}
for oversample in [1, 2, 4]:
    w = thick.scatter(pose, voltage=300.0, seed=0, oversample=oversample)
    images[oversample] = w.intensity(cc=None).pixels()[0].double()

reference = images[4]
contrast = (reference - reference.mean()).square().sum()
for oversample in [1, 2]:
    error = (images[oversample] - reference).square().sum() / contrast
    print(f"oversample {oversample}: error {error:.1e} of the contrast power")
# --8<-- [end:oversample]

fig, ax = plt.subplots(figsize=(4.6, 3.0))
k, p_ref = shell_power(reference)
ax.semilogy(k[1:], p_ref[1:], color="0.3", lw=1.2, label="image (oversample=4)")
for color, oversample in zip(PALETTE, [1, 2]):
    _, p_err = shell_power(images[oversample] - reference)
    ax.semilogy(
        k[1:], p_err[1:], color=color, lw=1.2, label=f"error, oversample={oversample}"
    )
ax.set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="Power",
    title="Exit-wave intensity in 2000 Å of ice",
)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "scattering-oversample")

transcript.start("voltage")
# --8<-- [start:voltage]
phases = {}
for voltage in [120.0, 200.0, 300.0]:
    w = specimen.scatter(pose, voltage=voltage, seed=0)
    phases[voltage] = torch.angle(w.pixels()[0])
    print(f"{voltage:3.0f} kV: largest phase shift {phases[voltage].max():.2f} rad")
# --8<-- [end:voltage]

vmax = float(max(p.max() for p in phases.values()))
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4), layout="constrained")
for i, (ax, (voltage, phase)) in enumerate(zip(axes, phases.items())):
    im = show_image(
        ax,
        phase,
        title=f"{voltage:.0f} kV",
        vmin=0,
        vmax=vmax,
        pixel_size=1.0,
        scalebar=50 if i == 0 else None,
    )
fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02, label="Phase (rad)")
save(fig, "scattering-voltage")

transcript.start("settings")
# --8<-- [start:settings]
from specter.settings import Propagation

propagation = Propagation(scattering_model="rytov", oversample=2)
a = specimen.scatter(pose, voltage=300.0, seed=0, propagation=propagation)
b = specimen.scatter(
    pose, voltage=300.0, seed=0, scattering_model="rytov", oversample=2
)
print(torch.equal(a.array, b.array))

try:
    specimen.scatter(pose, voltage=300.0, propagation=propagation, oversample=1)
except TypeError as error:
    print(f"TypeError: {error}")
# --8<-- [end:settings]

transcript.close()
