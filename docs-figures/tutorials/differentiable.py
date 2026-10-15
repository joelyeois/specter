"""
Figures and printed output for docs/tutorials/differentiable.md.

Run with: uv run python docs-figures/tutorials/differentiable.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import PALETTE, Transcript, as_numpy, save

transcript = Transcript("differentiable")

transcript.start("setup")
# --8<-- [start:setup]
from dataclasses import replace

import torch

from specter.imaging import CTF, Exposure, Poses, Structure

specimen = Structure.from_pdb("6bdf").render(voxel_size=2.0, box=128, device="cuda")
poses = Poses.random(16, seed=1)
exposure = Exposure(dose=20.0)


def image(scale: torch.Tensor, defocus: torch.Tensor) -> torch.Tensor:
    """Expected counts for every pose, a differentiable function of both."""
    scaled = replace(specimen, potential=specimen.potential * scale)
    waves = scaled.scatter(poses, voltage=300.0)
    ctf = CTF.from_values(defocus, n=len(poses))
    images = waves.apply_ctf(ctf).intensity(cc=None)
    return images.detect(exposure, noise=None).array


print(f"{len(poses)} images of {tuple(image(1.0, 15000.0).shape[1:])} pixels")
# --8<-- [end:setup]

transcript.start("gradcheck")
# --8<-- [start:gradcheck]
scale = torch.tensor(0.9, requires_grad=True)
defocus = torch.tensor(15000.0, requires_grad=True)

g = torch.Generator(device="cuda").manual_seed(0)
weights = torch.randn(len(poses), 128, 128, generator=g, device="cuda")


def f(scale, defocus):
    """A scalar that depends on every pixel."""
    return (weights * image(scale, defocus)).sum()


f(scale, defocus).backward()
print(f"autograd   df/dscale   = {scale.grad.item():.6g}")
print(f"autograd   df/ddefocus = {defocus.grad.item():.6g} per Å")

with torch.no_grad():
    for h in [1e-1, 1e-2, 1e-3]:
        fd = (f(0.9 + h, 15000.0) - f(0.9 - h, 15000.0)) / (2 * h)
        print(f"central difference, h = {h:<6g} df/dscale   = {fd.item():.6g}")
    for h in [100.0, 10.0, 1.0]:
        fd = (f(0.9, 15000.0 + h) - f(0.9, 15000.0 - h)) / (2 * h)
        print(f"central difference, h = {h:<6g} df/ddefocus = {fd.item():.6g}")
# --8<-- [end:gradcheck]

# The figure sweeps the step size more finely than the quoted section prints.
with torch.no_grad():
    sweep = {}
    for name, steps, at in [
        ("scale", np.logspace(-4, 0, 13), lambda h: (0.9 + h, 15000.0)),
        ("defocus", np.logspace(-1, 3, 13), lambda h: (0.9, 15000.0 + h)),
    ]:
        exact = (scale if name == "scale" else defocus).grad.item()
        errors = []
        for h in steps:
            hi = f(*at(h))
            lo = f(*at(-h))
            errors.append(abs((hi - lo).item() / (2 * h) - exact) / abs(exact))
        sweep[name] = (steps, np.array(errors))

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
for ax, (name, (steps, errors)), unit in zip(axes, sweep.items(), ["", " (Å)"]):
    ax.loglog(steps, errors, "o-", color=PALETTE[0], ms=3.5, lw=1.2)
    ax.set(
        xlabel=f"Finite-difference step{unit}",
        ylabel="Relative difference from autograd",
        title=f"d/d{name}",
    )
    ax.grid(True, which="major", alpha=0.3)
fig.tight_layout()
save(fig, "differentiable-gradcheck")

transcript.start("data")
# --8<-- [start:data]
true_scale, true_defocus = 0.85, 15000.0
with torch.no_grad():
    data = (
        replace(specimen, potential=specimen.potential * true_scale)
        .scatter(poses, voltage=300.0)
        .apply_ctf(CTF.from_values(true_defocus, n=len(poses)))
        .intensity(cc=None)
        .detect(exposure, seed=0)  # Poisson counts
        .array
    )


def loss(predicted: torch.Tensor) -> torch.Tensor:
    """Poisson negative log-likelihood per pixel, up to a constant."""
    return (predicted - data * torch.log(predicted)).mean()


print(f"data: {tuple(data.shape)}, {data.mean():.1f} counts per pixel")
# --8<-- [end:data]

transcript.start("landscape")
# --8<-- [start:landscape]
with torch.no_grad():
    scan = torch.arange(10000.0, 20001.0, 50.0)
    profile = torch.stack([loss(image(true_scale, d)) for d in scan]).cpu()
best = scan[profile.argmin()]
print(f"loss minimum on a 50 Å grid at {best:.0f} Å (true {true_defocus:.0f} Å)")
left, right = profile[: profile.argmin() + 1], profile[profile.argmin() :]
downhill = bool((left.diff() < 0).all() and (right.diff() > 0).all())
print(f"falls towards the minimum from both ends of the scan: {downhill}")
# --8<-- [end:landscape]

transcript.start("fit")
# --8<-- [start:fit]
trajectories = {}
for start in [1.1, 1.3, 1.4]:  # µm
    defocus_um = torch.tensor(start, requires_grad=True)
    scale = torch.tensor(1.1, requires_grad=True)
    optimizer = torch.optim.Adam([defocus_um, scale], lr=0.01)
    history = []
    for step in range(200):
        optimizer.zero_grad()
        value = loss(image(scale, defocus_um * 1e4))
        value.backward()
        optimizer.step()
        history.append((defocus_um.item() * 1e4, scale.item(), value.item()))
    trajectories[start] = history
    print(
        f"start {start * 1e4:5.0f} Å, scale 1.10 -> "
        f"defocus {history[-1][0]:7.1f} Å, scale {history[-1][1]:.4f}"
    )
# --8<-- [end:fit]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
minimum = float(profile.min())
axes[0].plot(as_numpy(scan), (as_numpy(profile) - minimum) * 1e3, color="0.3", lw=1)
axes[0].axvline(true_defocus, color="0.6", ls=":", lw=1)
for color, (start, history) in zip(PALETTE, trajectories.items()):
    d = np.array([h[0] for h in history])
    s = np.array([h[1] for h in history])
    axes[1].plot(d, color=color, lw=1.2, label=f"from {start:g} µm")
    axes[2].plot(s, color=color, lw=1.2)
    axes[0].plot(
        d[0],
        0.03,
        "^",
        color=color,
        ms=7,
        transform=axes[0].get_xaxis_transform(),
        clip_on=False,
    )
axes[0].set(
    xlabel="Defocus (Å)",
    ylabel="Loss above minimum (×10⁻³)",
    title="Loss against defocus, true scale",
)
axes[1].axhline(true_defocus, color="0.6", ls=":", lw=1)
axes[1].set(xlabel="Step", ylabel="Defocus (Å)", title="Defocus")
axes[1].legend(frameon=False, fontsize=7)
axes[2].axhline(true_scale, color="0.6", ls=":", lw=1)
axes[2].set(xlabel="Step", ylabel="Potential scale", title="Potential scale")
fig.tight_layout()
save(fig, "differentiable-fit")

transcript.start("atoms")
# --8<-- [start:atoms]
from specter.imaging import Specimen

ubiquitin = Structure.from_pdb("1ubq", scattering_factors="kirkland")
true_xyz = ubiquitin.pdb.coordinates.float().to("cuda")
elements = ubiquitin.pdb.atomic_numbers
views = Poses.random(24, seed=2)
lens = CTF.from_values(10000.0, n=len(views))


def project(xyz: torch.Tensor) -> torch.Tensor:
    """Noiseless images of the atoms at every view, differentiable in xyz."""
    atoms = Specimen.from_atoms(xyz, elements, voxel_size=1.0, box=56)
    waves = atoms.scatter(views, voltage=300.0)
    images = waves.apply_ctf(lens).intensity(cc=None)
    return images.detect(Exposure(dose=1000.0), noise=None).array


with torch.no_grad():
    target = project(true_xyz)
print(f"{len(true_xyz)} atoms, {len(views)} views of {tuple(target.shape[1:])}")
# --8<-- [end:atoms]

transcript.start("atoms-fit")


# --8<-- [start:atoms-fit]
def rmsd(xyz: torch.Tensor) -> float:
    return ((xyz - true_xyz) ** 2).sum(dim=1).mean().sqrt().item()


runs = {}
for sigma in [0.25, 0.5]:  # Å of noise per axis
    g = torch.Generator(device="cuda").manual_seed(0)
    start = true_xyz + sigma * torch.randn(true_xyz.shape, generator=g, device="cuda")
    xyz = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([xyz], lr=0.03)
    curve = []
    for step in range(300):
        optimizer.zero_grad()
        value = ((project(xyz) - target) ** 2).mean()
        value.backward()
        if step == 0:
            first_gradient = xyz.grad.clone()
        optimizer.step()
        curve.append(rmsd(xyz.detach()))
    runs[sigma] = (start, xyz.detach(), curve, first_gradient)
    error = (xyz.detach() - true_xyz).norm(dim=1)
    print(
        f"σ = {sigma} Å: RMSD {rmsd(start):.3f} -> {curve[-1]:.4f} Å, "
        f"median atom error {error.median():.4f} Å, "
        f"{int((error > 0.2).sum())} atoms more than 0.2 Å off, loss {value:.3g}"
    )
# --8<-- [end:atoms-fit]

transcript.start("swaps")
# --8<-- [start:swaps]
start, final, _, _ = runs[0.5]
error = (final - true_xyz).norm(dim=1)
stuck = (error > 0.2).nonzero().flatten()
distance = torch.cdist(final[stuck], true_xyz)  # to every true site
same = elements.to("cuda")[stuck, None] == elements.to("cuda")[None, :]
near_same = distance.masked_fill(~same, torch.inf).min(dim=1).values < 0.2
near_other = distance.masked_fill(same, torch.inf).min(dim=1).values < 0.2
print(f"{len(stuck)} atoms more than 0.2 Å from their own site:")
print(f"  {int(near_same.sum())} on the site of another atom of the same element")
print(f"  {int(near_other.sum())} on the site of an atom of another element")
print(f"  {int((~near_same & ~near_other).sum())} elsewhere")
symbol = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
taken = distance.masked_fill(same, torch.inf).argmin(dim=1)
pairs, gaps = {}, []
for i, j, other in zip(stuck.tolist(), taken.tolist(), near_other.tolist()):
    if other:
        key = f"{symbol[int(elements[i])]} on {symbol[int(elements[j])]}"
        pairs[key] = pairs.get(key, 0) + 1
        gaps.append((true_xyz[i] - true_xyz[j]).norm().item())
print("  cross-element swaps:", ", ".join(f"{k} {v}" for k, v in pairs.items()))
print(f"  their two sites are {min(gaps):.2f} to {max(gaps):.2f} Å apart")
starting = (start - true_xyz).norm(dim=1)
print(
    f"mean starting error: stuck {starting[stuck].mean():.2f} Å, "
    f"recovered {starting[error <= 0.2].mean():.2f} Å"
)
# --8<-- [end:swaps]

transcript.start("first-gradient")
# --8<-- [start:first-gradient]
start, _, _, gradient = runs[0.5]
cosine = torch.nn.functional.cosine_similarity(-gradient, true_xyz - start, dim=1)
print(
    f"atoms whose first step points within 90° of home: {(cosine > 0).float().mean():.0%}"
)
print(f"median cosine between the step and the way home: {cosine.median():.2f}")
# --8<-- [end:first-gradient]

# The figures: recovery curves, final against starting error, and the
# first gradient against the way home.
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))
for color, (sigma, (start, final, curve, _)) in zip(PALETTE, runs.items()):
    axes[0].semilogy(curve, color=color, lw=1.2, label=f"σ = {sigma} Å")
axes[0].set(xlabel="Step", ylabel="RMSD to the true coordinates (Å)", title="Recovery")
axes[0].legend(frameon=False)
start, final, _, gradient = runs[0.5]
e0 = as_numpy((start - true_xyz).norm(dim=1))
e1 = as_numpy((final - true_xyz).norm(dim=1))
axes[1].scatter(e0, np.maximum(e1, 1e-4), s=4, color=PALETTE[0], alpha=0.6)
axes[1].set_yscale("log")
axes[1].set(
    xlabel="Starting error (Å)", ylabel="Final error (Å)", title="σ = 0.5 Å, per atom"
)
axes[2].hist(as_numpy(cosine), bins=40, color=PALETTE[2])
axes[2].set(
    xlabel="cos(−gradient, way home)", ylabel="Atoms", title="First gradient, σ = 0.5 Å"
)
fig.tight_layout()
save(fig, "differentiable-atoms")

# The gallery thumbnail: the defocus loss landscape and where the fits began.
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(as_numpy(scan), as_numpy(profile) - minimum, color="0.25", lw=2)
ax.axvline(true_defocus, color="0.6", ls=":", lw=1.2)
for color, (start, history) in zip(PALETTE, trajectories.items()):
    d = np.array([h[0] for h in history])
    on_curve = np.interp(d, as_numpy(scan), as_numpy(profile) - minimum)
    ax.plot(d[::4], on_curve[::4], "o", color=color, ms=3.5, alpha=0.8)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
save(fig, "differentiable-thumbnail")

transcript.close()
