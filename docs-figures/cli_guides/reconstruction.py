"""
Figures and printed output for docs/user-guide/reconstruction.md.

A real gold-standard `specter reconstruct particle` run on data whose answer
is known. The 64-row EMPIAR-11377 refinement in docs-figures/data/ supplies a
template .cs; 1000 rows with random poses are written from it, `specter
simulate particles --cs_path` images 8B0X at those poses, and the guide's
reconstruction command reconstructs both halves from the simulated stack.

Run with: uv run python docs-figures/cli_guides/reconstruction.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch
from cryosparc.dataset import Dataset
from scipy.spatial.transform import Rotation

from _common import (
    PALETTE,
    REPO_ROOT,
    clipped,
    run,
    save,
    show_image,
    tidy,
    write_text,
)
from specter.fft import fourier_shell_correlation
from specter.imaging import Structure

N = 1000
DEVICE = "cuda"
PIXEL = 2.924
BOX = 128
SIMULATE = f"""specter simulate particles \\
    --pdb_source 8b0x \\
    --cs_path particles.cs \\
    --n_particles {N} \\
    --n_pixels {BOX} \\
    --dose 40 \\
    --batchsize 4 \\
    --seed 0 \\
    --device {DEVICE} \\
    --output_dir particles"""
RECONSTRUCT = f"""specter reconstruct particle \\
    --config configs/reconstruct.toml \\
    --cs_file particles.cs \\
    --mrc_file particles/particles.mrcs \\
    --dose_per_angstrom 40 \\
    --fsc_ref reference.mrc \\
    --device {DEVICE} \\
    --output_dir ."""

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-reconstruct-"))
(workdir / "configs").mkdir()
shutil.copy(REPO_ROOT / "configs" / "reconstruct.toml", workdir / "configs")

# The template's ctf/accel_kv, cs, amplitude contrast and pixel size, with
# random poses, defoci between 1 and 2 um, no aberrations beyond those, and
# alternating gold-standard halves.
template = Dataset.load(
    str(REPO_ROOT / "docs-figures" / "data" / "empiar-11377-64particles-pdbframe.cs")
)
rows = template.take(np.arange(N) % len(template))
rng = np.random.default_rng(0)
rows["uid"] = np.arange(1, N + 1, dtype=np.uint64)
rows["alignments3D/pose"] = (
    Rotation.random(N, random_state=0).as_rotvec().astype(np.float32)
)
rows["alignments3D/shift"] = np.zeros((N, 2), dtype=np.float32)
rows["alignments3D/split"] = (np.arange(N) % 2).astype(np.uint32)
defocus = rng.uniform(10000.0, 20000.0, N).astype(np.float32)
rows["ctf/df1_A"] = defocus
rows["ctf/df2_A"] = defocus
rows["ctf/df_angle_rad"] = np.zeros(N, dtype=np.float32)
for field in (
    "ctf/shift_A",
    "ctf/tilt_A",
    "ctf/trefoil_A",
    "ctf/tetra_A",
    "ctf/anisomag",
):
    rows[field] = np.zeros_like(rows[field])
rows.save(str(workdir / "particles.cs"))

# The model the stack is simulated from, as the map-to-model reference.
reference = Structure.from_pdb("8b0x").render(voxel_size=PIXEL, box=BOX, device=DEVICE)
with mrcfile.new(workdir / "reference.mrc") as mrc:
    mrc.set_data(reference.potential.cpu().numpy().astype(np.float32))
    mrc.voxel_size = PIXEL

write_text("reconstruction-simulate", SIMULATE + "\n")
write_text("reconstruction-command", RECONSTRUCT + "\n")
run(SIMULATE, workdir)
log = tidy(run(RECONSTRUCT, workdir))
write_text(
    "reconstruction-run",
    "\n".join(line for line in log.splitlines() if line.strip()) + "\n",
)

job = next(path.parent for path in sorted(workdir.rglob("job.json")))
record = json.loads((job / "job.json").read_text())
gold = record["params"].get("resolution_gold_standard")
gold_text = f"{gold:.1f} Å" if isinstance(gold, (int, float)) else str(gold)
half_a = torch.as_tensor(mrcfile.read(job / "volume_A.mrc").copy())
half_b = torch.as_tensor(mrcfile.read(job / "volume_B.mrc").copy())
truth = reference.potential.cpu()

k, fsc_halves = fourier_shell_correlation(half_a, half_b, PIXEL)
_, fsc_model = fourier_shell_correlation((half_a + half_b) / 2, truth, PIXEL)
# Shells past Nyquist reach only the corners of the Fourier cube, which the
# reconstruction's k-space mask leaves empty.
keep = (np.asarray(k) > 0) & (np.asarray(k) <= 1 / (2 * PIXEL) + 1e-6)
k, fsc_halves, fsc_model = (
    np.asarray(x, dtype=float)[keep] for x in (k, fsc_halves, fsc_model)
)


def crossing(curve: np.ndarray, threshold: float) -> float:
    """Resolution in Angstrom where `curve` first falls below `threshold`."""
    below = np.nonzero(curve < threshold)[0]
    return float(1 / k[below[0]]) if len(below) else float(1 / k[-1])


write_text(
    "reconstruction-result",
    f"{job.relative_to(workdir)}: {sorted(p.name for p in job.iterdir() if p.suffix == '.mrc')}\n"
    f"half-map FSC = 0.143 at {crossing(fsc_halves, 0.143):.1f} Å"
    f" (job.json: resolution_gold_standard = {gold_text})\n"
    f"average of the halves against the model, FSC = 0.5 at"
    f" {crossing(fsc_model, 0.5):.1f} Å\n",
)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), width_ratios=[1, 1, 1.5])
average = ((half_a + half_b) / 2).numpy()
for col, (volume, title) in enumerate(
    [(truth.numpy(), "8B0X, projected"), (average, "Reconstruction, projected")]
):
    projection = volume.sum(0)
    show_image(
        axes[col],
        projection,
        cmap="gray_r",
        title=title,
        pixel_size=PIXEL if col == 0 else None,
        scalebar=100 if col == 0 else None,
        **clipped(projection),
    )
axes[2].plot(k, fsc_halves, color=PALETTE[0], lw=1.2, label="half A against half B")
axes[2].plot(k, fsc_model, color=PALETTE[1], lw=1.2, label="map against the model")
axes[2].axhline(0.143, color="0.5", lw=0.6, ls=":")
axes[2].axhline(0.5, color="0.5", lw=0.6, ls="--")
axes[2].set(
    xlabel="Spatial frequency (1/Å)",
    ylabel="FSC",
    ylim=(-0.1, 1.02),
    xlim=(0, 1 / (2 * PIXEL)),
    title="Fourier shell correlation",
)
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "reconstruction-gold-standard")

shutil.rmtree(workdir)
