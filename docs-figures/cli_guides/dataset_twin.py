"""
Figures and printed output for docs/user-guide/dataset-twin.md.

Runs the guide's `specter match particles` command on the 64 EMPIAR-11377
particles kept in docs-figures/data/, in a scratch directory, and keeps the
report it writes.

Run with: uv run python docs-figures/cli_guides/dataset_twin.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from _common import OUT_DIR, REPO_ROOT, quantize_png, run, tidy, write_text

DATA = REPO_ROOT / "docs-figures" / "data"
COMMAND = """specter match particles \\
    --metadata_path empiar-11377-64particles-pdbframe.cs \\
    --images_path empiar-11377-64particles-128px.mrcs \\
    --pdb_source 8b0x \\
    --detector_model falcon4i_300kv \\
    --dose 40 --dose_rate 4.1 --energy_filter true \\
    --n_battery 64 \\
    --seed 0 \\
    --device cuda \\
    --output_dir match"""

workdir = Path(tempfile.mkdtemp(prefix="specter-guide-match-"))
for name in (
    "empiar-11377-64particles-pdbframe.cs",
    "empiar-11377-64particles-128px.mrcs",
):
    shutil.copy(DATA / name, workdir)
# Each probe is an ordinary `specter simulate particles` run that prints its
# own stage headers; the log keeps only what the match itself reports.
PROBE_LINES = (
    "Building 3D scattering potential",
    "Sampling poses, defocus",
    "Generating images on",
    "Post-processing",
    "Saving .mrcs + .star",
    "✓ match/probes/",
    "Total time:",
)
write_text("dataset_twin-command", COMMAND + "\n")
log = tidy(run(COMMAND, workdir))
write_text(
    "dataset_twin-run",
    "\n".join(
        line
        for line in log.splitlines()
        if line.strip() and not any(probe in line for probe in PROBE_LINES)
    )
    + "\n",
)

out = workdir / "match"
write_text("dataset_twin-toml", tidy((out / "matched.toml").read_text()))
OUT_DIR.mkdir(parents=True, exist_ok=True)
figure = OUT_DIR / "dataset_twin-report.png"
shutil.copy(out / "match_report.png", figure)
quantize_png(figure)
print(f"wrote {figure.relative_to(REPO_ROOT)}")

shutil.rmtree(workdir)
