"""
Phase 2: what happens when a user gives a flag a value that makes no sense?

Phase 1 asked whether a good value does something. This asks whether a bad one
fails *usefully*. The bar is not "must work" -- it is:

  * fail, rather than silently produce a plausible-looking wrong result;
  * fail fast, before minutes of PDB fetching and voxelization;
  * fail with a message naming the flag, so the user knows what to change.

A run that dies 6 frames deep in an FFT with a shape mismatch technically
"failed", but tells a researcher nothing about which of their 77 flags was
wrong. Those are reported as UNCLEAR. Only the final error line counts as
"naming" the flag -- a traceback whose *source lines* happen to contain
`config.potential_method` is not telling the user anything.

Usage
-----
    python qa/garbage.py particles
    python qa/garbage.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import PYTHON, REPO_ROOT, run_cli  # noqa: E402
from spec import SPECS  # noqa: E402
from sweep import WORK_DIR  # noqa: E402


@dataclass
class Case:
    """One bad-input probe."""

    flag: str
    value: str
    why: str
    # Set for values a Literal/Choice should reject: the CLI blocks these at
    # parse time, so they are injected through a TOML file instead, which is
    # the path that actually skips Click's validation.
    via_toml: bool = False

    @property
    def label(self) -> str:
        return f"{self.flag}={self.value}"


# Values that are wrong in a way no amount of downstream cleverness can fix.
_NUMERIC_ZERO_NEG = [
    Case("n_pixels", "0", "a box with no voxels"),
    Case("n_pixels", "-32", "negative box size"),
    Case("pixel_size", "0", "zero-size voxels: every physical scale divides by this"),
    Case("pixel_size", "-1.0", "negative voxel size"),
    Case("n_particles", "0", "asking for no particles"),
    Case("n_particles", "-5", "negative particle count"),
    Case("voltage", "0", "no accelerating voltage: wavelength is undefined"),
    Case("voltage", "-300", "negative accelerating voltage"),
    Case("dose", "0", "zero dose: no electrons, so no image"),
    Case("dose", "-20", "negative dose"),
    Case("alpha", "-0.1", "negative amplitude contrast ratio"),
    Case("alpha", "1.5", "amplitude contrast ratio above 1"),
    Case("cs", "-2.0", "negative spherical aberration"),
    Case("ice_thickness", "-100", "negative ice thickness"),
    Case("batchsize", "0", "zero-sized batches"),
    Case("batchsize", "-4", "negative batch size"),
]

_RANGES = [
    Case("dose", "60,20", "range given high,low"),
    Case("defocus", "15000,5000", "defocus range given high,low"),
]

_PATHS = [
    Case("pdb_code", "specter-data/pdb/definitely-not-here.cif", "missing structure"),
    Case("cs_path", "/nonexistent/particles.cs", "missing CryoSPARC file"),
    Case("star_path", "/nonexistent/particles.star", "missing RELION file"),
]

_LITERALS = [
    Case("scattering_model", "definitely-not-a-model", "invalid choice", via_toml=True),
    Case("potential_method", "4d", "invalid choice", via_toml=True),
    Case("ice_model", "frozen", "invalid choice", via_toml=True),
]

CASES: dict[str, list[Case]] = {
    "particles": _NUMERIC_ZERO_NEG + _RANGES + _PATHS + _LITERALS,
    "micrograph": [
        Case("n_pixels", "0", "a box with no voxels"),
        Case("pixel_size", "0", "zero-size voxels"),
        Case("micrograph_size", "0", "a micrograph with no pixels"),
        Case("micrograph_size", "-128", "negative micrograph size"),
        Case("n_micrographs", "0", "asking for no micrographs"),
        Case("voltage", "0", "no accelerating voltage"),
        Case("dose", "-20", "negative dose"),
        Case("ice_model", "frozen", "invalid choice", via_toml=True),
    ],
    "tiltseries": [
        Case("voxel_size", "0", "zero-size voxels"),
        Case("voxel_size", "-6", "negative voxel size"),
        Case("micrograph_size", "0", "a tilt with no pixels"),
        Case("n_tilts", "0", "a tilt series with no tilts"),
        Case("n_tilts", "-5", "negative tilt count"),
        Case("dose_per_tilt", "-3", "negative dose"),
        Case("min_tilt_angle", "60", "min tilt above max tilt (max is 30 here)"),
        Case("tilt_axis", "z", "invalid choice", via_toml=True),
    ],
    "tomogram": [
        Case("voxel_size", "0", "zero-size voxels"),
        Case("voxel_size", "-12", "negative voxel size"),
        Case("n_tomograms", "0", "asking for no tomograms"),
        Case("n_tomograms", "-2", "negative tomogram count"),
        Case("gap", "-50", "negative clearance between placed instances"),
        Case("filler_occupancy_fraction", "-0.1", "negative occupancy"),
    ],
}


def _import_floor_seconds() -> float:
    """
    Time to import the pipeline stack -- the earliest any run can fail.

    Measured rather than hardcoded: it is ~11 s here (torch + lightning +
    specter), dwarfs the validation it gates, and varies by machine. Calibrating
    against `--help` instead would be wrong, since `--help` never imports the
    pipelines at all.
    """
    t0 = time.perf_counter()
    subprocess.run(
        [PYTHON, "-c", "import specter.pipelines"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    return time.perf_counter() - t0


# A rejection within the import floor (plus slack for running many at once) was
# made from the config alone. Anything slower fetched a structure and voxelized
# it first, then failed -- which is what a user should not sit through to learn
# they typed a negative dose.
FAST_FAIL_SLACK_S = 6.0


def _toml_with(key: str, value: str, base_config: str) -> str:
    """Write a copy of `base_config` with one key overridden, and return it."""
    text = Path(base_config).read_text() if base_config else ""
    tmp = Path(tempfile.mkdtemp(prefix="specter-qa-toml-")) / "config.toml"
    quoted = (
        value if value.replace(".", "").replace("-", "").isdigit() else f'"{value}"'
    )
    tmp.write_text(f"{text}\n\n[qa_override]\n{key} = {quoted}\n")
    return str(tmp)


def probe(command: str, workers: int, budget: float) -> list[tuple[Case, str, str]]:
    """Run every bad-input case for one command and classify the outcome."""
    spec = SPECS[command]
    root = WORK_DIR / f"garbage_{command}"
    root.mkdir(parents=True, exist_ok=True)

    def _run(case: Case):
        out = root / case.label.replace("/", "_")
        argv = [*spec.argv, *spec.baseline]
        if case.via_toml:
            argv += ["--config", _toml_with(case.flag, case.value, spec.config_path)]
        else:
            argv += [f"--{case.flag}", case.value]
        argv += ["--output_dir", str(out), "--filename", "out"]
        return run_cli(case.label, argv, out, timeout=spec.timeout, threads=1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run, CASES[command]))

    rows = []
    for case, res in zip(CASES[command], results):
        err = res.error_line
        if res.ok:
            status, detail = "ACCEPTED", "ran to completion with a nonsense value"
        elif case.flag.lower() in err.lower():
            status = "NAMED" if res.wall_s <= budget else "NAMED-SLOW"
            detail = f"{res.wall_s}s: {err[:100]}"
        else:
            status, detail = "UNCLEAR", f"{res.wall_s}s: {err[:100]}"
        rows.append((case, status, detail))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=[*CASES, "all"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    commands = list(CASES) if args.command == "all" else [args.command]
    floor = _import_floor_seconds()
    budget = floor + FAST_FAIL_SLACK_S
    print(f"import floor {floor:.1f}s -> fast-fail budget {budget:.1f}s")
    totals: dict[str, int] = {}
    for command in commands:
        print(f"\n=== {command} ===")
        for case, status, detail in probe(command, args.workers, budget):
            totals[status] = totals.get(status, 0) + 1
            marker = "  " if status.startswith("NAMED") else "!!"
            print(f"{marker} [{status:11s}] --{case.label:52s} {case.why}")
            if not status.startswith("NAMED"):
                print(f"       {detail}")
    print(f"\nsummary: {totals}")
    print(f"repo: {REPO_ROOT}")


if __name__ == "__main__":
    main()
