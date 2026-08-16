"""Subprocess runner + output fingerprinting for the `specter` CLI QA sweep.

Every check in this harness goes through the real CLI entry point in a real
subprocess, so what is tested is exactly what a user types -- no in-process
shortcuts that could bypass argument parsing or config loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# The project venv if there is one (so a plain `python qa/sweep.py` still
# exercises the installed specter), else whatever interpreter is running us.
_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


@dataclass
class RunResult:
    """Outcome of one CLI invocation."""

    name: str
    argv: list[str]
    returncode: int
    wall_s: float
    stdout_tail: str
    stderr_tail: str
    out_dir: str = ""
    fingerprint: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def error_line(self) -> str:
        """Last non-empty stderr line -- the bit a user would actually read."""
        for line in reversed(self.stderr_tail.splitlines()):
            if line.strip():
                return line.strip()
        return ""


def _hash_array(arr: Any) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _fingerprint_mrc(path: Path) -> dict[str, Any]:
    import mrcfile
    import numpy as np

    with mrcfile.open(str(path), permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float64)
        voxel = mrc.voxel_size
        return {
            "kind": "mrc",
            "shape": list(data.shape),
            "voxel_size": round(float(voxel.x), 6),
            "mean": round(float(data.mean()), 8),
            "std": round(float(data.std()), 8),
            "min": round(float(data.min()), 8),
            "max": round(float(data.max()), 8),
            "hash": _hash_array(data.astype(np.float32)),
            "nan": bool(np.isnan(data).any()),
            "inf": bool(np.isinf(data).any()),
        }


def _fingerprint_star(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    rows = [
        ln for ln in text.splitlines() if ln.strip() and not ln.startswith(("_", "#"))
    ]
    return {
        "kind": "star",
        "n_lines": len(text.splitlines()),
        "n_rows": len(rows),
        "hash": hashlib.md5(text.encode()).hexdigest()[:16],
    }


def fingerprint_outputs(out_dir: Path) -> dict[str, Any]:
    """Summarise every artifact under `out_dir` in a comparable, seed-stable way."""
    fp: dict[str, Any] = {}
    if not out_dir.exists():
        return fp
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(out_dir))
        try:
            if path.suffix in (".mrc", ".mrcs", ".st"):
                fp[rel] = _fingerprint_mrc(path)
            elif path.suffix == ".star":
                fp[rel] = _fingerprint_star(path)
            else:
                fp[rel] = {"kind": "file", "bytes": path.stat().st_size}
        except Exception as exc:  # a corrupt output is itself a finding
            fp[rel] = {"kind": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    return fp


def run_cli(
    name: str,
    argv: list[str],
    out_dir: Path,
    cwd: Path | None = None,
    timeout: float = 900.0,
    threads: int = 4,
) -> RunResult:
    """
    Run `specter <argv>` in a subprocess and fingerprint whatever it wrote.

    Parameters
    ----------
    name : str
        Label for this run, used as the results key.
    argv : list[str]
        Arguments after the `specter` executable, e.g. ``["simulate", "particles", ...]``.
    out_dir : Path
        Directory the run is expected to write into; fingerprinted afterwards.
    cwd : Path or None
        Working directory for the subprocess. Defaults to the repo root.
    timeout : float
        Seconds before the run is killed and recorded as a timeout.
    threads : int
        Value for OMP/MKL thread env vars, so many runs can go in parallel.

    Returns
    -------
    RunResult
        Return code, timing, output tails, and the artifact fingerprint.
    """
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "COLUMNS": "160",
        }
    )
    cmd = [PYTHON, "-m", "specter.cli._cli", *argv]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd or REPO_ROOT),
            env=env,
            timeout=timeout,
        )
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = -9
        stdout = (
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = f"TIMEOUT after {timeout}s"
    wall = time.perf_counter() - t0
    return RunResult(
        name=name,
        argv=argv,
        returncode=rc,
        wall_s=round(wall, 2),
        stdout_tail="\n".join(stdout.splitlines()[-25:]),
        stderr_tail="\n".join(stderr.splitlines()[-40:]),
        out_dir=str(out_dir),
        fingerprint=fingerprint_outputs(out_dir),
    )


def save_results(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))


def load_results(path: Path) -> list[RunResult]:
    return [RunResult(**row) for row in json.loads(path.read_text())]


if __name__ == "__main__":
    print(f"python: {PYTHON}\nrepo:   {REPO_ROOT}\nargs:   {sys.argv[1:]}")
