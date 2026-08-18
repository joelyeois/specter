"""
Phase 3: the paths a real user takes that phases 1 and 2 deliberately skipped.

Phase 1 (`sweep.py`) ran everything single-threaded on CPU against local
structures, because that is the only way to get bit-reproducible comparisons.
That leaves whole surfaces untested, and they are exactly the ones a researcher
hits on day one:

  * **devices** -- `cuda`, `cuda:N`, multi-GPU DDP, and `batchsize="auto"`
    sizing itself to real free memory rather than a CPU fallback;
  * **output paths** -- a directory that does not exist yet, a relative path
    resolved against the cwd, a re-run over existing files, spaces in names;
  * **dataset-driven runs** -- `--cs_path`/`--star_path` through the CLI (the
    pipeline is covered by tests, the flags are not);
  * **`--assembly`** -- unverifiable offline, since every cached structure has
    an identical asymmetric unit and biological assembly.

Usage
-----
    python tools/cli-qa/phase3.py                # everything available
    python tools/cli-qa/phase3.py --skip-network
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import PYTHON, REPO_ROOT, run_cli  # noqa: E402
from spec import PDB_A  # noqa: E402
from sweep import WORK_DIR  # noqa: E402

ROOT = WORK_DIR / "phase3"

# Small enough to run many of, big enough to be a real forward pass.
BASE = [
    "simulate",
    "particles",
    # Absolute: the relative-output_dir check below runs from a different cwd,
    # where a repo-relative structure path would no longer resolve (and now
    # fails validation, correctly, but for the wrong reason).
    "--pdb_code",
    str(REPO_ROOT / PDB_A),
    "--n_particles",
    "2",
    "--n_pixels",
    "48",
    "--pixel_size",
    "3.0",
    "--seed",
    "1234",
    "--batchsize",
    "2",
]

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")


def gpu_count() -> int:
    out = subprocess.run(
        [PYTHON, "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    try:
        return int(out.stdout.strip())
    except ValueError:
        return 0


def check_devices(n_gpus: int) -> None:
    """Every device spelling the help text advertises must actually run."""
    print("\n=== devices ===")
    if n_gpus == 0:
        record("cuda", False, "no GPU visible -- cannot test")
        return

    for label, device in [("cuda", "cuda"), ("cuda:0", "cuda:0")]:
        out = ROOT / f"dev_{label.replace(':', '')}"
        res = run_cli(
            label,
            [*BASE, "--device", device, "--output_dir", str(out), "--filename", "out"],
            out,
        )
        record(f"--device {label}", res.ok, res.error_line or f"{res.wall_s}s")

    # batchsize="auto" is the shipped default and only does anything real on a
    # GPU, where it sizes itself to actual free memory.
    out = ROOT / "dev_auto_batch"
    res = run_cli(
        "auto",
        [
            "simulate",
            "particles",
            "--pdb_code",
            PDB_A,
            "--n_particles",
            "4",
            "--n_pixels",
            "48",
            "--pixel_size",
            "3.0",
            "--device",
            "cuda:0",
            "--output_dir",
            str(out),
            "--filename",
            "out",
        ],
        out,
    )
    record("batchsize=auto on GPU", res.ok, res.error_line or f"{res.wall_s}s")

    if n_gpus >= 2:
        out = ROOT / "dev_multi"
        res = run_cli(
            "multi",
            [
                "simulate",
                "particles",
                "--pdb_code",
                PDB_A,
                "--n_particles",
                "8",
                "--n_pixels",
                "48",
                "--pixel_size",
                "3.0",
                "--batchsize",
                "2",
                "--device",
                "0,1",
                "--output_dir",
                str(out),
                "--filename",
                "out",
            ],
            out,
            timeout=1200,
        )
        n = res.fingerprint.get("out.mrcs", {}).get("shape", [0])[0]
        record(
            "--device 0,1 (multi-GPU DDP)",
            res.ok and n == 8,
            res.error_line or f"{res.wall_s}s, {n} particles written",
        )
    else:
        record("multi-GPU DDP", False, f"only {n_gpus} GPU visible")


def check_paths() -> None:
    """Output-path handling, where a wrong answer costs a user their results."""
    print("\n=== output paths ===")

    nested = ROOT / "does" / "not" / "exist" / "yet"
    shutil.rmtree(ROOT / "does", ignore_errors=True)
    res = run_cli(
        "nested", [*BASE, "--output_dir", str(nested), "--filename", "out"], nested
    )
    record(
        "output_dir created if missing",
        res.ok and (nested / "out.mrcs").exists(),
        res.error_line or "created",
    )

    spaced = ROOT / "a directory with spaces"
    res = run_cli(
        "spaces", [*BASE, "--output_dir", str(spaced), "--filename", "out"], spaced
    )
    record(
        "spaces in output_dir",
        res.ok and (spaced / "out.mrcs").exists(),
        res.error_line or "ok",
    )

    # A second run must overwrite cleanly rather than append or half-write.
    rerun = ROOT / "rerun"
    # threads=1: multithreaded ice accumulation is not bit-reproducible, so a
    # hash comparison needs the single-threaded path to mean anything.
    first = run_cli(
        "rerun1",
        [*BASE, "--output_dir", str(rerun), "--filename", "out"],
        rerun,
        threads=1,
    )
    second = run_cli(
        "rerun2",
        [*BASE, "--output_dir", str(rerun), "--filename", "out"],
        rerun,
        threads=1,
    )
    same = first.fingerprint.get("out.mrcs", {}).get("hash") == second.fingerprint.get(
        "out.mrcs", {}
    ).get("hash")
    record(
        "re-run overwrites in place",
        second.ok and same,
        second.error_line or "identical output, no append",
    )

    # A relative output_dir must resolve against the cwd, not the repo root.
    cwd = ROOT / "cwd_test"
    (cwd / "sub").mkdir(parents=True, exist_ok=True)
    res = run_cli(
        "relative",
        [*BASE, "--output_dir", "sub", "--filename", "out"],
        cwd / "sub",
        cwd=cwd,
    )
    record(
        "relative output_dir uses cwd",
        res.ok and (cwd / "sub" / "out.mrcs").exists(),
        res.error_line or "resolved against cwd",
    )


def check_datasets() -> None:
    """--cs_path / --star_path through the CLI (the pipeline has tests; the
    flags do not)."""
    print("\n=== dataset-driven ===")
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from test_pipelines import _write_minimal_csfile, _write_minimal_starfile
    except Exception as exc:  # pragma: no cover
        record("cs/star fixtures", False, f"could not import builders: {exc}")
        return

    fixtures = ROOT / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for flag, builder, name in [
        ("--cs_path", _write_minimal_csfile, "minimal.cs"),
        ("--star_path", _write_minimal_starfile, "minimal.star"),
    ]:
        path = fixtures / name
        builder(path, 2)
        out = ROOT / f"dataset_{flag.strip('-')}"
        res = run_cli(
            flag,
            [
                "simulate",
                "particles",
                "--pdb_code",
                PDB_A,
                "--n_pixels",
                "48",
                "--n_particles",
                "2",
                "--batchsize",
                "2",
                "--device",
                "cpu",
                flag,
                str(path),
                "--output_dir",
                str(out),
                "--filename",
                "out",
            ],
            out,
        )
        record(f"{flag} via CLI", res.ok, res.error_line or f"{res.wall_s}s")


def check_assembly() -> None:
    """
    --assembly, which needs a fetch to test at all.

    Every locally cached structure has an identical asymmetric unit and
    biological assembly, so offline this flag cannot be distinguished from a
    no-op. 1BXN (RuBisCO) is a large multimer whose assembly genuinely expands
    its AU.
    """
    print("\n=== --assembly (network) ===")
    cache = ROOT / "pdb_cache"
    shapes = {}
    for value in ("true", "false"):
        out = ROOT / f"assembly_{value}"
        res = run_cli(
            value,
            [
                "simulate",
                "particles",
                "--pdb_code",
                "1BXN",
                "--assembly",
                value,
                "--n_particles",
                "1",
                "--n_pixels",
                "48",
                "--pixel_size",
                "4.0",
                "--batchsize",
                "1",
                "--device",
                "cpu",
                "--pdb_savefolder",
                str(cache),
                "--output_dir",
                str(out),
                "--filename",
                "out",
            ],
            out,
            timeout=1200,
        )
        if not res.ok:
            record(f"--assembly {value}", False, res.error_line)
            return
        shapes[value] = res.fingerprint.get("out.mrcs", {}).get("hash")
    record(
        "--assembly changes the structure used",
        shapes.get("true") != shapes.get("false"),
        f"assembly={shapes.get('true')} vs AU={shapes.get('false')}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-network", action="store_true")
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    n_gpus = gpu_count()
    print(f"repo: {REPO_ROOT}\nGPUs visible: {n_gpus}")

    check_devices(n_gpus)
    check_paths()
    check_datasets()
    if not args.skip_network:
        check_assembly()

    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\nsummary: {len(results) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
