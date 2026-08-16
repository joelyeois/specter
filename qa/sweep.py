"""Phase 1 sweep driver: does every CLI flag actually do something?

Usage
-----
    python dev/cli_qa/sweep.py particles [--workers 12] [--only ice_model]
    python dev/cli_qa/sweep.py all

For each command it runs the baseline twice (a determinism check -- if two
identical seeded invocations disagree, every later comparison is noise), then
one run per flag perturbation, and classifies each against its expectation.
Results land in dev/cli_qa/results/<key>.json plus a markdown report.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import REPO_ROOT, RunResult, run_cli, save_results  # noqa: E402
from spec import SPECS, CommandSpec, Flag  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Every run writes a full particle stack / micrograph / tomogram, so the working
# tree is not the place for it. $SPECTER_QA_WORKDIR overrides; otherwise a
# temp directory, which the OS clears on its own schedule.
WORK_DIR = Path(
    os.environ.get("SPECTER_QA_WORKDIR", "")
    or Path(tempfile.gettempdir()) / "specter-qa-runs"
)


# Two identical seeded runs are not bit-identical: the ice insertion
# accumulates atom contributions in a thread-order-dependent way, which moves
# pixels by ~2e-4 of a standard deviation (single-threaded runs *are* bit
# exact). Anything below this is that noise floor, not an effect of the flag.
CHANGE_TOL = 5e-3


def _data_files(fp: dict) -> dict:
    """Just the pixel-data artifacts, keyed by name."""
    return {k: v for k, v in fp.items() if v.get("kind") == "mrc"}


def _rel_diff(base: RunResult, run: RunResult, rel: str) -> float:
    """
    Max |a - b| between two runs' copies of one artifact, in units of its std.

    Returns `inf` when the arrays are not even the same shape -- that is a
    change by any definition.
    """
    import mrcfile
    import numpy as np

    a_path, b_path = Path(base.out_dir) / rel, Path(run.out_dir) / rel
    with mrcfile.open(str(a_path), permissive=True) as m:
        a = np.asarray(m.data, dtype=np.float64)
    with mrcfile.open(str(b_path), permissive=True) as m:
        b = np.asarray(m.data, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    scale = max(float(a.std()), 1e-30)
    return float(np.abs(a - b).max() / scale)


@dataclass
class Diff:
    """Baseline-vs-run comparison, or a terminal failure that preempts it."""

    terminal: str | None  # ERROR / NO-OUTPUT / NAN, else None
    detail: str
    changed: list[str] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)
    any_change: bool = False


def compare(base: RunResult, run: RunResult) -> Diff:
    """Diff one run's artifacts against the baseline's."""
    if not run.ok:
        return Diff("ERROR", f"rc={run.returncode}: {run.error_line}")

    base_data, run_data = _data_files(base.fingerprint), _data_files(run.fingerprint)
    if not run_data:
        return Diff("NO-OUTPUT", "run produced no .mrc/.mrcs output")

    bad = [k for k, v in run_data.items() if v.get("nan") or v.get("inf")]
    if bad:
        return Diff("NAN", f"non-finite values in {bad}")

    new_files = sorted(set(run.fingerprint) - set(base.fingerprint))
    # A flag that switches an output *off* (write_picks, write_segmentation)
    # changes the artifact set just as much as one that adds files.
    gone_files = sorted(set(base.fingerprint) - set(run.fingerprint))
    shared = sorted(set(base_data) & set(run_data))
    diffs = {k: _rel_diff(base, run, k) for k in shared}
    changed = [k for k, d in diffs.items() if d > CHANGE_TOL]
    any_change = bool(changed or new_files or set(run_data) != set(base_data))
    worst = max(diffs.values(), default=0.0)
    return Diff(
        None,
        f"max_rel_diff={worst:.2e} changed={changed or '[]'} "
        f"new={new_files or '[]'} gone={gone_files or '[]'}",
        changed,
        new_files + gone_files,
        any_change or bool(gone_files),
    )


def classify(flag: Flag, base: RunResult, run: RunResult) -> tuple[str, str]:
    """Apply `flag.expect` to the baseline-vs-run comparison."""
    diff = compare(base, run)
    if diff.terminal:
        return diff.terminal, diff.detail

    changed, new_files, any_change = diff.changed, diff.new_files, diff.any_change
    detail = diff.detail

    if flag.expect == "changes":
        return ("PASS" if any_change else "NO-OP", detail)
    if flag.expect == "unchanged":
        return ("UNEXPECTED-CHANGE" if changed else "PASS", detail)
    if flag.expect == "artifacts":
        return ("PASS" if new_files else "NO-OP", detail)
    if flag.expect == "metadata":
        star_changed = any(
            base.fingerprint.get(k, {}).get("hash") != v.get("hash")
            for k, v in run.fingerprint.items()
            if v.get("kind") == "star"
        )
        return ("PASS" if star_changed or any_change else "NO-OP", detail)
    return ("PASS", detail)


def coverage_report(spec: CommandSpec) -> list[str]:
    """Flags on the real command that the spec never mentions."""
    import click

    from specter.cli._cli import cli

    cmd = cli.commands[spec.argv[0]].commands[spec.argv[1]]
    actual = {p.name.lower() for p in cmd.params if isinstance(p, click.Option)}
    covered = {f.name.lower() for f in spec.flags}
    from spec import NOT_USER_FACING

    return sorted(actual - covered - {n.lower() for n in NOT_USER_FACING})


def resolve_values(spec: CommandSpec) -> list[str]:
    """
    Force every perturbation to actually differ from the effective config.

    A hand-written perturbation that happens to equal what the TOML already
    sets produces an identical run, which then reads as a NO-OP finding when
    nothing was ever perturbed in the first place. Booleans are flipped
    automatically; anything else is skipped and reported, so the spec can be
    corrected rather than quietly yielding a false negative.
    """
    from specter.config import load_config

    if spec.config_cls is None:
        return []
    cfg = load_config(spec.config_path, spec.config_cls)
    notes = []
    for flag in spec.flags:
        if flag.expect == "skip" or flag.value is None:
            continue
        current = getattr(cfg, flag.name, None)
        if current is None or str(current).lower() != str(flag.value).lower():
            continue
        if isinstance(current, bool):
            flag.value = "false" if current else "true"
            notes.append(
                f"--{flag.name}: equalled config ({current}), flipped to {flag.value}"
            )
        else:
            flag.expect = "skip"
            notes.append(
                f"--{flag.name}: perturbation equals config value {current!r}, SKIPPED"
            )
    return notes


def sweep(spec: CommandSpec, workers: int, only: str | None = None) -> list[RunResult]:
    """Run the baseline (twice) plus every non-skipped flag perturbation."""
    root = WORK_DIR / spec.key
    root.mkdir(parents=True, exist_ok=True)

    def _run(name: str, extra: list[str]) -> RunResult:
        out = root / name
        argv = [
            *spec.argv,
            *spec.baseline,
            *extra,
            "--output_dir",
            str(out),
            "--filename",
            "out",
        ]
        # One thread per run: the ice accumulation is thread-order dependent,
        # so multi-threaded runs are not bit-reproducible and small genuine
        # flag effects would be indistinguishable from that jitter. Parallelism
        # comes from running many single-threaded jobs at once instead.
        return run_cli(name, argv, out, timeout=spec.timeout, threads=1)

    for note in resolve_values(spec):
        print(f"  spec fix: {note}")
    todo = [f for f in spec.flags if f.expect != "skip"]
    if only:
        todo = [f for f in todo if f.name == only]

    # A flag that needs `context` to be meaningful must be judged against a run
    # with that context and nothing else -- otherwise the context's own effect
    # (crowding on vs off, say) is what gets measured, not the flag's.
    contexts = sorted({tuple(f.context) for f in todo if f.context})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        base_a, base_b = pool.map(lambda n: _run(n, []), ["baseline_a", "baseline_b"])
        ctx_futures = {
            ctx: pool.submit(
                _run,
                "ctx_" + "_".join(ctx).replace("-", "").replace("/", "_"),
                list(ctx),
            )
            for ctx in contexts
        }
        futures = {
            pool.submit(_run, f"{f.name}={f.value}".replace("/", "_"), f.argv()): f
            for f in todo
        }
        ctx_base = {ctx: fut.result() for ctx, fut in ctx_futures.items()}
        runs = [(futures[fut], fut.result()) for fut in futures]

    def reference(flag: Flag) -> RunResult:
        return ctx_base[tuple(flag.context)] if flag.context else base_a

    results = [base_a, base_b, *ctx_base.values()]
    print(f"\n=== {spec.key} ===")
    if not base_a.ok:
        print(f"  BASELINE FAILED rc={base_a.returncode}: {base_a.error_line}")
        return results + [r for _, r in runs]
    for ctx, ref in ctx_base.items():
        if not ref.ok:
            print(f"  CONTEXT RUN FAILED {' '.join(ctx)}: {ref.error_line}")

    det = compare(base_a, base_b)
    if det.terminal or det.any_change:
        print(f"  !! NON-DETERMINISTIC baseline under a fixed seed: {det.detail}")
    else:
        print(f"  baseline deterministic ({base_a.wall_s}s)")

    for flag, run in sorted(runs, key=lambda x: x[0].name):
        status, detail = classify(flag, reference(flag), run)
        results.append(run)
        if status != "PASS":
            print(f"  [{status:18s}] --{flag.name} {flag.value}  {detail[:110]}")
    n_pass = sum(1 for f, r in runs if classify(f, reference(f), r)[0] == "PASS")
    print(f"  {n_pass}/{len(runs)} flags behaved as expected")

    missing = coverage_report(spec)
    if missing:
        print(f"  spec does not cover: {missing}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=[*SPECS, "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    keys = list(SPECS) if args.command == "all" else [args.command]
    for key in keys:
        results = sweep(SPECS[key], args.workers, args.only)
        save_results(results, RESULTS_DIR / f"{key}.json")
        print(f"  -> {RESULTS_DIR / f'{key}.json'}")


if __name__ == "__main__":
    print(f"repo: {REPO_ROOT}")
    main()
