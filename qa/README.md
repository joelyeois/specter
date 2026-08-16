# CLI QA sweeps

Pre-release quality checks for the `specter` command line. Every check runs the
real CLI in a real subprocess, so what gets tested is exactly what a user types
— no in-process shortcuts that could bypass argument parsing or config loading.

Kept here rather than in `dev/` (gitignored scratch) for the same reason
`docs-figures/` is: it is tooling that has to stay reachable for anyone
reproducing a result, not a personal prototype.

## Running

```bash
python qa/sweep.py particles              # one command
python qa/sweep.py all --workers 24       # everything
python qa/sweep.py particles --only dose  # one flag, while debugging
```

Results land in `qa/results/<command>.json` (gitignored — they are run
artifacts, and the fingerprints are machine-specific).

Runs are **single-threaded on purpose**: ice insertion accumulates atom
contributions in a thread-order-dependent way, so multi-threaded runs are not
bit-reproducible and a small genuine flag effect would be indistinguishable
from that jitter. Parallelism comes from running many single-threaded jobs at
once — hence `--workers`.

### Environment

| Variable | Meaning |
|---|---|
| `SPECTER_QA_WORKDIR` | Where runs write their output. Defaults to a temp directory; each run writes a full stack/micrograph/tomogram, so it should not be the working tree. |
| `SPECTER_QA_VOLUME` | Specimen volume for the tilt-series sweep. Defaults to whatever `sweep.py tomogram` last wrote. Run the tomogram sweep first, or point this at any `.mrc`. |

## What the sweep asks

Phase 1 asks one question per flag: **does setting this actually change the
output?** A flag that parses cleanly, lands in the config, and then changes
nothing is a silent no-op — the failure mode a user never notices. Each flag in
`spec.py` carries a perturbed value, any `context` needed to make it meaningful
(an ice flag means nothing with `--ice_model none`), and an expectation:

| `expect` | Meaning |
|---|---|
| `changes` | pixel data must differ from the reference |
| `unchanged` | pixel data must **not** differ — an invariance check for performance knobs, not a no-op check |
| `artifacts` | the set of output files must change |
| `metadata` | the `.star` may differ even if pixels do not |
| `skip` | excluded, with a `note` saying why |

## Files

| File | Role |
|---|---|
| `runner.py` | subprocess execution + artifact fingerprinting |
| `spec.py` | per-command baseline, per-flag perturbation/context/expectation |
| `sweep.py` | orchestration, comparison, classification, coverage check |
| `qa_tomogram.toml` | shrunk `build tomogram` config — seconds not minutes, cached PDBs only, so no sweep run hits the network |
| `FINDINGS.md` | what the sweeps have found so far |

## Gotchas worth knowing before editing `spec.py`

These were all learned by getting them wrong first, and are handled in code now:

- A flag needing `context` is judged against a run with **that context alone**,
  not the plain baseline — otherwise the context's own effect is what gets
  measured.
- Perturbations must differ from the **effective TOML value**, not the
  dataclass default. `resolve_values()` flips booleans that match and skips
  anything else that would be a no-change, rather than letting it read as a
  finding.
- Flags that switch an output *off* remove files; removals count as changes.
- A seeded particle run must pin `--batchsize`, since `"auto"` is sized to
  free device memory at run time.
