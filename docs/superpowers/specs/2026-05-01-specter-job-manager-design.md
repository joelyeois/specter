# SPECTER Job Manager — Design Spec
*2026-05-01*

## Problem

Running SPECTER simulations and reconstructions from notebooks and scripts produces
outputs scattered across manually-typed paths. There is no record of which parameters
produced which outputs, making it hard to compare runs or revisit past work.

## Goal

A lightweight, opt-in job management layer that:
- Auto-creates a unique output folder per run
- Records a complete parameter snapshot (all constructor args + defaults + git commit) alongside outputs
- Enables CLI browsing and diffing of past jobs
- Does **not** break any existing notebook or script workflow

Out of scope for this iteration: web UI, job submission from a browser, PBS/HPC dispatch,
multi-user support.

---

## Mental Model: Project → Job

```
~/specter-data/                    ← configurable base dir
  empiar-12391/                    ← project
    J001/                          ← job
      job.json
      hparams.yaml                 (written by Lightning)
      epoch_0100.mrc               (written by VolumeMonitorCallback)
      final_volume.mrc
    J002/
      job.json
      ...
  lr-scheduler-test/               ← another project
    J001/
    J002/
```

The workspace level (as in CryoSPARC) is deliberately omitted for now — a flat
Project → Job hierarchy is simpler and sufficient for a single-user setup.
It can be added later without breaking the schema.

Job IDs are scoped per-project: each project has its own J001, J002, ... sequence.

---

## `specter.jobs` Module

New module at `src/specter/jobs/` with four files:

```
src/specter/jobs/
  __init__.py      ← exports Job, JobDatabase
  _job.py          ← Job context manager
  _database.py     ← JobDatabase (reads/queries job folders)
  _cli.py          ← CLI entry point
```

Exported from top-level `specter` package alongside existing exports.

---

## `Job` Context Manager

```python
from specter.jobs import Job

with Job("ghostbuster", project="empiar-12391") as job:
    model = job.create(Ghostbuster, volume_init, orig_pixel_size, rotations, ...)
    job.log({"dose_rescale_factor": dose_per_area, "n_particles": num_particles})
    trainer.fit(model, train_loader)
```

### Constructor

```python
Job(
    job_type: str,
    project: str,
    base_dir: str | Path | None = None,
)
```

- `job_type`: free-form string label, e.g. `"ghostbuster"`, `"tilt-series"`
- `project`: groups related jobs into a folder
- `base_dir`: overrides the default base directory (see Configuration)

### `job.dir`

`Path` to the auto-created job folder. Available if the user needs to save outputs
manually via `job.save()`. For classes with a `run_dir` parameter, this is handled
automatically by `job.create()` — the user does not need to reference `job.dir` directly.

### `job.create(cls, *args, **kwargs) -> instance`

Factory that captures a complete parameter snapshot, sets the output directory, and
instantiates the class.

Internally:
1. Binds `args` and `kwargs` to the class `__init__` signature via
   `inspect.signature(cls.__init__).bind(None, *args, **kwargs).apply_defaults()`
   (the leading `None` stands for `self`)
2. If the class `__init__` has a `run_dir` parameter, injects `run_dir=job.dir`
   automatically, overriding any user-supplied value. The user never needs to
   mention `job.dir` or `run_dir` explicitly.
3. Serializes the bound arguments into `job.json` under `params`:
   - Scalars, strings, booleans, lists of scalars → stored as-is
   - `torch.Tensor` → `{"__type__": "Tensor", "shape": [Z, Y, X], "dtype": "float32"}`
   - `dict` of tensors (e.g. `ctf_params`) → each value summarised the same way
   - Any other non-JSON-serializable object → `{"__type__": "<classname>", "repr": str(obj)[:200]}`
4. Instantiates `cls(*args, **kwargs)` with the (possibly modified) arguments and returns it

### `job.log(params: dict)`

Merges additional key-value pairs into `job.json["params"]`. Use this for anything
that happens outside the class constructor — pre-processing steps, dataset paths,
number of particles used, etc.

```python
job.log({"dose_rescale_factor": dose_per_area, "n_particles": len(images)})
```

Values must be JSON-serializable (scalars, strings, lists). Call multiple times if
needed; each call merges into the existing params.

### `job.save(tensor, filename)`

Saves a `torch.Tensor` to `job.dir / filename`. Converts to numpy internally.
Uses `mrcfile` for `.mrc`/`.mrcs`, `torch.save` for `.pt`, otherwise raises.

### `job.save_figure(fig, filename)`

Saves a `matplotlib.Figure` to `job.dir / filename`.

### Context manager behaviour

- `__enter__`: assigns next available job ID, creates the folder, writes initial `job.json`
  with `status: running`
- `__exit__` (no exception): updates `job.json` with `status: complete` and
  `completed_at` timestamp
- `__exit__` (exception raised): updates `job.json` with `status: failed` and
  `error` field containing the exception message; re-raises the exception

---

## `job.json` Schema

```json
{
  "id": "J001",
  "type": "ghostbuster",
  "project": "empiar-12391",
  "status": "complete",
  "created_at": "2026-05-01T14:32:00",
  "completed_at": "2026-05-01T16:45:12",
  "specter_version": "0.1.0",
  "specter_commit": "554229b",
  "params": {
    "voxel_size": 1.06,
    "energy": 300.0,
    "dose_per_angstrom": 40.0,
    "lr": 0.1,
    "lr_R": null,
    "lr_T": null,
    "lr_D": null,
    "lr_decay": 0.1,
    "scheduler": "LambdaLR",
    "scattering_model": "rytov",
    "aberration_model": "holography",
    "symmetry": "I1",
    "sparsity": 0,
    "V": {"__type__": "Tensor", "shape": [224, 224, 224], "dtype": "float32"},
    "quaternions": {"__type__": "Tensor", "shape": [5832, 4], "dtype": "float32"},
    "translations": {"__type__": "Tensor", "shape": [5832, 2], "dtype": "float32"},
    "ctf_params": {
      "dfu": {"__type__": "Tensor", "shape": [5832], "dtype": "float32"},
      "dfv": {"__type__": "Tensor", "shape": [5832], "dtype": "float32"}
    },
    "dose_rescale_factor": 44.944,
    "n_particles": 5832
  }
}
```

`specter_commit` is read from `git rev-parse --short HEAD` at job creation time;
falls back to `"unknown"` if git is unavailable.

### Schema evolution

The schema is append-only and unversioned at the field level. Old jobs simply lack
fields added in the future (e.g. if a new constructor argument is added to Ghostbuster
tomorrow, old jobs won't have it — which is correct, since those jobs didn't use it).
The `specter_commit` field pinpoints the exact code version that produced any job.

---

## `JobDatabase`

```python
from specter.jobs import JobDatabase

db = JobDatabase()                          # reads from default base_dir
db = JobDatabase(base_dir="/my/custom/dir")

db.list(project="empiar-12391")             # → list[dict]
db.get("empiar-12391", "J001")              # → dict (job.json contents)
db.diff("empiar-12391", "J001", "J002")     # → dict of changed keys only
```

`diff` compares the `params` dicts of two jobs, returning only keys where
the values differ. Keys present in one job but not the other are included
with `None` as the missing side's value. Tensor summary dicts are compared
by shape only.

---

## Configuration

Base directory is resolved in this order:
1. `base_dir` argument passed to `Job(...)` or `JobDatabase(...)`
2. Environment variable `SPECTER_JOBS_DIR`
3. Default: `~/specter-data/`

No config file needed for now.

---

## CLI

Registered as a console script `specter-jobs` in `pyproject.toml`:

```bash
specter-jobs list
specter-jobs list --project empiar-12391
specter-jobs show empiar-12391 J001
specter-jobs diff empiar-12391 J001 J002
```

`list` output (rendered with `rich`, which is already a dependency):

```
 Project          ID    Type          Status    Created               Params
 empiar-12391     J001  ghostbuster   complete  2026-04-30 14:32      lr=0.1 sym=I1
 empiar-12391     J002  ghostbuster   complete  2026-05-01 09:10      lr=0.05 sym=I1
```

`diff` output:

```
 Key   J001    J002
 lr    0.1  →  0.05
```

`list` shows a short summary of scalar params only (tensors omitted). `show` renders
the full `job.json` including tensor summaries.

---

## Integration With Existing Notebooks

**No existing code changes.** The `Job` wrapper is opt-in. Existing notebooks
that pass a literal `run_dir` string continue to work identically.

The only migration needed when adopting the job manager for a notebook:

```python
# before
model = Ghostbuster(volume_init, orig_pixel_size, rotations, ...,
                    run_dir="/scratch/loh/joel/my-run/")

# after — run_dir is handled automatically, user never specifies it
with Job("ghostbuster", project="empiar-12391") as job:
    model = job.create(Ghostbuster, volume_init, orig_pixel_size, rotations, ...)
    job.log({"dose_rescale_factor": dose_per_area, "n_particles": len(images)})
    trainer.fit(model, train_loader)
```

---

## Testing

- `test_job_creates_folder`: verify `job.dir` is created on `__enter__`
- `test_job_writes_json`: verify `job.json` contains expected fields
- `test_job_status_complete`: verify status is `complete` after clean exit
- `test_job_status_failed`: verify status is `failed` and error is recorded after exception
- `test_job_id_sequence`: verify J001, J002, J003 assigned in order
- `test_job_create_captures_defaults`: verify `job.create()` captures args not explicitly passed
- `test_job_create_injects_run_dir`: verify `run_dir` is set to `job.dir` automatically when the class accepts it
- `test_job_create_tensor_summary`: verify tensors are stored as shape/dtype dicts
- `test_job_log_merges`: verify repeated `job.log()` calls accumulate params
- `test_database_list`: verify `JobDatabase.list()` returns all jobs
- `test_database_diff`: verify diff returns only changed keys
- `test_cli_list`: smoke test CLI list command

---

## Files Changed / Created

| Path | Action |
|---|---|
| `src/specter/jobs/__init__.py` | create |
| `src/specter/jobs/_job.py` | create |
| `src/specter/jobs/_database.py` | create |
| `src/specter/jobs/_cli.py` | create |
| `src/specter/__init__.py` | add `from .jobs import Job, JobDatabase` |
| `pyproject.toml` | add `specter-jobs` console script entry point |
| `tests/test_jobs.py` | create |
