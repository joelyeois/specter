# Manage jobs

A job is a numbered, self-describing output folder: a directory named
`J001`, `J002`, and so on, holding one run's output files alongside a
`job.json` that records every parameter that produced them, the git
commit `specter` was running at, and (for reconstruction) the loss and
resolution history. It works like a RELION job tree or a CryoSPARC
workspace: a numbered sequence of runs under a project, so you can see
what ran and with what settings without digging through shell history.

`specter.jobs` is the package behind it (`Job`, `JobDatabase`, and the
`base_directory`/`SPECTER_JOBS_DIR` session helpers; see
[`specter.jobs`](../api/jobs.md) for the full Python API if you're
scripting against it directly rather than through the CLI). This page
covers the mechanics every `specter simulate`/`build`/`reconstruct`
command shares: when a run gets tracked, where its files land, what
`job.json` contains, and the `specter jobs list`/`show`/`diff` commands
you use to inspect and compare runs afterwards.

## Which commands track runs

Tracking is opt-in for every forward-simulation command, and mandatory for
reconstruction:

| Command | Job-type folder | Tracked by default? |
|---|---|---|
| `specter simulate particles` | `particles` | No -- set `--project` or `--job_id` |
| `specter simulate micrograph` | `micrographs` | No -- set `--project` or `--job_id` |
| `specter simulate tiltseries` | `tiltseries` | No -- set `--project` or `--job_id` |
| `specter build tomogram` | `tomograms` | No -- set `--project` or `--job_id` |
| `specter build ice` | -- | Never -- no `--project`/`--job_id` flags exist |
| `specter reconstruct particle` (`specter ghostbuster particle`) | `reconstructions` | **Always** |

`specter.pipelines._common.is_tracked` makes this decision for the
opt-in commands: a config is tracked if `project` or `job_id` is set,
and otherwise gets a flat, untracked write to `output_dir`.
Reconstruction has no untracked mode (neither RELION nor CryoSPARC has
one), so `specter reconstruct particle` always opens a job, with or
without `--project`.

`specter build ice` writes a replacement `IceBank` library rather than a
single dated run, so nothing about it fits the one-run/one-job model. It
has no `--project` or `--job_id` flag at all, and `specter jobs` doesn't
see its output.

## Turning tracking on

For the opt-in commands, either flag is sufficient on its own:

```bash
specter simulate particles --config configs/particle.toml --project apoferritin
```

```bash
specter simulate particles --config configs/particle.toml --job_id J001
```

`--project` groups related jobs under a shared folder and names it.
`--job_id` pins the specific job directory a run writes into, instead of
letting SPECTER auto-assign the next free `J0NN`. Passing only
`--job_id` still tracks the run: it lands directly under
`output_dir/<job_type>/<job_id>/`, in the implicit default project (see
[Directory layout](#directory-layout)). Passing both scopes the pinned id
to the named project: `output_dir/<project>/<job_type>/<job_id>/`.

Pinning `--job_id` is also how you resume a run: if that directory
already holds a `job.json`, the new invocation reads it back in and
merges into it instead of starting fresh. This is the mechanism behind
reconstructing gold-standard halfsets A and B as two separate command
invocations sharing one `job_id`; see [Reconstruct a
volume](reconstruction.md#reconstructing-halves-separately) for that
worked example, and [Resuming into an existing
job](#resuming-into-an-existing-job) below for what gets validated when
it happens.

## Directory layout

A tracked run's output lands at:

```
output_dir/[project/]job_type/J00N/
```

`project` is optional. When omitted, it drops from the path, but the run
stays tracked. It functions as the implicit default project for whatever
`output_dir` resolves to, useful when one directory holds only one
project's worth of jobs and a name would be redundant. `--project` is
for the opposite case: splitting one shared `output_dir` (one scratch
directory used across several unrelated structures, for instance) into
several named projects.

Numbering is a single continuous sequence per project, shared across every
job type within it, not restarted per job type. A project whose first run
is a reconstruction gets `J001` there; a particle stack simulated into the
same project afterwards gets `J002`, not a second `J001` under
`particles/`. `specter.jobs._job._next_job_id` enforces this by scanning
every job-type subfolder under the project directory and taking the
highest existing number, rather than scanning per job type.

### Resolving `output_dir`

There is exactly one output-path field on every config, `output_dir`, and
what it means depends on whether the run is tracked:

| `output_dir` | Untracked | Tracked |
|---|---|---|
| Unset (`None`) | `<job_type>/` | Project root found by walking up from cwd for an existing `.specter` marker |
| Set (TOML or `--output_dir`) | Used verbatim, as the leaf directory | Used verbatim, as the root of the `[project/]job_type/J00N/` tree |

Untracked, `output_dir` is the flat folder files land in directly.
Tracked, it is the root the numbered job tree grows under: turning on
`--project`/`--job_id` organises output *within* whatever directory you
already chose rather than relocating it. The unset default differs
between the two rows because the tracked layout supplies its own
`job_type` segment. Defaulting both cases to `<job_type>/` would produce
`tomograms/tomograms/J001`.

One consequence of this is easy to get backwards from the table alone:
**the canonical example configs under `configs/`** (`particle.toml`, `micrograph.toml`, `tilt_series.toml`,
`tomogram.toml`) **set `output_dir` explicitly** (`output_dir = "particles"`,
and so on), so those fields stay set for those configs even when you add
`--project` on the command line. Running

```bash
specter simulate particles --config configs/particle.toml --project apoferritin
```

writes to `particles/apoferritin/particles/J001/`, using the TOML's
literal `output_dir = "particles"` as the tracked root. It does **not**
walk up for a `.specter` marker, because `output_dir` was never unset to
begin with. To get the `.specter`-discovered project root instead, clear
`output_dir` in the config (comment out the line, or pass
`--output_dir ""` at the command line) before adding `--project`.
`configs/reconstruct.toml` ships with `output_dir` commented out for
exactly this reason: reconstruction is always tracked, so its canonical
config leaves the field unset and lets `.specter` discovery pick the root,
instead of hardcoding a path the way the untracked-by-default configs do.

## The `.specter` project marker

`.specter` is an empty marker file naming the root of a project, the same
role `.git` plays for a repository. Nothing reads its contents: its
presence at a given directory is the entire signal. Two functions resolve
it:

- **`find_specter_project_root()`** walks up from the current directory
  looking for `.specter`, the way `git` resolves the nearest ancestor
  containing `.git`. It never creates anything, so a non-main process (a
  DDP worker rank) can call it to agree on the same path as its siblings
  without racing anyone to create the marker.
- **`ensure_project_root()`** is the find-or-create counterpart, and the
  only one of the two with side effects. Whichever process owns a tracked
  run calls it once; it creates `.specter` at the discovered (or current)
  directory if none exists yet.

Creating the marker asks first at an interactive terminal (`Start a new
project here?`); otherwise it creates the marker without asking and
prints a notice instead. A blocking prompt would hang a batch job or CI
run with no way out, which is why `--do_projdir` exists as RELION's
non-interactive escape from the same dialog.

Because the search walks upward, running a tracked command from a
subdirectory of an already-initialised project still lands in that
project rather than starting a second, disconnected job tree with its own
`J001`. Given `.specter` at `/data/apoferritin/` and a job already at
`/data/apoferritin/particles/apoferritin-demo/particles/J001/`, running the
same tracked command from `/data/apoferritin/notebooks/` creates `J002` in
the same project, not a new `J001` under `notebooks/`.

## `job.json`

Every job directory holds exactly one `job.json`. A representative example,
trimmed to the fields that matter regardless of job type:

```json
{
  "id": "J002",
  "type": "particles",
  "project": "apoferritin-demo",
  "status": "complete",
  "created_at": "2026-08-24T20:46:33.400665+00:00",
  "specter_version": "0.1.0",
  "specter_commit": "a1b2c3d",
  "params": {
    "pdb_source": "6bdf",
    "n_particles": 4,
    "defocus": 8000,
    "scattering_model": "projection",
    "...": "every other config field, under the same names --config's TOML uses"
  }
}
```

- **`status`** is `"running"` while the job is in progress, `"complete"` on
  a clean exit, or `"failed"` (with an `"error"` field holding the
  exception message) if the process raised. `Job` writes it on entry and
  updates it on every subsequent write, so a `job.json` left at
  `"running"` after the process has exited means the run got killed
  rather than finishing cleanly.
- **`specter_commit`** is `git rev-parse --short HEAD` at run time (or
  `"unknown"` outside a git checkout), so you can trace a result back to
  the exact code that produced it.
- **`params`** holds every config field for the pipeline commands, or
  every constructor argument `Ghostbuster`/`Reconstructor` were built with
  for reconstruction (plus anything logged after construction, such as
  `Reconstructor.results_summary()`'s loss history and per-epoch
  resolutions; see [Reconstruct a volume](reconstruction.md) for what
  reconstruction specifically logs there). `specter` converts a value
  that isn't natively JSON-serializable: a 0-dimensional tensor (a scalar
  hyperparameter, e.g. a fitted defocus offset) becomes its plain number,
  a larger tensor becomes a `{"shape": ..., "dtype": ...}` summary rather
  than a dump of its contents, and anything else unrecognised becomes a
  truncated `repr()`.

`params` is the field `specter jobs diff` compares and `specter jobs list
--show` reads from; `id`/`type`/`project`/`status`/`created_at` are what
`specter jobs list`'s table columns come from.

### Resuming into an existing job

Pass a `--job_id` that already has a `job.json` on disk and the run
resumes instead of overwriting: `specter` merges incoming parameters into
what's already recorded (a plain `dict.update`, so a key present in both
keeps the new value). For the constructor-level parameters that produced
the run in the first place, resuming does something stricter than a
merge: it validates that every parameter this invocation would record
already matches what's stored, and raises `ValueError` naming every
mismatched key (stored value next to incoming value) if not. A second
`--halfset B` call sharing a `--job_id` with an earlier `--halfset A`
call must use the same defocus, the same PDB source, the same scattering
model, and so on, differing only in which half of the particle stack it
reconstructs. This is what makes gold-standard reconstruction across two
separate process invocations safe: a config typo on the second call
raises a clear error instead of producing an inconsistent halfmap pair
with no warning.

Resuming only checks the keys a given constructor call introspects. A
key present in `job.json` from an earlier `job.log(...)`
call (e.g. a result computed after the model finished fitting) stays
untouched, since the resuming call's constructor never claimed to agree
or disagree with it in the first place.

## `specter jobs list`

Lists every job under a base directory, one row per job, newest last:

```bash
specter jobs list --base-dir particles --project apoferritin-demo
```

```text
┏━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Project          ┃ ID   ┃ Type      ┃ Status   ┃ Created          ┃ Params                                                      ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ apoferritin-demo │ J001 │ particles │ complete │ 2026-08-24 20:44 │ pdb_source=6bdf  assembly=True  n_pixels=32  pixel_size=1.0 │
│ apoferritin-demo │ J002 │ particles │ complete │ 2026-08-24 20:44 │ pdb_source=6bdf  assembly=True  n_pixels=32  pixel_size=1.0 │
└──────────────────┴──────┴───────────┴──────────┴──────────────────┴─────────────────────────────────────────────────────────────┘
```

`Status` is colour-coded (green `complete`, yellow `running`, red
`failed`) in a terminal that supports it. Omit `--project` and the
command lists every job under `--base-dir`, across every project.

The default `Params` column shows the first four *scalar* params in
whatever order they were logged, not necessarily the ones that
distinguish one run from another. Pass `--show` with the exact keys to
display instead:

```bash
specter jobs list --base-dir particles --show n_particles,defocus
```

```text
┏━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Project          ┃ ID   ┃ Type      ┃ Status   ┃ Created          ┃ Params                                   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ apoferritin-demo │ J001 │ particles │ complete │ 2026-08-24 20:44 │ n_particles=2  defocus=[5000.0, 15000.0] │
│ apoferritin-demo │ J002 │ particles │ complete │ 2026-08-24 20:44 │ n_particles=4  defocus=8000              │
└──────────────────┴──────┴───────────┴──────────┴──────────────────┴──────────────────────────────────────────┘
```

A key `--show` names that is missing from a given job's `params` (e.g. a
key specific to a different job type, or a typo) prints as `key=-`
instead of dropping out, so the gap stays visible rather than leaving an
unexplained short row. This is how you compare many runs at once: pass
every field worth comparing across a batch of jobs (`--show
resolution_gold_standard,lr,epochs`) and read them off one table, instead
of opening each job's directory in turn.

### Locating `--base-dir`

`specter jobs` resolves its base directory independently of `.specter`.
It does **not** walk up from the current directory looking for a project
marker the way a tracked *write* does. Pass `--base-dir` explicitly, set
`SPECTER_JOBS_DIR` in the environment, or (from Python) call
`jobs.base_directory(...)` beforehand; without one of these, it raises
rather than guessing:

```text
RuntimeError: No job output directory configured. Do one of:
  1. jobs.base_directory('/your/data/path')  # recommended in notebooks
  2. export SPECTER_JOBS_DIR=/your/data/path  # in your shell / PBS script
  3. Job('type', 'project', base_dir='/your/data/path')  # per-job override
```

Pass whatever `output_dir` resolved to for the run(s) you're inspecting:
the tracked root from [Resolving `output_dir`](#resolving-output_dir),
not the job directory itself and not necessarily the project root. For a
project whose `output_dir` was left unset (the `.specter`-discovered
default), that root is the project root, and `export
SPECTER_JOBS_DIR=$(pwd)` from anywhere under it works for every
subsequent `specter jobs` call. For a project run against the shipped
`configs/particle.toml`, whose `output_dir = "particles"` SPECTER uses
verbatim even when tracked (see above), `--base-dir particles` is the
value that matches, not the project root.

## `specter jobs show`

Prints one job's complete `job.json`:

```bash
specter jobs show J002 --project apoferritin-demo --base-dir particles
```

```json
{
  "id": "J002",
  "type": "particles",
  "project": "apoferritin-demo",
  "status": "complete",
  "created_at": "2026-08-24T20:46:33.400665+00:00",
  "specter_version": "0.1.0",
  "specter_commit": "unknown",
  "params": {
    "pdb_source": "6bdf",
    "assembly": true,
    "n_pixels": 32,
    "n_particles": 4,
    "defocus": 8000,
    "scattering_model": "projection",
    "...": "..."
  }
}
```

`show` takes exactly one job id and prints its full record, params
included. Look here for a field `list`'s summary table doesn't show. It
has no multi-id form. To compare several jobs' params side by side, use
[`specter jobs list --show`](#specter-jobs-list) with a comma-separated
key list; to compare exactly two jobs field-by-field, use `specter jobs
diff`, below.

A job id that doesn't exist under the given project and base directory
fails with the same message `diff` and `get` share:

```text
No job 'J999' found under apoferritin-demo (searched every job-type subfolder)
```

## `specter jobs diff`

Compares two jobs' `params` dicts and prints only the keys that differ:

```bash
specter jobs diff J001 J002 --project apoferritin-demo --base-dir particles
```

```text
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
┃ Key         ┃ J001              ┃ J002 ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
│ defocus     │ [5000.0, 15000.0] │ 8000 │
│ n_particles │ 2                 │ 4    │
└─────────────┴───────────────────┴──────┘
```

`diff` omits keys where both jobs agree entirely: a config with dozens of
fields but one deliberate change (as here) prints a two-row table, not a
wall of duplicated agreement. A key present in one job's `params` but
absent from the other's (e.g. comparing a reconstruction against a
particle-stack job, or two reconstructions logged with different optional
fields) shows `None` on the missing side instead of getting skipped. Two
jobs with identical params print `Jobs are identical.` and nothing else.

## Worked example

Starting from an empty directory with no `.specter` marker:

```bash
specter simulate particles --config configs/particle.toml \
    --n_pixels 32 --n_particles 2 --device cpu --project apoferritin-demo
specter simulate particles --config configs/particle.toml \
    --n_pixels 32 --n_particles 4 --defocus 8000 --device cpu \
    --project apoferritin-demo
```

produces (since `configs/particle.toml` ships with `output_dir =
"particles"` set, per [Resolving `output_dir`](#resolving-output_dir)):

```text
particles/apoferritin-demo/particles/J001/{job.json,particles.mrcs,particles.star}
particles/apoferritin-demo/particles/J002/{job.json,particles.mrcs,particles.star}
```

From there, `specter jobs list --base-dir particles`, `specter jobs show
J002 --project apoferritin-demo --base-dir particles`, and `specter jobs
diff J001 J002 --project apoferritin-demo --base-dir particles` are the
three commands shown above, run against exactly this pair of jobs. The
second run's `--n_particles 4 --defocus 8000` overrides are what `diff`
reports as changed.

## See also

- [Reconstruct a volume](reconstruction.md): reconstruction's always-on
  tracking, the gold-standard two-halfset resume workflow, and what gets
  logged into `params` beyond the config fields (loss history, per-epoch
  resolutions).
- [Configure a run](configuration.md): the TOML/CLI override mechanics that
  produce the values recorded in `params`.
- [Generate a CryoSPARC dataset twin](dataset-twin.md): tracking a
  full-dataset run's parameters and provenance with `--project`.
- [`specter.jobs`](../api/jobs.md): the Python API (`Job`, `JobDatabase`,
  `base_directory`) behind every command on this page, for scripting job
  creation or inspection directly rather than through the CLI.
