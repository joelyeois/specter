# CLI QA — Phase 0/1 findings

Phase 1 asks one question per flag: **does setting it actually change the output?**
Every check runs the real `specter` CLI in a subprocess and fingerprints what it
wrote. Harness: `runner.py` (execution + fingerprinting), `spec.py` (what to
perturb), `sweep.py` (orchestration + classification).

Excluded as not-user-facing yet: the `torch_ctf` backend (no CLI flag exists at
all — Python API only) and fresh ice-cube generation (`GradientSKIcemaker` /
`build_ice_cache` have no CLI path). Also skipped: `ice_cache_dir`,
`ice_relax_steps`, `bulk_scattering_factors`, `shtyrov_params_path`, `conv_backend`,
`mmcif_filepath`.

## Result

| Command | Flags swept | Behaved as expected |
|---|---|---|
| `simulate particles` | 72 | 69 |
| `simulate micrograph` | 38 | 38 |
| `simulate tiltseries` | 31 | 31 |
| `build tomogram` | 20 | 13 → 18 after fixes |

## Bugs found and fixed

### 1. `specter build tomogram` crashed with the shipped config (critical)

`get_atom_species` returned a species list of the wrong length, surfacing as
`IndexError: shape of the mask [N] does not match the indexed tensor [M]` inside
`PotentialBuilder`. Since `scattering_factors = "shtyrov"` (then spelled
`potential_parameterization`) is the
default, affected structures could not be rendered at all — and
`filler_from_pei2016 = true` in `configs/tomogram.toml` pulls one in, so the
shipped default config failed outright.

**29 of 249 cached structures (12%) were affected**, via three independent root
causes, all in `pdb.py`:

| Cause | Detail | Example |
|---|---|---|
| Zero-occupancy atoms | `occ == 0.0` was the marker for "dummy H I added", but deposited structures carry real zero-occupancy atoms, which were then dropped | 1FA2 (208 atoms) |
| Missing insertion code | The atom key was `(chain, seqnum, atom_name)`; entries numbering residues `10, 10A, 10B…` collided and were discarded as duplicate conformers | 3DY4 (1828 atoms) |
| Hydrogen stripping | `HydrogenChange.ReAdd` removes every existing H before re-adding it from monomer-library geometry — without a library the strip happened and the re-add did not | 22FX (4968 atoms) |

Also fixed: `KeyError: ''` when the monomer library has no entry for a component.

After the fixes: **248/249 structures align**, element for element. The one
holdout (`7a4m-aligned.cif`) is a 24-model ensemble where Biopython walks every
model and gemmi types only the first; that now raises a named error explaining
the problem instead of misaligning silently.

### 2. `--deltaV_V` / `--deltaI_I` were silent no-ops

Click lowercases a parameter name derived from the flag, so `--deltaV_V` arrived
as `deltav_v`; `apply_overrides` did a blind `setattr`, attaching an attribute
nothing reads while `config.deltaV_V` kept its TOML value. The flag parsed, was
accepted, and did nothing (verified: exactly 0.0 difference before, 5.4 after).

Fixed in two places: the option now pins its own name, and `apply_overrides`
raises on any field it doesn't recognise, so this class of failure can't be
silent again.

### 3. `--n_particles 1` crashed

`random_quaternion(1)` squeezes the batch axis and returns `(4,)` instead of
`(1, 4)`, so the pipeline handed roma a length-1 vector: `IndexError` before
anything was written. `CrowdingSimulator` and `TomogramSpecimenGenerator` both
already guard against this on `random_rotation_matrix`; the particle pipeline
was the one that didn't.

### 4. `simulate micrograph` and `simulate tiltseries` had no seed at all

No config field, no flag, and neither pipeline ever called `specter.seed()` —
so neither command could produce reproducible output. Added `--seed` to both,
mirroring the particles pipeline (fixed seed used, else generated and logged).

## Not bugs, but worth knowing

- **`batchsize` changes results — now rejected rather than silently allowed.**
  Deterministic physics is bit-invariant under batching (verified: 0.0
  difference with ice and noise off), but stochastic stages consume the RNG
  stream in batch-sized chunks, so ice and noise draws shift. Since
  `batchsize="auto"` sizes to free GPU memory at run time, the same seed and
  config could give a different stack on a different machine.
  `run_particle_stack` now raises when a seed is set with `batchsize="auto"`,
  telling the user to pin an integer batch size. Pinned batching is
  reproducible; the deeper fix (deriving per-particle RNG from
  `(seed, particle_index)`) is deliberately not done.
- **Ice is not bit-reproducible multi-threaded.** Atom contributions accumulate
  in thread-order-dependent float order, moving pixels ~2e-4 of a standard
  deviation; single-threaded runs are bit-exact. Poisson sampling amplifies it
  to ~7e-2 on isolated pixels, since a rate change can flip a sampled count.
- **`--n_frames` does nothing unless `coincidence_radius > 0`** (which defaults
  to 0). Correct by design, undocumented in the flag help.
- **`--deltaV_V` / `--deltaI_I` are the only mixed-case flags** among 178. Easy
  to typo; click's suggestions soften it.
- **`configs/particle.toml` claimed `periodic` "forces potential_method='3d' if
  true".** It does not — `analytic` + `periodic` raises. Comment corrected.
  Note there are two unrelated `periodic` arguments: `PotentialBuilder`'s (the
  config/CLI flag, which nothing in the shipped pipeline sets) and the
  `arrays.py` splat helper's, which `ice/_bank.py` passes in its convolve path.
  Ice handles periodicity itself either way, so the "analytic can't do
  periodic" restriction is a `PotentialBuilder` limit, not a physics one.

## Phase 2 — bad input (`tools/cli-qa/garbage.py`)

Does a value that cannot mean anything fail *usefully*? Before: of 24 such
values on `simulate particles`, **9 ran to completion** and produced a
plausible-looking, meaningless stack (negative dose, negative voltage, alpha of
1.5, a defocus range given high-then-low), and **13 died ~10 s in** with
`ZeroDivisionError` / "tensor with negative dimension" / "cannot convert float
NaN to integer", naming nothing the user typed.

After `validate_config()`: **46 of 46 rejected across all four commands**, each
naming the field, its value and the requirement, within the process's own
~11 s import floor. It also closed a gap Phase 1 found — Click validates a
`--flag` against its `Choice`, but a TOML file bypasses that, and nothing
enforced a `Literal` at runtime.

## Phase 3 — the paths phases 1 and 2 skipped (`tools/cli-qa/phase3.py`)

10 of 10 pass: `--device cuda`/`cuda:0`/`0,1` (multi-GPU DDP writes all 8
particles), `batchsize="auto"` sizing itself against real GPU memory,
`output_dir` created when missing, spaces in paths, re-runs overwriting in
place, a relative `output_dir` resolving against the cwd, and
`--cs_path`/`--star_path` through the CLI.

`--assembly` is **verified** (network): 1BXN's biological assembly and its
asymmetric unit produce different structures. It cannot be tested offline —
every cached structure has AU == assembly.

## Resolved: the three findings Phase 1 left open

All three were artifacts of the Phase 1 baseline being too sparse, not no-ops:

- `membrane_min_transmembrane_spacing` binds once the membrane is crowded —
  at 60 transmembrane copies, spacing 40 vs 120 changes 28.6% of voxels. At
  the 4 copies Phase 1 used, the constraint never bound.
- `filler_table_min_mw_kda` 20 → 250 cuts the population from 27 species to
  19, and `filler_table_max_mw_kda` 100 leaves 9. Phase 1 had the reference
  tables switched off, so there was nothing for the cutoffs to cut.

## 2026-10-15 rerun, after the docs restructure and its fixes

The full Phase 1 sweep of all four commands, run after `potential_scale` was
removed, a tilt began keeping its FFT margin through the transfer function,
`Propagation.pad_fft` became a per-path default, and `blob/sign` began to be
honoured.

| Command | Full sweep | After the spec fixes below |
|---|---|---|
| `simulate particles` | 74/76 | 81/81 |
| `simulate micrograph` | 51/53 | 67/67 |
| `simulate tiltseries` | 35/37 | 41/41 |
| `build tomogram` | 17/21 | 21/21 |

Every flag of every command is now covered (`spec does not cover: []`).

**Harness fixes.** Two baselines could not start. `PDB_A`/`PDB_B` named
`1mbo.cif`, which a cache may hold only as `1mbo-assembly1.cif`, so the
particle and micrograph baselines were refused ("looks like a path, but no
such file"); `spec._cached` now takes whichever file the cache holds. With
`SPECTER_QA_WORKDIR` set, `QA_VOLUME` pointed at the work directory itself
rather than `tomogram/baseline_a/out.mrc` under it, so the tilt-series
baseline failed. `--only` now takes a comma-separated list.

**Spec fixes: contexts and perturbations that could not change anything.**

- `simulate micrograph --ice_profile meniscus` and `--ice_hole_radius 3000`
  had a context that already set both, so each run was identical to its
  reference by construction (since 2026-08-22). The meniscus now runs against
  a flat profile with the radius set, and the radius is perturbed to 1500.
- `simulate particles --potential_method 2d` and `--periodic true` were refused
  under the defaults (Shtyrov species are not supported by `2d`; `analytic`
  does not wrap). They run under `--scattering_factors kirkland` and
  `--potential_method 3d` respectively, the latter in a 48 A box: in the
  144 A baseline box the molecule's kernel never reaches a face to wrap.
- `simulate tiltseries --cc`/`--energy_spread`/`--temporal_coherence` move the
  6 A baseline's expected counts by ~1e-5 relative (measured 1.5e-6 for
  `--cc 2.0`), which no seeded Poisson count resolves and which sits under the
  sweep's 5e-3 tolerance; they are not no-ops. They run noiseless at 2 A voxels
  (`TILT_FINE`), `--cc` perturbed to 5.0 (2.0 gives 3.8e-3), the envelope at
  120 kV.
- `build tomogram --filler_table_{min,max}_mw_kda` act on the reference tables
  `qa_tomogram.toml` switches off; they run under `--filler_from_pei2016 true`.
  `--membrane_min_transmembrane_spacing` (binds at ~60 copies, a TOML-only
  count) and `--membrane_region_max_passes` (the gold-bead budget, which a few
  beads never exhaust) are skipped with that reason.
- New coverage: particles' absorption, `oversample`, `temporal_coherence`;
  the micrograph's `bfactor`, `klim`, `oversample`, `temporal_coherence`, the
  shape packer's knobs and the water-air interface bias; the tilt series'
  `bfactor`, `klim`, `oversample`; the tomogram's `packing_max_retries`
  (perturbed to 1, since 20 still placed every instance of the sparse
  baseline) and `packing_voxel_size`. The micrograph's `packing_max_retries`
  needs a crowded field (`--n_candidates 400`) to bind. Paths, job tracking
  and the Monomer Library flags are skipped with reasons.

**Fixed: `simulate micrograph --packing_seed` was ignored whenever `--seed`
was set.** `Field.draw` overwrote the crowd's `packing_seed` with the per-index
`derive_seed(seed, index, SEED_STAGE_SPECIMEN)` for the draw and restored it
after (`imaging/_field.py`, the `seed is not None` branch), so the flag did
nothing in any seeded run, every QA run included. A set value is now mixed
into the per-index seed (`imaging._stages.packer_seed`); unset, a seeded
packing is unchanged. 67/67 after the fix
(`tests/test_crowding.py::test_a_set_packing_seed_still_changes_a_seeded_packing`).

**Fixed, outside the sweep: `specter reconstruct particle` doubled its output
directory.** Run from the docs with no `--project` or `--job_id`, it wrote to
`reconstructions/reconstructions/J001` and created no `.specter` marker:
`resolve_output_dir` read the unset fields as an untracked run and returned the
`reconstructions/` default, under which `Job` grew its own
`reconstructions/J001`. Reconstruction tracks every run, so it now passes
`tracked=True`.

## Reproducing

```bash
python tools/cli-qa/sweep.py particles --workers 24
python tools/cli-qa/sweep.py all
```

Runs are single-threaded so the baseline is bit-reproducible; parallelism comes
from running many at once. Results land in `tools/cli-qa/results/<command>.json`.
