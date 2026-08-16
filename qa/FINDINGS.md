# CLI QA — Phase 0/1 findings

Phase 1 asks one question per flag: **does setting it actually change the output?**
Every check runs the real `specter` CLI in a subprocess and fingerprints what it
wrote. Harness: `runner.py` (execution + fingerprinting), `spec.py` (what to
perturb), `sweep.py` (orchestration + classification).

Excluded as not-user-facing yet: the `torch_ctf` backend (no CLI flag exists at
all — Python API only) and fresh ice-cube generation (`GradientSKIcemaker` /
`build_ice_cache` have no CLI path). Also skipped: `ice_cache_dir`,
`ice_relax_steps`, `ice_parameterization`, `shtyrov_params_path`, `conv_backend`,
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
`PotentialBuilder`. Since `potential_parameterization = "shtyrov"` is the
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

## Known gaps in this pass

- `--assembly` is unverified: every cached structure has an identical asymmetric
  unit and biological assembly, so distinguishing them needs a network fetch.
- `--membrane_min_transmembrane_spacing` showed no effect at 4 transmembrane
  copies — too few for the constraint to bind. Needs a denser test, not yet a
  finding either way.
- `filler_table_min/max_mw_kda` are only meaningful with a reference table
  enabled; the QA config keeps them off so the baseline is a run that works.
- `device`, path handling, and `.cs`/`.star`-driven runs are deliberately
  deferred to later phases.

## Reproducing

```bash
python qa/sweep.py particles --workers 24
python qa/sweep.py all
```

Runs are single-threaded so the baseline is bit-reproducible; parallelism comes
from running many at once. Results land in `qa/results/<command>.json`.
