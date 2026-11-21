# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not activate the `superpowers` skill set in this repository — it is too long, consumes too much context/tokens, and overcomplicates implementing what is usually a simple feature.

## Project Overview

**SPECTER** has two main objectives:

### 1. Physics-based cryo-EM / cryo-ET simulator
Generates training data that best matches experimental data. Accurate physics modelling is the top priority.

### 2. Ghostbuster — 3D reconstruction
Reconstructs a 3D map from many 2D experimental images paired with their imaging parameters, using the forward model defined in the `imaging` package. Additional features include parameter refinement (rotations, translations, defocus, and other imaging parameters). It is the inverse problem complement to the simulator. Covers both single-particle (`Ghostbuster`/`Reconstructor`) and cryo-ET tilt-series (`TomogramGhostbuster`/`TomogramReconstructor`) reconstruction.

## Environment & Package Management

- **Package manager**: `uv` only. Never use pip or conda directly.
- **Python**: 3.11+
- **Virtual environment**: always use `.venv` in the project root.

The environment is self-contained. No additional GPU or cluster setup is required — just activate `.venv`.

## Code Style

- **Type hints**: required on all function signatures.
- **Docstrings**: NumPy style for all public functions and classes.
- **Output channels**: library code (`imaging/`, `ghostbuster/`'s reconstructors, `pdb.py`, `specimen/`, ...) reports through `specter.logger` (`logger.info` for progress, `logger.warning` for something the caller should know about), never `print`; it is silent until `specter.set_verbosity` raises the level, which the pipelines do. User-driving code -- `cli/`, `pipelines/`, and the `Ghostbuster`/`TomogramGhostbuster` drivers -- prints through `specter.progress.console` (rich), which also owns the progress bars and section headers. `warnings.warn` is reserved for "your inputs may be wrong" (a fallback that changed the result, a missing library), since it carries the caller's location and can be filtered by category.

## Documentation Style

Prose in `docs/` (Concepts, user guide, etc.) follows a formal/academic register — precise, third person, terminology-first, closer to NumPy's reference docs or a PyTorch design note than a blog post or a chat reply. Concretely:

- **No em-dash aside fragments.** Don't tack on a compressed clause after an em-dash to restate or justify a point (e.g. "one render, not sixty"). Write it as a complete sentence, or cut it.
- **No hedging filler.** Avoid "worth being explicit about", "worth noting", "it's worth flagging" — just state the fact.
- **Don't cite the tool's own development history as justification.** A design choice (e.g. "this approximation matches what a prior/reference implementation did") doesn't need to lean on that prior tool's authority — the reader has no context for it and doesn't need it to trust the explanation. Legitimate bibliographic citations still belong in a `## References` section; the rule is about weaving lineage into prose as a rationale, not about citing sources.
- **State each caveat once**, in the section it belongs to (usually `## Limitations`), rather than explaining it at length inline and then again at the bottom. A one-line pointer ("see Limitations") from the first mention is fine.

- **Cite borrowed work where it is described.** A method, parameterization, measured value, dataset or structure taken from someone else's publication is cited author–year in the text (`Cowley & Moodie (1957)`, `(McMullan et al. 2015)`) and listed in a `## References` section at the very end of the page: `- Surname, I. I., & Surname, I. I. (Year). Title in sentence case. *Journal*, Vol(Issue), first–last. [doi:X](https://doi.org/X)`, all authors up to ten, else the first six and `et al.`. Check every entry against Crossref (`https://api.crossref.org/works/<doi>`) before adding it; never write a DOI, volume or page range from memory. Docstrings use a NumPy `References` section with `.. [n]` entries in the same form. `python docs-figures/bibliography.py` collects the pages' entries into `docs/references.md`.

`docs/concepts/cryoet-specimen/filaments.md` is the calibrated example if unsure how a rewrite should read.

## Development Workflow

1. **Prototype** in `dev/` — use these freely for experimentation.
2. **Implement** working code into `src/specter/` source modules.
3. **Update the docs examples** that the change touches. The walkthrough, tutorial and recipe pages quote runnable scripts in `docs-figures/` (`walkthrough/`, `tutorials/`); rerun the script so its figures and printed output stay current. There are no notebooks: `demo-notebooks/` was retired on 2026-11-19 in favour of these pages, so do not reintroduce `.ipynb` files.
4. **Add a test** in `tests/` covering the new behaviour (even a minimal smoke test is better than nothing).

When modifying physics-critical code, validate against known physical quantities (e.g. wavelength at 300 kV ≈ 1.969 pm) before committing.

## Git

- **Commit and push straight to `main`.** Do not create a feature branch, and do not open a pull request, unless explicitly asked to. This is a single-maintainer repo; the default "branch first when on the default branch" behaviour just adds a merge-and-delete round trip for no benefit here.
- Still commit and push only when asked — this overrides *where* work lands, not *whether* to push it.
- Commit only the files belonging to the task at hand. This working tree often carries unrelated in-flight edits; use explicit pathspecs (`git commit -- <paths>`) rather than `git commit -a`.
- Pre-commit runs `ruff format` and can rewrite staged files, aborting the commit. Re-stage the reformatted files and commit again.

## Superpowers working files

`.superpowers/` at the repo root (gitignored) is where Claude Code's superpowers
skill stores design specs and other working documents (e.g. `.superpowers/specs/`).
This is separate from `docs/`, which is reserved for Read the Docs / Zensical
content only — do not put specs or planning docs there.

## Testing

- Run tests with:

```bash
python -m pytest tests/                    # full suite, parallel by default (~3 min)
python -m pytest tests/test_imaging_regression.py::test_particle_regression -v -n 0  # single test
mypy src/                                  # type-check (not a pre-commit hook; run by hand)
```

- **Use `-n 0` for a single test, `-s`, or `--pdb`.** 32 worker startups cost ~6 s, which dwarfs a short test (2.9 s serial vs 9.0 s parallel), and xdist swallows `-s` output and breaks interactive debugging.
- GPU tests should gracefully skip or fall back to CPU when CUDA is unavailable.
- Do not mock physics calculations — test with real (small) inputs to catch numerical regressions.
- **Do not test what a shipped TOML config says.** No assertion should read `configs/*.toml`; test the code the config feeds instead.
- `tests/CLAUDE.md` (loaded when working under `tests/`) holds the rest: why the parallel settings must not be simplified, golden-file regeneration, and the full TOML rule.

## Off-Limits Files

Do **not** modify files in:

- `src/specter/atom_data/` — parameterised atomic potential data (Kirkland, Lobato, Shtyrov). These are fixed physical constants from published literature.
- `src/specter/ice_data/` — pre-computed ice data: the bundled `IceBank` library (`ice_cache/`) plus the MD reference frame and radial-average S(k) targets every `GradientSKIcemaker` run needs. Both this and `atom_data/` live inside the package, and are located via `importlib.resources` (`ice_data/__init__.py`'s `bundled_ice_data()`), so they ship in a wheel and resolve identically for editable and installed layouts. Never reintroduce the old repo-root `ice-data/` + `dirname`×3 pattern: that only resolves for an editable install from a checkout, and silently pointed inside the virtualenv's `lib/` otherwise.
- `ice-data/` — what remains at the repo root: two 32 MB legacy MD-derived `.pt` files (`mdsim_f_kernel_averaged_200x200x200_0.5A.pt`, `mdsim_frame100_ice_deltas.pt`) that no code path, test or notebook reads. `ice/_mdsim.py`'s `MDSimDump`/`ExtXYZDump` readers are kept on purpose (2026-11-07, the user's call) though only their tests use them. Deliberately *not* packaged, since shipping them would push the wheel past PyPI's 100 MB per-file limit for no benefit.

Changes to these would silently break the physical accuracy of all simulations.

## Repository conventions

- **`demo-scripts/` is gone**: every workflow is a `specter` subcommand; do not reintroduce the directory.
- **`configs/*.toml` order their tables like their command's `--help` panels, ending in `[advanced]`**; panel order comes from `cli/`'s `_*_GROUPS` lists. Deliberately not enforced by a test (see Testing).
- `dev/` is gitignored scratch, never pushed.

## Traps

Each of these looks like a safe simplification and is not. Nothing fails when one is broken; the output is just wrong.

### Physics and conventions
- **The inverse accepts only `Propagation` and `Optics`.** `Envelopes` and `Camera` are deliberately absent from the reconstructors: the data cannot determine them. Defocus reference and `specimen_absorption` are shared code between simulator and inverse (`tests/test_forward_model_parity.py`).
- **The inverse stays Cc-free.** `Reconstructor` calls `intensity(cc=None)` explicitly, and the ghostbuster golden fixtures must not move when a forward default changes.
- **An unset `rotate_mode` pairs with the scattering model** (`Propagation.rotation`, the user's call on 2026-11-20): `surface` for projection/firstborn/rytov/ctf, `exact` for multislice/kinematic, in simulation and reconstruction alike. It falls back to `exact` on a dressed specimen (crowding, ice, damage, atoms, aperture) or `inelastic_mfp`, and to `real` when `lr_R` or `lr_T` refines a pose where `exact` would apply (`surface` carries pose gradients, `exact` does not yet). The particle pipeline settles it once per run, so its clean exit waves are posed like its images.
- **A reconstruction under `rotate_mode="exact"` windows its map before rotating** (`Reconstructor.support`, `ghostbuster._resolution.exact_rotation_support`: flat to 0.925, zero at 0.98 of half the box). The rotation is periodic, and unwindowed the fit put noise in the corners that wrapped into its images: on 10551 at box 432, 20.8 nats/particle worse held-out. The solver's probes are drawn in the same window, so no prior rescaling goes with it (a 1/mean(window^2) factor only corrected whole-box probes, gone since da8b7bb). Simulation stays unwindowed.
- **`rotate_volume(V, build_affine_matrix(R))` turns the density by `R^T`.** A caller holding a template-to-world rotation passes `R.transpose(-1, -2)`.
- **Never z-flip V to "fix" the propagator sign.** `exp(-i pi lambda dz k^2)` is physical and validated (a4b36c9).
- **`dfang` in the `ctf_params` dict is in degrees**; `phaseshift`/`tiltx`/`tilty` are in radians.
- **Shtyrov needs bond types.** Every path that renders a PDB under `shtyrov` passes `compute_atom_species=True` and forwards `atom_species`, or every atom silently falls back to Peng. `scattering_factors` (biomolecules) and `bulk_scattering_factors` (ice, carbon, gold) stay two fields.
- **Pose optimisers use `weight_decay=0`.** AdamW's default pulls refined parameters towards zero.

### Dependencies and caches
- **torch-ctf is pinned `<0.6` because `ctf/_legacy.py` multiplies by `rho_max**3`.** Lift the pin and delete the workaround together, never separately. `ctf/` is the opt-in `Optics(aberration_backend="torch_ctf")` path (`imaging/_engines.py`, `imaging/_waves.py`) and the planned successor to the default `"legacy"` backend, not dead code.
- **Bump `_PARSED_CACHE_VERSION`** when changing `_build_typed_model`, `get_atoms_and_coordinates` or `get_atom_species`, or stale parsed structures are returned silently.
- **Never copy user files into the PDB cache.** Everything in it must be re-derivable, which is what makes `specter cache clean` safe.

### Concurrency (run directories are on NFS)
- **`flock` hangs on NFS.** A reconstruction writes only `job.json`, from the orchestrator. Workers return results through a polled queue (never a blocking `get()`). Half-map FSC epochs are claimed by creating the PNG with `O_CREAT | O_EXCL` (never an `exists()` check). A gold run opens its `Job` once, before spawning the halves.

### Performance
- **`limited_cpu_threads` is scoped, not process-wide, and never entered inside a loop.** Each resize of the 128-thread pool costs ~12 ms.

### Licensing
- **`auto_mask` is built from CryoSPARC's public documentation only.** Never read or reverse-engineer CryoSPARC's compiled code.
- **The Monomer Library is not packaged** (1.5 GB, LGPL-3.0); users install it themselves.
- **No `ghostbuster` console script.** PyPI's unrelated `ghostbuster` package would collide; the alias is a second click group.

### Removed on evidence: do not reintroduce
Alternative losses (`use_ncc`, `learn_noise_model`, `nps_weight`, `use_2d_mask`, `sparsity`); weighting the loss by alpha; a soft sphere mask before exact rotation in simulation (it deletes a known template's density); geometric (vdW) occupancy; `potential_scale`; a nonzero `klim`; the 2026-10-25 solver-stage options (absorption map, support mask, noise weight, arcsin(alpha) phase; recover from 00946a9); `crowd_max_distance_z` following `nz` in particle stacks; fitted B-factor, potential scale or coincidence radius in `specter match`. Re-adding any of them needs a held-out test showing it helps.

### Testing
- **Test `symmetrize_exact` on a compact, smooth object, never `torch.randn`.** Box-filling noise is not left unchanged by the averaging.
- **Never cross-check focal spread against PyPI abTEM 1.0.10.** It weights the wave amplitude rather than the intensity.

## Physics rules (always apply)

- Electron wavelength is computed relativistically — do not use the non-relativistic approximation.
- The interaction parameter `σ` is energy-dependent; always use `interaction_parameter()` from `constants.py` (re-exported by `specter.scattering`).
- Coincidence loss is modelled for direct electron detectors — do not remove this when simulating K3 detector outputs.
- CTF sign conventions follow the standard cryo-EM convention (defocus positive = underfocus).
- Detector MTF and DQE(0) are separate physical effects and must stay separate: bundled detector MTFs are derived from published DQE curves, so the shape is normalised as `sqrt(DQE(k)/DQE(0))` (`detectors/_catalogue.py`), and the zero-frequency counting efficiency is applied separately via `Detector(dqe0=...)` by scaling expected electron counts. Folding `DQE(0)` into the MTF instead would scale counts by `sqrt(DQE(0))` rather than `DQE(0)` and give the wrong shot-noise statistics.

## Reproducibility

- Use `specter.seed(n)` (re-exported from `random_seed.py`) to set a global seed before any simulation. It controls every draw specter makes, but **it does not make a run bit-reproducible**, and the gap is larger than float noise suggests.
- **A seeded particle stack is a function of `(seed, particle index)`, not of how it was batched** (since 2026-10-13). `Specimen.scatter(seed=)` and `Image.detect(seed=)` take it, and each particle's crowding, ice draw and detector noise come from a generator seeded by `random_seed.derive_seed(seed, poses.ids[i], stage)` (NumPy `SeedSequence` under the hood, abTEM's practice; stages are `imaging._stages.SEED_STAGE_*`): one batch of six, two of three and six singles draw the same crowding, ice and noise, and one or several worker processes give the same stack. Under multislice the images then agree to float rounding rather than bit for bit (since 2026-11-19): the fine-grid resampling uses real FFTs, whose rounding depends on the batch size on both MKL and cuFFT, so exit waves differ by ~1e-6 relative and a few Poisson counts flip by one; that was accepted for a ~3x faster resampling, and `scattering._kernels.fine_grid_potential` records it (`tests/test_particle_base.py`, `tests/test_imaging_particles.py::test_a_slice_of_the_stack_draws_what_the_stack_drew`, `tests/test_pipelines.py::test_run_particle_stack_seed_reproduces_across_batch_sizes`). That is why `batchsize="auto"` is no longer refused alongside a seed. A seed is an argument to each call, not object state, so a shard needs nothing broadcast. Three mechanisms, chosen per stage: the detector takes an explicit `torch.Generator` through every draw site (no global state touched); `IceBank` already took a `generator`; the crowd's draws go through roma and the Poisson-disk sampler, which take none, so they run under `random_seed.seeded_rng`, a CPU-only `fork_rng`. Without a `seed` (the direct-API default) everything still comes off the global stream and batching decides which draw reaches which particle. A micrograph is seeded per index too (`Field.draw(seed=, index=)`, stage `SEED_STAGE_SPECIMEN`), and a tilt's noise per `(seed, id, detector stage, tilt)`. The forward model is non-deterministic at ~2e-5 relative, from GPU reduction and FFT ordering, and Poisson sampling then DISCRETIZES that: most pixels round to the same integer count, a few cross a boundary and jump. Measured on a seeded `specter simulate tiltseries`, two identical runs differ on 0.1% of pixels by up to 74 counts; the same config at `noise_model="none"` differs on 98.8% of pixels but only at 2.1e-5 relative. `specter build tomogram` is likewise not bit-reproducible at a fixed seed, because its species are rendered in parallel and accumulated in a non-deterministic order (29% of voxels differ, at 6e-6 V on a 24 V volume). Both are inherent unless `torch.use_deterministic_algorithms(True)` is forced, which costs performance and is not set. What a seed DOES guarantee is that the specimen -- which species, how many, placed where -- is identical, and every one of those is exact: labels, picks and region volumes come back byte-identical run to run.
- **When A/B-ing a change for reproducibility, always run a baseline-vs-baseline control.** Two runs of the SAME code differ by the amounts above, so a comparison without that control reports the noise as a regression. This is not hypothetical: it made a 29%-of-voxels "difference" look like a real change until the control showed identical numbers.

## Dependency notes

- `finufft`/`cufinufft` back `nufft.py`; `cufinufft` is off on macOS, where `finufft` runs the same transform on the CPU, as it does when cuFINUFFT cannot load.
- `vesin-torch` replaced the old ASE-based neighbour search for the MLBOP ice energy.
- `nvidia-ml-py` (NVML) gives `ghostbuster/_memory.py` this process's own device memory, cuFINUFFT's allocations included, which neither PyTorch's statistics nor `mem_get_info` (shared with other jobs) isolate. Off on macOS; without it the batch probe falls back to device-wide use.
- `cupy-cuda12x` is a core dependency on Linux/Windows (membrane distance transform), absent on macOS where scipy's CPU path takes over.
- `connected-components-3d` (`cc3d`) is LGPL-3.0, used unmodified as a runtime dependency (3.7-5x faster than `scipy.ndimage.label` for the tomogram region labelling).
- `ase` and `seaborn` are `dev`-group dependencies only. `plots.py` hardcodes seaborn's "deep" palette, but seaborn stays in the dev group for development plotting — do not remove it.
