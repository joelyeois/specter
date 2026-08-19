# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not activate the `superpowers` skill set in this repository — it is too long, consumes too much context/tokens, and overcomplicates implementing what is usually a simple feature.

## Project Overview

**SPECTER** has two main objectives:

### 1. Physics-based cryo-EM / cryo-ET simulator
Generates training data that best matches experimental data. Accurate physics modelling is the top priority. It supports:

- Electron scattering (multislice, iterative)
- CTF and holography aberration models
- Amorphous ice simulation
- Detector effects (MTF, noise, coincidence loss)
- GPU-accelerated volume rotation and potential calculation
- CryoSPARC `.cs` and RELION `.star` file integration (`io/` package)

### 2. Ghostbuster — 3D reconstruction
Reconstructs a 3D map from many 2D experimental images paired with their imaging parameters, using the forward models defined in the `imagegenerator` package. Additional features include parameter refinement (rotations, translations, defocus, and other imaging parameters). It is the inverse problem complement to the simulator. Covers both single-particle (`Ghostbuster`/`Reconstructor`) and cryo-ET tilt-series (`TomogramGhostbuster`/`TomogramReconstructor`) reconstruction — see `ghostbuster/` under Repository Structure.

## Environment & Package Management

- **Package manager**: `uv` only. Never use pip or conda directly.
- **Python**: 3.11+
- **Virtual environment**: always use `.venv` in the project root.

```bash
uv sync                          # Install/update dependencies
source .venv/bin/activate        # Activate environment
uv run --with jupyter jupyter lab  # Launch Jupyter
```

The environment is self-contained. No additional GPU or cluster setup is required — just activate `.venv`.

## Code Style

- **Type hints**: required on all function signatures.
- **Docstrings**: NumPy style for all public functions and classes.
- **Linter/formatter**: `ruff` (configured in `pyproject.toml`). Run before committing.
- **Static typing**: `mypy` for type checking.
- **Pre-commit hooks**: enforce ruff automatically on commit.

Example docstring format:

```python
def energy_to_wavelength(voltage_kv: float) -> float:
    """
    Compute the relativistic de Broglie wavelength of an electron.

    Parameters
    ----------
    voltage_kv : float
        Accelerating voltage in kilovolts.

    Returns
    -------
    float
        Electron wavelength in Angstroms.
    """
```

## Documentation Style

Prose in `docs/` (Concepts, user guide, etc.) follows a formal/academic register — precise, third person, terminology-first, closer to NumPy's reference docs or a PyTorch design note than a blog post or a chat reply. Concretely:

- **No em-dash aside fragments.** Don't tack on a compressed clause after an em-dash to restate or justify a point (e.g. "one render, not sixty"). Write it as a complete sentence, or cut it.
- **No hedging filler.** Avoid "worth being explicit about", "worth noting", "it's worth flagging" — just state the fact.
- **Don't cite the tool's own development history as justification.** A design choice (e.g. "this approximation matches what a prior/reference implementation did") doesn't need to lean on that prior tool's authority — the reader has no context for it and doesn't need it to trust the explanation. Legitimate bibliographic citations still belong in a `## References` section; the rule is about weaving lineage into prose as a rationale, not about citing sources.
- **State each caveat once**, in the section it belongs to (usually `## Limitations`), rather than explaining it at length inline and then again at the bottom. A one-line pointer ("see Limitations") from the first mention is fine.

`docs/concepts/cryoet-specimen/filaments.md` is the calibrated example if unsure how a rewrite should read.

## Development Workflow

1. **Prototype** in `dev/` — use these freely for experimentation.
2. **Implement** working code into `src/specter/` source modules.
3. **Update demo notebooks** in `demo-notebooks/` to reflect the new functionality. These must always be kept up-to-date and working.
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

- Tests live in `tests/` and use `pytest`.
- The test suite is being actively built up — every new feature or algorithm should include at least a basic test.
- Run tests with:

```bash
python -m pytest tests/ -v
python -m pytest tests/test_generators.py::test_image_generator -v  # single test
python -m pytest tests/ --cov=src/specter  # with coverage
ruff check src/ tests/                     # lint
ruff format src/ tests/                    # format
mypy src/                                  # type-check
```

- GPU tests should gracefully skip or fall back to CPU when CUDA is unavailable.
- Do not mock physics calculations — test with real (small) inputs to catch numerical regressions.
- Regression tests in `tests/test_generators.py` use a **save-or-compare** pattern: on first run they save a golden `.pt` file under `tests/test_data/`; subsequent runs compare against it. Delete the fixture file and re-run to regenerate after intentional output changes.

## Off-Limits Files

Do **not** modify files in:

- `src/specter/atom_data/` — parameterised atomic potential data (Kirkland, Lobato, Shtyrov). These are fixed physical constants from published literature.
- `src/specter/ice_data/` — pre-computed ice data: the bundled `IceBank` library (`ice_cache/`) plus the MD reference frame and radial-average S(k) targets every `GradientSKIcemaker` run needs. Both this and `atom_data/` live inside the package, and are located via `importlib.resources` (`ice_data/__init__.py`'s `bundled_ice_data()`), so they ship in a wheel and resolve identically for editable and installed layouts. Never reintroduce the old repo-root `ice-data/` + `dirname`×3 pattern: that only resolves for an editable install from a checkout, and silently pointed inside the virtualenv's `lib/` otherwise.
- `ice-data/` — what remains at the repo root: two 32 MB legacy `MDSimDump`/`ExtXYZDump` inputs that no code path reads (notebooks only). Deliberately *not* packaged, since shipping them would push the wheel past PyPI's 100 MB per-file limit for no benefit.

Changes to these would silently break the physical accuracy of all simulations.

## Architecture

### Forward simulation pipeline

All major simulator classes inherit from `BaseImager(L.LightningModule)` so they run via Lightning's GPU/CPU dispatch:

```
PotentialBuilder            – builds 3D scattering potential from atomic coordinates (PDB/mmCIF)
    ↓ V [B, Z, Y, X]
ImageGenerator (or FromCoordinates) – base shared particle generation
    crowding → solvate (ice makers) → Scattering → Aberration → Detector
        ↓ images [B, Y, X]
MicrographGenerator         – assembles a full micrograph from many particles
TiltSeriesGenerator         – generates a tilt series
```

- `ImageGenerator` takes a **pre-built volume** tensor; `ImageGeneratorFromCoordinates` builds it from atomic coordinates on the fly via `PotentialBuilder`.
- `RandomIcemaker` (cheap, instant) and `GradientSKIcemaker` (S(k)/MLBOP-optimised, expensive) generate amorphous ice volumes. `IceBank` (in `ice/_bank.py`) does not build ice itself — it draws randomly rotated/translated crops from a bundled cache of pre-optimised `GradientSKIcemaker` configs (`src/specter/ice_data/ice_cache/`, shipped inside the package) at near-zero marginal cost, and tiles multiple crops together (with a short local MLBOP seam relaxation) for volumes larger than a single cached config.
- `Scattering` supports four propagation modes: `multislice`, `rytov`, `firstborn`, `projection` — multislice is most accurate and is the default.
- `Aberration` (in `aberrations/`) and `Detector` (in `microscope.py`) apply CTF, envelope, and detector MTF in Fourier space. `aberrations/_envelopes.py` holds the Fourier-space envelope functions (B-factor, Cc/spatial-coherence, dose) as pure functions, ported from teamtomo's `torch_fourier_filter.envelopes`.
- A second, opt-in CTF backend lives in `ctf/` (`CTFParameters`, `TransferFunction`), ported from `torch-ctf` conventions and verified term-by-term against `Aberration` (including against a real multi-particle CryoSPARC `.cs` file). Every `BaseImager` subclass takes `aberration_backend: Literal["legacy", "torch_ctf"] = "legacy"`; `"torch_ctf"` swaps in `ctf/_legacy.py`'s `LegacyAberrationAdapter`, which has the same `forward(exitwave, ctf_params_dict)` signature as `Aberration` so call sites don't change. The legacy `ctf_params` dict still mirrors CryoSPARC's own units (dfu/dfv/cs in Angstrom; `phaseshift`/`tiltx`/`tilty` in radians — but `dfang` in **degrees**, converted on the way in by `io/_cryosparc.py` and back to radians inside `aberrations/_functions.py`'s `defocus`, so both the `.cs`- and `.star`-driven paths agree with the synthetic `astigmatism_angle`) — not `CTFParameters`' native units (defocus in µm, Cs in mm, angles in degrees, dimensionless Zernike coefficients) — see [[project_torch_ctf_native_units_wrapper_todo]] memory for the still-open native-units-wrapper gap. `lpp_params` (laser phase plate) is a `LegacyAberrationAdapter` constructor-time argument, not a `ctf_params` dict key, since it's a shared instrument config rather than per-particle.

### Inverse problem — Reconstructor

`ghostbuster/` is a package, not a single file:

- `_reconstructor.py` — `Reconstructor(L.LightningModule)` reconstructs a 3D volume from 2D single-particle images by minimising the discrepancy between simulated and observed images using the same forward model as `ImageGenerator`. Jointly refines volume, rotations, translations, and defocus via separate learning rates.
- `_pipeline.py` — `Ghostbuster` + `compare_runs()`: end-to-end single-particle pipeline; loads CryoSPARC particle data, preprocesses images (sign flip, dose/scale normalisation), drives `Reconstructor` via a Lightning `Trainer`.
- `_tomogram_reconstructor.py` — `TomogramReconstructor(L.LightningModule)`: reconstructs a volume from a cryo-ET tilt series using the same forward model as `TiltSeriesGenerator`. One tilt per training step keeps GPU memory bounded regardless of tomogram size. The forward model is noiseless; observed images are compared directly to `|CTF(exitwave)|²`.
- `_tomogram_pipeline.py` — `TomogramGhostbuster`: end-to-end tomogram pipeline, mirrors `Ghostbuster`'s `run`/`test_run` API.
- `_helpers.py` — shared helpers (LR scheduler construction, k-space masking, image preprocessing) used by both reconstructors.

Pose/shift/defocus refinement (`lr_R`/`lr_T`/`lr_defocus` on `Reconstructor`/`TomogramReconstructor`) is wired in but still **unverified** for correctness — no test currently checks recovered rotations/translations/defocus against ground truth. The `--lr_R`/`--lr_T`/`--lr_D` CLI flags say so in their help text; the public reconstruction docs (`docs/user-guide/reconstruction.md`) are just a "work in progress" stub pending publication, so beyond that the status isn't documented anywhere outside this file and the code itself.

`Ghostbuster`/`TomogramGhostbuster`'s `run`/`test_run` take `device` as a GPU index, a list of them (DDP), or the string `"cpu"`. That last spelling is the only way to force CPU training: `_run_helpers.py`'s `resolve_device` otherwise decides from `torch.cuda.is_available()` alone, so on a GPU machine every other value targets the GPU.

### CLI & pipelines

- `cli/` — the `specter` command (entry point `specter.cli._cli:main`), built on `click`/`rich-click`. Exposes `specter simulate particles`, `specter simulate micrograph`, `specter simulate tiltseries`, `specter build tomogram`, `specter build ice`, `specter reconstruct particle` (aliased as `specter ghostbuster particle`). Each subcommand (`simulate.py`, `build.py`, `reconstruct.py`) loads a TOML config via `config.py`'s dataclasses (`ParticleStackConfig`, `MicrographConfig`, `TiltSeriesConfig`, `TomogramConfig`, `IceCacheConfig`, `ReconstructionConfig`) with `load_config()`, applies only the flags the user actually passed via `_click_options.py`'s `build_config_options()`/`collect_overrides()` (unset flags never clobber the TOML), then calls into `pipelines/`. `specter simulate micrograph` is single-device only (no multi-GPU DDP, unlike particles/tiltseries) — each micrograph needs its own freshly regenerated ice/crowding specimen between forward passes, and a single micrograph is already the GPU-memory-bound unit of work at `micrograph_size` resolution, so there's no batching to shard across devices. This is unrelated to the older `specter-jobs` entry point (`jobs/_cli.py`), a separate job-database CLI.
- `pipelines/` — `run_particle_stack()`, `run_micrograph()`, `run_tilt_series()`, `run_build_tomogram()`, `run_build_ice_cache()`, `run_reconstruction()`: the actual end-to-end implementations behind the `cli/` commands, kept separate so `cli/` stays a thin argument-parsing layer. `_common.py` holds logic shared across them (device parsing, scalar-or-range sampling, exit-wave saving, etc.); `run_micrograph` doesn't use `_common.py`'s multi-GPU DDP dispatch, since it doesn't apply here (see `cli/` above).
- **Multi-GPU comes in two flavours**, and `_common.py` has one device parser for each. `_parse_device` feeds Lightning DDP (particles/tiltseries: one job, sharded batches). `_parse_device_pool` feeds pipelines that own their own sharding of whole independent work units across devices — currently only `run_build_ice_cache`, which spawns one worker process per device, each generating a disjoint slice of the requested ice configs. Both accept `"0,1,2"`; only the pool parser accepts `"auto"`.
- `specter build ice` (`pipelines/_ice.py`) generates a *replacement* `IceBank` library, for users needing ice at a pixel size or cell size the bundled `src/specter/ice_data/ice_cache` doesn't cover. It never writes into `ice_data/` (off-limits, shipped); output goes to `specter-data/ice`, which a simulation config then points `ice_cache_dir` at. Configs are named after their seed (`ice_config_filename`), not their batch position — that is what makes resume (skip files already present) and extension (re-run at a higher `seed_start`) both correct. The optimisation recipe itself is deliberately not exposed as config: `mlbop_target=-0.413` eV/atom and `mlbop_strength=0.5` are measured/validated properties of the phase of ice being reproduced, and a library mixing recipes would have `IceBank` drawing from several phases interchangeably.
- `specter reconstruct particle` (`pipelines/_reconstruct.py`) is the one inverse-direction command. It is also reachable as **`specter ghostbuster particle`**: `cli/_cli.py` calls `build_reconstruct_group()` twice, once per name, so the solver answers to the name the docs and the Python API call it by. Two group objects rather than one registered under two keys, since click fixes a command's `name` at construction and a shared instance would print whichever name was registered first regardless of what the user typed. A standalone `ghostbuster` console script is deliberately *not* the mechanism: PyPI already has a `ghostbuster` package (an unrelated AWS tool) whose install writes a `ghostbuster` entry point, so in a shared environment the later install would silently win.
- **`reconstruct`'s subcommand noun names the modality**, not the input or the output: `particle` is single-particle reconstruction, and the planned `TomogramGhostbuster` sibling is `specter reconstruct tomogram`. Singular for the same reason — a modality is not a count of inputs, so it does not follow `simulate particles`' plural. Do not "fix" the noun toward the input (`tiltseries`) or toward a uniform output (`map`); both readings look consistent in isolation and neither survives the pair.
- **Every `specter reconstruct particle` run is numbered and tracked through `specter.jobs`** — there is no untracked mode, the way neither RELION nor CryoSPARC has one. The directory is `job_base_dir/[project/]reconstructions/J00N/` — project comes right after `job_base_dir`, ahead of the `reconstructions` job-type subfolder, since users group their own work by project first, not by which pipeline produced it (`Job`'s folder shape generally is `base_dir/[project/]job_type/J0NN`; `_next_job_id` scans every job-type subfolder under a project, so numbering is one continuous sequence per project rather than restarting per job type). `project` (on both `ReconstructionConfig` and `Job` itself) is optional, not required: leaving it unset doesn't skip tracking, it just drops the project-name segment and uses `job_base_dir`'s implicit default project — pass `--project` to split one `job_base_dir` into several named ones, e.g. one shared scratch directory used across unrelated structures. `job_base_dir` itself defaults to `find_specter_project_root()/SPECTER_DATA_DIR` — `find_specter_project_root()` walks up from cwd looking for an existing `specter-data/`, the same way `git` resolves the nearest ancestor `.git`, so running from a subdirectory of an already-initialised project lands in the same project rather than starting a disconnected new tree (and restarting numbering from `J001`). A `job.json` always records every parameter plus the git commit, and pinning `job_id` resumes into an existing directory (this is how a manual `halfset="A"` then `halfset="B"` pair shares one, with `Job.create` asserting every *other* setting matches). `nps_weight` (a per-frequency tensor) and `lpp_params` (a nested table) are the two `Ghostbuster` arguments with no config field, since neither has a spelling a flag can carry.
- **`halfset="gold"` is the default**, not `"all"`: a bare `specter reconstruct particle` reconstructs halfsets A and B and computes the halfmap FSC between them (`fsc_gold_standard.png`, plus `resolution_gold_standard` in `job.json`), rather than one single-volume run. `pipelines/_reconstruct.py`'s `_run_gold_standard` opens the one shared `Job` exactly *once*, before spawning either halfset as a separate worker process (parallel across devices when `device` names at least two, sequential on one otherwise) — letting each worker open its own `Job` independently would race on `Job`'s auto-numbering, which does `_next_job_id()` then a bare `mkdir()` with no `exist_ok`. Workers never open their own `Job`; they only write into the path the orchestrator hands them. `halfset="A"`/`"B"` still reconstructs a single half (e.g. for a quick test), and `"all"` still reconstructs every particle as one volume, ignoring the split.
- **Output layout**: everything specter writes lands under `./specter-data/`, relative to the current working directory — `specter-data/{pdb,particles,micrographs,tiltseries,tomograms,ice,reconstructions}`. Path resolution is one rule with no special cases: an explicit path (TOML or CLI) is used verbatim and resolves relative to the cwd; an omitted `output_dir`/`pdb_savefolder` falls back to `config.py`'s `default_output_dir(artifact)` / `default_pdb_cache_dir()`, both cwd-relative (not repo-root-anchored — that only resolves correctly for an editable install). `$SPECTER_PDB_CACHE` overrides the PDB cache location to an absolute path if you want one cache shared across working directories. `.gitignore` uses `/specter-data/` with a leading slash so the pattern doesn't also match `src/specter/`.

## Repository Structure

```
src/specter/                  # Main source package
  atom/                       # Atomic properties and potential functions
    atom.py                   # Atom symbols, numbers, masses
    atomic_potentials.py      # Kirkland, Lobato, Shtyrov parameterizations
  atom_data/                  # Scattering parameter tables — do not modify
  aberrations/                # Aberration phase model
    _functions.py             # Low-level, stateless phase functions (cs, defocus, beamtilt, trefoil, tetrafoil, phaseshift)
    _aberration.py            # Aberration(L.LightningModule) — composes the functions above into a transfer function
    _envelopes.py             # B-factor/Cc/spatial-coherence/dose envelope functions (pure functions of k-grid + params)
  ctf/                        # torch-ctf-backed CTF — opt-in second backend, verified parity with aberrations/
    _parameters.py             # CTFParameters, ParamField
    _transfer.py                # TransferFunction
    _legacy.py                   # LegacyAberrationAdapter — bridges the legacy ctf_params dict to CTFParameters
    _units.py                    # zernike_rho_max and other native-unit helpers
  imagegenerator/             # Image simulation classes
    _base.py                  # BaseImager base class
    _generator.py             # ImageGenerator, ImageGeneratorFromCoordinates
    _micrograph.py            # MicrographGenerator
    _tiltseries.py            # TiltSeriesGenerator
  ice/                        # Amorphous ice generation
    _random.py                # RandomIcemaker
    _gradient.py              # GradientSKIcemaker
    _bank.py                  # IceBank (cache) + build_one_ice_config()/build_ice_cache()
    _energy.py                # MLBOP coarse-grained water potential (structural diagnostic; neighbor search via vesin-torch, not ASE)
    _kernels.py               # Shared physics-kernel construction (atomic potential, S(k) target)
    _mdsim.py                 # MDSimDump/ExtXYZDump (legacy MD trajectory ingestion)
    _helpers.py               # Helper functions (water molecules, FFT, etc.)
  jobs/                       # Job management and persistence (specter-jobs entry point — unrelated to cli/)
    _job.py                   # Job class
    _database.py              # JobDatabase storage
    _cli.py                   # CLI interface
  cli/                        # `specter` CLI (specter simulate ..., specter build ...) — see "CLI & pipelines" below
  pipelines/                  # run_particle_stack/run_micrograph/run_tilt_series/run_build_tomogram/run_build_ice_cache/run_reconstruction — see "CLI & pipelines" below
  specimen/                   # Volume assembly (package) — under heavy active development, structure below is
                              # partial/illustrative only; read the package directly rather than trusting this list.
    single_particle.py        # MicrographSpecimenGenerator — populates a volume with template potentials + crowding + ice
    cytosolic_filler.py       # PEI2016_CROWDING_TABLE + CRYOETSIM_PARTICLE_TABLE + build_filler_pool_specs() — generic cytosolic background reference tables
    filament/                 # single-strand filaments (_path/_placement/_generator: F-actin,
                              # PROTOFILAMENT_SPEC) AND real microtubules (_lattice: surface-lattice
                              # geometry with constants measured off deposited MT reconstructions;
                              # _tube: whole-tube placement returning FilamentInstances so the
                              # tomogram generator's filament stamping renders them unchanged;
                              # _tubulin: extracts an ab-tubulin dimer from 3JAL in the MT frame,
                              # +Z = protofilament axis / +X = radially outward; _frames:
                              # parallel-transport frames, required so a bent tube's protofilaments
                              # don't shear apart). No supertwist -- deliberate, see _lattice's docstring.
    tomogram/, membrane/, packing/  # newer subpackages (tomogram/specimen assembly,
                              # organic membranes, sphere/tetris packing algorithms); also from_volume.py at the
                              # top level — still in flux, deliberately not detailed here
    _grid.py                  # BeadGenerator — gold fiducial bead physics, for specimen.tomogram.TomogramSpecimenGenerator
                              # (`specter build tomogram`); also the shared bulk-material density/potential helpers _carbon.py uses
    _carbon.py                 # CarbonFilmGenerator/GridSpec — carbon support film: alpha-shape rim geometry (from-scratch
                              # CTS gen_carbon.m port, dev/gen_carbon_replica.py) + MIP-calibrated flat deposition (dev/validate_carbon_mip.py)
    _carbon_delaunay.py        # Torch-free Delaunay/circumsphere worker for _carbon.py's blocked (spawn-context) alpha-complex build
  potential.py                # Scattering potential builder
  scattering.py               # Wave propagation (multislice, rytov, firstborn, projection)
  microscope.py               # Aberration and detector models
  detectors.py                # Detector MTF and noise models
  aretomo3.py                 # AreTomo3 .aln tilt-geometry → quaternions, for TiltSeriesGenerator
  constants.py                # Physical constants (rest_mass_energy, hc, energy_to_wavelength; CODATA via scipy.constants)
  memory.py                   # Peak-memory model + recommend_batchsize/resolve_batchsize for batchsize="auto"
  rotations/                  # Quaternion-based 3D rotations (built on the `roma` library)
    _rotation.py               # roma-wrapped translate_coordinates, rotate_coordinates
    _random.py                 # roma-wrapped random_quaternion/random_rotvec/random_rotation_matrix, rotations_angular_difference
    _volume.py                 # rotate_volume, rotate_volume_fourier, affine matrix helpers
    _volume_rotator.py         # VolumeRotator (LightningModule) for sampling rotated slices
  crowding.py                 # Molecular crowding simulation
  ghostbuster/                # 3D reconstruction (PyTorch Lightning) — see "Inverse problem" above
  arrays.py                   # Array utilities (soft voxelization, tiling, crops, fourier_crop)
  coords.py                   # Coordinate utilities (RDF, etc.)
  fft.py                      # FFT wrappers
  filters.py                  # Frequency-domain filters
  image.py                    # Image-level utilities
  pdb.py                      # PDB/mmCIF parsing helpers
  io/                          # Particle/micrograph metadata I/O (package)
    _cryosparc.py               # extract_parameters_from_csfile() — reads CryoSPARC .cs files
    _relion.py                   # RELION .star read/write: extract_parameters_from_starfile(), create_particle_starfile[_from_model](), create_micrograph_starfile()
    _common.py                   # _select_particles() — shared per-particle mask/truncate helper for both backends
  config.py                   # ParticleStackConfig/MicrographConfig/TiltSeriesConfig/TomogramConfig/IceCacheConfig/ReconstructionConfig dataclasses + load_config()/apply_overrides() for TOML-driven runs (shared by cli/ and direct Python callers)
  plots.py                    # Plotting helpers
  progress.py                 # Progress bar management (ProgressManager)
  random_seed.py              # Global seed control (exported as specter.seed)
  symmetries.py               # Symmetry operations
  qscore.py                   # Per-atom Q-score (map-model fit; Pintilie et al. 2020)
tests/                        # pytest test suite
  test_data/                  # Golden-output fixtures (.pt files) for regression tests
demo-notebooks/               # User-facing, always kept working
  create_particle_stack/       # script+notebook config pattern: notebook + its curated TOML
  create_particle_stack_modular/  # same pattern, modular forward-model pipeline variant
  create_micrograph/           # same pattern, for MicrographGenerator
  create_tilt_series/          # same pattern, for TiltSeriesGenerator
  create_tilt_series_modular/  # same pattern, modular variant
  simulate_particles_from_csfile/  # same pattern, driven from an existing CryoSPARC .cs file
                                # (plus standalone notebooks with no paired TOML, e.g.
                                # generate-and-reconstruct.ipynb, coordinates-to-images.ipynb,
                                # compare-atomic-potentials-with-kirkland.ipynb)
                              # (demo-scripts/ is gone: every workflow it held is now a `specter` subcommand.
                              # Particle stacks, synthetic OR driven from a real .cs/.star via cs_path/star_path,
                              # in `specter simulate particles`; micrographs in `specter simulate micrograph`;
                              # tilt series in `specter build tomogram` + `specter simulate tiltseries`;
                              # reconstruction in `specter reconstruct particle`. Do not reintroduce the
                              # directory for a workflow that could be a subcommand instead.)
configs/                      # TOML config files consumed by the `specter` CLI (flat, not nested)
  particle.toml                # canonical defaults for `specter simulate particles`
  micrograph.toml              # canonical defaults for `specter simulate micrograph`
  tilt_series.toml             # canonical defaults for `specter simulate tiltseries`
  tomogram.toml                 # canonical defaults for `specter build tomogram`
  ice.toml                      # canonical defaults for `specter build ice`
  reconstruct.toml              # canonical defaults for `specter reconstruct particle` -- the one config with
                                # no runnable default, since [data] names a real dataset
dev/                           # Prototyping and experimentation (not required to be clean; gitignored, never pushed)
docs-figures/                  # Tracked scripts that regenerate docs/assets/images/ figures for Concepts pages —
                              # one script per concept page (e.g. membrane_shape.py -> concepts/membrane-shape.md's
                              # figures). Kept separate from demo-notebooks/ (runnable end-user pipeline examples) and
                              # dev/ (gitignored scratch) since these are doc tooling that must stay reachable on
                              # GitHub for anyone regenerating a figure after an algorithm change. Also holds scripts
                              # that regenerate a docs *table* rather than an image (ice_cache_timing.py ->
                              # user-guide/ice-cache.md's cost table) — same rationale: hardware-specific numbers in
                              # prose need a reachable script to re-measure them on new silicon.
tools/cli-qa/                  # Tracked pre-release QA sweep for the `specter` CLI (renamed from qa/). Runs the real
                              # CLI in subprocesses and fingerprints artifacts, asking what tests/test_cli_*.py cannot:
                              # does each flag actually change the output, and does a nonsense value fail fast and
                              # legibly? Run it by hand before a release — it is deliberately NOT in CI, since a full
                              # sweep is minutes of real simulation. sweep.py checks that every CLI flag has a spec.py
                              # entry, so a newly added flag surfaces as a coverage failure; see its README for the
                              # phases and FINDINGS.md for what they caught. Run artifacts go to a gitignored results/.
pdb-data/                     # PDB structure files
ice-data/                     # Two 32 MB legacy MDSimDump inputs, notebook-only, NOT packaged (do not modify)
```

## Physics Accuracy Notes

- Electron wavelength is computed relativistically — do not use the non-relativistic approximation.
- The interaction parameter `σ` is energy-dependent; always use `interaction_parameter()` from `scattering.py`.
- Atomic potentials are parameterised; the Kirkland model is the default and most validated.
- **Shtyrov needs bond topology to be worth using.** It fits scattering factors per *bonded species* (`"C(HHHC)"`), so without `atom_species` every atom silently falls back to per-element Peng `c4322` factors — real published factors, but not what `parameterization="shtyrov"` implies. Every path that builds a `PDB` and renders it therefore passes `compute_atom_species=True` when (and only when) the parameterization is `shtyrov`, and forwards `pdb.atom_species` into `PotentialBuilder`: `TomogramSpecimenGenerator` (protein/filament/microtubule templates), `render_transmembrane_template`, `run_particle_stack`, and `run_micrograph` (which silently rendered 100% Peng until 2026-08-19 — it built a default-`shtyrov` `PotentialBuilder` with no `atom_species` at all). Typing costs ~1.3 s per unique structure (36k atoms) and changes a rendered template by ~4% relative RMS. Structures with no resolvable topology (legacy PDB format, isolated ions) degrade per-atom on their own — do not add a global guard for them. Paths that build potentials from bare coordinates (`ImageGeneratorFromCoordinates`, `membrane/_profile.py`'s lipid model) have no bonds to type and correctly stay per-element.
- **Shtyrov typing is hydrogen-limited without a Monomer Library.** 20 of the 42 species name an H neighbour, and a descriptor is built from atoms actually in the model, so in a hydrogen-free deposition a methyl carbon reads `C(C)`, not `C(HHHC)`, and misses the table: ~44% of a protein falls back to Peng (measured on 1A6M; 56% -> 99% coverage with a library). Supplying H needs the [Monomer Library](https://github.com/MonomerLibrary/monomers) via `$CLIBD_MON` or `PDB(monomer_library_path=...)`; it is deliberately **not** packaged (1.5 GB, LGPL-3.0, and specter is All Rights Reserved) — users install it themselves, see `docs/installation.md`. When one resolves, `PDB` takes `atomic_numbers`/`coordinates`/`atom_species` from the single H-completed gemmi model (`_build_typed_model`) instead of reconciling Biopython's atom list against gemmi's, which is what made the two disagree and raise. Rendered potential shifts 20-30% relative RMS. Three rules are copied from `sffit` (the reference implementation, at `~/sffit`): drop `occ == 0` atoms (gemmi zero-occupancies ambiguous H — rotatable hydroxyls, *both* His tautomers — which would otherwise render at full weight since `PotentialBuilder` has no occupancy weighting), drop `ConnectionType.MetalC` links (heme Fe types `Fe(NNNN)` instead of the untabulated `Fe(NNNNNOO)`), and restore `st.connections` between the two `prepare_topology` passes. Both of the first two apply **only** on the library path: without a library, dropping metal links degrades Fe to `Fe(NNN)`/`Fe(N)`, and dropping `occ == 0` would desync from Biopython. `expand_ncs` and sffit's altloc handling are deliberately not copied — specter fetches RCSB assemblies (would double-expand) and renders without occupancy weighting (would double-count conformers).
- Coincidence loss is modelled for direct electron detectors — do not remove this when simulating K3 detector outputs.
- CTF sign conventions follow the standard cryo-EM convention (defocus positive = underfocus).
- Detector MTF and DQE(0) are separate physical effects and must stay separate: bundled detector MTFs are derived from published DQE curves, so the shape is normalised as `sqrt(DQE(k)/DQE(0))` (`detectors.py`), and the zero-frequency counting efficiency is applied separately via `Detector(dqe0=...)` by scaling expected electron counts. Folding `DQE(0)` into the MTF instead would scale counts by `sqrt(DQE(0))` rather than `DQE(0)` and give the wrong shot-noise statistics.
- `IceBank` tiles volumes larger than a single cached config in **coordinate space**: it draws multiple independently rotated/translated crops (`_place_tiles`), places them side by side, and heals the tile boundaries with a short local MLBOP relaxation (`_relax_seams`) rather than voxel-space blending — do not replace this with a plain repeat/tile or hard-edge concatenation, which would leave visible seams (and, unrelaxed, measurably unfavorable energy at the boundaries). Relaxation cost is bounded to a halo band around each seam (`_place_tiles`'s `halo_margin`; only halo atoms are fed to the energy model, the untouched bulk is reattached unchanged), and is off by default — `generate_big_ice`/`generate_big_ice_deltas`'s `relax_steps` (exposed as `ice_relax_steps` on `TiltSeriesGenerator`/`ImageGenerator`/`MicrographGenerator`/`ParticleStackConfig`) defaults to 0. `generate_big_ice` is also memory-bounded for very large volumes. `tile_volume_from_blocks_blended()` (in `arrays.py`) is a separate, still-used utility for **voxel-space** tiling — overlap-add with random roll/flip/rotation augmentation per tile — used by `MDSimDump`/`ExtXYZDump` to assemble MD trajectory frames into larger volumes, not by `IceBank`. `RandomIcemaker`/`GradientSKIcemaker` only produce single unique blocks (`generate_ice`); they don't assemble large volumes themselves.
- Ice structure is driven by `GradientSKIcemaker` (optimised against pre-computed S(k)/MLBOP targets in `src/specter/ice_data/`) and cached via `IceBank`; `RandomIcemaker` is a fast, low-fidelity fallback for quick tests.

## Reproducibility

- Use `specter.seed(n)` (re-exported from `random_seed.py`) to set a global seed before any simulation for reproducible outputs.

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | GPU computation, all array ops |
| `lightning` | Distributed training (ghostbuster) |
| `biopython`, `biotite`, `gemmi` | PDB/mmCIF parsing |
| `cryosparc-tools` | `.cs` file I/O, isolated to `io/_cryosparc.py` |
| `mrcfile`, `starfile`, `eerfile` | Cryo-EM file formats; `starfile` backs RELION `.star` I/O in `io/_relion.py` |
| `roma` | Quaternion/rotation math (`rotations/`) |
| `vesin-torch` | Pairwise neighbor search for the MLBOP ice energy (`ice/_energy.py`), replaces the old ASE-based path |
| `cupy-cuda12x` | GPU exact Euclidean distance transform for the `spherical_harmonics` membrane backend (`specimen/membrane/_field.py`); core dep on Linux/Windows, absent on macOS where scipy's CPU path takes over |
| `click`, `rich-click` | The `specter` CLI (`cli/` package) |
| `ruff`, `mypy` | Code quality |

`ase` and `seaborn` are `dev`-group dependencies only, not runtime dependencies. `seaborn` is no longer a dependency of the package itself (`plots.py` hardcodes its "deep" palette instead), but it stays in the dev group because it is used for development plotting — do not remove it.
