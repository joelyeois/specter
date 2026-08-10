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

## Development Workflow

1. **Prototype** in `dev/` — use these freely for experimentation.
2. **Implement** working code into `src/specter/` source modules.
3. **Update demo notebooks** in `demo-notebooks/` to reflect the new functionality. These must always be kept up-to-date and working.
4. **Add a test** in `tests/` covering the new behaviour (even a minimal smoke test is better than nothing).

When modifying physics-critical code, validate against known physical quantities (e.g. wavelength at 300 kV ≈ 1.969 pm) before committing.

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
- `ice-data/` — pre-computed ice simulation data.

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
- `RandomIcemaker` (cheap, instant) and `GradientSKIcemaker` (S(k)/MLBOP-optimised, expensive) generate amorphous ice volumes. `IceBank` (in `ice/_bank.py`) does not build ice itself — it draws randomly rotated/translated crops from a bundled cache of pre-optimised `GradientSKIcemaker` configs (`ice-data/ice_cache/`, shipped with the repo) at near-zero marginal cost, and tiles multiple crops together (with a short local MLBOP seam relaxation) for volumes larger than a single cached config.
- `Scattering` supports four propagation modes: `multislice`, `rytov`, `firstborn`, `projection` — multislice is most accurate and is the default.
- `Aberration` (in `aberrations/`) and `Detector` (in `microscope.py`) apply CTF, envelope, and detector MTF in Fourier space. `aberrations/_envelopes.py` holds the Fourier-space envelope functions (B-factor, Cc/spatial-coherence, dose) as pure functions, ported from teamtomo's `torch_fourier_filter.envelopes`.
- A second, opt-in CTF backend lives in `ctf/` (`CTFParameters`, `TransferFunction`), ported from `torch-ctf` conventions and verified term-by-term against `Aberration` (including against a real multi-particle CryoSPARC `.cs` file). Every `BaseImager` subclass takes `aberration_backend: Literal["legacy", "torch_ctf"] = "legacy"`; `"torch_ctf"` swaps in `ctf/_legacy.py`'s `LegacyAberrationAdapter`, which has the same `forward(exitwave, ctf_params_dict)` signature as `Aberration` so call sites don't change. The legacy `ctf_params` dict still mirrors CryoSPARC's own units (dfu/dfv/cs in Angstrom, angles in radians) — not `CTFParameters`' native units (defocus in µm, Cs in mm, angles in degrees, dimensionless Zernike coefficients) — see [[project_torch_ctf_native_units_wrapper_todo]] memory for the still-open native-units-wrapper gap. `lpp_params` (laser phase plate) is a `LegacyAberrationAdapter` constructor-time argument, not a `ctf_params` dict key, since it's a shared instrument config rather than per-particle.

### Inverse problem — Reconstructor

`ghostbuster/` is a package, not a single file:

- `_reconstructor.py` — `Reconstructor(L.LightningModule)` reconstructs a 3D volume from 2D single-particle images by minimising the discrepancy between simulated and observed images using the same forward model as `ImageGenerator`. Jointly refines volume, rotations, translations, and defocus via separate learning rates.
- `_pipeline.py` — `Ghostbuster` + `compare_runs()`: end-to-end single-particle pipeline; loads CryoSPARC particle data, preprocesses images (sign flip, dose/scale normalisation), drives `Reconstructor` via a Lightning `Trainer`.
- `_tomogram_reconstructor.py` — `TomogramReconstructor(L.LightningModule)`: reconstructs a volume from a cryo-ET tilt series using the same forward model as `TiltSeriesGenerator`. One tilt per training step keeps GPU memory bounded regardless of tomogram size. The forward model is noiseless; observed images are compared directly to `|CTF(exitwave)|²`.
- `_tomogram_pipeline.py` — `TomogramGhostbuster`: end-to-end tomogram pipeline, mirrors `Ghostbuster`'s `run`/`test_run` API.
- `_helpers.py` — shared helpers (LR scheduler construction, k-space masking, image preprocessing) used by both reconstructors.

Pose/shift/defocus refinement (`lr_R`/`lr_T`/`lr_defocus` on `Reconstructor`/`TomogramReconstructor`) is wired in but still **unverified** for correctness — no test currently checks recovered rotations/translations/defocus against ground truth. The public reconstruction docs (`docs/user-guide/reconstruction.md`) are just a "work in progress" stub pending publication, so this status isn't documented anywhere outside this file and the code itself.

### CLI & pipelines

- `cli/` — the `specter` command (entry point `specter.cli._cli:main`), built on `click`/`rich-click`. Exposes `specter simulate particles`, `specter simulate tiltseries`, `specter build tomogram`. Each subcommand (`simulate.py`, `build.py`) loads a TOML config via `config.py`'s dataclasses (`ParticleStackConfig`, `TiltSeriesConfig`, `TomogramConfig`) with `load_config()`, applies only the flags the user actually passed via `_click_options.py`'s `build_config_options()`/`collect_overrides()` (unset flags never clobber the TOML), then calls into `pipelines/`. This is unrelated to the older `specter-jobs` entry point (`jobs/_cli.py`), a separate job-database CLI.
- `pipelines/` — `run_particle_stack()`, `run_tilt_series()`, `run_build_tomogram()`: the actual end-to-end implementations behind the `cli/` commands, kept separate so `cli/` stays a thin argument-parsing layer. `_common.py` holds logic shared across all three.

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
    _bank.py                  # IceBank (cache) + build_ice_cache()
    _energy.py                # MLBOP coarse-grained water potential (structural diagnostic; neighbor search via vesin-torch, not ASE)
    _kernels.py               # Shared physics-kernel construction (atomic potential, S(k) target)
    _mdsim.py                 # MDSimDump/ExtXYZDump (legacy MD trajectory ingestion)
    _helpers.py               # Helper functions (water molecules, FFT, etc.)
  jobs/                       # Job management and persistence (specter-jobs entry point — unrelated to cli/)
    _job.py                   # Job class
    _database.py              # JobDatabase storage
    _cli.py                   # CLI interface
  cli/                        # `specter` CLI (specter simulate ..., specter build ...) — see "CLI & pipelines" below
  pipelines/                  # run_particle_stack/run_tilt_series/run_build_tomogram — see "CLI & pipelines" below
  specimen/                   # Volume assembly (package) — under heavy active development, structure below is
                              # partial/illustrative only; read the package directly rather than trusting this list.
    single_particle.py        # MicrographSpecimenGenerator — populates a volume with template potentials + crowding + ice
    cytosolic_filler.py       # PEI2016_CROWDING_TABLE + CRYOETSIM_PARTICLE_TABLE + build_filler_pool_specs() — generic cytosolic background reference tables
    tomogram/, filament/, membrane/, packing/  # newer subpackages (tomogram/specimen assembly, filament placement,
                              # organic membranes, sphere/tetris packing algorithms); also from_volume.py at the
                              # top level — still in flux, deliberately not detailed here
    _grid.py                  # CarbonFilmGenerator/BeadGenerator — carbon support film + gold fiducial bead physics
                              # for specimen.tomogram.MembraneTomogramGenerator (`specter build tomogram`)
  potential.py                # Scattering potential builder
  scattering.py               # Wave propagation (multislice, rytov, firstborn, projection)
  microscope.py               # Aberration and detector models
  detectors.py                # Detector MTF and noise models
  aretomo3.py                 # AreTomo3 .aln tilt-geometry → quaternions, for TiltSeriesGenerator
  constants.py                # Physical constants (rest_mass_energy, hc, energy_to_wavelength; CODATA via scipy.constants)
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
  config.py                   # ParticleStackConfig/MicrographConfig/TiltSeriesConfig/TomogramConfig dataclasses + load_config()/apply_overrides() for TOML-driven runs (shared by demo-scripts/ and cli/)
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
demo-scripts/                 # Ready-to-run command-line scripts (generate_micrograph.py, generate_tilt_series.py,
                              # generate_particle_stack_from_csfile.py, generate_particle_stack_from_starfile.py,
                              # ghostbuster_reconstruct.py) — plain particle-stack generation now lives in the
                              # `specter simulate particles` CLI instead of a demo-script
configs/                      # TOML config files consumed by demo-scripts/ and the `specter` CLI (flat, not nested)
  particle.toml                # canonical defaults for `specter simulate particles`
  micrograph.toml              # canonical defaults for generate_micrograph.py
  tilt_series.toml             # canonical defaults for generate_tilt_series.py / `specter simulate tiltseries`
  tomogram.toml                 # canonical defaults for `specter build tomogram`
dev/                           # Prototyping and experimentation (not required to be clean; gitignored, never pushed)
docs-figures/                  # Tracked scripts that regenerate docs/assets/images/ figures for Concepts pages —
                              # one script per concept page (e.g. membrane_shape.py -> concepts/membrane-shape.md's
                              # figures). Kept separate from demo-scripts/ (runnable end-user pipeline examples) and
                              # dev/ (gitignored scratch) since these are doc tooling that must stay reachable on
                              # GitHub for anyone regenerating a figure after an algorithm change.
pdb-data/                     # PDB structure files
ice-data/                     # Pre-computed ice data (do not modify)
```

## Physics Accuracy Notes

- Electron wavelength is computed relativistically — do not use the non-relativistic approximation.
- The interaction parameter `σ` is energy-dependent; always use `interaction_parameter()` from `scattering.py`.
- Atomic potentials are parameterised; the Kirkland model is the default and most validated.
- Coincidence loss is modelled for direct electron detectors — do not remove this when simulating K3 detector outputs.
- CTF sign conventions follow the standard cryo-EM convention (defocus positive = underfocus).
- `IceBank` tiles volumes larger than a single cached config in **coordinate space**: it draws multiple independently rotated/translated crops (`_place_tiles`), places them side by side, and heals the tile boundaries with a short local MLBOP relaxation (`_relax_seams`) rather than voxel-space blending — do not replace this with a plain repeat/tile or hard-edge concatenation, which would leave visible seams (and, unrelaxed, measurably unfavorable energy at the boundaries). Relaxation cost is bounded to a halo band around each seam (`_place_tiles`'s `halo_margin`; only halo atoms are fed to the energy model, the untouched bulk is reattached unchanged), and is off by default — `generate_big_ice`/`generate_big_ice_deltas`'s `relax_steps` (exposed as `ice_relax_steps` on `TiltSeriesGenerator`/`ImageGenerator`/`MicrographGenerator`/`ParticleStackConfig`) defaults to 0. `generate_big_ice` is also memory-bounded for very large volumes. `tile_volume_from_blocks_blended()` (in `arrays.py`) is a separate, still-used utility for **voxel-space** tiling — overlap-add with random roll/flip/rotation augmentation per tile — used by `MDSimDump`/`ExtXYZDump` to assemble MD trajectory frames into larger volumes, not by `IceBank`. `RandomIcemaker`/`GradientSKIcemaker` only produce single unique blocks (`generate_ice`); they don't assemble large volumes themselves.
- Ice structure is driven by `GradientSKIcemaker` (optimised against pre-computed S(k)/MLBOP targets in `ice-data/`) and cached via `IceBank`; `RandomIcemaker` is a fast, low-fidelity fallback for quick tests.

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
| `click`, `rich-click` | The `specter` CLI (`cli/` package) |
| `ruff`, `mypy` | Code quality |

`ase` is an optional `dev`-group dependency only (not a runtime dependency); `seaborn` has been dropped entirely (`plots.py` hardcodes its "deep" palette instead).
