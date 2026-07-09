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
- CryoSPARC `.cs` file integration

### 2. Ghostbuster — 3D reconstruction
Reconstructs a 3D map from many 2D experimental images paired with their imaging parameters, using the forward models defined in the `imagegenerator` package. Additional features include parameter refinement (rotations, translations, defocus, and other imaging parameters). It is the inverse problem complement to the simulator.

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
def energy_to_wavelength(energy_kev: float) -> float:
    """
    Compute the relativistic de Broglie wavelength of an electron.

    Parameters
    ----------
    energy_kev : float
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    float
        Electron wavelength in Angstroms.
    """
```

## Development Workflow

1. **Prototype** in `dev-notebooks/` — use these freely for experimentation.
2. **Implement** working code into `src/specter/` source modules.
3. **Update demo notebooks** in `demo-notebooks/` to reflect the new functionality. These must always be kept up-to-date and working.
4. **Add a test** in `tests/` covering the new behaviour (even a minimal smoke test is better than nothing).

When modifying physics-critical code, validate against known physical quantities (e.g. wavelength at 300 kV ≈ 1.969 pm) before committing.

## Superpowers working files

`.superpowers/` at the repo root (gitignored) is where Claude Code's superpowers
skill stores design specs and other working documents (e.g. `.superpowers/specs/`).
This is separate from `docs/`, which is reserved for Read the Docs / Sphinx
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
- Ice makers (`RandomIcemaker`, `APIcemaker`, `MCMCIcemaker`, `GradientSKIcemaker`) generate amorphous ice volumes; `IceBank` (in `ice/_bank.py`) caches them for reuse.
- `Scattering` supports four propagation modes: `multislice`, `rytov`, `firstborn`, `projection` — multislice is most accurate and is the default.
- `Aberration` (in `aberrations/`) and `Detector` (in `microscope.py`) apply CTF, envelope, and detector MTF in Fourier space.

### Inverse problem — Reconstructor

`Reconstructor(L.LightningModule)` in `ghostbuster.py` reconstructs a 3D volume from 2D images by minimising the discrepancy between simulated and observed images using the same forward model as `ImageGenerator`. Jointly refines volume, rotations, translations, and defocus via separate learning rates.

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
  imagegenerator/             # Image simulation classes
    _base.py                  # BaseImager base class
    _generator.py             # ImageGenerator, ImageGeneratorFromCoordinates
    _micrograph.py            # MicrographGenerator
    _tiltseries.py            # TiltSeriesGenerator
  ice/                        # Amorphous ice generation
    _ap.py                    # APIcemaker (Atomic Potential-based)
    _mcmc.py                  # MCMCIcemaker
    _random.py                # RandomIcemaker
    _gradient.py              # GradientSKIcemaker
    _bank.py                  # IceBank (cache)
    _mdsim.py                 # MDSimDump (legacy support)
    _helpers.py               # Helper functions (water molecules, FFT, etc.)
  jobs/                       # Job management and persistence
    _job.py                   # Job class
    _database.py              # JobDatabase storage
    _cli.py                   # CLI interface
  specimen.py                 # Volume assembly
  potential.py                # Scattering potential builder
  scattering.py               # Wave propagation (multislice, rytov, firstborn, projection)
  microscope.py               # Aberration and detector models
  detectors.py                # Detector MTF and noise models
  rotations.py                # Quaternion-based 3D rotations
  crowding.py                 # Molecular crowding simulation
  ghostbuster.py              # 3D reconstruction (PyTorch Lightning)
  arrays.py                   # Array utilities (soft voxelization, tiling, crops, fourier_crop)
  coords.py                   # Coordinate utilities (RDF, etc.)
  fft.py                      # FFT wrappers
  filters.py                  # Frequency-domain filters
  image.py                    # Image-level utilities
  pdb.py                      # PDB/mmCIF parsing helpers
  cryosparc.py                # CryoSPARC .cs file I/O
  cuda.py                     # CUDA/device utilities
  config.py                   # ParticleStackConfig dataclass + load_config()/apply_overrides() for TOML-driven runs
  plots.py                    # Plotting helpers
  progress.py                 # Progress bar management (ProgressManager)
  random_seed.py              # Global seed control (exported as specter.seed)
  simulate_particles.py       # Particle simulation pipeline
  symmetries.py               # Symmetry operations
  welling_rotation.py         # Welling rotation sampling
  qscore.py                   # Per-atom Q-score (map-model fit; Pintilie et al. 2020)
tests/                        # pytest test suite
  test_data/                  # Golden-output fixtures (.pt files) for regression tests
demo-notebooks/               # User-facing, always kept working
  particle_stack/              # script+notebook config pattern: notebook + its curated TOML
demo-scripts/                 # Ready-to-run command-line scripts
configs/                      # TOML config files consumed by demo-scripts/ and demo-notebooks/
  particle_stack/
    default.toml                # canonical defaults for generate_particle_stack.py
dev-notebooks/                # Prototyping and experimentation (not required to be clean)
pdb-data/                     # PDB structure files
ice-data/                     # Pre-computed ice data (do not modify)
```

## Physics Accuracy Notes

- Electron wavelength is computed relativistically — do not use the non-relativistic approximation.
- The interaction parameter `σ` is energy-dependent; always use `interaction_parameter()` from `scattering.py`.
- Atomic potentials are parameterised; the Kirkland model is the default and most validated.
- Coincidence loss is modelled for direct electron detectors — do not remove this when simulating K3 detector outputs.
- CTF sign conventions follow the standard cryo-EM convention (defocus positive = underfocus).
- Ice volumes are assembled via `tile_volume_from_blocks()` (in `arrays.py`) with random roll/flip/rotation augmentation per tile — do not replace this with a plain repeat/tile which would produce visible seams.
- MD simulation dump ingestion (`get_mdsim`, `get_mdsim_file`, etc.) was removed from `Icemaker`; ice structure is now driven purely by pre-computed kernels in `ice-data/`.

## Reproducibility

- Use `specter.seed(n)` (re-exported from `random_seed.py`) to set a global seed before any simulation for reproducible outputs.
- `MCMCIcemaker.init_random()` no longer accepts a seed argument — call `specter.seed()` before instantiating if you need reproducibility.

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | GPU computation, all array ops |
| `lightning` | Distributed training (ghostbuster) |
| `biopython`, `gemmi` | PDB/mmCIF parsing |
| `cryosparc-tools` | `.cs` file I/O |
| `mrcfile`, `starfile` | Cryo-EM file formats |
| `ruff`, `mypy` | Code quality |
