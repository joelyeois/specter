# CLAUDE.md — SPECTER

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
Reconstructs a 3D map from many 2D experimental images paired with their imaging parameters, using the forward models defined in `imagegenerator.py`. Additional features include parameter refinement (rotations, translations, defocus, and other imaging parameters). It is the inverse problem complement to the simulator.

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

## Testing

- Tests live in `tests/` and use `pytest`.
- The test suite is being actively built up — every new feature or algorithm should include at least a basic test.
- Run tests with:

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src/specter  # with coverage
```

- GPU tests should gracefully skip or fall back to CPU when CUDA is unavailable.
- Do not mock physics calculations — test with real (small) inputs to catch numerical regressions.

## Off-Limits Files

Do **not** modify files in:

- `src/specter/atom/atom_data/` — parameterised atomic potential data (Kirkland, Lobato, Shtyrov). These are fixed physical constants from published literature.
- `ice-data/` — pre-computed ice simulation data.

Changes to these would silently break the physical accuracy of all simulations.

## Repository Structure

```
src/specter/          # Main source package
  atom/               # Atomic properties and potential functions
  imagegenerator.py   # Top-level image simulation classes
  specimen.py         # Volume assembly (TomogramGenerator)
  potential.py        # Scattering potential builder
  scattering.py       # Wave propagation (multislice, iterative)
  microscope.py       # Aberration and detector models
  rotations.py        # Quaternion-based 3D rotations
  icemaker.py         # Amorphous ice generation
  crowding.py         # Molecular crowding simulation
  ghostbuster.py      # 3D reconstruction (PyTorch Lightning)
tests/                # pytest test suite
demo-notebooks/       # User-facing, always kept working
dev-notebooks/        # Prototyping and experimentation (not required to be clean)
pdb-data/             # PDB structure files
ice-data/             # Pre-computed ice data (do not modify)
```

## Physics Accuracy Notes

- Electron wavelength is computed relativistically — do not use the non-relativistic approximation.
- The interaction parameter `σ` is energy-dependent; always use `interaction_parameter()` from `scattering.py`.
- Atomic potentials are parameterised; the Kirkland model is the default and most validated.
- Coincidence loss is modelled for direct electron detectors — do not remove this when simulating K3 detector outputs.
- CTF sign conventions follow the standard cryo-EM convention (defocus positive = underfocus).

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | GPU computation, all array ops |
| `lightning` | Distributed training (ghostbuster) |
| `biopython`, `gemmi` | PDB/mmCIF parsing |
| `cryosparc-tools` | `.cs` file I/O |
| `mrcfile`, `starfile` | Cryo-EM file formats |
| `ruff`, `mypy` | Code quality |
