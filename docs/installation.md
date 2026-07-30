# Installation

SPECTER standardises on [uv](https://docs.astral.sh/uv/) and Python
3.11+. The most common mistake is installing the dependencies but not the
package itself.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or conda/pip

## 1. Clone and install

```bash
git clone https://github.com/joelyeois/specter.git
cd specter

# dependencies from pyproject.toml, into .venv
uv sync
source .venv/bin/activate

# and the package itself — skipping this breaks every notebook import
uv pip install -e .
```

## 2. Launch the notebooks

```bash
uv run --with jupyter jupyter lab
```

## 3. Confirm it works

The fastest end-to-end check is a small particle stack on CPU. It downloads
a structure, builds a potential, and writes an image stack with metadata.

```bash
python demo-scripts/generate_particle_stack.py \
    --config configs/particle.toml \
    --n_particles 4 --num_pixels 128 \
    --device cpu --output_dir ./output/
```

You should get `output/particles.mrcs` and `output/particles.star`.

## Reproducibility

Set a global seed before anything else if you need runs to be repeatable.

```python
import specter
specter.seed(0)
specter.set_verbosity("INFO")  # logs each pipeline stage as it runs
```

## Installing with conda and pip instead

Supported, but not what the project's own guidance assumes.

```bash
conda create -n specter python=3.11
conda activate specter
pip install -r requirements.txt
pip install -e .
```

You will need to install Jupyter yourself in this route.

## GPU support

SPECTER uses PyTorch for all array operations, pinned to a CUDA 12.1 index.
GPU acceleration is available automatically when a CUDA-capable device is
present; all classes fall back to CPU otherwise.

## What gets pulled in

Notable dependencies and why they are there:

| Package | Purpose |
|---|---|
| `torch` | All array computation; pinned to a CUDA 12.1 index |
| `lightning` | Device dispatch and distributed training |
| `vesin-torch` | Differentiable neighbour lists for the MLBOP water energy |
| `polnet` | Cryo-ET specimen packing, pinned to v1.1.2 |
| `gemmi`, `biotite`, `biopython` | Structure file parsing |
| `mrcfile`, `starfile`, `eerfile` | Cryo-EM file formats |
| `cryosparc-tools` | Reading `.cs` files |

## If you install outside a checkout

Only `atom_data/*.txt` is declared as package data. The ice cache in
`ice-data/` is **not**, and is located by walking up to the repository
root. A `pip install` from outside a clone will find no ice cache, and
any run with `ice_model="gd"` will fail until you copy the cache or set
`--ice_cache_dir`. See [Using the ice cache](user-guide/ice-cache.md) for details.

## Checking your work

```bash
python -m pytest tests/ -v
ruff check src/ tests/
mypy src/
```

Regression tests use a save-or-compare pattern: the first run writes a
golden `.pt` fixture under `tests/test_data/`, and later runs compare
against it. When you change outputs deliberately, delete the fixture and
re-run to regenerate.
